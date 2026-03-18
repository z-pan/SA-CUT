"""ResNet-based generator for SA-CUT virtual H&E staining.

Architecture follows the CUT paper (Park et al., ECCV 2020):
  Encoder  : 7×7 conv → 2× downsampling (stride-2 conv)
  Bottleneck: N ResNet blocks (ReflectionPad + Conv + InstanceNorm + ReLU × 2)
  Decoder  : 2× transposed conv upsampling → 7×7 conv → Tanh

Key SA-CUT modification: ``input_nc=2`` concatenates the TPAF channel with the
pre-computed nuclear segmentation mask, providing a structural anchor signal to
the encoder from the very first layer (early-fusion strategy).

No skip connections are used (ResNet generator, not U-Net) to prevent the
segmentation mask from leaking directly into the output without translation.

Normalization: always ``InstanceNorm2d`` — BatchNorm must not be used because
the training batch size is 1.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from models.networks import get_norm_layer, init_weights


# ---------------------------------------------------------------------------
# ResNet building block
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """Single residual block used in the generator bottleneck.

    Uses ReflectionPad2d to avoid border artifacts (preferred over zero-padding
    for image translation tasks).

    Args:
        dim: Number of channels (constant throughout the block).
        norm_layer: Callable that accepts ``dim`` and returns a norm module.
        use_dropout: If ``True``, insert Dropout(0.5) between the two convolutions.
        use_bias: Whether convolutions include a bias term.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Callable,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input feature map of shape ``(B, dim, H, W)``.

        Returns:
            Output feature map of the same shape as ``x``.
        """
        return x + self.conv_block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ResNetGenerator(nn.Module):
    """Mask-conditioned ResNet generator for TPAF → virtual H&E translation.

    The generator is built as a flat ``nn.ModuleList`` so that
    ``get_encoder_features`` can efficiently extract intermediate activations
    at arbitrary layer indices (required by the SA-PatchNCE loss).

    Flat layer layout for ``n_resnet_blocks=9`` (default):

    +---------+------------------------------------------------------+
    | Index   | Layer                                                |
    +=========+======================================================+
    | 0       | ReflectionPad2d(3)                                   |
    | 1       | Conv2d(input_nc → ngf, 7×7)                          |
    | 2       | InstanceNorm2d(ngf)                                  |
    | 3       | ReLU                                                 |
    | 4       | Conv2d(ngf → ngf×2, 3×3 stride 2) — downsample 1    |
    | 5       | InstanceNorm2d(ngf×2)                                |
    | 6       | ReLU                                                 |
    | 7       | Conv2d(ngf×2 → ngf×4, 3×3 stride 2) — downsample 2  |
    | 8       | InstanceNorm2d(ngf×4)                                |
    | 9       | ReLU                                                 |
    | 10–18   | ResnetBlock × 9                                      |
    | 19      | ConvTranspose2d(ngf×4 → ngf×2) — upsample 1         |
    | 20      | InstanceNorm2d(ngf×2)                                |
    | 21      | ReLU                                                 |
    | 22      | ConvTranspose2d(ngf×2 → ngf) — upsample 2           |
    | 23      | InstanceNorm2d(ngf)                                  |
    | 24      | ReLU                                                 |
    | 25      | ReflectionPad2d(3)                                   |
    | 26      | Conv2d(ngf → output_nc, 7×7)                         |
    | 27      | Tanh                                                 |
    +---------+------------------------------------------------------+

    Default PatchNCE feature indices ``[0, 4, 8, 12, 16]`` extract:
      - 0  : padded input  (spatial reference)
      - 4  : first downsampled feature  (ngf×2, H/2)
      - 8  : second downsampled, post-norm  (ngf×4, H/4)
      - 12 : after ResBlock 3  (ngf×4, H/4)
      - 16 : after ResBlock 7  (ngf×4, H/4)

    Args:
        config: Namespace with fields:

            - ``input_nc`` (int): Input channels. Default 2 (TPAF + mask).
            - ``output_nc`` (int): Output channels. Default 3 (RGB H&E).
            - ``ngf`` (int): Base feature depth. Default 64.
            - ``n_resnet_blocks`` (int): Number of ResNet blocks. Default 9.
            - ``norm_type`` (str): ``"instance"`` | ``"batch"`` | ``"none"``.
            - ``use_dropout`` (bool): Dropout in ResNet blocks. Default False.
    """

    # Number of spatial downsampling / upsampling steps (fixed at 2 per CUT design)
    _N_DOWNSAMPLING: int = 2

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        input_nc: int = config.input_nc
        output_nc: int = config.output_nc
        ngf: int = config.ngf
        n_resnet_blocks: int = config.n_resnet_blocks
        norm_type: str = getattr(config, "norm_type", "instance")
        use_dropout: bool = getattr(config, "use_dropout", False)

        norm_layer = get_norm_layer(norm_type)
        # InstanceNorm absorbs bias → disable conv bias to avoid redundancy
        use_bias: bool = norm_type == "instance"

        layers: list[nn.Module] = []

        # ------------------------------------------------------------------
        # Encoder — initial 7×7 conv (indices 0–3)
        # ------------------------------------------------------------------
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        # ------------------------------------------------------------------
        # Encoder — downsampling blocks (indices 4–9 for n_downsampling=2)
        # ------------------------------------------------------------------
        for i in range(self._N_DOWNSAMPLING):
            mult = 2**i
            layers += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # ------------------------------------------------------------------
        # Bottleneck — ResNet blocks (indices 10 … 10+n_resnet_blocks-1)
        # ------------------------------------------------------------------
        mult = 2**self._N_DOWNSAMPLING  # 4
        for _ in range(n_resnet_blocks):
            layers.append(
                ResnetBlock(ngf * mult, norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            )

        # ------------------------------------------------------------------
        # Decoder — upsampling blocks
        # ------------------------------------------------------------------
        for i in range(self._N_DOWNSAMPLING):
            mult = 2 ** (self._N_DOWNSAMPLING - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # ------------------------------------------------------------------
        # Decoder — output 7×7 conv + Tanh
        # ------------------------------------------------------------------
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.ModuleList(layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full generator forward pass.

        Args:
            x: Input tensor of shape ``(B, input_nc, H, W)``.
                For SA-CUT: channel 0 = TPAF, channel 1 = nuclear mask.

        Returns:
            Generated H&E image of shape ``(B, output_nc, H, W)``,
            values in ``[-1, 1]``.
        """
        for layer in self.model:
            x = layer(x)
        return x

    # ------------------------------------------------------------------
    # Feature extraction for PatchNCE loss
    # ------------------------------------------------------------------

    def get_encoder_features(
        self,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Extract intermediate feature maps at specified layer indices.

        Runs a partial forward pass stopping at ``max(layer_indices)`` and
        collects activations at every requested index.  Used by the
        SA-PatchNCE loss to sample query / key patches from multiple encoder
        scales.

        Default indices ``[0, 4, 8, 12, 16]`` correspond to (with ngf=64,
        9 resblocks): padded input, first downsampled features, second
        downsampled post-norm, and two intermediate resblock outputs.

        Args:
            x: Input tensor of shape ``(B, input_nc, H, W)``.
            layer_indices: List of integer indices into ``self.model``.
                Defaults to ``[0, 4, 8, 12, 16]``.

        Returns:
            List of feature tensors in the order of ``layer_indices``.

        Raises:
            ValueError: If any index exceeds the number of model layers.
        """
        if layer_indices is None:
            layer_indices = [0, 4, 8, 12, 16]

        n_layers = len(self.model)
        for idx in layer_indices:
            if idx >= n_layers:
                raise ValueError(
                    f"Layer index {idx} is out of range (model has {n_layers} layers)."
                )

        index_set = set(layer_indices)
        max_idx = max(layer_indices)

        features: List[torch.Tensor] = []
        # Preserve order matching layer_indices (not sorted by appearance)
        collected: dict[int, torch.Tensor] = {}

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in index_set:
                collected[i] = x
            if i >= max_idx:
                break

        for idx in layer_indices:
            features.append(collected[idx])

        return features


# ---------------------------------------------------------------------------
# Shape / sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    cfg = SimpleNamespace(
        input_nc=2,
        output_nc=3,
        ngf=64,
        n_resnet_blocks=9,
        norm_type="instance",
        use_dropout=False,
    )

    generator = ResNetGenerator(cfg)
    init_weights(generator, init_type="normal", init_gain=0.02)
    generator.eval()

    dummy = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        output = generator(dummy)

    assert output.shape == (1, 3, 256, 256), f"Unexpected shape: {output.shape}"
    assert output.min() >= -1.0 and output.max() <= 1.0, (
        f"Output range [{output.min():.4f}, {output.max():.4f}] not in [-1, 1]"
    )

    feats = generator.get_encoder_features(dummy, layer_indices=[0, 4, 8, 12, 16])
    expected_shapes = [
        (1, 2, 262, 262),   # index 0: after ReflectionPad2d(3) on 256×256
        (1, 128, 128, 128), # index 4: first downsampled conv output
        (1, 256, 64, 64),   # index 8: second downsample InstanceNorm output
        (1, 256, 64, 64),   # index 12: after ResBlock 3
        (1, 256, 64, 64),   # index 16: after ResBlock 7
    ]
    for i, (feat, expected) in enumerate(zip(feats, expected_shapes)):
        assert feat.shape == torch.Size(expected), (
            f"Feature {i} shape {tuple(feat.shape)} != expected {expected}"
        )

    print("All assertions passed.")
    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output range : [{output.min():.4f}, {output.max():.4f}]")
    for i, (feat, idx) in enumerate(zip(feats, [0, 4, 8, 12, 16])):
        print(f"  Feature[{idx:2d}]  : {tuple(feat.shape)}")
