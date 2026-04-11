"""PatchGAN discriminator for SA-CUT virtual H&E staining.

Implements the NLayerDiscriminator from pix2pix (Isola et al., CVPR 2017).
The discriminator operates on overlapping 70×70 patches rather than the full
image, making it agnostic to image size and better at capturing local texture
fidelity — exactly what matters for histology.

No sigmoid in the final layer: we use LSGAN (least-squares loss), which
directly optimises the raw logit values and provides smoother, more stable
gradients than vanilla GAN.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from models.networks import get_norm_layer, init_weights


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with configurable depth.

    For ``n_layers=3`` and a 256×256 input the output is ``[B, 1, 30, 30]``,
    where each output unit has a 70×70-pixel receptive field in the input.

    Layer layout (``n_layers=3``, ``ndf=64``):

    +--------+---------------------------------------------------+----------+
    | Index  | Operation                                         | Spatial  |
    +========+===================================================+==========+
    | 0      | Conv2d(input_nc→64, 4×4, s=2) + LeakyReLU(0.2)  | 128×128  |
    | 1      | Conv2d(64→128, 4×4, s=2) + IN + LeakyReLU(0.2)  |  64×64   |
    | 2      | Conv2d(128→256, 4×4, s=2) + IN + LeakyReLU(0.2) |  32×32   |
    | 3      | Conv2d(256→512, 4×4, s=1) + IN + LeakyReLU(0.2) |  31×31   |
    | 4      | Conv2d(512→1, 4×4, s=1)                          |  30×30   |
    +--------+---------------------------------------------------+----------+

    Receptive field computation for ``n_layers=3``::

        RF(conv_0) = 4
        RF(conv_1) = 4 + (4−1)×2  = 10
        RF(conv_2) = 10 + (4−1)×4 = 22
        RF(conv_3) = 22 + (4−1)×8 = 46
        RF(output) = 46 + (4−1)×8 = 70   ← classic PatchGAN

    Args:
        config: Namespace with fields:

            - ``input_nc``  (int): Input channels. Default 3 (RGB H&E).
            - ``ndf``       (int): Base feature depth. Default 64.
            - ``n_layers``  (int): Number of stride-2 conv layers. Default 3.
            - ``norm_type`` (str): ``"instance"`` | ``"batch"`` | ``"none"``.
            - ``use_spectral_norm`` (bool): Wrap the first conv with spectral
              normalisation to further stabilise discriminator training.
              Default ``False``.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        input_nc: int = config.input_nc
        ndf: int = config.ndf
        n_layers: int = config.n_layers
        norm_type: str = getattr(config, "norm_type", "instance")
        use_spectral_norm: bool = getattr(config, "use_spectral_norm", False)

        kw: int = 4   # kernel width (4×4 — standard PatchGAN)
        padw: int = 1  # padding

        # ------------------------------------------------------------------
        # Spectral-norm mode (SNGAN-style): SN on every Conv2d, no IN.
        # SN constrains the Lipschitz constant of D, preventing overconfidence
        # without requiring careful R1 lambda tuning.
        # Standard mode: InstanceNorm between convolutions (pix2pix convention).
        # ------------------------------------------------------------------
        if use_spectral_norm:
            use_bias: bool = True  # no IN to absorb bias
            norm_fn = get_norm_layer("none")
            def _wrap(conv: nn.Module) -> nn.Module:
                return nn.utils.spectral_norm(conv)
        else:
            norm_layer = get_norm_layer(norm_type)
            use_bias = norm_type == "instance"
            norm_fn = norm_layer
            def _wrap(conv: nn.Module) -> nn.Module:
                return conv

        # ------------------------------------------------------------------
        # Layer 0 — first conv, no normalisation (standard pix2pix convention)
        # ------------------------------------------------------------------
        layers: list[nn.Module] = [
            _wrap(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # ------------------------------------------------------------------
        # Layers 1 … n_layers−1 — stride-2 blocks with normalisation
        # ------------------------------------------------------------------
        nf_mult: int = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                _wrap(nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                )),
                norm_fn(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # ------------------------------------------------------------------
        # Layer n_layers — stride-1 block (transitions to output resolution)
        # ------------------------------------------------------------------
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            _wrap(nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            )),
            norm_fn(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # ------------------------------------------------------------------
        # Output — 1-channel patch logits, no sigmoid (LSGAN)
        # ------------------------------------------------------------------
        layers.append(
            _wrap(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-patch real/fake logits.

        Args:
            x: Input image tensor of shape ``(B, input_nc, H, W)``.

        Returns:
            Patch logit map of shape ``(B, 1, H', W')``.
            For a 256×256 input with ``n_layers=3``: ``(B, 1, 30, 30)``.
            No sigmoid applied — compatible with LSGAN and vanilla GAN losses.
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Shape / sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    cfg = SimpleNamespace(
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_type="instance",
        use_spectral_norm=False,
    )

    disc = NLayerDiscriminator(cfg)
    init_weights(disc, init_type="normal", init_gain=0.02)
    disc.eval()

    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = disc(dummy)

    assert output.shape == (1, 1, 30, 30), f"Unexpected shape: {output.shape}"
    # No sigmoid — output should NOT be clipped to [0, 1]
    assert not (output.min() >= 0 and output.max() <= 1), (
        "Output appears to have sigmoid applied — LSGAN requires raw logits"
    )

    # Also verify spectral norm variant produces same output shape
    cfg_sn = SimpleNamespace(
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_type="instance",
        use_spectral_norm=True,
    )
    disc_sn = NLayerDiscriminator(cfg_sn)
    with torch.no_grad():
        output_sn = disc_sn(dummy)
    assert output_sn.shape == (1, 1, 30, 30), (
        f"Spectral-norm variant unexpected shape: {output_sn.shape}"
    )

    print("All assertions passed.")
    print(f"  Input  shape : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output range : [{output.min():.4f}, {output.max():.4f}]  (raw logits, no sigmoid)")
    print(f"  Spectral-norm output shape : {tuple(output_sn.shape)}")

    # Print per-layer spatial dimensions for n_layers=3
    print("\n  Spatial dimensions through the discriminator (n_layers=3):")
    x = dummy
    for i, layer in enumerate(disc.model):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d,)):
            print(f"    After conv layer {i}: {tuple(x.shape)}")
