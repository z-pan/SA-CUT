"""Structure consistency loss (L_struct) for SA-CUT.

Self-consistency constraint: nuclear regions in the generated H&E image must
spatially align with the input nuclear mask produced by AFN-DeSeg/Cellpose.

This is **not** a cross-domain pixel-wise loss.  The comparison is between:
  * ``input_mask``      — nuclear segmentation from the TPAF patch (input side)
  * ``extracted_mask``  — hematoxylin regions extracted from ``generated_he``

Both masks describe *the same* physical slide region, so spatial agreement is
biologically grounded (whereas comparing source TPAF intensities to generated
H&E pixels is meaningless for unpaired data).

Modes
-----
``'threshold'`` (default, fully differentiable)
    1. Convert generated H&E from ``[-1, 1]`` → ``[0, 1]``.
    2. Decompose RGB into HED (Hematoxylin–Eosin–DAB) stain space using a
       differentiable PyTorch reimplementation of Ruifrok & Johnston (2001).
    3. Apply a sigmoid soft-threshold on the H channel to obtain a soft
       nuclear mask.
    4. Compute soft Dice loss between the extracted mask and ``input_mask``.
    Gradient flows: loss → extracted mask → HED decomp → tanh output of G.

``'pretrained_detector'``
    Runs a frozen pretrained nucleus detector on the generated H&E for a
    more accurate ``extracted_mask``.  Gradient to the generator is obtained
    via the differentiable HED path with the detector's output used as the
    supervision target (straight-through estimator).
    Not yet implemented — raises ``NotImplementedError``.

HED decomposition details
--------------------------
Following ``skimage.color.separate_stains`` (Ruifrok & Johnston, 2001):

    log_od  = log(RGB.clamp(1e-6)) / log(1e-6)   # Beer–Lambert OD
    stains  = log_od @ HED_FROM_RGB               # unmix stain concentrations
    stains  = stains.clamp(min=0)                 # non-negative concentrations

The 3×3 unmixing matrix ``HED_FROM_RGB`` is the inverse of the standard stain
matrix (hematoxylin, eosin, DAB absorption vectors in OD space).

Warm-up and ramp-up
-------------------
During the warm-up phase the generator output does not yet resemble H&E, so
the HED extraction is meaningless.  ``forward`` returns zero loss for
``epoch < warmup_epochs``, then linearly ramps from 0 to 1 over ``ramp_epochs``
epochs before reaching full weight at ``epoch >= warmup_epochs + ramp_epochs``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# HED unmixing matrix (Ruifrok & Johnston, 2001)
# Inverse of the standard stain-vector matrix:
#   rgb_from_hed = [[0.65, 0.70, 0.29],   # hematoxylin OD vector
#                   [0.07, 0.99, 0.11],   # eosin OD vector
#                   [0.27, 0.57, 0.78]]   # DAB OD vector
# Matches skimage.color.hed_from_rgb exactly.
# ---------------------------------------------------------------------------

_HED_FROM_RGB: Tensor = torch.tensor(
    [
        [ 1.87798274, -1.00767869, -0.55611582],
        [-0.06590806,  1.13473037, -0.13552180],
        [-0.60190736, -0.48041419,  1.57358807],
    ],
    dtype=torch.float32,
)

_LOG_EPS: float = math.log(1e-6)  # ≈ −13.8155


# ---------------------------------------------------------------------------
# Standalone utilities (importable for testing)
# ---------------------------------------------------------------------------


def rgb_to_hed(rgb_01: Tensor, hed_matrix: Tensor) -> Tensor:
    """Differentiable RGB → HED stain-space decomposition.

    Implements the Beer–Lambert unmixing from Ruifrok & Johnston (2001),
    matching ``skimage.color.rgb2hed`` to floating-point precision.

    Args:
        rgb_01: RGB image tensor of shape ``(B, 3, H, W)``, values in
            ``[0, 1]``.
        hed_matrix: 3×3 unmixing matrix (``_HED_FROM_RGB``), on the same
            device as ``rgb_01``.

    Returns:
        Stain-concentration tensor of shape ``(B, 3, H, W)``, values ≥ 0.
        Channel 0 = hematoxylin, 1 = eosin, 2 = DAB.
    """
    B, C, H, W = rgb_01.shape
    # Optical density: log(I) / log(eps), maps (0,1] → [0,1] monotonically
    log_od = torch.log(rgb_01.clamp(min=1e-6)) / _LOG_EPS  # (B, 3, H, W)
    # Reshape to (B, N, 3) for batched matrix multiply
    od_flat = log_od.permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B, H*W, 3)
    stains_flat = od_flat @ hed_matrix                        # (B, H*W, 3)
    stains = stains_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
    return stains.clamp(min=0.0)


def soft_dice_loss(pred: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """Soft Dice loss averaged over the batch.

    Differentiable — ``pred`` can be any value in ``[0, 1]`` (soft mask).
    ``target`` is typically a hard binary mask in ``{0, 1}``.

    For binary inputs with perfect overlap (``pred == target``), the loss is
    exactly 0 regardless of the ``smooth`` value.

    Args:
        pred:   Predicted soft mask, shape ``(B, 1, H, W)``, values in ``[0, 1]``.
        target: Ground-truth mask,   shape ``(B, 1, H, W)``, values in ``[0, 1]``.
        smooth: Laplace smoothing constant to avoid division by zero.
                Set to 0 for a strict Dice coefficient (not recommended for
                training due to instability when both masks are empty).

    Returns:
        Scalar Dice loss in ``[0, 1]``.
    """
    pred_flat = pred.flatten(1)    # (B, N)
    tgt_flat = target.flatten(1)   # (B, N)

    intersection = (pred_flat * tgt_flat).sum(dim=1)  # (B,)
    cardinality = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)  # (B,)

    dice_per_image = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice_per_image.mean()


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------


class StructureConsistencyLoss(nn.Module):
    """Self-consistency loss between the input nuclear mask and the nuclei
    extracted from the generated H&E image.

    Args:
        mode: Extraction strategy for the nuclear mask from generated H&E.

            ``'luminance'``
                Nuclei are the darkest structures in H&E regardless of exact
                hue.  Convert to grayscale, invert, sigmoid-threshold, then
                soft Dice against the input mask.  Works from epoch 0 (no
                chicken-and-egg dependency on correct colour).

            ``'threshold'``
                Differentiable HED colour deconvolution followed by sigmoid
                on the H (hematoxylin) channel.  Requires the generated image
                to already roughly resemble H&E for meaningful gradients.

            ``'pretrained_detector'``
                Not yet implemented.

        threshold: Value above which a pixel is considered nuclear.
            For ``'luminance'``: inverted-grayscale threshold, typical 0.5.
            For ``'threshold'``: H-channel threshold, typical 0.05–0.30.
        sharpness: Steepness of the sigmoid soft-threshold.  Higher values
            approximate a hard threshold.  Default ``15.0``.
        learnable_threshold: If ``True``, ``threshold`` and ``sharpness`` are
            registered as ``nn.Parameter`` objects and updated by any
            optimizer that receives ``loss_fn.parameters()``.  Default ``False``.
        smooth: Soft Dice smoothing constant.  Default ``1.0``.
        warmup_epochs: Epochs before the loss activates.  Default ``10``.
        ramp_epochs: Epochs over which the loss weight linearly ramps from
            0 to 1 after the warm-up phase.  Default ``5``.
        detector_checkpoint: Path to a pretrained nucleus detector checkpoint.
            Only used when ``mode='pretrained_detector'``.
    """

    def __init__(
        self,
        mode: str = "luminance",
        threshold: float = 0.5,
        sharpness: float = 15.0,
        learnable_threshold: bool = False,
        smooth: float = 1.0,
        warmup_epochs: int = 10,
        ramp_epochs: int = 5,
        detector_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        if mode not in {"luminance", "threshold", "pretrained_detector"}:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'luminance', 'threshold', "
                "or 'pretrained_detector'."
            )
        self.mode = mode
        self.smooth = smooth
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = max(ramp_epochs, 1)  # avoid division by zero

        # HED unmixing matrix — buffer (device-aware, not a parameter)
        self.register_buffer("hed_matrix", _HED_FROM_RGB.clone())

        # Threshold & sharpness — optionally learnable
        self.threshold_val = nn.Parameter(
            torch.tensor(threshold, dtype=torch.float32),
            requires_grad=learnable_threshold,
        )
        self.sharpness_val = nn.Parameter(
            torch.tensor(sharpness, dtype=torch.float32),
            requires_grad=learnable_threshold,
        )

        # Cached extracted mask for debug logging (detached)
        self._last_hem_mask: Optional[Tensor] = None

        if mode == "pretrained_detector":
            self._init_detector(detector_checkpoint)

    # ------------------------------------------------------------------
    # Mode B: pretrained detector (stub)
    # ------------------------------------------------------------------

    def _init_detector(self, checkpoint_path: Optional[str]) -> None:
        """Initialise the pretrained nucleus detector (Mode B).

        Args:
            checkpoint_path: Path to the detector checkpoint.

        Raises:
            NotImplementedError: Always — Mode B is not yet implemented.
        """
        raise NotImplementedError(
            "Mode 'pretrained_detector' is not yet implemented.\n"
            "Planned design:\n"
            "  1. Load a frozen HoVer-Net (or similar) from checkpoint_path.\n"
            "  2. Run detector on generated_he.detach() to obtain a soft mask.\n"
            "  3. Use the detected mask as the supervision target for the\n"
            "     differentiable HED extraction path (straight-through estimator):\n"
            "       loss = soft_dice(hed_extracted_mask, detector_mask.detach())\n"
            "  This keeps the gradient path through the generator while using\n"
            "  the detector's more accurate mask as the learning target.\n"
            "Set mode='threshold' to use the fully differentiable default."
        )

    # ------------------------------------------------------------------
    # Warm-up schedule
    # ------------------------------------------------------------------

    def schedule_weight(self, epoch: int) -> float:
        """Compute the scalar loss weight for the given epoch.

        Returns ``0.0`` during warm-up, then linearly ramps to ``1.0``.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Float weight in ``[0.0, 1.0]``.
        """
        if epoch < self.warmup_epochs:
            return 0.0
        elapsed = epoch - self.warmup_epochs
        if elapsed >= self.ramp_epochs:
            return 1.0
        return float(elapsed) / float(self.ramp_epochs)

    # ------------------------------------------------------------------
    # HED extraction
    # ------------------------------------------------------------------

    def extract_hem_mask(self, generated_he: Tensor) -> Tensor:
        """Extract a soft nuclear (hematoxylin) mask from a generated H&E image.

        Pipeline:

        1. ``[-1, 1]`` → ``[0, 1]`` (tanh output range correction).
        2. RGB → HED via differentiable Beer–Lambert unmixing.
        3. Sigmoid soft-threshold on the H channel:
           ``mask = σ((H − threshold) × sharpness)``

        The full operation is differentiable — gradients flow back to
        ``generated_he``.

        Args:
            generated_he: Generated H&E tensor, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]`` (tanh output range).

        Returns:
            Soft nuclear mask, shape ``(B, 1, H, W)``, values in ``(0, 1)``.
        """
        # Step 1: [-1, 1] → [0, 1]
        rgb_01 = generated_he * 0.5 + 0.5

        # Step 2: HED decomposition
        hed = rgb_to_hed(rgb_01, self.hed_matrix)  # (B, 3, H, W)
        h_channel = hed[:, 0:1, :, :]              # (B, 1, H, W)

        # Step 3: differentiable soft threshold
        extracted = torch.sigmoid((h_channel - self.threshold_val) * self.sharpness_val)
        return extracted

    # ------------------------------------------------------------------
    # Luminance extraction
    # ------------------------------------------------------------------

    def extract_luminance_mask(self, generated_he: Tensor) -> Tensor:
        """Extract a soft nuclear mask based on image darkness (luminance).

        In H&E staining, nuclei are always the **darkest** structures
        regardless of exact hue — hematoxylin absorbs more light than eosin.
        This property holds even when the generated colours are not yet
        correct (e.g. blue-green instead of purple), making luminance-based
        extraction robust from epoch 0.

        Pipeline:

        1. ``[-1, 1]`` → ``[0, 1]`` (tanh output range correction).
        2. RGB → luminance-weighted grayscale.
        3. Invert: dark regions (nuclei) → high values.
        4. Sigmoid soft-threshold on inverted grayscale:
           ``mask = σ((inv_gray − threshold) × sharpness)``

        Args:
            generated_he: Generated H&E tensor, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]``.

        Returns:
            Soft nuclear mask, shape ``(B, 1, H, W)``, values in ``(0, 1)``.
        """
        # Step 1: [-1, 1] → [0, 1]
        rgb_01 = generated_he * 0.5 + 0.5

        # Step 2: luminance-weighted grayscale (ITU-R BT.601)
        gray = (
            0.299 * rgb_01[:, 0:1]
            + 0.587 * rgb_01[:, 1:2]
            + 0.114 * rgb_01[:, 2:3]
        )

        # Step 3: invert — dark nuclei become high values
        inv_gray = 1.0 - gray

        # Step 4: differentiable soft threshold
        extracted = torch.sigmoid(
            (inv_gray - self.threshold_val) * self.sharpness_val
        )
        return extracted

    # ------------------------------------------------------------------
    # Mode dispatcher
    # ------------------------------------------------------------------

    def extract_mask(self, generated_he: Tensor) -> Tensor:
        """Extract a soft nuclear mask using the configured mode.

        Dispatches to :meth:`extract_luminance_mask` (mode ``'luminance'``)
        or :meth:`extract_hem_mask` (mode ``'threshold'``).

        Args:
            generated_he: Generated H&E tensor, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]``.

        Returns:
            Soft nuclear mask, shape ``(B, 1, H, W)``, values in ``(0, 1)``.
        """
        if self.mode == "luminance":
            return self.extract_luminance_mask(generated_he)
        return self.extract_hem_mask(generated_he)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        generated_he: Tensor,
        input_mask: Tensor,
        epoch: int = 0,
    ) -> Tensor:
        """Compute the structure consistency loss.

        Args:
            generated_he: Generated H&E image, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]``.
            input_mask: Nuclear segmentation mask from AFN-DeSeg/Cellpose,
                shape ``(B, 1, H, W)``, float32 in ``[0, 1]``.
            epoch: Current training epoch (0-indexed) for warm-up scheduling.

        Returns:
            Scalar loss tensor.  Returns a detached zero during warm-up so that
            the caller's ``total_loss.backward()`` is unaffected.
        """
        weight = self.schedule_weight(epoch)

        if weight == 0.0:
            return torch.zeros((), device=generated_he.device, dtype=generated_he.dtype)

        extracted = self.extract_mask(generated_he)

        # Cache for external logging (e.g. TensorBoard) — detached, no gradient
        self._last_hem_mask = extracted.detach()

        loss = soft_dice_loss(extracted, input_mask, smooth=self.smooth)
        return weight * loss

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def get_last_hem_mask(self) -> Optional[Tensor]:
        """Return the hematoxylin mask extracted during the most recent forward.

        Returns ``None`` if ``forward`` has not yet been called or if the last
        call was in the warm-up phase.  The returned tensor is detached and safe
        to pass directly to a TensorBoard ``add_image`` call.

        Returns:
            Float32 tensor of shape ``(B, 1, H, W)`` or ``None``.
        """
        return self._last_hem_mask
