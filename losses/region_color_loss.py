"""Region-conditioned colour statistics matching loss for SA-CUT.

Motivation — why global colour matching fails
----------------------------------------------
:class:`losses.color_stats_loss.ColorStatsLoss` matches only the *global*
channel-wise mean/std of the generated image to the real H&E distribution.
Its degenerate minimum is a spatially **uniform** colour: a flat pink fills the
whole patch, the global statistics match, and the loss saturates while the
image is biologically wrong (Run 7, epoch 95 — "overly uniform pink").

The deeper problem is an **asymmetry of spatial supervision** in SA-CUT:

  * Nuclei get a strong, dedicated, spatial loss — ``L_struct`` (soft Dice on
    the dark / hematoxylin regions).
  * Eosin (cytoplasm / stroma) has **no** spatial anchor.  The only colour
    signals are ``L_adv`` (weak, saturates when D collapses) and global
    ``L_color`` (degenerates to uniform pink).

So the generator optimises the strongly-supervised objective (dark purple
nuclei) and lets the unconstrained eosin drift to whatever satisfies the global
statistics — pale, uniform pink.

Solution — match colour statistics *per region*
-----------------------------------------------
Partition the image into three regions and match colour moments **within each
region** separately.  This gives eosin a dedicated colour anchor symmetric to
the one nuclei already have, and it removes the uniform-pink minimum because a
flat colour can no longer satisfy three region targets at once.

    +-------------+-------------------------------+---------------------------+
    | Region      | Generated-side weight source  | Real-side weight source   |
    +=============+===============================+===========================+
    | nuclear     | input mask (mask > 0.5)       | mask extracted from real  |
    |             |                               | H&E via L_struct extractor|
    +-------------+-------------------------------+---------------------------+
    | cytoplasm   | non-mask AND bright TPAF      | non-nuclear AND non-white |
    | (eosin)     | (signal present)              |                           |
    +-------------+-------------------------------+---------------------------+
    | background  | non-mask AND dark TPAF        | non-nuclear AND white     |
    |             | (no signal / lumen)           | (high luminance)          |
    +-------------+-------------------------------+---------------------------+

Unpaired-safe by construction
-----------------------------
No pixel of the generated image is ever compared to a pixel of the real image.
Only **region-aggregated statistics** are compared, and the real-side targets
are maintained as an exponential moving average (EMA) over batches — the same
mechanism as the global loss.  The region weights are derived from the TPAF
input mask (generated side) and the real H&E itself (real side); both are
constants with respect to the generator, so the only gradient path is
generated pixels → region statistics → loss.

Usage::

    criterion_color = RegionColorStatsLoss(...).to(device)

    # In the training loop, once per iteration:
    real_nuc = struct_loss.extract_mask(real_he)          # (B,1,H,W) soft, no grad
    criterion_color.update_stats(real_he, real_nuc)       # EMA of real region stats
    loss_color = criterion_color(fake_he, tpaf, mask) * lambda_color
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Region identifiers.  Order fixed for reproducible buffer registration.
_REGIONS: tuple[str, ...] = ("nuclear", "cytoplasm", "background")

# ITU-R BT.601 luminance weights (match _rgb_to_gray in the trainer / L_struct).
_LUMA = (0.299, 0.587, 0.114)


def _luminance(rgb_01: Tensor) -> Tensor:
    """Luminance-weighted grayscale of an ``[0, 1]`` RGB tensor.

    Args:
        rgb_01: ``(B, 3, H, W)`` float in ``[0, 1]``.

    Returns:
        ``(B, 1, H, W)`` luminance in ``[0, 1]``.
    """
    return (
        _LUMA[0] * rgb_01[:, 0:1]
        + _LUMA[1] * rgb_01[:, 1:2]
        + _LUMA[2] * rgb_01[:, 2:3]
    )


def _weighted_channel_stats(
    x: Tensor, weight: Tensor, eps: float = 1e-5
) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted per-channel mean and std, aggregated over batch and space.

    Args:
        x: Image tensor ``(B, C, H, W)``.
        weight: Non-negative pixel weights ``(B, 1, H, W)`` (broadcast over C).
        eps: Numerical floor for the weight sum and variance.

    Returns:
        Tuple ``(mean, std, wsum)`` where ``mean`` and ``std`` are ``(C,)``
        and ``wsum`` is a scalar tensor (total weight mass in the batch).
        Gradients flow through ``x`` (the weights are treated as constants).
    """
    wsum = weight.sum()                                    # scalar
    denom = wsum.clamp(min=eps)
    mean = (weight * x).sum(dim=(0, 2, 3)) / denom         # (C,)
    diff2 = (x - mean.view(1, -1, 1, 1)) ** 2
    var = (weight * diff2).sum(dim=(0, 2, 3)) / denom      # (C,)
    std = torch.sqrt(var.clamp(min=eps))                   # (C,)
    return mean, std, wsum


class RegionColorStatsLoss(nn.Module):
    """Region-conditioned colour statistics matching loss.

    Maintains a per-region EMA of the real H&E channel-wise mean and std, then
    penalises the generated image whose per-region statistics deviate from
    those targets.  See the module docstring for the full motivation.

    Loss = Σ_r w_r · [ L1(gen_mean_r, ema_mean_r)
                       + β · L1(gen_std_r, ema_std_r) ]

    where the sum runs over the active regions and ``β = std_weight``.

    Args:
        momentum: EMA update rate for the real-side region statistics.
            Lower is more stable. Default ``0.01``.
        warmup_batches: ``forward`` returns zero until this many
            ``update_stats`` calls have populated the EMA. Default ``50``.
        std_weight: Relative weight of the std-matching term (``β``).
            Default ``1.0``.
        region_weights: Per-region loss multipliers. Regions with weight
            ``0`` (or absent) are dropped from the loss. Default emphasises
            cytoplasm because eosin is the under-supervised region.
        white_threshold: Real-side luminance above which a non-nuclear pixel
            is treated as white background. Default ``0.8``.
        tpaf_fg_threshold: Generated-side TPAF intensity above which a
            non-nuclear pixel is treated as cytoplasm (signal present) rather
            than background. Default ``0.1``.
        sharpness: Sigmoid steepness for the soft background / foreground
            partitions. Default ``10.0``.
        min_region_mass: A region whose total pixel weight in the current
            batch is below this is skipped for that batch (both EMA update and
            loss), avoiding noisy statistics from near-empty regions.
            Default ``4.0``.
    """

    def __init__(
        self,
        momentum: float = 0.01,
        warmup_batches: int = 50,
        std_weight: float = 1.0,
        region_weights: Optional[Dict[str, float]] = None,
        white_threshold: float = 0.8,
        tpaf_fg_threshold: float = 0.1,
        sharpness: float = 10.0,
        min_region_mass: float = 4.0,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.warmup_batches = warmup_batches
        self.std_weight = std_weight
        self.white_threshold = white_threshold
        self.tpaf_fg_threshold = tpaf_fg_threshold
        self.sharpness = sharpness
        self.min_region_mass = min_region_mass

        if region_weights is None:
            region_weights = {"nuclear": 1.0, "cytoplasm": 1.5, "background": 0.5}
        self.region_weights: Dict[str, float] = {
            r: float(region_weights.get(r, 0.0)) for r in _REGIONS
        }

        # Per-region running statistics — buffers (device-aware, checkpointed).
        for r in _REGIONS:
            self.register_buffer(f"running_mean_{r}", torch.zeros(3))
            self.register_buffer(f"running_std_{r}", torch.ones(3))
        # Shared update counter (number of update_stats calls actually applied).
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))

        # Detached per-region coverage from the last update, for logging.
        self._last_coverage: Dict[str, float] = {r: 0.0 for r in _REGIONS}

    # ------------------------------------------------------------------
    # Region weight construction
    # ------------------------------------------------------------------

    def _real_region_weights(
        self, real_he: Tensor, real_nuc_mask: Tensor
    ) -> Dict[str, Tensor]:
        """Soft region weight maps for a batch of real H&E images.

        Args:
            real_he: Real H&E ``(B, 3, H, W)`` in ``[-1, 1]``.
            real_nuc_mask: Soft nuclear mask extracted from ``real_he`` (e.g.
                via ``StructureConsistencyLoss.extract_mask``), ``(B, 1, H, W)``
                in ``[0, 1]``; high = nuclear.

        Returns:
            Dict region → weight map ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        rgb_01 = real_he * 0.5 + 0.5
        lum = _luminance(rgb_01)
        nuc = real_nuc_mask.clamp(0.0, 1.0)
        # Bright, non-nuclear pixels are white background.
        white = torch.sigmoid((lum - self.white_threshold) * self.sharpness)
        bg = white * (1.0 - nuc)
        cyto = (1.0 - nuc) * (1.0 - white)
        return {"nuclear": nuc, "cytoplasm": cyto, "background": bg}

    def _generated_region_weights(
        self, tpaf: Tensor, mask: Tensor
    ) -> Dict[str, Tensor]:
        """Soft region weight maps for the generated image, from TPAF + mask.

        The weights are constants with respect to the generator — they depend
        only on the (fixed) TPAF input and its precomputed nuclear mask — so no
        gradient flows through them.

        Args:
            tpaf: TPAF input ``(B, 1, H, W)`` in ``[0, 1]``.
            mask: Precomputed nuclear mask ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            Dict region → weight map ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        nuc = mask.clamp(0.0, 1.0)
        # Bright, non-nuclear TPAF = cytoplasm (signal); dark = background/lumen.
        fg = torch.sigmoid((tpaf - self.tpaf_fg_threshold) * self.sharpness)
        cyto = (1.0 - nuc) * fg
        bg = (1.0 - nuc) * (1.0 - fg)
        return {"nuclear": nuc, "cytoplasm": cyto, "background": bg}

    # ------------------------------------------------------------------
    # EMA update (real side)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_stats(self, real_he: Tensor, real_nuc_mask: Tensor) -> None:
        """Update the per-region EMA statistics from a batch of real H&E.

        Call once per training iteration *before* computing the loss.

        Args:
            real_he: Real H&E ``(B, 3, H, W)`` in ``[-1, 1]``.
            real_nuc_mask: Soft nuclear mask extracted from ``real_he``
                (detached), ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        weights = self._real_region_weights(real_he, real_nuc_mask)
        first = self.n_updates == 0

        for r in _REGIONS:
            w = weights[r]
            mean, std, wsum = _weighted_channel_stats(real_he, w)
            self._last_coverage[r] = (wsum / real_he[:, :1].numel()).item()

            if wsum < self.min_region_mass:
                continue  # too few pixels this batch — skip to avoid noise

            run_mean = getattr(self, f"running_mean_{r}")
            run_std = getattr(self, f"running_std_{r}")
            if first:
                run_mean.copy_(mean)
                run_std.copy_(std)
            else:
                run_mean.lerp_(mean, self.momentum)
                run_std.lerp_(std, self.momentum)

        self.n_updates += 1

    # ------------------------------------------------------------------
    # Forward (generated side)
    # ------------------------------------------------------------------

    def forward(self, generated_he: Tensor, tpaf: Tensor, mask: Tensor) -> Tensor:
        """Compute the region-conditioned colour matching loss.

        Args:
            generated_he: Generated H&E ``(B, 3, H, W)`` in ``[-1, 1]`` (carries
                the gradient to the generator).
            tpaf: TPAF input ``(B, 1, H, W)`` in ``[0, 1]``.
            mask: Precomputed nuclear mask ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            Scalar loss tensor.  Returns zero during EMA warm-up.
        """
        if self.n_updates < self.warmup_batches:
            return torch.zeros((), device=generated_he.device, dtype=generated_he.dtype)

        weights = self._generated_region_weights(tpaf.detach(), mask.detach())

        total = torch.zeros((), device=generated_he.device, dtype=generated_he.dtype)
        for r in _REGIONS:
            lam = self.region_weights[r]
            if lam <= 0.0:
                continue
            w = weights[r]
            if w.sum() < self.min_region_mass:
                continue  # region essentially absent in this batch
            gen_mean, gen_std, _ = _weighted_channel_stats(generated_he, w)
            target_mean = getattr(self, f"running_mean_{r}")
            target_std = getattr(self, f"running_std_{r}")
            region_loss = F.l1_loss(gen_mean, target_mean) + self.std_weight * F.l1_loss(
                gen_std, target_std
            )
            total = total + lam * region_loss

        return total

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def get_last_coverage(self) -> Dict[str, float]:
        """Return the per-region pixel-coverage fraction from the last update.

        Useful for TensorBoard debugging — if ``cytoplasm`` coverage is near
        zero the background / foreground thresholds are likely mis-set.

        Returns:
            Dict region → coverage fraction in ``[0, 1]``.
        """
        return dict(self._last_coverage)
