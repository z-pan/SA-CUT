"""Color statistics matching loss for SA-CUT.

Matches the channel-wise mean and standard deviation of generated H&E images
to a running estimate of the real H&E distribution.  This provides an explicit
colour guidance signal that is missing from the other SA-CUT losses:

  * ``L_struct`` constrains nuclear (dark) regions but is neutral on eosin.
  * ``L_adv`` should enforce realistic colours, but the discriminator
    consistently trains too fast, saturating its colour gradients.
  * ``L_NCE`` preserves local structure, not colour.
  * ``L_idt`` is near-zero once the generator learns identity reconstruction.

By directly penalising deviation from the real H&E colour moments, this loss
pushes the generator toward producing pink eosin in stromal/cytoplasm regions
instead of defaulting to white.

The loss operates in the generator output space (``[-1, 1]``).  Running
statistics are updated with an exponential moving average (EMA) so that the
target adapts across batches without requiring a pre-computed dataset pass.

Usage::

    criterion_color = ColorStatsLoss(momentum=0.01).to(device)

    # In training loop:
    criterion_color.update_stats(real_he)                # update EMA
    loss_color = criterion_color(fake_he) * lambda_color  # compute loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ColorStatsLoss(nn.Module):
    """Channel-wise colour statistics matching loss.

    Maintains an exponential moving average (EMA) of the per-channel mean and
    standard deviation of real H&E images, then penalises generated images
    whose statistics deviate from this target.

    Loss = L1(gen_mean, ema_mean) + L1(gen_std, ema_std)

    Both terms are averaged over the 3 RGB channels, producing a scalar.

    Args:
        momentum: EMA update rate.  Lower values make the target more stable;
            higher values track recent batches more closely.  Default ``0.01``.
        warmup_batches: Number of ``update_stats`` calls before the loss
            activates.  During warm-up, ``forward`` returns zero so that the
            EMA has time to converge to a reasonable estimate.  Default ``50``.
    """

    def __init__(
        self,
        momentum: float = 0.01,
        warmup_batches: int = 50,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.warmup_batches = warmup_batches

        # Running statistics — buffers (device-aware, saved in checkpoints)
        self.register_buffer("running_mean", torch.zeros(3))
        self.register_buffer("running_std", torch.ones(3))
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_stats(self, real_he: Tensor) -> None:
        """Update running colour statistics with a batch of real H&E images.

        Call once per training iteration *before* computing the loss.

        Args:
            real_he: Real H&E image tensor, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]``.
        """
        batch_mean = real_he.mean(dim=[0, 2, 3])  # (3,)
        batch_std = real_he.std(dim=[0, 2, 3])     # (3,)

        if self.n_updates == 0:
            # First call: initialise directly instead of lerping from zeros
            self.running_mean.copy_(batch_mean)
            self.running_std.copy_(batch_std)
        else:
            self.running_mean.lerp_(batch_mean, self.momentum)
            self.running_std.lerp_(batch_std, self.momentum)

        self.n_updates += 1

    def forward(self, generated_he: Tensor) -> Tensor:
        """Compute colour statistics matching loss.

        Args:
            generated_he: Generated H&E image, shape ``(B, 3, H, W)``,
                values in ``[-1, 1]``.

        Returns:
            Scalar loss tensor.  Returns zero during EMA warm-up.
        """
        if self.n_updates < self.warmup_batches:
            return torch.zeros((), device=generated_he.device,
                               dtype=generated_he.dtype)

        gen_mean = generated_he.mean(dim=[2, 3])  # (B, 3)
        gen_std = generated_he.std(dim=[2, 3])     # (B, 3)

        target_mean = self.running_mean.unsqueeze(0)  # (1, 3)
        target_std = self.running_std.unsqueeze(0)    # (1, 3)

        loss = (
            F.l1_loss(gen_mean, target_mean.expand_as(gen_mean))
            + F.l1_loss(gen_std, target_std.expand_as(gen_std))
        )
        return loss
