"""GAN adversarial loss supporting LSGAN, vanilla GAN, and Wasserstein modes.

Usage::

    criterion = GANLoss(gan_mode='lsgan').to(device)

    # Discriminator update
    loss_D_real = criterion(D(real), target_is_real=True)
    loss_D_fake = criterion(D(fake.detach()), target_is_real=False)

    # Generator update
    loss_G = criterion(D(fake), target_is_real=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GANLoss(nn.Module):
    """Adversarial loss for GAN training.

    Supports three modes:

    * ``'lsgan'`` — Least-squares GAN (MSELoss). Real target = 1.0, fake = 0.0.
      Recommended: stable gradients, no vanishing.
    * ``'vanilla'`` — Original GAN (BCEWithLogitsLoss). Real = 1.0, fake = 0.0.
    * ``'wgangp'`` — Wasserstein GAN with gradient penalty. No sigmoid; the
      discriminator output is used directly (real: maximise, fake: minimise).

    Target value tensors are registered as **buffers** so they move with the
    module to the correct device via ``.to(device)`` and are excluded from
    optimizer parameter groups.

    Args:
        gan_mode: One of ``'lsgan'``, ``'vanilla'``, or ``'wgangp'``.
        real_label: Target value for real samples. Default ``1.0``.
        fake_label: Target value for fake samples. Default ``0.0``.

    Raises:
        NotImplementedError: If *gan_mode* is not recognised.
    """

    def __init__(
        self,
        gan_mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        if gan_mode not in {"lsgan", "vanilla", "wgangp"}:
            raise NotImplementedError(
                f"gan_mode '{gan_mode}' is not supported. "
                "Choose from: 'lsgan', 'vanilla', 'wgangp'."
            )

        self.gan_mode = gan_mode

        # Register as buffers — not parameters, but travel with .to(device)
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:  # wgangp — no criterion object needed
            self.loss = None

    def _target_tensor(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        """Expand the scalar target to match *prediction*'s shape.

        Args:
            prediction: Discriminator output tensor of any shape.
            target_is_real: Use the real-label value when ``True``.

        Returns:
            Tensor filled with the target value, same shape as *prediction*.
        """
        label = self.real_label if target_is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        """Compute the GAN loss.

        Args:
            prediction: Raw discriminator output (logits for vanilla/lsgan,
                unbounded scores for wgangp).  Shape is arbitrary — typically
                ``(N, 1, H, W)`` from a PatchGAN discriminator.
            target_is_real: If ``True``, treat *prediction* as coming from
                real data; otherwise as coming from generated (fake) data.

        Returns:
            Scalar loss tensor.
        """
        if self.gan_mode in {"lsgan", "vanilla"}:
            target = self._target_tensor(prediction, target_is_real)
            return self.loss(prediction, target)

        # wgangp: maximise D(real), minimise D(fake)  →  loss = ±mean(D(x))
        if target_is_real:
            return -prediction.mean()
        return prediction.mean()
