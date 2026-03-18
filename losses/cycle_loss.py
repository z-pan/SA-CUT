"""Cycle-consistency and identity losses for the CycleGAN ablation baseline.

These losses are **not** used in the main SA-CUT model.  They are provided
solely for the ``ablation_cyclegan`` experiment so that the CycleGAN baseline
can be trained within the same framework.

Background
----------
CycleGAN (Zhu et al., ICCV 2017) trains two generators simultaneously:

* ``G_A``: domain A → domain B  (here: TPAF → virtual H&E)
* ``G_B``: domain B → domain A  (here: virtual H&E → TPAF)

and enforces that the composition of the two generators is close to the
identity in each domain:

    forward cycle:  ``||G_B(G_A(x_A)) − x_A||₁ ≤ ε``
    backward cycle: ``||G_A(G_B(x_B)) − x_B||₁ ≤ ε``

This prevents the generators from mapping every input to the same output
(mode collapse) and provides the only cross-domain structural signal in the
absence of paired training data.

An optional **identity loss** (Zhu et al., Section 5.2) is recommended when
the input and output domains share colour information (e.g. painting → photo).
It penalises the generator for changing the colour of images that are already
in the target domain:

    ``||G_A(x_B) − x_B||₁``   (G_A should act as identity on H&E input)
    ``||G_B(x_A) − x_A||₁``   (G_B should act as identity on TPAF input)

In the TPAF → H&E setting the identity loss is *less* critical than in the
painting task because the domains differ substantially in colour.  The loss
is controlled via ``losses.lambda_idt`` and can be set to ``0.0`` to disable.

Loss weighting (from Zhu et al., Table 1)
------------------------------------------
Recommended: ``lambda_cycle = 10.0``, ``lambda_idt = 0.5 * lambda_cycle = 5.0``.
These are set in ``configs/ablation_cyclegan.yaml`` and should not be
hard-coded here.

Module design
-------------
Both losses are thin wrappers around ``F.l1_loss``.  The trainer is
responsible for:

1. Running the forward passes to obtain ``fake_B = G_A(real_A)`` and
   ``fake_A = G_B(real_B)``.
2. Running the backward passes to obtain ``rec_A = G_B(fake_B)`` and
   ``rec_B = G_A(fake_A)``.
3. Calling ``CycleConsistencyLoss.forward(real_A, rec_A)`` and
   ``CycleConsistencyLoss.forward(real_B, rec_B)`` separately, then summing
   and scaling by ``lambda_cycle``.
4. Optionally calling ``IdentityLoss.forward(real_B, G_A(real_B))`` and
   ``IdentityLoss.forward(real_A, G_B(real_A))``, scaled by ``lambda_idt``.

Domain-specific normalisation note
------------------------------------
The TPAF domain uses 1-channel float32 input in ``[0, 1]`` (after percentile
normalisation — **never** min-max normalisation, per CLAUDE.md).  The H&E
domain uses 3-channel float32 in ``[-1, 1]`` (tanh generator output range).
L1 loss is scale-invariant up to a constant, so both domains can share the
same loss module with no changes.  The trainer must ensure the ``source``
and ``reconstructed`` tensors are on the same device and have consistent
normalisation before calling ``forward()``.

References
----------
* Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
  Adversarial Networks," ICCV 2017.
  https://arxiv.org/abs/1703.10593
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CycleConsistencyLoss(nn.Module):
    """Per-direction L1 reconstruction loss for CycleGAN cycle consistency.

    Computes the mean absolute error between *source* (the original domain
    image) and *reconstructed* (the image after two generator passes):

        ``L_cyc_A = ||G_B(G_A(x_A)) − x_A||₁``

    The caller is responsible for scaling by ``lambda_cycle`` and summing
    both directions.

    Args:
        reduction: Passed to ``F.l1_loss``.  Default ``'mean'``.

    Example::

        cycle_loss = CycleConsistencyLoss()

        fake_he   = G_tpaf2he(real_tpaf)   # (B, 3, H, W)  in [-1, 1]
        rec_tpaf  = G_he2tpaf(fake_he)     # (B, 1, H, W)  in [ 0, 1]

        fake_tpaf = G_he2tpaf(real_he)     # (B, 1, H, W)
        rec_he    = G_tpaf2he(fake_tpaf)   # (B, 3, H, W)

        loss = lambda_cycle * (
            cycle_loss(real_tpaf, rec_tpaf) +
            cycle_loss(real_he,   rec_he)
        )
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction='{reduction}' is not valid. Choose 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, source: Tensor, reconstructed: Tensor) -> Tensor:
        """Compute the L1 cycle-consistency loss for one direction.

        Args:
            source: Original domain image, shape ``(B, C, H, W)``.
                Values in ``[0, 1]`` for TPAF or ``[-1, 1]`` for H&E.
            reconstructed: Reconstructed image after two generator passes,
                same shape and normalisation as *source*.

        Returns:
            Scalar L1 loss tensor (or elementwise if ``reduction='none'``).
        """
        return F.l1_loss(reconstructed, source, reduction=self.reduction)


class IdentityLoss(nn.Module):
    """L1 identity-mapping loss for CycleGAN colour preservation.

    Penalises a generator for modifying an image that is already in its
    target domain.  For generator ``G_A: A → B``, the identity loss is:

        ``L_idt_A = ||G_A(x_B) − x_B||₁``

    Useful when the two domains share colour information, preventing the
    generators from gratuitously altering hue.  In the TPAF → H&E setting
    this encourages ``G_TPAF2HE`` to preserve H&E staining when fed a real
    H&E patch, providing a weak colour-anchor that complements the adversarial
    loss.

    Set ``losses.lambda_idt = 0.0`` in the experiment config to disable.

    Args:
        reduction: Passed to ``F.l1_loss``.  Default ``'mean'``.

    Example::

        idt_loss = IdentityLoss()

        # G_tpaf2he should map real H&E → real H&E unchanged
        idt_he   = G_tpaf2he(real_he)
        loss_idt = lambda_idt * idt_loss(real_he, idt_he)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction='{reduction}' is not valid. Choose 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, source: Tensor, generated: Tensor) -> Tensor:
        """Compute the L1 identity loss.

        Args:
            source: Real image already in the target domain, shape
                ``(B, C, H, W)``.
            generated: Output of the generator given *source* as input,
                same shape as *source*.

        Returns:
            Scalar L1 loss tensor.
        """
        return F.l1_loss(generated, source, reduction=self.reduction)


class CycleLoss(nn.Module):
    """Combined CycleGAN cycle-consistency + identity loss module.

    Convenience wrapper that holds :class:`CycleConsistencyLoss` and
    :class:`IdentityLoss` as submodules and exposes a single ``forward``
    method matching the calling convention used in the CycleGAN trainer.

    This module is *not* used in the SA-CUT model.  It exists only for the
    ``ablation_cyclegan`` experiment config.

    Args:
        lambda_cycle: Weight for the cycle-consistency terms.  Default ``10.0``
            (Zhu et al., 2017 recommendation).
        lambda_idt: Weight for the identity-mapping terms.  Default ``5.0``
            (= ``0.5 × lambda_cycle``).  Set ``0.0`` to disable.
        reduction: Passed to the sub-losses.  Default ``'mean'``.

    Example::

        cycle = CycleLoss(lambda_cycle=10.0, lambda_idt=5.0).to(device)

        loss, breakdown = cycle(
            real_A=real_tpaf,  rec_A=rec_tpaf,  idt_A=idt_tpaf,
            real_B=real_he,    rec_B=rec_he,    idt_B=idt_he,
        )
        loss.backward()
    """

    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_idt: float = 5.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_idt = lambda_idt
        self.cycle_loss = CycleConsistencyLoss(reduction=reduction)
        self.idt_loss = IdentityLoss(reduction=reduction)

    def forward(
        self,
        real_A: Tensor,
        rec_A: Tensor,
        idt_A: Tensor,
        real_B: Tensor,
        rec_B: Tensor,
        idt_B: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined cycle + identity loss.

        Args:
            real_A: Original domain-A images (e.g. TPAF), ``(B, C_A, H, W)``.
            rec_A:  Reconstructed domain-A images: ``G_B(G_A(real_A))``.
            idt_A:  Identity output for domain-A generator: ``G_B(real_A)``.
                Must have the same shape as *real_A*.
            real_B: Original domain-B images (e.g. H&E), ``(B, C_B, H, W)``.
            rec_B:  Reconstructed domain-B images: ``G_A(G_B(real_B))``.
            idt_B:  Identity output for domain-B generator: ``G_A(real_B)``.
                Must have the same shape as *real_B*.

        Returns:
            ``(total_loss, breakdown_dict)`` where *total_loss* is a scalar
            tensor and *breakdown_dict* maps component names to Python floats
            suitable for logging::

                {
                    'cycle_A': float,   # L_cyc for domain A
                    'cycle_B': float,   # L_cyc for domain B
                    'idt_A':   float,   # L_idt for domain A  (0 if lambda_idt=0)
                    'idt_B':   float,   # L_idt for domain B  (0 if lambda_idt=0)
                    'total':   float,   # weighted sum (= total_loss.item())
                }
        """
        loss_cyc_A = self.cycle_loss(real_A, rec_A)
        loss_cyc_B = self.cycle_loss(real_B, rec_B)

        total = self.lambda_cycle * (loss_cyc_A + loss_cyc_B)

        if self.lambda_idt > 0.0:
            loss_idt_A = self.idt_loss(real_A, idt_A)
            loss_idt_B = self.idt_loss(real_B, idt_B)
            total = total + self.lambda_idt * (loss_idt_A + loss_idt_B)
        else:
            loss_idt_A = torch.zeros((), device=real_A.device, dtype=real_A.dtype)
            loss_idt_B = torch.zeros((), device=real_B.device, dtype=real_B.dtype)

        breakdown = {
            "cycle_A": loss_cyc_A.item(),
            "cycle_B": loss_cyc_B.item(),
            "idt_A":   loss_idt_A.item(),
            "idt_B":   loss_idt_B.item(),
            "total":   total.item(),
        }
        return total, breakdown
