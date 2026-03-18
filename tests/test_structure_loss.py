"""Unit tests for losses/structure_loss.py.

Run with::

    python -m pytest tests/test_structure_loss.py -v
    python -m unittest tests.test_structure_loss -v

Test coverage
-------------
Soft Dice loss (standalone function)
  1. Loss is exactly 0 when pred == target (binary masks).
  2. Loss is close to 1 when pred and target are perfectly disjoint.
  3. Loss is in (0, 1) for partially overlapping masks.
  4. Loss is differentiable — gradient flows through pred.

HED decomposition
  5. White pixels (no stain) produce near-zero H channel.
  6. A synthetically constructed "purple" pixel produces a higher H channel
     than a "pink" eosin pixel.
  7. Output shape matches input (B, 3, H, W).
  8. All stain values are non-negative.

StructureConsistencyLoss — threshold mode
  9.  Loss is low when generated image consists of hematoxylin-coloured
      nuclei over a white background, and the input mask marks those nuclei.
  10. Loss is high when the generated image and input mask are spatially
      anti-correlated (eosin where mask=1, hematoxylin where mask=0).
  11. Gradient flows back to the generated_he input tensor.
  12. Warm-up: forward returns zero (no gradient) before warmup_epochs.
  13. Ramp-up: weight increases linearly from 0 to 1 over ramp_epochs.
  14. get_last_hem_mask returns None before forward is called; a valid
      tensor afterwards.
  15. Learnable threshold: threshold_val is in model parameters.
  16. Non-learnable threshold: threshold_val is NOT in parameters.
  17. Unknown mode raises ValueError at construction time.
  18. pretrained_detector mode raises NotImplementedError.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.structure_loss import (  # noqa: E402
    StructureConsistencyLoss,
    _HED_FROM_RGB,
    rgb_to_hed,
    soft_dice_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cpu")


def _make_loss(**kwargs) -> StructureConsistencyLoss:
    defaults = dict(
        mode="threshold",
        threshold=0.15,
        sharpness=15.0,
        learnable_threshold=False,
        smooth=1.0,
        warmup_epochs=5,
        ramp_epochs=5,
    )
    defaults.update(kwargs)
    return StructureConsistencyLoss(**defaults)


def _hed_matrix() -> torch.Tensor:
    return _HED_FROM_RGB.clone()


def _hem_rgb_01() -> torch.Tensor:
    """Return approximate RGB [0,1] colour of concentrated hematoxylin stain.

    Based on Beer–Lambert with OD vector [0.65, 0.70, 0.29] at concentration 2:
        RGB = exp(-[0.65, 0.70, 0.29] * 2) ≈ [0.27, 0.25, 0.56]
    This is a dark purple-blue — strongly hematoxylin.
    """
    return torch.tensor([[[0.27]], [[0.25]], [[0.56]]])  # (3, 1, 1) broadcastable


def _eos_rgb_01() -> torch.Tensor:
    """Return approximate RGB [0,1] colour of eosin stain.

    Based on Beer–Lambert with OD vector [0.07, 0.99, 0.11] at concentration 2:
        RGB = exp(-[0.07, 0.99, 0.11] * 2) ≈ [0.87, 0.14, 0.80]
    This is a pinkish-magenta.
    """
    return torch.tensor([[[0.87]], [[0.14]], [[0.80]]])  # (3, 1, 1) broadcastable


# ---------------------------------------------------------------------------
# 1–4. soft_dice_loss
# ---------------------------------------------------------------------------


class TestSoftDiceLoss(unittest.TestCase):
    """Tests for the standalone soft_dice_loss function."""

    def test_zero_loss_on_identical_binary_masks(self) -> None:
        """Dice loss is exactly 0 when pred == target (binary values).

        For binary inputs p == g ∈ {0, 1}:
            intersection = Σ p² = Σ p
            cardinality  = 2 Σ p
            Dice = 1 − (2Σp + s)/(2Σp + s) = 0   for any smooth s.
        """
        for fill in [0.0, 1.0]:
            with self.subTest(fill=fill):
                mask = torch.full((2, 1, 64, 64), fill)
                loss = soft_dice_loss(mask, mask, smooth=1.0)
                self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_near_one_loss_for_disjoint_masks(self) -> None:
        """Dice loss approaches 1 for perfectly disjoint binary masks."""
        pred = torch.zeros(1, 1, 8, 8)
        pred[:, :, :4, :] = 1.0   # top half
        tgt = torch.zeros(1, 1, 8, 8)
        tgt[:, :, 4:, :] = 1.0    # bottom half

        loss = soft_dice_loss(pred, tgt, smooth=0.0)
        # intersection = 0  →  Dice = 1 − 0 / (sum_pred + sum_tgt) = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_partial_overlap_in_open_interval(self) -> None:
        """Dice loss is strictly between 0 and 1 for partial overlap."""
        pred = torch.zeros(1, 1, 4, 4)
        tgt = torch.zeros(1, 1, 4, 4)
        pred[:, :, :2, :] = 1.0  # top two rows
        tgt[:, :, 1:3, :] = 1.0  # rows 1–2 (one row shared)
        loss = soft_dice_loss(pred, tgt, smooth=0.0)
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 1.0)

    def test_gradient_flows_through_pred(self) -> None:
        """Gradient must be defined on pred after backward."""
        pred = torch.rand(1, 1, 8, 8, requires_grad=True)
        tgt = torch.randint(0, 2, (1, 1, 8, 8)).float()
        loss = soft_dice_loss(pred, tgt)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(pred.grad.abs().sum().item() == 0.0)


# ---------------------------------------------------------------------------
# 5–8. rgb_to_hed
# ---------------------------------------------------------------------------


class TestRgbToHed(unittest.TestCase):
    """Tests for the differentiable HED decomposition."""

    def test_white_pixel_produces_zero_stains(self) -> None:
        """White background (RGB=[1,1,1]) yields near-zero stain concentrations."""
        white = torch.ones(1, 3, 4, 4)
        hed = rgb_to_hed(white, _hed_matrix())
        self.assertTrue(
            (hed < 0.01).all().item(),
            f"White pixel HED values should be ~0, got max={hed.max().item():.4f}",
        )

    def test_hem_channel_higher_for_purple_than_pink(self) -> None:
        """Hematoxylin channel must be higher for purple than for eosin-pink."""
        hem_pixel = _hem_rgb_01().unsqueeze(0).expand(1, 3, 4, 4)
        eos_pixel = _eos_rgb_01().unsqueeze(0).expand(1, 3, 4, 4)

        hed_hem = rgb_to_hed(hem_pixel, _hed_matrix())
        hed_eos = rgb_to_hed(eos_pixel, _hed_matrix())

        h_hem = hed_hem[:, 0].mean().item()
        h_eos = hed_eos[:, 0].mean().item()
        self.assertGreater(
            h_hem,
            h_eos,
            f"H-channel: hematoxylin ({h_hem:.4f}) should exceed eosin ({h_eos:.4f})",
        )

    def test_output_shape(self) -> None:
        """Output shape matches (B, 3, H, W) of the input."""
        rgb = torch.rand(3, 3, 32, 32)
        hed = rgb_to_hed(rgb, _hed_matrix())
        self.assertEqual(tuple(hed.shape), (3, 3, 32, 32))

    def test_stains_non_negative(self) -> None:
        """All stain concentration values must be ≥ 0."""
        rgb = torch.rand(2, 3, 16, 16)
        hed = rgb_to_hed(rgb, _hed_matrix())
        self.assertTrue((hed >= 0).all().item(), "Negative stain concentrations found.")


# ---------------------------------------------------------------------------
# 9–18. StructureConsistencyLoss
# ---------------------------------------------------------------------------


class TestStructureConsistencyLoss(unittest.TestCase):
    """Integration-level tests for StructureConsistencyLoss."""

    # ------------------------------------------------------------------
    # 9. Low loss when image/mask are semantically consistent
    # ------------------------------------------------------------------

    def test_low_loss_when_nuclei_are_hematoxylin(self) -> None:
        """Loss is low when nuclear pixels are hematoxylin-coloured.

        Construct a 2-region image:
          * Nuclear patch (mask=1): dark purple RGB ≈ [0.27, 0.25, 0.56]
          * Stroma patch  (mask=0): white RGB = [1, 1, 1]
        After HED extraction the H channel should be high in the nuclear
        patch and near-zero in the stroma → extracted mask ≈ input mask
        → Dice loss ≈ 0.
        """
        B, H, W = 1, 16, 16
        half = W // 2

        # Build image in [0, 1] then convert to [-1, 1]
        img_01 = torch.ones(B, 3, H, W)
        hem_rgb = _hem_rgb_01()  # (3, 1, 1)
        img_01[:, :, :, :half] = hem_rgb.expand(B, 3, H, half)

        gen_he = img_01 * 2.0 - 1.0  # [-1, 1]
        gen_he.requires_grad_(True)

        mask = torch.zeros(B, 1, H, W)
        mask[:, :, :, :half] = 1.0  # same nuclear half

        loss_fn = _make_loss(
            threshold=0.05,   # low threshold captures concentrated stain
            sharpness=30.0,
            smooth=1.0,
            warmup_epochs=0,
            ramp_epochs=1,
        )
        loss = loss_fn(gen_he, mask, epoch=1)
        self.assertLess(loss.item(), 0.3, f"Expected low loss, got {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 10. High loss when mask is anti-correlated with H-channel
    # ------------------------------------------------------------------

    def test_high_loss_when_masks_are_anticorrelated(self) -> None:
        """Loss is high when eosin-pink fills the nuclear mask region.

        Nuclear mask marks the LEFT half, but the generated image has
        eosin (pink, low H-channel) on the LEFT and hematoxylin (purple)
        on the RIGHT.  The extracted mask is roughly the opposite of the
        input mask → high Dice loss.
        """
        B, H, W = 1, 16, 16
        half = W // 2

        # Image: left=eosin (pink), right=hematoxylin (purple)
        img_01 = torch.ones(B, 3, H, W)
        img_01[:, :, :, :half] = _eos_rgb_01().expand(B, 3, H, half)
        img_01[:, :, :, half:] = _hem_rgb_01().expand(B, 3, H, half)

        gen_he = img_01 * 2.0 - 1.0
        gen_he.requires_grad_(True)

        # Mask marks LEFT (eosin) half as "nuclear"
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, :, :half] = 1.0

        loss_fn = _make_loss(
            threshold=0.05,
            sharpness=30.0,
            smooth=0.0,
            warmup_epochs=0,
            ramp_epochs=1,
        )
        loss = loss_fn(gen_he, mask, epoch=1)
        self.assertGreater(loss.item(), 0.5, f"Expected high loss, got {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 11. Gradient flows back to generated_he
    # ------------------------------------------------------------------

    def test_gradient_flows_to_generated_image(self) -> None:
        """Gradient of the loss must reach generated_he (end-to-end differentiable)."""
        gen_he = torch.randn(1, 3, 16, 16, requires_grad=True)
        mask = torch.randint(0, 2, (1, 1, 16, 16)).float()

        loss_fn = _make_loss(warmup_epochs=0, ramp_epochs=1)
        loss = loss_fn(gen_he, mask, epoch=1)
        loss.backward()

        self.assertIsNotNone(gen_he.grad, "gen_he.grad is None — gradient did not flow.")
        self.assertGreater(
            gen_he.grad.abs().sum().item(),
            0.0,
            "gen_he.grad is all zeros — gradient did not flow.",
        )

    # ------------------------------------------------------------------
    # 12. Warm-up returns zero loss (no gradient)
    # ------------------------------------------------------------------

    def test_warmup_returns_zero_detached(self) -> None:
        """Before warmup_epochs, forward returns a detached scalar zero."""
        gen_he = torch.randn(1, 3, 8, 8, requires_grad=True)
        mask = torch.ones(1, 1, 8, 8)

        loss_fn = _make_loss(warmup_epochs=10, ramp_epochs=5)
        for epoch in range(10):
            loss = loss_fn(gen_he, mask, epoch=epoch)
            self.assertAlmostEqual(loss.item(), 0.0, places=6, msg=f"epoch={epoch}")
            self.assertFalse(
                loss.requires_grad,
                f"Warmup loss should be detached (no grad) at epoch={epoch}.",
            )

    # ------------------------------------------------------------------
    # 13. Ramp-up increases weight linearly
    # ------------------------------------------------------------------

    def test_ramp_up_schedule(self) -> None:
        """schedule_weight increases linearly from 0 to 1 over ramp_epochs."""
        loss_fn = _make_loss(warmup_epochs=10, ramp_epochs=5)

        self.assertAlmostEqual(loss_fn.schedule_weight(9), 0.0)   # last warmup epoch
        self.assertAlmostEqual(loss_fn.schedule_weight(10), 0.0)  # first ramp epoch
        self.assertAlmostEqual(loss_fn.schedule_weight(11), 0.2)  # 1/5 of ramp
        self.assertAlmostEqual(loss_fn.schedule_weight(12), 0.4)  # 2/5
        self.assertAlmostEqual(loss_fn.schedule_weight(13), 0.6)  # 3/5
        self.assertAlmostEqual(loss_fn.schedule_weight(14), 0.8)  # 4/5
        self.assertAlmostEqual(loss_fn.schedule_weight(15), 1.0)  # full weight
        self.assertAlmostEqual(loss_fn.schedule_weight(100), 1.0) # beyond ramp

    def test_ramp_loss_is_scaled(self) -> None:
        """Loss at ramp mid-point is half the loss at full weight."""
        gen_he = torch.randn(1, 3, 8, 8)
        mask = torch.randint(0, 2, (1, 1, 8, 8)).float()

        loss_fn = _make_loss(warmup_epochs=0, ramp_epochs=10)
        loss_full = loss_fn(gen_he, mask, epoch=10).item()  # weight=1.0
        loss_half = loss_fn(gen_he, mask, epoch=5).item()   # weight=0.5
        self.assertAlmostEqual(loss_half, loss_full * 0.5, places=5)

    # ------------------------------------------------------------------
    # 14. get_last_hem_mask
    # ------------------------------------------------------------------

    def test_last_hem_mask_none_before_forward(self) -> None:
        """get_last_hem_mask returns None before the first forward call."""
        loss_fn = _make_loss()
        self.assertIsNone(loss_fn.get_last_hem_mask())

    def test_last_hem_mask_valid_after_forward(self) -> None:
        """get_last_hem_mask returns a (B, 1, H, W) tensor after forward."""
        loss_fn = _make_loss(warmup_epochs=0, ramp_epochs=1)
        gen_he = torch.randn(2, 3, 16, 16)
        mask = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss_fn(gen_he, mask, epoch=1)

        hem = loss_fn.get_last_hem_mask()
        self.assertIsNotNone(hem)
        self.assertEqual(tuple(hem.shape), (2, 1, 16, 16))
        self.assertFalse(hem.requires_grad, "Cached hem mask should be detached.")

    def test_last_hem_mask_none_in_warmup(self) -> None:
        """get_last_hem_mask stays None during the warm-up phase."""
        loss_fn = _make_loss(warmup_epochs=10)
        gen_he = torch.randn(1, 3, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        loss_fn(gen_he, mask, epoch=3)  # still in warmup
        self.assertIsNone(loss_fn.get_last_hem_mask())

    # ------------------------------------------------------------------
    # 15–16. Learnable vs fixed threshold
    # ------------------------------------------------------------------

    def test_learnable_threshold_is_in_parameters(self) -> None:
        """threshold_val and sharpness_val appear in parameters() when learnable."""
        loss_fn = _make_loss(learnable_threshold=True)
        param_names = [name for name, _ in loss_fn.named_parameters()]
        self.assertIn("threshold_val", param_names)
        self.assertIn("sharpness_val", param_names)
        self.assertTrue(loss_fn.threshold_val.requires_grad)

    def test_fixed_threshold_not_in_trainable_parameters(self) -> None:
        """threshold_val has requires_grad=False when learnable_threshold=False."""
        loss_fn = _make_loss(learnable_threshold=False)
        self.assertFalse(loss_fn.threshold_val.requires_grad)
        self.assertFalse(loss_fn.sharpness_val.requires_grad)

        # Should not appear in list(parameters()) since requires_grad=False
        trainable = [p for p in loss_fn.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 0)

    # ------------------------------------------------------------------
    # 17–18. Invalid modes
    # ------------------------------------------------------------------

    def test_unknown_mode_raises_value_error(self) -> None:
        """ValueError is raised for an unrecognised mode string."""
        with self.assertRaises(ValueError) as ctx:
            StructureConsistencyLoss(mode="magic_detector")
        self.assertIn("magic_detector", str(ctx.exception))

    def test_pretrained_detector_raises_not_implemented(self) -> None:
        """Attempting to build a 'pretrained_detector' loss raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            StructureConsistencyLoss(mode="pretrained_detector")

    # ------------------------------------------------------------------
    # General sanity
    # ------------------------------------------------------------------

    def test_output_is_scalar(self) -> None:
        loss_fn = _make_loss(warmup_epochs=0, ramp_epochs=1)
        gen_he = torch.randn(2, 3, 32, 32)
        mask = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(gen_he, mask, epoch=1)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_finite(self) -> None:
        loss_fn = _make_loss(warmup_epochs=0, ramp_epochs=1)
        gen_he = torch.randn(1, 3, 16, 16)
        mask = torch.randint(0, 2, (1, 1, 16, 16)).float()
        loss = loss_fn(gen_he, mask, epoch=1)
        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")

    def test_hed_matrix_on_correct_device(self) -> None:
        """HED buffer must move to the same device as the loss module."""
        loss_fn = _make_loss()
        # On CPU by default
        self.assertEqual(loss_fn.hed_matrix.device.type, "cpu")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
