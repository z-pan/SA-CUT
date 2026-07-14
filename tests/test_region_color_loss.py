"""Unit tests for losses/region_color_loss.py.

Run with::

    python -m pytest tests/test_region_color_loss.py -v
    python -m unittest tests.test_region_color_loss -v

Test coverage
-------------
_weighted_channel_stats (standalone)
  1. Uniform weight reproduces the plain per-channel mean/std.
  2. A binary region weight selects only the masked pixels' statistics.
  3. Gradient flows through the value tensor (weights are constants).

RegionColorStatsLoss
  4.  Warm-up: forward returns exactly zero before warmup_batches updates.
  5.  After warm-up the loss is a positive scalar.
  6.  Gradient flows back to the generated image.
  7.  Region weight maps form a soft partition (sum ≈ 1 per pixel).
  8.  A region with zero loss-multiplier is excluded from the loss.
  9.  A near-empty region (mass < min_region_mass) is skipped, not NaN.
  10. Perfectly matched statistics give ~zero loss (per region).
  11. state_dict round-trips (per-region mean/std buffers + counter).
  12. get_last_coverage returns a fraction per region after an update.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.region_color_loss import (  # noqa: E402
    _REGIONS,
    RegionColorStatsLoss,
    _weighted_channel_stats,
)

_DEVICE = torch.device("cpu")


def _make_loss(**kwargs) -> RegionColorStatsLoss:
    defaults = dict(warmup_batches=1)
    defaults.update(kwargs)
    return RegionColorStatsLoss(**defaults).to(_DEVICE)


class TestWeightedChannelStats(unittest.TestCase):
    def test_uniform_weight_matches_plain_stats(self):
        x = torch.randn(2, 3, 16, 16)
        w = torch.ones(2, 1, 16, 16)
        mean, std, wsum = _weighted_channel_stats(x, w)
        plain_mean = x.mean(dim=(0, 2, 3))
        # population std (unbiased=False) matches the weighted formula
        plain_std = x.var(dim=(0, 2, 3), unbiased=False).sqrt()
        self.assertTrue(torch.allclose(mean, plain_mean, atol=1e-4))
        self.assertTrue(torch.allclose(std, plain_std, atol=1e-3))
        self.assertAlmostEqual(wsum.item(), 2 * 16 * 16, places=2)

    def test_binary_region_selects_pixels(self):
        x = torch.randn(1, 3, 8, 8)
        w = torch.zeros(1, 1, 8, 8)
        w[..., :4, :] = 1.0  # top half only
        mean, _, _ = _weighted_channel_stats(x, w)
        ref = x[:, :, :4, :].mean(dim=(0, 2, 3))
        self.assertTrue(torch.allclose(mean, ref, atol=1e-4))

    def test_gradient_flows_through_values(self):
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        w = torch.ones(1, 1, 8, 8)
        mean, std, _ = _weighted_channel_stats(x, w)
        (mean.sum() + std.sum()).backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


class TestRegionColorStatsLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.W = 2, 32, 32
        self.real = torch.rand(self.B, 3, self.H, self.W) * 2 - 1
        self.real_nuc = (torch.rand(self.B, 1, self.H, self.W) > 0.6).float()
        self.tpaf = torch.rand(self.B, 1, self.H, self.W)
        self.mask = (torch.rand(self.B, 1, self.H, self.W) > 0.6).float()

    def _gen(self, requires_grad=False):
        g = torch.rand(self.B, 3, self.H, self.W) * 2 - 1
        return g.requires_grad_(requires_grad)

    def test_warmup_returns_zero(self):
        loss_fn = _make_loss(warmup_batches=3)
        loss_fn.update_stats(self.real, self.real_nuc)
        out = loss_fn(self._gen(), self.tpaf, self.mask)
        self.assertEqual(out.item(), 0.0)

    def test_active_loss_is_positive_scalar(self):
        loss_fn = _make_loss(warmup_batches=1)
        loss_fn.update_stats(self.real, self.real_nuc)
        out = loss_fn(self._gen(), self.tpaf, self.mask)
        self.assertEqual(out.dim(), 0)
        self.assertGreater(out.item(), 0.0)

    def test_gradient_reaches_generated_image(self):
        loss_fn = _make_loss(warmup_batches=1)
        loss_fn.update_stats(self.real, self.real_nuc)
        gen = self._gen(requires_grad=True)
        loss_fn(gen, self.tpaf, self.mask).backward()
        self.assertIsNotNone(gen.grad)
        self.assertGreater(gen.grad.abs().sum().item(), 0.0)

    def test_region_weights_form_soft_partition(self):
        loss_fn = _make_loss()
        w = loss_fn._generated_region_weights(self.tpaf, self.mask)
        total = sum(w[r] for r in _REGIONS)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))
        w2 = loss_fn._real_region_weights(self.real, self.real_nuc)
        total2 = sum(w2[r] for r in _REGIONS)
        self.assertTrue(torch.allclose(total2, torch.ones_like(total2), atol=1e-5))

    def test_zero_multiplier_region_excluded(self):
        full = _make_loss(warmup_batches=1)
        full.update_stats(self.real, self.real_nuc)
        gen = self._gen()
        loss_full = full(gen, self.tpaf, self.mask).item()

        no_cyto = _make_loss(
            warmup_batches=1,
            region_weights={"nuclear": 1.0, "cytoplasm": 0.0, "background": 0.5},
        )
        no_cyto.update_stats(self.real, self.real_nuc)
        loss_no_cyto = no_cyto(gen, self.tpaf, self.mask).item()
        self.assertNotAlmostEqual(loss_full, loss_no_cyto, places=4)

    def test_empty_region_skipped_no_nan(self):
        loss_fn = _make_loss(warmup_batches=1)
        loss_fn.update_stats(self.real, self.real_nuc)
        # All-nuclear mask → cytoplasm & background have zero mass on gen side.
        full_mask = torch.ones(self.B, 1, self.H, self.W)
        out = loss_fn(self._gen(), self.tpaf, full_mask)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(out).all())

    def test_matched_stats_give_low_loss(self):
        loss_fn = _make_loss(warmup_batches=1, std_weight=1.0)
        # Feed the real image as both target and generated → per-region stats
        # should match closely (weights differ real vs gen, but overall small).
        for _ in range(5):
            loss_fn.update_stats(self.real, self.real_nuc)
        # Use the real image on the generated side with its own mask proxy.
        out = loss_fn(self.real, self.tpaf, self.real_nuc)
        self.assertLess(out.item(), 0.5)

    def test_state_dict_roundtrip(self):
        loss_fn = _make_loss(warmup_batches=1)
        loss_fn.update_stats(self.real, self.real_nuc)
        sd = loss_fn.state_dict()
        # 3 means + 3 stds + 1 counter
        self.assertEqual(len(sd), 2 * len(_REGIONS) + 1)
        clone = _make_loss(warmup_batches=1)
        clone.load_state_dict(sd)
        for r in _REGIONS:
            self.assertTrue(
                torch.allclose(
                    getattr(loss_fn, f"running_mean_{r}"),
                    getattr(clone, f"running_mean_{r}"),
                )
            )

    def test_coverage_reported_per_region(self):
        loss_fn = _make_loss(warmup_batches=1)
        loss_fn.update_stats(self.real, self.real_nuc)
        cov = loss_fn.get_last_coverage()
        self.assertEqual(set(cov.keys()), set(_REGIONS))
        self.assertAlmostEqual(sum(cov.values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
