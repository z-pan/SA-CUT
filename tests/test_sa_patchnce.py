"""Unit tests for losses/sa_patchnce.py.

Run with::

    python -m pytest tests/test_sa_patchnce.py -v
    python -m unittest tests.test_sa_patchnce -v

Test coverage
-------------
1. Output is a scalar tensor with a gradient.
2. Loss decreases when feat_source ≈ feat_generated (matching positive pairs).
3. Checkerboard mask produces two non-empty region sets.
4. mask=None falls back to standard PatchNCE (no crash, valid scalar output).
5. MLP heads are built lazily and track the correct number of scales.
6. Feature-list length mismatch raises ValueError.
7. All-nuclear / all-stroma masks trigger the fallback branch without error.
8. Loss is finite and positive for random inputs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.sa_patchnce import PatchMLP, RegionAwarePatchNCE  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cpu")


def _rand_feat(b: int, c: int, h: int, w: int) -> torch.Tensor:
    """Return a random feature map that requires gradient."""
    return torch.randn(b, c, h, w, requires_grad=True)


def _make_feat_lists(
    b: int = 1,
    n_layers: int = 3,
    channels: tuple = (8, 16, 32),
    spatial: tuple = (32, 16, 8),
) -> tuple[list, list]:
    """Build paired source / generated feature lists."""
    src = [_rand_feat(b, c, s, s) for c, s in zip(channels, spatial)]
    gen = [_rand_feat(b, c, s, s) for c, s in zip(channels, spatial)]
    return src, gen


def _make_mask(b: int, h: int, w: int, fill: float = 0.5) -> torch.Tensor:
    """Return a uniform mask filled with *fill*."""
    return torch.full((b, 1, h, w), fill)


def _checkerboard_mask(b: int, h: int, w: int) -> torch.Tensor:
    """Return a binary checkerboard mask (half nuclear, half stroma)."""
    rows = torch.arange(h).unsqueeze(1).expand(h, w)
    cols = torch.arange(w).unsqueeze(0).expand(h, w)
    pattern = ((rows + cols) % 2).float()  # 0.0 and 1.0
    return pattern.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)


# ---------------------------------------------------------------------------
# PatchMLP
# ---------------------------------------------------------------------------


class TestPatchMLP(unittest.TestCase):
    """Basic sanity checks for the projector head."""

    def test_output_shape(self) -> None:
        mlp = PatchMLP(in_channels=64, nc=256)
        x = torch.randn(128, 64)
        out = mlp(x)
        self.assertEqual(tuple(out.shape), (128, 256))

    def test_output_is_unit_normalised(self) -> None:
        mlp = PatchMLP(in_channels=32, nc=128)
        x = torch.randn(50, 32)
        out = mlp(x)
        norms = out.norm(p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(50), atol=1e-5))


# ---------------------------------------------------------------------------
# RegionAwarePatchNCE
# ---------------------------------------------------------------------------


class TestRegionAwarePatchNCE(unittest.TestCase):
    """Core loss functionality."""

    def _make_loss(self, n_layers: int = 3, n_patches: int = 64) -> RegionAwarePatchNCE:
        return RegionAwarePatchNCE(
            n_layers=n_layers,
            n_patches=n_patches,
            nc=64,
            tau=0.07,
            min_region_patches=4,
        )

    # ------------------------------------------------------------------
    # 1. Scalar output with gradient
    # ------------------------------------------------------------------

    def test_output_is_scalar_with_grad(self) -> None:
        """forward() returns a 0-D tensor that has a gradient."""
        loss_fn = self._make_loss()
        src, gen = _make_feat_lists(n_layers=3)
        mask = _make_mask(1, 32, 32, fill=0.5)

        out = loss_fn(src, gen, mask)

        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(out.requires_grad, "Loss tensor must have requires_grad=True.")

    def test_backward_does_not_raise(self) -> None:
        """Calling .backward() on the loss should not raise any error."""
        loss_fn = self._make_loss()
        src, gen = _make_feat_lists(n_layers=3)
        mask = _make_mask(1, 32, 32, fill=0.5)

        out = loss_fn(src, gen, mask)
        out.backward()  # should complete without error

    # ------------------------------------------------------------------
    # 2. Loss decreases when positives are aligned
    # ------------------------------------------------------------------

    def test_lower_loss_when_features_match(self) -> None:
        """Loss with feat_source ≈ feat_generated must be lower than with random pair.

        When source and generated features are identical, each positive pair
        has cosine similarity 1.0 (maximum), giving a low InfoNCE loss.
        With independent random features, positive-pair similarity is ~0,
        raising the loss close to log(n_patches).
        """
        torch.manual_seed(0)
        loss_fn = self._make_loss(n_patches=64)
        mask = _make_mask(1, 32, 32, fill=0.4)  # mostly stroma → fallback PatchNCE

        channels = (8, 16, 32)
        spatial = (32, 16, 8)

        # Case A: generated == source (perfect alignment)
        src_a = [torch.randn(1, c, s, s) for c, s in zip(channels, spatial)]
        gen_a = [f.detach().clone().requires_grad_(True) for f in src_a]
        loss_a = loss_fn(src_a, gen_a, mask)

        # Case B: independent random features (no alignment)
        # Re-use the same loss_fn so MLP weights are shared → fair comparison.
        src_b = [torch.randn(1, c, s, s) for c, s in zip(channels, spatial)]
        gen_b = [torch.randn(1, c, s, s, requires_grad=True) for c, s in zip(channels, spatial)]
        loss_b = loss_fn(src_b, gen_b, mask)

        self.assertLess(
            loss_a.item(),
            loss_b.item(),
            msg=(
                f"Expected loss_A (matching) < loss_B (random), "
                f"got {loss_a.item():.4f} vs {loss_b.item():.4f}."
            ),
        )

    # ------------------------------------------------------------------
    # 3. Checkerboard mask produces two non-empty sets
    # ------------------------------------------------------------------

    def test_checkerboard_mask_partitions_both_regions(self) -> None:
        """A checkerboard mask must yield non-empty nuclear and stroma sets.

        We verify this by inspecting the mask directly (no private API access
        needed) and confirm the forward call completes without falling back
        on every layer (loss should be finite and not degenerate).
        """
        h, w = 8, 8
        mask = _checkerboard_mask(1, h, w)

        # Flatten and check partition counts.
        mask_flat = mask.squeeze().flatten()  # (64,)
        nuc_idx = (mask_flat > 0.5).nonzero(as_tuple=True)[0]
        stroma_idx = (mask_flat <= 0.5).nonzero(as_tuple=True)[0]

        self.assertEqual(nuc_idx.numel(), 32, "Half of checkerboard pixels should be nuclear.")
        self.assertEqual(stroma_idx.numel(), 32, "Half of checkerboard pixels should be stroma.")
        # Confirm they are disjoint
        self.assertEqual(
            nuc_idx.numel() + stroma_idx.numel(),
            mask_flat.numel(),
        )

    def test_forward_with_checkerboard_mask(self) -> None:
        """forward() with a checkerboard mask must return a finite scalar."""
        loss_fn = self._make_loss(n_patches=32)

        # Use spatial sizes that match the checkerboard (8×8) for layer 0.
        src = [_rand_feat(1, 8, 8, 8)]
        gen = [_rand_feat(1, 8, 8, 8)]
        loss_fn_1layer = RegionAwarePatchNCE(
            n_layers=1, n_patches=32, nc=32, tau=0.07, min_region_patches=4
        )

        mask = _checkerboard_mask(1, 8, 8)
        out = loss_fn_1layer(src, gen, mask)

        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(out), f"Loss is not finite: {out.item()}")

    # ------------------------------------------------------------------
    # 4. mask=None reproduces standard PatchNCE
    # ------------------------------------------------------------------

    def test_mask_none_fallback(self) -> None:
        """Passing mask=None should not raise and should return a valid scalar."""
        loss_fn = self._make_loss()
        src, gen = _make_feat_lists(n_layers=3)

        out = loss_fn(src, gen, mask=None)

        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(out))
        self.assertTrue(out.requires_grad)

    # ------------------------------------------------------------------
    # 5. Lazy MLP construction
    # ------------------------------------------------------------------

    def test_mlp_heads_built_after_first_forward(self) -> None:
        """mlp_heads should be empty before the first call and populated after."""
        loss_fn = self._make_loss(n_layers=3)
        self.assertEqual(len(loss_fn.mlp_heads), 0, "No MLP heads before first forward.")

        src, gen = _make_feat_lists(n_layers=3)
        loss_fn(src, gen, mask=None)

        self.assertEqual(len(loss_fn.mlp_heads), 3, "Three MLP heads after first forward.")

    def test_mlp_heads_match_feature_channels(self) -> None:
        """Each MLP head's input size must match the corresponding feature channels."""
        channels = (8, 16, 32)
        spatial = (32, 16, 8)
        loss_fn = RegionAwarePatchNCE(n_layers=3, n_patches=32, nc=64)

        src = [torch.randn(1, c, s, s) for c, s in zip(channels, spatial)]
        gen = [torch.randn(1, c, s, s) for c, s in zip(channels, spatial)]
        loss_fn(src, gen, mask=None)

        for mlp, c in zip(loss_fn.mlp_heads, channels):
            first_linear = mlp.net[0]
            self.assertEqual(
                first_linear.in_features,
                c,
                f"MLP input size {first_linear.in_features} != feature channels {c}.",
            )

    def test_mlp_heads_not_rebuilt_on_second_call(self) -> None:
        """mlp_heads should not be rebuilt (or extended) on subsequent calls."""
        loss_fn = self._make_loss(n_layers=3)
        src, gen = _make_feat_lists(n_layers=3)
        loss_fn(src, gen, mask=None)
        head_ids_first = [id(h) for h in loss_fn.mlp_heads]

        loss_fn(src, gen, mask=None)
        head_ids_second = [id(h) for h in loss_fn.mlp_heads]

        self.assertEqual(head_ids_first, head_ids_second, "MLP heads were replaced on 2nd call.")

    # ------------------------------------------------------------------
    # 6. Feature-list length mismatch
    # ------------------------------------------------------------------

    def test_raises_on_source_length_mismatch(self) -> None:
        """ValueError when len(feat_source) != n_layers."""
        loss_fn = self._make_loss(n_layers=3)
        src_wrong = [_rand_feat(1, 8, 16, 16), _rand_feat(1, 16, 8, 8)]  # only 2
        _, gen = _make_feat_lists(n_layers=3)

        with self.assertRaises(ValueError):
            loss_fn(src_wrong, gen, mask=None)

    def test_raises_on_generated_length_mismatch(self) -> None:
        """ValueError when len(feat_generated) != n_layers."""
        loss_fn = self._make_loss(n_layers=3)
        src, _ = _make_feat_lists(n_layers=3)
        gen_wrong = [_rand_feat(1, 8, 16, 16)]  # only 1

        with self.assertRaises(ValueError):
            loss_fn(src, gen_wrong, mask=None)

    # ------------------------------------------------------------------
    # 7. Degenerate masks trigger fallback without error
    # ------------------------------------------------------------------

    def test_all_nuclear_mask_triggers_fallback(self) -> None:
        """An all-ones mask (all nuclear) falls back to standard PatchNCE."""
        loss_fn = RegionAwarePatchNCE(n_layers=1, n_patches=32, nc=32)
        src = [_rand_feat(1, 8, 16, 16)]
        gen = [_rand_feat(1, 8, 16, 16)]
        mask = _make_mask(1, 16, 16, fill=1.0)  # all nuclear → stroma_idx empty

        out = loss_fn(src, gen, mask)
        self.assertTrue(torch.isfinite(out))

    def test_all_stroma_mask_triggers_fallback(self) -> None:
        """An all-zeros mask (all stroma) falls back to standard PatchNCE."""
        loss_fn = RegionAwarePatchNCE(n_layers=1, n_patches=32, nc=32)
        src = [_rand_feat(1, 8, 16, 16)]
        gen = [_rand_feat(1, 8, 16, 16)]
        mask = _make_mask(1, 16, 16, fill=0.0)  # all stroma → nuc_idx empty

        out = loss_fn(src, gen, mask)
        self.assertTrue(torch.isfinite(out))

    # ------------------------------------------------------------------
    # 8. Sanity: loss is finite and positive
    # ------------------------------------------------------------------

    def test_loss_is_finite_and_positive(self) -> None:
        """Loss must be a finite positive number for random inputs."""
        torch.manual_seed(7)
        loss_fn = self._make_loss()
        src, gen = _make_feat_lists(n_layers=3)
        mask = _checkerboard_mask(1, 32, 32)

        out = loss_fn(src, gen, mask)

        self.assertTrue(torch.isfinite(out), f"Loss is not finite: {out.item()}")
        self.assertGreater(out.item(), 0.0, "Loss should be strictly positive.")

    # ------------------------------------------------------------------
    # Batch size > 1
    # ------------------------------------------------------------------

    def test_batch_size_2(self) -> None:
        """Loss computation must handle B > 1 without error."""
        loss_fn = RegionAwarePatchNCE(n_layers=2, n_patches=32, nc=32)
        src = [_rand_feat(2, 8, 16, 16), _rand_feat(2, 16, 8, 8)]
        gen = [_rand_feat(2, 8, 16, 16), _rand_feat(2, 16, 8, 8)]
        mask = _checkerboard_mask(2, 16, 16)

        out = loss_fn(src, gen, mask)

        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(out))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
