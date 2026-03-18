"""Unit tests for the data pipeline: UnpairedTPAFDataset and transforms.

Run with either::

    python -m pytest tests/test_data_pipeline.py -v
    python -m unittest tests.test_data_pipeline -v

Test coverage
-------------
1. Synthetic patch generation and persistence as .npy files.
2. UnpairedTPAFDataset: output shapes, value ranges, mask shape.
3. Unpaired indexing: TPAF and H&E indices are independent (modulo cycling).
4. Transform pipeline: geometric flips are applied identically to TPAF and mask
   (shared random state via PairedFlipRotate) and independently to H&E.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make the package importable when run from the repo root or tests/ dir.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import UnpairedTPAFDataset  # noqa: E402
from data.transforms import (  # noqa: E402
    PairedFlipRotate,
    UnpairedTransform,
)


# ---------------------------------------------------------------------------
# Synthetic patch factories
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_PATCH = 256


def _make_tpaf(rng: np.random.Generator = _RNG) -> np.ndarray:
    """Return a synthetic TPAF patch: (H, W) float32 in [0, 1].

    Models low-SNR autofluorescence: mostly low-intensity values with a few
    zero-signal void regions (simulating nuclei).
    """
    patch = rng.random((_PATCH, _PATCH)).astype(np.float32) * 0.3
    # Insert void regions (nuclear areas have near-zero TPAF signal).
    patch[40:70, 40:70] = 0.0
    patch[120:160, 130:170] = 0.0
    return patch


def _make_he(rng: np.random.Generator = _RNG) -> np.ndarray:
    """Return a synthetic H&E patch: (H, W, 3) uint8 in [0, 255]."""
    return rng.integers(0, 256, size=(_PATCH, _PATCH, 3), dtype=np.uint8)


def _make_mask(rng: np.random.Generator = _RNG) -> np.ndarray:
    """Return a synthetic nuclear mask: (H, W) float32 binary {0.0, 1.0}."""
    mask = np.zeros((_PATCH, _PATCH), dtype=np.float32)
    mask[40:70, 40:70] = 1.0   # matches the void regions in the TPAF
    mask[120:160, 130:170] = 1.0
    return mask


def _save_domain_patches(
    root: Path,
    n_tpaf: int,
    n_he: int,
    n_masks: int = 0,
) -> tuple[Path, Path, Path | None]:
    """Write synthetic patches to *root* and return (tpaf_dir, he_dir, mask_dir).

    Args:
        root:    Temp directory root.
        n_tpaf:  Number of TPAF patches to write.
        n_he:    Number of H&E patches to write.
        n_masks: Number of TPAF-matched masks to write (0 → no mask_dir).

    Returns:
        ``(tpaf_dir, he_dir, mask_dir)`` where ``mask_dir`` is ``None`` when
        *n_masks* == 0.
    """
    tpaf_dir = root / "tpaf"
    he_dir = root / "he"
    tpaf_dir.mkdir(parents=True)
    he_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)

    tpaf_names: list[str] = []
    for i in range(n_tpaf):
        name = f"tpaf_patch_{i:04d}.npy"
        np.save(tpaf_dir / name, _make_tpaf(rng))
        tpaf_names.append(name)

    for i in range(n_he):
        he_uint8 = _make_he(rng)
        # Normalise to float32 [0, 1] before saving — matches PatchExtractor output.
        he_float = he_uint8.astype(np.float32) / 255.0
        np.save(he_dir / f"he_patch_{i:04d}.npy", he_float)

    mask_dir: Path | None = None
    if n_masks > 0:
        mask_dir = root / "masks"
        mask_dir.mkdir(parents=True)
        for i in range(n_masks):
            # File names must match TPAF names for the mask-lookup to work.
            np.save(mask_dir / tpaf_names[i], _make_mask(rng))

    return tpaf_dir, he_dir, mask_dir


# ---------------------------------------------------------------------------
# 1. Synthetic patch generation
# ---------------------------------------------------------------------------


class TestSyntheticPatchGeneration(unittest.TestCase):
    """Verify the synthetic patch factories produce correctly shaped/typed data."""

    def test_tpaf_shape_and_dtype(self) -> None:
        patch = _make_tpaf()
        self.assertEqual(patch.shape, (_PATCH, _PATCH))
        self.assertEqual(patch.dtype, np.float32)

    def test_tpaf_value_range(self) -> None:
        patch = _make_tpaf()
        self.assertGreaterEqual(float(patch.min()), 0.0)
        self.assertLessEqual(float(patch.max()), 1.0)

    def test_tpaf_has_void_regions(self) -> None:
        """Void regions (nuclei) must contain exactly zero signal."""
        patch = _make_tpaf()
        self.assertEqual(float(patch[40:70, 40:70].max()), 0.0)

    def test_he_shape_and_dtype(self) -> None:
        patch = _make_he()
        self.assertEqual(patch.shape, (_PATCH, _PATCH, 3))
        self.assertEqual(patch.dtype, np.uint8)

    def test_mask_shape_and_binary(self) -> None:
        mask = _make_mask()
        self.assertEqual(mask.shape, (_PATCH, _PATCH))
        unique = set(np.unique(mask).tolist())
        self.assertEqual(unique, {0.0, 1.0})

    def test_npy_round_trip(self) -> None:
        """Patches saved and reloaded via numpy preserve dtype and values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "tpaf.npy"
            original = _make_tpaf()
            np.save(p, original)
            loaded = np.load(p)
        np.testing.assert_array_equal(original, loaded)
        self.assertEqual(loaded.dtype, np.float32)


# ---------------------------------------------------------------------------
# 2. UnpairedTPAFDataset — shapes, ranges, mask
# ---------------------------------------------------------------------------


class TestUnpairedTPAFDatasetShapesAndRanges(unittest.TestCase):
    """Test output tensor shapes and value ranges returned by __getitem__."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        self.tpaf_dir, self.he_dir, self.mask_dir = _save_domain_patches(
            root, n_tpaf=4, n_he=4, n_masks=4
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_dataset(self, with_mask: bool = True) -> UnpairedTPAFDataset:
        return UnpairedTPAFDataset(
            tpaf_dir=self.tpaf_dir,
            he_dir=self.he_dir,
            mask_dir=self.mask_dir if with_mask else None,
            patch_size=_PATCH,
            transform=None,
        )

    # --- tpaf ---

    def test_tpaf_tensor_shape(self) -> None:
        ds = self._make_dataset()
        sample = ds[0]
        self.assertEqual(tuple(sample["tpaf"].shape), (1, _PATCH, _PATCH))

    def test_tpaf_dtype_is_float32(self) -> None:
        ds = self._make_dataset()
        self.assertEqual(ds[0]["tpaf"].dtype, torch.float32)

    def test_tpaf_range_zero_to_one(self) -> None:
        ds = self._make_dataset()
        for idx in range(len(ds)):
            t = ds[idx]["tpaf"]
            self.assertGreaterEqual(t.min().item(), 0.0,
                                    msg=f"TPAF min < 0 at index {idx}")
            self.assertLessEqual(t.max().item(), 1.0,
                                 msg=f"TPAF max > 1 at index {idx}")

    # --- he ---

    def test_he_tensor_shape(self) -> None:
        ds = self._make_dataset()
        self.assertEqual(tuple(ds[0]["he"].shape), (3, _PATCH, _PATCH))

    def test_he_dtype_is_float32(self) -> None:
        ds = self._make_dataset()
        self.assertEqual(ds[0]["he"].dtype, torch.float32)

    def test_he_range_minus_one_to_one(self) -> None:
        """H&E must be scaled to [-1, 1] (tanh output range)."""
        ds = self._make_dataset()
        for idx in range(len(ds)):
            he = ds[idx]["he"]
            self.assertGreaterEqual(he.min().item(), -1.0,
                                    msg=f"H&E min < -1 at index {idx}")
            self.assertLessEqual(he.max().item(), 1.0,
                                 msg=f"H&E max > 1 at index {idx}")

    # --- mask ---

    def test_mask_tensor_shape_with_mask_dir(self) -> None:
        ds = self._make_dataset(with_mask=True)
        self.assertEqual(tuple(ds[0]["mask"].shape), (1, _PATCH, _PATCH))

    def test_mask_dtype_is_float32(self) -> None:
        ds = self._make_dataset(with_mask=True)
        self.assertEqual(ds[0]["mask"].dtype, torch.float32)

    def test_mask_range_zero_to_one(self) -> None:
        ds = self._make_dataset(with_mask=True)
        for idx in range(len(ds)):
            m = ds[idx]["mask"]
            self.assertGreaterEqual(m.min().item(), 0.0)
            self.assertLessEqual(m.max().item(), 1.0)

    def test_mask_tensor_shape_without_mask_dir(self) -> None:
        """Absent mask_dir → zero tensor of correct shape."""
        ds = self._make_dataset(with_mask=False)
        mask = ds[0]["mask"]
        self.assertEqual(tuple(mask.shape), (1, _PATCH, _PATCH))
        self.assertEqual(mask.sum().item(), 0.0)

    # --- metadata ---

    def test_sample_contains_path_strings(self) -> None:
        ds = self._make_dataset()
        sample = ds[0]
        self.assertIsInstance(sample["tpaf_path"], str)
        self.assertIsInstance(sample["he_path"], str)


# ---------------------------------------------------------------------------
# 3. Unpaired indexing
# ---------------------------------------------------------------------------


class TestUnpairedIndexing(unittest.TestCase):
    """Verify that TPAF and H&E use independent modulo-cycled indices."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_dataset_length_is_max_of_domain_sizes(self) -> None:
        tpaf_dir, he_dir, _ = _save_domain_patches(self.root, n_tpaf=3, n_he=5)
        ds = UnpairedTPAFDataset(tpaf_dir, he_dir)
        self.assertEqual(len(ds), 5)

    def test_different_indices_yield_different_he_patches(self) -> None:
        """With more H&E patches than TPAF, consecutive indices should differ in H&E."""
        tpaf_dir, he_dir, _ = _save_domain_patches(
            self.root / "sep", n_tpaf=2, n_he=5
        )
        ds = UnpairedTPAFDataset(tpaf_dir, he_dir)
        # Collect all (tpaf_path, he_path) pairs across the dataset.
        pairs = [(ds[i]["tpaf_path"], ds[i]["he_path"]) for i in range(len(ds))]
        he_paths = [p for _, p in pairs]
        # All 5 H&E patches should be visited (no collisions from cycling).
        self.assertEqual(len(set(he_paths)), 5)

    def test_tpaf_and_he_paths_differ(self) -> None:
        """Returned tpaf_path and he_path must point to different domain dirs."""
        tpaf_dir, he_dir, _ = _save_domain_patches(
            self.root / "diff", n_tpaf=3, n_he=3
        )
        ds = UnpairedTPAFDataset(tpaf_dir, he_dir)
        for i in range(len(ds)):
            sample = ds[i]
            self.assertNotEqual(
                sample["tpaf_path"],
                sample["he_path"],
                msg=f"Index {i}: tpaf_path equals he_path — patches are not unpaired.",
            )

    def test_tpaf_cycles_when_smaller_than_he(self) -> None:
        """When n_tpaf < n_he, the TPAF paths should cycle (mod n_tpaf)."""
        tpaf_dir, he_dir, _ = _save_domain_patches(
            self.root / "cycle", n_tpaf=2, n_he=4
        )
        ds = UnpairedTPAFDataset(tpaf_dir, he_dir)
        self.assertEqual(len(ds), 4)
        # Index 0 and 2 should map to the same TPAF file (cycling with period 2).
        self.assertEqual(ds[0]["tpaf_path"], ds[2]["tpaf_path"])
        # Index 1 and 3 should also map to the same TPAF file.
        self.assertEqual(ds[1]["tpaf_path"], ds[3]["tpaf_path"])
        # But indices 0 and 1 should map to different H&E files (no cycling yet).
        self.assertNotEqual(ds[0]["he_path"], ds[1]["he_path"])

    def test_he_cycles_when_smaller_than_tpaf(self) -> None:
        """When n_he < n_tpaf, the H&E paths should cycle (mod n_he)."""
        tpaf_dir, he_dir, _ = _save_domain_patches(
            self.root / "cycle2", n_tpaf=4, n_he=2
        )
        ds = UnpairedTPAFDataset(tpaf_dir, he_dir)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds[0]["he_path"], ds[2]["he_path"])
        self.assertEqual(ds[1]["he_path"], ds[3]["he_path"])
        self.assertNotEqual(ds[0]["tpaf_path"], ds[1]["tpaf_path"])


# ---------------------------------------------------------------------------
# 4. Transform pipeline — TPAF/mask flip consistency
# ---------------------------------------------------------------------------


class TestTransformFlipConsistency(unittest.TestCase):
    """Verify that PairedFlipRotate applies identical transforms to TPAF and mask."""

    # ------------------------------------------------------------------
    # PairedFlipRotate: forced p=1.0 flips
    # ------------------------------------------------------------------

    def _asymmetric_image(self) -> torch.Tensor:
        """Return a (1, 8, 8) tensor with a distinct left-right asymmetry."""
        t = torch.zeros(1, 8, 8)
        t[0, :, 0] = 1.0   # left column = 1
        t[0, :, 1:] = 0.0  # rest = 0
        return t

    def test_hflip_applied_identically_to_tpaf_and_mask(self) -> None:
        """With p_hflip=1 the same horizontal flip must be applied to both."""
        aug = PairedFlipRotate(p_hflip=1.0, p_vflip=0.0, p_rot90=0.0)
        image = self._asymmetric_image()
        mask = image.clone()

        aug_image, aug_mask = aug(image.clone(), mask.clone())

        # After hflip the distinctive column moves from left to right.
        torch.testing.assert_close(aug_image, aug_mask)

    def test_vflip_applied_identically_to_tpaf_and_mask(self) -> None:
        """With p_vflip=1 the same vertical flip must be applied to both."""
        aug = PairedFlipRotate(p_hflip=0.0, p_vflip=1.0, p_rot90=0.0)
        image = torch.zeros(1, 8, 8)
        image[0, 0, :] = 1.0   # top row = 1
        mask = image.clone()

        aug_image, aug_mask = aug(image.clone(), mask.clone())

        torch.testing.assert_close(aug_image, aug_mask)

    def test_rot90_applied_identically_to_tpaf_and_mask(self) -> None:
        """With p_rot90=1 the same rotation must be applied to both."""
        aug = PairedFlipRotate(p_hflip=0.0, p_vflip=0.0, p_rot90=1.0)
        image = self._asymmetric_image()
        mask = image.clone()

        aug_image, aug_mask = aug(image.clone(), mask.clone())

        torch.testing.assert_close(aug_image, aug_mask)

    def test_none_mask_returned_unchanged(self) -> None:
        """A None mask must pass through PairedFlipRotate unmodified."""
        aug = PairedFlipRotate(p_hflip=1.0, p_vflip=1.0, p_rot90=1.0)
        image = self._asymmetric_image()
        _, returned_mask = aug(image, None)
        self.assertIsNone(returned_mask)

    def test_hflip_actually_flips_image_geometry(self) -> None:
        """Confirm the horizontal flip reverses the spatial content."""
        aug = PairedFlipRotate(p_hflip=1.0, p_vflip=0.0, p_rot90=0.0)
        image = self._asymmetric_image()   # left column = 1, rest = 0
        flipped, _ = aug(image.clone(), None)

        # After hflip, the rightmost column should be 1, leftmost should be 0.
        self.assertAlmostEqual(flipped[0, 0, -1].item(), 1.0)
        self.assertAlmostEqual(flipped[0, 0, 0].item(), 0.0)

    # ------------------------------------------------------------------
    # UnpairedTransform via dataset integration
    # ------------------------------------------------------------------

    def test_unpaired_transform_tpaf_mask_consistency_via_dataset(self) -> None:
        """TPAF and mask must receive the same geometric transform in the dataset.

        Strategy: create TPAF and mask with a known asymmetry.  After a forced
        horizontal flip (p_hflip=1, others 0) the asymmetry should be mirrored
        identically in both output tensors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tpaf_dir = root / "tpaf"
            he_dir = root / "he"
            mask_dir = root / "masks"
            tpaf_dir.mkdir()
            he_dir.mkdir()
            mask_dir.mkdir()

            # Build a TPAF patch with a left-bright, right-dark pattern.
            tpaf = np.zeros((_PATCH, _PATCH), dtype=np.float32)
            tpaf[:, : _PATCH // 2] = 0.8    # left half bright
            tpaf[:, _PATCH // 2 :] = 0.1    # right half dim

            # Mask mirrors the TPAF pattern for easy comparison.
            mask = np.zeros((_PATCH, _PATCH), dtype=np.float32)
            mask[:, : _PATCH // 2] = 1.0
            mask[:, _PATCH // 2 :] = 0.0

            # H&E is uniform; its value doesn't matter for this test.
            he = np.full((_PATCH, _PATCH, 3), 0.5, dtype=np.float32)

            fname = "patch_0000.npy"
            np.save(tpaf_dir / fname, tpaf)
            np.save(he_dir / fname, he)
            np.save(mask_dir / fname, mask)

            # Force horizontal flip only.
            transform = UnpairedTransform(
                p_hflip=1.0, p_vflip=0.0, p_rot90=0.0, he_color_jitter=False
            )
            ds = UnpairedTPAFDataset(
                tpaf_dir, he_dir, mask_dir=mask_dir, transform=transform
            )
            sample = ds[0]

        out_tpaf = sample["tpaf"]  # (1, H, W)
        out_mask = sample["mask"]  # (1, H, W)

        # After hflip: right half of tpaf should be bright (0.8) not dim (0.1).
        right_tpaf = out_tpaf[0, 0, _PATCH // 2 :].mean().item()
        self.assertGreater(right_tpaf, 0.5,
                           msg="TPAF was not flipped as expected.")

        # Mask must follow the same flip: right half should be 1.0.
        right_mask = out_mask[0, 0, _PATCH // 2 :].mean().item()
        self.assertGreater(right_mask, 0.5,
                           msg="Mask was not flipped in sync with TPAF.")

    def test_he_transforms_independent_of_tpaf(self) -> None:
        """H&E geometry must be drawn independently, not shared with TPAF/mask.

        We cannot deterministically test *different* transforms (the random draws
        are independent, so on any single sample they might coincidentally agree).
        Instead we verify the *structural* guarantee: that the H&E tensor is
        augmented without altering TPAF or mask geometry in a correlated way
        across many samples.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tpaf_dir = root / "tpaf"
            he_dir = root / "he"
            tpaf_dir.mkdir()
            he_dir.mkdir()

            # 8 patches per domain so we get enough samples.
            rng = np.random.default_rng(7)
            for i in range(8):
                tpaf = rng.random((_PATCH, _PATCH)).astype(np.float32)
                he = (rng.random((_PATCH, _PATCH, 3)).astype(np.float32))
                np.save(tpaf_dir / f"p{i:02d}.npy", tpaf)
                np.save(he_dir / f"p{i:02d}.npy", he)

            transform = UnpairedTransform(
                p_hflip=0.5, p_vflip=0.5, p_rot90=0.0, he_color_jitter=False
            )
            ds = UnpairedTPAFDataset(tpaf_dir, he_dir, transform=transform)

            # Collect whether left-half > right-half mean for each modality.
            tpaf_left_gt_right: list[bool] = []
            he_left_gt_right: list[bool] = []
            torch.manual_seed(123)
            for i in range(len(ds)):
                s = ds[i]
                tpaf_left_gt_right.append(
                    s["tpaf"][0, :, : _PATCH // 2].mean().item()
                    > s["tpaf"][0, :, _PATCH // 2 :].mean().item()
                )
                he_left_gt_right.append(
                    s["he"][0, :, : _PATCH // 2].mean().item()
                    > s["he"][0, :, _PATCH // 2 :].mean().item()
                )

        # TPAF and H&E flip decisions should not be perfectly correlated.
        # If they were always equal, it would indicate shared random state.
        # (With 8 samples and p=0.5, the chance of all 8 being equal is 2*(0.5^8)≈0.8%.)
        n_agree = sum(a == b for a, b in zip(tpaf_left_gt_right, he_left_gt_right))
        self.assertLess(
            n_agree,
            len(ds),
            msg=(
                "TPAF and H&E flips agreed on every sample, suggesting "
                "shared random state rather than independent augmentation."
            ),
        )


# ---------------------------------------------------------------------------
# 5. Edge cases / error handling
# ---------------------------------------------------------------------------


class TestDatasetEdgeCases(unittest.TestCase):
    """Test error handling and boundary conditions."""

    def test_missing_tpaf_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            he_dir = root / "he"
            he_dir.mkdir()
            np.save(he_dir / "p.npy", np.zeros((_PATCH, _PATCH, 3), dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                UnpairedTPAFDataset(root / "nonexistent_tpaf", he_dir)

    def test_missing_he_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tpaf_dir = root / "tpaf"
            tpaf_dir.mkdir()
            np.save(tpaf_dir / "p.npy", np.zeros((_PATCH, _PATCH), dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                UnpairedTPAFDataset(tpaf_dir, root / "nonexistent_he")

    def test_empty_tpaf_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tpaf_dir = root / "tpaf"
            he_dir = root / "he"
            tpaf_dir.mkdir()
            he_dir.mkdir()
            np.save(he_dir / "p.npy", np.zeros((_PATCH, _PATCH, 3), dtype=np.float32))
            with self.assertRaises(RuntimeError):
                UnpairedTPAFDataset(tpaf_dir, he_dir)

    def test_missing_mask_for_tpaf_returns_zeros(self) -> None:
        """If a TPAF file has no matching mask, a zero-tensor must be returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tpaf_dir = root / "tpaf"
            he_dir = root / "he"
            mask_dir = root / "masks"
            tpaf_dir.mkdir()
            he_dir.mkdir()
            mask_dir.mkdir()

            np.save(tpaf_dir / "tpaf.npy", np.zeros((_PATCH, _PATCH), dtype=np.float32))
            np.save(he_dir / "he.npy", np.zeros((_PATCH, _PATCH, 3), dtype=np.float32))
            # Mask directory exists but contains no file named 'tpaf.npy'.
            np.save(mask_dir / "other.npy", np.ones((_PATCH, _PATCH), dtype=np.float32))

            ds = UnpairedTPAFDataset(tpaf_dir, he_dir, mask_dir=mask_dir)
            mask = ds[0]["mask"]

        self.assertEqual(mask.sum().item(), 0.0)
        self.assertEqual(tuple(mask.shape), (1, _PATCH, _PATCH))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
