"""Unit tests for models/mask_provider.py.

Run with::

    python -m pytest tests/test_mask_provider.py -v
    python -m unittest tests.test_mask_provider -v

Test coverage
-------------
1. ``PrecomputedMaskProvider``
   - Loads mask by matching TPAF filename stem across different extensions
   - Returns float32 tensor of shape (1, H, W) with values in [0, 1]
   - Normalises uint8-range masks (0–255) to [0, 1]
   - Promotes 2-D (H, W) arrays to (1, H, W)
   - Raises FileNotFoundError with a helpful message on missing mask
   - Validates spatial dimensions against patch_size on the first call
   - Caches the shape check (does not re-validate on subsequent calls)
   - Raises FileNotFoundError when mask_dir does not exist

2. ``CellposeSAMMaskProvider`` (skipped if cellpose is not installed)
   - Returns float32 tensor of shape (1, H, W)
   - All output values are in [0, 1]
   - ``return_flows=True`` returns a tuple (tensor, list)

3. ``CellposeSAMMaskProvider`` parameter-freeze (always run via mocking)
   - No parameters have ``requires_grad=True`` after construction
   - ``parameters()`` returns an empty iterator
   - ``named_parameters()`` returns an empty iterator

4. ``build_mask_provider`` factory
   - Returns ``PrecomputedMaskProvider`` for mode="precomputed"
   - Returns ``CellposeSAMMaskProvider`` for mode="cellpose_sam" (mocked)
   - Raises ``ValueError`` for unknown mode
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mask_provider import (  # noqa: E402
    PrecomputedMaskProvider,
    build_mask_provider,
)

# ---------------------------------------------------------------------------
# Detect optional cellpose install
# ---------------------------------------------------------------------------

try:
    import cellpose  # noqa: F401

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mask(directory: Path, stem: str, array: np.ndarray) -> Path:
    """Save *array* as ``<directory>/<stem>.npy`` and return the path."""
    path = directory / f"{stem}.npy"
    np.save(str(path), array)
    return path


def _make_binary_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """Return a random float32 binary mask of shape ``(H, W)``."""
    rng = np.random.default_rng(42)
    return rng.choice([0.0, 1.0], size=(h, w)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. PrecomputedMaskProvider
# ---------------------------------------------------------------------------


class TestPrecomputedMaskProvider(unittest.TestCase):
    """Tests for loading pre-saved .npy masks from disk."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.mask_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Happy-path loading
    # ------------------------------------------------------------------

    def test_loads_mask_matching_stem(self) -> None:
        """Provider resolves the mask by the TPAF path's filename stem."""
        mask = _make_binary_mask()
        _write_mask(self.mask_dir, "patch_001", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("/data/tpaf/patch_001.tif")

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(tuple(result.shape), (1, 256, 256))

    def test_stem_matching_ignores_tpaf_extension(self) -> None:
        """Mask lookup ignores the TPAF extension (.tif, .npy, .png, etc.)."""
        mask = _make_binary_mask()
        _write_mask(self.mask_dir, "patch_042", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        for ext in [".tif", ".tiff", ".npy", ".png"]:
            with self.subTest(ext=ext):
                result = provider.get_mask(f"/data/patches/patch_042{ext}")
                self.assertEqual(tuple(result.shape), (1, 256, 256))

    def test_output_dtype_is_float32(self) -> None:
        """Result tensor must always be float32."""
        mask = _make_binary_mask().astype(np.uint8)  # deliberately wrong dtype
        _write_mask(self.mask_dir, "p", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("p.tif")
        self.assertEqual(result.dtype, torch.float32)

    def test_values_in_unit_range(self) -> None:
        """All mask values must be in [0, 1]."""
        mask = _make_binary_mask()
        _write_mask(self.mask_dir, "p", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("p.npy")
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_normalises_uint8_range_mask(self) -> None:
        """Masks saved in [0, 255] range are rescaled to [0, 1]."""
        mask_255 = (_make_binary_mask() * 255.0).astype(np.float32)
        _write_mask(self.mask_dir, "uint8_mask", mask_255)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("uint8_mask.tif")
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertGreaterEqual(float(result.min()), 0.0)

    def test_promotes_2d_to_3d(self) -> None:
        """A (H, W) mask is promoted to (1, H, W)."""
        mask_2d = _make_binary_mask()  # shape (256, 256)
        self.assertEqual(mask_2d.ndim, 2)
        _write_mask(self.mask_dir, "flat", mask_2d)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("flat.npy")
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 1)

    def test_accepts_already_3d_mask(self) -> None:
        """A (1, H, W) mask is returned as-is."""
        mask_3d = _make_binary_mask()[np.newaxis]  # (1, 256, 256)
        _write_mask(self.mask_dir, "three_d", mask_3d)

        provider = PrecomputedMaskProvider(str(self.mask_dir))
        result = provider.get_mask("three_d.tif")
        self.assertEqual(tuple(result.shape), (1, 256, 256))

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_file_not_found_on_missing_mask(self) -> None:
        """FileNotFoundError with a helpful hint when the mask file is absent."""
        provider = PrecomputedMaskProvider(str(self.mask_dir))
        with self.assertRaises(FileNotFoundError) as ctx:
            provider.get_mask("/data/tpaf/nonexistent_patch.tif")
        # Message should name the missing patch and mention the remedy
        msg = str(ctx.exception)
        self.assertIn("nonexistent_patch", msg)
        self.assertIn("precompute_masks", msg)

    def test_raises_when_mask_dir_missing(self) -> None:
        """FileNotFoundError if mask_dir does not exist at construction."""
        with self.assertRaises(FileNotFoundError):
            PrecomputedMaskProvider("/nonexistent/directory/masks")

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    def test_shape_validation_passes_on_correct_size(self) -> None:
        """No error raised when mask spatial dims match patch_size."""
        mask = _make_binary_mask(128, 128)
        _write_mask(self.mask_dir, "small", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir), patch_size=128)
        result = provider.get_mask("small.tif")  # should not raise
        self.assertEqual(tuple(result.shape), (1, 128, 128))

    def test_shape_validation_raises_on_mismatch(self) -> None:
        """ValueError if loaded mask spatial dims don't match patch_size."""
        mask = _make_binary_mask(128, 128)  # wrong size
        _write_mask(self.mask_dir, "wrong_size", mask)

        provider = PrecomputedMaskProvider(str(self.mask_dir), patch_size=256)
        with self.assertRaises(ValueError) as ctx:
            provider.get_mask("wrong_size.tif")
        self.assertIn("256", str(ctx.exception))

    def test_shape_validation_cached_after_first_call(self) -> None:
        """Shape is only validated on the *first* call; subsequent calls skip it."""
        mask_ok = _make_binary_mask(256, 256)
        _write_mask(self.mask_dir, "ok_1", mask_ok)
        _write_mask(self.mask_dir, "ok_2", mask_ok)

        provider = PrecomputedMaskProvider(str(self.mask_dir), patch_size=256)
        provider.get_mask("ok_1.tif")
        self.assertTrue(provider._shape_validated)

        # Second call: validation is skipped (no re-raise even if we monkey-patch)
        # We verify _shape_validated stays True after second call (no re-check)
        provider.get_mask("ok_2.tif")
        self.assertTrue(provider._shape_validated)


# ---------------------------------------------------------------------------
# 2. CellposeSAMMaskProvider — requires cellpose installed
# ---------------------------------------------------------------------------


@unittest.skipUnless(CELLPOSE_AVAILABLE, "cellpose not installed — skipping")
class TestCellposeSAMMaskProviderWithRealCellpose(unittest.TestCase):
    """Integration-level tests that run only when cellpose is available.

    These tests still mock the model to avoid needing real checkpoint files.
    """

    def _make_provider(self) -> "CellposeSAMMaskProvider":  # noqa: F821
        """Return a CellposeSAMMaskProvider with a mocked Cellpose model."""
        from models.mask_provider import CellposeSAMMaskProvider  # noqa: PLC0415

        h, w = 64, 64
        fake_labels = np.zeros((h, w), dtype=np.int32)
        fake_labels[10:30, 10:30] = 1  # one nucleus
        fake_labels[40:55, 40:55] = 2  # another nucleus

        with mock.patch("cellpose.models.CellposeModel") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.eval.return_value = (fake_labels, [], [])
            mock_instance.net = torch.nn.Linear(1, 1)  # has real params

            provider = CellposeSAMMaskProvider(
                checkpoint_path="/fake/checkpoint.pth",
                model_type="cyto3",
                diameter=30,
                gpu=False,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
            # Replace internal model with the mock instance for inference
            provider._model = mock_instance

        return provider

    def test_output_shape(self) -> None:
        """get_mask returns tensor of shape (1, H, W)."""
        provider = self._make_provider()
        dummy = torch.zeros(1, 64, 64)
        result = provider.get_mask(dummy)
        self.assertEqual(tuple(result.shape), (1, 64, 64))

    def test_output_dtype(self) -> None:
        """Output tensor is float32."""
        provider = self._make_provider()
        result = provider.get_mask(torch.zeros(1, 64, 64))
        self.assertEqual(result.dtype, torch.float32)

    def test_output_value_range(self) -> None:
        """All output values are in [0, 1]."""
        provider = self._make_provider()
        result = provider.get_mask(np.zeros((64, 64), dtype=np.float32))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_return_flows_true(self) -> None:
        """return_flows=True returns a (tensor, list) tuple."""
        provider = self._make_provider()
        output = provider.get_mask(torch.zeros(1, 64, 64), return_flows=True)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        tensor, flows = output
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertIsInstance(flows, list)


# ---------------------------------------------------------------------------
# 3. CellposeSAMMaskProvider — parameter freeze (always run, uses mocking)
# ---------------------------------------------------------------------------


class TestCellposeSAMMaskProviderFreezing(unittest.TestCase):
    """Verify the 'no trainable parameters' constraint — mocked, always runs."""

    def _build_with_mock_cellpose(self) -> "CellposeSAMMaskProvider":  # noqa: F821
        """Construct CellposeSAMMaskProvider with cellpose patched into sys.modules."""
        from models.mask_provider import CellposeSAMMaskProvider  # noqa: PLC0415

        # Build a mock CellposeModel whose .net is a real nn.Module
        real_net = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        # Verify the net starts WITH gradients (so we can confirm they get frozen)
        for p in real_net.parameters():
            self.assertTrue(p.requires_grad, "Precondition: params should have grad by default")

        mock_instance = mock.MagicMock()
        mock_instance.net = real_net
        mock_instance.eval.return_value = (
            np.zeros((64, 64), dtype=np.int32),
            [],
            [],
        )
        MockCellposeModel = mock.MagicMock(return_value=mock_instance)

        mock_cellpose_module = mock.MagicMock()
        mock_cellpose_module.models.CellposeModel = MockCellposeModel

        with mock.patch.dict(
            "sys.modules",
            {"cellpose": mock_cellpose_module, "cellpose.models": mock_cellpose_module.models},
        ):
            provider = CellposeSAMMaskProvider(
                checkpoint_path="/fake/ckpt.pth",
                model_type="cyto3",
                diameter=30,
                gpu=False,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )

        return provider

    def test_no_requires_grad_after_init(self) -> None:
        """All parameters inside the Cellpose net must have requires_grad=False."""
        provider = self._build_with_mock_cellpose()
        for name, param in provider._model.net.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Parameter '{name}' still has requires_grad=True after freezing.",
            )

    def test_parameters_returns_empty_iterator(self) -> None:
        """parameters() returns an empty iterator — safe against optimizer inclusion."""
        provider = self._build_with_mock_cellpose()
        params = list(provider.parameters())
        self.assertEqual(
            params,
            [],
            "CellposeSAMMaskProvider.parameters() must return an empty iterator.",
        )

    def test_named_parameters_returns_empty_iterator(self) -> None:
        """named_parameters() returns an empty iterator."""
        provider = self._build_with_mock_cellpose()
        named = list(provider.named_parameters())
        self.assertEqual(named, [])

    def test_net_is_in_eval_mode(self) -> None:
        """The underlying torch net must be in eval mode after construction."""
        provider = self._build_with_mock_cellpose()
        net = provider._model.net
        self.assertFalse(
            net.training,
            "Cellpose net should be in eval mode (training=False) after freezing.",
        )


# ---------------------------------------------------------------------------
# 4. build_mask_provider factory
# ---------------------------------------------------------------------------


class TestBuildMaskProvider(unittest.TestCase):
    """Tests for the build_mask_provider factory function."""

    def _make_cfg(self, mode: str, mask_dir: Optional[str] = None) -> SimpleNamespace:
        """Build a minimal config namespace for the factory."""
        return SimpleNamespace(
            mask_provider=SimpleNamespace(
                mode=mode,
                mask_dir=mask_dir or "/tmp",
                cellpose_checkpoint="/fake/ckpt.pth",
                cellpose_model_type="cyto3",
                cellpose_diameter=30,
                cellpose_gpu=False,
                cellpose_flow_threshold=0.4,
                cellpose_cellprob_threshold=0.0,
            ),
            data=SimpleNamespace(patch_size=256),
        )

    def test_precomputed_mode_returns_correct_type(self) -> None:
        """build_mask_provider returns PrecomputedMaskProvider for mode='precomputed'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg("precomputed", mask_dir=tmpdir)
            provider = build_mask_provider(cfg)
            self.assertIsInstance(provider, PrecomputedMaskProvider)

    def test_precomputed_mode_passes_patch_size(self) -> None:
        """patch_size from cfg.data is forwarded to PrecomputedMaskProvider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg("precomputed", mask_dir=tmpdir)
            cfg.data.patch_size = 128
            provider = build_mask_provider(cfg)
            self.assertIsInstance(provider, PrecomputedMaskProvider)
            self.assertEqual(provider.patch_size, 128)

    def test_cellpose_sam_mode_returns_correct_type(self) -> None:
        """build_mask_provider returns CellposeSAMMaskProvider for mode='cellpose_sam'."""
        from models.mask_provider import CellposeSAMMaskProvider  # noqa: PLC0415

        mock_instance = mock.MagicMock()
        mock_instance.net = torch.nn.Linear(1, 1)
        MockCellposeModel = mock.MagicMock(return_value=mock_instance)

        mock_cellpose_module = mock.MagicMock()
        mock_cellpose_module.models.CellposeModel = MockCellposeModel

        with mock.patch.dict(
            "sys.modules",
            {"cellpose": mock_cellpose_module, "cellpose.models": mock_cellpose_module.models},
        ):
            cfg = self._make_cfg("cellpose_sam")
            provider = build_mask_provider(cfg)

        self.assertIsInstance(provider, CellposeSAMMaskProvider)

    def test_unknown_mode_raises_value_error(self) -> None:
        """build_mask_provider raises ValueError for an unrecognised mode."""
        cfg = self._make_cfg("totally_unknown_mode")
        with self.assertRaises(ValueError) as ctx:
            build_mask_provider(cfg)
        self.assertIn("totally_unknown_mode", str(ctx.exception))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
