"""Unpaired TPAF / H&E patch dataset for SA-CUT training.

Key design constraints (from CLAUDE.md)
-----------------------------------------
* **Unpaired data**: TPAF and H&E patches have NO pixel-level correspondence.
  Never compute pixel-wise losses (L1/MSE) between source TPAF and target H&E.
* **Independent indexing**: if the two domain lists differ in size, the shorter
  list is cycled via modulo so every epoch sees all patches from both domains.
* **TPAF normalisation**: patches are expected to be float32 in ``[0, 1]``,
  already percentile-normalised by :mod:`data.patch_extractor`.  A defensive
  clamp + warning fires if a patch is found outside ``[0, 1]``.
* **H&E scaling**: loaded as float32 ``[0, 1]`` then scaled to ``[-1, 1]`` in
  ``__getitem__`` **after** the optional transform.  This ordering ensures that
  :class:`data.transforms.HEColorJitter` always receives a ``[0, 1]`` image.
* **Mask dtype**: float32 in ``[0, 1]`` (hard binary: 0.0/1.0; soft probability
  maps: continuous).  A zero mask is returned when no pre-computed mask exists;
  the trainer then calls the frozen AFN-DeSeg model on-the-fly.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# Supported patch file formats for TPAF and H&E directories.
# Masks are always .npy (output of scripts/precompute_masks.py).
_PATCH_EXTENSIONS: tuple[str, ...] = (".npy", ".png", ".tif", ".tiff")

logger = logging.getLogger(__name__)


class UnpairedTPAFDataset(Dataset):
    """Unpaired dataset of TPAF and H&E microscopy patches.

    TPAF (source domain) and H&E (target domain) patches are loaded from separate
    directories with **no assumed spatial or semantic correspondence** between them.
    This is the fundamental unpaired-translation assumption: matching is performed
    implicitly by the adversarial and contrastive losses during training, not by
    data alignment.

    Cycling
    ~~~~~~~
    When the two domains differ in size the shorter list is cycled via modulo
    indexing so that one full epoch exhausts the **larger** domain list:

    .. code-block:: text

        tpaf_index = index % len(tpaf_files)
        he_index   = index % len(he_files)
        len(dataset) = max(len(tpaf_files), len(he_files))

    Expected file layout
    ~~~~~~~~~~~~~~~~~~~~
    The layout matches the output of :class:`data.patch_extractor.PatchExtractor`::

        tpaf_dir/
          slide_A/slide_A_y0000000_x0000000.npy   # float32 (H,W)  [0,1]
          slide_A/slide_A_y0000000_x0000256.npy
          ...
        he_dir/
          slide_B/slide_B_y0000000_x0000000.npy   # float32 (H,W,3) [0,1]
          ...
        mask_dir/  (optional)
          slide_A/slide_A_y0000000_x0000000.npy   # float32 (H,W)  [0,1]
          ...

    Returned sample dict
    ~~~~~~~~~~~~~~~~~~~~
    +----------------+--------------------------------+-----------------------------+
    | Key            | Shape                          | Range / notes               |
    +================+================================+=============================+
    | ``'tpaf'``     | ``(1, H, W)`` float32          | ``[0, 1]``                  |
    +----------------+--------------------------------+-----------------------------+
    | ``'he'``       | ``(3, H, W)`` float32          | ``[-1, 1]``  (tanh range)   |
    +----------------+--------------------------------+-----------------------------+
    | ``'mask'``     | ``(1, H, W)`` float32          | ``[0, 1]``; zeros if absent |
    +----------------+--------------------------------+-----------------------------+
    | ``'tpaf_path'``| ``str``                        | absolute path               |
    +----------------+--------------------------------+-----------------------------+
    | ``'he_path'``  | ``str``                        | absolute path               |
    +----------------+--------------------------------+-----------------------------+

    Transform ordering
    ~~~~~~~~~~~~~~~~~~
    The optional *transform* callable receives the sample dict **before** H&E is
    scaled to ``[-1, 1]``.  All tensors are in ``[0, 1]`` at transform time.
    The ``[-1, 1]`` scaling is applied by :meth:`__getitem__` immediately after
    the transform returns.  This guarantees that
    :class:`data.transforms.HEColorJitter` (which uses torchvision's
    ``ColorJitter``) always receives values in the expected ``[0, 1]`` range.

    Args:
        tpaf_dir:   Directory containing TPAF patches (recursive search).
                    Supported formats: ``.npy``, ``.png``, ``.tif``, ``.tiff``.
        he_dir:     Directory containing H&E patches (recursive search).
                    Supported formats: ``.npy``, ``.png``, ``.tif``, ``.tiff``.
        mask_dir:   Optional directory of pre-computed nuclear masks (``.npy``
                    only, produced by ``scripts/precompute_masks.py``).  Matched
                    to TPAF patches by **filename stem** (no extension), so a TPAF
                    file ``foo.png`` matches mask ``foo.npy``.  When ``None``, a
                    zero mask of shape ``(1, H, W)`` is returned and the trainer
                    calls the Cellpose-SAM model on-the-fly.
        patch_size: Expected spatial size used for shape validation.  Pass ``None``
                    to skip (e.g. if using variable-size patches).
        transform:  Optional callable ``(dict) -> dict`` applied to the sample
                    before H&E is scaled to ``[-1, 1]``.  See
                    :class:`data.transforms.UnpairedTransform`.

    Raises:
        FileNotFoundError: If *tpaf_dir* or *he_dir* (or *mask_dir*, if given)
            does not exist.
        RuntimeError: If either domain directory contains no supported patch files.
    """

    def __init__(
        self,
        tpaf_dir: str | Path,
        he_dir: str | Path,
        mask_dir: Optional[str | Path] = None,
        patch_size: Optional[int] = 256,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.tpaf_dir = Path(tpaf_dir)
        self.he_dir = Path(he_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.patch_size = patch_size
        self.transform = transform

        self.tpaf_files: list[Path] = _find_patches(self.tpaf_dir, label="TPAF")
        self.he_files: list[Path] = _find_patches(self.he_dir, label="H&E")

        # Build O(1) mask lookup keyed by file stem (no extension).
        # This allows TPAF files in any format (e.g. foo.png) to match their
        # corresponding mask (foo.npy) produced by scripts/precompute_masks.py.
        self._mask_index: dict[str, Path] = {}
        if self.mask_dir is not None:
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"mask_dir not found: {self.mask_dir}")
            for p in self.mask_dir.glob("**/*.npy"):
                if p.stem in self._mask_index:
                    logger.debug(
                        "Mask stem collision: '%s' — keeping %s, ignoring %s.",
                        p.stem,
                        self._mask_index[p.stem],
                        p,
                    )
                else:
                    self._mask_index[p.stem] = p

        logger.info(
            "%s: %d TPAF | %d H&E | %d masks | len=%d",
            type(self).__name__,
            len(self.tpaf_files),
            len(self.he_files),
            len(self._mask_index),
            len(self),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return dataset length = max(n_tpaf, n_he).

        Cycling the shorter domain means every position in [0, len) maps to
        a valid patch in both domains.
        """
        return max(len(self.tpaf_files), len(self.he_files))

    def __getitem__(self, index: int) -> dict:
        """Load and return one (TPAF, H&E, mask) sample.

        TPAF and H&E are drawn from independent modulo-cycled streams::

            tpaf_path = tpaf_files[index % n_tpaf]
            he_path   = he_files[index % n_he]

        The transform (if any) is called while H&E is still in ``[0, 1]``.
        H&E is scaled to ``[-1, 1]`` immediately after the transform returns.

        Args:
            index: Sample index in ``[0, len(dataset))``.

        Returns:
            Sample dict — see class docstring for key/shape/range details.
        """
        tpaf_path = self.tpaf_files[index % len(self.tpaf_files)]
        he_path = self.he_files[index % len(self.he_files)]

        tpaf = self._load_tpaf(tpaf_path)                     # (1, H, W)  [0,1]
        he = self._load_he(he_path)                            # (3, H, W)  [0,1]
        mask = self._load_mask(tpaf_path, tpaf.shape[-2], tpaf.shape[-1])  # (1,H,W)

        sample: dict = {
            "tpaf": tpaf,
            "he": he,
            "mask": mask,
            "tpaf_path": str(tpaf_path),
            "he_path": str(he_path),
        }

        # Apply augmentation while tensors are in [0, 1].  ColorJitter requires
        # this range; scaling H&E to [-1, 1] beforehand would break it.
        if self.transform is not None:
            sample = self.transform(sample)

        # Scale H&E to [-1, 1] for GAN generator with tanh output activation.
        # TPAF and mask remain in [0, 1].
        sample["he"] = sample["he"] * 2.0 - 1.0

        return sample

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            f"  n_tpaf={len(self.tpaf_files)},\n"
            f"  n_he={len(self.he_files)},\n"
            f"  n_masks={len(self._mask_index)},\n"
            f"  patch_size={self.patch_size},\n"
            f"  transform={self.transform},\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_tpaf(self, path: Path) -> Tensor:
        """Load a TPAF patch as a ``(1, H, W)`` float32 tensor in ``[0, 1]``.

        Supported formats:

        * ``.npy`` — expected to be already float32 in ``[0, 1]`` (percentile-
          normalised by :class:`data.patch_extractor.PatchExtractor`).
        * ``.png`` — loaded as uint8 grayscale; percentile-clipped [1%, 99%] and
          rescaled to ``[0, 1]`` (per CLAUDE.md: never use min-max normalisation).
        * ``.tif`` / ``.tiff`` — loaded via ``tifffile`` (16-bit safe); same
          percentile normalisation as PNG.

        For multi-channel TPAF (NADH + FAD, shape ``(H, W, 2)``), the array is
        transposed to ``(2, H, W)`` and returned as a 2-channel tensor.  The
        generator config must set ``input_nc=3`` (NADH + FAD + mask) in that case.

        Args:
            path: Path to a TPAF patch file.

        Returns:
            Float32 tensor, shape ``(C, H, W)`` where C is 1 or 2.

        Raises:
            RuntimeError: If the loaded array has an unexpected shape or format.
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            array = np.load(path)
            if array.dtype != np.float32:
                logger.warning(
                    "TPAF patch '%s' has dtype %s (expected float32). "
                    "Was percentile normalization applied by the patch extractor? Casting.",
                    path.name,
                    array.dtype,
                )
                array = array.astype(np.float32)
        elif suffix == ".png":
            raw = np.array(Image.open(path).convert("L"), dtype=np.float32)
            lo, hi = np.percentile(raw, [1.0, 99.0])
            array = np.clip((raw - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        elif suffix in (".tif", ".tiff"):
            import tifffile  # optional dependency; only imported when needed
            raw = tifffile.imread(str(path)).astype(np.float32)
            if raw.ndim == 3 and raw.shape[-1] in (1, 2):
                raw = raw[..., 0] if raw.shape[-1] == 1 else raw  # keep (H,W,2) intact
            lo, hi = np.percentile(raw, [1.0, 99.0])
            array = np.clip((raw - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        else:
            raise RuntimeError(
                f"Unsupported TPAF format '{path.suffix}' for '{path}'. "
                f"Supported: {_PATCH_EXTENSIONS}"
            )

        # Shape normalisation → (C, H, W)
        if array.ndim == 2:
            # (H, W) — standard single-channel TPAF
            array = array[np.newaxis, :, :]
        elif array.ndim == 3 and array.shape[-1] in (1, 2):
            # (H, W, C) — channel-last, transpose
            array = np.ascontiguousarray(array.transpose(2, 0, 1))
        elif array.ndim == 3 and array.shape[0] in (1, 2):
            # (C, H, W) — already channel-first
            pass
        else:
            raise RuntimeError(
                f"Unexpected TPAF patch shape {array.shape} in '{path}'. "
                "Expected (H, W), (H, W, C), or (C, H, W) with C ∈ {1, 2}."
            )

        self._check_spatial(array, path)

        tensor = torch.from_numpy(array)
        if tensor.min() < -1e-3 or tensor.max() > 1.0 + 1e-3:
            logger.warning(
                "TPAF patch '%s' values outside [0, 1]: min=%.4f max=%.4f. "
                "Clamping. Verify that the patch extractor applied percentile "
                "normalization (min-max normalization is prohibited per CLAUDE.md).",
                path.name,
                tensor.min().item(),
                tensor.max().item(),
            )
            tensor = tensor.clamp(0.0, 1.0)

        return tensor

    def _load_he(self, path: Path) -> Tensor:
        """Load an H&E patch as a ``(3, H, W)`` float32 tensor in ``[0, 1]``.

        Supported formats:

        * ``.npy`` — expected to be already float32 in ``[0, 1]``.
        * ``.png`` — loaded as uint8 RGB; divided by 255.
        * ``.tif`` / ``.tiff`` — loaded via ``tifffile``; divided by 255 if
          values exceed 1.0 (assumes 8-bit H&E TIFF).

        Scaling to ``[-1, 1]`` is performed by :meth:`__getitem__` **after** the
        transform, not here.

        Args:
            path: Path to an H&E patch file.

        Returns:
            Float32 tensor of shape ``(3, H, W)`` in ``[0, 1]``.

        Raises:
            RuntimeError: If the loaded array is not 3-channel or format unsupported.
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            array = np.load(path)
            if array.dtype != np.float32:
                logger.warning(
                    "H&E patch '%s' has dtype %s (expected float32). Casting.",
                    path.name,
                    array.dtype,
                )
                array = array.astype(np.float32)
        elif suffix == ".png":
            array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        elif suffix in (".tif", ".tiff"):
            import tifffile  # optional dependency; only imported when needed
            raw = tifffile.imread(str(path)).astype(np.float32)
            array = raw / 255.0 if raw.max() > 1.0 else raw
        else:
            raise RuntimeError(
                f"Unsupported H&E format '{path.suffix}' for '{path}'. "
                f"Supported: {_PATCH_EXTENSIONS}"
            )

        # Shape normalisation → (3, H, W)
        if array.ndim == 3 and array.shape[-1] == 3:
            # (H, W, 3) — channel-last (patch extractor default)
            array = np.ascontiguousarray(array.transpose(2, 0, 1))
        elif array.ndim == 3 and array.shape[0] == 3:
            # (3, H, W) — already channel-first
            pass
        else:
            raise RuntimeError(
                f"Unexpected H&E patch shape {array.shape} in '{path}'. "
                "Expected (H, W, 3) or (3, H, W)."
            )

        self._check_spatial(array, path)

        tensor = torch.from_numpy(array)
        if tensor.min() < -1e-3 or tensor.max() > 1.0 + 1e-3:
            logger.warning(
                "H&E patch '%s' values outside [0, 1]: min=%.4f max=%.4f. Clamping.",
                path.name,
                tensor.min().item(),
                tensor.max().item(),
            )
            tensor = tensor.clamp(0.0, 1.0)

        return tensor

    def _load_mask(self, tpaf_path: Path, height: int, width: int) -> Tensor:
        """Load the nuclear segmentation mask matching *tpaf_path*, or return zeros.

        Masks are matched by **filename** (basename only), so they can live in a
        flat directory or mirror the TPAF subdirectory structure.

        Mask dtype must be float32 in ``[0, 1]`` per CLAUDE.md:
        hard binary masks use {0.0, 1.0}; soft probability maps use continuous
        values.  A clamp to ``[0, 1]`` is applied defensively.

        Args:
            tpaf_path: TPAF patch path whose mask is to be located.
            height:    Spatial height for the fallback zero tensor.
            width:     Spatial width for the fallback zero tensor.

        Returns:
            Float32 tensor of shape ``(1, H, W)`` in ``[0, 1]``.
        """
        if self.mask_dir is None:
            return torch.zeros(1, height, width, dtype=torch.float32)

        mask_path = self._mask_index.get(tpaf_path.stem)
        if mask_path is None:
            logger.warning(
                "No pre-computed mask found for stem '%s' in mask_dir '%s'. "
                "Returning zero mask; Cellpose-SAM will compute it on-the-fly.",
                tpaf_path.stem,
                self.mask_dir,
            )
            return torch.zeros(1, height, width, dtype=torch.float32)

        array = np.load(mask_path).astype(np.float32)

        # Shape normalisation → (1, H, W)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.ascontiguousarray(array.transpose(2, 0, 1))
        elif array.ndim == 3 and array.shape[0] == 1:
            pass
        else:
            raise RuntimeError(
                f"Unexpected mask shape {array.shape} for '{mask_path}'. "
                "Expected (H, W) or (1, H, W)."
            )

        return torch.from_numpy(array).clamp(0.0, 1.0)

    def _check_spatial(self, array: np.ndarray, path: Path) -> None:
        """Log a warning if patch spatial dimensions differ from *patch_size*.

        Args:
            array: Array of shape ``(C, H, W)``.
            path:  Source path for the warning message.
        """
        if self.patch_size is None:
            return
        h, w = array.shape[-2], array.shape[-1]
        if h != self.patch_size or w != self.patch_size:
            logger.warning(
                "Patch '%s' has spatial size (%d, %d), expected (%d, %d).",
                path.name,
                h,
                w,
                self.patch_size,
                self.patch_size,
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _find_patches(directory: Path, label: str) -> list[Path]:
    """Recursively discover and sort patch files under *directory*.

    Searches for all formats in :data:`_PATCH_EXTENSIONS` (``.npy``, ``.png``,
    ``.tif``, ``.tiff``).  Files are de-duplicated by stem when the same name
    appears in multiple formats (e.g. ``foo.npy`` and ``foo.png``): the format
    with the highest priority in ``_PATCH_EXTENSIONS`` wins.

    Args:
        directory: Root directory to search.
        label:     Human-readable domain name for error messages.

    Returns:
        Sorted list of patch file paths.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        RuntimeError:      If no supported patch files are found.
    """
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory not found: {directory}")

    # Collect all supported files; de-duplicate by stem (first extension wins
    # per _PATCH_EXTENSIONS priority order).
    seen: dict[str, Path] = {}
    for ext in _PATCH_EXTENSIONS:
        for p in directory.glob(f"**/*{ext}"):
            if p.stem not in seen:
                seen[p.stem] = p

    files = sorted(seen.values(), key=lambda p: (p.parent, p.name))
    if not files:
        raise RuntimeError(
            f"No patch files found in {label} directory: {directory}\n"
            f"Supported formats: {_PATCH_EXTENSIONS}"
        )
    return files
