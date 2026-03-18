"""WSI patch extraction pipeline for SA-CUT.

Handles two imaging domains:

* **TPAF** (``.tif``, 16-bit): loaded with ``tifffile`` (PIL/OpenCV silently clip
  16-bit to 8-bit and must never be used).  Z-stacks are collapsed via max-intensity
  projection.  Normalization is percentile-based [p_low, p_high] → [0, 1] float32;
  min-max normalization is explicitly prohibited (see CLAUDE.md).

* **H&E** (``.tif``, ``.svs``, ``.ndpi``, 8-bit RGB): ``.svs``/``.ndpi`` files are
  read tile-by-tile via ``openslide`` (full-image load is infeasible for GB-scale WSIs).
  Plain ``.tif`` files are read with ``tifffile``.  Normalization scales uint8 → [0, 1].

Extracted patches are saved as float32 ``.npy`` files.  Each patch gets a sidecar
``.json`` with provenance metadata.  A ``manifest.csv`` is written to the output
directory covering all patches from a run.

Typical usage::

    from data.patch_extractor import PatchExtractor, save_manifest

    extractor = PatchExtractor(domain="tpaf", patch_size=256)
    records = extractor.extract_from_directory(
        input_dir=Path("data/raw/tpaf"),
        output_dir=Path("data/patches/tpaf"),
    )
    save_manifest(records, Path("data/patches/tpaf"))
"""

import csv
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: openslide (required only for .svs / .ndpi)
# ---------------------------------------------------------------------------
try:
    import openslide  # type: ignore[import-untyped]

    _OPENSLIDE_AVAILABLE = True
except ImportError:
    _OPENSLIDE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Domain = Literal["tpaf", "he"]

_TPAF_EXTENSIONS: frozenset[str] = frozenset({".tif", ".tiff"})
_HE_TIFF_EXTENSIONS: frozenset[str] = frozenset({".tif", ".tiff"})
_HE_OPENSLIDE_EXTENSIONS: frozenset[str] = frozenset({".svs", ".ndpi"})
_HE_EXTENSIONS: frozenset[str] = _HE_TIFF_EXTENSIONS | _HE_OPENSLIDE_EXTENSIONS


# ---------------------------------------------------------------------------
# Patch record
# ---------------------------------------------------------------------------


@dataclass
class PatchRecord:
    """Provenance metadata for a single extracted patch.

    Attributes:
        patch_path: Absolute path to the saved ``.npy`` file.
        source_wsi: Basename of the source WSI file.
        x: Left-edge x-coordinate (pixels at level-0) within the source image.
        y: Top-edge y-coordinate (pixels at level-0) within the source image.
        patch_size: Spatial size of the (square) patch in pixels.
        domain: Imaging domain — ``"tpaf"`` or ``"he"``.
        dtype_original: NumPy dtype of the raw image prior to normalization.
        norm_p_low: Lower percentile used for TPAF clipping (``None`` for H&E).
        norm_p_high: Upper percentile used for TPAF clipping (``None`` for H&E).
        norm_p_low_value: Raw intensity at ``norm_p_low`` (``None`` for H&E).
        norm_p_high_value: Raw intensity at ``norm_p_high`` (``None`` for H&E).
    """

    patch_path: Path
    source_wsi: str
    x: int
    y: int
    patch_size: int
    domain: Domain
    dtype_original: str
    norm_p_low: Optional[float] = None
    norm_p_high: Optional[float] = None
    norm_p_low_value: Optional[float] = None
    norm_p_high_value: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (converts ``Path`` to ``str``)."""
        d = asdict(self)
        d["patch_path"] = str(self.patch_path)
        return d


# ---------------------------------------------------------------------------
# Normalization functions (public — used by data pipeline and tests)
# ---------------------------------------------------------------------------


def normalize_tpaf(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """Clip a TPAF image to [p_low, p_high] percentile then scale to [0, 1] float32.

    This is the *only* permitted normalization strategy for TPAF images.  Min-max
    normalization is explicitly prohibited: TPAF has heavy-tailed intensity
    distributions with hot pixels that would compress the biologically relevant range
    into a narrow band.

    For multi-channel inputs (shape ``(H, W, C)``), each channel is normalised
    independently using its own percentile values.

    Args:
        image: Raw TPAF array.  May be uint16, uint12, float, etc.
            Shape ``(H, W)`` or ``(H, W, C)``.
        p_low: Lower percentile for clipping (default 1.0).
        p_high: Upper percentile for clipping (default 99.0).

    Returns:
        A tuple ``(normalized, p_low_value, p_high_value)`` where:

        * ``normalized`` is float32 in ``[0, 1]`` with the same spatial shape.
        * ``p_low_value`` and ``p_high_value`` are the raw intensity values at the
          chosen percentiles (for channel 0 when multi-channel).

    Raises:
        ValueError: If ``p_low >= p_high``.
    """
    if p_low >= p_high:
        raise ValueError(f"p_low ({p_low}) must be strictly less than p_high ({p_high}).")

    if image.ndim == 3:
        # Normalize each channel independently; return channel-0 stats for metadata.
        channels: list[np.ndarray] = []
        lo0 = hi0 = 0.0
        for c in range(image.shape[-1]):
            ch_norm, lo, hi = normalize_tpaf(image[:, :, c], p_low, p_high)
            channels.append(ch_norm)
            if c == 0:
                lo0, hi0 = lo, hi
        return np.stack(channels, axis=-1).astype(np.float32), lo0, hi0

    lo = float(np.percentile(image, p_low))
    hi = float(np.percentile(image, p_high))

    if hi <= lo:
        warnings.warn(
            f"TPAF image has zero dynamic range (p{p_low}={lo}, p{p_high}={hi}). "
            "Returning zeros.",
            stacklevel=2,
        )
        return np.zeros(image.shape, dtype=np.float32), lo, hi

    clipped = np.clip(image.astype(np.float32), lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return normalized.astype(np.float32), lo, hi


def normalize_he(image: np.ndarray) -> np.ndarray:
    """Scale a uint8 H&E RGB image to [0, 1] float32.

    Args:
        image: uint8 array of shape ``(H, W, 3)``.

    Returns:
        float32 array of shape ``(H, W, 3)`` with values in ``[0, 1]``.
    """
    return (image.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------


def _infer_axes(data: np.ndarray) -> str:
    """Heuristically infer TIFF axis order from array shape.

    Used as a fallback when ``tifffile`` series metadata is unavailable.

    Args:
        data: Raw array loaded from TIFF.

    Returns:
        Axis-order string consistent with tifffile convention (e.g. ``"ZYX"``).
    """
    if data.ndim == 2:
        return "YX"
    if data.ndim == 3:
        # Last dim ≤ 4 → channel axis; otherwise depth/Z axis.
        return "YXC" if data.shape[-1] <= 4 else "ZYX"
    if data.ndim == 4:
        return "ZYXC" if data.shape[-1] <= 4 else "CZYX"
    # Unusual shape — fall back to treating as 2-D.
    return "YX"


def _load_tpaf_array(path: Path) -> tuple[np.ndarray, str]:
    """Load a TPAF TIFF into a 2-D (or 3-D channel) float32-ready array.

    Uses ``tifffile`` exclusively — PIL and OpenCV silently clip 16-bit values
    to 8-bit and must never be used for TPAF.

    Z-stacks are collapsed via max-intensity projection so that the brightest
    signal along the optical axis is preserved.  After projection, the returned
    array has shape ``(H, W)`` (single channel) or ``(H, W, C)`` (multi-channel).

    Args:
        path: Path to a ``.tif`` / ``.tiff`` TPAF file.

    Returns:
        ``(data, dtype_str)`` where ``data`` is the spatial array (not yet
        normalised) and ``dtype_str`` is the original NumPy dtype.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the resulting array is not 2-D or 3-D after projection.
    """
    if not path.exists():
        raise FileNotFoundError(f"TPAF file not found: {path}")

    with tifffile.TiffFile(path) as tif:
        if tif.series:
            series = tif.series[0]
            axes = series.axes.upper()
            data: np.ndarray = series.asarray()
        else:
            data = tif.asarray()
            axes = _infer_axes(data)

    dtype_str = str(data.dtype)

    # ---- Collapse Z axis via max-intensity projection ----
    if "Z" in axes:
        z_idx = axes.index("Z")
        logger.info(
            "Z-stack detected in %s (shape=%s, axes=%s): max-intensity projection.",
            path.name,
            data.shape,
            axes,
        )
        data = data.max(axis=z_idx)
        axes = axes.replace("Z", "")

    # ---- Reorder to (H, W) or (H, W, C) ----
    channel_char = next((c for c in ("C", "S") if c in axes), None)
    if channel_char is not None:
        c_idx = axes.index(channel_char)
        if c_idx == 0:
            # (C, H, W) → (H, W, C)
            data = np.moveaxis(data, 0, -1)
            axes = axes[1:] + axes[0]

    if data.ndim not in (2, 3):
        raise RuntimeError(
            f"Unexpected TPAF array shape {data.shape} (axes='{axes}') after "
            "z-projection.  Expected 2-D (H, W) or 3-D (H, W, C)."
        )

    logger.debug("Loaded TPAF %s → shape=%s dtype=%s", path.name, data.shape, dtype_str)
    return data, dtype_str


def _load_he_tifffile(path: Path) -> tuple[np.ndarray, str]:
    """Load an H&E TIFF into a (H, W, 3) uint8 array using ``tifffile``.

    Args:
        path: Path to a ``.tif`` / ``.tiff`` H&E file.

    Returns:
        ``(data, dtype_str)`` where ``data`` is shape ``(H, W, 3)`` uint8.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the loaded image is not 3-channel.
    """
    if not path.exists():
        raise FileNotFoundError(f"H&E TIFF not found: {path}")

    data = tifffile.imread(str(path))
    dtype_str = str(data.dtype)

    # Pyramidal TIFFs can load as (levels, H, W, 3); take the highest resolution.
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 2:
        # Grayscale — replicate to RGB.
        data = np.stack([data, data, data], axis=-1)
    if data.shape[-1] == 4:
        data = data[:, :, :3]  # drop alpha

    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(
            f"Expected (H, W, 3) after loading {path.name}, got shape {data.shape}."
        )

    logger.debug("Loaded H&E TIFF %s → shape=%s dtype=%s", path.name, data.shape, dtype_str)
    return data, dtype_str


# ---------------------------------------------------------------------------
# Background rejection predicates (public for testing)
# ---------------------------------------------------------------------------


def is_background_tpaf(
    patch: np.ndarray,
    global_p5_threshold: float,
    bg_fraction: float = 0.8,
) -> bool:
    """Return ``True`` if the TPAF patch is predominantly background.

    A patch is considered background when more than *bg_fraction* of its pixels
    fall below the global 5th-percentile intensity of the source WSI.

    Args:
        patch: Float32 patch of shape ``(H, W)`` or ``(H, W, C)``.  If multi-
            channel, only channel 0 is evaluated.
        global_p5_threshold: Raw-intensity threshold derived from the 5th percentile
            of *non-zero* pixels in the source WSI (computed once per WSI).  Using
            only non-zero pixels ensures the threshold reflects actual tissue signal
            rather than the slide background level.
        bg_fraction: Fraction threshold; patches with more than this proportion of
            sub-threshold pixels are rejected (default 0.8).

    Returns:
        ``True`` if the patch should be discarded as background.
    """
    channel = patch[:, :, 0] if patch.ndim == 3 else patch
    return float(np.mean(channel < global_p5_threshold)) > bg_fraction


def is_background_he(patch_f32: np.ndarray, bg_fraction: float = 0.8) -> bool:
    """Return ``True`` if the H&E patch is predominantly white background.

    A patch is considered background when more than *bg_fraction* of its pixels
    have a grayscale value above 0.9 (near-white slide background).

    Args:
        patch_f32: Float32 patch of shape ``(H, W, 3)`` in ``[0, 1]``.
        bg_fraction: Fraction threshold (default 0.8).

    Returns:
        ``True`` if the patch should be discarded as background.
    """
    gray = patch_f32.mean(axis=-1) if patch_f32.ndim == 3 else patch_f32
    return float(np.mean(gray > 0.9)) > bg_fraction


# ---------------------------------------------------------------------------
# Patch I/O helpers
# ---------------------------------------------------------------------------


def _patch_stem(source_wsi: str, y: int, x: int) -> str:
    """Build a deterministic filename stem from WSI name and patch coordinates.

    Zero-padded coordinates ensure correct lexicographic ordering.

    Args:
        source_wsi: Basename of the source WSI file.
        y: Top-edge row coordinate (pixels).
        x: Left-edge column coordinate (pixels).

    Returns:
        Stem string, e.g. ``"slide01_y001024_x002048"``.
    """
    stem = Path(source_wsi).stem
    return f"{stem}_y{y:07d}_x{x:07d}"


def _save_patch(
    patch: np.ndarray,
    record: PatchRecord,
    output_dir: Path,
) -> Path:
    """Save *patch* as ``.npy`` and write a sidecar ``.json`` with *record* metadata.

    Args:
        patch: Float32 array of shape ``(H, W)`` or ``(H, W, C)``.
        record: Metadata record whose ``patch_path`` will be updated in-place.
        output_dir: Directory to write files into (created if absent).

    Returns:
        Absolute path to the saved ``.npy`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _patch_stem(record.source_wsi, record.y, record.x)
    npy_path = output_dir / f"{stem}.npy"
    json_path = output_dir / f"{stem}.json"

    np.save(npy_path, patch)

    record.patch_path = npy_path.resolve()
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2)

    return npy_path


def _iter_patch_positions(
    height: int,
    width: int,
    patch_size: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(y, x)`` top-left corners for all non-overlapping patches.

    Patches that would extend beyond the image boundary are omitted.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        patch_size: Side length of square patches.

    Yields:
        ``(y, x)`` coordinate pairs.
    """
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            yield y, x


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class PatchExtractor:
    """Extract non-overlapping patches from TPAF or H&E whole-slide images.

    For TPAF images the full WSI is loaded into memory (single-channel 16-bit
    is ~200 MB for a 10 k×10 k slide — feasible for research hardware).
    For H&E ``.svs``/``.ndpi`` files, tiles are read on-demand via ``openslide``
    to avoid loading multi-GB files into memory.

    Args:
        domain: ``"tpaf"`` or ``"he"``.
        patch_size: Side length of square output patches (default 256).
        p_low: Lower percentile for TPAF normalization (default 1.0).
        p_high: Upper percentile for TPAF normalization (default 99.0).
        bg_fraction: Maximum proportion of background pixels before a patch is
            rejected (default 0.8, i.e. 80 %).

    Example::

        extractor = PatchExtractor(domain="tpaf", patch_size=256)
        records = extractor.extract_from_directory(
            Path("data/raw/tpaf"), Path("data/patches/tpaf")
        )
        save_manifest(records, Path("data/patches/tpaf"))
    """

    def __init__(
        self,
        domain: Domain,
        patch_size: int = 256,
        p_low: float = 1.0,
        p_high: float = 99.0,
        bg_fraction: float = 0.8,
    ) -> None:
        if domain not in ("tpaf", "he"):
            raise ValueError(f"domain must be 'tpaf' or 'he', got '{domain}'.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if not (0.0 <= bg_fraction <= 1.0):
            raise ValueError(f"bg_fraction must be in [0, 1], got {bg_fraction}.")

        self.domain = domain
        self.patch_size = patch_size
        self.p_low = p_low
        self.p_high = p_high
        self.bg_fraction = bg_fraction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_from_file(self, wsi_path: Path, output_dir: Path) -> list[PatchRecord]:
        """Extract patches from a single WSI file.

        Args:
            wsi_path: Path to the WSI file.
            output_dir: Directory where patches and sidecars will be written.

        Returns:
            List of :class:`PatchRecord` for every patch that passed background
            filtering.
        """
        wsi_path = Path(wsi_path)
        suffix = wsi_path.suffix.lower()

        if self.domain == "tpaf":
            if suffix not in _TPAF_EXTENSIONS:
                logger.warning("Skipping non-TPAF file: %s", wsi_path.name)
                return []
            return self._extract_tpaf(wsi_path, output_dir)

        # H&E
        if suffix in _HE_OPENSLIDE_EXTENSIONS:
            return self._extract_he_openslide(wsi_path, output_dir)
        if suffix in _HE_TIFF_EXTENSIONS:
            return self._extract_he_tifffile(wsi_path, output_dir)

        logger.warning("Skipping unsupported H&E format: %s", wsi_path.name)
        return []

    def extract_from_directory(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> list[PatchRecord]:
        """Extract patches from every recognised WSI file in *input_dir*.

        Args:
            input_dir: Directory containing WSI files (searched non-recursively).
            output_dir: Root directory for output patches.  Per-WSI sub-directories
                are created automatically.

        Returns:
            Aggregated list of :class:`PatchRecord` from all WSI files.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        extensions = _TPAF_EXTENSIONS if self.domain == "tpaf" else _HE_EXTENSIONS
        wsi_files = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in extensions
        )

        if not wsi_files:
            logger.warning("No %s files found in %s.", self.domain.upper(), input_dir)
            return []

        all_records: list[PatchRecord] = []
        for wsi_path in wsi_files:
            wsi_out = Path(output_dir) / wsi_path.stem
            logger.info("Processing %s → %s", wsi_path.name, wsi_out)
            records = self.extract_from_file(wsi_path, wsi_out)
            all_records.extend(records)
            logger.info(
                "  %s: %d patches accepted.", wsi_path.name, len(records)
            )

        return all_records

    # ------------------------------------------------------------------
    # Domain-specific extraction
    # ------------------------------------------------------------------

    def _extract_tpaf(self, wsi_path: Path, output_dir: Path) -> list[PatchRecord]:
        """Load a TPAF TIFF, compute global statistics, and extract patches.

        Args:
            wsi_path: Path to a TPAF ``.tif`` file.
            output_dir: Output directory for this WSI's patches.

        Returns:
            List of accepted :class:`PatchRecord`.
        """
        data, dtype_str = _load_tpaf_array(wsi_path)
        h, w = data.shape[:2]

        # Compute global statistics on the raw image (before normalization).
        # The 5th-percentile threshold is used for background rejection.
        # Per CLAUDE.md: always use percentile clipping, never min-max.
        raw_for_stats = data[:, :, 0] if data.ndim == 3 else data

        # Background pixels (e.g. zero-signal areas outside tissue) can constitute
        # a large fraction of the WSI.  Computing the 5th percentile over the whole
        # image (including zeros) would yield 0, making the threshold meaningless.
        # We therefore restrict the statistic to non-zero pixels so the threshold
        # reflects the low end of the actual tissue-signal distribution.
        nonzero_pixels = raw_for_stats[raw_for_stats > 0]
        if nonzero_pixels.size > 0:
            global_p5 = float(np.percentile(nonzero_pixels, 5))
        else:
            global_p5 = 0.0
            logger.warning("%s: image contains no non-zero pixels.", wsi_path.name)

        logger.debug(
            "%s global stats (non-zero pixels): p5=%.1f, p%g=%.1f, p%g=%.1f",
            wsi_path.name,
            global_p5,
            self.p_low,
            float(np.percentile(raw_for_stats, self.p_low)),
            self.p_high,
            float(np.percentile(raw_for_stats, self.p_high)),
        )

        # Normalize the whole image once; slice patches from the normalised array.
        normalized, p_low_val, p_high_val = normalize_tpaf(data, self.p_low, self.p_high)

        records: list[PatchRecord] = []
        n_total = n_rejected = 0
        for y, x in _iter_patch_positions(h, w, self.patch_size):
            raw_patch = raw_for_stats[y : y + self.patch_size, x : x + self.patch_size]
            n_total += 1
            if is_background_tpaf(raw_patch, global_p5, self.bg_fraction):
                n_rejected += 1
                continue

            patch = normalized[y : y + self.patch_size, x : x + self.patch_size]
            record = PatchRecord(
                patch_path=Path(),  # filled in by _save_patch
                source_wsi=wsi_path.name,
                x=x,
                y=y,
                patch_size=self.patch_size,
                domain="tpaf",
                dtype_original=dtype_str,
                norm_p_low=self.p_low,
                norm_p_high=self.p_high,
                norm_p_low_value=p_low_val,
                norm_p_high_value=p_high_val,
            )
            _save_patch(patch, record, output_dir)
            records.append(record)

        logger.info(
            "%s: %d/%d patches passed background filter (rejected %d).",
            wsi_path.name,
            len(records),
            n_total,
            n_rejected,
        )
        return records

    def _extract_he_tifffile(
        self, wsi_path: Path, output_dir: Path
    ) -> list[PatchRecord]:
        """Extract patches from an H&E TIFF using ``tifffile``.

        Args:
            wsi_path: Path to an H&E ``.tif`` file.
            output_dir: Output directory for this WSI's patches.

        Returns:
            List of accepted :class:`PatchRecord`.
        """
        data, dtype_str = _load_he_tifffile(wsi_path)
        h, w = data.shape[:2]

        if h * w > 250_000_000:
            logger.warning(
                "%s is very large (%dx%d pixels).  Consider using openslide "
                "for memory-efficient extraction.",
                wsi_path.name,
                h,
                w,
            )

        records: list[PatchRecord] = []
        n_total = n_rejected = 0
        for y, x in _iter_patch_positions(h, w, self.patch_size):
            raw_patch = data[y : y + self.patch_size, x : x + self.patch_size]
            patch_f32 = normalize_he(raw_patch)
            n_total += 1
            if is_background_he(patch_f32, self.bg_fraction):
                n_rejected += 1
                continue

            record = PatchRecord(
                patch_path=Path(),
                source_wsi=wsi_path.name,
                x=x,
                y=y,
                patch_size=self.patch_size,
                domain="he",
                dtype_original=dtype_str,
            )
            _save_patch(patch_f32, record, output_dir)
            records.append(record)

        logger.info(
            "%s: %d/%d patches passed background filter (rejected %d).",
            wsi_path.name,
            len(records),
            n_total,
            n_rejected,
        )
        return records

    def _extract_he_openslide(
        self, wsi_path: Path, output_dir: Path
    ) -> list[PatchRecord]:
        """Extract patches from an H&E slide using ``openslide`` (tile-by-tile).

        Reads each patch region individually so the full WSI is never loaded into
        memory.  All coordinates are at resolution level 0.

        Args:
            wsi_path: Path to an ``.svs`` or ``.ndpi`` file.
            output_dir: Output directory for this WSI's patches.

        Returns:
            List of accepted :class:`PatchRecord`.

        Raises:
            ImportError: If ``openslide-python`` is not installed.
        """
        if not _OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python is required to read .svs/.ndpi files. "
                "Install it with: pip install openslide-python"
            )

        slide = openslide.OpenSlide(str(wsi_path))
        slide_w, slide_h = slide.dimensions  # (width, height) at level 0
        dtype_str = "uint8"

        records: list[PatchRecord] = []
        n_total = n_rejected = 0

        try:
            for y, x in _iter_patch_positions(slide_h, slide_w, self.patch_size):
                # read_region returns a PIL RGBA image.
                region_pil = slide.read_region(
                    location=(x, y),
                    level=0,
                    size=(self.patch_size, self.patch_size),
                )
                raw_patch = np.asarray(region_pil)[:, :, :3]  # drop alpha
                patch_f32 = normalize_he(raw_patch)
                n_total += 1

                if is_background_he(patch_f32, self.bg_fraction):
                    n_rejected += 1
                    continue

                record = PatchRecord(
                    patch_path=Path(),
                    source_wsi=wsi_path.name,
                    x=x,
                    y=y,
                    patch_size=self.patch_size,
                    domain="he",
                    dtype_original=dtype_str,
                )
                _save_patch(patch_f32, record, output_dir)
                records.append(record)
        finally:
            slide.close()

        logger.info(
            "%s: %d/%d patches passed background filter (rejected %d).",
            wsi_path.name,
            len(records),
            n_total,
            n_rejected,
        )
        return records


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def save_manifest(records: list[PatchRecord], output_dir: Path) -> Path:
    """Write a ``manifest.csv`` listing all extracted patches to *output_dir*.

    The CSV contains one row per patch with all fields from :class:`PatchRecord`.
    If *records* is empty a header-only CSV is still written so that downstream
    tools can detect the (empty) run.

    Args:
        records: Patch records returned by :meth:`PatchExtractor.extract_from_directory`.
        output_dir: Directory where ``manifest.csv`` will be written.

    Returns:
        Path to the written manifest file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    fieldnames = list(PatchRecord.__dataclass_fields__.keys())
    # Ensure patch_path is serialised as a string.
    fieldnames_str = fieldnames  # to_dict() handles the conversion

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    logger.info("Manifest written: %s (%d rows).", manifest_path, len(records))
    return manifest_path
