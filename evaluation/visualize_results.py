"""Publication-quality figure generation for SA-CUT virtual H&E staining.

Generates four figure types intended for a thesis chapter (Nature Biomedical
Engineering submission quality):

1. **Comparison** (``--mode comparison``)
   N-row × 6-column panel comparing methods on the same TPAF patches::

       TPAF | Nuclear Mask | CycleGAN | CUT baseline | SA-CUT | Real H&E

2. **Ablation** (``--mode ablation``)
   N-row × 6-column panel isolating contributions of SA-CUT components::

       TPAF | Mask | CUT baseline | CUT+mask | SA-CUT w/o struct | SA-CUT full

3. **Failure cases** (``--mode failures``)
   N-row × 4-column panel showing the *worst* structure-consistency patches
   (lowest IoU), either by reading a pre-computed ``per_image.csv`` from
   :mod:`evaluation.structure_consistency` or by computing IoU on the fly::

       TPAF | Input Mask | SA-CUT output | Extracted Mask (HED+Otsu)

4. **Mask overlay** (``--mode overlay``)
   N-row × 3-column panel that superimposes the input nuclear-mask contour
   on the generated H&E for fine-grained spatial alignment inspection::

       TPAF | Generated H&E + mask contour | Input Mask

Mode ``all`` generates all four figures in sequence.

Scale bar
---------
When ``--pixel_size_um`` (µm / pixel, e.g. ``0.5`` for 0.5 µm/px) and
``--scale_bar_um`` (bar length in µm, default ``50``) are both provided, each
TPAF and H&E panel in the **bottom-right cell of the last row** receives an
anchored white scale bar.  Otherwise scale bars are omitted.

Input directories
-----------------
Every method directory is optional except ``--tpaf_dir`` and ``--mask_dir``
(required by all modes).  Absent columns are shown as grey "N/A" placeholder
cells so a figure can still be generated with whatever subset is available.

File-matching convention
------------------------
All directories are searched for ``.png / .jpg / .npy / .tiff`` files and
matched to TPAF patches by **filename stem** (same convention used in
:mod:`evaluation.structure_consistency`).

Output
------
Each mode saves a ``.png`` (300 dpi, rasterised) and a ``.pdf`` (vector) to
``--output_dir``.  File names: ``{mode}_{n}samples.{ext}``.

CLI example
-----------
::

    python evaluation/visualize_results.py \\
        --mode comparison \\
        --tpaf_dir     data/patches/tpaf \\
        --mask_dir     data/patches/masks \\
        --cyclegan_dir results/cyclegan \\
        --cut_dir      results/cut_baseline \\
        --sa_cut_dir   results/sa_cut \\
        --real_he_dir  data/patches/he \\
        --output_dir   results/figures \\
        --n_samples 8 \\
        --pixel_size_um 0.5 \\
        --scale_bar_um  50 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PIL import Image
from skimage.measure import find_contours

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.structure_consistency import (  # noqa: E402
    _find_files_by_stem,
    _load_he_image,
    _load_mask,
    _load_tpaf,
    evaluate_image,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication style constants
# ---------------------------------------------------------------------------

# Nature Biomedical Engineering two-column width: 180 mm ≈ 7.087 in.
_NBE_FULL_WIDTH_IN: float = 7.087
_CELL_ASPECT: float = 1.0           # height / width per image cell (square patches)
_HEADER_HEIGHT_IN: float = 0.26     # extra height reserved for column-label row

# Cyan contour for mask overlay – accessible to red-green colour-blind readers
# and visible on both pink H&E and dark TPAF backgrounds.
_CONTOUR_COLOR: str = "#00C8FF"
_CONTOUR_LW: float = 0.75

# Green tint for mask overlay on TPAF (green channel only for colour-safety)
_MASK_OVERLAY_ALPHA: float = 0.45
_MASK_OVERLAY_RGB = np.array([0.05, 0.90, 0.25], dtype=np.float64)


# ---------------------------------------------------------------------------
# Publication style setup
# ---------------------------------------------------------------------------


def apply_publication_style() -> None:
    """Configure matplotlib rcParams for thesis / journal figure quality.

    Targets Nature Biomedical Engineering figure guidelines:

    * Sans-serif font (Liberation Sans if available, otherwise DejaVu Sans).
    * Body text 8 pt, column headers 8 pt bold, tick labels 7 pt.
    * No gridlines, all spines hidden on image axes.
    * White figure background, minimal padding.
    * PDF / PS fonts embedded as TrueType (fonttype 42) for vector output.
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    sans = "Liberation Sans" if "Liberation Sans" in available_fonts else "DejaVu Sans"

    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      [sans, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":            8,
        "axes.titlesize":       8,
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        # Spacing
        "figure.dpi":           300,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        # Axes chrome – all hidden on image panels
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     False,
        "axes.spines.bottom":   False,
        "axes.grid":            False,
        "xtick.bottom":         False,
        "ytick.left":           False,
        # Vector PDF/PS settings
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "svg.fonttype":         "none",
        # Lines
        "lines.linewidth":      0.8,
    })


# ---------------------------------------------------------------------------
# Low-level image loading helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".npy", ".tiff", ".tif"}


def _display_he(path: Path) -> np.ndarray:
    """Load generated H&E as uint8 (H, W, 3) for ``imshow``."""
    arr = _load_he_image(path)   # float64 [0,1]
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _display_tpaf(path: Path) -> np.ndarray:
    """Load TPAF as float64 (H, W) [0,1] for ``imshow`` with 'gray' cmap."""
    return _load_tpaf(path)


def _display_mask(path: Path) -> np.ndarray:
    """Load nuclear mask as float64 (H, W) in {0.0, 1.0} for ``imshow``."""
    return _load_mask(path)


def _mask_on_tpaf(
    tpaf: np.ndarray,
    mask: np.ndarray,
    alpha: float = _MASK_OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend a semi-transparent green nuclear-mask tint onto a TPAF image.

    Args:
        tpaf: Float64 (H, W) TPAF in [0, 1].
        mask: Float64 (H, W) binary nuclear mask in {0, 1}.
        alpha: Overlay opacity over mask-positive pixels.

    Returns:
        Uint8 (H, W, 3) RGB composite.
    """
    rgb = np.stack([tpaf, tpaf, tpaf], axis=-1)
    overlay = np.zeros_like(rgb)
    overlay[:] = _MASK_OVERLAY_RGB
    blend = mask[..., np.newaxis]
    composite = rgb * (1.0 - blend * alpha) + overlay * blend * alpha
    return (composite * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Axes drawing primitives
# ---------------------------------------------------------------------------


def _placeholder(ax: plt.Axes, label: str = "N/A", size: int = 256) -> None:
    """Fill *ax* with a dark-grey tile and a centred label text.

    Args:
        ax: Target axes.
        label: Short descriptive string shown in the centre.
        size: Edge size of the placeholder square in pixels.
    """
    ax.imshow(np.full((size, size, 3), 38, dtype=np.uint8))
    ax.text(
        0.5, 0.5, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=7, color="#aaaaaa", fontweight="bold",
    )


def _strip(ax: plt.Axes) -> None:
    """Remove all ticks, tick labels, and spines from *ax*."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scalebar(
    ax: plt.Axes,
    pixel_size_um: float,
    bar_um: float,
    image_w_px: int,
) -> None:
    """Draw an anchored white scale bar in the bottom-right corner of *ax*.

    The bar label has a thin black stroke so it is readable on both white
    and dark backgrounds.

    Args:
        ax: Target axes.
        pixel_size_um: µm per pixel of the displayed image.
        bar_um: Desired scale bar length in µm.
        image_w_px: Image width in pixels (used to scale bar thickness).
    """
    bar_px = bar_um / pixel_size_um
    thickness = max(2, image_w_px // 64)
    label = f"{int(bar_um):d} µm"
    font_props = fm.FontProperties(family="sans-serif", size=6)
    sb = AnchoredSizeBar(
        ax.transData,
        size=bar_px,
        label=label,
        loc="lower right",
        pad=0.25,
        borderpad=0.5,
        sep=2,
        color="white",
        frameon=False,
        size_vertical=thickness,
        fontproperties=font_props,
        label_top=False,
    )
    sb.txt_label.set_path_effects(
        [pe.withStroke(linewidth=1.5, foreground="black")]
    )
    ax.add_artist(sb)


def _contour_overlay(
    ax: plt.Axes,
    he_uint8: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Render H&E then draw nuclear-mask boundary contours on top.

    Contours are extracted at iso-level 0.5 using
    ``skimage.measure.find_contours``.  Each contour is plotted as a solid
    cyan line (colour chosen for colour-blind accessibility).

    Args:
        ax: Target axes.
        he_uint8: Uint8 (H, W, 3) generated H&E image.
        mask: Float64 (H, W) binary nuclear mask in [0, 1].
    """
    ax.imshow(he_uint8, interpolation="lanczos")
    for contour in find_contours(mask, 0.5):
        ax.plot(
            contour[:, 1], contour[:, 0],
            linewidth=_CONTOUR_LW,
            color=_CONTOUR_COLOR,
            solid_capstyle="round",
            antialiased=True,
        )


def _col_headers(first_row_axes: np.ndarray, labels: Sequence[str]) -> None:
    """Set bold column-header titles on the first row of axes.

    Args:
        first_row_axes: 1-D array of ``Axes`` objects (first grid row).
        labels: One label string per column.
    """
    for ax, label in zip(first_row_axes, labels):
        ax.set_title(label, fontsize=8, fontweight="bold", pad=3, loc="center")


def _row_label(ax: plt.Axes, label: str) -> None:
    """Attach a short rotated y-label to the leftmost axes in a row.

    Args:
        ax: Leftmost axes in the row.
        label: Row label text (truncated to 20 characters).
    """
    ax.set_ylabel(
        label[:20],
        fontsize=6,
        rotation=90,
        labelpad=4,
        va="center",
        ha="right",
    )


# ---------------------------------------------------------------------------
# Figure size calculation
# ---------------------------------------------------------------------------


def _figsize(n_cols: int, n_rows: int) -> Tuple[float, float]:
    """Compute (width, height) in inches for a square-cell figure grid.

    Targets the NBE two-column width.  For figures with fewer columns the
    width is proportionally reduced so cells stay square.

    Args:
        n_cols: Number of columns.
        n_rows: Number of image rows.

    Returns:
        ``(width_in, height_in)`` tuple.
    """
    cell_w = _NBE_FULL_WIDTH_IN / max(n_cols, 1)
    height = cell_w * _CELL_ASPECT * n_rows + _HEADER_HEIGHT_IN
    width = cell_w * n_cols
    return width, height


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save *fig* as a 300-dpi PNG and a vector PDF.

    Args:
        fig: Figure to save.
        output_dir: Destination directory (created if needed).
        stem: Base filename without extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 300}), ("pdf", {})]:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext, **kwargs)
        logger.info("Saved  %s", path)


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------


def _sample_uniform(
    stems: List[str],
    n: int,
    rng: np.random.Generator,
) -> List[str]:
    """Sample up to *n* stems uniformly at random without replacement.

    Args:
        stems: Candidate list.
        n: Desired sample size.
        rng: NumPy random generator.

    Returns:
        Sampled subset, preserving sorted order.
    """
    if len(stems) <= n:
        return list(stems)
    idx = sorted(rng.choice(len(stems), size=n, replace=False).tolist())
    return [stems[i] for i in idx]


def _sample_worst_iou(
    stems: List[str],
    mask_stems: Dict[str, Path],
    he_stems: Dict[str, Path],
    n: int,
    struct_csv: Optional[Path],
) -> List[str]:
    """Return the *n* stems with the lowest IoU (worst structure consistency).

    Reads pre-computed IoU values from *struct_csv* if available and the stem
    is present in it; otherwise re-computes IoU via
    :func:`~evaluation.structure_consistency.evaluate_image`.

    Args:
        stems: Candidate list.
        mask_stems: Stem → mask path mapping.
        he_stems: Stem → SA-CUT H&E path mapping.
        n: Number of worst-case stems to return.
        struct_csv: Optional pre-computed CSV from structure_consistency.py.

    Returns:
        Up to *n* stems sorted ascending by IoU (worst first).
    """
    iou_map: Dict[str, float] = {}

    # Try reading from CSV first
    if struct_csv and struct_csv.exists():
        with struct_csv.open() as fh:
            reader = csv_mod.DictReader(fh)
            stem_set = set(stems)
            for row in reader:
                s = row.get("image_stem", "")
                if s in stem_set:
                    try:
                        iou_map[s] = float(row["iou"])
                    except (KeyError, ValueError):
                        pass
        if iou_map:
            logger.info(
                "Read %d IoU scores from '%s'.", len(iou_map), struct_csv
            )

    # Fall back to on-the-fly computation for any missing stems
    missing = [s for s in stems if s not in iou_map and s in he_stems and s in mask_stems]
    if missing:
        logger.info(
            "Computing IoU on-the-fly for %d stems …", len(missing)
        )
        for stem in missing:
            try:
                m, _ = evaluate_image(stem, he_stems[stem], mask_stems[stem])
                iou_map[stem] = m.iou
            except Exception as exc:  # noqa: BLE001
                logger.warning("evaluate_image failed for '%s': %s", stem, exc)

    if not iou_map:
        logger.warning("No IoU values available — using arbitrary order.")
        return stems[:n]

    ranked = sorted(iou_map, key=lambda s: iou_map[s])  # ascending = worst first
    return [s for s in ranked if s in set(stems)][:n]


# ---------------------------------------------------------------------------
# Figure 1 — method comparison
# ---------------------------------------------------------------------------


def make_comparison_figure(
    tpaf_stems: Dict[str, Path],
    mask_stems: Dict[str, Path],
    he_dirs: Dict[str, Optional[Dict[str, Path]]],
    output_dir: Path,
    n_samples: int = 8,
    pixel_size_um: Optional[float] = None,
    scale_bar_um: float = 50.0,
    seed: int = 42,
) -> None:
    """Generate the side-by-side method comparison figure.

    Columns::

        TPAF | Nuclear Mask | CycleGAN | CUT Baseline | SA-CUT | Real H&E

    Missing method directories produce grey "N/A" placeholder cells.

    Args:
        tpaf_stems: Stem → TPAF path mapping.
        mask_stems: Stem → nuclear mask path mapping.
        he_dirs: Dict keyed by ``'cyclegan'``, ``'cut'``, ``'sa_cut'``,
            ``'real_he'``.  Each value is either a stem-map or ``None``.
        output_dir: Destination directory.
        n_samples: Number of patch rows.
        pixel_size_um: µm/pixel.  ``None`` disables scale bars.
        scale_bar_um: Scale bar length in µm.
        seed: RNG seed for reproducible patch selection.
    """
    col_labels = [
        "TPAF", "Nuclear Mask",
        "CycleGAN", "CUT Baseline", "SA-CUT", "Real H&E",
    ]
    method_keys = ["cyclegan", "cut", "sa_cut", "real_he"]

    common = sorted(tpaf_stems.keys() & mask_stems.keys())
    rng = np.random.default_rng(seed)
    selected = _sample_uniform(common, n_samples, rng)
    n_rows = len(selected)
    if n_rows == 0:
        logger.warning("No matched TPAF/mask stems — skipping comparison figure.")
        return

    n_cols = len(col_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figsize(n_cols, n_rows), squeeze=False)
    _col_headers(axes[0], col_labels)

    for ri, stem in enumerate(selected):
        last_row = ri == n_rows - 1

        # ── Col 0: TPAF ──────────────────────────────────────────────────
        ax = axes[ri, 0]
        if stem in tpaf_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            ax.imshow(tpaf, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
            if pixel_size_um and last_row:
                _scalebar(ax, pixel_size_um, scale_bar_um, tpaf.shape[1])
        else:
            _placeholder(ax, "TPAF\nN/A")
        _strip(ax)
        _row_label(ax, f"#{ri + 1}")

        # ── Col 1: Mask overlay on TPAF ──────────────────────────────────
        ax = axes[ri, 1]
        if stem in tpaf_stems and stem in mask_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            mask = _display_mask(mask_stems[stem])
            ax.imshow(_mask_on_tpaf(tpaf, mask), interpolation="lanczos")
        elif stem in mask_stems:
            ax.imshow(
                _display_mask(mask_stems[stem]),
                cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos",
            )
        else:
            _placeholder(ax, "Mask\nN/A")
        _strip(ax)

        # ── Cols 2-5: method H&E outputs ─────────────────────────────────
        for ci, key in enumerate(method_keys):
            ax = axes[ri, 2 + ci]
            stems_map = he_dirs.get(key)
            if stems_map and stem in stems_map:
                he = _display_he(stems_map[stem])
                ax.imshow(he, interpolation="lanczos")
                if pixel_size_um and last_row and ci == len(method_keys) - 1:
                    _scalebar(ax, pixel_size_um, scale_bar_um, he.shape[1])
            else:
                _placeholder(ax)
            _strip(ax)

    fig.subplots_adjust(
        left=0.06, right=0.995, top=0.97, bottom=0.01,
        wspace=0.03, hspace=0.03,
    )
    _save(fig, output_dir, f"comparison_{n_rows}samples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — ablation comparison
# ---------------------------------------------------------------------------


def make_ablation_figure(
    tpaf_stems: Dict[str, Path],
    mask_stems: Dict[str, Path],
    ablation_dirs: Dict[str, Optional[Dict[str, Path]]],
    output_dir: Path,
    n_samples: int = 8,
    pixel_size_um: Optional[float] = None,
    scale_bar_um: float = 50.0,
    seed: int = 42,
) -> None:
    """Generate the ablation study comparison figure.

    Columns::

        TPAF | Mask | CUT | CUT+mask | SA-CUT (w/o L_struct) | SA-CUT (full)

    Args:
        tpaf_stems: Stem → TPAF path mapping.
        mask_stems: Stem → nuclear mask path mapping.
        ablation_dirs: Dict keyed by ``'cut'``, ``'cut_mask'``,
            ``'sa_cut_no_struct'``, ``'sa_cut'``.  Each value is a stem-map
            or ``None``.
        output_dir: Destination directory.
        n_samples: Number of patch rows.
        pixel_size_um: µm/pixel; ``None`` disables scale bars.
        scale_bar_um: Scale bar length in µm.
        seed: RNG seed.
    """
    # LaTeX-style labels render in matplotlib's mathtext engine
    col_labels = [
        "TPAF", "Mask",
        "CUT", "CUT + Mask",
        r"SA-CUT ($\lambda_{struct}=0$)", "SA-CUT (full)",
    ]
    abl_keys = ["cut", "cut_mask", "sa_cut_no_struct", "sa_cut"]

    common = sorted(tpaf_stems.keys() & mask_stems.keys())
    rng = np.random.default_rng(seed)
    selected = _sample_uniform(common, n_samples, rng)
    n_rows = len(selected)
    if n_rows == 0:
        logger.warning("No matched stems — skipping ablation figure.")
        return

    n_cols = len(col_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figsize(n_cols, n_rows), squeeze=False)
    _col_headers(axes[0], col_labels)

    for ri, stem in enumerate(selected):
        last_row = ri == n_rows - 1

        # ── Col 0: TPAF ──────────────────────────────────────────────────
        ax = axes[ri, 0]
        if stem in tpaf_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            ax.imshow(tpaf, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
            if pixel_size_um and last_row:
                _scalebar(ax, pixel_size_um, scale_bar_um, tpaf.shape[1])
        else:
            _placeholder(ax, "TPAF\nN/A")
        _strip(ax)
        _row_label(ax, f"#{ri + 1}")

        # ── Col 1: Mask overlay on TPAF ──────────────────────────────────
        ax = axes[ri, 1]
        if stem in tpaf_stems and stem in mask_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            mask = _display_mask(mask_stems[stem])
            ax.imshow(_mask_on_tpaf(tpaf, mask), interpolation="lanczos")
        elif stem in mask_stems:
            ax.imshow(
                _display_mask(mask_stems[stem]),
                cmap="gray", vmin=0.0, vmax=1.0,
            )
        else:
            _placeholder(ax, "Mask\nN/A")
        _strip(ax)

        # ── Cols 2-5: ablation variants ───────────────────────────────────
        for ci, key in enumerate(abl_keys):
            ax = axes[ri, 2 + ci]
            stems_map = ablation_dirs.get(key)
            if stems_map and stem in stems_map:
                he = _display_he(stems_map[stem])
                ax.imshow(he, interpolation="lanczos")
                if pixel_size_um and last_row and ci == len(abl_keys) - 1:
                    _scalebar(ax, pixel_size_um, scale_bar_um, he.shape[1])
            else:
                _placeholder(ax)
            _strip(ax)

    fig.subplots_adjust(
        left=0.06, right=0.995, top=0.97, bottom=0.01,
        wspace=0.03, hspace=0.03,
    )
    _save(fig, output_dir, f"ablation_{n_rows}samples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — failure cases
# ---------------------------------------------------------------------------


def make_failure_figure(
    tpaf_stems: Dict[str, Path],
    mask_stems: Dict[str, Path],
    sa_cut_stems: Dict[str, Path],
    output_dir: Path,
    n_samples: int = 8,
    pixel_size_um: Optional[float] = None,
    scale_bar_um: float = 50.0,
    struct_csv: Optional[Path] = None,
) -> None:
    """Generate the failure-case visualisation figure.

    Patches are selected as the *worst* N by IoU score (lowest structure
    consistency).  For each patch::

        TPAF | Input Mask | SA-CUT Output | Extracted Mask (HED+Otsu)

    Extracted masks are freshly computed from each SA-CUT output rather than
    cached, so the figure always reflects current checkpoints.

    Args:
        tpaf_stems: Stem → TPAF path mapping.
        mask_stems: Stem → nuclear mask path mapping.
        sa_cut_stems: Stem → SA-CUT generated H&E path mapping.
        output_dir: Destination directory.
        n_samples: Number of worst-case rows to include.
        pixel_size_um: µm/pixel; ``None`` disables scale bars.
        scale_bar_um: Scale bar length in µm.
        struct_csv: Optional ``per_image.csv`` for fast IoU lookup.
    """
    col_labels = [
        "TPAF",
        "Input Mask",
        "SA-CUT Output",
        "Extracted Mask\n(HED + Otsu)",
    ]

    common = sorted(tpaf_stems.keys() & mask_stems.keys() & sa_cut_stems.keys())
    if not common:
        logger.warning(
            "No stems shared by TPAF, mask, and SA-CUT dirs — "
            "skipping failures figure."
        )
        return

    selected = _sample_worst_iou(
        common, mask_stems, sa_cut_stems, n_samples, struct_csv
    )
    n_rows = len(selected)
    if n_rows == 0:
        logger.warning("No stems selected for failure figure.")
        return

    n_cols = len(col_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figsize(n_cols, n_rows), squeeze=False)
    _col_headers(axes[0], col_labels)

    for ri, stem in enumerate(selected):
        last_row = ri == n_rows - 1

        # Re-compute IoU + extracted mask fresh from the SA-CUT output
        try:
            m, extracted = evaluate_image(
                stem, sa_cut_stems[stem], mask_stems[stem]
            )
            iou_val: Optional[float] = m.iou
        except Exception as exc:  # noqa: BLE001
            logger.warning("evaluate_image failed for '%s': %s", stem, exc)
            fallback = 256
            extracted = np.zeros((fallback, fallback), dtype=np.float64)
            iou_val = None

        iou_label = f"IoU={iou_val:.3f}" if iou_val is not None else "IoU=N/A"

        # ── Col 0: TPAF ──────────────────────────────────────────────────
        ax = axes[ri, 0]
        if stem in tpaf_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            ax.imshow(tpaf, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
            if pixel_size_um and last_row:
                _scalebar(ax, pixel_size_um, scale_bar_um, tpaf.shape[1])
        else:
            _placeholder(ax, "TPAF\nN/A")
        _strip(ax)
        _row_label(ax, iou_label)   # Row label shows IoU so reader can rank

        # ── Col 1: Input mask ─────────────────────────────────────────────
        ax = axes[ri, 1]
        mask = _display_mask(mask_stems[stem])
        ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
        _strip(ax)

        # ── Col 2: SA-CUT output ──────────────────────────────────────────
        ax = axes[ri, 2]
        he = _display_he(sa_cut_stems[stem])
        ax.imshow(he, interpolation="lanczos")
        _strip(ax)

        # ── Col 3: HED-extracted mask ─────────────────────────────────────
        ax = axes[ri, 3]
        ax.imshow(extracted, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
        # Overlay dice score as small annotation
        if iou_val is not None:
            ax.text(
                0.97, 0.03, f"IoU {iou_val:.3f}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=5.5, color="white",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
            )
        _strip(ax)

    fig.subplots_adjust(
        left=0.10, right=0.995, top=0.97, bottom=0.01,
        wspace=0.03, hspace=0.03,
    )
    _save(fig, output_dir, f"failures_{n_rows}samples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — mask contour overlay
# ---------------------------------------------------------------------------


def make_overlay_figure(
    tpaf_stems: Dict[str, Path],
    mask_stems: Dict[str, Path],
    sa_cut_stems: Dict[str, Path],
    output_dir: Path,
    n_samples: int = 8,
    pixel_size_um: Optional[float] = None,
    scale_bar_um: float = 50.0,
    seed: int = 42,
) -> None:
    """Generate the nuclear-mask contour overlay figure.

    For each selected patch::

        TPAF | SA-CUT output + input-mask contour | Input Mask

    The cyan contour lines trace the boundary of the input nuclear mask
    on top of the generated H&E, allowing direct visual assessment of how
    well the generator respected the structural prior.

    Args:
        tpaf_stems: Stem → TPAF path mapping.
        mask_stems: Stem → nuclear mask path mapping.
        sa_cut_stems: Stem → SA-CUT generated H&E path mapping.
        output_dir: Destination directory.
        n_samples: Number of patch rows.
        pixel_size_um: µm/pixel; ``None`` disables scale bars.
        scale_bar_um: Scale bar length in µm.
        seed: RNG seed.
    """
    col_labels = [
        "TPAF",
        "SA-CUT + Mask Contour",
        "Input Mask",
    ]

    common = sorted(tpaf_stems.keys() & mask_stems.keys() & sa_cut_stems.keys())
    rng = np.random.default_rng(seed)
    selected = _sample_uniform(common, n_samples, rng)
    n_rows = len(selected)
    if n_rows == 0:
        logger.warning("No common stems for overlay figure — skipping.")
        return

    n_cols = len(col_labels)
    # 3-column figures: widen each cell slightly for contour legibility
    cell_w = (_NBE_FULL_WIDTH_IN / 6) * 2.0
    fig_w = cell_w * n_cols
    fig_h = cell_w * _CELL_ASPECT * n_rows + _HEADER_HEIGHT_IN + 0.22  # legend space

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    _col_headers(axes[0], col_labels)

    # Single shared legend for the contour colour
    legend_patch = mpatches.Patch(
        facecolor="none",
        edgecolor=_CONTOUR_COLOR,
        linewidth=_CONTOUR_LW + 0.4,
        label="Input mask boundary",
    )
    fig.legend(
        handles=[legend_patch],
        loc="lower center",
        ncol=1,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    for ri, stem in enumerate(selected):
        last_row = ri == n_rows - 1

        # ── Col 0: TPAF ──────────────────────────────────────────────────
        ax = axes[ri, 0]
        if stem in tpaf_stems:
            tpaf = _display_tpaf(tpaf_stems[stem])
            ax.imshow(tpaf, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
            if pixel_size_um and last_row:
                _scalebar(ax, pixel_size_um, scale_bar_um, tpaf.shape[1])
        else:
            _placeholder(ax, "TPAF\nN/A")
        _strip(ax)
        _row_label(ax, f"#{ri + 1}")

        # ── Col 1: H&E + contour overlay ─────────────────────────────────
        ax = axes[ri, 1]
        mask = _display_mask(mask_stems[stem])
        if stem in sa_cut_stems:
            he = _display_he(sa_cut_stems[stem])
            _contour_overlay(ax, he, mask)
            if pixel_size_um and last_row:
                _scalebar(ax, pixel_size_um, scale_bar_um, he.shape[1])
        else:
            _placeholder(ax, "H&E\nN/A")
        _strip(ax)

        # ── Col 2: Input mask ─────────────────────────────────────────────
        ax = axes[ri, 2]
        ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="lanczos")
        _strip(ax)

    fig.subplots_adjust(
        left=0.08, right=0.995, top=0.97, bottom=0.05,
        wspace=0.03, hspace=0.03,
    )
    _save(fig, output_dir, f"overlay_{n_rows}samples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate publication-quality SA-CUT result figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["comparison", "ablation", "failures", "overlay", "all"],
        default="all",
        help="Figure type(s) to generate.  'all' runs every mode.",
    )

    # ── Required inputs ────────────────────────────────────────────────
    req = p.add_argument_group("Required inputs (all modes)")
    req.add_argument("--tpaf_dir",  required=True, metavar="DIR",
                     help="TPAF patch directory.")
    req.add_argument("--mask_dir",  required=True, metavar="DIR",
                     help="Nuclear mask directory (.npy float32 [0,1]).")

    # ── Method directories ─────────────────────────────────────────────
    meth = p.add_argument_group("Method result directories (optional)")
    meth.add_argument("--sa_cut_dir",           default=None, metavar="DIR",
                      help="SA-CUT full model H&E output.")
    meth.add_argument("--cyclegan_dir",          default=None, metavar="DIR",
                      help="CycleGAN H&E output.")
    meth.add_argument("--cut_dir",               default=None, metavar="DIR",
                      help="CUT baseline H&E output.")
    meth.add_argument("--cut_mask_dir",          default=None, metavar="DIR",
                      help="CUT+mask ablation H&E output.")
    meth.add_argument("--sa_cut_no_struct_dir",  default=None, metavar="DIR",
                      help="SA-CUT without L_struct H&E output.")
    meth.add_argument("--real_he_dir",           default=None, metavar="DIR",
                      help="Real (unpaired) reference H&E patches.")

    # ── Failure-case options ───────────────────────────────────────────
    fail = p.add_argument_group("Failure-case options")
    fail.add_argument(
        "--struct_csv", default=None, metavar="CSV",
        help=(
            "per_image.csv from evaluation/structure_consistency.py. "
            "When provided IoU values are read directly rather than recomputed."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument("--output_dir",  default="results/figures", metavar="DIR",
                   help="Destination directory for PNG and PDF figures.")

    # ── Display options ───────────────────────────────────────────────
    disp = p.add_argument_group("Display options")
    disp.add_argument("--n_samples",     type=int,   default=8,
                      help="Number of patch rows per figure.")
    disp.add_argument("--pixel_size_um", type=float, default=None, metavar="FLOAT",
                      help="µm per pixel.  Enables scale bars.")
    disp.add_argument("--scale_bar_um",  type=float, default=50.0, metavar="FLOAT",
                      help="Scale bar length in µm.")
    disp.add_argument("--seed",          type=int,   default=42,
                      help="RNG seed for reproducible patch sampling.")
    return p


def _opt_stems(path_str: Optional[str]) -> Optional[Dict[str, Path]]:
    """Load a directory's stem-map, or return ``None`` if path is absent.

    Args:
        path_str: Directory path as a string, or ``None``.

    Returns:
        Stem → Path dict, or ``None``.
    """
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        logger.warning("Directory not found (will use N/A placeholders): '%s'", p)
        return None
    try:
        return _find_files_by_stem(p)
    except FileNotFoundError:
        return None


def main() -> None:
    """Parse CLI arguments and run the requested figure generation modes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()
    apply_publication_style()

    output_dir = Path(args.output_dir)

    # Required directories
    try:
        tpaf_stems = _find_files_by_stem(Path(args.tpaf_dir))
        mask_stems = _find_files_by_stem(Path(args.mask_dir))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info(
        "Found %d TPAF patches, %d masks.",
        len(tpaf_stems), len(mask_stems),
    )

    # Optional method directories
    sa_cut_stems    = _opt_stems(args.sa_cut_dir)          or {}
    cyclegan_stems  = _opt_stems(args.cyclegan_dir)
    cut_stems       = _opt_stems(args.cut_dir)
    cut_mask_stems  = _opt_stems(args.cut_mask_dir)
    sa_cut_ns_stems = _opt_stems(args.sa_cut_no_struct_dir)
    real_he_stems   = _opt_stems(args.real_he_dir)

    struct_csv = Path(args.struct_csv) if args.struct_csv else None

    modes = (
        ["comparison", "ablation", "failures", "overlay"]
        if args.mode == "all"
        else [args.mode]
    )

    shared = dict(
        n_samples=args.n_samples,
        pixel_size_um=args.pixel_size_um,
        scale_bar_um=args.scale_bar_um,
    )

    for mode in modes:
        logger.info("─── mode: %s ───", mode)

        if mode == "comparison":
            make_comparison_figure(
                tpaf_stems=tpaf_stems,
                mask_stems=mask_stems,
                he_dirs={
                    "cyclegan": cyclegan_stems,
                    "cut":      cut_stems,
                    "sa_cut":   sa_cut_stems or None,
                    "real_he":  real_he_stems,
                },
                output_dir=output_dir,
                seed=args.seed,
                **shared,
            )

        elif mode == "ablation":
            make_ablation_figure(
                tpaf_stems=tpaf_stems,
                mask_stems=mask_stems,
                ablation_dirs={
                    "cut":              cut_stems,
                    "cut_mask":         cut_mask_stems,
                    "sa_cut_no_struct": sa_cut_ns_stems,
                    "sa_cut":           sa_cut_stems or None,
                },
                output_dir=output_dir,
                seed=args.seed,
                **shared,
            )

        elif mode == "failures":
            if not sa_cut_stems:
                logger.warning("--sa_cut_dir not provided — skipping failures figure.")
                continue
            make_failure_figure(
                tpaf_stems=tpaf_stems,
                mask_stems=mask_stems,
                sa_cut_stems=sa_cut_stems,
                output_dir=output_dir,
                struct_csv=struct_csv,
                **shared,
            )

        elif mode == "overlay":
            if not sa_cut_stems:
                logger.warning("--sa_cut_dir not provided — skipping overlay figure.")
                continue
            make_overlay_figure(
                tpaf_stems=tpaf_stems,
                mask_stems=mask_stems,
                sa_cut_stems=sa_cut_stems,
                output_dir=output_dir,
                seed=args.seed,
                **shared,
            )

    logger.info("Done.  All figures written to '%s'.", output_dir.resolve())


if __name__ == "__main__":
    main()
