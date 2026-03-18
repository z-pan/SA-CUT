"""Structure consistency evaluation for SA-CUT generated virtual H&E images.

Measures how faithfully the generator preserves input nuclear segmentation
masks in the generated H&E output.  This is the primary quantitative metric
for the structural anchor contribution of SA-CUT.

Pipeline per image
------------------
1. Load the generated H&E patch (PNG or .npy, float32 [0,1] or [0,255]).
2. Extract nuclei from H&E via HED colour deconvolution (Ruifrok & Johnston,
   2001) + Otsu thresholding on the hematoxylin channel.  This is the
   non-differentiable NumPy analogue of the ``StructureConsistencyLoss``
   threshold mode used during training.
3. Load the corresponding input nuclear mask (.npy float32 [0,1]).
4. Binarise the mask at 0.5 (if not already hard-binary).
5. Compute pixel-level metrics between the two binary masks:
   - **Dice coefficient**   = 2|A∩B| / (|A| + |B|)
   - **IoU**               = |A∩B| / |A∪B|
   - **Pixel accuracy**    = (TP + TN) / total
   - **Precision & Recall** (nuclei treated as positive class)
6. Compute connected-component metrics:
   - ``n_components_input``   — number of nuclei in the input mask
   - ``n_components_extracted`` — number of nuclei in the extracted mask
   - ``component_ratio``      — extracted / input  (1.0 is perfect)

Aggregation
-----------
Mean ± std of every metric is computed across all images and written to
``{output_dir}/summary.json``.  Per-image values are written to
``{output_dir}/per_image.csv``.

Visualisation
-------------
A multi-row figure (one row per sampled image, up to ``--max_vis_rows``
rows) with four columns:

  | TPAF | Input Mask | Generated H&E | Extracted Mask (HED+Otsu) |

TPAF column is skipped (column shown as a grey placeholder) when
``--tpaf_dir`` is not provided.

Matching strategy
-----------------
Generated H&E images (.png, .jpg, .npy) are matched to masks (.npy) by
**filename stem**.  The stem of the generated image must match the stem of
the corresponding mask file.  A warning is logged for every unmatched pair.
The same stem-based lookup applies for TPAF files when ``--tpaf_dir`` is given.

CLI
---
::

    python evaluation/structure_consistency.py \\
        --he_dir    results/generated_he \\
        --mask_dir  data/patches/masks \\
        --output_dir results/struct_eval \\
        --tpaf_dir  data/patches/tpaf \\
        --max_vis_rows 16

Do **not** compute PSNR/SSIM between generated H&E and real H&E — there is
no paired ground truth (CLAUDE.md constraint).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.measure import label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"}

# Matches skimage.color.hed_from_rgb exactly (Ruifrok & Johnston, 2001).
# Stored here for reference / assertion — rgb2hed() is used directly.
_HED_FROM_RGB = np.array(
    [[ 1.87798274, -1.00767869, -0.55611582],
     [-0.06590806,  1.13473037, -0.13552180],
     [-0.60190736, -0.48041419,  1.57358807]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ImageMetrics(NamedTuple):
    """Per-image structure consistency metrics."""

    image_stem: str
    dice: float
    iou: float
    pixel_accuracy: float
    precision: float
    recall: float
    n_components_input: int
    n_components_extracted: int
    component_ratio: float            # extracted / input  (0 if input has 0 components)
    input_mask_coverage: float        # fraction of pixels that are nuclear (input)
    extracted_mask_coverage: float    # fraction of pixels that are nuclear (extracted)
    otsu_threshold: float             # Otsu threshold chosen for this image


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _load_he_image(path: Path) -> np.ndarray:
    """Load a generated H&E image as a float64 (H, W, 3) array in [0, 1].

    Accepts .npy (float32 or uint8, shape (H,W,3) or (3,H,W)) and
    standard raster formats (.png, .jpg, etc.) via PIL.

    Args:
        path: Path to the generated H&E file.

    Returns:
        Float64 ndarray of shape (H, W, 3) in [0, 1].

    Raises:
        RuntimeError: If the array shape cannot be normalised to (H, W, 3).
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] == 3:         # (3, H, W) → (H, W, 3)
            arr = arr.transpose(1, 2, 0)
        elif arr.ndim == 3 and arr.shape[-1] == 3:       # already (H, W, 3)
            pass
        else:
            raise RuntimeError(
                f"Cannot load '{path}': shape {arr.shape} is not 3-channel RGB."
            )
        arr = arr.astype(np.float64)
        # Normalise: uint8-range [0,255] → [0,1], clamp float range
        if arr.max() > 1.5:
            arr /= 255.0
        return np.clip(arr, 0.0, 1.0)
    else:
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        return np.clip(rgb, 0.0, 1.0)


def _load_mask(path: Path) -> np.ndarray:
    """Load a nuclear mask .npy file as a binary float64 (H, W) array.

    Mask values are binarised at 0.5 (hard binary masks use {0, 1};
    soft probability maps are thresholded).

    Args:
        path: Path to a .npy mask file (float32 [0,1]).

    Returns:
        Binary float64 ndarray of shape (H, W) with values in {0.0, 1.0}.
    """
    arr = np.load(path).astype(np.float64)
    if arr.ndim == 3:
        arr = arr.squeeze()  # (1, H, W) or (H, W, 1) → (H, W)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape {arr.shape} in '{path}'.")
    return (arr > 0.5).astype(np.float64)


def _load_tpaf(path: Path) -> np.ndarray:
    """Load a TPAF .npy patch as a float64 (H, W) array in [0, 1].

    Args:
        path: Path to the .npy TPAF patch file.

    Returns:
        Float64 ndarray of shape (H, W) in [0, 1].
    """
    arr = np.load(path).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    return np.clip(arr, 0.0, 1.0)


def _find_files_by_stem(directory: Path) -> Dict[str, Path]:
    """Build a stem → path mapping for all supported image files in *directory*.

    Recurses into subdirectories.  If two files share a stem the last one
    (in sorted order) wins and a warning is logged.

    Args:
        directory: Root directory to search.

    Returns:
        Dict mapping filename stem (no extension) to file Path.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    stem_map: Dict[str, Path] = {}
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in _IMAGE_EXTS:
            if p.stem in stem_map:
                logger.warning(
                    "Stem collision in '%s': keeping %s, ignoring %s.",
                    directory, stem_map[p.stem].name, p.name,
                )
            stem_map[p.stem] = p
    return stem_map


# ---------------------------------------------------------------------------
# HED colour deconvolution + Otsu extraction (NumPy, non-differentiable)
# ---------------------------------------------------------------------------


def extract_nuclear_mask_hed(
    he_rgb: np.ndarray,
    otsu_multiplier: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Extract a binary nuclear mask from an H&E RGB image via HED + Otsu.

    This is the non-differentiable NumPy analogue of the ``StructureConsistencyLoss``
    ``'threshold'`` mode used during training.  It is used at evaluation time
    only — gradients are not needed.

    Steps:

    1. Convert float64 RGB [0,1] → HED stain space via ``skimage.color.rgb2hed``
       (Ruifrok & Johnston, 2001).  The hematoxylin channel (index 0) captures
       nuclear staining; positive values indicate hematoxylin presence.
    2. Apply Otsu thresholding on the H channel to find the optimal binary split.
       Multiplying the Otsu threshold by ``otsu_multiplier`` allows slight
       relaxation (> 1.0) or tightening (< 1.0) of the cut-off.
    3. Return the resulting binary mask and the threshold value.

    Note on H-channel sign convention: ``skimage.rgb2hed`` returns *positive*
    H values where hematoxylin is present (nuclear regions are high-H, not
    low-H).  Otsu on this positive H map correctly separates nuclear from
    background.

    Args:
        he_rgb: Float64 (H, W, 3) RGB image in [0, 1].
        otsu_multiplier: Scalar applied to the Otsu threshold before binarisation.
            Default 1.0 (pure Otsu).

    Returns:
        ``(binary_mask, threshold)`` where *binary_mask* is a float64 (H, W)
        array with values in {0.0, 1.0} and *threshold* is the applied Otsu
        cut-off value.
    """
    # rgb2hed expects float64 in [0, 1]; always passes our clip guarantee
    hed = rgb2hed(he_rgb)                        # (H, W, 3)
    h_channel = hed[:, :, 0]                     # hematoxylin channel

    # Otsu on H channel; raises ValueError if channel is uniform
    try:
        thresh = threshold_otsu(h_channel) * otsu_multiplier
    except ValueError:
        # Uniform H channel (blank patch) — default to mid-range
        thresh = 0.0

    binary = (h_channel > thresh).astype(np.float64)
    return binary, float(thresh)


# ---------------------------------------------------------------------------
# Binary mask metrics
# ---------------------------------------------------------------------------


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute pixel-level binary segmentation metrics.

    Both inputs must be float64 (H, W) arrays with values in {0.0, 1.0}.
    The *positive* class represents nuclear pixels.

    Args:
        pred:   Extracted binary mask (model prediction).
        target: Input binary mask (ground-truth structural prior).

    Returns:
        Dict with keys: ``'dice'``, ``'iou'``, ``'pixel_accuracy'``,
        ``'precision'``, ``'recall'``.
    """
    p = pred.astype(bool)
    t = target.astype(bool)

    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    tn = float((~p & ~t).sum())

    n = tp + fp + fn + tn

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    pixel_accuracy = (tp + tn) / (n + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy,
        "precision": precision,
        "recall": recall,
    }


def count_components(binary: np.ndarray) -> int:
    """Count the number of connected components in a binary mask.

    Uses 4-connectivity (cross-shaped structuring element) matching the
    standard biomedical image analysis convention for nucleus counting.

    Args:
        binary: Float64 (H, W) array with values in {0.0, 1.0}.

    Returns:
        Number of distinct connected foreground components.
    """
    _, n = label(binary.astype(np.uint8), connectivity=1, return_num=True)
    return int(n)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------


def evaluate_image(
    stem: str,
    he_path: Path,
    mask_path: Path,
    otsu_multiplier: float = 1.0,
) -> Tuple[ImageMetrics, np.ndarray]:
    """Run the full structure consistency evaluation pipeline for one image pair.

    Args:
        stem: Filename stem (shared by he_path and mask_path).
        he_path: Path to the generated H&E image.
        mask_path: Path to the corresponding input nuclear mask (.npy).
        otsu_multiplier: Passed to :func:`extract_nuclear_mask_hed`.

    Returns:
        ``(metrics, extracted_mask)`` — the per-image metric struct and the
        binary mask extracted from H&E (float64 (H,W), values in {0,1}).
    """
    he_rgb = _load_he_image(he_path)
    input_mask = _load_mask(mask_path)

    # Resize extracted mask to match input mask if spatial sizes differ
    if he_rgb.shape[:2] != input_mask.shape[:2]:
        logger.warning(
            "%s: H&E shape %s ≠ mask shape %s — resizing H&E for metric computation.",
            stem, he_rgb.shape[:2], input_mask.shape[:2],
        )
        he_pil = Image.fromarray((he_rgb * 255).astype(np.uint8))
        he_pil = he_pil.resize(
            (input_mask.shape[1], input_mask.shape[0]), Image.BILINEAR
        )
        he_rgb = np.asarray(he_pil, dtype=np.float64) / 255.0

    extracted, otsu_thresh = extract_nuclear_mask_hed(he_rgb, otsu_multiplier)

    pix = binary_metrics(extracted, input_mask)
    n_in = count_components(input_mask)
    n_ex = count_components(extracted)
    comp_ratio = n_ex / n_in if n_in > 0 else 0.0

    metrics = ImageMetrics(
        image_stem=stem,
        dice=pix["dice"],
        iou=pix["iou"],
        pixel_accuracy=pix["pixel_accuracy"],
        precision=pix["precision"],
        recall=pix["recall"],
        n_components_input=n_in,
        n_components_extracted=n_ex,
        component_ratio=comp_ratio,
        input_mask_coverage=float(input_mask.mean()),
        extracted_mask_coverage=float(extracted.mean()),
        otsu_threshold=otsu_thresh,
    )
    return metrics, extracted


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _he_array_for_display(he_path: Path) -> np.ndarray:
    """Load a generated H&E image as uint8 (H, W, 3) for matplotlib imshow."""
    arr = _load_he_image(he_path)  # float64 [0,1]
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _tpaf_array_for_display(path: Path) -> np.ndarray:
    """Load a TPAF patch as float64 (H, W) in [0, 1] for matplotlib imshow (grey)."""
    return _load_tpaf(path)


def generate_visual_grid(
    records: List[Tuple[str, Path, Path, np.ndarray, Optional[Path]]],
    output_path: Path,
    max_rows: int = 16,
) -> None:
    """Save a 4-column × N-row comparison figure.

    Column layout per row:

        | TPAF (grey) | Input Mask | Generated H&E | Extracted Mask (HED+Otsu) |

    When a TPAF file is not available for a given row the TPAF cell is shown
    as a solid grey placeholder.

    Args:
        records: List of ``(stem, he_path, mask_path, extracted_mask, tpaf_path)``
            tuples.  *tpaf_path* may be ``None``.
        output_path: Destination for the saved figure (e.g. ``.png`` or ``.pdf``).
        max_rows: Maximum number of image rows to include.  Rows are sampled
            uniformly when ``len(records) > max_rows``.
    """
    if not records:
        logger.warning("No records to visualise — skipping figure generation.")
        return

    # Uniform sampling when there are more images than rows.
    if len(records) > max_rows:
        step = len(records) / max_rows
        indices = [int(i * step) for i in range(max_rows)]
        records = [records[i] for i in indices]
    n_rows = len(records)

    n_cols = 4
    fig_h = max(2.5 * n_rows, 4)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, fig_h), squeeze=False
    )

    col_titles = ["TPAF", "Input Mask", "Generated H&E", "Extracted Mask (HED+Otsu)"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=4)

    for row_idx, (stem, he_path, mask_path, extracted, tpaf_path) in enumerate(records):
        # Column 0: TPAF
        ax_tpaf = axes[row_idx, 0]
        if tpaf_path is not None and tpaf_path.exists():
            tpaf_arr = _tpaf_array_for_display(tpaf_path)
            ax_tpaf.imshow(tpaf_arr, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            # Grey placeholder
            h_px = extracted.shape[0]
            ax_tpaf.imshow(
                np.full((h_px, h_px), 0.5), cmap="gray", vmin=0.0, vmax=1.0
            )
            ax_tpaf.text(
                0.5, 0.5, "N/A", transform=ax_tpaf.transAxes,
                ha="center", va="center", fontsize=8, color="white",
            )

        # Column 1: Input nuclear mask
        input_mask_arr = _load_mask(mask_path)
        axes[row_idx, 1].imshow(input_mask_arr, cmap="gray", vmin=0.0, vmax=1.0)

        # Column 2: Generated H&E
        he_arr = _he_array_for_display(he_path)
        axes[row_idx, 2].imshow(he_arr)

        # Column 3: Extracted mask (HED + Otsu)
        axes[row_idx, 3].imshow(extracted, cmap="gray", vmin=0.0, vmax=1.0)

        # Row label on the left
        axes[row_idx, 0].set_ylabel(stem[:24], fontsize=6, rotation=0, labelpad=60, va="center")

    # Remove all tick marks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0.08, 0, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Visualisation saved → %s", output_path)


# ---------------------------------------------------------------------------
# Aggregation + I/O
# ---------------------------------------------------------------------------


def aggregate(metrics_list: List[ImageMetrics]) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std for every scalar metric across all images.

    Args:
        metrics_list: Per-image metrics from :func:`evaluate_image`.

    Returns:
        Nested dict: ``{metric_name: {'mean': float, 'std': float}}``.
    """
    scalar_fields = [
        "dice", "iou", "pixel_accuracy", "precision", "recall",
        "n_components_input", "n_components_extracted", "component_ratio",
        "input_mask_coverage", "extracted_mask_coverage", "otsu_threshold",
    ]
    agg: Dict[str, Dict[str, float]] = {}
    for field in scalar_fields:
        vals = np.array([getattr(m, field) for m in metrics_list], dtype=np.float64)
        agg[field] = {"mean": float(vals.mean()), "std": float(vals.std())}
    agg["n_images"] = {"mean": float(len(metrics_list)), "std": 0.0}
    return agg


def save_csv(metrics_list: List[ImageMetrics], path: Path) -> None:
    """Write per-image metrics to a CSV file.

    Args:
        metrics_list: List of :class:`ImageMetrics` tuples.
        path: Destination CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ImageMetrics._fields)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m._asdict())
    logger.info("Per-image CSV saved → %s", path)


def save_json(summary: Dict, path: Path) -> None:
    """Write an aggregated summary dict to a pretty-printed JSON file.

    Args:
        summary: Dict produced by :func:`aggregate` plus metadata.
        path: Destination JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Aggregate JSON saved → %s", path)


def _print_summary(agg: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary table to stdout."""
    print(f"\n{'─' * 52}")
    print(f"  Structure Consistency Evaluation  (N={int(agg['n_images']['mean'])})")
    print(f"{'─' * 52}")
    metrics_order = [
        ("dice",                    "Dice coefficient"),
        ("iou",                     "IoU"),
        ("pixel_accuracy",          "Pixel accuracy"),
        ("precision",               "Precision"),
        ("recall",                  "Recall"),
        ("component_ratio",         "Component ratio (extr/input)"),
        ("n_components_input",      "# nuclei  (input mask)"),
        ("n_components_extracted",  "# nuclei  (extracted)"),
        ("input_mask_coverage",     "Coverage  (input mask)"),
        ("extracted_mask_coverage", "Coverage  (extracted)"),
        ("otsu_threshold",          "Otsu threshold"),
    ]
    for key, label in metrics_order:
        if key in agg:
            m, s = agg[key]["mean"], agg[key]["std"]
            print(f"  {label:<32s}  {m:8.4f} ± {s:.4f}")
    print(f"{'─' * 52}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate structural consistency of SA-CUT generated H&E images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--he_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of generated H&E images (.png, .jpg, .npy, etc.).",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of input nuclear masks (.npy, float32 [0,1]).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory to write per_image.csv, summary.json, and comparison.png.",
    )
    parser.add_argument(
        "--tpaf_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Optional: directory of TPAF .npy patches for the visualisation figure. "
            "Matched to H&E images by filename stem. "
            "When omitted the TPAF column shows a grey placeholder."
        ),
    )
    parser.add_argument(
        "--otsu_multiplier",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help=(
            "Scalar applied to the Otsu threshold before binarisation. "
            "Values > 1.0 produce a stricter (smaller) nuclear mask; "
            "values < 1.0 produce a more permissive mask. "
            "Default 1.0 (pure Otsu)."
        ),
    )
    parser.add_argument(
        "--max_vis_rows",
        type=int,
        default=16,
        metavar="N",
        help="Maximum number of image rows in the visualisation figure.",
    )
    return parser


def main() -> None:
    """Parse args, run per-image evaluation, save outputs."""
    args = _build_parser().parse_args()

    he_dir = Path(args.he_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    tpaf_dir = Path(args.tpaf_dir) if args.tpaf_dir else None

    # ── Discover files ─────────────────────────────────────────────────
    he_stems = _find_files_by_stem(he_dir)
    mask_stems = _find_files_by_stem(mask_dir)
    tpaf_stems = _find_files_by_stem(tpaf_dir) if tpaf_dir else {}

    # Match by stem: only evaluate images that have a corresponding mask.
    common_stems = sorted(he_stems.keys() & mask_stems.keys())
    if not common_stems:
        raise RuntimeError(
            f"No matching stems found between he_dir ('{he_dir}') and "
            f"mask_dir ('{mask_dir}'). Check that filenames share the same "
            "stem (e.g. 'patch_001.png' matches 'patch_001.npy')."
        )

    unmatched_he = sorted(he_stems.keys() - mask_stems.keys())
    unmatched_mask = sorted(mask_stems.keys() - he_stems.keys())
    if unmatched_he:
        logger.warning(
            "%d H&E image(s) have no matching mask (e.g. '%s').",
            len(unmatched_he), unmatched_he[0],
        )
    if unmatched_mask:
        logger.warning(
            "%d mask file(s) have no matching H&E image (e.g. '%s').",
            len(unmatched_mask), unmatched_mask[0],
        )
    logger.info("Evaluating %d matched image pairs.", len(common_stems))

    # ── Per-image evaluation ───────────────────────────────────────────
    metrics_list: List[ImageMetrics] = []
    vis_records: List[Tuple] = []

    for stem in common_stems:
        he_path = he_stems[stem]
        mask_path = mask_stems[stem]
        tpaf_path = tpaf_stems.get(stem)

        try:
            m, extracted = evaluate_image(
                stem, he_path, mask_path, args.otsu_multiplier
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping '%s': %s", stem, exc)
            continue

        metrics_list.append(m)
        vis_records.append((stem, he_path, mask_path, extracted, tpaf_path))

    if not metrics_list:
        raise RuntimeError("No images could be evaluated. Check input directories.")

    # ── Aggregate ─────────────────────────────────────────────────────
    agg = aggregate(metrics_list)

    import datetime  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    summary = {
        "aggregated": agg,
        "he_dir": str(he_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "tpaf_dir": str(tpaf_dir.resolve()) if tpaf_dir else None,
        "n_evaluated": len(metrics_list),
        "n_skipped": len(common_stems) - len(metrics_list),
        "otsu_multiplier": args.otsu_multiplier,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_hash,
    }

    # ── Save outputs ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(metrics_list, output_dir / "per_image.csv")
    save_json(summary, output_dir / "summary.json")

    # ── Visualisation ─────────────────────────────────────────────────
    generate_visual_grid(
        vis_records,
        output_dir / "comparison.png",
        max_rows=args.max_vis_rows,
    )

    _print_summary(agg)


if __name__ == "__main__":
    main()
