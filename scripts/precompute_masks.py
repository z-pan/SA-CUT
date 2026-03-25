"""Batch-generate nuclear segmentation masks from TPAF patches.

Loads a fine-tuned Cellpose-SAM model and runs inference on every TPAF patch
in *input_dir*, writing binary float32 ``.npy`` masks to *output_dir* with
matching filenames.

Outputs
-------
``<output_dir>/<stem>.npy``
    Float32 binary mask, shape ``(1, H, W)``, values ``{0.0, 1.0}``.
``<output_dir>/qc_summary.csv``
    Per-patch QC statistics: filename, num_nuclei, mask_coverage_ratio,
    mean_nucleus_area.
``<output_dir>/qc_montage.png``
    4 × 4 grid of ``[TPAF | mask overlay]`` panels for visual spot-checking.
    Up to 16 randomly sampled patches are shown.

Usage
-----
::

    python scripts/precompute_masks.py \\
        --input_dir  data/patches/tpaf \\
        --output_dir data/patches/masks \\
        --checkpoint checkpoints/cellpose_sam_tpaf.pth \\
        --model_type cyto3

    # Override diameter and disable GPU
    python scripts/precompute_masks.py \\
        --input_dir  data/patches/tpaf \\
        --output_dir data/patches/masks \\
        --checkpoint checkpoints/cellpose_sam_tpaf.pth \\
        --diameter   40 \\
        --no_gpu

Exit codes: 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running directly as `python scripts/precompute_masks.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported input file extensions
# ---------------------------------------------------------------------------

_TPAF_EXTENSIONS = {".tif", ".tiff", ".npy", ".png"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_tpaf_patch(path: Path) -> np.ndarray:
    """Load a TPAF patch and return a float32 array in ``[0, 1]`` of shape ``(H, W)``.

    Args:
        path: Path to the TPAF patch file.

    Returns:
        Float32 ndarray of shape ``(H, W)``, normalised to ``[0, 1]`` using
        percentile clipping (1st–99th percentile) as required by CLAUDE.md.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        import tifffile  # noqa: PLC0415  (import inside fn — optional dep)

        img = tifffile.imread(str(path)).astype(np.float32)
    elif suffix == ".npy":
        img = np.load(str(path)).astype(np.float32)
    elif suffix == ".png":
        from PIL import Image  # noqa: PLC0415

        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported TPAF file extension '{suffix}' for {path}. "
            f"Supported: {sorted(_TPAF_EXTENSIONS)}"
        )

    # Squeeze to (H, W)
    if img.ndim == 3:
        img = img.squeeze()
    if img.ndim != 2:
        raise ValueError(f"Unexpected TPAF array shape {img.shape} in {path}.")

    # Percentile clipping — NEVER min-max (see CLAUDE.md pitfall #2)
    # Cast percentiles to float32: np.percentile returns float64 scalars, and
    # arithmetic with float64 would promote the result array to float64.
    p_low = np.float32(np.percentile(img, 1.0))
    p_high = np.float32(np.percentile(img, 99.0))
    if p_high <= p_low:
        return np.zeros_like(img)
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------


def _patch_qc_stats(
    binary_mask: np.ndarray,
    instance_labels: np.ndarray,
    filename: str,
) -> dict:
    """Compute QC statistics for one patch.

    Args:
        binary_mask: Boolean or float32 array of shape ``(H, W)``.
        instance_labels: Integer instance-label array from Cellpose.
        filename: Original TPAF filename (for the CSV row).

    Returns:
        Dict with keys: ``filename``, ``num_nuclei_detected``,
        ``mask_coverage_ratio``, ``mean_nucleus_area``.
    """
    binary = (binary_mask > 0.5)
    total_pixels = binary.size
    coverage = float(binary.sum()) / total_pixels

    unique = np.unique(instance_labels)
    num_nuclei = int((unique > 0).sum())
    mean_area = (float(binary.sum()) / num_nuclei) if num_nuclei > 0 else 0.0

    return {
        "filename": filename,
        "num_nuclei_detected": num_nuclei,
        "mask_coverage_ratio": round(coverage, 6),
        "mean_nucleus_area": round(mean_area, 2),
    }


def _save_qc_csv(rows: List[dict], output_dir: Path) -> Path:
    """Write QC stats to ``<output_dir>/qc_summary.csv``.

    Args:
        rows: List of dicts from :func:`_patch_qc_stats`.
        output_dir: Directory where the CSV is saved.

    Returns:
        Path to the written CSV file.
    """
    csv_path = output_dir / "qc_summary.csv"
    fieldnames = ["filename", "num_nuclei_detected", "mask_coverage_ratio", "mean_nucleus_area"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("QC summary written to %s", csv_path)
    return csv_path


def _save_qc_montage(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    n_cols: int = 4,
) -> Path:
    """Save a QC montage image showing TPAF patches and mask overlays.

    Each cell in the grid shows the TPAF image in greyscale with the binary
    nuclear mask overlaid in semi-transparent red.

    Args:
        samples: List of ``(tpaf_hw, binary_mask_hw)`` tuples (float32, [0,1]).
            Up to ``n_cols * n_cols`` samples are shown (extras are dropped).
        output_dir: Directory where ``qc_montage.png`` is saved.
        n_cols: Number of columns (and rows) in the grid.  Default 4.

    Returns:
        Path to the saved montage PNG.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not installed — skipping QC montage.")
        return output_dir / "qc_montage.png"

    max_cells = n_cols * n_cols
    samples = samples[:max_cells]
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    # Normalise axes to 2-D array even when n_rows==1
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (tpaf_hw, mask_hw) in enumerate(samples):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Greyscale TPAF as RGB background
        rgb = np.stack([tpaf_hw] * 3, axis=-1)
        # Red overlay: mask channel → red, alpha proportional to mask value
        overlay = np.zeros((*tpaf_hw.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0          # red channel
        overlay[..., 3] = mask_hw * 0.45  # alpha

        ax.imshow(rgb, cmap="gray", vmin=0, vmax=1)
        ax.imshow(overlay)
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(samples), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("SA-CUT QC montage — TPAF + nuclear mask overlay (red)", fontsize=10)
    fig.tight_layout()

    montage_path = output_dir / "qc_montage.png"
    fig.savefig(str(montage_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("QC montage saved to %s", montage_path)
    return montage_path


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def run_precompute(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Optional[Path],
    model_type: str,
    diameter: Optional[int],
    gpu: bool,
    flow_threshold: float,
    cellprob_threshold: float,
    montage_seed: int = 0,
) -> None:
    """Run batch Cellpose-SAM inference and save masks + QC artefacts.

    Args:
        input_dir: Directory of TPAF patch files.
        output_dir: Directory to write ``.npy`` masks and QC files.
        checkpoint: Path to the Cellpose-SAM checkpoint (``.pth`` fine-tuned on
            TPAF).  Pass ``None`` to use the Cellpose built-in weights specified
            by *model_type* (e.g. ``"cyto3"``).
        model_type: Cellpose model type string (e.g. ``"cyto3"``, ``"nuclei"``).
        diameter: Nucleus diameter in pixels (``None`` = auto-estimate).
        gpu: Use CUDA.
        flow_threshold: Cellpose flow threshold.
        cellprob_threshold: Cellpose cell-probability threshold.
        montage_seed: Random seed for patch selection in the QC montage.
    """
    # Collect input files
    patch_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _TPAF_EXTENSIONS
    )
    if not patch_files:
        raise RuntimeError(
            f"No TPAF patches found in {input_dir}. "
            f"Expected files with extensions: {sorted(_TPAF_EXTENSIONS)}"
        )
    logger.info("Found %d TPAF patches in %s", len(patch_files), input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Cellpose model once
    try:
        from cellpose import models as _cellpose_models  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "cellpose is required to run precompute_masks.py. "
            "Install it with: pip install cellpose"
        ) from exc

    if checkpoint is not None:
        logger.info("Loading Cellpose-SAM from %s (model_type=%s)", checkpoint, model_type)
        model = _cellpose_models.CellposeModel(
            pretrained_model=str(checkpoint),
            model_type=model_type,
            gpu=gpu,
        )
    else:
        logger.info("No checkpoint provided — using built-in Cellpose weights (model_type=%s)", model_type)
        model = _cellpose_models.CellposeModel(
            model_type=model_type,
            gpu=gpu,
        )
    # Freeze (this script never trains the model)
    net = getattr(model, "net", None)
    if net is not None:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    qc_rows: List[dict] = []
    montage_samples: List[Tuple[np.ndarray, np.ndarray]] = []

    try:
        from tqdm import tqdm  # noqa: PLC0415

        patch_iter = tqdm(patch_files, desc="Generating masks", unit="patch")
    except ImportError:
        patch_iter = iter(patch_files)  # type: ignore[assignment]

    import torch  # noqa: PLC0415  (defer import — may not be available early)

    for patch_path in patch_iter:
        try:
            tpaf_hw = _load_tpaf_patch(patch_path)
        except Exception as exc:
            logger.warning("Skipping %s — load error: %s", patch_path.name, exc)
            continue

        # Scale [0,1] → [0,255] for Cellpose
        img_cp = (tpaf_hw * 255.0).astype(np.float32)

        with torch.no_grad():
            masks_inst, flows, _ = model.eval(
                img_cp,
                diameter=diameter,
                channels=[0, 0],
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=True,
            )

        # Instance labels → binary float32 (1, H, W)
        binary_hw = (masks_inst > 0).astype(np.float32)
        mask_out = binary_hw[np.newaxis]  # (1, H, W)

        # Save mask
        out_path = output_dir / f"{patch_path.stem}.npy"
        np.save(str(out_path), mask_out)

        # QC stats
        qc_rows.append(_patch_qc_stats(binary_hw, masks_inst, patch_path.name))
        montage_samples.append((tpaf_hw, binary_hw))

    if not qc_rows:
        logger.error("No patches were processed successfully.")
        sys.exit(1)

    # Write QC artefacts
    _save_qc_csv(qc_rows, output_dir)

    # Select up to 16 random patches for the montage
    rng = random.Random(montage_seed)
    selected = rng.sample(montage_samples, min(16, len(montage_samples)))
    _save_qc_montage(selected, output_dir)

    logger.info(
        "Done. %d/%d patches processed. Masks saved to %s",
        len(qc_rows),
        len(patch_files),
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-generate nuclear segmentation masks from TPAF patches using "
            "a fine-tuned Cellpose-SAM model.  Run this once before training "
            "SA-CUT to avoid on-the-fly inference overhead."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing TPAF patch files (.tif/.tiff/.npy/.png).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory where binary .npy masks and QC artefacts will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PTH",
        help=(
            "Path to the Cellpose-SAM checkpoint (.pth) fine-tuned on TPAF. "
            "Omit to use the Cellpose built-in weights for --model_type."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cyto3",
        metavar="TYPE",
        help="Cellpose model type (e.g. 'cyto3', 'nuclei').",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=30,
        metavar="PX",
        help="Expected nucleus diameter in pixels. Pass 0 to auto-estimate.",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.4,
        help="Cellpose flow threshold for mask filtering.",
    )
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=0.0,
        help="Cellpose cell-probability threshold.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        default=False,
        help="Disable GPU; run Cellpose on CPU.",
    )
    parser.add_argument(
        "--montage_seed",
        type=int,
        default=0,
        help="Random seed for patch selection in the QC montage.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the precompute_masks CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        logger.error("--input_dir does not exist: %s", args.input_dir)
        sys.exit(1)

    if args.checkpoint is not None and not args.checkpoint.exists():
        logger.error("--checkpoint does not exist: %s", args.checkpoint)
        sys.exit(1)

    diameter = args.diameter if args.diameter > 0 else None

    try:
        run_precompute(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            diameter=diameter,
            gpu=not args.no_gpu,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            montage_seed=args.montage_seed,
        )
    except Exception as exc:
        logger.error("Precomputation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
