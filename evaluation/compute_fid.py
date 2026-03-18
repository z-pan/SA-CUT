"""FID and KID evaluation for SA-CUT generated virtual H&E images.

Computes two distribution-level realism metrics between a directory of
generated H&E patches and a reference directory of real H&E patches:

FID (Fréchet Inception Distance)
    Measures the Fréchet distance between multivariate Gaussians fitted to
    InceptionV3 pool3 activations (2048-d by default).  Lower is better.
    Sensitive to sample size — reliable from ~2 000 images upwards.

KID (Kernel Inception Distance)
    An unbiased, sample-size-robust alternative to FID that uses a polynomial
    kernel MMD on the same Inception activations.  Lower is better.
    Reported as (mean, std) estimated via ``n_kid_subsets`` random subsets of
    ``kid_subset_size`` images.

Supported input formats
-----------------------
* ``.npy`` patches: loaded as-is and assumed to be float32 in ``[0, 1]``
  (RGB, shape ``(H, W, 3)`` or ``(3, H, W)``).  This is the format written
  by :class:`~data.patch_extractor.PatchExtractor` and
  ``scripts/test.py``.
* ``.png`` / ``.jpg`` / ``.jpeg`` / ``.tiff`` / ``.tif``: loaded via PIL
  and scaled to ``[0, 1]`` float32.

When both image formats are found, all are loaded together.

CLI usage
---------
::

    python evaluation/compute_fid.py \\
        --generated results/generated_he \\
        --real      data/patches/he \\
        --output    results/fid_scores.json

Advanced options::

    python evaluation/compute_fid.py \\
        --generated results/generated_he \\
        --real      data/patches/he \\
        --output    results/fid_scores.json \\
        --dims 2048 \\
        --batch_size 64 \\
        --kid_subset_size 1000 \\
        --n_kid_subsets 100 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Make project root importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Supported image extensions (lowercase, with leading dot).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"}


# ---------------------------------------------------------------------------
# Dataset for loading H&E patches
# ---------------------------------------------------------------------------


class _HEPatchDataset(Dataset):
    """Loads H&E patches from a directory, returning (3, 299, 299) float tensors.

    InceptionV3 is pre-trained on ImageNet at 299×299.  The pytorch-fid
    library handles resizing internally when ``resize_input=True`` (default),
    but we normalise to ``[0, 1]`` here to match that expectation.

    Supported formats: ``.npy``, ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``, ``.tif``.

    Args:
        directory: Root directory to search recursively for patch files.
    """

    def __init__(self, directory: Path) -> None:
        self.files: List[Path] = _find_image_files(directory)
        if not self.files:
            raise RuntimeError(
                f"No image files found under '{directory}'. "
                f"Supported extensions: {sorted(_IMAGE_EXTS)}"
            )
        logger.info("Found %d images in '%s'.", len(self.files), directory)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor:
        """Return a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
        return _load_image(self.files[idx])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _find_image_files(directory: Path) -> List[Path]:
    """Recursively find all supported image files under *directory*.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of file paths with supported extensions.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)


def _load_image(path: Path) -> Tensor:
    """Load one image file as a ``(3, H, W)`` float32 tensor in ``[0, 1]``.

    For ``.npy`` files: array must be float32 in ``[0, 1]`` with shape
    ``(H, W, 3)`` or ``(3, H, W)`` (the format written by the SA-CUT pipeline).

    For raster image files: loaded via PIL and converted to RGB float32.

    Args:
        path: Path to a supported image file.

    Returns:
        Float32 tensor of shape ``(3, H, W)`` in ``[0, 1]``.

    Raises:
        RuntimeError: If the array shape cannot be normalised to ``(3, H, W)``.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # Normalise shape to (3, H, W)
        if arr.ndim == 3 and arr.shape[-1] == 3:          # (H, W, 3) channel-last
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] == 3:         # (3, H, W) channel-first
            pass
        elif arr.ndim == 2:                                 # (H, W) grayscale → replicate
            arr = np.stack([arr, arr, arr], axis=0)
        else:
            raise RuntimeError(
                f"Cannot load '{path}': unexpected npy shape {arr.shape}. "
                "Expected (H,W,3), (3,H,W) or (H,W)."
            )
        tensor = torch.from_numpy(arr).clamp(0.0, 1.0)
    else:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as img:
            rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0   # (H, W, 3) [0,1]
        tensor = torch.from_numpy(
            np.ascontiguousarray(arr.transpose(2, 0, 1))   # (3, H, W)
        )
    return tensor


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def _get_activations(
    files: List[Path],
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    """Extract InceptionV3 activations for all files.

    Wraps ``pytorch_fid.fid_score.get_activations`` but uses our own
    ``_HEPatchDataset`` loader so that ``.npy`` files are also supported.

    Args:
        files: List of image file paths.
        model: Frozen InceptionV3 model.
        batch_size: Number of images per inference batch.
        dims: Feature dimensionality (must match *model* output block).
        device: Compute device.
        num_workers: DataLoader worker count.

    Returns:
        Float64 ndarray of shape ``(N, dims)`` where N = len(files).
    """
    model.eval()

    class _FileDataset(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            return _load_image(files[i])

    loader = DataLoader(
        _FileDataset(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    acts: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # InceptionV3 from pytorch-fid outputs a list; take index 0 (pool3 or
            # the requested block).  Blocks 0–2 have spatial dims > 1×1; apply
            # global average pooling to produce a flat feature vector.
            out = model(batch)[0]           # (B, C, H, W) or (B, C, 1, 1)
            if out.ndim == 4:
                out = out.mean(dim=(-2, -1))  # → (B, C) via global avg pool
            acts.append(out.cpu().numpy())

    return np.concatenate(acts, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------


def compute_fid(
    acts_gen: np.ndarray,
    acts_real: np.ndarray,
) -> float:
    """Compute FID from pre-extracted activation arrays.

    Fits Gaussian statistics (μ, Σ) to each activation set and returns
    the Fréchet distance between them.

    Args:
        acts_gen:  ``(N_gen,  D)`` float64 activation array for generated images.
        acts_real: ``(N_real, D)`` float64 activation array for real images.

    Returns:
        FID scalar (lower is better).
    """
    from pytorch_fid.fid_score import calculate_frechet_distance  # noqa: PLC0415

    mu_gen, sigma_gen = acts_gen.mean(axis=0), np.cov(acts_gen, rowvar=False)
    mu_real, sigma_real = acts_real.mean(axis=0), np.cov(acts_real, rowvar=False)
    return float(calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real))


# ---------------------------------------------------------------------------
# KID computation
# ---------------------------------------------------------------------------


def _polynomial_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute the polynomial kernel matrix K(X, Y).

    ``K[i, j] = (gamma * <X[i], Y[j]> + coef0) ** degree``

    If *gamma* is ``None`` it defaults to ``1 / X.shape[1]`` (standard
    normalisation for high-dimensional features).

    Args:
        X: ``(n, d)`` array.
        Y: ``(m, d)`` array.
        degree: Polynomial degree (default 3, matching the TTUR/KID paper).
        gamma: Kernel bandwidth.  ``None`` → ``1/d``.
        coef0: Additive constant (default 1.0).

    Returns:
        ``(n, m)`` kernel matrix.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X @ Y.T + coef0) ** degree


def compute_kid(
    acts_gen: np.ndarray,
    acts_real: np.ndarray,
    subset_size: int = 1000,
    n_subsets: int = 100,
    degree: int = 3,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Compute KID (Kernel Inception Distance) via random-subset MMD.

    KID is an unbiased, variance-aware estimator of distributional similarity.
    It is better than FID for small sample sizes because FID's Gaussian
    assumption breaks down and it is biased for ``N < 10 000``.

    The estimator follows Bińkowski et al., "Demystifying MMD GANs" (ICLR 2018):

    .. math::

        \\widehat{\\mathrm{KID}} = \\mathbb{E}_{\\text{subset}}\\left[
            \\frac{1}{m(m-1)} \\sum_{i \\neq j} k(x_i, x_j)
            + \\frac{1}{n(n-1)} \\sum_{i \\neq j} k(y_i, y_j)
            - \\frac{2}{mn} \\sum_{i,j} k(x_i, y_j)
        \\right]

    where ``m = n = subset_size``.

    Args:
        acts_gen:    ``(N_gen,  D)`` float64 InceptionV3 activations.
        acts_real:   ``(N_real, D)`` float64 InceptionV3 activations.
        subset_size: Samples per random subset from each domain.  If either
                     domain has fewer samples, ``subset_size`` is clamped.
        n_subsets:   Number of random subsets to average.
        degree:      Polynomial kernel degree (default 3).
        rng_seed:    NumPy RNG seed for reproducibility.

    Returns:
        ``(kid_mean, kid_std)`` tuple (both lower is better, std reflects
        estimation variance across subsets).
    """
    rng = np.random.default_rng(rng_seed)

    # Clamp subset size to the smaller domain.
    n_gen, n_real = acts_gen.shape[0], acts_real.shape[0]
    ss = min(subset_size, n_gen, n_real)
    if ss < subset_size:
        logger.warning(
            "KID subset_size clamped from %d to %d (limited by dataset size: "
            "generated=%d, real=%d).",
            subset_size, ss, n_gen, n_real,
        )

    mmd_values: List[float] = []

    for _ in range(n_subsets):
        idx_gen = rng.choice(n_gen, size=ss, replace=False)
        idx_real = rng.choice(n_real, size=ss, replace=False)
        x = acts_gen[idx_gen]    # (ss, D)
        y = acts_real[idx_real]  # (ss, D)

        # Unbiased MMD² with polynomial kernel
        k_xx = _polynomial_kernel(x, x, degree=degree)
        k_yy = _polynomial_kernel(y, y, degree=degree)
        k_xy = _polynomial_kernel(x, y, degree=degree)

        # Zero diagonal for unbiased within-set terms
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd = (
            k_xx.sum() / (ss * (ss - 1))
            + k_yy.sum() / (ss * (ss - 1))
            - 2.0 * k_xy.mean()
        )
        mmd_values.append(float(mmd))

    arr = np.array(mmd_values)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def compute_metrics(
    generated_dir: Path,
    real_dir: Path,
    dims: int = 2048,
    batch_size: int = 64,
    kid_subset_size: int = 1000,
    n_kid_subsets: int = 100,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
) -> Dict[str, float]:
    """Compute FID and KID between *generated_dir* and *real_dir*.

    This is the main programmatic API.  Loads a frozen InceptionV3, extracts
    activations for both directories, then computes both metrics.

    Args:
        generated_dir: Directory of generated H&E patches/images.
        real_dir: Directory of real H&E patches/images (reference set).
        dims: InceptionV3 feature dimension.  One of ``64, 192, 768, 2048``.
        batch_size: Images per forward pass through InceptionV3.
        kid_subset_size: Subset size per KID estimate.
        n_kid_subsets: Number of KID subsets (more → lower variance).
        device: Torch device.  Auto-detected if ``None``.
        num_workers: DataLoader workers for image loading.

    Returns:
        Dict with keys:
        ``'fid'``, ``'kid_mean'``, ``'kid_std'``,
        ``'n_generated'``, ``'n_real'``,
        ``'dims'``, ``'inception_block'``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validate dims ──────────────────────────────────────────────────
    from pytorch_fid.inception import InceptionV3  # noqa: PLC0415

    if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        raise ValueError(
            f"dims={dims} is not supported. "
            f"Choose from: {sorted(InceptionV3.BLOCK_INDEX_BY_DIM.keys())}."
        )
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # ── Build InceptionV3 ──────────────────────────────────────────────
    logger.info("Loading InceptionV3 (block_idx=%d, dims=%d) on %s.", block_idx, dims, device)
    model = InceptionV3(output_blocks=(block_idx,), resize_input=True, normalize_input=True)
    model = model.to(device).eval()

    # ── Discover files ─────────────────────────────────────────────────
    gen_files = _find_image_files(generated_dir)
    real_files = _find_image_files(real_dir)
    logger.info(
        "Images — generated: %d | real: %d", len(gen_files), len(real_files)
    )
    if not gen_files:
        raise RuntimeError(f"No images found in generated directory: {generated_dir}")
    if not real_files:
        raise RuntimeError(f"No images found in real directory: {real_dir}")

    # ── Extract activations ────────────────────────────────────────────
    logger.info("Extracting activations for generated images...")
    t0 = time.perf_counter()
    acts_gen = _get_activations(gen_files, model, batch_size, dims, device, num_workers)
    logger.info(
        "Generated activations: shape=%s  (%.1fs)", acts_gen.shape, time.perf_counter() - t0
    )

    logger.info("Extracting activations for real images...")
    t0 = time.perf_counter()
    acts_real = _get_activations(real_files, model, batch_size, dims, device, num_workers)
    logger.info(
        "Real activations: shape=%s  (%.1fs)", acts_real.shape, time.perf_counter() - t0
    )

    # ── FID ────────────────────────────────────────────────────────────
    logger.info("Computing FID...")
    t0 = time.perf_counter()
    fid = compute_fid(acts_gen, acts_real)
    logger.info("FID = %.4f  (%.1fs)", fid, time.perf_counter() - t0)

    # ── KID ────────────────────────────────────────────────────────────
    logger.info(
        "Computing KID (subset_size=%d, n_subsets=%d)...", kid_subset_size, n_kid_subsets
    )
    t0 = time.perf_counter()
    kid_mean, kid_std = compute_kid(
        acts_gen, acts_real, subset_size=kid_subset_size, n_subsets=n_kid_subsets
    )
    logger.info(
        "KID = %.6f ± %.6f  (%.1fs)", kid_mean, kid_std, time.perf_counter() - t0
    )

    return {
        "fid": fid,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "n_generated": len(gen_files),
        "n_real": len(real_files),
        "dims": dims,
        "inception_block": block_idx,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FID and KID between generated H&E patches and real H&E reference set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of generated H&E patches (.npy or image files).",
    )
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of real H&E patches (.npy or image files).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to write JSON results. "
            "Defaults to '<generated_dir>/fid_kid_scores.json'."
        ),
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        metavar="{64,192,768,2048}",
        help="InceptionV3 feature dimensionality.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for InceptionV3 feature extraction.",
    )
    parser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of images per random subset for KID estimation.",
    )
    parser.add_argument(
        "--n_kid_subsets",
        type=int,
        default=100,
        metavar="N",
        help="Number of random subsets for KID estimation (more → lower variance).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="DataLoader worker processes for image loading.",
    )
    return parser


def main() -> None:
    """CLI entry point: parse args, compute metrics, save JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _build_parser().parse_args()

    generated_dir = Path(args.generated)
    real_dir = Path(args.real)
    output_path = Path(args.output) if args.output else generated_dir / "fid_kid_scores.json"
    device = torch.device(args.device) if args.device else None

    logger.info("Generated : %s", generated_dir)
    logger.info("Real      : %s", real_dir)
    logger.info("Output    : %s", output_path)

    results = compute_metrics(
        generated_dir=generated_dir,
        real_dir=real_dir,
        dims=args.dims,
        batch_size=args.batch_size,
        kid_subset_size=args.kid_subset_size,
        n_kid_subsets=args.n_kid_subsets,
        device=device,
        num_workers=args.num_workers,
    )

    # Add metadata for traceability.
    import datetime  # noqa: PLC0415

    results["generated_dir"] = str(generated_dir.resolve())
    results["real_dir"] = str(real_dir.resolve())
    results["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    # Attempt to record the git commit hash.
    try:
        import subprocess  # noqa: PLC0415

        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        results["git_commit"] = git_hash
    except Exception:
        results["git_commit"] = "unknown"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Results saved → %s", output_path)
    print(f"\n{'─' * 40}")
    print(f"  FID              : {results['fid']:.4f}")
    print(f"  KID mean         : {results['kid_mean']:.6f}")
    print(f"  KID std          : {results['kid_std']:.6f}")
    print(f"  N generated      : {results['n_generated']}")
    print(f"  N real           : {results['n_real']}")
    print(f"  Inception dims   : {results['dims']}")
    print(f"{'─' * 40}\n")


if __name__ == "__main__":
    main()
