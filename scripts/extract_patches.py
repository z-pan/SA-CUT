"""CLI entry point for WSI patch extraction.

Delegates all logic to :mod:`data.patch_extractor`.

Usage examples::

    # Extract TPAF patches (16-bit .tif, percentile normalization)
    python scripts/extract_patches.py \\
        --input_dir  data/raw/tpaf \\
        --output_dir data/patches/tpaf \\
        --domain     tpaf \\
        --patch_size 256

    # Extract H&E patches from a directory of .svs files
    python scripts/extract_patches.py \\
        --input_dir  data/raw/he \\
        --output_dir data/patches/he \\
        --domain     he \\
        --patch_size 256

    # Stricter background filtering and custom percentile clipping
    python scripts/extract_patches.py \\
        --input_dir   data/raw/tpaf \\
        --output_dir  data/patches/tpaf_strict \\
        --domain      tpaf \\
        --p_low       2.0 \\
        --p_high      98.0 \\
        --bg_fraction 0.7

Exit code: 0 on success, 1 on error.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly as `python scripts/extract_patches.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.patch_extractor import PatchExtractor, save_manifest  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract non-overlapping patches from TPAF or H&E whole-slide images. "
            "For TPAF, percentile normalization is mandatory (never min-max). "
            "See data/patch_extractor.py and CLAUDE.md for full details."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing source WSI files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Root directory for extracted patches, sidecars, and manifest.csv.",
    )
    parser.add_argument(
        "--domain",
        choices=["tpaf", "he"],
        required=True,
        help=(
            "Imaging domain.  'tpaf': 16-bit .tif, percentile normalization. "
            "'he': 8-bit RGB .tif/.svs/.ndpi, scaled to [0, 1]."
        ),
    )

    # --- Patch geometry ---
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        metavar="N",
        help="Side length of square patches in pixels.",
    )

    # --- TPAF normalization (ignored for H&E) ---
    parser.add_argument(
        "--p_low",
        type=float,
        default=1.0,
        metavar="PCT",
        help=(
            "[TPAF only] Lower percentile for intensity clipping. "
            "Per CLAUDE.md, min-max normalization is prohibited."
        ),
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.0,
        metavar="PCT",
        help="[TPAF only] Upper percentile for intensity clipping.",
    )

    # --- Background filtering ---
    parser.add_argument(
        "--bg_fraction",
        type=float,
        default=0.8,
        metavar="F",
        help=(
            "Maximum fraction of background pixels allowed before a patch is "
            "rejected.  TPAF: pixels below global 5th-percentile.  "
            "H&E: near-white pixels (grayscale > 0.9)."
        ),
    )

    # --- Logging ---
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run patch extraction and return an exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # --- Validate inputs ---
    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return 1
    if args.patch_size < 1:
        logger.error("--patch_size must be positive, got %d.", args.patch_size)
        return 1
    if not (0.0 < args.bg_fraction < 1.0):
        logger.error("--bg_fraction must be in (0, 1), got %.3f.", args.bg_fraction)
        return 1
    if args.domain == "tpaf" and args.p_low >= args.p_high:
        logger.error(
            "--p_low (%.1f) must be strictly less than --p_high (%.1f).",
            args.p_low,
            args.p_high,
        )
        return 1

    logger.info(
        "SA-CUT patch extraction — domain=%s, patch_size=%d, input=%s, output=%s",
        args.domain,
        args.patch_size,
        args.input_dir,
        args.output_dir,
    )
    if args.domain == "tpaf":
        logger.info(
            "TPAF normalization: percentile clipping [p_low=%.1f, p_high=%.1f]",
            args.p_low,
            args.p_high,
        )

    # --- Extract ---
    try:
        extractor = PatchExtractor(
            domain=args.domain,
            patch_size=args.patch_size,
            p_low=args.p_low,
            p_high=args.p_high,
            bg_fraction=args.bg_fraction,
        )
        records = extractor.extract_from_directory(args.input_dir, args.output_dir)
    except Exception:
        logger.exception("Patch extraction failed.")
        return 1

    # --- Manifest ---
    try:
        manifest_path = save_manifest(records, args.output_dir)
    except Exception:
        logger.exception("Failed to write manifest.")
        return 1

    logger.info(
        "Done. %d patches extracted. Manifest: %s",
        len(records),
        manifest_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
