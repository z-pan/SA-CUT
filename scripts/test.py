"""SA-CUT inference entry point.

Loads a trained generator checkpoint, runs it over a directory of TPAF
patches, and writes generated virtual H&E images as PNG files.

Mask acquisition
----------------
Masks are obtained by one of two strategies (auto-detected from CLI flags):

1. **Precomputed** (``--mask_dir``): load pre-saved ``.npy`` masks that
   were computed during training pre-processing.  Fast — zero inference
   overhead per patch.

2. **On-the-fly Cellpose-SAM** (``--mask_checkpoint``): run the frozen
   Cellpose-SAM model on each TPAF image at test time.  Slower but
   requires no pre-processing step.

CRITICAL: always use the same Cellpose checkpoint (or the same pre-computed
masks) that were used during training.  Different segmentations produce
different mask distributions, which degrades translation quality.

Mosaic reconstruction
---------------------
Pass ``--make_mosaic`` to tile the generated patches back into a
WSI-level image.  Patch positions are inferred from the filename stem
(``{slide_name}_y{Y:07d}_x{X:07d}.npy``), matching the convention used
by :class:`~data.patch_extractor.PatchExtractor`.  Each slide found in
the input directory produces one ``.png`` mosaic in ``{output_dir}/mosaics/``.

Usage
-----
Precomputed masks::

    python scripts/test.py \\
        --checkpoint checkpoints/sa_cut_full/latest.pth \\
        --input_dir data/patches/tpaf \\
        --output_dir results/generated_he \\
        --mask_dir data/patches/masks

On-the-fly masks::

    python scripts/test.py \\
        --checkpoint checkpoints/sa_cut_full/latest.pth \\
        --input_dir data/patches/tpaf \\
        --output_dir results/generated_he \\
        --mask_checkpoint checkpoints/cellpose_sam_tpaf.pth

With mosaic reconstruction::

    python scripts/test.py \\
        --checkpoint checkpoints/sa_cut_full/latest.pth \\
        --input_dir data/patches/tpaf \\
        --output_dir results/generated_he \\
        --mask_dir data/patches/masks \\
        --make_mosaic
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

# Make project root importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_utils import ConfigNamespace
from models.generator import ResNetGenerator
from models.mask_provider import (
    CellposeSAMMaskProvider,
    MaskProvider,
    PrecomputedMaskProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Regex matching PatchExtractor filename convention:
#   {slide_stem}_y{Y:07d}_x{X:07d}.npy
_PATCH_STEM_RE = re.compile(r"^(.+)_y(\d+)_x(\d+)$")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the inference entry point."""
    parser = argparse.ArgumentParser(
        description="SA-CUT inference: generate virtual H&E from TPAF patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a .pth checkpoint written by SACUTTrainer.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of TPAF .npy patches to translate (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory where generated H&E PNG images are written.",
    )
    # ── Mask source (mutually exclusive) ──────────────────────────────────
    mask_group = parser.add_mutually_exclusive_group(required=False)
    mask_group.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory of pre-computed .npy nuclear masks.  "
            "Masks are matched to TPAF patches by filename stem."
        ),
    )
    mask_group.add_argument(
        "--mask_checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a Cellpose-SAM .pth checkpoint for on-the-fly mask inference.  "
            "Use the same checkpoint that was active during training."
        ),
    )
    # ── Cellpose-SAM settings (ignored unless --mask_checkpoint is set) ───
    parser.add_argument(
        "--cellpose_model_type",
        type=str,
        default="cyto3",
        metavar="TYPE",
        help="Cellpose model type (default: cyto3). Only used with --mask_checkpoint.",
    )
    parser.add_argument(
        "--cellpose_diameter",
        type=int,
        default=30,
        metavar="PX",
        help="Expected nucleus diameter in pixels (default: 30). Only used with --mask_checkpoint.",
    )
    # ── Mosaic reconstruction ─────────────────────────────────────────────
    parser.add_argument(
        "--make_mosaic",
        action="store_true",
        help=(
            "Tile generated patches into a WSI-level mosaic PNG per slide. "
            "Requires patches to follow the PatchExtractor naming convention: "
            "{slide}_y{Y:07d}_x{X:07d}.npy."
        ),
    )
    parser.add_argument(
        "--mosaic_bg_color",
        type=int,
        default=255,
        metavar="GRAY",
        help="Background colour (0–255) for mosaic regions with no patch (default: 255 = white).",
    )
    # ── Generator tweaks ──────────────────────────────────────────────────
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Override patch_size from the checkpoint config. "
            "Useful when running on full-size images instead of 256×256 crops."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        metavar="N",
        help="Number of patches to process in a single GPU batch (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Torch device override, e.g. 'cpu' or 'cuda:1'. Auto-detected by default.",
    )
    return parser


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _find_tpaf_patches(input_dir: Path) -> List[Path]:
    """Recursively find all .npy patch files under *input_dir*.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of .npy file paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        RuntimeError: If no .npy files are found.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    patches = sorted(input_dir.glob("**/*.npy"))
    if not patches:
        raise RuntimeError(f"No .npy patch files found under: {input_dir}")
    return patches


def _load_tpaf(path: Path) -> Tensor:
    """Load a TPAF .npy patch and return a ``(1, H, W)`` float32 tensor in ``[0, 1]``.

    Mirrors the loading logic in :meth:`~data.dataset.UnpairedTPAFDataset._load_tpaf`.

    Args:
        path: Path to a .npy TPAF patch (float32, already percentile-normalised).

    Returns:
        Float32 tensor of shape ``(1, H, W)`` in ``[0, 1]``.
    """
    array = np.load(path).astype(np.float32)

    if array.ndim == 2:
        array = array[np.newaxis]
    elif array.ndim == 3 and array.shape[-1] in (1, 2):
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
    elif array.ndim == 3 and array.shape[0] in (1, 2):
        pass
    else:
        raise RuntimeError(
            f"Unexpected TPAF shape {array.shape} in '{path}'. "
            "Expected (H,W), (H,W,C) or (C,H,W) with C∈{{1,2}}."
        )

    return torch.from_numpy(array).clamp(0.0, 1.0)


def _tensor_to_pil(fake_he: Tensor) -> Image.Image:
    """Convert a ``(3, H, W)`` ``[-1, 1]`` tensor to an 8-bit RGB PIL image.

    Args:
        fake_he: Float32 tensor of shape ``(3, H, W)`` in ``[-1, 1]``.

    Returns:
        8-bit RGB PIL image.
    """
    rgb = (fake_he * 0.5 + 0.5).clamp(0.0, 1.0)   # → [0, 1]
    rgb_uint8 = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
    return Image.fromarray(rgb_uint8, mode="RGB")


def _output_path(patch_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Compute the output PNG path, mirroring the input subdirectory structure.

    Example::

        patch_path = input_dir / "slide_A" / "slide_A_y0000000_x0000000.npy"
        → output_dir / "slide_A" / "slide_A_y0000000_x0000000.png"

    Args:
        patch_path: Absolute path of the TPAF .npy file.
        input_dir: Root input directory.
        output_dir: Root output directory.

    Returns:
        Absolute path for the output .png file.
    """
    rel = patch_path.relative_to(input_dir)
    return (output_dir / rel).with_suffix(".png")


# ---------------------------------------------------------------------------
# Mosaic reconstruction
# ---------------------------------------------------------------------------


def _parse_patch_coords(stem: str) -> Optional[Tuple[str, int, int]]:
    """Parse a PatchExtractor filename stem into (slide_name, y, x).

    Args:
        stem: Filename without extension, e.g. ``"slide_A_y0000512_x0001024"``.

    Returns:
        ``(slide_name, y, x)`` tuple, or ``None`` if the stem does not match
        the expected pattern.
    """
    m = _PATCH_STEM_RE.match(stem)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _build_mosaics(
    patch_image_map: Dict[Path, Image.Image],
    input_dir: Path,
    output_dir: Path,
    bg_color: int = 255,
) -> None:
    """Tile generated patch images into per-slide WSI-level mosaics.

    Only patches whose filenames match the PatchExtractor naming convention
    ``{slide}_y{Y:07d}_x{X:07d}.npy`` are included; others are silently
    skipped.  Each unique slide name produces one PNG mosaic.

    The mosaic canvas size is determined by the maximum (y + patch_height) and
    (x + patch_width) coordinates found across all patches for that slide.

    Args:
        patch_image_map: Mapping from TPAF patch path → generated PIL image.
        input_dir: Root input directory (used for relative-path computation).
        output_dir: Root output directory.  Mosaics are written to
            ``{output_dir}/mosaics/``.
        bg_color: Background fill colour (0–255) for uncovered pixels.
    """
    mosaic_dir = output_dir / "mosaics"
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    # Group patches by slide name.
    slide_patches: Dict[str, List[Tuple[int, int, Image.Image]]] = defaultdict(list)
    for patch_path, img in patch_image_map.items():
        parsed = _parse_patch_coords(patch_path.stem)
        if parsed is None:
            logger.debug(
                "Skipping mosaic for '%s': filename does not match "
                "{slide}_y{Y:07d}_x{X:07d} convention.",
                patch_path.name,
            )
            continue
        slide_name, y, x = parsed
        slide_patches[slide_name].append((y, x, img))

    if not slide_patches:
        logger.warning(
            "No patches matched the PatchExtractor naming convention; no mosaics produced."
        )
        return

    for slide_name, entries in slide_patches.items():
        # Determine canvas size from bounding box.
        patch_h, patch_w = entries[0][2].size[1], entries[0][2].size[0]
        max_y = max(y + patch_h for y, _, _ in entries)
        max_x = max(x + patch_w for _, x, _ in entries)

        canvas = Image.new("RGB", (max_x, max_y), color=(bg_color, bg_color, bg_color))
        for y, x, img in entries:
            canvas.paste(img, (x, y))

        out_path = mosaic_dir / f"{slide_name}_mosaic.png"
        canvas.save(out_path)
        logger.info(
            "Mosaic saved: %s  (%d patches, canvas %d×%d)",
            out_path.name,
            len(entries),
            max_x,
            max_y,
        )


# ---------------------------------------------------------------------------
# Generator loading
# ---------------------------------------------------------------------------


def _load_generator(
    checkpoint_path: Path, device: torch.device
) -> Tuple[ResNetGenerator, dict]:
    """Load the generator from a SACUTTrainer checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        device: Target device.

    Returns:
        ``(generator, checkpoint_config_dict)`` tuple.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If the checkpoint does not contain 'G_state_dict' or 'config'.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "G_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint '{checkpoint_path}' has no 'G_state_dict' key.")
    if "config" not in ckpt:
        raise KeyError(f"Checkpoint '{checkpoint_path}' has no 'config' key.")

    cfg_dict: dict = ckpt["config"]
    gen_cfg_dict: dict = cfg_dict.get("generator", {})

    # Build a SimpleNamespace so ResNetGenerator receives the expected interface.
    gen_cfg = SimpleNamespace(**gen_cfg_dict)

    G = ResNetGenerator(gen_cfg).to(device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()

    epoch = ckpt.get("epoch", "unknown")
    logger.info(
        "Generator loaded from '%s' (epoch %s).  "
        "input_nc=%d  output_nc=%d  ngf=%d",
        checkpoint_path.name,
        epoch,
        gen_cfg_dict.get("input_nc", "?"),
        gen_cfg_dict.get("output_nc", "?"),
        gen_cfg_dict.get("ngf", "?"),
    )
    return G, cfg_dict


# ---------------------------------------------------------------------------
# Mask provider construction (test-time)
# ---------------------------------------------------------------------------


def _build_test_mask_provider(
    mask_dir: Optional[str],
    mask_checkpoint: Optional[str],
    cellpose_model_type: str,
    cellpose_diameter: int,
    patch_size: Optional[int],
    device: torch.device,
) -> MaskProvider:
    """Construct the appropriate mask provider for test-time inference.

    Args:
        mask_dir: Path to precomputed mask directory (takes priority).
        mask_checkpoint: Path to Cellpose-SAM checkpoint (used when mask_dir is None).
        cellpose_model_type: Cellpose model type string.
        cellpose_diameter: Expected nucleus diameter in pixels.
        patch_size: Expected patch spatial size (for PrecomputedMaskProvider validation).
        device: Compute device (used to decide GPU usage for Cellpose).

    Returns:
        Initialised :class:`~models.mask_provider.MaskProvider`.

    Raises:
        ValueError: If neither mask_dir nor mask_checkpoint is provided.
    """
    if mask_dir is not None:
        logger.info("Mask source: precomputed from '%s'.", mask_dir)
        return PrecomputedMaskProvider(mask_dir=mask_dir, patch_size=patch_size)

    if mask_checkpoint is not None:
        logger.info(
            "Mask source: on-the-fly Cellpose-SAM (checkpoint='%s', model_type='%s').",
            mask_checkpoint,
            cellpose_model_type,
        )
        use_gpu = device.type == "cuda"
        return CellposeSAMMaskProvider(
            checkpoint_path=mask_checkpoint,
            model_type=cellpose_model_type,
            diameter=cellpose_diameter,
            gpu=use_gpu,
        )

    # Neither provided — return a zero-mask fallback with a clear warning.
    logger.warning(
        "No mask source specified (--mask_dir or --mask_checkpoint). "
        "The generator will receive all-zero masks, which may degrade quality. "
        "Specify a mask source for best results."
    )
    return _ZeroMaskProvider()


class _ZeroMaskProvider(MaskProvider):
    """Fallback that returns all-zero masks when no mask source is configured.

    This should only be used for quick debugging.  Zero masks disable
    the structural prior, producing output identical to baseline CUT.
    """

    def get_mask(self, source) -> Tensor:
        """Return a zero mask matching the spatial dimensions of *source*.

        Args:
            source: A TPAF tensor of shape ``(C, H, W)``.

        Returns:
            Float32 zero tensor of shape ``(1, H, W)``.
        """
        if isinstance(source, Tensor):
            return torch.zeros(1, source.shape[-2], source.shape[-1], dtype=torch.float32)
        raise TypeError(f"_ZeroMaskProvider expects a Tensor, got {type(source)}.")


# ---------------------------------------------------------------------------
# Per-patch inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_inference(
    G: ResNetGenerator,
    patches: List[Path],
    mask_provider: MaskProvider,
    device: torch.device,
    batch_size: int,
) -> Dict[Path, Image.Image]:
    """Run the generator over all patches and return a path→image mapping.

    Patches are processed in batches of *batch_size* for GPU efficiency.
    For ``CellposeSAMMaskProvider`` (which operates on single images), the
    batch is assembled by processing masks one-by-one and stacking.

    Args:
        G: Generator in eval mode.
        patches: List of TPAF .npy file paths.
        mask_provider: Mask provider instance.
        device: Compute device.
        batch_size: Number of patches per GPU batch.

    Returns:
        Mapping from TPAF patch path → generated PIL image.
    """
    use_precomputed = isinstance(mask_provider, PrecomputedMaskProvider)
    results: Dict[Path, Image.Image] = {}

    # Process in batches
    for start in tqdm(range(0, len(patches), batch_size), desc="Generating", unit="batch"):
        batch_paths = patches[start : start + batch_size]

        tpaf_tensors: List[Tensor] = []
        mask_tensors: List[Tensor] = []

        for p in batch_paths:
            tpaf = _load_tpaf(p)  # (1, H, W)

            if use_precomputed:
                mask = mask_provider.get_mask(str(p))  # pass path string
            else:
                mask = mask_provider.get_mask(tpaf)    # pass tensor

            # Ensure mask is (1, H, W) float32 in [0, 1]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float().clamp(0.0, 1.0)

            tpaf_tensors.append(tpaf)
            mask_tensors.append(mask)

        tpaf_batch = torch.stack(tpaf_tensors, dim=0).to(device)  # (B, 1, H, W)
        mask_batch = torch.stack(mask_tensors, dim=0).to(device)  # (B, 1, H, W)

        gen_input = torch.cat([tpaf_batch, mask_batch], dim=1)    # (B, 2, H, W)
        fake_he_batch = G(gen_input)                               # (B, 3, H, W) [-1,1]

        for i, p in enumerate(batch_paths):
            results[p] = _tensor_to_pil(fake_he_batch[i])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, load checkpoint, run inference, save outputs."""
    parser = _build_parser()
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load generator ────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    G, ckpt_cfg = _load_generator(ckpt_path, device)

    # Resolve patch_size: CLI override > checkpoint config
    patch_size: Optional[int] = args.patch_size or ckpt_cfg.get("data", {}).get("patch_size")

    # ── Build mask provider ───────────────────────────────────────────────
    mask_provider = _build_test_mask_provider(
        mask_dir=args.mask_dir,
        mask_checkpoint=args.mask_checkpoint,
        cellpose_model_type=args.cellpose_model_type,
        cellpose_diameter=args.cellpose_diameter,
        patch_size=patch_size,
        device=device,
    )

    # ── Find input patches ────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = _find_tpaf_patches(input_dir)
    logger.info("Found %d TPAF patches under '%s'.", len(patches), input_dir)

    # ── Inference ─────────────────────────────────────────────────────────
    results = _run_inference(
        G=G,
        patches=patches,
        mask_provider=mask_provider,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Save patches ──────────────────────────────────────────────────────
    saved = 0
    for patch_path, img in results.items():
        out_path = _output_path(patch_path, input_dir, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        saved += 1

    logger.info("Saved %d generated H&E images to '%s'.", saved, output_dir)

    # ── Optional mosaic reconstruction ────────────────────────────────────
    if args.make_mosaic:
        logger.info("Building WSI mosaics...")
        _build_mosaics(
            patch_image_map=results,
            input_dir=input_dir,
            output_dir=output_dir,
            bg_color=args.mosaic_bg_color,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
