"""SA-CUT training entry point.

Loads and resolves configuration, seeds all RNGs, then delegates to
:class:`~trainers.sa_cut_trainer.SACUTTrainer` for the full training loop.

Usage
-----
Basic (defaults from ``configs/default.yaml``)::

    python scripts/train.py --config configs/experiment_full.yaml

Override individual config keys from the command line::

    python scripts/train.py \\
        --config configs/experiment_full.yaml \\
        --training.n_epochs=100 \\
        --losses.lambda_struct=3.0 \\
        --experiment.name=sa_cut_20260318_ablation

Resume from a checkpoint::

    python scripts/train.py \\
        --config configs/experiment_full.yaml \\
        --resume checkpoints/sa_cut_full/latest.pth

The resolved config (base ← experiment ← CLI overrides) is saved to the
experiment log directory immediately before training starts, so every run
is fully reproducible.
"""

import argparse
import logging
import os
import pprint
import random
import sys
from pathlib import Path

# Smoke test runs on CPU: hide CUDA before torch initialises so device selection
# (cuda-if-available) resolves to CPU even on a machine with a GPU.
if "--fast_dev_run" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" is ambiguous; -1 reliably hides GPUs

import numpy as np
import torch

# Make project root importable when the script is executed from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_utils import load_config, save_config
from trainers.sa_cut_trainer import SACUTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_BASE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the training entry point."""
    parser = argparse.ArgumentParser(
        description="SA-CUT training entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        # allow_abbrev=False so that --training.lr_G is not accidentally
        # abbreviated to --training.l, which could silently match a different key.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config (merged on top of default.yaml).",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=str(_DEFAULT_BASE_CONFIG),
        metavar="PATH",
        help=(
            "Path to the base YAML config (default: configs/default.yaml). "
            "The experiment config is deep-merged on top."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to a checkpoint .pth file to resume training from.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build trainer and validate config, then exit without training.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help=(
            "Smoke test: force CPU, 1 epoch on tiny synthetic data "
            "(data/smoke/, see scripts/make_smoke_data.py), throwaway output dirs. "
            "Runs data -> G/D -> losses -> optimizer step -> checkpoint end-to-end "
            "in seconds to catch wiring/shape/dtype bugs before pushing."
        ),
    )
    return parser


# Config overrides applied by --fast_dev_run (appended last so they win).
_SMOKE_OVERRIDES = [
    "--training.n_epochs=1",
    "--training.n_epochs_decay=0",
    "--data.num_workers=0",
    "--data.augment=false",
    "--data.patch_size=128",
    "--data.tpaf_dir=data/smoke/tpaf",
    "--data.he_dir=data/smoke/he",
    "--mask_provider.mode=precomputed",
    "--mask_provider.mask_dir=data/smoke/masks",
    "--experiment.name=_smoke",
    "--experiment.log_dir=results/_smoke_logs",
    "--experiment.checkpoint_dir=checkpoints/_smoke",
    "--experiment.save_freq=1",
    "--experiment.log_freq=1",
    "--experiment.vis_freq=1",
    "--experiment.use_tensorboard=false",
    "--experiment.use_wandb=false",
    "--losses.struct_warmup_epochs=0",
    "--losses.color_stats_warmup=0",
]


# ---------------------------------------------------------------------------
# Seed helper (redundant with SACUTTrainer, but called early so that any
# code between parse_args() and trainer construction is also seeded)
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, load config, build trainer, run training."""
    parser = _build_parser()

    # Split known args so that --key=value overrides can be forwarded to
    # parse_overrides without argparse complaining about unknown flags.
    known, override_tokens = parser.parse_known_args()

    # Smoke test: append shrink-to-CPU overrides last so they take precedence.
    if known.fast_dev_run:
        override_tokens = override_tokens + _SMOKE_OVERRIDES
        logger.info("--fast_dev_run: CPU smoke test on tiny synthetic data (data/smoke/).")

    # ── Load and resolve config ───────────────────────────────────────────
    cfg = load_config(
        config_path=known.config,
        base_path=known.base_config,
        cli_args=override_tokens,
    )

    # ── Seed early ───────────────────────────────────────────────────────
    seed = getattr(cfg.training, "seed", 42)
    _seed_all(seed)
    logger.info("Global seed set to %d", seed)

    # ── Print resolved config ─────────────────────────────────────────────
    logger.info("Resolved configuration:\n%s", pprint.pformat(cfg.to_dict(), width=100))

    # ── Save config to experiment dir (pre-flight) ────────────────────────
    run_dir = Path(cfg.experiment.log_dir) / cfg.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_flight_path = run_dir / "config_preflight.yaml"
    save_config(cfg, str(pre_flight_path))
    logger.info("Pre-flight config saved → %s", pre_flight_path)

    # ── Build trainer ─────────────────────────────────────────────────────
    trainer = SACUTTrainer(cfg)

    # ── Optional resume ───────────────────────────────────────────────────
    if known.resume is not None:
        logger.info("Resuming from checkpoint: %s", known.resume)
        trainer.load_checkpoint(known.resume)

    # ── Dry-run exit ──────────────────────────────────────────────────────
    if known.dry_run:
        logger.info("--dry_run: trainer built successfully. Exiting.")
        return

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(
        "Starting training: %s  (epochs %d + %d decay)",
        cfg.experiment.name,
        cfg.training.n_epochs,
        cfg.training.n_epochs_decay,
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl-C).")
        logger.info("Saving emergency checkpoint...")
        try:
            trainer.save_checkpoint(trainer.start_epoch, tag="emergency")
            logger.info(
                "Emergency checkpoint saved to '%s/emergency.pth'.",
                trainer.ckpt_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save emergency checkpoint: %s", exc)
        sys.exit(0)

    logger.info("Training finished successfully.")


if __name__ == "__main__":
    main()
