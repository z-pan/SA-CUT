"""SA-CUT training orchestrator.

Implements the three-phase training protocol from CLAUDE.md:

  Phase 1 — Warm-up (epochs 0 … struct_warmup_epochs−1)
      Active losses: ``L_adv + L_SA-PatchNCE``
      ``L_struct`` is disabled (generator hasn't learned color mapping yet).

  Phase 2 — Full training (epochs struct_warmup_epochs … n_epochs−1)
      Active losses: ``L_adv + L_SA-PatchNCE + L_struct + L_idt``
      ``L_struct`` ramps linearly 0 → ``lambda_struct`` over
      ``struct_rampup_epochs`` epochs.

  Phase 3 — LR decay (epochs n_epochs … n_epochs + n_epochs_decay − 1)
      Linear learning-rate decay from the base LR to 0.
      All losses remain active.

Reproducibility
---------------
Every run logs its git commit hash, resolved config YAML, and all random
seeds (Python, NumPy, PyTorch, CUDA) to the experiment directory.

Mask acquisition
----------------
``cfg.mask_provider.mode`` selects the mask source:

  ``'precomputed'``
      Masks are loaded from disk by :class:`~data.dataset.UnpairedTPAFDataset`
      (mask_dir is passed to the dataset constructor).  Zero compute overhead.

  ``'cellpose_sam'``
      The dataset returns zero masks.  After each batch, the trainer calls
      ``mask_provider.get_mask(tpaf[b])`` for each sample.  The frozen
      Cellpose model's parameters are asserted absent from all optimizers.

Generated H&E re-encoding for PatchNCE
----------------------------------------
``G.get_encoder_features`` expects a 2-channel input (TPAF + mask) but the
generated image is 3-channel RGB.  We convert the generated H&E to
luminance-weighted grayscale ``[0, 1]`` and append the *same* nuclear mask
as the second channel.  This gives the encoder identical structural
conditioning for both source and generated representations, making the
PatchNCE features directly comparable across domains.
"""

from __future__ import annotations

import contextlib
import warnings
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from configs.config_utils import ConfigNamespace, save_config
from data.dataset import UnpairedTPAFDataset
from losses.gan_loss import GANLoss
from losses.sa_patchnce import RegionAwarePatchNCE
from losses.structure_loss import StructureConsistencyLoss
from models.discriminator import NLayerDiscriminator
from models.generator import ResNetGenerator
from models.mask_provider import (
    CellposeSAMMaskProvider,
    MaskProvider,
    build_mask_provider,
)
from models.networks import init_weights

logger = logging.getLogger(__name__)

# Default PatchNCE layer indices — match the ResNetGenerator table in CLAUDE.md.
# Indices 0, 4, 8, 12, 16 capture: padded input, first downsample, second
# downsample post-norm, ResBlock 3, ResBlock 7 outputs.
_DEFAULT_PATCHNCE_INDICES: List[int] = [0, 4, 8, 12, 16]


# ---------------------------------------------------------------------------
# Logging back-ends
# ---------------------------------------------------------------------------


class _NullWriter:
    """Silent no-op writer used when TensorBoard / wandb is unavailable."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _build_tb_writer(log_dir: str) -> object:
    """Return a TensorBoard SummaryWriter, or a _NullWriter if unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging → %s", log_dir)
        return writer
    except ImportError:
        logger.warning("tensorboard not installed — TensorBoard logging disabled.")
        return _NullWriter()


def _init_wandb(cfg: ConfigNamespace, run_dir: str) -> object:
    """Initialise a wandb run, or return a _NullWriter if disabled/unavailable."""
    if not getattr(cfg.experiment, "use_wandb", False):
        return _NullWriter()
    try:
        import wandb  # noqa: PLC0415

        run = wandb.init(
            project=getattr(cfg.experiment, "wandb_project", "SA-CUT"),
            name=cfg.experiment.name,
            config=cfg.to_dict(),
            dir=run_dir,
            resume="allow",
        )
        logger.info("wandb run: %s", run.url)
        return run
    except ImportError:
        logger.warning("wandb not installed — wandb logging disabled.")
        return _NullWriter()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _set_requires_grad(net: nn.Module, flag: bool) -> None:
    """Toggle ``requires_grad`` for all parameters of *net* in-place."""
    for p in net.parameters():
        p.requires_grad_(flag)


def _rgb_to_gray(rgb_neg1_1: Tensor) -> Tensor:
    """Convert ``[-1, 1]`` RGB to luminance-weighted grayscale ``[0, 1]``.

    Args:
        rgb_neg1_1: ``(B, 3, H, W)`` float32 in ``[-1, 1]``.

    Returns:
        ``(B, 1, H, W)`` float32 in ``[0, 1]``.
    """
    rgb_01 = rgb_neg1_1 * 0.5 + 0.5
    gray = (
        0.299 * rgb_01[:, 0:1]
        + 0.587 * rgb_01[:, 1:2]
        + 0.114 * rgb_01[:, 2:3]
    )
    return gray.clamp(0.0, 1.0)


def _seed_everything(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_git_hash() -> str:
    """Return the short HEAD git hash, or 'unknown' on any failure."""
    try:
        import subprocess  # noqa: PLC0415

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class SACUTTrainer:
    """Main SA-CUT training orchestrator.

    Constructs and wires all components, then exposes :meth:`train` for the
    full run and :meth:`train_one_epoch` for a single pass.

    Args:
        cfg: Fully resolved config namespace (from
            :func:`configs.config_utils.load_config`).
    """

    def __init__(self, cfg: ConfigNamespace) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        # ── Reproducibility ────────────────────────────────────────────────
        seed: int = getattr(cfg.training, "seed", 42)
        _seed_everything(seed)
        logger.info("Seed: %d  |  git: %s", seed, _get_git_hash())

        # ── Experiment directories ─────────────────────────────────────────
        self.run_dir = Path(cfg.experiment.log_dir) / cfg.experiment.name
        self.ckpt_dir = Path(cfg.experiment.checkpoint_dir) / cfg.experiment.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, str(self.run_dir / "resolved_config.yaml"))
        (self.run_dir / "git_hash.txt").write_text(_get_git_hash() + "\n")

        # ── Models ────────────────────────────────────────────────────────
        self.G: ResNetGenerator = ResNetGenerator(cfg.generator).to(self.device)
        self.D: NLayerDiscriminator = NLayerDiscriminator(cfg.discriminator).to(self.device)
        init_weights(self.G, cfg.generator.init_type, cfg.generator.init_gain)
        init_weights(self.D, cfg.discriminator.init_type, cfg.discriminator.init_gain)
        logger.info(
            "Generator  params: %d",
            sum(p.numel() for p in self.G.parameters()),
        )
        logger.info(
            "Discriminator params: %d",
            sum(p.numel() for p in self.D.parameters()),
        )

        # ── Mask provider ──────────────────────────────────────────────────
        self.mask_provider: MaskProvider = build_mask_provider(cfg)
        self._onthefly_mask: bool = isinstance(self.mask_provider, CellposeSAMMaskProvider)
        logger.info(
            "Mask mode: %s",
            "on-the-fly Cellpose-SAM" if self._onthefly_mask else "precomputed",
        )

        # ── PatchNCE layer configuration ───────────────────────────────────
        self.layer_indices: List[int] = list(
            getattr(cfg.losses, "patchnce_layer_indices", _DEFAULT_PATCHNCE_INDICES)
        )
        n_layers = len(self.layer_indices)

        # ── Losses ────────────────────────────────────────────────────────
        self.criterion_gan = GANLoss(cfg.losses.gan_mode).to(self.device)
        self.criterion_patchnce = RegionAwarePatchNCE(
            n_layers=n_layers,
            n_patches=cfg.losses.n_patches_per_layer,
            tau=getattr(cfg.losses, "patchnce_tau", 0.07),
        ).to(self.device)
        self.criterion_struct = StructureConsistencyLoss(
            mode=cfg.losses.struct_loss_mode,
            threshold=getattr(cfg.losses, "struct_threshold_hematoxylin", 0.15),
            sharpness=getattr(cfg.losses, "struct_sharpness", 15.0),
            warmup_epochs=cfg.losses.struct_warmup_epochs,
            ramp_epochs=cfg.losses.struct_rampup_epochs,
        ).to(self.device)

        # Trigger lazy MLP head construction before the optimizer is created.
        # This ensures MLP parameters are captured by Adam from step 1.
        self._init_patchnce_mlps()

        # ── Optimizers ────────────────────────────────────────────────────
        # G optimizer includes PatchNCE MLP heads (they are part of G's branch).
        self._g_params: List[nn.Parameter] = (
            list(self.G.parameters())
            + list(self.criterion_patchnce.mlp_heads.parameters())
        )
        self.optimizer_G = Adam(
            self._g_params,
            lr=cfg.training.lr_G,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.optimizer_D = Adam(
            self.D.parameters(),
            lr=cfg.training.lr_D,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )

        # Safety: Cellpose-SAM must never appear in any optimizer.
        self._assert_cellpose_not_in_optimizers()

        # ── LR schedulers ─────────────────────────────────────────────────
        lr_lambda = self._make_lr_lambda(
            cfg.training.n_epochs, cfg.training.n_epochs_decay
        )
        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=lr_lambda)

        # ── AMP ───────────────────────────────────────────────────────────
        self.use_amp: bool = (
            getattr(cfg.training, "use_amp", False) and torch.cuda.is_available()
        )
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if self.use_amp else None
        )

        # ── Dataset + DataLoader ───────────────────────────────────────────
        mask_dir_for_dataset = (
            cfg.mask_provider.mask_dir
            if cfg.mask_provider.mode == "precomputed"
            else None
        )
        dataset = UnpairedTPAFDataset(
            tpaf_dir=cfg.data.tpaf_dir,
            he_dir=cfg.data.he_dir,
            mask_dir=mask_dir_for_dataset,
            patch_size=cfg.data.patch_size,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=getattr(cfg.data, "num_workers", 4),
            pin_memory=getattr(cfg.data, "pin_memory", True),
            drop_last=True,
        )
        logger.info("Dataset: %d samples, %d batches/epoch", len(dataset), len(self.dataloader))

        # ── Logging ───────────────────────────────────────────────────────
        self.tb_writer = (
            _build_tb_writer(str(self.run_dir / "tb"))
            if getattr(cfg.experiment, "use_tensorboard", True)
            else _NullWriter()
        )
        self.wandb_run = _init_wandb(cfg, str(self.run_dir))

        # ── Training state ─────────────────────────────────────────────────
        self.start_epoch: int = 0
        self.global_step: int = 0
        self.grad_clip: float = getattr(cfg.training, "grad_clip_norm", 0.0)
        self.d_steps_per_g: int = getattr(cfg.training, "d_steps_per_g", 1)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_patchnce_mlps(self) -> None:
        """Run a dummy PatchNCE forward to populate MLP heads before optimizer creation.

        RegionAwarePatchNCE builds its per-scale MLP heads on the first forward
        call (lazy initialisation).  Running a dummy pass here ensures the heads
        exist — and their parameters are included in Adam — before training starts.
        """
        ps = self.cfg.data.patch_size
        dummy = torch.zeros(
            1, self.cfg.generator.input_nc, ps, ps, device=self.device
        )
        with torch.no_grad():
            dummy_feats = self.G.get_encoder_features(dummy, self.layer_indices)
            self.criterion_patchnce(dummy_feats, dummy_feats, mask=None)
        logger.info(
            "PatchNCE: %d MLP heads built (nc=%d)",
            len(self.criterion_patchnce.mlp_heads),
            self.criterion_patchnce.nc,
        )

    def _assert_cellpose_not_in_optimizers(self) -> None:
        """Assert that frozen Cellpose-SAM parameters are absent from all optimizers.

        Raises:
            AssertionError: If any Cellpose parameter is found in optimizer_G
                or optimizer_D.
        """
        if not self._onthefly_mask:
            return  # precomputed mode — no inference model to check

        cellpose_ids = {id(p) for p in self.mask_provider.parameters()}
        # CellposeSAMMaskProvider.parameters() returns iter([]) by design,
        # so cellpose_ids is always empty.  We keep the assertion to guard
        # against any future refactor that might accidentally expose params.
        for opt_name, opt in [("optimizer_G", self.optimizer_G), ("optimizer_D", self.optimizer_D)]:
            for group in opt.param_groups:
                for p in group["params"]:
                    assert id(p) not in cellpose_ids, (
                        f"Frozen CellposeSAMMaskProvider parameter found in {opt_name}. "
                        "The mask provider must never appear in any optimizer.  "
                        "Check _g_params construction in SACUTTrainer.__init__."
                    )
        logger.info("Optimizer safety check: Cellpose parameters absent from all optimizers. ✓")

    @staticmethod
    def _make_lr_lambda(n_epochs: int, n_epochs_decay: int):
        """Build the LR schedule function for LambdaLR.

        Returns 1.0 for the first *n_epochs* (Phases 1 and 2), then decays
        linearly to 0.0 over the next *n_epochs_decay* epochs (Phase 3).

        Args:
            n_epochs: Epochs with constant LR.
            n_epochs_decay: Epochs for linear decay.

        Returns:
            Callable ``epoch → float`` in ``[0.0, 1.0]``.
        """
        def lr_lambda(epoch: int) -> float:
            if epoch < n_epochs:
                return 1.0
            return max(0.0, 1.0 - (epoch - n_epochs) / max(n_epochs_decay, 1))

        return lr_lambda

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the complete training loop (all phases).

        Saves a ``latest.pth`` checkpoint after every epoch and a dated
        checkpoint every ``cfg.experiment.save_freq`` epochs.
        """
        total_epochs = self.cfg.training.n_epochs + self.cfg.training.n_epochs_decay
        logger.info(
            "Starting training: epochs %d → %d",
            self.start_epoch,
            total_epochs - 1,
        )

        for epoch in range(self.start_epoch, total_epochs):
            t0 = time.time()
            epoch_losses = self.train_one_epoch(epoch)

            # Advance LR schedulers (epoch-level)
            self.scheduler_G.step()
            self.scheduler_D.step()

            lr_g = self.scheduler_G.get_last_lr()[0]
            elapsed = time.time() - t0
            logger.info(
                "Epoch %03d/%d  %.0fs  lr=%.2e  "
                "G=%.4f  D=%.4f  nce=%.4f  struct=%.4f  idt=%.4f",
                epoch, total_epochs - 1, elapsed, lr_g,
                epoch_losses.get("loss_G", float("nan")),
                epoch_losses.get("loss_D", float("nan")),
                epoch_losses.get("loss_nce", float("nan")),
                epoch_losses.get("loss_struct", float("nan")),
                epoch_losses.get("loss_idt", float("nan")),
            )

            for k, v in epoch_losses.items():
                self.tb_writer.add_scalar(f"epoch/{k}", v, epoch)

            # Always save latest; save dated checkpoint periodically
            self.save_checkpoint(epoch, tag="latest")
            if (epoch + 1) % self.cfg.experiment.save_freq == 0:
                self.save_checkpoint(epoch, tag=f"epoch_{epoch:04d}")

        logger.info("Training complete.")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one full pass over the DataLoader.

        Args:
            epoch: Current epoch index (0-indexed).

        Returns:
            Dict of mean loss values over the epoch.
        """
        self.G.train()
        self.D.train()

        accum: Dict[str, float] = defaultdict(float)
        n_iters = 0

        for batch_idx, batch in enumerate(self.dataloader):
            tpaf: Tensor = batch["tpaf"].to(self.device)      # (B, 1, H, W) [0,1]
            real_he: Tensor = batch["he"].to(self.device)      # (B, 3, H, W) [-1,1]
            mask: Tensor = batch["mask"].to(self.device)       # (B, 1, H, W) [0,1]

            # ── On-the-fly mask (Cellpose-SAM mode) ───────────────────────
            if self._onthefly_mask:
                masks = []
                for b in range(tpaf.shape[0]):
                    with torch.no_grad():
                        m = self.mask_provider.get_mask(tpaf[b])  # (1, H, W)
                    masks.append(m.to(self.device))
                mask = torch.stack(masks, dim=0)  # (B, 1, H, W)

            # ── Single generator forward (reused by D and G updates) ──────
            # fake_he is computed WITH gradient so G's graph is intact for
            # the G update.  The D update receives fake_he.detach().
            gen_input = torch.cat([tpaf, mask], dim=1)  # (B, 2, H, W)
            with self._amp_ctx():
                fake_he: Tensor = self.G(gen_input)

            # ── Discriminator update(s) ────────────────────────────────────
            d_losses: Dict[str, float] = {}
            for _ in range(self.d_steps_per_g):
                d_losses = self.update_D(real_he, fake_he.detach())

            # ── Generator update ───────────────────────────────────────────
            g_losses = self.update_G(tpaf, mask, real_he, fake_he, gen_input, epoch)

            # ── Accumulate ────────────────────────────────────────────────
            all_losses = {**d_losses, **g_losses}
            for k, v in all_losses.items():
                accum[k] += v
            n_iters += 1

            if self.global_step % self.cfg.experiment.log_freq == 0:
                self.log_metrics(all_losses, epoch, self.global_step)

            if self.global_step % self.cfg.experiment.vis_freq == 0:
                self.log_images(
                    tpaf.detach(), mask.detach(),
                    fake_he.detach(), real_he.detach(),
                    epoch,
                )

            self.global_step += 1

        return {k: v / max(n_iters, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    # D update
    # ------------------------------------------------------------------

    def update_D(self, real_he: Tensor, fake_he: Tensor) -> Dict[str, float]:
        """Discriminator gradient step with optional R1 gradient penalty.

        Training:
          1. Real loss:  ``0.5 * L_adv(D(real), True)``
          2. Fake loss:  ``0.5 * L_adv(D(fake.detach()), False)``
          3. R1 penalty: ``lambda_r1 * 0.5 * E[||∇_x D(x)||^2]`` (if enabled)

        ``fake_he`` **must** be detached by the caller to prevent G from
        receiving gradients through this step.

        Args:
            real_he: Real H&E patches, ``(B, 3, H, W)`` in ``[-1, 1]``.
            fake_he: Detached generated H&E, same shape.

        Returns:
            Dict with keys ``'loss_D'``, ``'loss_D_real'``, ``'loss_D_fake'``,
            ``'loss_D_r1'``.
        """
        _set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()

        with self._amp_ctx():
            loss_D_real = self.criterion_gan(self.D(real_he), target_is_real=True)
            loss_D_fake = self.criterion_gan(self.D(fake_he), target_is_real=False)
            loss_D_gan = (loss_D_real + loss_D_fake) * 0.5

        # R1 penalty is computed OUTSIDE _amp_ctx so that autograd.grad runs in
        # float32.  create_graph=True is required so backward() can differentiate
        # through the penalty into D's parameters.
        lambda_r1: float = getattr(self.cfg.losses, "lambda_r1", 0.0)
        if lambda_r1 > 0.0:
            loss_D_r1 = self._compute_r1_penalty(real_he) * lambda_r1
            loss_D = loss_D_gan + loss_D_r1
        else:
            loss_D_r1 = torch.zeros((), device=real_he.device)
            loss_D = loss_D_gan

        self._backward(loss_D, self.optimizer_D, list(self.D.parameters()))

        return {
            "loss_D":      loss_D.item(),
            "loss_D_real": loss_D_real.item(),
            "loss_D_fake": loss_D_fake.item(),
            "loss_D_r1":   loss_D_r1.item(),
        }

    # ------------------------------------------------------------------
    # G update
    # ------------------------------------------------------------------

    def update_G(
        self,
        tpaf: Tensor,
        mask: Tensor,
        real_he: Tensor,
        fake_he: Tensor,
        gen_input: Tensor,
        epoch: int,
    ) -> Dict[str, float]:
        """Generator gradient step (all G-side losses).

        Loss composition:

        +-------------------+----------------------------------------------+
        | Loss              | Active epochs                                |
        +===================+==============================================+
        | ``L_adv``         | always                                       |
        +-------------------+----------------------------------------------+
        | ``L_SA-PatchNCE`` | always                                       |
        +-------------------+----------------------------------------------+
        | ``L_struct``      | epoch ≥ struct_warmup_epochs (ramps linearly)|
        +-------------------+----------------------------------------------+
        | ``L_idt``         | always (if lambda_idt > 0)                   |
        +-------------------+----------------------------------------------+

        The discriminator is frozen (``requires_grad=False``) during this step.

        Args:
            tpaf: TPAF patches, ``(B, 1, H, W)`` in ``[0, 1]``.
            mask: Nuclear mask, ``(B, 1, H, W)`` in ``[0, 1]``.
            real_he: Real H&E, ``(B, 3, H, W)`` in ``[-1, 1]``.
            fake_he: Generated H&E from the *current* forward pass with its
                     gradient graph still intact.
            gen_input: ``cat([tpaf, mask], dim=1)`` cached from before the D
                       update.
            epoch: Current epoch for warm-up scheduling.

        Returns:
            Dict of scalar loss values for logging.
        """
        _set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()

        with self._amp_ctx():
            # ── 1. Adversarial (fool D) ────────────────────────────────────
            loss_adv = (
                self.criterion_gan(self.D(fake_he), target_is_real=True)
                * self.cfg.losses.lambda_adv
            )

            # ── 2. SA-PatchNCE ─────────────────────────────────────────────
            # Source features: encode the TPAF + mask input.
            feat_src = self.G.get_encoder_features(gen_input, self.layer_indices)

            # Generated features: encode fake H&E as [luminance, same mask].
            # Appending the same nuclear mask provides the encoder with identical
            # structural conditioning in both domains, so features are comparable.
            fake_gray = _rgb_to_gray(fake_he)                          # (B, 1, H, W)
            enc_gen = torch.cat([fake_gray, mask], dim=1)              # (B, 2, H, W)
            feat_gen = self.G.get_encoder_features(enc_gen, self.layer_indices)

            loss_nce = (
                self.criterion_patchnce(feat_src, feat_gen, mask)
                * self.cfg.losses.lambda_patchnce
            )

            # ── 3. Structure consistency (handles warm-up internally) ──────
            loss_struct = (
                self.criterion_struct(fake_he, mask, epoch)
                * self.cfg.losses.lambda_struct
            )

            # ── 4. Identity (optional) ────────────────────────────────────
            # Input:  [real H&E grayscale, extracted nuclear mask]
            # Output: should reproduce real H&E
            # Regularises the generator's colour mapping without requiring
            # paired TPAF↔H&E data.  The nuclear mask is extracted from
            # real H&E via Macenko HED decomposition (same pipeline as
            # L_struct) so that the mask-colour association is consistent:
            # mask=1 regions correspond to actual hematoxylin in the target.
            loss_idt = torch.zeros((), device=self.device)
            if self.cfg.losses.lambda_idt > 0.0:
                real_gray = _rgb_to_gray(real_he)                      # (B, 1, H, W)
                with torch.no_grad():
                    real_hem_mask = self.criterion_struct.extract_hem_mask(real_he)
                idt_input = torch.cat([real_gray, real_hem_mask], dim=1)  # (B, 2, H, W)
                idt_out = self.G(idt_input)                            # (B, 3, H, W)
                loss_idt = F.l1_loss(idt_out, real_he) * self.cfg.losses.lambda_idt

            loss_G = loss_adv + loss_nce + loss_struct + loss_idt

        self._backward(loss_G, self.optimizer_G, self._g_params)
        _set_requires_grad(self.D, True)

        return {
            "loss_G": loss_G.item(),
            "loss_adv_G": loss_adv.item(),
            "loss_nce": loss_nce.item(),
            "loss_struct": loss_struct.item(),
            "loss_idt": loss_idt.item(),
        }

    # ------------------------------------------------------------------
    # AMP + backward helper
    # ------------------------------------------------------------------

    def _amp_ctx(self):
        """Return ``torch.cuda.amp.autocast()`` or a null context."""
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def _backward(
        self,
        loss: Tensor,
        optimizer: torch.optim.Optimizer,
        params: Optional[List[nn.Parameter]] = None,
    ) -> None:
        """Backward + optional grad clipping + optimizer step.

        Handles AMP scaling transparently.

        Args:
            loss: Scalar loss tensor.
            optimizer: Optimizer to step.
            params: Parameter list for gradient norm clipping.
                    Ignored when ``grad_clip == 0``.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0 and params:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0 and params:
                nn.utils.clip_grad_norm_(params, self.grad_clip)
            optimizer.step()

    def _compute_r1_penalty(self, real_he: Tensor) -> Tensor:
        """R1 gradient penalty: ``0.5 * E[mean(||∇_x D(x)||^2)]`` on real images.

        Prevents discriminator overconfidence by penalising large gradients
        w.r.t. real inputs.  Must be called **outside** any AMP autocast
        context so that ``torch.autograd.grad`` operates in float32.

        Uses **mean** (not sum) over spatial and channel dimensions so that
        the penalty magnitude is independent of input resolution.  Without
        this, 512×512 inputs produce 4× the penalty of 256×256 for identical
        gradient magnitudes, making ``lambda_r1`` resolution-dependent.

        Args:
            real_he: Real H&E batch ``(B, 3, H, W)`` in ``[-1, 1]``.

        Returns:
            Scalar penalty tensor attached to D's computation graph.
        """
        x = real_he.detach().float().requires_grad_(True)
        with torch.enable_grad():
            d_out = self.D(x)
            grads = torch.autograd.grad(
                outputs=d_out.sum(),
                inputs=x,
                create_graph=True,
            )[0]
        return 0.5 * grads.pow(2).mean()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, tag: str = "latest") -> None:
        """Save a training checkpoint to ``{ckpt_dir}/{tag}.pth``.

        Checkpoint format (from CLAUDE.md)::

            {
                'epoch':               int,
                'G_state_dict':        ...,
                'D_state_dict':        ...,
                'optimizer_G':         ...,
                'optimizer_D':         ...,
                'patchnce_state_dict': ...,
                'global_step':         int,
                'config':              dict,
                'scaler':              ...  (AMP only),
            }

        Args:
            epoch: Epoch index to record in the file.
            tag: Filename stem (e.g. ``'latest'``, ``'epoch_0050'``).
        """
        ckpt: dict = {
            "epoch": epoch,
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "patchnce_state_dict": self.criterion_patchnce.state_dict(),
            "global_step": self.global_step,
            "config": self.cfg.to_dict(),
        }
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()

        path = self.ckpt_dir / f"{tag}.pth"
        torch.save(ckpt, path)
        logger.info("Checkpoint → %s  (epoch %d)", path.name, epoch)

    def load_checkpoint(self, path: str) -> None:
        """Restore all training state from a checkpoint file.

        Resumes from the epoch immediately after the saved one.  LR
        schedulers are fast-forwarded to match.

        Args:
            path: Path to a ``.pth`` checkpoint written by
                :meth:`save_checkpoint`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.G.load_state_dict(ckpt["G_state_dict"])
        self.D.load_state_dict(ckpt["D_state_dict"])
        self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
        self.optimizer_D.load_state_dict(ckpt["optimizer_D"])

        if "patchnce_state_dict" in ckpt:
            self.criterion_patchnce.load_state_dict(ckpt["patchnce_state_dict"])

        if "scaler" in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)

        # Fast-forward LR schedulers to match the resumed epoch.
        # We step without a prior optimizer.step(), which is intentional here
        # (we are replaying history, not training).  Suppress the PyTorch
        # "step before optimizer.step" UserWarning that would otherwise fire.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(self.start_epoch):
                self.scheduler_G.step()
                self.scheduler_D.step()

        logger.info(
            "Resumed from '%s'  (epoch %d → continuing from %d, step %d)",
            ckpt_path.name,
            ckpt["epoch"],
            self.start_epoch,
            self.global_step,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        losses_dict: Dict[str, float],
        epoch: int,
        step: int,
    ) -> None:
        """Write scalar metrics to TensorBoard and/or wandb.

        Args:
            losses_dict: Loss name → value mapping.
            epoch: Current epoch (used by wandb).
            step: Global iteration index (TensorBoard x-axis).
        """
        for k, v in losses_dict.items():
            self.tb_writer.add_scalar(f"train/{k}", v, step)

        # Also log the extracted hematoxylin mask coverage (structure loss debug)
        hem_mask = self.criterion_struct.get_last_hem_mask()
        if hem_mask is not None:
            self.tb_writer.add_scalar(
                "train/hem_mask_coverage",
                hem_mask.mean().item(),
                step,
            )

        try:
            import wandb  # noqa: PLC0415

            if wandb.run is not None:
                wandb.log({"step": step, **{f"train/{k}": v for k, v in losses_dict.items()}})
        except ImportError:
            pass

    def log_images(
        self,
        tpaf: Tensor,
        mask: Tensor,
        fake_he: Tensor,
        real_he: Tensor,
        epoch: int,
    ) -> None:
        """Log a 4-column image grid to TensorBoard and/or wandb.

        Columns: TPAF (grey→RGB) | nuclear mask | generated H&E | real H&E.
        Only the first sample in the batch is shown.

        Args:
            tpaf:    ``(B, 1, H, W)`` in ``[0, 1]``.
            mask:    ``(B, 1, H, W)`` in ``[0, 1]``.
            fake_he: ``(B, 3, H, W)`` in ``[-1, 1]``.
            real_he: ``(B, 3, H, W)`` in ``[-1, 1]``.
            epoch: Current epoch (for wandb caption).
        """
        tpaf_rgb = tpaf[0:1].expand(-1, 3, -1, -1)          # grey→RGB  [0,1]
        mask_rgb = mask[0:1].expand(-1, 3, -1, -1)           # grey→RGB  [0,1]
        fake_rgb = fake_he[0:1] * 0.5 + 0.5                  # [-1,1]→[0,1]
        real_rgb = real_he[0:1] * 0.5 + 0.5                  # [-1,1]→[0,1]

        grid = make_grid(
            torch.cat([tpaf_rgb, mask_rgb, fake_rgb, real_rgb], dim=0),
            nrow=4, normalize=False, padding=2,
        )  # (3, H, W*4+padding)

        self.tb_writer.add_image("samples/tpaf_mask_fake_real", grid, self.global_step)

        # Also log the latest extracted hematoxylin mask for structure loss debugging
        hem_mask = self.criterion_struct.get_last_hem_mask()
        if hem_mask is not None:
            hem_rgb = hem_mask[0:1].expand(-1, 3, -1, -1)
            self.tb_writer.add_image("debug/hem_extracted", hem_rgb[0], self.global_step)

        try:
            import wandb  # noqa: PLC0415

            if wandb.run is not None:
                wandb.log({
                    "samples": wandb.Image(
                        grid.permute(1, 2, 0).cpu().numpy(),
                        caption=f"[epoch {epoch}] TPAF | mask | generated H&E | real H&E",
                    ),
                    "epoch": epoch,
                })
        except ImportError:
            pass
