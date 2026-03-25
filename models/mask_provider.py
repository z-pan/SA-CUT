"""Unified interface for nuclear segmentation mask acquisition.

The SA-CUT pipeline requires a float32 nuclear mask (shape ``[1, H, W]``,
values in ``[0, 1]``) for every TPAF patch.  Two concrete providers cover the
two operational phases:

* :class:`PrecomputedMaskProvider` — loads pre-saved ``.npy`` mask files.
  Zero inference overhead; preferred during training.
* :class:`CellposeSAMMaskProvider` — runs frozen Cellpose-SAM on-the-fly.
  Use for test-time inference on unseen TPAF images.

Use :func:`build_mask_provider` to construct the correct provider from a YAML
config namespace.

Critical constraints
--------------------
* Neither provider is an :class:`~torch.nn.Module`.  They must **never** be
  passed to an optimizer.
* The Cellpose network is frozen at construction: ``eval()`` mode and
  ``requires_grad=False`` on all parameters.
* All inference is wrapped in :func:`torch.no_grad`.
* Masks are always returned as ``torch.float32`` tensors in ``[0, 1]``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_mask_stats(mask: np.ndarray, instance_labels: Optional[np.ndarray] = None) -> dict:
    """Compute QC statistics for a single binary/soft mask.

    Args:
        mask: Binary mask array of shape ``(H, W)`` with values in ``[0, 1]``.
        instance_labels: Optional instance-label array of shape ``(H, W)``
            returned by Cellpose.  If supplied, ``num_nuclei`` is the count of
            unique non-zero labels; otherwise it is estimated via connected
            components.

    Returns:
        Dict with keys ``coverage_ratio`` (float), ``num_nuclei`` (int),
        ``mean_nucleus_area`` (float).
    """
    binary = (mask > 0.5).astype(np.uint8)
    total_pixels = binary.size
    coverage = float(binary.sum()) / total_pixels

    if instance_labels is not None:
        unique_labels = np.unique(instance_labels)
        num_nuclei = int((unique_labels > 0).sum())
    else:
        try:
            from scipy import ndimage

            _, num_nuclei = ndimage.label(binary)
        except ImportError:
            num_nuclei = -1  # scipy not available; mark as unknown

    mean_area = (float(binary.sum()) / num_nuclei) if num_nuclei > 0 else 0.0

    return {
        "coverage_ratio": coverage,
        "num_nuclei": num_nuclei,
        "mean_nucleus_area": mean_area,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MaskProvider(ABC):
    """Abstract base class for nuclear mask acquisition.

    All concrete subclasses must return a ``torch.float32`` tensor of shape
    ``(1, H, W)`` with values in ``[0, 1]``.
    """

    @abstractmethod
    def get_mask(
        self,
        source: Union[str, Tensor, np.ndarray],
    ) -> Tensor:
        """Return a nuclear segmentation mask.

        Args:
            source: For :class:`PrecomputedMaskProvider` this is a path string
                to the corresponding TPAF patch file.  For
                :class:`CellposeSAMMaskProvider` this is the TPAF image as a
                ``Tensor[1, H, W]`` or ``ndarray[H, W]``.

        Returns:
            Float32 tensor of shape ``(1, H, W)`` with values in ``[0, 1]``.
        """


# ---------------------------------------------------------------------------
# Mode A — precomputed
# ---------------------------------------------------------------------------


class PrecomputedMaskProvider(MaskProvider):
    """Load pre-saved binary/soft mask ``.npy`` files from disk.

    Masks are expected to share the filename *stem* of the corresponding TPAF
    patch.  For example, TPAF patch ``patch_001.tif`` → mask
    ``<mask_dir>/patch_001.npy``.

    Args:
        mask_dir: Directory that contains the pre-generated ``.npy`` mask
            files.  The directory must exist at construction time.
        patch_size: Expected spatial size (height == width) of each mask.
            If provided, the first loaded mask is validated against this value
            and subsequent loads skip the check (cached).  Pass ``None`` to
            disable the check.
    """

    def __init__(self, mask_dir: str, patch_size: Optional[int] = None) -> None:
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"mask_dir does not exist: {self.mask_dir}. "
                "Run scripts/precompute_masks.py to generate masks first."
            )
        self.patch_size = patch_size
        self._shape_validated = False

    def get_mask(self, tpaf_path: str) -> Tensor:  # type: ignore[override]
        """Load the precomputed mask matching *tpaf_path*'s filename stem.

        Args:
            tpaf_path: Absolute or relative path to the TPAF patch file.
                Only the filename stem is used for the lookup.

        Returns:
            Float32 tensor of shape ``(1, H, W)`` with values in ``[0, 1]``.

        Raises:
            FileNotFoundError: If no matching ``.npy`` file exists in
                ``mask_dir``.
            ValueError: If the loaded mask's spatial dimensions do not match
                ``patch_size`` (checked on first call only).
        """
        stem = Path(tpaf_path).stem
        mask_path = self.mask_dir / f"{stem}.npy"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"No precomputed mask found for '{Path(tpaf_path).name}'. "
                f"Expected: {mask_path}\n"
                "Tip: run scripts/precompute_masks.py to generate all masks "
                "before training."
            )

        mask = np.load(str(mask_path)).astype(np.float32)

        # Normalize [0, 255] → [0, 1] if the array was saved as uint8-like
        if mask.max() > 1.0:
            mask = mask / 255.0
            mask = np.clip(mask, 0.0, 1.0)

        # Ensure shape is (1, H, W)
        if mask.ndim == 2:
            mask = mask[np.newaxis]  # (H, W) → (1, H, W)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            raise ValueError(
                f"Unexpected mask shape {mask.shape} in {mask_path}. "
                "Expected (H, W) or (1, H, W)."
            )

        # Validate spatial dimensions against expected patch_size (first call only)
        if not self._shape_validated and self.patch_size is not None:
            h, w = mask.shape[-2], mask.shape[-1]
            if h != self.patch_size or w != self.patch_size:
                raise ValueError(
                    f"Mask spatial dimensions ({h}×{w}) do not match "
                    f"expected patch_size={self.patch_size} for {mask_path}."
                )
            self._shape_validated = True

        return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Mode B — Cellpose-SAM (on-the-fly inference)
# ---------------------------------------------------------------------------


class CellposeSAMMaskProvider(MaskProvider):
    """Run a frozen Cellpose-SAM model to produce nuclear masks on the fly.

    The Cellpose network is loaded once at construction, immediately put into
    ``eval()`` mode, and all parameters are frozen (``requires_grad=False``).
    Inference is always wrapped in :func:`torch.no_grad`.

    This provider has **no trainable parameters** and must never be passed to
    a PyTorch optimizer.

    Args:
        checkpoint_path: Path to the Cellpose-SAM checkpoint (a ``.pth``
            file fine-tuned on TPAF data).
        model_type: Cellpose model type string.  Must match the checkpoint
            architecture: ``"cpsam"`` for Cellpose-SAM fine-tuned models
            (filenames start with ``cpsam_*``); ``"cyto3"`` / ``"nuclei"``
            for classic Cellpose U-Net models.  Mixing architectures causes
            a weight shape-mismatch error at load time.
        diameter: Expected nucleus diameter in pixels.  Pass ``None`` to
            auto-estimate per image (slower).
        gpu: Use CUDA if available.
        flow_threshold: Cellpose flow threshold for mask filtering.
        cellprob_threshold: Cellpose cell-probability threshold.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "cpsam",
        diameter: Optional[int] = 30,
        gpu: bool = True,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
    ) -> None:
        try:
            from cellpose import models as _cellpose_models
        except ImportError as exc:
            raise ImportError(
                "cellpose is required for CellposeSAMMaskProvider. "
                "Install it with: pip install cellpose"
            ) from exc

        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

        logger.info("Loading Cellpose-SAM from %s (model_type=%s)", checkpoint_path, model_type)
        self._model = _cellpose_models.CellposeModel(
            pretrained_model=checkpoint_path,
            model_type=model_type,
            gpu=gpu,
        )

        # Freeze — eval mode + disable gradients on the underlying torch net
        self._freeze()
        logger.info("Cellpose-SAM frozen (eval mode, requires_grad=False on all params).")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        """Put the Cellpose network in eval mode and disable all gradients."""
        # Cellpose stores its PyTorch network as `.net` (CellposeModel) or
        # `.net` inside `.net` for older versions.  We walk every attribute
        # that exposes `.parameters()`.
        net = getattr(self._model, "net", None)
        if net is not None and hasattr(net, "parameters"):
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)
        # Also freeze any secondary network (e.g. size model)
        net2 = getattr(self._model, "net2", None)
        if net2 is not None and hasattr(net2, "parameters"):
            net2.eval()
            for param in net2.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_mask(  # type: ignore[override]
        self,
        tpaf_image: Union[Tensor, np.ndarray],
        return_flows: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, list]]:
        """Run Cellpose-SAM inference and return a binary nuclear mask.

        Args:
            tpaf_image: TPAF patch as a ``Tensor[1, H, W]`` (float32, ``[0, 1]``)
                or a ``ndarray[H, W]`` (float32 or uint16).
            return_flows: If ``True``, also return the raw Cellpose flow list
                for downstream uncertainty weighting.

        Returns:
            If ``return_flows=False``: float32 ``Tensor[1, H, W]`` in ``[0, 1]``.
            If ``return_flows=True``: tuple of (mask tensor, cellpose flows list).
        """
        # --- Prepare numpy HxW input for Cellpose ---------------------------
        if isinstance(tpaf_image, Tensor):
            img_np = tpaf_image.detach().cpu().numpy()
        else:
            img_np = np.asarray(tpaf_image, dtype=np.float32)

        if img_np.ndim == 3:
            # (1, H, W) → (H, W)
            img_np = img_np.squeeze(0)
        elif img_np.ndim != 2:
            raise ValueError(
                f"Expected 2-D (H, W) or 3-D (1, H, W) image array, "
                f"got shape {img_np.shape}."
            )

        # Scale [0, 1] → [0, 255] — Cellpose's internal normalisation expects
        # a broader range; many fine-tuned models were trained on scaled inputs.
        img_np = (img_np * 255.0).astype(np.float32)

        # --- Inference (no gradient tracking) --------------------------------
        with torch.no_grad():
            masks_inst, flows, _ = self._model.eval(
                img_np,
                diameter=self.diameter,
                channels=[0, 0],  # grayscale: single-channel input
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                normalize=True,  # Cellpose internal per-image normalisation
            )

        # Instance labels → float32 binary mask (1, H, W)
        binary = (masks_inst > 0).astype(np.float32)[np.newaxis]  # (1, H, W)
        mask_tensor = torch.from_numpy(binary)

        if return_flows:
            return mask_tensor, flows
        return mask_tensor

    def parameters(self):
        """Return an empty iterator — this provider has no trainable parameters.

        Explicitly overriding to prevent accidental inclusion in an optimizer
        parameter group.
        """
        return iter([])

    def named_parameters(self, *args, **kwargs):
        """Return an empty iterator — this provider has no trainable parameters."""
        return iter([])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mask_provider(cfg) -> MaskProvider:
    """Construct the appropriate :class:`MaskProvider` from a config namespace.

    Args:
        cfg: Config object with a ``mask_provider`` sub-namespace containing:

            - ``mode`` (str): ``"precomputed"`` or ``"cellpose_sam"``.
            - ``mask_dir`` (str): used when ``mode="precomputed"``.
            - ``cellpose_checkpoint`` (str): used when ``mode="cellpose_sam"``.
            - ``cellpose_model_type`` (str)
            - ``cellpose_diameter`` (int)
            - ``cellpose_gpu`` (bool)
            - ``cellpose_flow_threshold`` (float)
            - ``cellpose_cellprob_threshold`` (float)

            Optionally ``cfg.data.patch_size`` (int) for shape validation in
            precomputed mode.

    Returns:
        A fully-initialised :class:`MaskProvider` instance.

    Raises:
        ValueError: If ``cfg.mask_provider.mode`` is not recognised.
    """
    mode: str = cfg.mask_provider.mode

    if mode == "precomputed":
        patch_size = getattr(getattr(cfg, "data", None), "patch_size", None)
        return PrecomputedMaskProvider(
            mask_dir=cfg.mask_provider.mask_dir,
            patch_size=patch_size,
        )

    if mode == "cellpose_sam":
        return CellposeSAMMaskProvider(
            checkpoint_path=cfg.mask_provider.cellpose_checkpoint,
            model_type=cfg.mask_provider.cellpose_model_type,
            diameter=cfg.mask_provider.cellpose_diameter,
            gpu=cfg.mask_provider.cellpose_gpu,
            flow_threshold=cfg.mask_provider.cellpose_flow_threshold,
            cellprob_threshold=cfg.mask_provider.cellpose_cellprob_threshold,
        )

    raise ValueError(
        f"Unknown mask_provider mode: '{mode}'. "
        "Choose from: 'precomputed', 'cellpose_sam'."
    )
