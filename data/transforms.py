"""Domain-specific augmentation transforms for SA-CUT unpaired training.

Design principles
-----------------
* **Geometric transforms**: applied to TPAF + mask *jointly* (shared random state)
  and to H&E *independently* (separate random state).  The two domains are unpaired,
  so no geometric correspondence between them should be assumed.

* **Color jitter**: H&E only.  TPAF is single-channel greyscale fluorescence; colour
  jitter is biologically meaningless and would destroy intensity semantics.  H&E
  jitter is deliberately mild (brightness/contrast ±0.1, saturation ±0.05, hue ±0.02)
  to avoid over-distorting the haematoxylin/eosin chromatic relationship that the
  discriminator must learn to evaluate.

* **Random state**: flip/rotation decisions use ``torch.rand`` / ``torch.randint``
  so that seeding with ``torch.manual_seed`` fully controls reproducibility.
  ``torchvision.transforms.ColorJitter`` uses Python's ``random`` module internally;
  call ``random.seed`` alongside ``torch.manual_seed`` for exact reproducibility.

* **Tensor range**: all transforms expect and return float32 tensors in ``[0, 1]``.
  H&E scaling to ``[-1, 1]`` is NOT done here; it is applied by
  :class:`data.dataset.UnpairedTPAFDataset` *after* the transform so that
  ``ColorJitter`` always operates on the natural ``[0, 1]`` range.

Typical usage::

    from data.transforms import UnpairedTransform
    from data.dataset import UnpairedTPAFDataset

    transform = UnpairedTransform(p_hflip=0.5, p_vflip=0.5, p_rot90=0.5)
    dataset = UnpairedTPAFDataset(
        tpaf_dir="data/patches/tpaf",
        he_dir="data/patches/he",
        transform=transform,
    )
"""

from typing import Optional

import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.transforms import ColorJitter


# ---------------------------------------------------------------------------
# Geometric: paired (TPAF + mask share state)
# ---------------------------------------------------------------------------


class PairedFlipRotate:
    """Apply random flip and 90° rotation to a (image, mask) pair with a shared state.

    Both tensors receive **identical** geometric transforms.  Spatial correspondence
    between a TPAF patch and its nuclear segmentation mask is preserved.

    Args:
        p_hflip: Probability of a horizontal flip.
        p_vflip: Probability of a vertical flip.
        p_rot90: Probability of applying a random 90° rotation (k sampled from {1,2,3}).
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
    ) -> None:
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(
        self,
        image: Tensor,
        mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Apply the same random transform to *image* and *mask*.

        Args:
            image: Float32 tensor of shape ``(C, H, W)``.
            mask:  Float32 tensor of shape ``(1, H, W)``, or ``None`` (returned
                   unchanged).

        Returns:
            ``(image, mask)`` after identical geometric augmentation.
        """
        if torch.rand(1).item() < self.p_hflip:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if torch.rand(1).item() < self.p_vflip:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        if torch.rand(1).item() < self.p_rot90:
            k = int(torch.randint(low=1, high=4, size=(1,)).item())
            image = torch.rot90(image, k, dims=[-2, -1])
            if mask is not None:
                mask = torch.rot90(mask, k, dims=[-2, -1])

        return image, mask

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p_hflip={self.p_hflip}, p_vflip={self.p_vflip}, p_rot90={self.p_rot90})"
        )


# ---------------------------------------------------------------------------
# Geometric: independent (H&E uses its own random state)
# ---------------------------------------------------------------------------


class IndependentFlipRotate:
    """Apply random flip and 90° rotation to a single tensor independently.

    Generates its own random numbers, so it is statistically independent from any
    concurrent :class:`PairedFlipRotate` call.  Used for H&E images in the unpaired
    training loop — augmenting H&E with independent geometry prevents the model from
    learning a spurious geometric correspondence between the two unpaired domains.

    Args:
        p_hflip: Probability of a horizontal flip.
        p_vflip: Probability of a vertical flip.
        p_rot90: Probability of applying a random 90° rotation (k sampled from {1,2,3}).
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
    ) -> None:
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, image: Tensor) -> Tensor:
        """Apply random geometric augmentation.

        Args:
            image: Float32 tensor of shape ``(C, H, W)``.

        Returns:
            Augmented tensor of the same shape.
        """
        if torch.rand(1).item() < self.p_hflip:
            image = TF.hflip(image)

        if torch.rand(1).item() < self.p_vflip:
            image = TF.vflip(image)

        if torch.rand(1).item() < self.p_rot90:
            k = int(torch.randint(low=1, high=4, size=(1,)).item())
            image = torch.rot90(image, k, dims=[-2, -1])

        return image

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p_hflip={self.p_hflip}, p_vflip={self.p_vflip}, p_rot90={self.p_rot90})"
        )


# ---------------------------------------------------------------------------
# Colour: H&E only
# ---------------------------------------------------------------------------


class HEColorJitter:
    """Mild colour jitter for H&E patches.

    Wraps :class:`torchvision.transforms.ColorJitter` with conservative defaults
    suited to H&E virtual staining.  The jitter is intentionally subtle:

    * **brightness / contrast** ±0.10 — accommodates slide-to-slide staining
      intensity variation without washing out hematoxylin/eosin contrast.
    * **saturation** ±0.05 — small, preserves the characteristic purple-pink
      palette that the discriminator and downstream pathologist depend on.
    * **hue** ±0.02 — very small; aggressive hue shifts would rotate
      hematoxylin (purple) toward eosin (pink) and confuse the model.

    This transform operates on **float32 tensors in [0, 1]** (torchvision
    ``ColorJitter`` supports tensor inputs from v0.8+).  Do NOT apply after
    scaling H&E to ``[-1, 1]``.

    Args:
        brightness: Max ± brightness offset as a fraction of the original.
        contrast:   Max ± contrast scale factor offset.
        saturation: Max ± saturation scale factor offset.
        hue:        Max ± hue shift in ``[0, 0.5]`` cycles.
    """

    def __init__(
        self,
        brightness: float = 0.10,
        contrast: float = 0.10,
        saturation: float = 0.05,
        hue: float = 0.02,
    ) -> None:
        self._jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image: Tensor) -> Tensor:
        """Apply colour jitter to an H&E tensor in ``[0, 1]``.

        Args:
            image: Float32 tensor of shape ``(3, H, W)`` with values in ``[0, 1]``.

        Returns:
            Colour-jittered tensor, same shape and range.
        """
        return self._jitter(image)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(jitter={self._jitter})"


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------


class UnpairedTransform:
    """Full augmentation pipeline for :class:`data.dataset.UnpairedTPAFDataset`.

    Applies transforms to a sample dict according to the unpaired-domain rules:

    +-----------+-----------------------+---------------------+
    | Modality  | Geometric transform   | Colour transform    |
    +===========+=======================+=====================+
    | TPAF      | Shared with mask      | None (greyscale)    |
    +-----------+-----------------------+---------------------+
    | Mask      | Shared with TPAF      | None                |
    +-----------+-----------------------+---------------------+
    | H&E       | Independent           | Mild colour jitter  |
    +-----------+-----------------------+---------------------+

    **Input contract**: the sample dict must contain ``'tpaf'``, ``'he'``, and
    ``'mask'`` as float32 tensors in ``[0, 1]``.  Extra keys (e.g. ``'tpaf_path'``)
    are passed through unchanged.

    **Output contract**: same dict with augmented tensors, still in ``[0, 1]``.
    H&E is NOT scaled to ``[-1, 1]`` here; :class:`data.dataset.UnpairedTPAFDataset`
    applies that conversion **after** calling this transform, ensuring
    :class:`HEColorJitter` always receives a ``[0, 1]`` image.

    Args:
        p_hflip:         Probability of horizontal flip for each modality.
        p_vflip:         Probability of vertical flip for each modality.
        p_rot90:         Probability of a random 90° rotation for each modality.
        he_color_jitter: Whether to apply colour jitter to H&E.  Set ``False``
                         for validation / inference runs.
        brightness:      Colour jitter brightness magnitude.
        contrast:        Colour jitter contrast magnitude.
        saturation:      Colour jitter saturation magnitude.
        hue:             Colour jitter hue shift magnitude.

    Example::

        transform = UnpairedTransform(
            p_hflip=0.5, p_vflip=0.5, p_rot90=0.5,
            he_color_jitter=True,
        )
        # Disable all augmentation for validation:
        val_transform = UnpairedTransform(
            p_hflip=0.0, p_vflip=0.0, p_rot90=0.0, he_color_jitter=False
        )
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        he_color_jitter: bool = True,
        brightness: float = 0.10,
        contrast: float = 0.10,
        saturation: float = 0.05,
        hue: float = 0.02,
    ) -> None:
        self._tpaf_mask_aug = PairedFlipRotate(p_hflip, p_vflip, p_rot90)
        self._he_geom_aug = IndependentFlipRotate(p_hflip, p_vflip, p_rot90)
        self._he_color_aug: Optional[HEColorJitter] = (
            HEColorJitter(brightness, contrast, saturation, hue)
            if he_color_jitter
            else None
        )

    def __call__(self, sample: dict) -> dict:
        """Augment a sample dict from :class:`data.dataset.UnpairedTPAFDataset`.

        Args:
            sample: Dict with at minimum keys ``'tpaf'``, ``'he'``, ``'mask'``
                    as float32 tensors in ``[0, 1]``.

        Returns:
            Augmented sample dict; tensor values remain in ``[0, 1]``.
        """
        tpaf: Tensor = sample["tpaf"]
        he: Tensor = sample["he"]
        mask: Optional[Tensor] = sample.get("mask")

        # --- Geometric: TPAF + mask share the same random draw ---
        tpaf, mask = self._tpaf_mask_aug(tpaf, mask)

        # --- Geometric: H&E draws independently ---
        he = self._he_geom_aug(he)

        # --- Colour: H&E only ---
        if self._he_color_aug is not None:
            he = self._he_color_aug(he)

        return {**sample, "tpaf": tpaf, "he": he, "mask": mask}

    def __repr__(self) -> str:
        lines = [
            f"{type(self).__name__}(",
            f"  tpaf_mask_aug = {self._tpaf_mask_aug},",
            f"  he_geom_aug   = {self._he_geom_aug},",
            f"  he_color_aug  = {self._he_color_aug},",
            ")",
        ]
        return "\n".join(lines)
