"""Region-Aware PatchNCE loss (SA-PatchNCE) — the core innovation of SA-CUT.

Background: why standard PatchNCE fails for TPAF → H&E
---------------------------------------------------------
CUT's PatchNCE loss encourages mutual information between corresponding patches
in the source (TPAF) and generated (H&E) images.  For each sampled spatial
location *i*, the source patch at *i* is the **positive key** and source
patches at all *other* locations are **negatives**.

The problem: nothing in this formulation distinguishes *nuclear* patches from
*stromal* patches.  The discriminator can correctly fool the PatchNCE loss by
mapping **dark nuclear voids** in TPAF → **bright eosin-pink** in H&E, because:

  1. The dark-void source feature (nuclear) at location *i* is the positive for
     the generated feature at location *i* — regardless of colour.
  2. The negatives (other source patches) are a *mixture* of nuclear and stromal
     features.  If the generator consistently inverts the two regions, every
     "positive" pair is still the closest pair after the MLP projection, because
     the *ordering* of patches is preserved.

SA-PatchNCE solution: cross-region negatives
---------------------------------------------
By partitioning sampled locations into Ω_nuc (mask > 0.5) and Ω_stroma
(mask ≤ 0.5) and then using **all** sampled source patches as negatives for
*both* region queues, we make cross-region pairs explicitly negative:

  • A generated nuclear patch (dark void → should become purple) that instead
    looks pink-stromal will be *more similar* to stromal source keys than to the
    nuclear source key at the same location.  The InfoNCE loss penalises this
    because the positive (source nuclear, same location) now competes against
    many stromal keys that the incorrectly-coloured generated patch resembles.

  • Symmetrically, a generated stromal patch coloured as purple nuclei will be
    penalised because it resembles nuclear source keys, which are negatives for
    the stromal query.

The result is an explicit gradient signal that enforces semantic alignment:

    dark void in TPAF  ──→  hematoxylin (purple)  in H&E
    bright NADH/FAD    ──→  eosin (pink/red)       in H&E

Algorithm summary
-----------------
For each of L encoder layers::

    1. Downsample mask  → (B, 1, H_l, W_l), nearest-neighbor
    2. Flatten spatial  → (B, C_l, N_l),   N_l = H_l × W_l
    3. Sample n_patches random locations (shared for source & generated)
    4. Project through shared 2-layer MLP → L2-normalise → (n_patches, nc=256)
    5. Partition:  nuc_idx  = {i : mask[i] > 0.5}
                   stroma_idx = {i : mask[i] ≤ 0.5}
    6. Nuclear InfoNCE:
         logits = q_nuc @ K_all.T / τ     shape (|nuc|, n_patches)
         labels = nuc_idx                  (positive column = same spatial loc)
    7. Stroma InfoNCE:  analogous
    8. layer_loss = 0.5 * (nce_nuc + nce_stroma)
    9. Fallback to standard PatchNCE if |nuc| < min_region_patches  OR
                                        |stroma| < min_region_patches
   10. Final loss = mean over L layers and B images

Notes
-----
* MLP heads are built **lazily** on the first forward call so that the caller
  does not need to know feature channel sizes at construction time.
* When ``mask=None``, an all-zero mask is synthesised, which triggers the
  fallback in every layer — reproducing standard PatchNCE for the CUT baseline
  ablation.
* The MLP heads are registered in an ``nn.ModuleList`` and travel with the
  module via ``.to(device)``.  They should be included in the *generator*
  optimiser parameter group (not the discriminator's).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# MLP projector head
# ---------------------------------------------------------------------------


class PatchMLP(nn.Module):
    """Two-layer MLP projector that maps raw patch features to a unit-sphere.

    After projection, dot products between feature vectors are cosine
    similarities, enabling stable InfoNCE computation with a temperature τ.

    Args:
        in_channels: Feature dimensionality at this encoder scale (C_l).
        nc: Output embedding dimensionality. Default 256.
    """

    def __init__(self, in_channels: int, nc: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, nc),
            nn.ReLU(inplace=True),
            nn.Linear(nc, nc),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalise a batch of patch descriptors.

        Args:
            x: Raw patch features, shape ``(N, C)``.

        Returns:
            L2-normalised embeddings, shape ``(N, nc)``.
        """
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Region-aware PatchNCE
# ---------------------------------------------------------------------------


class RegionAwarePatchNCE(nn.Module):
    """SA-PatchNCE: mask-guided region-partitioned contrastive patch loss.

    Extends CUT's PatchNCE with nuclear/stromal region partitioning to enforce
    biologically correct semantic mapping and prevent semantic inversion
    artifacts.  See module docstring for the full theoretical motivation.

    Args:
        n_layers: Number of encoder feature scales (length of the feature
            lists passed to ``forward``). Default 5.
        n_patches: Spatial locations sampled per layer per image. Default 256.
        nc: MLP output embedding dimensionality. Default 256.
        tau: InfoNCE temperature. Default 0.07.
        min_region_patches: Minimum patch count required in *each* region to
            activate region-aware mode.  Layers with too few patches in either
            region fall back to standard PatchNCE. Default 4.
    """

    def __init__(
        self,
        n_layers: int = 5,
        n_patches: int = 256,
        nc: int = 256,
        tau: float = 0.07,
        min_region_patches: int = 4,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_patches = n_patches
        self.nc = nc
        self.tau = tau
        self.min_region_patches = min_region_patches
        # Populated lazily on the first forward call.
        self.mlp_heads: nn.ModuleList = nn.ModuleList()

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _build_mlps(self, feat_list: List[Tensor]) -> None:
        """Construct one PatchMLP per encoder scale on the first call.

        Args:
            feat_list: Source feature list; used only to read channel counts.
        """
        if len(self.mlp_heads) > 0:
            return  # already initialised

        for feat in feat_list:
            c = feat.shape[1]
            self.mlp_heads.append(PatchMLP(c, self.nc))

        # Bring newly added submodules to the same device as the features.
        device = feat_list[0].device
        self.mlp_heads.to(device)

    # ------------------------------------------------------------------
    # Per-layer loss computation
    # ------------------------------------------------------------------

    def _nce_loss_for_layer(
        self,
        feat_s: Tensor,
        feat_g: Tensor,
        mask_l: Tensor,
        mlp: PatchMLP,
    ) -> Tensor:
        """Compute region-aware InfoNCE for one encoder scale.

        The MLP head is **shared** between source and generated features:
        source → keys (K), generated → queries (Q).  Positive pairs are at the
        same sampled spatial location; negatives are all *other* sampled source
        locations, including those from the opposite region.

        Args:
            feat_s: Source features, shape ``(B, C, H_l, W_l)``.
            feat_g: Generated features, shape ``(B, C, H_l, W_l)``.
            mask_l: Nuclear mask downsampled to ``(B, 1, H_l, W_l)``,
                values in ``[0, 1]``.
            mlp: Shared MLP projector head for this scale.

        Returns:
            Scalar mean loss across the batch.
        """
        B, C, H, W = feat_s.shape
        N = H * W
        n_sample = min(self.n_patches, N)

        # Flatten spatial dimension: (B, C, N)
        fs_flat = feat_s.flatten(2)
        fg_flat = feat_g.flatten(2)
        mask_flat = mask_l.squeeze(1).flatten(1)  # (B, N)

        batch_losses: List[Tensor] = []

        for b in range(B):
            # ── Sample n_patches random spatial locations ───────────────
            perm = torch.randperm(N, device=feat_s.device)[:n_sample]

            # Extract and transpose to (n_sample, C)
            fs_b = fs_flat[b, :, perm].T
            fg_b = fg_flat[b, :, perm].T
            mask_b = mask_flat[b, perm]  # (n_sample,)

            # ── Project + L2-normalise → (n_sample, nc) ─────────────────
            ks = mlp(fs_b)  # source keys
            qs = mlp(fg_b)  # generated queries

            # ── Partition by region ──────────────────────────────────────
            nuc_idx = (mask_b > 0.5).nonzero(as_tuple=True)[0]    # (m_nuc,)
            stroma_idx = (mask_b <= 0.5).nonzero(as_tuple=True)[0]  # (m_stroma,)

            use_region = (
                nuc_idx.numel() >= self.min_region_patches
                and stroma_idx.numel() >= self.min_region_patches
            )

            if use_region:
                # ── Region-aware InfoNCE ─────────────────────────────────
                # Nuclear queries against ALL source keys.
                # labels[i] = nuc_idx[i]: the positive key for nuclear query i
                # is the source feature at the same spatial location.
                logits_nuc = qs[nuc_idx] @ ks.T / self.tau  # (m_nuc, n_sample)
                loss_nuc = F.cross_entropy(logits_nuc, nuc_idx)

                logits_stroma = qs[stroma_idx] @ ks.T / self.tau  # (m_stroma, n_sample)
                loss_stroma = F.cross_entropy(logits_stroma, stroma_idx)

                batch_losses.append((loss_nuc + loss_stroma) * 0.5)

            else:
                # ── Fallback: standard PatchNCE (diagonal labels) ────────
                # Triggered when a region is empty or too small.  Preserves
                # gradient flow without region partitioning.
                labels = torch.arange(n_sample, device=feat_s.device)
                batch_losses.append(F.cross_entropy(qs @ ks.T / self.tau, labels))

        return torch.stack(batch_losses).mean()

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feat_source: List[Tensor],
        feat_generated: List[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute SA-PatchNCE loss across all L encoder layers.

        Args:
            feat_source: L feature maps from ``G.get_encoder_features(tpaf_input)``.
                Each tensor has shape ``(B, C_l, H_l, W_l)``.
            feat_generated: L feature maps from ``G.get_encoder_features(generated_he)``.
                Same shapes as ``feat_source``.
            mask: Nuclear segmentation mask of shape ``(B, 1, H, W)``, float32
                in ``[0, 1]``.  Pass ``None`` to reproduce standard PatchNCE
                (CUT baseline ablation — region partitioning is disabled).

        Returns:
            Scalar loss tensor with gradient attached.

        Raises:
            ValueError: If the length of either feature list does not equal
                ``self.n_layers``.
        """
        if len(feat_source) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} source feature maps, "
                f"got {len(feat_source)}."
            )
        if len(feat_generated) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} generated feature maps, "
                f"got {len(feat_generated)}."
            )

        # Build MLP heads from observed feature channel sizes (first call only).
        self._build_mlps(feat_source)

        layer_losses: List[Tensor] = []

        for fs, fg, mlp in zip(feat_source, feat_generated, self.mlp_heads):
            _, _, H_l, W_l = fs.shape

            if mask is not None:
                mask_l = F.interpolate(mask, size=(H_l, W_l), mode="nearest")
            else:
                # All-zero mask → nuc_idx always empty → standard PatchNCE fallback.
                mask_l = torch.zeros(
                    fs.shape[0], 1, H_l, W_l,
                    device=fs.device,
                    dtype=torch.float32,
                )

            layer_losses.append(self._nce_loss_for_layer(fs, fg, mask_l, mlp))

        return torch.stack(layer_losses).mean()
