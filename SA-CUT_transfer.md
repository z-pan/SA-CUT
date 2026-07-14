# SA-CUT Project Transfer Document

> Generated: 2026-07-14
> Branch: `claude/debug-sacut-staining-m1dBn`
> Purpose: Continue debugging, training, and development in Claude Code CLI

---

## 1. Project Summary

**SA-CUT** (Structure-Anchored Contrastive Unpaired Translation) generates virtual H&E-stained images from label-free two-photon autofluorescence (TPAF) microscopy of ovarian cancer tissue.

Core architecture: CUT framework + nuclear mask conditioning + region-aware contrastive loss + structure consistency loss.

**The central problem SA-CUT solves:** Standard CUT produces semantic inversion — bright TPAF regions (cytoplasm) mapped to purple (nuclei), dark voids (actual nuclei) become white background. SA-CUT injects pre-computed nuclear masks to force correct TPAF-to-H&E semantic mapping.

---

## 2. Debugging History (Runs 1-7)

### Run 1-4 (pre-session): Baseline failures
- CUT baseline without mask: severe semantic inversion
- Adding mask input helped but nuclei color was blue instead of purple
- L_struct (HED-based) was in dead zone (Dice stuck at 0.72-0.75)

### Run 5: Spectral Norm + Luminance L_struct
**Changes:** Added spectral norm to all D conv layers; switched L_struct from HED threshold mode to luminance mode.

**Results:**
- Purple nuclei achieved (correct)
- L_struct active and meaningful
- **But:** Eosin (pink cytoplasm) pale/missing from epoch 47 onward
- D_loss collapsed from 0.25 to 0.06 by epoch 99 → yellow-green halos appeared

### Run 6: lr_D halved + lambda_idt raised
**Changes:** `lr_D: 2e-4 → 1e-4`, `lambda_idt: 0.2 → 0.5`

**Results:**
- Eosin still pale at epoch 47 — confirmed this is a root cause issue, not just D collapse symptom
- Identity loss was near-zero (~0.016), so raising lambda_idt didn't help much

### Run 7: Added L_color (Color Statistics Matching Loss)
**Changes:** New `ColorStatsLoss` — EMA of real H&E channel mean/std, L1 penalty for deviation. `lambda_color: 5.0`.

**Results:**
- **Epoch 66: Best result yet** — pink eosin visible, purple nuclei correct
- **Epoch 95: Degradation** — cyan/green halos, overly uniform pink
- D_loss had dropped to 0.059 again
- L_color plateaued (~0.39) — G found local minimum of uniform pink

### Run 8 (pending): D Update Gating
**Changes (most recent commit):** When `EMA(D_loss) < 0.1`, skip D gradient updates. D is still evaluated (no grad) so the EMA tracks recovery and automatically re-enables D updates.

**Status:** Code committed and pushed. Training has NOT started yet.

---

## 3. All Code Changes Made (Chronological Commits)

```
40b99d5 losses,trainer,config: fix L_struct dead zone, identity mask, and D stability
d334e51 trainer,config,notebook: add R1 gradient penalty to fix discriminator collapse
20d33e8 trainer,config: fix R1 gradient penalty resolution scaling bug
2269906 losses,models,trainer,config: add spectral norm D + luminance L_struct mode
8b70488 config: fix D-G imbalance and eosin color deficiency for Run 6
cfc6c82 losses,trainer,config: add color statistics matching loss (L_color)
35e8210 trainer,config: add D update gating to prevent discriminator collapse
```

---

## 4. Current Architecture & Loss Composition

### Models
| Component | Architecture | Key Config |
|-----------|-------------|------------|
| Generator G | ResNet-9blocks, early fusion | input_nc=2 (TPAF+Mask), output_nc=3 (RGB) |
| Discriminator D | PatchGAN 70x70, 3 layers | Spectral norm on ALL conv layers, no InstanceNorm |
| Mask Provider | Precomputed binary masks | Loaded from `data/patches/masks/` |

### Loss Function
```
L_total = L_adv                                          # always active
        + lambda_patchnce * L_SA-PatchNCE(mask)          # always active
        + w(t) * lambda_struct * L_struct                # warm-up epochs 0-2, ramp epochs 3-7
        + lambda_idt * L_idt                             # always active
        + lambda_color * L_color                         # warmup 50 batches, then active
        + lambda_r1 * L_R1                               # D-side, always active
```

### Current Hyperparameters (experiment_sa_cut_full.yaml)
```yaml
losses:
  lambda_adv: 1.0
  lambda_patchnce: 1.0
  lambda_struct: 5.0
  lambda_idt: 0.5
  lambda_color: 5.0
  lambda_r1: 1.0
  gan_mode: lsgan
  patchnce_mode: region_aware
  struct_loss_mode: luminance
  struct_warmup_epochs: 3
  struct_rampup_epochs: 5
  color_stats_momentum: 0.01
  color_stats_warmup: 50

training:
  lr_G: 2.0e-4
  lr_D: 1.0e-4
  d_loss_gate_threshold: 0.1
  d_loss_gate_ema_momentum: 0.05

discriminator:
  use_spectral_norm: true
```

### Three-Phase Training Protocol
- **Phase 1 — Warm-up** (epochs 0-2): L_adv + L_SA-PatchNCE only
- **Phase 2 — Full** (epochs 3-199): All losses; L_struct ramps linearly over epochs 3-7
- **Phase 3 — LR decay** (epochs 200-399): Linear LR decay to 0

---

## 5. Key Files Modified

### `trainers/sa_cut_trainer.py` (main training loop)
- **Line 69:** `from losses.color_stats_loss import ColorStatsLoss`
- **Lines 256-273:** Loss initialization — mode-dependent struct threshold, ColorStatsLoss init
- **Lines 344-368:** Training state — D gating variables (threshold, EMA, momentum)
- **Lines 466-479:** Epoch logging — includes `D_ema` and `D_gate%`
- **Lines 497-498:** `criterion_color.update_stats(real_he)` before G forward
- **Lines 531-567:** D update gating logic (core mechanism)
- **Lines ~690-710:** L_color computation in `update_G`
- **Lines ~800-810:** Checkpoint save includes `d_loss_ema`
- **Lines ~885-888:** Checkpoint load restores `d_loss_ema`

### `losses/color_stats_loss.py` (NEW — 119 lines)
- `ColorStatsLoss` class with EMA tracking of real H&E channel-wise mean/std
- `update_stats(real_he)` — EMA update, first batch initializes directly
- `forward(generated_he)` — L1(gen_mean, ema_mean) + L1(gen_std, ema_std)
- Returns zero during warmup (first 50 batches)

### `losses/structure_loss.py`
- Added `extract_luminance_mask()` method — RGB→grayscale→invert→sigmoid threshold
- Added `extract_mask()` dispatcher (luminance vs HED mode)
- `forward()` calls `extract_mask()` instead of `extract_hem_mask()`

### `models/discriminator.py`
- Added `use_spectral_norm` option — `nn.utils.spectral_norm()` on all Conv2d layers
- When enabled, InstanceNorm replaced with Identity (SN handles normalization)

### `configs/experiment_sa_cut_full.yaml`
- `use_spectral_norm: true`, `struct_loss_mode: luminance`, `lambda_color: 5.0`
- `lr_D: 1.0e-4`, `lambda_idt: 0.5`
- `d_loss_gate_threshold: 0.1`, `d_loss_gate_ema_momentum: 0.05`

### `configs/default.yaml`
- Same additions as experiment config (SN, luminance, L_color, D gating)
- D gating disabled by default (`threshold: 0.0`) so ablation configs are unaffected

---

## 6. Known Issues & Unsolved Problems

### Problem 1: D Collapse (partially solved)
**Pattern:** D_loss drops from ~0.25 to ~0.06 across all runs after epoch 60-70. Adversarial gradients saturate, G loses color guidance, image quality degrades (cyan/green halos).

**Mitigations applied:**
- Spectral norm on D (slowed decline but didn't prevent it)
- lr_D = half of lr_G
- R1 gradient penalty (lambda_r1 = 1.0)
- D update gating (newly added, untested)

**What to watch in Run 8:**
- D_ema should stabilize around 0.1 (oscillating above/below threshold)
- D_gate% should be 0% for epochs 0-50, then increase as D gets stronger
- If gating activates too early (before epoch 30), threshold may be too high
- If image quality still degrades despite gating, D may not be the sole cause

### Problem 2: Eosin Color (partially solved)
**Pattern:** Pink eosin (cytoplasm/stroma) consistently pale or absent.

**Root cause analysis:**
- L_struct only constrains dark (nuclear) regions, neutral on eosin
- L_adv's color gradients saturate when D trains too fast
- L_idt is near-zero once G learns identity reconstruction
- L_NCE preserves structure, not color

**Mitigations applied:**
- ColorStatsLoss (L_color) — explicit channel-wise mean/std matching
- Raised lambda_idt from 0.2 to 0.5

**What to watch:**
- loss_color should decrease from ~0.5 to ~0.15 over training
- If L_color causes uniform flat pink (epoch 95 pattern), may need spatial awareness
- Consider: spatial color loss (eosin in non-mask regions only) as future improvement

### Problem 3: L_color Local Minimum
**Observed:** At epoch 95, L_color plateaued at ~0.39. G produced spatially uniform pink instead of location-appropriate colors (pink cytoplasm, white background, dark purple nuclei).

**L_color by design matches global statistics, not spatial patterns.** If this recurs with D gating active, a more targeted approach may be needed:
- Region-conditioned color loss (separate targets for mask/non-mask regions)
- Perceptual color loss (VGG-based, style transfer approach)

---

## 7. Domain Constraints (MUST follow)

1. **TPAF → H&E semantic mapping:**
   - Dark TPAF void → Purple nuclei (hematoxylin)
   - Bright TPAF → Pink cytoplasm (eosin)
   - Very bright TPAF → Deep pink collagen
   - Zero signal large area → White background

2. **Unpaired data:** Never use pixel-wise loss (L1/MSE) between TPAF and H&E. The only pixel-wise loss is L_struct (input mask vs mask extracted from generated H&E — self-consistency).

3. **TPAF normalization:** Always percentile clipping [1st, 99th], never min-max.

4. **Cellpose-SAM:** Frozen, never trained during SA-CUT. Same segmentor at train and test time.

5. **D collapse prevention:** lr_D = half of lr_G, lambda_r1 >= 1.0, spectral norm enabled.

---

## 8. Monitoring Checklist for Next Training Run

### Healthy Training Indicators
```
Epochs 0-10:   D_loss ~0.25-0.40, G_loss ~1.5-3.0, struct=0.0 (warm-up)
Epochs 10-30:  D_loss ~0.15-0.30, struct ramping up, color ~0.4-0.5
Epochs 30-60:  D_loss ~0.12-0.20, color decreasing, struct ~0.15-0.30
Epochs 60-100: D_loss ~0.10-0.15, D_gate starting to activate occasionally
Epochs 100+:   D_gate ~20-50%, D_loss oscillating around threshold
```

### Red Flags
- `D_loss < 0.08` sustained → D gating should be catching this
- `D_gate = 100%` for many epochs → threshold too high or G fundamentally weak
- `loss_color` stuck at 0.4+ after epoch 50 → G not learning global color
- `loss_struct` stuck at initial value → struct extraction may be broken
- Cyan/green halos in generated images → D collapse, adversarial gradient gone
- Uniform flat pink everywhere → L_color local minimum (global stats matched but spatial variation lost)

### Key Epochs to Save & Inspect
- Epoch 10 (end of warm-up)
- Epoch 30 (early full training)
- Epoch 50 (mid training)
- Epoch 70 (historically when degradation starts)
- Epoch 100 (verify D gating effectiveness)

---

## 9. File Structure (relevant files only)

```
SA-CUT/
├── CLAUDE.md                              # Project conventions & domain rules
├── configs/
│   ├── config_utils.py                    # Config loading, merging, CLI override
│   ├── default.yaml                       # Base config (all keys)
│   ├── experiment_sa_cut_full.yaml        # Full model experiment config
│   ├── ablation_cut_baseline.yaml         # Ablation: vanilla CUT
│   ├── ablation_cut_mask_input.yaml       # Ablation: CUT + mask
│   └── ablation_sa_cut_no_struct.yaml     # Ablation: SA-CUT minus L_struct
├── losses/
│   ├── __init__.py                        # (empty)
│   ├── color_stats_loss.py                # NEW: ColorStatsLoss (L_color)
│   ├── gan_loss.py                        # GANLoss (lsgan/vanilla/wgangp)
│   ├── sa_patchnce.py                     # RegionAwarePatchNCE (SA-PatchNCE)
│   └── structure_loss.py                  # StructureConsistencyLoss (L_struct)
├── models/
│   ├── discriminator.py                   # NLayerDiscriminator + spectral norm
│   ├── generator.py                       # ResNetGenerator (9-block, early fusion)
│   ├── mask_provider.py                   # Precomputed / Cellpose-SAM mask loading
│   └── networks.py                        # Weight init utilities
├── trainers/
│   └── sa_cut_trainer.py                  # SACUTTrainer (full training loop)
├── data/
│   └── dataset.py                         # UnpairedTPAFDataset
├── scripts/
│   ├── train.py                           # CLI entry point
│   └── test.py                            # Inference entry point
└── .claude/
    └── skills/
        └── domain-biology.md              # Extended training protocol reference
```

---

## 10. Quick Start Commands

```bash
# Activate environment
conda activate sa-cut  # or your env name

# Train with full config (from scratch)
python scripts/train.py --config configs/experiment_sa_cut_full.yaml

# Train with CLI overrides
python scripts/train.py --config configs/experiment_sa_cut_full.yaml \
  --set training.d_loss_gate_threshold=0.12

# Resume from checkpoint
python scripts/train.py --config configs/experiment_sa_cut_full.yaml \
  --resume checkpoints/sa_cut_full/sa_cut_full/latest.pth

# Monitor training
tensorboard --logdir results/logs/sa_cut_full
```

---

## 11. Next Steps (Priority Order)

1. **Run 8:** Train from scratch with D gating (`d_loss_gate_threshold=0.1`). Monitor D_ema and D_gate% closely.

2. **If D gating works but eosin still flat:** Implement region-conditioned color loss — separate color targets for mask vs non-mask regions.

3. **If D gating fires too aggressively:** Raise threshold to 0.12-0.15 or lower EMA momentum to 0.02 for smoother tracking.

4. **If quality is good at epoch 70 but still degrades at epoch 100+:** The three-phase schedule may need adjustment — consider earlier LR decay onset or reducing total epochs.

5. **Long-term:** Run full ablation table (4 configs) with the final hyperparameters for the paper.

---

## 12. Communication Notes

- User communicates in Chinese; technical terms in English
- Conversation language: Chinese responses preferred, English for code/configs
- User runs training on their own GPU, provides epoch screenshots for evaluation
- User expects concrete analysis of training logs and generated images
