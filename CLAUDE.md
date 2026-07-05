# CLAUDE.md — SA-CUT: Structure-Anchored Virtual H&E Staining

> This file instructs Claude Code on the project context, architecture, coding conventions, and domain constraints. Read this before generating or modifying any code.

## Project Overview

**SA-CUT** (Structure-Anchored Contrastive Unpaired Translation) is a deep learning framework for generating virtual H&E-stained images from label-free two-photon autofluorescence (TPAF) microscopy images of ovarian cancer tissue.

**Core problem:** TPAF images are grayscale, low-SNR, and unintuitive for pathologists. Standard unpaired translation methods (CycleGAN, CUT) produce severe semantic inversion artifacts — bright TPAF regions (cytoplasm/collagen) are incorrectly mapped to dark purple nuclei in H&E, while dark void regions (actual nuclei) become blank background.

**Solution:** Inject pre-computed nuclear segmentation masks as structural anchors into the CUT framework, forcing biologically correct semantic mapping: mask regions → hematoxylin (purple nuclei), non-mask bright regions → eosin (pink cytoplasm/stroma).

## Architecture

| Module | Description | Key Files |
|--------|-------------|-----------|
| **Generator G** | ResNet-9blocks encoder-decoder, input = [TPAF, Mask] (2-ch) | `models/generator.py` |
| **Discriminator D** | PatchGAN (NLayerDiscriminator, n_layers=3) | `models/discriminator.py` |
| **SA-PatchNCE** | Region-aware contrastive loss, mask-partitioned patch sampling | `losses/sa_patchnce.py` |
| **L_struct** | Structure consistency via color-space thresholding or pretrained detector | `losses/structure_loss.py` |
| **Mask Provider** | Precomputed masks (file-based) or Cellpose-SAM (frozen, on-the-fly) | `models/mask_provider.py` |
| **Data Pipeline** | Unpaired TPAF/H&E patch loading from WSI | `data/` |

## Tech Stack & Dependencies

```
Python          3.9–3.11
PyTorch         2.0+
torchvision     0.15+
CUDA            11.8 or 12.x
```

**Required packages:** `torch` `torchvision` `numpy` `scipy` `scikit-image` `opencv-python` `Pillow` `tifffile` `openslide-python` `cellpose` `monai` `pytorch-fid` `tensorboard` `tqdm` `pyyaml`

## Coding Conventions

- Model classes: `PascalCase`; config keys: `snake_case`; file names: `snake_case.py`
- Experiment names: `{method}_{date}_{note}` (e.g., `sa_cut_20260317_baseline`)
- All hyperparameters live in YAML config files under `configs/`. Never hardcode them.

## Local Smoke Test

Before pushing, run `bash scripts/smoke_test.sh` (= `train.py --fast_dev_run`): forces
CPU, generates tiny synthetic patches (`scripts/make_smoke_data.py` → `data/smoke/`,
gitignored), and runs 1 epoch end-to-end (data → G/D → all losses → optimizer step →
checkpoint) in ~20 s. Catches wiring/shape/dtype bugs without a GPU or real data. Real
training still runs on Colab (A100) via `notebooks/SA_CUT_Colab_Train.ipynb`.

## Key Training Rule

**D collapse prevention is mandatory.** If `loss_D` drops below 0.1 (should be 0.3–0.7 for LSGAN), the adversarial gradient to G vanishes and G produces colorized-TPAF output instead of H&E. Set `lambda_r1 ≥ 1.0` and `lr_D = 1e-4` (half of `lr_G`) in all runs. See `.claude/skills/domain-biology.md` for full training protocol.

## Domain-Critical Constraints

These constraints encode biomedical knowledge. Violating them produces biologically wrong results.

### 1. Semantic Inversion Rule

**TPAF → H&E mapping must respect biology:**

| TPAF signal | Biological structure | H&E appearance |
|-------------|---------------------|----------------|
| Dark void (low/no fluorescence) | Nucleus | Purple (hematoxylin) |
| Bright (NADH/FAD) | Cytoplasm | Pink (eosin) |
| Very bright (SHG/collagen) | Collagen fibers | Deep pink/red |
| Zero signal, large area | Background/lumen | White |

If the generated H&E shows bright TPAF regions as purple, or dark voids as white background, the model is **wrong**. This is the central problem SA-CUT solves.

### 2. Mask Quality Dependency

The entire pipeline depends on the nuclear mask quality, whether from precomputed files or Cellpose-SAM. Code must:
- Log mask quality metrics (estimated coverage ratio, connected component stats) during training.
- Support uncertainty-weighted masks (soft probability maps, not just binary).
- Never modify or fine-tune the frozen Cellpose-SAM model during SA-CUT training.
- The mask provider operates in one of two modes (configured via `mask_provider.mode`):
  - `precomputed`: load masks from disk (default, fastest). Mask files must exist for every TPAF patch, matched by filename.
  - `cellpose_sam`: run Cellpose-SAM inference on-the-fly (slower, useful for test-time on new data). The model checkpoint is frozen.

### 3. Unpaired Data Assumptions

- TPAF and H&E patches are **never** pixel-aligned. Do not compute pixel-wise losses (MSE, L1) between source TPAF and target H&E.
- The only pixel-wise loss is L_struct: between the input nuclear mask and a mask extracted from the **generated** H&E (self-consistency, not cross-domain).

### 4. Image Properties

| Property | TPAF | H&E |
|----------|------|-----|
| Channels | 1 (grayscale) or 2 (NADH + FAD) | 3 (RGB) |
| Bit depth | 16-bit (uint16) or 12-bit | 8-bit (uint8) |
| Dynamic range | Very low SNR, heavy tail | Standard natural image |
| Patch size | 256×256 (default) | 256×256 (default) |
| Normalization | Percentile-based (1st–99th), then [0, 1] | Standard [0, 1] or [-1, 1] |

**Critical:** TPAF normalization must use percentile clipping, NOT min-max, because outlier hot pixels distort the range. Always clip to [1st, 99th] percentile before scaling.

## Common Pitfalls to Avoid

1. **Never use pixel-wise loss (L1/MSE) between TPAF source and H&E target.** This is unpaired translation.
2. **Never normalize TPAF with min-max.** Use percentile clipping.
3. **Never train or fine-tune Cellpose-SAM during SA-CUT training.** It must be frozen.
4. **Never use batch normalization in the generator for batch_size=1.** Use instance normalization.
5. **Always detach generator output when updating discriminator.** Standard GAN practice.
6. **Always verify mask dtype.** Mask should be `float32` in [0, 1], not `uint8` or `bool`. Binary hard masks use 0.0/1.0; soft masks use probability values.
7. **Structure loss warm-up is mandatory.** Enabling L_struct from epoch 0 produces garbage gradients because the generator output doesn't resemble H&E yet.
8. **Test-time: masks must come from the same segmentation source used during training.** If training used precomputed Cellpose-SAM masks, inference must also use Cellpose-SAM (same checkpoint) or masks precomputed by it. Switching segmentors between train and test changes the mask distribution and degrades results.
9. **TPAF 16-bit images must be loaded with `tifffile`, not PIL/OpenCV.** PIL silently clips to 8-bit.

## Git Conventions

- Branch naming: `feature/{module}`, `fix/{issue}`, `ablation/{name}`
- Commit messages: imperative mood, prefix with module name. E.g., `losses: implement region-aware PatchNCE with mask partitioning`
- Never commit model checkpoints or dataset files. Use `.gitignore`.
- Tag reproducible experiment runs: `v0.1-baseline`, `v0.2-sa-patchnce`, etc.

## Extended Reference

Full training protocol, ablation configurations, evaluation metrics, and paper references: `.claude/skills/domain-biology.md`
