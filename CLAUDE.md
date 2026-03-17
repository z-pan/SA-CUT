# CLAUDE.md — SA-CUT: Structure-Anchored Virtual H&E Staining

> This file instructs Claude Code on the project context, architecture, coding conventions, and domain constraints. Read this before generating or modifying any code.

---

## Project Overview

**SA-CUT** (Structure-Anchored Contrastive Unpaired Translation) is a deep learning framework for generating virtual H&E-stained images from label-free two-photon autofluorescence (TPAF) microscopy images of ovarian cancer tissue.

**Core problem:** TPAF images are grayscale, low-SNR, and unintuitive for pathologists. Standard unpaired translation methods (CycleGAN, CUT) produce severe semantic inversion artifacts — bright TPAF regions (cytoplasm/collagen) are incorrectly mapped to dark purple nuclei in H&E, while dark void regions (actual nuclei) become blank background.

**Solution:** Inject pre-computed nuclear segmentation masks as structural anchors into the CUT framework, forcing biologically correct semantic mapping: mask regions → hematoxylin (purple nuclei), non-mask bright regions → eosin (pink cytoplasm/stroma).

---

## Architecture Summary

```
TPAF image ──┬──→ [AFN-DeSeg (frozen)] → Nuclear Mask ──┐
             │                                            ├→ Concat → Generator G → Virtual H&E
             └────────────────────────────────────────────┘
                                                                    ↓
                                              ┌─── L_adv (PatchGAN Discriminator)
                                              ├─── L_SA-PatchNCE (Region-Aware Contrastive)
                              L_total ←───────┤
                                              ├─── L_struct (Mask Consistency)
                                              └─── L_idt (Identity, optional)
```

### Module Breakdown

| Module | Description | Key Files |
|--------|-------------|-----------|
| **Generator G** | ResNet-9blocks encoder-decoder, input = [TPAF, Mask] (2-ch) | `models/generator.py` |
| **Discriminator D** | PatchGAN (NLayerDiscriminator, n_layers=3) | `models/discriminator.py` |
| **SA-PatchNCE** | Region-aware contrastive loss, mask-partitioned patch sampling | `losses/sa_patchnce.py` |
| **L_struct** | Structure consistency via color-space thresholding or pretrained detector | `losses/structure_loss.py` |
| **AFN-DeSeg** | Frozen pretrained nuclear segmentation, provides mask | `models/afn_deseg.py` (wrapper only) |
| **Data Pipeline** | Unpaired TPAF/H&E patch loading from WSI | `data/` |

---

## Tech Stack & Dependencies

```
Python          3.9–3.11
PyTorch         2.0+
torchvision     0.15+
CUDA            11.8 or 12.x
```

### Required Packages

```
torch, torchvision
numpy, scipy, scikit-image
opencv-python (cv2)
Pillow
tifffile              # for reading TPAF .tif WSI
openslide-python      # for reading H&E .svs/.ndpi WSI (optional)
monai                 # medical image transforms, Dice loss
pytorch-fid           # FID evaluation
tensorboard or wandb  # experiment tracking
tqdm
pyyaml                # config management
```

### Optional

```
timm                  # if using pretrained encoders beyond ResNet
hover_net             # for L_struct Plan B (pretrained H&E nucleus detector)
albumentations        # advanced augmentation
```

---

## Project Directory Structure

```
SA-CUT/
├── CLAUDE.md                  # THIS FILE
├── configs/
│   ├── default.yaml           # default hyperparameters
│   ├── ablation_cut.yaml      # CUT baseline (no mask)
│   ├── ablation_no_struct.yaml
│   └── experiment_full.yaml
├── data/
│   ├── __init__.py
│   ├── dataset.py             # UnpairedTPAFDataset
│   ├── patch_extractor.py     # WSI → patch extraction pipeline
│   └── transforms.py          # domain-specific augmentations
├── models/
│   ├── __init__.py
│   ├── generator.py           # ResNetGenerator (mask-conditioned)
│   ├── discriminator.py       # NLayerDiscriminator (PatchGAN)
│   ├── networks.py            # weight init, normalization helpers
│   └── afn_deseg.py           # frozen segmentation model wrapper
├── losses/
│   ├── __init__.py
│   ├── gan_loss.py            # LSGAN / vanilla GAN loss
│   ├── sa_patchnce.py         # Region-Aware PatchNCE
│   └── structure_loss.py      # L_struct (mask consistency)
├── trainers/
│   ├── __init__.py
│   └── sa_cut_trainer.py      # main training loop
├── evaluation/
│   ├── compute_fid.py
│   ├── structure_consistency.py  # mask IoU between input mask & generated H&E nuclei
│   └── visualize_results.py
├── scripts/
│   ├── train.py               # entry point
│   ├── test.py                # inference
│   ├── extract_patches.py     # WSI preprocessing
│   └── run_ablations.sh       # ablation experiment launcher
├── checkpoints/               # saved model weights (gitignored)
├── results/                   # generated images & metrics (gitignored)
└── requirements.txt
```

---

## Coding Conventions

### Style

- **PEP 8** strictly. Use `black` formatter, `isort` for imports.
- Type hints on all function signatures.
- Docstrings: Google style. Every public class and function must have a docstring.
- Max line length: 100 characters.

### Naming

- Model classes: `PascalCase` (e.g., `ResNetGenerator`, `RegionAwarePatchNCE`)
- Config keys: `snake_case` (e.g., `lambda_struct`, `n_resnet_blocks`)
- File names: `snake_case.py`
- Experiment names: `{method}_{date}_{note}` (e.g., `sa_cut_20260317_baseline`)

### Architecture Patterns

- **Config-driven:** All hyperparameters live in YAML config files, never hardcoded. Use `argparse` + YAML override pattern.
- **Modular losses:** Each loss is a standalone `nn.Module` with a `forward()` that returns a scalar tensor. The trainer composes them.
- **Reproducibility:** Every experiment logs: config YAML, git commit hash, random seeds (Python, NumPy, PyTorch, CUDA).
- **Checkpoint format:** `{'epoch': int, 'G_state_dict': ..., 'D_state_dict': ..., 'optimizer_G': ..., 'optimizer_D': ..., 'config': dict}`

### PyTorch Patterns

```python
# Correct: explicit device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Correct: gradient control for frozen modules
with torch.no_grad():
    mask = seg_model(tpaf_image)

# Correct: GAN training order
# 1. Update D (detach G output)
# 2. Update G (don't detach)
```

---

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

The entire pipeline depends on AFN-DeSeg's segmentation quality. Code must:
- Log mask quality metrics (estimated coverage ratio, connected component stats) during training.
- Support uncertainty-weighted masks (soft probability maps, not just binary).
- Never modify or fine-tune the frozen segmentation model during SA-CUT training.

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

---

## Hyperparameter Reference

```yaml
# Generator
generator:
  input_nc: 2           # TPAF (1ch) + Mask (1ch); or 3 if NADH+FAD+Mask
  output_nc: 3           # RGB H&E
  ngf: 64
  n_resnet_blocks: 9
  norm_type: instance
  use_dropout: false
  mask_injection: early_fusion   # Options: early_fusion | multi_scale

# Discriminator
discriminator:
  input_nc: 3
  ndf: 64
  n_layers: 3
  norm_type: instance

# Losses
losses:
  lambda_adv: 1.0
  lambda_patchnce: 1.0
  lambda_struct: 5.0       # high — structure preservation is the core contribution
  lambda_idt: 0.5          # optional, set 0 to disable
  gan_mode: lsgan          # lsgan (recommended) | vanilla | wgangp
  n_patchnce_layers: 5     # number of encoder layers to sample patches from
  n_patches_per_layer: 256
  struct_loss_mode: threshold  # threshold | pretrained_detector
  struct_warmup_epochs: 10     # epochs before L_struct activates

# Training
training:
  batch_size: 1
  lr_G: 2.0e-4
  lr_D: 2.0e-4
  beta1: 0.5
  beta2: 0.999
  n_epochs: 200
  n_epochs_decay: 200     # linear LR decay over these epochs
  patch_size: 256
  seed: 42
```

---

## Training Protocol

### Phase 1: Warm-up (epochs 1–10)
- Active losses: `L_adv + L_PatchNCE`
- L_struct disabled (generator hasn't learned color mapping yet)
- Purpose: generator learns basic TPAF → H&E color distribution

### Phase 2: Full training (epochs 11–200)
- All losses active: `L_adv + L_SA-PatchNCE + L_struct + L_idt`
- L_struct ramps linearly from 0 to `lambda_struct` over 5 epochs

### Phase 3: Decay (epochs 201–400)
- Linear LR decay to 0
- All losses remain active

### Ablation Configurations

| Experiment | Config | What changes |
|------------|--------|-------------|
| CUT baseline | `ablation_cut.yaml` | `input_nc=1`, no mask, standard PatchNCE |
| CUT + mask input | custom | `input_nc=2`, standard PatchNCE, no L_struct |
| SA-CUT w/o struct | `ablation_no_struct.yaml` | SA-PatchNCE but `lambda_struct=0` |
| **SA-CUT full** | `experiment_full.yaml` | All components |

---

## Evaluation Metrics

| Metric | Script | Purpose |
|--------|--------|---------|
| FID | `evaluation/compute_fid.py` | Distribution-level realism |
| Nuclei Mask IoU | `evaluation/structure_consistency.py` | Input mask vs generated H&E nuclei mask |
| Nuclei F1 Score | `evaluation/structure_consistency.py` | Per-nucleus detection agreement |
| LPIPS (optional) | — | Perceptual similarity within each generated image |

Do **not** compute PSNR/SSIM between generated H&E and real H&E — there is no paired ground truth.

---

## Common Pitfalls to Avoid

1. **Never use pixel-wise loss (L1/MSE) between TPAF source and H&E target.** This is unpaired translation.
2. **Never normalize TPAF with min-max.** Use percentile clipping.
3. **Never train or fine-tune AFN-DeSeg during SA-CUT training.** It must be frozen.
4. **Never use batch normalization in the generator for batch_size=1.** Use instance normalization.
5. **Always detach generator output when updating discriminator.** Standard GAN practice.
6. **Always verify mask dtype.** Mask should be `float32` in [0, 1], not `uint8` or `bool`. Binary hard masks use 0.0/1.0; soft masks use probability values.
7. **Structure loss warm-up is mandatory.** Enabling L_struct from epoch 0 produces garbage gradients because the generator output doesn't resemble H&E yet.
8. **Test-time: mask must come from the same AFN-DeSeg checkpoint used during training.** Different segmentation models produce different mask distributions.
9. **TPAF 16-bit images must be loaded with `tifffile`, not PIL/OpenCV.** PIL silently clips to 8-bit.

---

## Git Conventions

- Branch naming: `feature/{module}`, `fix/{issue}`, `ablation/{name}`
- Commit messages: imperative mood, prefix with module name. E.g., `losses: implement region-aware PatchNCE with mask partitioning`
- Never commit model checkpoints or dataset files. Use `.gitignore`.
- Tag reproducible experiment runs: `v0.1-baseline`, `v0.2-sa-patchnce`, etc.

---

## References

These papers define the baseline methods. The codebase should be consistent with their formulations.

- **CUT:** Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation," ECCV 2020.
- **CycleGAN:** Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," ICCV 2017.
- **UTOM:** Li et al., "Unsupervised content-preserving transformation for optical microscopy," Light: Science & Applications, 2021.
- **PatchGAN:** Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.
- **Virtual Staining (paired):** Rivenson et al., "Virtual histological staining of unlabelled tissue-autofluorescence images via deep learning," Nature Biomedical Engineering, 2019.

---

## Contact

- **Author:** Pan Zhengyuan (潘正元)
- **Affiliation:** Shanghai Jiao Tong University, Department of Biomedical Engineering
- **Thesis Chapter:** Chapter 5 — Structure-Anchored Virtual H&E Staining
