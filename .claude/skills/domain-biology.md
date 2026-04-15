# SA-CUT Domain Knowledge & Training Protocol

> Load this skill when debugging training behavior, designing ablations, or evaluating results.

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

## References

These papers define the baseline methods. The codebase should be consistent with their formulations.

- **CUT:** Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation," ECCV 2020.
- **CycleGAN:** Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," ICCV 2017.
- **UTOM:** Li et al., "Unsupervised content-preserving transformation for optical microscopy," Light: Science & Applications, 2021.
- **PatchGAN:** Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.
- **Virtual Staining (paired):** Rivenson et al., "Virtual histological staining of unlabelled tissue-autofluorescence images via deep learning," Nature Biomedical Engineering, 2019.
