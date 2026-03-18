#!/usr/bin/env bash
# =============================================================================
# SA-CUT Ablation Experiment Runner
# =============================================================================
#
# Runs all five ablation experiments sequentially, then evaluates each one
# with FID/KID (distribution-level realism) and structure consistency
# (structural preservation), and finally produces a summary comparison CSV.
#
# Experiments (in training order)
# --------------------------------
#   1. ablation_cyclegan         CycleGAN baseline (no mask, cycle loss)
#   2. ablation_cut_baseline     Standard CUT (no mask, standard PatchNCE)
#   3. ablation_cut_mask_input   CUT with mask as 2nd input channel
#   4. ablation_sa_cut_no_struct SA-CUT without L_struct (SA-PatchNCE only)
#   5. sa_cut_full               Full SA-CUT (all components)
#
# Output layout
# -------------
#   results/ablations/
#   ├── ablation_cyclegan/
#   │   ├── checkpoints/        model weights (.pth)
#   │   ├── logs/               TensorBoard event files
#   │   ├── generated_he/       test-set H&E outputs
#   │   └── eval/
#   │       ├── fid.json        FID + KID scores
#   │       └── struct/         structure consistency CSV + JSON + figure
#   ├── ablation_cut_baseline/  (same layout)
#   ├── ...
#   └── summary_ablations.csv   ← cross-experiment comparison table
#
# Usage
# -----
#   # Full run (train + evaluate all 5 experiments):
#   bash scripts/run_ablations.sh
#
#   # Skip training, run evaluation only (assumes checkpoints exist):
#   SKIP_TRAIN=1 bash scripts/run_ablations.sh
#
#   # Skip inference, run evaluation only (assumes generated_he exists):
#   SKIP_TRAIN=1 SKIP_INFER=1 bash scripts/run_ablations.sh
#
#   # Run a specific subset of experiments (space-separated):
#   EXPERIMENTS="ablation_cut_baseline sa_cut_full" bash scripts/run_ablations.sh
#
#   # Quick-test mode: 2 epochs each (for CI / smoke testing):
#   QUICK_TEST=1 bash scripts/run_ablations.sh
#
#   # Use a custom Python interpreter or virtual environment:
#   PYTHON=/path/to/venv/bin/python bash scripts/run_ablations.sh
#
# Environment variables
# ---------------------
#   PYTHON          Python interpreter (default: python)
#   RESULTS_ROOT    Root output directory (default: results/ablations)
#   TPAF_DIR        TPAF test patches for inference (default: data/patches/tpaf)
#   MASK_DIR        Precomputed masks (default: data/patches/masks)
#   REAL_HE_DIR     Real H&E for FID reference (default: data/raw/he)
#   SKIP_TRAIN      Set to 1 to skip the training phase
#   SKIP_INFER      Set to 1 to skip the inference phase
#   SKIP_EVAL       Set to 1 to skip FID + struct-consistency evaluation
#   QUICK_TEST      Set to 1 to run only 2 epochs per experiment (smoke test)
#   EXPERIMENTS     Space-separated list of experiment names to run
#                   (default: all five; see EXPERIMENT_NAMES array below)
#
# Prerequisites
# -------------
#   pip install torch torchvision scikit-image scipy matplotlib
#   pip install pytorch-fid   # or evaluation/compute_fid.py handles it
#
# Notes
# -----
#   * The script runs experiments sequentially.  For parallel execution on a
#     cluster, submit each experiment as a separate job and then call the
#     evaluation and summary steps once all jobs complete.
#   * set -euo pipefail ensures any non-zero exit code immediately aborts the
#     entire script.  Remove 'e' from set flags if you want to continue on
#     individual experiment failures.
#   * CycleGAN trains two generators and two discriminators.  Its checkpoint
#     will only contain the forward generator (G_TPAF2HE) for inference.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (overridable via environment)
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablations}"
TPAF_DIR="${TPAF_DIR:-data/patches/tpaf}"
MASK_DIR="${MASK_DIR:-data/patches/masks}"
REAL_HE_DIR="${REAL_HE_DIR:-data/raw/he}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
QUICK_TEST="${QUICK_TEST:-0}"

# Ordered list: CycleGAN first (weakest), SA-CUT full last (strongest)
declare -a EXPERIMENT_NAMES_DEFAULT=(
    "ablation_cyclegan"
    "ablation_cut_baseline"
    "ablation_cut_mask_input"
    "ablation_sa_cut_no_struct"
    "sa_cut_full"
)
# Allow caller to override with a space-separated list
if [[ -n "${EXPERIMENTS:-}" ]]; then
    IFS=' ' read -r -a EXPERIMENT_NAMES <<< "${EXPERIMENTS}"
else
    EXPERIMENT_NAMES=("${EXPERIMENT_NAMES_DEFAULT[@]}")
fi

# Map experiment name → config file
declare -A EXPERIMENT_CONFIGS=(
    ["ablation_cyclegan"]="configs/ablation_cyclegan.yaml"
    ["ablation_cut_baseline"]="configs/ablation_cut_baseline.yaml"
    ["ablation_cut_mask_input"]="configs/ablation_cut_mask_input.yaml"
    ["ablation_sa_cut_no_struct"]="configs/ablation_sa_cut_no_struct.yaml"
    ["sa_cut_full"]="configs/experiment_sa_cut_full.yaml"
)

# Map experiment name → human-readable display name
declare -A EXPERIMENT_LABELS=(
    ["ablation_cyclegan"]="CycleGAN"
    ["ablation_cut_baseline"]="CUT (baseline)"
    ["ablation_cut_mask_input"]="CUT + Mask"
    ["ablation_sa_cut_no_struct"]="SA-CUT (no L_struct)"
    ["sa_cut_full"]="SA-CUT (full)"
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_ROOT}/run_${TIMESTAMP}.log"
SUMMARY_CSV="${RESULTS_ROOT}/summary_ablations.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    local line
    line="$(printf '[%s]  %-8s  %s' "${ts}" "${level}" "${msg}")"
    echo "${line}"
    echo "${line}" >> "${LOG_FILE}"
}

info()    { log "INFO"    "$@"; }
success() { log "OK"      "$@"; }
warn()    { log "WARN"    "$@"; }
fail()    { log "ERROR"   "$@"; exit 1; }

section() {
    local title="$*"
    local line
    line="$(printf '═%.0s' {1..72})"
    { echo ""; echo "${line}"; echo "  ${title}"; echo "${line}"; } \
        | tee -a "${LOG_FILE}"
}

check_python() {
    if ! "${PYTHON}" -c "import torch" &>/dev/null; then
        fail "PyTorch not importable from '${PYTHON}'. Check your Python environment."
    fi
    local py_ver
    py_ver="$("${PYTHON}" --version 2>&1)"
    info "Python: ${py_ver}"
}

# ---------------------------------------------------------------------------
# Phase 0: Setup
# ---------------------------------------------------------------------------

mkdir -p "${RESULTS_ROOT}"

section "SA-CUT Ablation Runner  |  ${TIMESTAMP}"
info "Results root  : ${RESULTS_ROOT}"
info "TPAF dir      : ${TPAF_DIR}"
info "Mask dir      : ${MASK_DIR}"
info "Real H&E dir  : ${REAL_HE_DIR}"
info "Experiments   : ${EXPERIMENT_NAMES[*]}"
info "Log file      : ${LOG_FILE}"
[[ "${QUICK_TEST}" == "1" ]] && warn "QUICK_TEST=1 — training for 2 epochs only (smoke test)"

check_python

# Verify required scripts exist
for script in scripts/train.py scripts/test.py \
              evaluation/compute_fid.py evaluation/structure_consistency.py; do
    [[ -f "${script}" ]] || fail "Required script not found: ${script}"
done

# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------

if [[ "${SKIP_TRAIN}" == "1" ]]; then
    warn "SKIP_TRAIN=1 — skipping training phase"
else
    section "Phase 1: Training"

    for exp_name in "${EXPERIMENT_NAMES[@]}"; do
        config="${EXPERIMENT_CONFIGS[${exp_name}]:-}"
        [[ -z "${config}" ]] && fail "Unknown experiment: '${exp_name}'"
        [[ -f "${config}" ]] || fail "Config not found: ${config}"

        exp_out="${RESULTS_ROOT}/${exp_name}"
        ckpt_dir="${exp_out}/checkpoints"
        log_dir="${exp_out}/logs"

        section "Training: ${EXPERIMENT_LABELS[${exp_name}]}  (${exp_name})"
        info "Config : ${config}"
        info "Output : ${exp_out}"

        mkdir -p "${ckpt_dir}" "${log_dir}"

        # Build CLI overrides: redirect checkpoints and logs into ablations/
        local_overrides=(
            "--experiment.checkpoint_dir=${ckpt_dir}"
            "--experiment.log_dir=${log_dir}"
            "--experiment.name=${exp_name}"
        )

        # Quick-test: override epoch counts to 2
        if [[ "${QUICK_TEST}" == "1" ]]; then
            local_overrides+=(
                "--training.n_epochs=2"
                "--training.n_epochs_decay=0"
                "--losses.struct_warmup_epochs=0"
                "--losses.struct_rampup_epochs=0"
            )
        fi

        "${PYTHON}" scripts/train.py \
            --config "${config}" \
            "${local_overrides[@]}" \
            2>&1 | tee "${exp_out}/train.log"

        success "Training complete: ${exp_name}"
    done

    success "Phase 1 (training) complete."
fi

# ---------------------------------------------------------------------------
# Phase 2: Inference (generate test-set H&E images)
# ---------------------------------------------------------------------------

if [[ "${SKIP_INFER}" == "1" ]]; then
    warn "SKIP_INFER=1 — skipping inference phase"
else
    section "Phase 2: Inference"

    for exp_name in "${EXPERIMENT_NAMES[@]}"; do
        exp_out="${RESULTS_ROOT}/${exp_name}"
        ckpt_dir="${exp_out}/checkpoints"
        out_he="${exp_out}/generated_he"

        # Locate the latest checkpoint
        latest_ckpt=""
        for candidate in "${ckpt_dir}/latest.pth" "${ckpt_dir}/best.pth"; do
            if [[ -f "${candidate}" ]]; then
                latest_ckpt="${candidate}"
                break
            fi
        done
        # Fall back to the highest-epoch checkpoint
        if [[ -z "${latest_ckpt}" ]]; then
            latest_ckpt="$(ls -t "${ckpt_dir}"/*.pth 2>/dev/null | head -1 || true)"
        fi

        if [[ -z "${latest_ckpt}" ]]; then
            warn "No checkpoint found for '${exp_name}' in '${ckpt_dir}' — skipping inference."
            continue
        fi

        section "Inference: ${EXPERIMENT_LABELS[${exp_name}]}"
        info "Checkpoint : ${latest_ckpt}"
        info "Output     : ${out_he}"

        mkdir -p "${out_he}"

        "${PYTHON}" scripts/test.py \
            --checkpoint "${latest_ckpt}" \
            --input_dir  "${TPAF_DIR}" \
            --output_dir "${out_he}" \
            --mask_dir   "${MASK_DIR}" \
            2>&1 | tee "${exp_out}/infer.log"

        success "Inference complete: ${exp_name} → ${out_he}"
    done

    success "Phase 2 (inference) complete."
fi

# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------

if [[ "${SKIP_EVAL}" == "1" ]]; then
    warn "SKIP_EVAL=1 — skipping evaluation phase"
else
    section "Phase 3: Evaluation"

    for exp_name in "${EXPERIMENT_NAMES[@]}"; do
        exp_out="${RESULTS_ROOT}/${exp_name}"
        out_he="${exp_out}/generated_he"
        eval_dir="${exp_out}/eval"

        if [[ ! -d "${out_he}" ]] || [[ -z "$(ls -A "${out_he}" 2>/dev/null)" ]]; then
            warn "generated_he directory empty or missing for '${exp_name}' — skipping eval."
            continue
        fi

        mkdir -p "${eval_dir}"

        # ── 3a. FID / KID evaluation ──────────────────────────────────────
        section "  FID/KID: ${EXPERIMENT_LABELS[${exp_name}]}"
        "${PYTHON}" evaluation/compute_fid.py \
            --generated "${out_he}" \
            --real      "${REAL_HE_DIR}" \
            --output    "${eval_dir}/fid.json" \
            2>&1 | tee "${eval_dir}/fid.log"

        success "  FID done: ${exp_name}"

        # ── 3b. Structure consistency evaluation ──────────────────────────
        section "  Struct: ${EXPERIMENT_LABELS[${exp_name}]}"
        "${PYTHON}" evaluation/structure_consistency.py \
            --he_dir     "${out_he}" \
            --mask_dir   "${MASK_DIR}" \
            --output_dir "${eval_dir}/struct" \
            --tpaf_dir   "${TPAF_DIR}" \
            2>&1 | tee "${eval_dir}/struct.log"

        success "  Struct done: ${exp_name}"
    done

    success "Phase 3 (evaluation) complete."
fi

# ---------------------------------------------------------------------------
# Phase 4: Summary comparison table
# ---------------------------------------------------------------------------

section "Phase 4: Building summary table"

"${PYTHON}" - <<'PYEOF'
"""Generate a CSV summary table across all ablation experiments.

Reads per-experiment JSON files produced by compute_fid.py and
structure_consistency.py, then writes a single CSV where each row
corresponds to one experiment and columns are evaluation metrics.

Missing files are filled with empty strings so partial runs still produce
a usable (if incomplete) table.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config (mirrors shell variables above) ────────────────────────────────
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results/ablations")
SUMMARY_CSV  = os.path.join(RESULTS_ROOT, "summary_ablations.csv")

EXPERIMENT_ORDER = [
    "ablation_cyclegan",
    "ablation_cut_baseline",
    "ablation_cut_mask_input",
    "ablation_sa_cut_no_struct",
    "sa_cut_full",
]
EXPERIMENT_LABELS = {
    "ablation_cyclegan":         "CycleGAN",
    "ablation_cut_baseline":     "CUT (baseline)",
    "ablation_cut_mask_input":   "CUT + Mask",
    "ablation_sa_cut_no_struct": "SA-CUT (no L_struct)",
    "sa_cut_full":               "SA-CUT (full)",
}


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON silently, returning None on any error."""
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def fmt(value: Any, decimals: int = 4) -> str:
    """Format a numeric value, or return empty string for None / missing."""
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def mean_std(agg: Dict, key: str) -> tuple[str, str]:
    """Extract mean ± std from an aggregated metrics dict."""
    entry = agg.get(key, {})
    return fmt(entry.get("mean")), fmt(entry.get("std"))


rows: List[Dict[str, str]] = []

for exp_name in EXPERIMENT_ORDER:
    exp_dir  = Path(RESULTS_ROOT) / exp_name
    fid_path = exp_dir / "eval" / "fid.json"
    struct_path = exp_dir / "eval" / "struct" / "summary.json"

    fid_data    = safe_load_json(fid_path)
    struct_data = safe_load_json(struct_path)

    # ── FID / KID metrics ─────────────────────────────────────────────
    fid_score = kid_mean = kid_std = ""
    if fid_data:
        fid_score = fmt(fid_data.get("fid"))
        kid_entry = fid_data.get("kid", {})
        kid_mean  = fmt(kid_entry.get("mean"))
        kid_std   = fmt(kid_entry.get("std"))

    # ── Structure consistency metrics ────────────────────────────────
    agg = struct_data.get("aggregated", {}) if struct_data else {}
    n_eval = ""
    if struct_data:
        n_eval = str(struct_data.get("n_evaluated", ""))

    dice_m,   dice_s   = mean_std(agg, "dice")
    iou_m,    iou_s    = mean_std(agg, "iou")
    prec_m,   prec_s   = mean_std(agg, "precision")
    rec_m,    rec_s    = mean_std(agg, "recall")
    comp_m,   comp_s   = mean_std(agg, "component_ratio")
    cov_in_m, cov_in_s = mean_std(agg, "input_mask_coverage")
    cov_ex_m, cov_ex_s = mean_std(agg, "extracted_mask_coverage")

    row = {
        "experiment":              exp_name,
        "label":                   EXPERIMENT_LABELS.get(exp_name, exp_name),
        # ── Realism ──────────────────────────────────────────────────
        "fid":                     fid_score,
        "kid_mean":                kid_mean,
        "kid_std":                 kid_std,
        # ── Structure consistency (mean ± std) ───────────────────────
        "dice_mean":               dice_m,
        "dice_std":                dice_s,
        "iou_mean":                iou_m,
        "iou_std":                 iou_s,
        "precision_mean":          prec_m,
        "precision_std":           prec_s,
        "recall_mean":             rec_m,
        "recall_std":              rec_s,
        "component_ratio_mean":    comp_m,
        "component_ratio_std":     comp_s,
        "coverage_input_mean":     cov_in_m,
        "coverage_input_std":      cov_in_s,
        "coverage_extracted_mean": cov_ex_m,
        "coverage_extracted_std":  cov_ex_s,
        "n_evaluated":             n_eval,
    }
    rows.append(row)

# ── Write CSV ────────────────────────────────────────────────────────────
if not rows:
    print("No experiment results found — summary CSV not written.", file=sys.stderr)
    sys.exit(0)

summary_path = Path(SUMMARY_CSV)
summary_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0].keys())

with summary_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary table written → {summary_path}")

# ── Pretty-print to terminal ─────────────────────────────────────────────
col_w = 26
line = "─" * (col_w + 8 + 9 + 9 + 11 + 11)
print()
print(line)
print(f"{'Experiment':<{col_w}}  {'FID':>7}  {'KID':>8}  {'Dice':>9}  {'IoU':>9}  {'CompRatio':>10}")
print(line)
for r in rows:
    kid_str = f"{r['kid_mean']}±{r['kid_std']}" if r["kid_mean"] else ""
    dice_str = f"{r['dice_mean']}±{r['dice_std']}" if r["dice_mean"] else ""
    iou_str  = f"{r['iou_mean']}±{r['iou_std']}"  if r["iou_mean"]  else ""
    comp_str = f"{r['component_ratio_mean']}±{r['component_ratio_std']}" \
               if r["component_ratio_mean"] else ""
    print(
        f"{r['label']:<{col_w}}  "
        f"{r['fid']:>7}  "
        f"{kid_str:>8}  "
        f"{dice_str:>9}  "
        f"{iou_str:>9}  "
        f"{comp_str:>10}"
    )
print(line)
print()
PYEOF

success "Summary CSV written to: ${SUMMARY_CSV}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

section "All phases complete"
info "Summary CSV  : ${SUMMARY_CSV}"
info "Full log     : ${LOG_FILE}"
info "Results root : $(realpath "${RESULTS_ROOT}" 2>/dev/null || echo "${RESULTS_ROOT}")"
