#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — fast local CPU sanity check for SA-CUT.
#
# Generates tiny synthetic patches, then runs one training iteration on CPU
# (1 epoch, throwaway output). Exercises the whole path — data -> G/D -> losses
# (adv, SA-PatchNCE, struct, color) -> optimizer step -> checkpoint — in a
# minute or two, catching wiring/shape/dtype bugs BEFORE you push or train on
# the A100. No GPU and no real data needed.
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --config configs/experiment_full.yaml   # any config
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/default.yaml"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *)        EXTRA+=("$1"); shift ;;   # forwarded to train.py
    esac
done

export CUDA_VISIBLE_DEVICES="-1"   # force CPU (train.py also guards on --fast_dev_run)

echo "== SMOKE: generating synthetic patches =="
python scripts/make_smoke_data.py

echo "== SMOKE: config=$CONFIG (CPU, 1 epoch, tiny synthetic data) =="
python scripts/train.py --config "$CONFIG" --fast_dev_run "${EXTRA[@]}"

echo "== SMOKE PASSED =="
