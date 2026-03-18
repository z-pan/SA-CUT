#!/usr/bin/env bash
# Run all ablation experiments sequentially.
# Usage: bash scripts/run_ablations.sh

set -euo pipefail

PYTHON=python

echo "=== Ablation 1: CUT Baseline ==="
$PYTHON scripts/train.py --config configs/ablation_cut.yaml

echo "=== Ablation 2: SA-CUT without Structure Loss ==="
$PYTHON scripts/train.py --config configs/ablation_no_struct.yaml

echo "=== Full SA-CUT Model ==="
$PYTHON scripts/train.py --config configs/experiment_full.yaml

echo "All ablation experiments complete."
