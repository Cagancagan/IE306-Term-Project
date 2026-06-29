#!/usr/bin/env bash
# Reproduce the final centralized-dispatch evaluation table.
# Config and seeds are overridable so the instructor can swap in held-out settings.
set -euo pipefail

CONFIG="${1:-configs/eval_standard.yaml}"
SEEDS="${2:-0,1,2,3,4,5,6,7,8,9}"

python run_all.py --config "$CONFIG" --seeds "$SEEDS"
