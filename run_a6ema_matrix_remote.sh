#!/usr/bin/env bash
# Remote-only A6ema matrix runner for c240g5-110223.wisc.cloudlab.us.
# Uses the existing dataset in place; does not generate, copy, or modify data.
set -euo pipefail
cd /users/qinglong/transformer
export OMP_NUM_THREADS=1
PYTHON=/users/qinglong/venv/bin/python

for s in 1 2 3; do
  "$PYTHON" -u -m tools.compare \
    --experiments A6ema A6ema_nophase A6ema_constbasis A6ema_nophase_constbasis \
      A6_noema A6ema_fno_baseline A6ema_fno_film A6ema_fno_attention A6ema_unet \
    --epochs 24 --batch-size 16 --num-workers 2 --device cuda \
    --seed 42 --train-seed "$s" \
    --runs-dir "runs/a6ema_matrix_seed${s}"
done
