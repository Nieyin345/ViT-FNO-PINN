#!/usr/bin/env bash
# Remote-only worker: run the full A6ema comparison matrix for one training seed.
# Launch separate seeds in parallel; each seed has its own result root.
set -euo pipefail
case "${1:-}" in
  1|2|3) seed="$1" ;;
  *) echo "usage: $0 {1|2|3}" >&2; exit 2 ;;
esac
cd /users/qinglong/transformer
export OMP_NUM_THREADS=1
exec /users/qinglong/venv/bin/python -u -m tools.compare \
  --experiments A6ema A6ema_nophase A6ema_constbasis A6ema_nophase_constbasis \
    A6_noema A6ema_fno_baseline A6ema_fno_film A6ema_fno_attention A6ema_unet \
  --epochs 24 --batch-size 16 --num-workers 2 --device cuda \
  --seed 42 --train-seed "$seed" \
  --runs-dir "runs/a6ema_matrix_seed${seed}"
