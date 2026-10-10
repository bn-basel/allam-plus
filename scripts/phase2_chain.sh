#!/bin/sh
# Phase 2: starting-point dev score, the long-context run, dev scores of the candidate checkpoints.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
{
run scripts/long_dev.py --base
run scripts/train_long.py --run long_v1
for s in 40 80 120 160 183; do
  [ -d runs/long_v1/checkpoint-$s ] && run scripts/long_dev.py --adapter runs/long_v1/checkpoint-$s
done
run scripts/long_dev.py --report long_v1
echo "=== $(date +%H:%M) phase2 chain done"
} >> notes/logs/phase2_chain.log
