#!/bin/sh
# Tool-calling starting points before Phase 3: ArabFuncBench (now downloadable) for every version, AISA for the
# Phase 2 model (the model Phase 3 continues from).
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
REC=C:/WIP/projects/saudi-dialect/runs/allam_v1/checkpoint-414
LONG=runs/long_v1/checkpoint-183
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
{
run scripts/tools_bench.py --bench arabfuncbench --protocol paper
run scripts/tools_bench.py --bench arabfuncbench --protocol text
run scripts/tools_bench.py --bench arabfuncbench --protocol text --adapter $REC --name receptionist
run scripts/tools_bench.py --bench arabfuncbench --protocol text --rope yarn4 --adapter $LONG --name long_v1
run scripts/tools_bench.py --bench aisa_dev --protocol text --rope yarn4 --adapter $LONG --name long_v1
echo "=== $(date +%H:%M) phase3 baselines done"
} >> notes/logs/phase3.log
