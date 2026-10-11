#!/bin/sh
# "Before" numbers in the allam-plus tool format (native), for a like-for-like comparison after Phase 3.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
LONG=runs/long_v1/checkpoint-183
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
until grep -q "phase3 baselines done" notes/logs/phase3.log; do sleep 30; done
{
run scripts/tools_bench.py --bench arabfuncbench --protocol native
run scripts/tools_bench.py --bench aisa_dev --protocol native
run scripts/tools_bench.py --bench arabfuncbench --protocol native --rope yarn4 --adapter $LONG --name long_v1
run scripts/tools_bench.py --bench aisa_dev --protocol native --rope yarn4 --adapter $LONG --name long_v1
run scripts/receptionist_eval.py --name base_native
run scripts/receptionist_eval.py --name long_v1_native --rope yarn4 --adapter $LONG
echo "=== $(date +%H:%M) phase3 baselines2 done"
} >> notes/logs/phase3.log
