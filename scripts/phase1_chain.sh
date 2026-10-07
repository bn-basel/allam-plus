#!/bin/sh
# Phase 1 GPU runs, one after another (each script resumes if interrupted). Logs in notes/logs/.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
REC=C:/WIP/projects/saudi-dialect/runs/allam_v1/checkpoint-414
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
{
run scripts/needle.py run --rope shipped
run scripts/needle.py run --rope yarn4
run scripts/needle.py report
run scripts/arabicmmlu.py --rope shipped
run scripts/arabicmmlu.py --rope yarn4
run scripts/tools_bench.py --bench aisa_dev --protocol text
run scripts/tools_bench.py --bench aisa_dev --protocol text --adapter $REC --name receptionist
run scripts/lceval.py --rope shipped
run scripts/lceval.py --rope yarn4
run scripts/ifeval.py --rope shipped
run scripts/ifeval.py --rope yarn4
run scripts/arabicmmlu.py --rope shipped --adapter $REC --name receptionist
run scripts/ifeval.py --rope shipped --adapter $REC --name receptionist
run scripts/train_probe.py --lengths 4096 8192 12288 16384
run scripts/train_probe.py --lengths 8192 12288 16384 --chunked-loss
echo "=== $(date +%H:%M) chain done"
} >> notes/logs/phase1_chain.log
