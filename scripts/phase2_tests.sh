#!/bin/sh
# Phase 2 tests on the checkpoint chosen by the dev rule.  Usage: sh scripts/phase2_tests.sh <step>
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
C=runs/long_v1/checkpoint-$1
NAME=long_v1
SERVER=C:/WIP/tools/llama.cpp/bin/llama-server.exe
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096"
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
{
echo "=== $(date +%H:%M) phase 2 tests on $C"
run scripts/needle.py run --rope yarn4 --adapter $C --name $NAME
run scripts/needle.py report
run scripts/lceval.py --rope yarn4 --adapter $C --name $NAME
run scripts/arabicmmlu.py --rope yarn4 --adapter $C --name $NAME
run scripts/arabicmmlu.py --rope shipped --adapter $C --name ${NAME}__shipped_rope
run scripts/export_gguf.py --model allam --adapter $C --name allam-plus-$NAME
Q8=C:/WIP/release-work/allam-plus-$NAME/gguf/allam-plus-$NAME-Q8_0.gguf
"$SERVER" -m $Q8 -ngl 99 -c 4096 -np 1 --port 8095 $YARN > notes/logs/llama_server.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; done
run scripts/ifeval.py --server http://127.0.0.1:8095 --name ${NAME}_q8_yarn4
kill $SPID; wait $SPID 2>/dev/null
echo "=== $(date +%H:%M) phase2 tests done"
} >> notes/logs/phase2_chain.log
