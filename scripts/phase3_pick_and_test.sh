#!/bin/sh
# Dev-score the candidate checkpoints of tools_v1 (every 100th step and the last), pick by rule D16, test the pick.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
SERVER=C:/WIP/tools/llama.cpp/bin/llama-server.exe
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096"
LAST=$(ls -d runs/tools_v1/checkpoint-* | sed 's/.*-//' | sort -n | tail -1)
for s in 100 200 300 400 500 $LAST; do
  [ -f notes/STOP ] && exit 0
  [ -d runs/tools_v1/checkpoint-$s ] && [ ! -f results/tool_dev/tools_v1__checkpoint-$s.json ] && \
    uv run python scripts/tool_dev.py --adapter runs/tools_v1/checkpoint-$s
done
uv run python scripts/tool_dev.py --report tools_v1
N=$(uv run python scripts/tool_dev.py --report tools_v1 2>/dev/null | grep "chosen by the rule" | sed 's/.*checkpoint-//')
C=runs/tools_v1/checkpoint-$N
echo "phase 3 tests on $C"
for cmd in \
  "scripts/tools_bench.py --bench arabfuncbench --protocol native --rope yarn4 --adapter $C --name tools_v1" \
  "scripts/tools_bench.py --bench aisa_dev --protocol native --rope yarn4 --adapter $C --name tools_v1" \
  "scripts/receptionist_eval.py --name tools_v1_native --rope yarn4 --adapter $C" \
  "scripts/needle.py run --rope yarn4 --adapter $C --name tools_v1" \
  "scripts/needle.py report" \
  "scripts/lceval.py --rope yarn4 --adapter $C --name tools_v1" \
  "scripts/arabicmmlu.py --rope yarn4 --adapter $C --name tools_v1" \
  "scripts/export_gguf.py --model allam --adapter $C --name allam-plus-tools_v1"; do
  [ -f notes/STOP ] && exit 0
  echo "--- $(date +%H:%M) $cmd"; uv run python $cmd
done
Q8=C:/WIP/release-work/allam-plus-tools_v1/gguf/allam-plus-tools_v1-Q8_0.gguf
"$SERVER" -m $Q8 -ngl 99 -c 4096 -np 1 --port 8095 $YARN > notes/logs/llama_server.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; done
uv run python scripts/ifeval.py --server http://127.0.0.1:8095 --name tools_v1_q8_yarn4
kill $SPID; wait $SPID 2>/dev/null
echo "phase 3 tests done"
