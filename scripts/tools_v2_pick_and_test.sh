#!/bin/sh
# Receptionist tuning round, after training: dev-score the candidates of tools_v2, pick by rule D19 (or keep tools_v1),
# then test the pick, the deciding tests first, and compare with tools_v1 (scripts/compare_runs.py). Safe to re-run.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
SERVER=C:/WIP/tools/llama.cpp/bin/llama-server.exe
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096"
LAST=$(ls -d runs/tools_v2/checkpoint-* | sed 's/.*-//' | sort -n | tail -1)
for s in 60 120 180 $LAST; do
  [ -f notes/STOP ] && exit 0
  [ -d runs/tools_v2/checkpoint-$s ] && [ ! -f results/tool_dev/tools_v2__checkpoint-$s.json ] && \
    uv run python scripts/tool_dev.py --adapter runs/tools_v2/checkpoint-$s
done
uv run python scripts/tool_dev.py --report tools_v2 --rule receptionist --ref tools_v1__checkpoint-400
PICK=$(uv run python scripts/tool_dev.py --report tools_v2 --rule receptionist --ref tools_v1__checkpoint-400 2>/dev/null \
       | grep "chosen by the rule" | sed 's/.*: //')
case "$PICK" in
  tools_v2__*) ;;
  *) echo "no tools_v2 checkpoint beats tools_v1 on the receptionist dev turns without losing general ones: tools_v1 stays"
     exit 0 ;;
esac
C=runs/$(echo "$PICK" | sed 's|__|/|')
echo "tools_v2 tests on $C"
for cmd in \
  "scripts/receptionist_eval.py --name tools_v2_native --rope yarn4 --adapter $C" \
  "scripts/tools_bench.py --bench arabfuncbench --protocol native --rope yarn4 --adapter $C --name tools_v2" \
  "scripts/tools_bench.py --bench aisa_dev --protocol native --rope yarn4 --adapter $C --name tools_v2" \
  "scripts/arabicmmlu.py --rope yarn4 --adapter $C --name tools_v2" \
  "scripts/needle.py run --rope yarn4 --adapter $C --name tools_v2" \
  "scripts/compare_runs.py --a tools_v1 --b tools_v2" \
  "scripts/lceval.py --rope yarn4 --adapter $C --name tools_v2" \
  "scripts/export_gguf.py --model allam --adapter $C --name allam-plus-tools_v2"; do
  [ -f notes/STOP ] && exit 0
  echo "--- $(date +%H:%M) $cmd"; uv run python $cmd
done
Q8=C:/WIP/release-work/allam-plus-tools_v2/gguf/allam-plus-tools_v2-Q8_0.gguf
if [ ! -f results/ifeval/tools_v2_q8_yarn4/summary.json ] && [ -f $Q8 ]; then
  "$SERVER" -m $Q8 -ngl 99 -c 4096 -np 1 --port 8095 $YARN > notes/logs/llama_server.log 2>&1 &
  SPID=$!
  until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; done
  uv run python scripts/ifeval.py --server http://127.0.0.1:8095 --name tools_v2_q8_yarn4
  kill $SPID; wait $SPID 2>/dev/null
fi
uv run python scripts/compare_runs.py --a tools_v1 --b tools_v2
echo "tools_v2 tests done"
