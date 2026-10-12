#!/bin/sh
# Fix round, after training: dev-score the candidates of tools_v3 on dev_v3, pick by rule D23 (or keep tools_v2),
# then the deciding tests first: release GGUF + the 90-task agent test, then the guards; compare with tools_v2
# (scripts/compare_runs.py --rule d23). Safe to re-run: finished steps are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
SERVER=C:/WIP/tools/llama.cpp/bin/llama-server.exe
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096"
LAST=$(ls -d runs/tools_v3/checkpoint-* | sed 's/.*-//' | sort -n | tail -1)
for s in 60 120 180 $LAST; do
  [ -f notes/STOP ] && exit 0
  [ -d runs/tools_v3/checkpoint-$s ] && [ ! -f results/tool_dev_v3/tools_v3__checkpoint-$s.json ] && \
    uv run python scripts/tool_dev.py --dev v3 --adapter runs/tools_v3/checkpoint-$s
done
uv run python scripts/tool_dev.py --dev v3 --report tools_v3 --rule fix --ref tools_v2__checkpoint-180
PICK=$(uv run python scripts/tool_dev.py --dev v3 --report tools_v3 --rule fix --ref tools_v2__checkpoint-180 2>/dev/null \
       | grep "chosen by the rule" | sed 's/.*: //')
case "$PICK" in
  tools_v3__*) ;;
  *) echo "no tools_v3 checkpoint beats tools_v2 on the new dev turns without losing others: tools_v2 stays"
     exit 0 ;;
esac
C=runs/$(echo "$PICK" | sed 's|__|/|')
echo "tools_v3 tests on $C"
W=C:/WIP/release-work/allam-plus-tools_v3
[ -f $W/gguf/manifest.json ] || uv run python scripts/export_gguf.py --model allam --adapter $C --name allam-plus-tools_v3
[ -f $W/release/manifest.json ] || uv run python scripts/finalize_gguf.py --src $W/gguf --name allam-plus-7b
[ -f notes/STOP ] && exit 0
sh scripts/agent_test_served.sh tools_v3
uv run python scripts/agent_bench.py --compare tools_v2_q4 tools_v3_q4
for cmd in \
  "scripts/receptionist_eval.py --name tools_v3_native --rope yarn4 --adapter $C" \
  "scripts/tools_bench.py --bench arabfuncbench --protocol native --rope yarn4 --adapter $C --name tools_v3" \
  "scripts/tools_bench.py --bench aisa_dev --protocol native --rope yarn4 --adapter $C --name tools_v3" \
  "scripts/arabicmmlu.py --rope yarn4 --adapter $C --name tools_v3" \
  "scripts/needle.py run --rope yarn4 --adapter $C --name tools_v3" \
  "scripts/compare_runs.py --a tools_v2 --b tools_v3 --rule d23" \
  "scripts/lceval.py --rope yarn4 --adapter $C --name tools_v3"; do
  [ -f notes/STOP ] && exit 0
  echo "--- $(date +%H:%M) $cmd"; uv run python $cmd
done
Q8=$W/gguf/allam-plus-tools_v3-Q8_0.gguf
if [ ! -f results/ifeval/tools_v3_q8_yarn4/summary.json ] && [ -f $Q8 ]; then
  "$SERVER" -m $Q8 -ngl 99 -c 4096 -np 1 --port 8095 $YARN > notes/logs/llama_server.log 2>&1 &
  SPID=$!
  until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; done
  uv run python scripts/ifeval.py --server http://127.0.0.1:8095 --name tools_v3_q8_yarn4
  kill $SPID; wait $SPID 2>/dev/null
fi
uv run python scripts/compare_runs.py --a tools_v2 --b tools_v3 --rule d23
echo "tools_v3 tests done"
