#!/bin/sh
# Phase 4: test a release GGUF the way apps use it: llama-server with only the file (template and 16K inside), the
# OpenAI chat API with "tools", standard tool_calls back. Safe to re-run: finished benchmarks are skipped.
#   sh scripts/phase4_serve_tests.sh Q8_0 [release folder]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
Q=${1:-Q8_0}
DIR=${2:-C:/WIP/release-work/allam-plus-tools_v1/release}
GGUF=$DIR/allam-plus-7b-$Q.gguf
NAME=$(basename "$(dirname "$DIR")" | sed 's/allam-plus-//')_$(echo "$Q" | tr 'A-Z' 'a-z')
URL=http://127.0.0.1:8095
[ -f "$GGUF" ] || { echo "missing $GGUF (run scripts/finalize_gguf.py first)"; exit 1; }
C:/WIP/tools/llama.cpp/bin/llama-server.exe -m "$GGUF" -ngl 99 -c 16384 -np 1 -fa on --jinja --port 8095 \
  > notes/logs/llama_server_$NAME.log 2>&1 &
SPID=$!
until curl -sf $URL/health > /dev/null; do sleep 2; kill -0 $SPID 2>/dev/null || { echo "server did not start"; exit 1; }; done
echo "--- $(date +%H:%M) $NAME: server up"
grep -E "n_ctx_train|rope scaling|freq_scale_train|n_ctx_orig_yarn|add_space_prefix|n_ctx " notes/logs/llama_server_$NAME.log | head -12
for cmd in \
  "scripts/check_gguf_template.py --url $URL" \
  "scripts/check_gguf_template.py --url $URL --string-args" \
  "scripts/tools_bench.py --bench arabfuncbench --protocol served --server $URL --name $NAME" \
  "scripts/tools_bench.py --bench aisa_dev --protocol served --server $URL --name $NAME" \
  "scripts/receptionist_eval.py --server $URL --name ${NAME}_served" \
  "scripts/needle.py run --rope yarn4 --server $URL --name ${NAME}_served" \
  "demo/agent.py --server $URL --tasks --name $NAME"; do
  [ -f notes/STOP ] && break
  echo "--- $(date +%H:%M) $cmd"; uv run python $cmd 2>&1 | tail -25
done
kill $SPID; wait $SPID 2>/dev/null
echo "--- $(date +%H:%M) $NAME: done"
