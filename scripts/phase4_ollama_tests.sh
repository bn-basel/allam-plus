#!/bin/sh
# Phase 4: the Q4_K_M release through Ollama (portable, C:\WIP\tools\ollama), with the Modelfile in release/.
# Ollama 0.40 serves through llama-server with the GGUF's own chat template, so the scores should match the
# llama-server ones. Then the same with the Go-template Modelfile (OLLAMA_GO_TEMPLATE=1), for older Ollama versions.
#   sh scripts/phase4_ollama_tests.sh [release folder]
# The two test models stay in Ollama's store (C:/WIP/tools/ollama/models, about 4.3 GB as one shared blob); remove
# them yourself with: ollama rm allam-plus-jinja allam-plus-go
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
DIR=${1:-C:/WIP/release-work/allam-plus-tools_v1/release}
NAME=$(basename "$(dirname "$DIR")" | sed 's/allam-plus-//')
OLLAMA=C:/WIP/tools/ollama/ollama.exe
export OLLAMA_MODELS='C:\WIP\tools\ollama\models' OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1
URL=http://127.0.0.1:11434
for mode in jinja go; do
  [ -f notes/STOP ] && break
  if [ $mode = go ]; then export OLLAMA_GO_TEMPLATE=1; MF=Modelfile.go-template; else unset OLLAMA_GO_TEMPLATE; MF=Modelfile; fi
  "$OLLAMA" serve > notes/logs/ollama_$mode.log 2>&1 &
  OPID=$!
  until curl -sf $URL/api/version > /dev/null; do sleep 2; done
  cp release/$MF "$DIR/$MF"
  (cd "$DIR" && "$OLLAMA" create allam-plus-$mode -f $MF)
  echo "--- $(date +%H:%M) ollama $mode"
  for cmd in \
    "scripts/receptionist_eval.py --server $URL --server-model allam-plus-$mode --name ${NAME}_q4_ollama_$mode" \
    "scripts/tools_bench.py --bench aisa_dev --protocol served --server $URL --server-model allam-plus-$mode --name ${NAME}_q4_ollama_$mode"; do
    [ -f notes/STOP ] && break
    echo "--- $(date +%H:%M) $cmd"; uv run python $cmd 2>&1 | tail -20
  done
  kill $OPID; wait $OPID 2>/dev/null
  taskkill //F //IM ollama.exe > /dev/null 2>&1
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
done
echo "--- $(date +%H:%M) ollama tests done"
