#!/bin/sh
# Run the 90-task agent test (scripts/agent_bench.py) on a run's release Q4_K_M GGUF, served by llama-server.
#   sh scripts/agent_test_served.sh tools_v2
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
RUN=$1
GGUF=C:/WIP/release-work/allam-plus-$RUN/release/allam-plus-7b-Q4_K_M.gguf
[ -f "$GGUF" ] || { echo "missing $GGUF"; exit 1; }
C:/WIP/tools/llama.cpp/bin/llama-server.exe -m "$GGUF" -ngl 99 -c 16384 -np 1 -fa on --jinja --port 8095 \
  > notes/logs/llama_server_agent_$RUN.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; kill -0 $SPID 2>/dev/null || { echo "server did not start"; exit 1; }; done
uv run python scripts/agent_bench.py --server http://127.0.0.1:8095 --name ${RUN}_q4 2>&1 | tail -3
kill $SPID; wait $SPID 2>/dev/null
