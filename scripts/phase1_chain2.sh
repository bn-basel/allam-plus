#!/bin/sh
# Phase 1, remaining GPU runs (after the IFEval switch to llama-server). Logs in notes/logs/phase1_chain.log.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
REC=C:/WIP/projects/saudi-dialect/runs/allam_v1/checkpoint-414
BASE_Q8=C:/WIP/release-work/allam-7b-instruct-base/gguf/allam-7b-instruct-base-Q8_0.gguf
BASE_Q4=C:/WIP/release-work/allam-7b-instruct-base/gguf/allam-7b-instruct-base-Q4_K_M.gguf
REC_Q8=C:/WIP/release-work/allam-7b-saudi-receptionist/gguf/allam-7b-saudi-receptionist-Q8_0.gguf
SERVER=C:/WIP/tools/llama.cpp/bin/llama-server.exe
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096"
run() { echo "=== $(date +%H:%M) $*"; uv run python "$@" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn"; }
serve() {  # serve <gguf> [extra flags...]: start llama-server on 8095 and wait until it answers
  "$SERVER" -m "$@" -ngl 99 -c 4096 -np 1 --port 8095 > notes/logs/llama_server.log 2>&1 &
  SPID=$!
  until curl -sf http://127.0.0.1:8095/health > /dev/null; do sleep 2; done
  echo "=== $(date +%H:%M) server up: $*"
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3; }
{
run scripts/arabicmmlu.py --rope shipped
run scripts/arabicmmlu.py --rope shipped --adapter $REC --name receptionist
serve $BASE_Q8;          run scripts/ifeval.py --server http://127.0.0.1:8095 --name base_q8;          stop
serve $BASE_Q8 $YARN;    run scripts/ifeval.py --server http://127.0.0.1:8095 --name base_q8_yarn4;    stop
serve $REC_Q8;           run scripts/ifeval.py --server http://127.0.0.1:8095 --name receptionist_q8;  stop
run scripts/train_probe.py --lengths 4096 8192 12288 16384
run scripts/train_probe.py --lengths 8192 12288 16384 --chunked-loss
run scripts/speed_memory.py --gguf $BASE_Q4
run scripts/speed_memory.py --gguf $BASE_Q8 --kv f16 q8_0
echo "=== $(date +%H:%M) chain2 done"
} >> notes/logs/phase1_chain.log
