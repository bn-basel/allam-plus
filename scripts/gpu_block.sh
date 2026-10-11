#!/bin/sh
# Run a list of GPU steps for at most a set time, then stop. Every step resumes where it left off, so a block can be
# re-run later to continue.   Usage: sh scripts/gpu_block.sh <hours> <steps file>
# Stops early (before starting the next step) at the deadline or when notes/STOP exists (scripts/stop_gpu.ps1).
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
DEADLINE=$(awk -v h="$1" -v n="$(date +%s)" 'BEGIN { printf "%d", n + h * 3600 }')
LOG=notes/logs/gpu_blocks.log
echo "=== $(date '+%m-%d %H:%M') block start: $2, $1 h" >> $LOG
while IFS= read -r step || [ -n "$step" ]; do
  case "$step" in ''|\#*) continue ;; esac
  if [ -f notes/STOP ]; then echo "=== $(date +%H:%M) STOP file: block ended" >> $LOG; mv -f notes/STOP notes/STOP.seen; exit 0; fi
  if [ $(date +%s) -ge $DEADLINE ]; then echo "=== $(date +%H:%M) deadline reached: block ended" >> $LOG; exit 0; fi
  echo "=== $(date +%H:%M) $step" >> $LOG
  sh -c "$step" 2>&1 | grep -v -i "triton\|UserWarning\|warnings.warn" >> $LOG
done < "$2"
echo "=== $(date +%H:%M) block done: $2" >> $LOG
