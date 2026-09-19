#!/usr/bin/env bash
# Detached launch for Grouped SpectralMR 50-epoch controlled run.
set -euo pipefail
cd "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal"

RUN_NAME="DBCR_SpectralMR_grouped_seed42_epochs50_20260918"
PY="${PY:-/aul/homes/mmazi007/.conda/envs/pylrt/bin/python}"
LOG="outputs/${RUN_NAME}.log"
PIDFILE="outputs/${RUN_NAME}.pid"
GPU="${GPU:-2}"

# Stop prior instance via pidfile only (avoid pkill -f self-match).
if [[ -f "$PIDFILE" ]]; then
  oldpid="$(cat "$PIDFILE" || true)"
  if [[ -n "${oldpid}" ]] && kill -0 "$oldpid" 2>/dev/null; then
    kill "$oldpid" 2>/dev/null || true
    sleep 2
    kill -9 "$oldpid" 2>/dev/null || true
  fi
fi

rm -rf "outputs/${RUN_NAME}"
rm -f "$LOG" "$PIDFILE"

setsid nohup "$PY" -u -m src.train \
  --gpu "$GPU" \
  --config configs/dbcr_spectral_mr_grouped.json \
  --seed 42 \
  --epochs 50 \
  --run_name "$RUN_NAME" \
  --skip_test \
  --val_endpoint \
  --num_workers 4 \
  >"$LOG" 2>&1 < /dev/null &

echo $! >"$PIDFILE"
disown $! 2>/dev/null || true
echo "started_pid=$(cat "$PIDFILE") gpu=$GPU py=$PY"
sleep 2
ps -o pid,ppid,sid,stat,etime,cmd -p "$(cat "$PIDFILE")" || true
