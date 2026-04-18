#!/bin/bash
# After benchmark_guoq.py (advk, PID 261805) finishes:
#   1. In parallel:
#      (a) Re-run base+advk OOM circuits sequentially at 32GB
#      (b) Run original Quartz on 250 GUOQ circuits (32 workers × 8GB)
#   2. When Quartz finishes, re-run its OOM circuits sequentially at 32GB.
set -u

TARGET_PID=261805
cd /home/cc/mingkuan/qalm

echo "[$(date)] Waiting for benchmark_guoq.py (PID $TARGET_PID) to finish..."
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] benchmark_guoq.py finished."

# ── Stage 1: launch both in parallel ─────────────────────────────────────────
echo "[$(date)] Launching base+advk OOM re-run (32GB, 1 worker) in background..."
nohup python3 rerun_oom_circuits.py > rerun_oom_circuits.log 2>&1 &
OOM_PID=$!
echo "[$(date)]   OOM re-run PID: $OOM_PID"

echo "[$(date)] Launching benchmark_quartz_guoq.py (8GB, 32 workers) in background..."
nohup bash -c 'PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_quartz_guoq.py' \
    > benchmark_quartz_guoq.log 2>&1 &
QUARTZ_PID=$!
echo "[$(date)]   Quartz benchmark PID: $QUARTZ_PID"

# ── Stage 2: wait for Quartz, then re-run its OOM circuits ───────────────────
echo "[$(date)] Waiting for Quartz benchmark (PID $QUARTZ_PID) to finish..."
wait "$QUARTZ_PID"
echo "[$(date)] Quartz benchmark finished. Starting Quartz OOM re-run (32GB)..."

# If the base+advk OOM re-run is still going, this will contend for the
# single-worker 32GB slot but run sequentially inside the script, which is fine.
nohup python3 rerun_oom_circuits.py --quartz-only >> rerun_oom_circuits.log 2>&1 &
Q_OOM_PID=$!
echo "[$(date)]   Quartz OOM re-run PID: $Q_OOM_PID"

# Wait for both OOM re-runs to finish before exiting.
wait "$OOM_PID" 2>/dev/null || true
wait "$Q_OOM_PID" 2>/dev/null || true

echo "[$(date)] All done. See rerun_oom_circuits.log and benchmark_quartz_guoq.log."
