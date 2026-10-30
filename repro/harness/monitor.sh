#!/usr/bin/env bash
# Exit (to wake the agent) on: scheduler not active, CPU mode not ok, new anomaly statuses,
# low memory, or after MAX_S seconds (heartbeat). Prints a summary either way.
# Usage: repro/harness/monitor.sh [max_seconds]
R=/home/cc/mingkuan/qalm/repro/runs
MAX_S=${1:-7200}
start=$(date +%s)
base_bad=$(grep -h -o -E '"status": "(INFRA_GAVE_UP|COLLECT_ERROR|CRASH)"' $R/jobs/*/done.json 2>/dev/null | wc -l)
reason=heartbeat
while true; do
  sleep 60
  systemctl is-active -q qalmrepro-sched || { reason="scheduler not active"; break; }
  python3 - "$R" <<'PY' || { reason="status anomaly"; break; }
import json, sys
s = json.load(open(sys.argv[1] + "/status.json"))
sys.exit(0 if s["cpu_mode_ok"] and s["mem_available_gib"] > 15 else 1)
PY
  bad=$(grep -h -o -E '"status": "(INFRA_GAVE_UP|COLLECT_ERROR|CRASH)"' $R/jobs/*/done.json 2>/dev/null | wc -l)
  [ "$bad" -gt "$base_bad" ] && { reason="new INFRA_GAVE_UP/COLLECT_ERROR/CRASH ($bad)"; break; }
  [ $(( $(date +%s) - start )) -ge $MAX_S ] && break
done
echo "monitor exit: $reason at $(date -u +%FT%TZ)"
cat $R/status.json; echo
REPRO_RUNS_DIR=$R python3 /home/cc/mingkuan/qalm/repro/harness/run.py status
tail -5 $R/sched.log
