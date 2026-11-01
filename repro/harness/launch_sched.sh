#!/usr/bin/env bash
# Start the scheduler as a system unit (survives SSH logout; restarted if it crashes). Usage:
#   repro/harness/launch_sched.sh <plan.jsonl> [runs_dir] [extra run.py launch args...]
# Before starting it (re)installs the campaign slice and keeps every other process
# (user sessions, system services, docker daemon, the scheduler) on the housekeeping CPUs,
# so that job cores and their SMT siblings run nothing but the job.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv/bin/python   # the uv environment (uv sync)
[ -x "$PY" ] || { echo "no $PY: run 'uv sync' in $REPO first" >&2; exit 1; }
PLAN=$(realpath "$1"); shift
RUNS=${1:-$REPO/repro/runs}; shift || true
mkdir -p "$RUNS"
HK=$("$PY" -c "import yaml;print(yaml.safe_load(open('$REPO/repro/config/campaign.yaml'))['cores']['housekeeping'])")
sudo -n install -m644 "$REPO/repro/env/qalmrepro.slice" /run/systemd/system/qalmrepro.slice
sudo -n systemctl daemon-reload
# never stop or restart the slice here: running jobs live in it and must survive a scheduler restart
sudo -n systemctl start qalmrepro.slice
if [ "${REPRO_ALLOW_CPU_MODE:-0}" != "1" ]; then
  for u in user.slice system.slice init.scope; do
    sudo -n systemctl set-property --runtime "$u" AllowedCPUs="$HK"
  done
fi
sudo -n systemctl reset-failed qalmrepro-sched.service 2>/dev/null || true
sudo -n systemd-run --unit=qalmrepro-sched --uid="$(id -un)" --gid="$(id -gn)" --quiet \
  --working-directory=$REPO -p AllowedCPUs="$HK" -p OOMScoreAdjust=-900 \
  -p Restart=on-failure -p RestartSec=30 \
  -E REPRO_RUNS_DIR="$RUNS" -E HOME="$HOME" \
  ${REPRO_CAP_TIERS_GIB:+-E REPRO_CAP_TIERS_GIB=$REPRO_CAP_TIERS_GIB} ${REPRO_PRIOR_SCALE:+-E REPRO_PRIOR_SCALE=$REPRO_PRIOR_SCALE} \
  ${REPRO_ALLOW_CPU_MODE:+-E REPRO_ALLOW_CPU_MODE=$REPRO_ALLOW_CPU_MODE} \
  -p StandardOutput=append:"$RUNS"/sched.log -p StandardError=append:"$RUNS"/sched.log \
  "$PY" $REPO/repro/harness/run.py launch --plan "$PLAN" "$@"
echo "scheduler started; log: $RUNS/sched.log"
