#!/usr/bin/env bash
# Live test of cpuwatch.py on the server. Run it ONLY while no experiments run (it refuses
# otherwise). Takes about 2 minutes. It starts short-lived test loads (busy loops, 5 GiB
# allocations) in cgroups that it stops again, runs a test copy of the watcher as the unit
# "cwtest" (scan every 5 s, watchdog 20 s) logging to /tmp/cwtest.log, and prints PASS/FAIL.
#   bash cwtest.sh            (expects ~/awagner/cpuwatch.py and an active qalmrepro.slice)
# Overrides for other machines: W (watcher path), SUDO, SCOPE (--user), JOBARGS (--job-cores ..),
# ONJOB_CPU / SIB_CPU / INJOB_CPU / INJOB_CPU2, START_PATTERN, OUTSIDE_CG, ALLOC_GIB.
set -u
W=${W:-$HOME/awagner/cpuwatch.py}
SUDO=${SUDO-sudo}
SCOPE=${SCOPE:-}
JOBARGS=${JOBARGS:-}
ONJOB_CPU=${ONJOB_CPU:-10}      # a job core
SIB_CPU=${SIB_CPU:-130}         # the idle SMT sibling of job core 2
INJOB_CPU=${INJOB_CPU:-11}      # job cores used by the fake "jobs" inside qalmrepro.slice
INJOB_CPU2=${INJOB_CPU2:-12}
START_PATTERN=${START_PATTERN:-"job cores 2-63,66-127; idle siblings 130-191,194-255"}
OUTSIDE_CG=${OUTSIDE_CG:-/user.slice}   # cgroup prefix of processes started from this shell
ALLOC_GIB=${ALLOC_GIB:-5}               # size of each test allocation (watcher threshold: 4 GiB)
LOG=/tmp/cwtest.log
SD="$SUDO systemd-run $SCOPE --quiet"
SC="$SUDO systemctl $SCOPE"
Q="systemctl $SCOPE"
UNITS="cwt-onjob cwt-sib cwt-injob cwt-injobmem"

if [ -z "$SCOPE" ]; then
  if $Q is-active -q qalmrepro-sched || \
     [ "$(find /sys/fs/cgroup/qalmrepro.slice -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" != 0 ]; then
    echo "experiments are running (scheduler active or jobs in qalmrepro.slice): not testing"; exit 1
  fi
  $Q is-active -q qalmrepro.slice || { echo "qalmrepro.slice is not active"; exit 1; }
fi
[ -f "$W" ] || { echo "no watcher at $W"; exit 1; }
echo "watcher $(sha256sum "$W" | cut -c1-16); this takes about 2 minutes"

cleanup() {
  $SC stop cwtest $UNITS 2>/dev/null; $SC reset-failed cwtest $UNITS 2>/dev/null
  $SC stop cwtest.slice 2>/dev/null
}
cleanup
rm -f "$LOG"
pass=0; fail=0
check() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1)); else echo "FAIL  $1"; fail=$((fail+1)); fi; }

# low-memory threshold one allocation below what is free now; the two test allocations cross it
LOW=$(awk -v a="$ALLOC_GIB" '/MemAvailable/{printf "%d", $2/1048576 - a}' /proc/meminfo)
$SD --unit=cwtest -p Restart=on-failure -p RestartSec=3 -p WatchdogSec=20 -p TimeoutStopSec=5 \
  /usr/bin/python3 -I "$W" --log "$LOG" --interval 5 --low-mem-gib "$LOW" $JOBARGS
sleep 4

echo "test loads for 15 s ..."
LOOP='while :; do :; done'
ALLOC="import time; a = b'x' * ($ALLOC_GIB << 30); time.sleep(13)"
timeout 15 sh -c "$LOOP" &                                    # busy, outside the slice, housekeeping CPUs
timeout 15 /usr/bin/python3 -c "$ALLOC" &                     # memory, outside the slice
$SD --unit=cwt-onjob --slice=cwtest.slice -p AllowedCPUs=$ONJOB_CPU -p RuntimeMaxSec=15 sh -c "$LOOP"
$SD --unit=cwt-sib --slice=cwtest.slice -p AllowedCPUs=$SIB_CPU -p RuntimeMaxSec=15 sh -c "$LOOP"
$SD --unit=cwt-injob --slice=qalmrepro.slice -p AllowedCPUs=$INJOB_CPU -p RuntimeMaxSec=15 sh -c "$LOOP"
$SD --unit=cwt-injobmem --slice=qalmrepro.slice -p AllowedCPUs=$INJOB_CPU2 -p RuntimeMaxSec=15 \
  /usr/bin/python3 -c "$ALLOC"
wait
sleep 12   # two more scans after the loads: memory recovers

echo "watchdog: freezing the watcher (simulated hang) ..."
P=$($Q show cwtest -p MainPID --value)
$SUDO kill -STOP "$P"
sleep 35
R1=$($Q show cwtest -p NRestarts --value)
echo "crash: kill -9 ..."
P=$($Q show cwtest -p MainPID --value)
$SUDO kill -9 "$P"
sleep 8
R2=$($Q show cwtest -p NRestarts --value)
$SC stop cwtest; sleep 3
ST=$($Q is-active cwtest)
WD=$(journalctl $SCOPE -u cwtest --since "-5min" --no-pager 2>/dev/null | grep -c "Watchdog timeout")

echo
check "START: found the job cores and idle siblings"         'grep -q "$START_PATTERN" "$LOG"'
check "onjob: unconfined process allowed on job core"         'grep -q "WARN onjob .*cwt-onjob.service.*job cores $ONJOB_CPU\$" "$LOG"'
check "sibling: idle SMT sibling busy"                        'grep -q "WARN sibling cpu$SIB_CPU " "$LOG"'
check "busy: process outside the slice above 20%"             'grep -q "WARN busy .*cgroup $OUTSIDE_CG" "$LOG"'
check "mem: big process outside the slice"                    'grep -q "WARN mem .*cgroup $OUTSIDE_CG.*resident $ALLOC_GIB.0 GiB" "$LOG"'
check "lowmem: warning and recovery"                          'grep -q "WARN lowmem" "$LOG" && grep -q "INFO memok" "$LOG"'
check "fake jobs in qalmrepro.slice ignored (CPU and memory)" '! grep -q "qalmrepro.slice" "$LOG"'
check "hang: watchdog killed and restarted it"                '[ "${R1:-0}" -ge 1 ] && [ "$WD" -ge 1 ]'
check "crash: restarted after kill -9"                        '[ "${R2:-0}" -gt "${R1:-0}" ]'
check "stop: stays stopped"                                   '[ "$ST" != active ] && [ "$ST" != activating ]'
check "a START line for every (re)start"                      '[ "$(grep -c START "$LOG")" -eq $(( ${R2:-0} + 1 )) ]'
cleanup
if [ -z "$SCOPE" ]; then
  check "qalmrepro.slice empty again"                         '[ "$(find /sys/fs/cgroup/qalmrepro.slice -mindepth 1 -maxdepth 1 -type d | wc -l)" = 0 ]'
fi
echo; echo "$pass passed, $fail failed; log: $LOG"
