#!/usr/bin/env bash
# Fix the CPU frequency regime for the whole campaign: boost off, governor performance.
#   repro/env/set_cpu_mode.sh fixed     # campaign setting
#   repro/env/set_cpu_mode.sh restore   # back to boost=1, schedutil (the machine's default)
set -euo pipefail
case "${1:-}" in
  fixed)   boost=0; gov=performance ;;
  restore) boost=1; gov=schedutil ;;
  *) echo "usage: $0 fixed|restore" >&2; exit 2 ;;
esac
echo "$boost" | sudo -n tee /sys/devices/system/cpu/cpufreq/boost > /dev/null
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$gov" | sudo -n tee "$g" > /dev/null; done
echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost) governors: $(sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c | tr '\n' ' ')"
