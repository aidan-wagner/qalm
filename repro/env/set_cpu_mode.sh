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
C=/sys/devices/system/cpu
if [ -e $C/cpufreq/boost ]; then                 # acpi-cpufreq (AMD)
  echo "$boost" | sudo -n tee $C/cpufreq/boost > /dev/null
elif [ -e $C/intel_pstate/no_turbo ]; then       # intel_pstate
  echo $((1 - boost)) | sudo -n tee $C/intel_pstate/no_turbo > /dev/null
else
  echo "no CPU boost control on this machine; run the scheduler with REPRO_ALLOW_CPU_MODE=1" >&2; exit 1
fi
for g in $C/cpu*/cpufreq/scaling_governor; do echo "$gov" | sudo -n tee "$g" > /dev/null; done
echo "boost=$boost governors: $(sort $C/cpu*/cpufreq/scaling_governor | uniq -c | tr '\n' ' ')"
