#!/usr/bin/env bash
# Read-only snapshot of the host environment. Writes repro/env/<host>_<utc>.txt
set -u
cd "$(dirname "$0")"
out="$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "## date";        date -u
  echo "## uname";       uname -a
  echo "## lscpu";       lscpu
  echo "## numactl";     numactl -H
  echo "## microcode";   grep -m1 microcode /proc/cpuinfo
  echo "## boost";       cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
  echo "## governors";   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
  echo "## scaling_max_freq"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq 2>/dev/null | sort | uniq -c
  echo "## numa_balancing"; cat /proc/sys/kernel/numa_balancing
  echo "## thp";         cat /sys/kernel/mm/transparent_hugepage/enabled
  echo "## free";        free -g
  echo "## systemd";     systemctl --version | head -1
  echo "## docker";      sudo -n docker version --format '{{.Server.Version}}' 2>&1
  echo "## gcc";         gcc --version | head -1
  echo "## cmake";       cmake --version | head -1
  echo "## rustc";       ~/.cargo/bin/rustc +stable -V 2>&1; ~/.cargo/bin/cargo +stable -V 2>&1
  echo "## python";      python3 -V
  python3 - <<'PY'
import importlib
for m in ["numpy", "matplotlib", "scipy", "yaml", "qiskit"]:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "MISSING", e)
PY
} > "$out" 2>&1
echo "wrote repro/env/$out"
