#!/usr/bin/env bash
# Record the native build the campaign will use: writes repro/env/build_info.txt with the
# toolchain, the compile flags and the sha256 of the binaries and the library named in
# repro/config/campaign.yaml. The scheduler refuses to start if they no longer match.
# Usage (after building): repro/env/record_build.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
paths=$(python3 -c "
import yaml
c = yaml.safe_load(open('repro/config/campaign.yaml'))
print(' '.join(list(c['binaries'].values()) + [c['lib']]))")
build_dir=$(dirname "$(echo "$paths" | awk '{print $1}')")
{
  echo "## recorded $(date -u +%FT%TZ) on $(hostname)"
  echo "## git"
  git rev-parse HEAD
  git status --short -- src roqc CMakeLists.txt | head -20
  echo "## toolchain"
  rustc -V 2>/dev/null || echo "rustc: not on PATH"
  cargo -V 2>/dev/null || echo "cargo: not on PATH"
  gcc --version | head -1
  cmake --version | head -1
  echo "## compile flags (quartz_runtime)"
  grep -h "CXX_FLAGS" "$build_dir"/CMakeFiles/quartz_runtime.dir/flags.make 2>/dev/null || echo "not found"
  echo "## binaries"
  for p in $paths; do
    echo "$(sha256sum "$p" | awk '{print $1}')  $p"
  done
} > repro/env/build_info.txt
cat repro/env/build_info.txt
