#!/usr/bin/env bash
# Record the native build the campaign will use: writes repro/env/build_info.txt with the git
# commit and every uncommitted change of the source, the toolchain the build used, the compile
# flags and the sha256 of the binaries and the library named in repro/config/campaign.yaml, and
# repro/env/build_info.diff with the uncommitted changes themselves (so that a build from a
# modified working tree can be rebuilt). The scheduler refuses to start if the binaries no
# longer match.
# Usage (after building): repro/env/record_build.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY=$REPO/.venv/bin/python   # the uv environment (uv sync)
[ -x "$PY" ] || { echo "no $PY: run 'uv sync' in $REPO first" >&2; exit 1; }
paths=$("$PY" -c "
import yaml
c = yaml.safe_load(open('repro/config/campaign.yaml'))
print(' '.join(list(c['binaries'].values()) + [c['lib']]))")
build_dir=$(dirname "$(echo "$paths" | awk '{print $1}')")
SRC="src roqc CMakeLists.txt"
DIFF=repro/env/build_info.diff
# uncommitted changes of the source: tracked files (staged or not) and new untracked files
git diff HEAD -- $SRC > "$DIFF"
for f in $(git ls-files --others --exclude-standard -- $SRC); do
  git diff --no-index -- /dev/null "$f" >> "$DIFF" || true   # exit 1 = "files differ"
done
{
  echo "## recorded $(date -u +%FT%TZ) on $(hostname)"
  echo "## git"
  git rev-parse HEAD
  git status --short -- $SRC
  if [ -s "$DIFF" ]; then
    echo "uncommitted changes: $DIFF ($(wc -l < "$DIFF") lines, sha256 $(sha256sum "$DIFF" | awk '{print $1}'))"
  else
    echo "uncommitted changes: none"
  fi
  echo "## toolchain"
  # the rustc cargo actually compiled roqc with (CMakeCache's Rust_VERSION is detected at
  # configure time and can differ when RUSTUP_TOOLCHAIN is set only for the build)
  info="$build_dir/cargo/build/.rustc_info.json"
  if [ -f "$info" ]; then
    echo "rustc used by the build: $("$PY" -I -c '
import json, sys
outs = json.load(open(sys.argv[1])).get("outputs", {}).values()
print(next((o["stdout"].splitlines()[0] for o in outs if o.get("stdout", "").startswith("rustc ")), "unknown"))' "$info")"
  else
    echo "rustc used by the build: unknown (no $info)"
  fi
  echo "RUSTUP_TOOLCHAIN when recording: ${RUSTUP_TOOLCHAIN:-<unset>}"
  grep -E '^(Rust_TOOLCHAIN|CMAKE_CXX_COMPILER|CMAKE_BUILD_TYPE)(:[A-Z]+)?=' "$build_dir/CMakeCache.txt" 2>/dev/null \
    || echo "no $build_dir/CMakeCache.txt"
  cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$build_dir/CMakeCache.txt" 2>/dev/null || true)
  if [ -n "$cxx" ]; then echo "C++ compiler used by the build: $("$cxx" --version | head -1)"; fi
  echo "on PATH: $(rustc -V 2>/dev/null || echo 'no rustc'); $(cargo -V 2>/dev/null || echo 'no cargo')"
  cmake --version | head -1
  echo "## compile flags (quartz_runtime)"
  grep -h "CXX_FLAGS" "$build_dir"/CMakeFiles/quartz_runtime.dir/flags.make 2>/dev/null || echo "not found"
  echo "## binaries"
  for p in $paths; do
    echo "$(sha256sum "$p" | awk '{print $1}')  $p"
  done
} > repro/env/build_info.txt
cat repro/env/build_info.txt
