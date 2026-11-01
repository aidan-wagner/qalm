#!/usr/bin/env bash
# Regenerate every table, figure and number of the paper from repro/results into
# repro/analysis/out (or the directory given as the first argument):
#   a/, b/      tables and S248 figures for QALM configuration (a) (the paper) and (b)
#   curves/     ablation figures (greedy k, single k, N_pool, N_branch) and their numbers
#   numbers/    every number quoted in the text (numbers.tex macros, numbers.json provenance)
# Needs the GUOQ benchmarks (repro/inputs/fetch_guoq_benchmarks.sh) only for numSuiteSourceCircuits.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
PY=$REPO/.venv/bin/python   # the uv environment (uv sync)
[ -x "$PY" ] || { echo "no $PY: run 'uv sync' in $REPO first" >&2; exit 1; }
OUT=${1:-repro/analysis/out}
A=repro/analysis
R=repro/results
for c in a b; do
  "$PY" $A/tables.py --results $R --qalm-config $c --out "$OUT/$c"
  "$PY" $A/fig_percircuit.py --results $R --qalm-config $c --out "$OUT/$c"
done
"$PY" $A/fig_curves.py --results $R --out "$OUT/curves"
"$PY" $A/numbers.py --results $R --events $R/events.csv --out "$OUT/numbers"
