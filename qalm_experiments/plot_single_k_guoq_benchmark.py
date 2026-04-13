"""
Plotting script for the GUOQ 250-circuit single-k ablation.

Reads pickled results from single_k_guoq_benchmark_results/pkl/ and
greedy_k_guoq_benchmark_results/pkl/ (for the advancing-k overlay).
No experiments are run — this is plot-only.

Usage:
    python3 qalm_experiments/plot_single_k_guoq_benchmark.py
"""

import glob
import os
import sys

# Build CIRCUIT_LIST from the benchmark directory
_BENCH_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")
_qasm_files = sorted(glob.glob(os.path.join(_BENCH_DIR, "*.qasm")))
if not _qasm_files:
    print(f"ERROR: no .qasm files found in {_BENCH_DIR}", file=sys.stderr)
    sys.exit(1)

# Import the plot module and override its config
import plot_single_k as _plot

_EXCLUDE = {"qft_10", "qft_16"}
_plot.CIRCUIT_LIST = [
    (path, os.path.splitext(os.path.basename(path))[0])
    for path in _qasm_files
    if os.path.splitext(os.path.basename(path))[0] not in _EXCLUDE
]
_plot.RESULTS_DIR = "single_k_guoq_benchmark_results"
_plot.GREEDY_K_RESULTS_DIR = "greedy_k_guoq_benchmark_results"
_plot.REDUCTION_YLIM = None

if __name__ == "__main__":
    print(f"Using {len(_plot.CIRCUIT_LIST)} circuits from {_BENCH_DIR}")
    print(f"Results dir → {_plot.RESULTS_DIR}/")
    _plot.main()
