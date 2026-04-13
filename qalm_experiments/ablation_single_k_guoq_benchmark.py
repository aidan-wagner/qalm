"""
Single-k ablation on the GUOQ 250-circuit benchmark set.

Same experiment as ablation_single_k.py but with 250 circuits from
~/guoq/benchmarks/nam_rz/ instead of the 26 nam_circs.

Results go to single_k_guoq_benchmark_results/ (separate from the 26-circuit results).

Run from repo root:
    python3 qalm_experiments/ablation_single_k_guoq_benchmark.py
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

# Import the experiment module and override its config
import ablation_single_k as _exp

_exp.CIRCUIT_LIST = [
    (path, os.path.splitext(os.path.basename(path))[0])
    for path in _qasm_files
]
_exp.RESULTS_DIR = "single_k_guoq_benchmark_results"

if __name__ == "__main__":
    print(f"Using {len(_exp.CIRCUIT_LIST)} circuits from {_BENCH_DIR}")
    print(f"Results → {_exp.RESULTS_DIR}/")
    _exp.main()
