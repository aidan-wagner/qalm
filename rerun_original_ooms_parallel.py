"""
Re-run the pure-Quartz (test_original) circuits that OOM'd in the 8GB sweep,
this time with 32GB memory limit and 8 workers.

Reads status from original_benchmark_results/results_3600s.csv; reruns any
circuit with status in {OOM, OOM_AFTER_GREEDY} by deleting the stale pkl and
calling benchmark_original_guoq.run_one under a higher memory cap.

Run from repo root:
    python3 rerun_original_ooms_parallel.py
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the run_one from the main benchmark module, but override constants so
# it uses 32GB and writes to the same OUT_DIR (overwrites stale OOM pkls).
sys.path.insert(0, "qalm_experiments")
import benchmark_original_guoq as bog

bog.MEM_LIMIT_GB = 32  # override

N_WORKERS = 8
OUT_DIR   = bog.OUT_DIR
TIMEOUT   = bog.TIMEOUT
GUOQ_DIR  = os.path.expanduser("~/guoq/benchmarks/nam_rz")
CSV_PATH  = f"{OUT_DIR}/results_{TIMEOUT}s.csv"


def load_failed_circuits():
    failed = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["status"].strip() in {"OOM", "OOM_AFTER_GREEDY"}:
                failed.append((row["circuit"], row["status"].strip()))
    return failed


def _run_one(args):
    circ_name, prev_status = args
    circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
    pkl_path  = f"{OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
    # Remove stale pkl so run_one actually re-executes the subprocess.
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    print(f"[START] {circ_name} (prev={prev_status})", flush=True)
    result = bog.run_one((circ_path, circ_name, "guoq"))
    (post_greedy_time, post_greedy_total, post_greedy_cx,
     times, totals, final_total, final_cx, peak_kb, status) = result
    print(
        f"[DONE ] {circ_name}: prev={prev_status} → {status}, "
        f"greedy={post_greedy_total}/{post_greedy_cx} @{post_greedy_time}s, "
        f"final={final_total}/{final_cx}, peak={peak_kb/1024:.0f}MB",
        flush=True,
    )
    return circ_name, prev_status, status


def main():
    failed = load_failed_circuits()
    print(
        f"=== Re-running {len(failed)} OOM/OOM_AFTER_GREEDY circuits "
        f"with {bog.MEM_LIMIT_GB}GB limit, {N_WORKERS} workers ===",
        flush=True,
    )

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, a): a for a in failed}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:
                a = futs[fut]
                print(f"[FAIL ] {a}: {exc}", flush=True)

    print("\n=== Summary ===")
    for status in ("OK", "OOM_AFTER_GREEDY", "OOM", "FAILED"):
        n = sum(1 for _, _, s in results if s == status)
        print(f"  {status:<20} {n}")


if __name__ == "__main__":
    main()
