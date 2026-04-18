"""
Re-run the 15 circuits that previously hung in greedy_optimize() and were
mis-classified as OOM. Now that greedy_optimize has a timeout parameter
(half of the overall budget), these should complete with at least a greedy
snapshot, and likely get some optimize_original improvements too.

8 workers × 32GB × 3600s budget.

Run from repo root:
    python3 rerun_hung_greedy_parallel.py
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "qalm_experiments")
import benchmark_original_guoq as bog

bog.MEM_LIMIT_GB = 32  # override

N_WORKERS = 8
OUT_DIR   = bog.OUT_DIR
TIMEOUT   = bog.TIMEOUT
GUOQ_DIR  = os.path.expanduser("~/guoq/benchmarks/nam_rz")
CSV_PATH  = f"{OUT_DIR}/results_{TIMEOUT}s.csv"


def load_hung_circuits():
    """Pick up only status=OOM rows: those never produced a greedy snapshot."""
    hung = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["status"].strip() == "OOM":
                hung.append(row["circuit"])
    return hung


def _run_one(circ_name):
    circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
    pkl_path  = f"{OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    print(f"[START] {circ_name}", flush=True)
    result = bog.run_one((circ_path, circ_name, "guoq"))
    (post_greedy_time, post_greedy_total, post_greedy_cx,
     times, totals, final_total, final_cx, peak_kb, status) = result
    print(
        f"[DONE ] {circ_name}: status={status}, "
        f"greedy={post_greedy_total}/{post_greedy_cx} @{post_greedy_time}s, "
        f"final={final_total}/{final_cx}, peak={peak_kb/1024:.0f}MB",
        flush=True,
    )
    return circ_name, status


def main():
    circuits = load_hung_circuits()
    print(
        f"=== Re-running {len(circuits)} hung-greedy circuits with "
        f"{bog.MEM_LIMIT_GB}GB limit, {N_WORKERS} workers, timeout={TIMEOUT}s "
        f"(greedy gets the full budget) ===",
        flush=True,
    )

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, c): c for c in circuits}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:
                c = futs[fut]
                print(f"[FAIL ] {c}: {exc}", flush=True)

    print("\n=== Summary ===")
    for s in ("OK", "OOM_AFTER_GREEDY", "OOM", "FAILED"):
        n = sum(1 for _, status in results if status == s)
        print(f"  {s:<20} {n}")


if __name__ == "__main__":
    main()
