"""
Parallel re-run of the 51 Quartz (test_optimize) circuits that OOMed at
startup under the 8GB limit. Uses 32GB RLIMIT_AS per worker and 8 workers.

Reuses _launch + run_quartz_circuit from rerun_oom_circuits.py.

Run from repo root:
    python3 rerun_quartz_ooms_parallel.py
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from rerun_oom_circuits import (
    ECCSET,
    GUOQ_DIR,
    QUARTZ_CONFIG,
    QUARTZ_OUT_DIR,
    TIMEOUT,
    run_quartz_circuit,
)

N_WORKERS = 8
CSV_PATH = f"{QUARTZ_OUT_DIR}/results_3600s.csv"


def load_oom_circuits():
    oom = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["status"].strip() == "OOM":
                oom.append(row["circuit"])
    return oom


def _run_one(circ_name):
    roqc_interval, preprocess, two_way_rm = QUARTZ_CONFIG
    circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
    pkl_path  = f"{QUARTZ_OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
    qasm_path = f"{QUARTZ_OUT_DIR}/qasm/guoq_{circ_name}_{TIMEOUT}.qasm"
    cmd = [
        "./build/test_optimize",
        circ_path, circ_name, str(TIMEOUT),
        str(roqc_interval), str(preprocess), str(two_way_rm),
        ECCSET,
    ]
    print(f"[START] {circ_name}", flush=True)
    result = run_quartz_circuit(cmd, circ_name, pkl_path, qasm_path)
    times, totals, final_total, final_cx, peak_kb, status = result
    print(f"[DONE ] {circ_name}: status={status}, total={final_total}, "
          f"cx={final_cx}, peak={peak_kb/1024:.0f}MB", flush=True)
    return circ_name, status, final_total, final_cx, peak_kb


def main():
    circuits = load_oom_circuits()
    print(f"=== Re-running {len(circuits)} Quartz OOM circuits "
          f"with 32GB limit, {N_WORKERS} workers ===\n", flush=True)

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
    ok   = sum(1 for _, s, *_ in results if s == "OK")
    oom  = sum(1 for _, s, *_ in results if s == "OOM")
    fail = sum(1 for _, s, *_ in results if s == "FAILED")
    print(f"OK={ok}, OOM={oom}, FAILED={fail}, total={len(results)}")


if __name__ == "__main__":
    sys.exit(main())
