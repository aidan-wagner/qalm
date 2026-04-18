"""
Benchmark the new termination-on-queue-degradation build on 26 nam_circs
(circuit/nam_circs/ minus hwb6 and ham15-low), 1h timeout.

Uses the current paper "best config": gk=2 (greedy k=1 + k=2 pre-pass, then
advancing QALM starting at k=3), N_pool=1, N_branch=1, enqueue_intermediate=1.

Output is parallel to guoq_benchmark_advk_results/ so the CSVs can be
diffed side-by-side.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_nam_with_termination.py
"""

import csv
import os
import pickle
import resource
import signal
import subprocess
import threading
import multiprocessing

import numpy as np

# ── settings ─────────────────────────────────────────────────────────────────
TIMEOUT      = 3600
N_WORKERS    = 26
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "nam_termination_plus2_results"

CHECKPOINTS  = [10, 30, 60, 120, 180, 240, 360, 480, 600, 900, 1200, 1800, 2400, 3000, 3600]

# Paper best config (same as benchmark_guoq.py BEST_CONFIG).
# (n_pool, n_branch, greedy_k, repeat_tol, exp_incr, no_incr, local, two_way)
BEST_CONFIG = (1, 1, 2, 1.5, 0, 0, 1, 0)

# 26 circuits: nam_circs minus hwb6 and ham15-low.
NAM_CIRCUITS = [
    ("circuit/nam_circs/adder_8.qasm",        "adder_8"),
    ("circuit/nam_circs/barenco_tof_3.qasm",  "barenco_tof_3"),
    ("circuit/nam_circs/barenco_tof_4.qasm",  "barenco_tof_4"),
    ("circuit/nam_circs/barenco_tof_5.qasm",  "barenco_tof_5"),
    ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
    ("circuit/nam_circs/csla_mux_3.qasm",     "csla_mux_3"),
    ("circuit/nam_circs/csum_mux_9.qasm",     "csum_mux_9"),
    ("circuit/nam_circs/gf2^4_mult.qasm",     "gf2^4_mult"),
    ("circuit/nam_circs/gf2^5_mult.qasm",     "gf2^5_mult"),
    ("circuit/nam_circs/gf2^6_mult.qasm",     "gf2^6_mult"),
    ("circuit/nam_circs/gf2^7_mult.qasm",     "gf2^7_mult"),
    ("circuit/nam_circs/gf2^8_mult.qasm",     "gf2^8_mult"),
    ("circuit/nam_circs/gf2^9_mult.qasm",     "gf2^9_mult"),
    ("circuit/nam_circs/gf2^10_mult.qasm",    "gf2^10_mult"),
    ("circuit/nam_circs/mod5_4.qasm",         "mod5_4"),
    ("circuit/nam_circs/mod_mult_55.qasm",    "mod_mult_55"),
    ("circuit/nam_circs/mod_red_21.qasm",     "mod_red_21"),
    ("circuit/nam_circs/qcla_adder_10.qasm",  "qcla_adder_10"),
    ("circuit/nam_circs/qcla_com_7.qasm",     "qcla_com_7"),
    ("circuit/nam_circs/qcla_mod_7.qasm",     "qcla_mod_7"),
    ("circuit/nam_circs/rc_adder_6.qasm",     "rc_adder_6"),
    ("circuit/nam_circs/tof_3.qasm",          "tof_3"),
    ("circuit/nam_circs/tof_4.qasm",          "tof_4"),
    ("circuit/nam_circs/tof_5.qasm",          "tof_5"),
    ("circuit/nam_circs/tof_10.qasm",         "tof_10"),
    ("circuit/nam_circs/vbe_adder_3.qasm",    "vbe_adder_3"),
]


# ── helpers (same pattern as benchmark_guoq.py) ──────────────────────────────
def _set_mem_limit():
    lim = MEM_LIMIT_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def _monitor(proc, mem_limit_kb, peak_ref, stop_evt):
    peak = 0
    while not stop_evt.is_set():
        try:
            with open(f"/proc/{proc.pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        if kb > peak:
                            peak = kb
                        if kb > mem_limit_kb:
                            try:
                                os.kill(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        break
        except (FileNotFoundError, ProcessLookupError):
            break
        stop_evt.wait(0.5)
    peak_ref[0] = peak


def _orig_counts(circ_path):
    import qiskit
    try:
        qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
        return qc.size(), qc.count_ops().get("cx", 0)
    except Exception:
        return -1, -1


def _interp_at(times, values, t):
    if not times:
        return None
    idx = next((i for i, ts in enumerate(times) if ts > t), len(times)) - 1
    return values[max(0, idx)]


def run_one(args):
    circ_path, circ_name = args
    (n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr,
     local, two_way) = BEST_CONFIG

    pkl_path = f"{OUT_DIR}/pkl/{circ_name}_{TIMEOUT}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    cmd = [
        "./build/test_greedy_k_ablation",
        circ_path, circ_name,
        str(TIMEOUT),
        str(greedy_k),
        str(n_pool), str(n_branch),
        str(rep_tol),
        str(int(exp_incr)), str(int(no_incr)),
        str(int(local)), str(int(two_way)),
        ECCSET,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_set_mem_limit,
    )

    mem_limit_kb = MEM_LIMIT_GB * 1024 * 1024
    peak_ref = [0]
    stop_evt = threading.Event()
    mon = threading.Thread(
        target=_monitor, args=(proc, mem_limit_kb, peak_ref, stop_evt), daemon=True
    )
    mon.start()

    wall_limit = TIMEOUT + 180
    try:
        stdout, _ = proc.communicate(timeout=wall_limit)
    except subprocess.TimeoutExpired:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
    stop_evt.set()
    mon.join()

    peak_kb = peak_ref[0]
    oom_killed = (proc.returncode not in (0, None))

    times, totals, cxs = [], [], []
    qalm_k_starts = []
    term_events = 0
    for line in stdout.splitlines():
        words = line.split()
        if len(words) >= 12 and words[0] == f"[{circ_name}]" and words[1] == "Best":
            try:
                totals.append(float(words[3]))
                cxs.append(int(words[5]))
                times.append(float(words[10]))
            except (ValueError, IndexError):
                pass
        if "QALM_K_START" in line:
            qalm_k_starts.append(line)
        if "queue degraded" in line:
            term_events += 1

    if not totals and oom_killed:
        status = "OOM"
    elif not totals:
        status = "FAILED"
    else:
        status = "OK"

    # Largest k reached
    import re
    max_k = 0
    for s in qalm_k_starts:
        m = re.search(r'k=(\d+)', s)
        if m:
            max_k = max(max_k, int(m.group(1)))

    print(
        f"{circ_name}: {len(totals)} improvements, peak={peak_kb/1024:.0f}MB, "
        f"max_k={max_k}, term_events={term_events}, status={status}",
        flush=True,
    )

    result = (times, totals, cxs, peak_kb, status, max_k, term_events)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    print(
        f"Benchmarking {len(NAM_CIRCUITS)} nam_circs (hwb6/ham15-low excluded) "
        f"with {N_WORKERS} workers, timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB"
    )
    print("Binary: ./build/test_greedy_k_ablation (rebuilt with termination check)")
    print(f"Config: gk=2 advancing, N_pool=1, N_branch=1, local=1, enqueue=1")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, NAM_CIRCUITS)

    chk_cols = []
    for t in CHECKPOINTS:
        chk_cols += [f"total_t{t}", f"cx_t{t}"]

    csv_path = f"{OUT_DIR}/results_{TIMEOUT}s.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["circuit",
             "orig_total", "orig_cx",
             "opt_total", "opt_cx",
             "total_reduction_pct", "cx_reduction_pct",
             "peak_rss_mb", "status", "max_k", "term_events"]
            + chk_cols
        )
        for (circ_path, circ_name), result in zip(NAM_CIRCUITS, results):
            # Handle both new and legacy pickle tuple lengths
            if len(result) == 7:
                times, totals, cxs, peak_kb, status, max_k, term_events = result
            else:
                times, totals, cxs, peak_kb, status = result[:5]
                max_k, term_events = 0, 0
            orig_total, orig_cx = _orig_counts(circ_path)
            opt_total = int(totals[-1]) if totals else -1
            opt_cx    = int(cxs[-1])    if cxs    else -1

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            chk_vals = []
            for t in CHECKPOINTS:
                chk_vals.append(_interp_at(times, totals, t) or "")
                chk_vals.append(_interp_at(times, cxs, t) or "")

            writer.writerow(
                [circ_name,
                 orig_total, orig_cx,
                 opt_total, opt_cx,
                 pct(opt_total, orig_total), pct(opt_cx, orig_cx),
                 f"{peak_kb/1024:.1f}", status, max_k, term_events]
                + chk_vals
            )

    print(f"\nCSV → {csv_path}")

    # Summary
    ratios = []
    for (_, _), r in zip(NAM_CIRCUITS, results):
        totals = r[1]
        if totals:
            ratios.append(totals[-1])
    if ratios:
        print(f"OK: {len(ratios)}/{len(NAM_CIRCUITS)} circuits completed")


if __name__ == "__main__":
    main()
