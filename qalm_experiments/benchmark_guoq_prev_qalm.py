"""
Previous QALM (gate-count cost, no-enq) on the 250-circuit GUOQ benchmark.

Config matches the Nam weighted-cost experiment so results pair cleanly:
  npool=1, nbranch=3, greedy_k=2, advancing k, enqueue_intermediate=0,
  cost_mode=0 (gate count).

After the run, reports avg and geomean reductions for total gate count and
CX count over 248 circuits (excludes qft_10 and qft_16).

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_guoq_prev_qalm.py
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

# ── top-level parameters ─────────────────────────────────────────────────────
TIMEOUT      = 60
N_WORKERS    = 16
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "guoq_benchmark_prev_qalm_results"

COST_MODE    = 0

# (n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr, local, two_way)
CONFIG = (1, 3, 2, 1.5, 0, 0, 1, 0)

EXCLUDE = {"qft_10", "qft_16"}

# ── circuits ──────────────────────────────────────────────────────────────────
_GUOQ_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")
GUOQ_CIRCUITS = sorted([
    (_GUOQ_DIR + "/" + f, os.path.splitext(f)[0], "guoq")
    for f in os.listdir(_GUOQ_DIR) if f.endswith(".qasm")
])

ALL_CIRCUITS = GUOQ_CIRCUITS


# ── helpers ───────────────────────────────────────────────────────────────────
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
        total = qc.size()
        cx    = qc.count_ops().get("cx", 0)
        return total, cx
    except Exception:
        return -1, -1


# ── worker ────────────────────────────────────────────────────────────────────
def run_one(args):
    circ_path, circ_name, source = args
    (n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr,
     local, two_way) = CONFIG

    pkl_path = f"{OUT_DIR}/pkl/{source}_{circ_name}_{TIMEOUT}.pkl"
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
        "0",                 # fixed_k=0 (advancing)
        "0",                 # enqueue_intermediate=0 (no-enq)
        str(COST_MODE),
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        preexec_fn=_set_mem_limit,
    )

    mem_limit_kb = MEM_LIMIT_GB * 1024 * 1024
    peak_ref = [0]
    stop_evt = threading.Event()
    mon_thread = threading.Thread(
        target=_monitor, args=(proc, mem_limit_kb, peak_ref, stop_evt), daemon=True
    )
    mon_thread.start()

    wall_limit = TIMEOUT + 120
    try:
        stdout, _ = proc.communicate(timeout=wall_limit)
    except subprocess.TimeoutExpired:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
    stop_evt.set()
    mon_thread.join()

    peak_kb = peak_ref[0]
    oom_killed = (proc.returncode not in (0, None))

    # "[name] Best cost: <cost> CX: <cx> candidate number: <n> after <t> seconds."
    # With cost_mode=0, cost == total gate count.
    times, totals, cxs = [], [], []
    for line in stdout.splitlines():
        words = line.split()
        if len(words) < 12:
            continue
        if words[0] == f"[{circ_name}]" and words[1] == "Best":
            try:
                totals.append(float(words[3]))
                cxs.append(int(words[5]))
                times.append(float(words[10]))
            except (ValueError, IndexError):
                pass

    if not totals and oom_killed:
        status = "OOM"
    elif not totals:
        status = "FAILED"
    else:
        status = "OK"

    print(
        f"[{source}] {circ_name}: {len(totals)} improvements, "
        f"peak={peak_kb/1024:.0f} MB, status={status}",
        flush=True,
    )

    result = (times, totals, cxs, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    print(
        f"Benchmarking {len(ALL_CIRCUITS)} guoq circuits with {N_WORKERS} workers, "
        f"timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB, cost_mode={COST_MODE}, "
        f"npool={CONFIG[0]}, nbranch={CONFIG[1]}, no-enq"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, ALL_CIRCUITS)

    csv_path = f"{OUT_DIR}/results_{TIMEOUT}s.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source", "circuit",
             "orig_total", "orig_cx",
             "opt_total", "opt_cx",
             "total_reduction_pct", "cx_reduction_pct",
             "peak_rss_mb", "status"]
        )
        rows = []
        for (cp, cn, src), result in zip(ALL_CIRCUITS, results):
            times, totals, cxs, peak_kb, status = result
            orig_total, orig_cx = _orig_counts(cp)
            opt_total = int(totals[-1]) if totals else -1
            opt_cx    = int(cxs[-1])    if cxs    else -1

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            row = [src, cn, orig_total, orig_cx, opt_total, opt_cx,
                   pct(opt_total, orig_total), pct(opt_cx, orig_cx),
                   f"{peak_kb/1024:.1f}", status]
            writer.writerow(row)
            rows.append((cn, orig_total, orig_cx, opt_total, opt_cx))

    print(f"\nCSV saved -> {csv_path}")

    # ── summary over 248 circuits (exclude qft_10, qft_16) ─────────────────────
    total_reds, cx_reds = [], []
    total_ratios, cx_ratios = [], []
    missing, excluded = [], []
    for cn, orig_total, orig_cx, opt_total, opt_cx in rows:
        if cn in EXCLUDE:
            excluded.append(cn); continue
        if opt_total < 0 or orig_total <= 0:
            missing.append(cn); continue
        total_reds.append((1 - opt_total/orig_total) * 100)
        total_ratios.append(opt_total/orig_total)
        if orig_cx > 0 and opt_cx >= 0:
            cx_reds.append((1 - opt_cx/orig_cx) * 100)
            cx_ratios.append(opt_cx/orig_cx if opt_cx > 0 else 1e-9)

    def geomean_red(ratios):
        if not ratios: return float('nan')
        return (1 - np.exp(np.mean(np.log(ratios)))) * 100

    print(f"\n=== Summary over 248 circuits (excluded: {sorted(excluded)}, missing: {sorted(missing)}) ===")
    print(f"Total gate:  arith-mean reduction = {np.mean(total_reds):.2f}%  "
          f"geomean reduction = {geomean_red(total_ratios):.2f}%  "
          f"(n={len(total_reds)})")
    print(f"CX (2Q):     arith-mean reduction = {np.mean(cx_reds):.2f}%  "
          f"geomean reduction = {geomean_red(cx_ratios):.2f}%  "
          f"(n={len(cx_reds)})")


if __name__ == "__main__":
    main()
