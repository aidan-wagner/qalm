"""
Weighted-cost study on the 250-circuit GUOQ benchmark suite (~/guoq/benchmarks/nam_rz).

Same plumbing as benchmark_guoq.py, but sets cost_mode=1 on
test_greedy_k_ablation so the search optimizes (1Q + 10*2Q) instead of
raw gate count.

The per-line log format is:
    "[name] Best cost: <weighted>  CX: <cx>  candidate number: <n>  after <t> s."
so we parse `weighted` and `cx` directly; total gates are recovered via
total = weighted - 9 * cx (exact for Nam gate set where all 2Q gates = cx).

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_guoq_weighted_cost.py
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
TIMEOUT      = 3600
N_WORKERS    = 32
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "guoq_benchmark_weighted_cost_results"

COST_MODE    = 1   # 1 = weighted (1Q + 10*2Q)

CHECKPOINTS  = [10, 30, 60, 120, 180, 240, 360, 480, 600, 900, 1200, 1800, 2400, 3000, 3600]

# Config matched to benchmark_guoq_prev_qalm.py (N_pool=1, N_branch=3,
# greedy_k=2, advancing k, local=1, no-enq). Baseline-vs-weighted comparison
# isolates cost_mode.
BEST_CONFIG = (1, 3, 2, 1.5, 0, 0, 1, 0)

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
        one_q = total - cx
        weighted = one_q + 10 * cx
        return total, cx, weighted
    except Exception:
        return -1, -1, -1


def _interp_at(times, values, t):
    if not times:
        return None
    idx = next((i for i, ts in enumerate(times) if ts > t), len(times)) - 1
    return values[max(0, idx)]


# ── worker ────────────────────────────────────────────────────────────────────
def run_one(args):
    circ_path, circ_name, source = args
    (n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr,
     local, two_way) = BEST_CONFIG

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
        str(COST_MODE),      # weighted cost
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

    # "[name] Best cost: <weighted> CX: <cx> candidate number: <n> after <t> seconds."
    times, weighteds, cxs = [], [], []
    for line in stdout.splitlines():
        words = line.split()
        if len(words) < 12:
            continue
        if words[0] == f"[{circ_name}]" and words[1] == "Best":
            try:
                weighteds.append(float(words[3]))
                cxs.append(int(words[5]))
                times.append(float(words[10]))
            except (ValueError, IndexError):
                pass

    totals = [int(w) - 9 * c for w, c in zip(weighteds, cxs)]

    if not weighteds and oom_killed:
        status = "OOM"
    elif not weighteds:
        status = "FAILED"
    else:
        status = "OK"

    print(
        f"[{source}] {circ_name}: {len(weighteds)} improvements, "
        f"peak={peak_kb/1024:.0f} MB, status={status}",
        flush=True,
    )

    result = (times, totals, cxs, weighteds, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    print(
        f"Benchmarking {len(ALL_CIRCUITS)} guoq circuits with {N_WORKERS} workers, "
        f"timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB, cost_mode={COST_MODE}"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, ALL_CIRCUITS)

    chk_cols = []
    for t in CHECKPOINTS:
        chk_cols += [f"total_t{t}", f"cx_t{t}", f"weighted_t{t}"]

    csv_path = f"{OUT_DIR}/results_{TIMEOUT}s.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source", "circuit",
             "orig_total", "orig_cx", "orig_weighted",
             "opt_total", "opt_cx", "opt_weighted",
             "total_reduction_pct", "cx_reduction_pct", "weighted_reduction_pct",
             "peak_rss_mb", "status"]
            + chk_cols
        )
        for (circ_path, circ_name, source), result in zip(ALL_CIRCUITS, results):
            times, totals, cxs, weighteds, peak_kb, status = result
            orig_total, orig_cx, orig_w = _orig_counts(circ_path)
            opt_total = int(totals[-1]) if totals else -1
            opt_cx    = int(cxs[-1])    if cxs    else -1
            opt_w     = int(weighteds[-1]) if weighteds else -1

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            chk_vals = []
            for t in CHECKPOINTS:
                chk_vals.append(_interp_at(times, totals, t) or "")
                chk_vals.append(_interp_at(times, cxs, t) or "")
                chk_vals.append(_interp_at(times, weighteds, t) or "")

            writer.writerow(
                [source, circ_name,
                 orig_total, orig_cx, orig_w,
                 opt_total, opt_cx, opt_w,
                 pct(opt_total, orig_total), pct(opt_cx, orig_cx), pct(opt_w, orig_w),
                 f"{peak_kb/1024:.1f}", status]
                + chk_vals
            )

    print(f"\nCSV saved → {csv_path}")

    # summary: geomean reduction for each metric
    for label, idx in [("total", 1), ("cx", 2), ("weighted", 3)]:
        ratios = []
        for (cp, _, _), r in zip(ALL_CIRCUITS, results):
            series = r[idx]
            orig = _orig_counts(cp)[idx - 1]
            if series and orig > 0:
                ratios.append(series[-1] / orig)
        if ratios:
            print(
                f"guoq-{label}: {len(ratios)}/{len(ALL_CIRCUITS)} OK, "
                f"avg={np.mean(ratios):.4f}, geomean={np.exp(np.mean(np.log(ratios))):.4f}"
            )


if __name__ == "__main__":
    main()
