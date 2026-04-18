"""
Benchmark QALM (best config, advancing k) on:
  - ~/guoq/benchmarks/nam_rz/  (250 circuits)
  - circuit/nam_circs/          (26 circuits, for cross-checking)
  - circuit/nam_rm_circs/       (28 circuits, for cross-checking)

Outputs a CSV per timeout with original/optimized total-gate and CX counts,
plus interpolated gate counts at fixed checkpoints, suitable for tables like
eval_end_to_end.tex.

Best config (from exploration_k_comparison experiments):
  N_pool=1, N_branch=1, advancing k starting at 3 (i.e. greedy_k=2 →
  greedy(k=1) + greedy(k=2), then QALM at k=3, 4, 5, …),
  local=1, two_way=0, Nam_(5,3) ECC set.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_guoq.py
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

# ── top-level parameters (edit here) ─────────────────────────────────────────
TIMEOUT      = 3600         # seconds; use 3600 for the full one-hour run
N_WORKERS    = 32
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "guoq_benchmark_advk_results"

# Checkpoints at which we record (total gates, CX gates) by interpolation.
CHECKPOINTS  = [10, 30, 60, 120, 180, 240, 360, 480, 600, 900, 1200, 1800, 2400, 3000, 3600]

# Best QALM config for test_greedy_k_ablation:
#   (initial_pool_size, exploration_pool_size, greedy_k, repeat_tolerance,
#    exploration_increase, strictly_reducing, only_do_local_transformations,
#    two_way_rm)
# greedy_k=2  →  greedy(k=1) + greedy(k=2) then advancing QALM at k=3,4,5,...
BEST_CONFIG = (1, 1, 2, 1.5, 0, 0, 1, 0)

# ── circuit lists ─────────────────────────────────────────────────────────────
_GUOQ_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")
GUOQ_CIRCUITS = sorted([
    (_GUOQ_DIR + "/" + f, os.path.splitext(f)[0], "guoq")
    for f in os.listdir(_GUOQ_DIR) if f.endswith(".qasm")
])

NAM_CIRCUITS = [
    ("circuit/nam_circs/adder_8.qasm",        "adder_8",        "nam"),
    ("circuit/nam_circs/barenco_tof_3.qasm",  "barenco_tof_3",  "nam"),
    ("circuit/nam_circs/barenco_tof_4.qasm",  "barenco_tof_4",  "nam"),
    ("circuit/nam_circs/barenco_tof_5.qasm",  "barenco_tof_5",  "nam"),
    ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10", "nam"),
    ("circuit/nam_circs/csla_mux_3.qasm",     "csla_mux_3",     "nam"),
    ("circuit/nam_circs/csum_mux_9.qasm",     "csum_mux_9",     "nam"),
    ("circuit/nam_circs/gf2^4_mult.qasm",     "gf2^4_mult",     "nam"),
    ("circuit/nam_circs/gf2^5_mult.qasm",     "gf2^5_mult",     "nam"),
    ("circuit/nam_circs/gf2^6_mult.qasm",     "gf2^6_mult",     "nam"),
    ("circuit/nam_circs/gf2^7_mult.qasm",     "gf2^7_mult",     "nam"),
    ("circuit/nam_circs/gf2^8_mult.qasm",     "gf2^8_mult",     "nam"),
    ("circuit/nam_circs/gf2^9_mult.qasm",     "gf2^9_mult",     "nam"),
    ("circuit/nam_circs/gf2^10_mult.qasm",    "gf2^10_mult",    "nam"),
    ("circuit/nam_circs/hwb6.qasm",           "hwb6",           "nam"),
    ("circuit/nam_circs/ham15-low.qasm",      "ham15-low",      "nam"),
    ("circuit/nam_circs/mod5_4.qasm",         "mod5_4",         "nam"),
    ("circuit/nam_circs/mod_mult_55.qasm",    "mod_mult_55",    "nam"),
    ("circuit/nam_circs/mod_red_21.qasm",     "mod_red_21",     "nam"),
    ("circuit/nam_circs/qcla_adder_10.qasm",  "qcla_adder_10",  "nam"),
    ("circuit/nam_circs/qcla_com_7.qasm",     "qcla_com_7",     "nam"),
    ("circuit/nam_circs/qcla_mod_7.qasm",     "qcla_mod_7",     "nam"),
    ("circuit/nam_circs/rc_adder_6.qasm",     "rc_adder_6",     "nam"),
    ("circuit/nam_circs/tof_3.qasm",          "tof_3",          "nam"),
    ("circuit/nam_circs/tof_4.qasm",          "tof_4",          "nam"),
    ("circuit/nam_circs/tof_5.qasm",          "tof_5",          "nam"),
    ("circuit/nam_circs/tof_10.qasm",         "tof_10",         "nam"),
    ("circuit/nam_circs/vbe_adder_3.qasm",    "vbe_adder_3",    "nam"),
]

NAM_RM_CIRCUITS = [
    ("circuit/nam_rm_circs/adder_8.qasm",        "adder_8",        "nam_rm"),
    ("circuit/nam_rm_circs/barenco_tof_3.qasm",  "barenco_tof_3",  "nam_rm"),
    ("circuit/nam_rm_circs/barenco_tof_4.qasm",  "barenco_tof_4",  "nam_rm"),
    ("circuit/nam_rm_circs/barenco_tof_5.qasm",  "barenco_tof_5",  "nam_rm"),
    ("circuit/nam_rm_circs/barenco_tof_10.qasm", "barenco_tof_10", "nam_rm"),
    ("circuit/nam_rm_circs/csla_mux_3.qasm",     "csla_mux_3",     "nam_rm"),
    ("circuit/nam_rm_circs/csum_mux_9.qasm",     "csum_mux_9",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^4_mult.qasm",     "gf2^4_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^5_mult.qasm",     "gf2^5_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^6_mult.qasm",     "gf2^6_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^7_mult.qasm",     "gf2^7_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^8_mult.qasm",     "gf2^8_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^9_mult.qasm",     "gf2^9_mult",     "nam_rm"),
    ("circuit/nam_rm_circs/gf2^10_mult.qasm",    "gf2^10_mult",    "nam_rm"),
    ("circuit/nam_rm_circs/hwb6.qasm",           "hwb6",           "nam_rm"),
    ("circuit/nam_rm_circs/ham15-low.qasm",      "ham15-low",      "nam_rm"),
    ("circuit/nam_rm_circs/mod5_4.qasm",         "mod5_4",         "nam_rm"),
    ("circuit/nam_rm_circs/mod_mult_55.qasm",    "mod_mult_55",    "nam_rm"),
    ("circuit/nam_rm_circs/mod_red_21.qasm",     "mod_red_21",     "nam_rm"),
    ("circuit/nam_rm_circs/qcla_adder_10.qasm",  "qcla_adder_10",  "nam_rm"),
    ("circuit/nam_rm_circs/qcla_com_7.qasm",     "qcla_com_7",     "nam_rm"),
    ("circuit/nam_rm_circs/qcla_mod_7.qasm",     "qcla_mod_7",     "nam_rm"),
    ("circuit/nam_rm_circs/rc_adder_6.qasm",     "rc_adder_6",     "nam_rm"),
    ("circuit/nam_rm_circs/tof_3.qasm",          "tof_3",          "nam_rm"),
    ("circuit/nam_rm_circs/tof_4.qasm",          "tof_4",          "nam_rm"),
    ("circuit/nam_rm_circs/tof_5.qasm",          "tof_5",          "nam_rm"),
    ("circuit/nam_rm_circs/tof_10.qasm",         "tof_10",         "nam_rm"),
    ("circuit/nam_rm_circs/vbe_adder_3.qasm",    "vbe_adder_3",    "nam_rm"),
]

ALL_CIRCUITS = NAM_CIRCUITS + NAM_RM_CIRCUITS + GUOQ_CIRCUITS


# ── helpers ───────────────────────────────────────────────────────────────────
def _set_mem_limit():
    """preexec_fn: set virtual-address-space hard limit to MEM_LIMIT_GB."""
    lim = MEM_LIMIT_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def _monitor(proc, mem_limit_kb, peak_ref, stop_evt):
    """Background thread: poll RSS every 0.5 s; kill proc if over limit."""
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
    """Return (total_gates, cx_gates) for the input circuit."""
    import qiskit
    try:
        qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
        total = qc.size()
        cx    = qc.count_ops().get("cx", 0)
        return total, cx
    except Exception:
        return -1, -1


def _interp_at(times, values, t):
    """Step-interpolate: last value at or before t, or first value if t < times[0]."""
    if not times:
        return None
    idx = next((i for i, ts in enumerate(times) if ts > t), len(times)) - 1
    return values[max(0, idx)]


# ── worker function ───────────────────────────────────────────────────────────
def run_one(args):
    circ_path, circ_name, source = args
    (n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr,
     local, two_way) = BEST_CONFIG

    pkl_path = f"{OUT_DIR}/pkl/{source}_{circ_name}_{TIMEOUT}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    # test_greedy_k_ablation: runs greedy(k=1..greedy_k), then advancing QALM
    # starting at k=greedy_k+1 and incrementing each iteration.
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
    peak_ref   = [0]
    stop_evt   = threading.Event()
    mon_thread = threading.Thread(
        target=_monitor, args=(proc, mem_limit_kb, peak_ref, stop_evt), daemon=True
    )
    mon_thread.start()

    # Hard wall-clock limit: TIMEOUT + 120s grace. The greedy phases check
    # their deadline only between iterations, so they may overrun by tens of
    # seconds on large circuits.
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

    peak_kb    = peak_ref[0]
    oom_killed = (proc.returncode not in (0, None))

    # Parse output: "[circ_name] Best cost: {total}  CX: {cx}  candidate number: {n}  after {t} seconds."
    times, totals, cxs = [], [], []
    for line in stdout.splitlines():
        words = line.split()
        if len(words) < 12:
            continue
        if words[0] == f"[{circ_name}]" and words[1] == "Best":
            try:
                totals.append(float(words[3]))   # total gate count
                cxs.append(int(words[5]))         # CX gate count
                times.append(float(words[10]))    # elapsed seconds
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


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    n_total = len(ALL_CIRCUITS)
    print(
        f"Benchmarking {n_total} circuits "
        f"({len(NAM_CIRCUITS)} nam + {len(NAM_RM_CIRCUITS)} nam_rm + {len(GUOQ_CIRCUITS)} guoq) "
        f"with {N_WORKERS} workers, timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, ALL_CIRCUITS)

    # ── write CSV ──────────────────────────────────────────────────────────────
    # Checkpoint columns: total_t{s} and cx_t{s} for each checkpoint
    chk_cols = []
    for t in CHECKPOINTS:
        chk_cols += [f"total_t{t}", f"cx_t{t}"]

    csv_path = f"{OUT_DIR}/results_{TIMEOUT}s.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source", "circuit",
             "orig_total", "orig_cx",
             "opt_total", "opt_cx",
             "total_reduction_pct", "cx_reduction_pct",
             "peak_rss_mb", "status"]
            + chk_cols
        )
        for (circ_path, circ_name, source), result in zip(ALL_CIRCUITS, results):
            times, totals, cxs, peak_kb, status = result
            orig_total, orig_cx = _orig_counts(circ_path)
            opt_total = int(totals[-1]) if totals else -1
            opt_cx    = int(cxs[-1])    if cxs    else -1

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            # Checkpoint values: step-interpolate the time series
            chk_vals = []
            for t in CHECKPOINTS:
                chk_vals.append(_interp_at(times, totals, t) or "")
                chk_vals.append(_interp_at(times, cxs, t) or "")

            writer.writerow(
                [source, circ_name,
                 orig_total, orig_cx,
                 opt_total, opt_cx,
                 pct(opt_total, orig_total), pct(opt_cx, orig_cx),
                 f"{peak_kb/1024:.1f}", status]
                + chk_vals
            )

    print(f"\nCSV saved → {csv_path}")

    # ── summary ────────────────────────────────────────────────────────────────
    for src_label in ["nam", "nam_rm", "guoq"]:
        subset = [
            (r, _orig_counts(c[0]))
            for c, r in zip(ALL_CIRCUITS, results)
            if c[2] == src_label
        ]
        ratios = [
            r[0][1][-1] / orig_total
            for r, (orig_total, orig_cx) in subset
            if r[0][1] and orig_total > 0
        ]
        if ratios:
            print(
                f"{src_label}: {len(ratios)}/{len(subset)} OK, "
                f"avg={np.mean(ratios):.4f}, geomean={np.exp(np.mean(np.log(ratios))):.4f}"
            )


if __name__ == "__main__":
    main()
