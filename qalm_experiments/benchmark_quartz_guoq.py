"""
Benchmark original Quartz (no ROQC, no greedy preprocessing, no 2-way RM)
on the 250 GUOQ Nam-gate-set circuits.

Binary: ./build/test_optimize
Args:   roqc_interval=-1, preprocess=0, two_way_rotation_merging=0

test_optimize stdout:
  - Progress lines:   "[circ] Best cost: <total>  candidate number: <N>  after <t> seconds."
  - Final QASM block: everything after the "Optimized graph:" marker.

We save the final optimized QASM per-circuit and parse it with qiskit to get
the exact final total- and CX-gate counts.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_quartz_guoq.py
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

TIMEOUT      = 3600
N_WORKERS    = 32
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "quartz_benchmark_results"

# test_optimize: roqc_interval, preprocess (greedy_start), two_way_rotation_merging
ROQC_INTERVAL = -1   # -1 disables ROQC
PREPROCESS    = 0    # no greedy preprocessing
TWO_WAY_RM    = 0    # no two-way rotation merging

_GUOQ_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")
GUOQ_CIRCUITS = sorted([
    (_GUOQ_DIR + "/" + f, os.path.splitext(f)[0], "guoq")
    for f in os.listdir(_GUOQ_DIR) if f.endswith(".qasm")
])


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


def _counts_from_qasm_str(qasm_str):
    import qiskit
    try:
        qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        return qc.size(), qc.count_ops().get("cx", 0)
    except Exception:
        return -1, -1


def run_one(args):
    circ_path, circ_name, source = args

    pkl_path  = f"{OUT_DIR}/pkl/{source}_{circ_name}_{TIMEOUT}.pkl"
    qasm_path = f"{OUT_DIR}/qasm/{source}_{circ_name}_{TIMEOUT}.qasm"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    cmd = [
        "./build/test_optimize",
        circ_path, circ_name,
        str(TIMEOUT),
        str(ROQC_INTERVAL),
        str(PREPROCESS),
        str(TWO_WAY_RM),
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

    # Split stdout into pre-QASM log and the final QASM block.
    qasm_marker = "Optimized graph:"
    if qasm_marker in stdout:
        log_part, qasm_part = stdout.split(qasm_marker, 1)
        final_qasm = qasm_part.strip()
    else:
        log_part, final_qasm = stdout, ""

    # Parse progress lines for the time series.
    # Format: "[circ] Best cost: <total>  candidate number: <N>  after <t> seconds."
    # idx:      0    1    2      3        4        5       6     7      8    9
    times, totals = [], []
    for line in log_part.splitlines():
        words = line.split()
        if len(words) < 10:
            continue
        if words[0] == f"[{circ_name}]" and words[1] == "Best":
            try:
                totals.append(float(words[3]))
                times.append(float(words[8]))
            except (ValueError, IndexError):
                pass

    # Save the final optimized QASM and extract true total/CX counts from it.
    if final_qasm:
        os.makedirs(os.path.dirname(qasm_path), exist_ok=True)
        with open(qasm_path, "w") as fh:
            fh.write(final_qasm)
        final_total, final_cx = _counts_from_qasm_str(final_qasm)
    else:
        final_total, final_cx = -1, -1

    if not totals and oom_killed:
        status = "OOM"
    elif not totals:
        status = "FAILED"
    else:
        status = "OK"

    print(
        f"[{source}] {circ_name}: {len(totals)} improvements, "
        f"final total={final_total} cx={final_cx}, "
        f"peak={peak_kb/1024:.0f} MB, status={status}",
        flush=True,
    )

    # pkl schema: (times, totals, final_total, final_cx, peak_kb, status)
    result = (times, totals, final_total, final_cx, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/qasm", exist_ok=True)
    circuits = GUOQ_CIRCUITS
    print(
        f"Benchmarking original Quartz on {len(circuits)} GUOQ circuits "
        f"with {N_WORKERS} workers, timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, circuits)

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
        for (circ_path, circ_name, source), result in zip(circuits, results):
            times, totals, final_total, final_cx, peak_kb, status = result
            orig_total, orig_cx = _orig_counts(circ_path)

            # For OOM circuits, fall back to the original gate counts
            # (i.e. 0% reduction) so downstream plots/averages don't drop them.
            if status == "OOM":
                final_total, final_cx = orig_total, orig_cx

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            writer.writerow(
                [source, circ_name,
                 orig_total, orig_cx,
                 final_total, final_cx,
                 pct(final_total, orig_total), pct(final_cx, orig_cx),
                 f"{peak_kb/1024:.1f}", status]
            )

    print(f"\nCSV saved → {csv_path}")

    ratios_total = [
        r[2] / _orig_counts(c[0])[0]
        for c, r in zip(circuits, results)
        if r[2] > 0 and _orig_counts(c[0])[0] > 0
    ]
    if ratios_total:
        print(
            f"guoq: {len(ratios_total)}/{len(circuits)} OK, "
            f"avg={np.mean(ratios_total):.4f}, "
            f"geomean={np.exp(np.mean(np.log(ratios_total))):.4f}"
        )


if __name__ == "__main__":
    main()
