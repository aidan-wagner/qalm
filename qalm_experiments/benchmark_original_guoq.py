"""
Benchmark pure Quartz (optimize_original — no ROQC, no preprocess prefix)
on the 250 GUOQ Nam-gate-set circuits.

Binary: ./build/test_original
Args:   <input_qasm> <circuit_name> <timeout_s> <ecc_set>

test_original stdout layout (after the patch):
  "number of xfers: <n>"
  "Post-greedy time: <t> seconds"
  "Post-greedy graph:"
  <intermediate QASM>
  "End post-greedy graph."
  "[circ] Best cost: <cost>  candidate number: <n>  after <t> seconds."  (repeated)
  "Optimized graph:"
  <final QASM>

If optimize_original OOMs (or is killed by the mem monitor), we still have the
post-greedy intermediate QASM + time — those are reported as the run's final
result so no circuit is lost.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/benchmark_original_guoq.py
"""

import csv
import multiprocessing
import os
import pickle
import resource
import signal
import subprocess
import threading

import numpy as np

TIMEOUT      = 3600
N_WORKERS    = 32
MEM_LIMIT_GB = 8
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "original_benchmark_results"

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


POST_GREEDY_TIME_MARK = "Post-greedy time: "
POST_GREEDY_BEG_MARK  = "Post-greedy graph:"
POST_GREEDY_END_MARK  = "End post-greedy graph."
FINAL_MARK            = "Optimized graph:"


def _parse_stdout(stdout):
    """Return (post_greedy_time, post_greedy_qasm, progress_times,
    progress_totals, final_qasm).

    Any field may be None/""/[] if the corresponding section never appeared
    (e.g. OOM before greedy finished).
    """
    # Post-greedy time
    post_greedy_time = None
    for line in stdout.splitlines():
        if line.startswith(POST_GREEDY_TIME_MARK):
            try:
                post_greedy_time = float(
                    line[len(POST_GREEDY_TIME_MARK):].split()[0]
                )
            except (ValueError, IndexError):
                pass
            break

    # Post-greedy QASM block
    post_greedy_qasm = ""
    if POST_GREEDY_BEG_MARK in stdout and POST_GREEDY_END_MARK in stdout:
        _, rest = stdout.split(POST_GREEDY_BEG_MARK, 1)
        qasm_block, _ = rest.split(POST_GREEDY_END_MARK, 1)
        post_greedy_qasm = qasm_block.strip()

    # Final QASM (after optimize_original)
    final_qasm = ""
    if FINAL_MARK in stdout:
        _, final_block = stdout.split(FINAL_MARK, 1)
        final_qasm = final_block.strip()

    # Progress lines come between POST_GREEDY_END_MARK and FINAL_MARK (if any)
    if POST_GREEDY_END_MARK in stdout:
        _, after_greedy = stdout.split(POST_GREEDY_END_MARK, 1)
    else:
        after_greedy = stdout
    if FINAL_MARK in after_greedy:
        progress_part, _ = after_greedy.split(FINAL_MARK, 1)
    else:
        progress_part = after_greedy

    times, totals = [], []
    for line in progress_part.splitlines():
        words = line.split()
        if len(words) < 10:
            continue
        if words[1:2] == ["Best"]:
            try:
                totals.append(float(words[3]))
                times.append(float(words[8]))
            except (ValueError, IndexError):
                pass

    return post_greedy_time, post_greedy_qasm, times, totals, final_qasm


def run_one(args):
    circ_path, circ_name, source = args

    pkl_path       = f"{OUT_DIR}/pkl/{source}_{circ_name}_{TIMEOUT}.pkl"
    final_qasm_path   = f"{OUT_DIR}/qasm/{source}_{circ_name}_{TIMEOUT}.qasm"
    greedy_qasm_path  = f"{OUT_DIR}/qasm_greedy/{source}_{circ_name}_{TIMEOUT}.qasm"

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    cmd = [
        "./build/test_original",
        circ_path, circ_name,
        str(TIMEOUT),
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
        target=_monitor, args=(proc, mem_limit_kb, peak_ref, stop_evt),
        daemon=True,
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

    (post_greedy_time, post_greedy_qasm,
     times, totals, final_qasm) = _parse_stdout(stdout)

    # Save the greedy snapshot QASM (if available)
    post_greedy_total, post_greedy_cx = -1, -1
    if post_greedy_qasm:
        os.makedirs(os.path.dirname(greedy_qasm_path), exist_ok=True)
        with open(greedy_qasm_path, "w") as fh:
            fh.write(post_greedy_qasm)
        post_greedy_total, post_greedy_cx = _counts_from_qasm_str(post_greedy_qasm)

    # Save the final QASM (if optimize_original reached completion)
    final_total, final_cx = -1, -1
    if final_qasm:
        os.makedirs(os.path.dirname(final_qasm_path), exist_ok=True)
        with open(final_qasm_path, "w") as fh:
            fh.write(final_qasm)
        final_total, final_cx = _counts_from_qasm_str(final_qasm)

    # Classify run
    if final_qasm:
        status = "OK"
    elif post_greedy_qasm and oom_killed:
        status = "OOM_AFTER_GREEDY"
    elif oom_killed:
        status = "OOM"
    else:
        status = "FAILED"

    print(
        f"[{source}] {circ_name}: greedy={post_greedy_total}/{post_greedy_cx} "
        f"@{post_greedy_time}s, final={final_total}/{final_cx}, "
        f"improv={len(totals)}, peak={peak_kb/1024:.0f} MB, status={status}",
        flush=True,
    )

    # pkl schema:
    # (post_greedy_time, post_greedy_total, post_greedy_cx,
    #  times, totals, final_total, final_cx, peak_kb, status)
    result = (post_greedy_time, post_greedy_total, post_greedy_cx,
              times, totals, final_total, final_cx, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def _write_csv(circuits, results):
    csv_path = f"{OUT_DIR}/results_{TIMEOUT}s.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source", "circuit",
             "orig_total", "orig_cx",
             "post_greedy_total", "post_greedy_cx", "post_greedy_time_s",
             "final_total", "final_cx",
             "opt_total", "opt_cx",
             "total_reduction_pct", "cx_reduction_pct",
             "peak_rss_mb", "status"]
        )
        for (circ_path, circ_name, source), result in zip(circuits, results):
            (post_greedy_time, post_greedy_total, post_greedy_cx,
             times, totals, final_total, final_cx, peak_kb, status) = result
            orig_total, orig_cx = _orig_counts(circ_path)

            # opt_total / opt_cx fall back to greedy snapshot if optimize_original
            # never finished; if even greedy didn't produce output, fall back to
            # original gate counts (0% reduction).
            if final_total > 0:
                opt_total, opt_cx = final_total, final_cx
            elif post_greedy_total > 0:
                opt_total, opt_cx = post_greedy_total, post_greedy_cx
            else:
                opt_total, opt_cx = orig_total, orig_cx

            def pct(opt, orig):
                return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

            writer.writerow(
                [source, circ_name,
                 orig_total, orig_cx,
                 post_greedy_total, post_greedy_cx,
                 f"{post_greedy_time:.2f}" if post_greedy_time is not None else "nan",
                 final_total, final_cx,
                 opt_total, opt_cx,
                 pct(opt_total, orig_total), pct(opt_cx, orig_cx),
                 f"{peak_kb/1024:.1f}", status]
            )
    print(f"CSV saved → {csv_path}")


def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/qasm", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/qasm_greedy", exist_ok=True)
    circuits = GUOQ_CIRCUITS
    print(
        f"Benchmarking pure Quartz (optimize_original) on {len(circuits)} "
        f"GUOQ circuits with {N_WORKERS} workers, timeout={TIMEOUT}s, "
        f"mem_limit={MEM_LIMIT_GB}GB"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, circuits)

    _write_csv(circuits, results)

    # Quick summary
    ratios = []
    for (circ_path, _, _), r in zip(circuits, results):
        orig_total, _ = _orig_counts(circ_path)
        final_total = r[5] if r[5] > 0 else (r[1] if r[1] > 0 else -1)
        if final_total > 0 and orig_total > 0:
            ratios.append(final_total / orig_total)
    if ratios:
        print(
            f"guoq: {len(ratios)}/{len(circuits)} accounted, "
            f"avg_ratio={np.mean(ratios):.4f}, "
            f"geomean={np.exp(np.mean(np.log(ratios))):.4f}"
        )


if __name__ == "__main__":
    main()
