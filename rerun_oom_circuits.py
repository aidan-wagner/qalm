"""
Re-run circuits that were killed by the 8GB RLIMIT_AS before 3600s,
this time with a 32GB memory limit.

Base (test_qalm) killed circuits are known:
  - grover_pigeon3, rd73_252, misex1_241

Advk (test_greedy_k_ablation) and Quartz (test_optimize) killed circuits are
detected at runtime by scanning pkl files for peak RSS > 7.5GB AND last log
time < 3590s.

Overwrites the old pkl files so downstream plotting picks up the new results.

Run from repo root:
    python3 rerun_oom_circuits.py                  # re-run base + advk
    python3 rerun_oom_circuits.py --quartz-only    # re-run only Quartz OOMs
"""

import glob
import os
import pickle
import resource
import signal
import subprocess
import sys
import threading

TIMEOUT = 3600
MEM_LIMIT_GB = 32
ECCSET = "eccset/Nam_5_3_complete_ECC_set.json"

GUOQ_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")

# ── configs ──────────────────────────────────────────────────────────────────
# Base: test_qalm with (n_pool, n_branch, k, rep_tol, exp_incr, no_incr,
#                        local, greedy, two_way)
BASE_CONFIG = (1, 1, 3, 1.5, 0, 0, 1, 1, 0)
BASE_OUT_DIR = "guoq_benchmark_results"
BASE_CIRCUITS = ["grover_pigeon3", "rd73_252", "misex1_241"]

# Advk: test_greedy_k_ablation with (n_pool, n_branch, greedy_k, rep_tol,
#                                     exp_incr, no_incr, local, two_way)
ADVK_CONFIG = (1, 1, 2, 1.5, 0, 0, 1, 0)
ADVK_OUT_DIR = "guoq_benchmark_advk_results"

# Quartz: test_optimize with (roqc_interval, preprocess, two_way_rm)
QUARTZ_CONFIG = (-1, 0, 0)
QUARTZ_OUT_DIR = "quartz_benchmark_results"


def _set_mem_limit():
    lim = MEM_LIMIT_GB * 1024**3
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


def _launch(cmd):
    """Launch a subprocess with RLIMIT_AS set, plus an RSS monitor."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, preexec_fn=_set_mem_limit,
    )
    mem_limit_kb = MEM_LIMIT_GB * 1024 * 1024
    peak_ref = [0]
    stop_evt = threading.Event()
    mon = threading.Thread(
        target=_monitor, args=(proc, mem_limit_kb, peak_ref, stop_evt),
        daemon=True,
    )
    mon.start()

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
    mon.join()
    return stdout, proc.returncode, peak_ref[0]


def run_qalm_circuit(cmd, circ_name, pkl_path):
    """Run test_qalm / test_greedy_k_ablation and save pkl (with CX column)."""
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Output:  {pkl_path}")

    stdout, returncode, peak_kb = _launch(cmd)

    # Format: "[circ] Best cost: <t>\tCX: <c>\tcandidate number: <n>\tafter <s> seconds."
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

    oom_killed = returncode not in (0, None)
    if not totals and oom_killed:
        status = "OOM"
    elif not totals:
        status = "FAILED"
    else:
        status = "OK"

    print(f"  Result: {len(totals)} improvements, peak={peak_kb/1024:.0f} MB, "
          f"status={status}, returncode={returncode}")
    if totals:
        print(f"  Final total={totals[-1]:.0f}, last_time={times[-1]:.1f}s")

    result = (times, totals, cxs, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def run_quartz_circuit(cmd, circ_name, pkl_path, qasm_path):
    """Run test_optimize: save pkl (6-tuple schema) and the final QASM."""
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Output:  {pkl_path}")
    print(f"  QASM:    {qasm_path}")

    stdout, returncode, peak_kb = _launch(cmd)

    # Split stdout into progress log and the final QASM block.
    qasm_marker = "Optimized graph:"
    if qasm_marker in stdout:
        log_part, qasm_part = stdout.split(qasm_marker, 1)
        final_qasm = qasm_part.strip()
    else:
        log_part, final_qasm = stdout, ""

    # Parse progress lines. Format: no CX column.
    # "[circ] Best cost: <t>  candidate number: <n>  after <s> seconds."
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

    final_total, final_cx = -1, -1
    if final_qasm:
        os.makedirs(os.path.dirname(qasm_path), exist_ok=True)
        with open(qasm_path, "w") as fh:
            fh.write(final_qasm)
        try:
            import qiskit
            qc = qiskit.QuantumCircuit.from_qasm_str(final_qasm)
            final_total = qc.size()
            final_cx = qc.count_ops().get("cx", 0)
        except Exception as exc:
            print(f"  (qiskit parse error: {exc})")

    oom_killed = returncode not in (0, None)
    if not totals and oom_killed:
        status = "OOM"
    elif not totals:
        status = "FAILED"
    else:
        status = "OK"

    print(f"  Result: {len(totals)} improvements, final total={final_total} cx={final_cx}, "
          f"peak={peak_kb/1024:.0f} MB, status={status}, returncode={returncode}")

    # Quartz pkl schema: (times, totals, final_total, final_cx, peak_kb, status)
    result = (times, totals, final_total, final_cx, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def _strip_source_prefix(fname):
    for prefix in ("guoq_", "nam_rm_", "nam_"):
        if fname.startswith(prefix):
            return fname[len(prefix):]
    return fname


def detect_killed_circuits(pkl_dir, peak_gb_threshold=7.5, time_threshold=3590,
                            schema="qalm"):
    """Scan pkl files and return circuit names that were likely killed by OOM.

    schema="qalm":   (times, totals, cxs, peak_kb, status)  — 5-tuple
    schema="quartz": (times, totals, final_total, final_cx, peak_kb, status)  — 6-tuple
    """
    killed = []
    for p in sorted(glob.glob(f"{pkl_dir}/*_{TIMEOUT}.pkl")):
        with open(p, "rb") as f:
            data = pickle.load(f)
        if schema == "quartz":
            times, totals, _final_t, _final_c, peak_kb, status = data
        else:
            times, totals, _cxs, peak_kb, status = data
        if not times:
            continue
        peak_gb = peak_kb / 1024 / 1024
        last_time = times[-1]
        if last_time < time_threshold and peak_gb > peak_gb_threshold:
            fname = os.path.basename(p).replace(f"_{TIMEOUT}.pkl", "")
            circ_name = _strip_source_prefix(fname)
            lost = TIMEOUT - last_time
            killed.append((circ_name, peak_gb, last_time, lost))
            print(f"  Detected: {circ_name} (peak={peak_gb:.2f}GB, "
                  f"last_log={last_time:.1f}s, lost={lost:.0f}s)")
    return killed


def rerun_base():
    n_pool, n_branch, k, rep_tol, exp_incr, no_incr, local, greedy, two_way = BASE_CONFIG
    print(f"--- Base (test_qalm), {len(BASE_CIRCUITS)} known killed circuits ---")
    for circ_name in BASE_CIRCUITS:
        circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
        pkl_path = f"{BASE_OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
        cmd = [
            "./build/test_qalm",
            circ_path, circ_name, str(TIMEOUT),
            str(n_pool), str(n_branch), str(k),
            str(rep_tol),
            str(int(exp_incr)), str(int(no_incr)),
            str(int(local)), str(int(greedy)), str(int(two_way)),
            ECCSET,
        ]
        run_qalm_circuit(cmd, circ_name, pkl_path)
        print()


def rerun_advk():
    n_pool, n_branch, greedy_k, rep_tol, exp_incr, no_incr, local, two_way = ADVK_CONFIG
    print(f"--- Advk (test_greedy_k_ablation), detecting killed circuits ---")
    advk_killed = detect_killed_circuits(f"{ADVK_OUT_DIR}/pkl", schema="qalm")
    if not advk_killed:
        print("  No killed circuits detected.")
        return
    print(f"  Re-running {len(advk_killed)} circuits...\n")
    for circ_name, *_ in advk_killed:
        circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
        pkl_path = f"{ADVK_OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
        cmd = [
            "./build/test_greedy_k_ablation",
            circ_path, circ_name, str(TIMEOUT),
            str(greedy_k),
            str(n_pool), str(n_branch),
            str(rep_tol),
            str(int(exp_incr)), str(int(no_incr)),
            str(int(local)), str(int(two_way)),
            ECCSET,
        ]
        run_qalm_circuit(cmd, circ_name, pkl_path)
        print()


def rerun_quartz():
    roqc_interval, preprocess, two_way_rm = QUARTZ_CONFIG
    print(f"--- Quartz (test_optimize), detecting killed circuits ---")
    q_killed = detect_killed_circuits(f"{QUARTZ_OUT_DIR}/pkl", schema="quartz")
    if not q_killed:
        print("  No killed circuits detected.")
        return
    print(f"  Re-running {len(q_killed)} circuits...\n")
    for circ_name, *_ in q_killed:
        circ_path = f"{GUOQ_DIR}/{circ_name}.qasm"
        pkl_path  = f"{QUARTZ_OUT_DIR}/pkl/guoq_{circ_name}_{TIMEOUT}.pkl"
        qasm_path = f"{QUARTZ_OUT_DIR}/qasm/guoq_{circ_name}_{TIMEOUT}.qasm"
        cmd = [
            "./build/test_optimize",
            circ_path, circ_name, str(TIMEOUT),
            str(roqc_interval), str(preprocess), str(two_way_rm),
            ECCSET,
        ]
        run_quartz_circuit(cmd, circ_name, pkl_path, qasm_path)
        print()


def main():
    quartz_only = "--quartz-only" in sys.argv
    print(f"=== Re-running OOM circuits with {MEM_LIMIT_GB}GB limit ===\n")

    if quartz_only:
        rerun_quartz()
    else:
        rerun_base()
        rerun_advk()

    print("=== Done ===")


if __name__ == "__main__":
    main()
