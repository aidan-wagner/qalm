"""
xor5_254 no-enqueue sweep, round 3.

Round 1 plateaued at 15 across 22 configs; round 2 (killed) showed fixed_k up
to 15 still stuck at 15 when N_branch=1. Round 3 drops N_branch=1 entirely
(collapses to greedy path under no-enqueue) and sweeps nb ∈ {2,3,5,8}.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/xor5_254_no_enqueue_v3.py
"""

import os
import pickle
import resource
import signal
import subprocess
import threading
import multiprocessing
import time

TIMEOUT      = 3600
N_WORKERS    = 8
MEM_LIMIT_GB = 4
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "xor5_254_no_enqueue_v3_results"

CIRC_PATH = os.path.expanduser("~/guoq/benchmarks/nam_rz/xor5_254.qasm")
CIRC_NAME = "xor5_254"

# (greedy_k, n_pool, n_branch, fixed_k, exploration_increase)
CONFIGS = [
    (0, 1, 2, 10, 0),   # minimal non-trivial width, deep single-shot
    (0, 1, 3, 10, 0),   # nb=3 single-shot
    (0, 1, 5, 10, 0),   # wider single-shot
    (0, 1, 8, 10, 0),   # very wide single-shot
    (0, 3, 5, 15, 0),   # wide + deep single-shot
    (0, 1, 5, 0,  1),   # wide advancing + exploration_increase
    (0, 3, 5, 0,  1),   # wider pool + branch + exp_incr
    (10, 1, 5, 0, 0),   # advance starts at k=11 + nb=5
]


def _label(cfg):
    gk, np_, nb, fk, ei = cfg
    mode = f"fk{fk}" if fk > 0 else f"adv{'_ei' if ei else ''}"
    return f"gk{gk}_np{np_}_nb{nb}_{mode}"


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


def run_one(cfg):
    gk, n_pool, n_branch, fixed_k, exp_incr = cfg
    label = _label(cfg)

    pkl_path = f"{OUT_DIR}/pkl/{label}_{TIMEOUT}.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            r = pickle.load(fh)
        totals = r[1]
        best = min(totals) if totals else -1
        print(f"  [cached] {label}: best={best}", flush=True)
        return label, r

    cmd = [
        "./build/test_greedy_k_ablation",
        CIRC_PATH, CIRC_NAME,
        str(TIMEOUT),
        str(gk),
        str(n_pool), str(n_branch),
        "1.5",              # repeat_tolerance
        str(exp_incr),      # exploration_increase
        "0",                # strictly_reducing_rules
        "1",                # only_do_local_transformations
        "0",                # two_way_rotation_merging
        ECCSET,
        str(fixed_k),       # fixed_k
        "0",                # enqueue_intermediate = 0
    ]

    t0 = time.time()
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

    wall_limit = TIMEOUT + 300
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

    elapsed = time.time() - t0
    peak_kb = peak_ref[0]
    oom_killed = (proc.returncode not in (0, None))

    times, totals, cxs = [], [], []
    for line in stdout.splitlines():
        words = line.split()
        if len(words) < 12:
            continue
        if words[0] == f"[{CIRC_NAME}]" and words[1] == "Best":
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

    best = min(totals) if totals else -1
    t_best = times[totals.index(min(totals))] if totals else -1

    print(
        f"  {label}: best={best} at t={t_best:.1f}s, "
        f"improvements={len(totals)}, peak={peak_kb/1024:.0f}MB, status={status}, "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )

    result = (times, totals, cxs, peak_kb, status)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    return label, result


def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    print(
        f"Launching {len(CONFIGS)} configs on {CIRC_NAME} with {N_WORKERS} workers, "
        f"timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB, enqueue_intermediate=OFF"
    )
    for cfg in CONFIGS:
        print(f"  {_label(cfg)}  cfg={cfg}")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, CONFIGS)

    print("\n=== SUMMARY (sorted by best, then time-to-best) ===")
    rows = []
    for label, (times, totals, cxs, peak_kb, status) in results:
        if totals:
            best = int(min(totals))
            t_best = times[totals.index(min(totals))]
        else:
            best = 999
            t_best = -1
        rows.append((best, t_best, label, status, peak_kb, len(totals)))
    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"{'label':<26} {'best':<6} {'t_best(s)':<12} {'status':<8} {'peak(MB)':<10} {'n_imp':<8}")
    for best, t_best, label, status, peak_kb, n_imp in rows:
        best_s = "FAIL" if best == 999 else str(best)
        print(f"{label:<26} {best_s:<6} {t_best:<12.1f} {status:<8} {peak_kb/1024:<10.0f} {n_imp:<8}")

    csv_path = f"{OUT_DIR}/summary_{TIMEOUT}s.csv"
    with open(csv_path, "w") as fh:
        fh.write("label,best_total,time_to_best_s,status,peak_mb,n_improvements\n")
        for best, t_best, label, status, peak_kb, n_imp in rows:
            best_s = "FAIL" if best == 999 else str(best)
            fh.write(f"{label},{best_s},{t_best:.1f},{status},{peak_kb/1024:.0f},{n_imp}\n")
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
