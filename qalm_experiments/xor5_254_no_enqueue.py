"""
xor5_254 no-enqueue config sweep.

Hunt for no-enqueue QALM configs that reach 7 gates on xor5_254 within 1 hour.
Context: greedy-0 with enqueue=1 reaches 7 gates in ~183s. We want to know which
enqueue_intermediate=0 configs can still reach that minimum, and how fast.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/xor5_254_no_enqueue.py
"""

import os
import pickle
import resource
import signal
import subprocess
import threading
import multiprocessing
import time

# ── settings ─────────────────────────────────────────────────────────────────
TIMEOUT      = 3600
N_WORKERS    = 32
MEM_LIMIT_GB = 4
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "xor5_254_no_enqueue_results"

CIRC_PATH = os.path.expanduser("~/guoq/benchmarks/nam_rz/xor5_254.qasm")
CIRC_NAME = "xor5_254"

# Configs: (greedy_k, n_pool, n_branch, fixed_k)
#   fixed_k=0 → advancing (k starts at greedy_k+1 and increments each iter)
#   fixed_k>0 → QALM runs once at that fixed exploration depth
CONFIGS = []
# Group A: vary greedy_k with (1,1) advancing
for gk in [0, 1, 2, 3]:
    CONFIGS.append((gk, 1, 1, 0))
# Group B: vary greedy_k with (3,3) advancing
for gk in [0, 1, 2]:
    CONFIGS.append((gk, 3, 3, 0))
# Group C: gk=0 advancing, other pool/branch shapes
for np_, nb in [(1, 3), (3, 1), (5, 5)]:
    CONFIGS.append((0, np_, nb, 0))
# Group D: gk=0, (1,1), fixed_k sweep
for fk in [1, 2, 3, 4, 5]:
    CONFIGS.append((0, 1, 1, fk))
# Group E: gk=1 advancing, pool/branch shapes
for np_, nb in [(1, 3), (3, 1), (5, 5)]:
    CONFIGS.append((1, np_, nb, 0))
# Group F: gk=2, (3,3), fixed_k
for fk in [3, 4]:
    CONFIGS.append((2, 3, 3, fk))
# Group G: gk=0, (3,3), fixed_k
for fk in [2, 3]:
    CONFIGS.append((0, 3, 3, fk))


def _config_label(cfg):
    gk, np_, nb, fk = cfg
    mode = f"fk{fk}" if fk > 0 else "adv"
    return f"gk{gk}_np{np_}_nb{nb}_{mode}"


# ── helpers (adapted from benchmark_guoq.py) ─────────────────────────────────

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
    gk, n_pool, n_branch, fixed_k = cfg
    label = _config_label(cfg)

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
        "0",                # exploration_increase
        "0",                # strictly_reducing_rules
        "1",                # only_do_local_transformations
        "0",                # two_way_rotation_merging
        ECCSET,
        str(fixed_k),       # fixed_k
        "0",                # enqueue_intermediate = 0 (no-enqueue)
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

    elapsed = time.time() - t0
    peak_kb = peak_ref[0]
    oom_killed = (proc.returncode not in (0, None))

    # Parse "[xor5_254] Best cost: {total}  CX: {cx}  candidate number: {n}  after {t} seconds."
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(f"{OUT_DIR}/pkl", exist_ok=True)

    print(
        f"Launching {len(CONFIGS)} configs on {CIRC_NAME} with {N_WORKERS} workers, "
        f"timeout={TIMEOUT}s, mem_limit={MEM_LIMIT_GB}GB, enqueue_intermediate=OFF"
    )
    for cfg in CONFIGS:
        print(f"  {_config_label(cfg)}")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_one, CONFIGS)

    # Summary: sort by best, then by time-to-best
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
    print(f"{'label':<22} {'best':<6} {'t_best(s)':<12} {'status':<8} {'peak(MB)':<10} {'n_imp':<8}")
    for best, t_best, label, status, peak_kb, n_imp in rows:
        best_s = "FAIL" if best == 999 else str(best)
        print(f"{label:<22} {best_s:<6} {t_best:<12.1f} {status:<8} {peak_kb/1024:<10.0f} {n_imp:<8}")

    # Write a CSV summary too
    csv_path = f"{OUT_DIR}/summary_{TIMEOUT}s.csv"
    with open(csv_path, "w") as fh:
        fh.write("label,best_total,time_to_best_s,status,peak_mb,n_improvements\n")
        for best, t_best, label, status, peak_kb, n_imp in rows:
            best_s = "FAIL" if best == 999 else str(best)
            fh.write(f"{label},{best_s},{t_best:.1f},{status},{peak_kb/1024:.0f},{n_imp}\n")
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
