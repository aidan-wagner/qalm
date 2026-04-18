"""
Single config run on xor5_254 with the new termination-on-queue-degradation check.
Config: gk=2, np=1, nb=3, advancing, enqueue_intermediate=0.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/xor5_254_single_adv.py
"""

import os
import pickle
import resource
import signal
import subprocess
import threading
import time

TIMEOUT      = 3600
MEM_LIMIT_GB = 4
ECCSET       = "eccset/Nam_5_3_complete_ECC_set.json"
OUT_DIR      = "xor5_254_single_adv_plus2_results"

CIRC_PATH = os.path.expanduser("~/guoq/benchmarks/nam_rz/xor5_254.qasm")
CIRC_NAME = "xor5_254"

# (greedy_k, n_pool, n_branch, fixed_k, exploration_increase, enqueue_intermediate)
CFG = (2, 1, 3, 0, 0, 0)


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gk, n_pool, n_branch, fixed_k, exp_incr, enq = CFG
    label = f"gk{gk}_np{n_pool}_nb{n_branch}_adv_enq{enq}"
    stdout_path = os.path.join(OUT_DIR, f"{label}.stdout")
    pkl_path    = os.path.join(OUT_DIR, f"{label}_{TIMEOUT}.pkl")

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
        str(fixed_k),       # fixed_k = 0 → advancing
        str(enq),           # enqueue_intermediate
    ]
    print(f"Launching {label}")
    print("CMD:", " ".join(cmd))

    t0 = time.time()
    with open(stdout_path, "w") as stdout_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_fh,
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
            _, err = proc.communicate(timeout=wall_limit)
        except subprocess.TimeoutExpired:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, err = proc.communicate()
        stop_evt.set()
        mon.join()

    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.1f}s rc={proc.returncode} peak={peak_ref[0]/1024:.0f}MB")
    if err and err.strip():
        print("STDERR tail:", err[-300:])

    # Parse stdout (re-read to produce the pickle + summary)
    times, totals, cxs = [], [], []
    k_starts = []
    term_events = 0
    with open(stdout_path) as fh:
        for line in fh:
            words = line.split()
            if not words:
                continue
            if words[0] == f"[{CIRC_NAME}]" and len(words) >= 12 and words[1] == "Best":
                try:
                    totals.append(float(words[3]))
                    cxs.append(int(words[5]))
                    times.append(float(words[10]))
                except (ValueError, IndexError):
                    pass
            if "QALM_K_START" in line:
                k_starts.append(line.strip())
            if "queue degraded" in line:
                term_events += 1

    best = int(min(totals)) if totals else -1
    t_best = times[totals.index(min(totals))] if totals else -1
    print(f"best={best} at t={t_best:.1f}s, improvements={len(totals)}, "
          f"QALM_K_START events={len(k_starts)}, queue-degraded terminations={term_events}")
    if k_starts:
        # Show largest k reached
        import re
        ks = [int(re.search(r'k=(\d+)', s).group(1)) for s in k_starts]
        print(f"k range: [{min(ks)}, {max(ks)}], last 3 k values: {ks[-3:]}")

    with open(pkl_path, "wb") as fh:
        pickle.dump((times, totals, cxs, peak_ref[0], proc.returncode), fh)
    print(f"pkl -> {pkl_path}")
    print(f"stdout -> {stdout_path}")


if __name__ == "__main__":
    main()
