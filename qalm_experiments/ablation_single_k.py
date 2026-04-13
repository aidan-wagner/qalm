"""
Single-k ablation study: fixed exploration depth (no k-advancement).

Two groups of experiments, each with 5 configs (fixed k=1,2,3,4,5):
  Group 1: greedy_k=0 (no greedy preprocessing)
  Group 2: greedy_k=2 (greedy k=1,2 first)

32 workers, 1h timeout, 8GB mem limit, 26 circuits.

Run from repo root:
    python3 qalm_experiments/ablation_single_k.py
"""

import os
import pickle
import resource
import signal
import subprocess
import threading

import multiprocessing
import numpy as np

# ── benchmark circuits (26) ────────────────────────────────────────────────────
CIRCUIT_LIST = [
    ("circuit/nam_circs/adder_8.qasm", "adder_8"),
    ("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
    ("circuit/nam_circs/barenco_tof_4.qasm", "barenco_tof_4"),
    ("circuit/nam_circs/barenco_tof_5.qasm", "barenco_tof_5"),
    ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
    ("circuit/nam_circs/csla_mux_3.qasm", "csla_mux_3"),
    ("circuit/nam_circs/csum_mux_9.qasm", "csum_mux_9"),
    ("circuit/nam_circs/gf2^4_mult.qasm", "gf2^4_mult"),
    ("circuit/nam_circs/gf2^5_mult.qasm", "gf2^5_mult"),
    ("circuit/nam_circs/gf2^6_mult.qasm", "gf2^6_mult"),
    ("circuit/nam_circs/gf2^7_mult.qasm", "gf2^7_mult"),
    ("circuit/nam_circs/gf2^8_mult.qasm", "gf2^8_mult"),
    ("circuit/nam_circs/gf2^9_mult.qasm", "gf2^9_mult"),
    ("circuit/nam_circs/gf2^10_mult.qasm", "gf2^10_mult"),
    ("circuit/nam_circs/mod5_4.qasm", "mod5_4"),
    ("circuit/nam_circs/mod_mult_55.qasm", "mod_mult_55"),
    ("circuit/nam_circs/mod_red_21.qasm", "mod_red_21"),
    ("circuit/nam_circs/qcla_adder_10.qasm", "qcla_adder_10"),
    ("circuit/nam_circs/qcla_com_7.qasm", "qcla_com_7"),
    ("circuit/nam_circs/qcla_mod_7.qasm", "qcla_mod_7"),
    ("circuit/nam_circs/rc_adder_6.qasm", "rc_adder_6"),
    ("circuit/nam_circs/tof_3.qasm", "tof_3"),
    ("circuit/nam_circs/tof_4.qasm", "tof_4"),
    ("circuit/nam_circs/tof_5.qasm", "tof_5"),
    ("circuit/nam_circs/tof_10.qasm", "tof_10"),
    ("circuit/nam_circs/vbe_adder_3.qasm", "vbe_adder_3"),
]

FIXED_K_VALUES = [1, 2, 3, 4, 5]
GREEDY_K_GROUPS = [0, 2]

TIMEOUT = 3600  # seconds
N_WORKERS = 32
MEM_LIMIT_GB = 8
ECCSET = "eccset/Nam_5_3_complete_ECC_set.json"

N_POOL = 1
N_BRANCH = 1
REPEAT_TOL = 1.5
EXP_INCREASE = 0
STRICTLY_REDUCING = 0
LOCAL_ONLY = 1
TWO_WAY_RM = 0

RESULTS_DIR = "single_k_results"


# ── helpers ────────────────────────────────────────────────────────────────────

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


def run_one(args):
    """Run test_greedy_k_ablation with a fixed exploration k."""
    circ_path, circ_name, greedy_k, fixed_k = args

    pkl_path = os.path.join(
        RESULTS_DIR, "pkl",
        f"{circ_name}_gk{greedy_k}_fk{fixed_k}_{TIMEOUT}.pkl",
    )
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    proc = subprocess.Popen(
        [
            "./build/test_greedy_k_ablation",
            circ_path, circ_name,
            str(TIMEOUT),
            str(greedy_k),
            str(N_POOL), str(N_BRANCH),
            str(REPEAT_TOL),
            str(int(EXP_INCREASE)),
            str(int(STRICTLY_REDUCING)),
            str(int(LOCAL_ONLY)),
            str(int(TWO_WAY_RM)),
            ECCSET,
            str(fixed_k),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_set_mem_limit,
    )

    mem_limit_kb = MEM_LIMIT_GB * 1024 * 1024
    peak_ref = [0]
    stop_evt = threading.Event()
    mon_thread = threading.Thread(
        target=_monitor,
        args=(proc, mem_limit_kb, peak_ref, stop_evt),
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

    peak_kb = peak_ref[0]
    times, costs = [], []
    qasm_lines = []
    in_qasm = False

    prefix = f"[{circ_name}]"
    for line in stdout.splitlines():
        if in_qasm:
            qasm_lines.append(line)
            continue
        if line.startswith("OPENQASM"):
            in_qasm = True
            qasm_lines.append(line)
            continue
        words = line.split()
        if not words:
            continue
        if words[0] == prefix:
            if len(words) >= 2 and words[1] == "[QALM_K_START]":
                continue
            try:
                costs.append(float(words[3]))
                after_idx = words.index("after")
                times.append(float(words[after_idx + 1]))
            except (IndexError, ValueError):
                pass

    print(
        f"{circ_name} gk={greedy_k} fk={fixed_k}: "
        f"{len(costs)} improvements, peak={peak_kb / 1024:.0f} MB",
        flush=True,
    )

    out = (times, costs, peak_kb)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)

    # Save optimized QASM
    if qasm_lines:
        qasm_dir = os.path.join(RESULTS_DIR, "qasm")
        os.makedirs(qasm_dir, exist_ok=True)
        qasm_path = os.path.join(
            qasm_dir,
            f"{circ_name}_gk{greedy_k}_fk{fixed_k}_{TIMEOUT}.qasm",
        )
        with open(qasm_path, "w") as f:
            f.write("\n".join(qasm_lines) + "\n")

    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(RESULTS_DIR, "pkl"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "qasm"), exist_ok=True)

    tasks = [
        (circ_path, circ_name, gk, fk)
        for gk in GREEDY_K_GROUPS
        for (circ_path, circ_name) in CIRCUIT_LIST
        for fk in FIXED_K_VALUES
    ]

    n_tasks = len(tasks)
    print(
        f"Launching {n_tasks} tasks "
        f"({len(CIRCUIT_LIST)} circuits × {len(FIXED_K_VALUES)} k values "
        f"× {len(GREEDY_K_GROUPS)} greedy_k groups) "
        f"with {N_WORKERS} workers, timeout={TIMEOUT}s, "
        f"mem_limit={MEM_LIMIT_GB}GB …"
    )

    with multiprocessing.Pool(N_WORKERS) as pool:
        flat_results = pool.map(run_one, tasks)

    print(f"\nAll {n_tasks} tasks completed. Results pickled in {RESULTS_DIR}/pkl/")
    print("Run plot_single_k.py to generate figures.")


if __name__ == "__main__":
    main()
