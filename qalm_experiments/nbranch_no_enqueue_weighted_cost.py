"""
Weighted-cost study: cost_function = 1Q + 10 * 2Q.

Single config — npool=1, nbranch=3, no intermediate enqueue, advancing k,
cost_mode=1 (new weighted cost added via test_greedy_k_ablation argv[15]).

Plots report both (a) gate count reduction (the traditional metric) and
(b) weighted-cost reduction (the metric actually driving the search).

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/nbranch_no_enqueue_weighted_cost.py
"""

import os
import pickle
import resource
import subprocess
import multiprocessing
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qiskit

# -- settings ------------------------------------------------------------------
TIMEOUT = 3600
ECCSET = "eccset/Nam_5_3_complete_ECC_set.json"
N_WORKERS = 32
MEM_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB

N_POOL = 1
N_BRANCH = 3
COST_MODE = 1  # 1 = weighted 1Q + 10*2Q

OUT_DIR = "nbranch_no_enqueue_weighted_cost_results"

# 26 circuits (hwb6 and ham15-low excluded — consistent with other study plots)
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

TWO_Q_GATES = {"cx", "cz", "cy", "ch", "cp", "crx", "cry", "crz", "cu", "cu1",
               "cu3", "swap", "iswap", "rxx", "ryy", "rzz", "rzx"}


# -- helpers -------------------------------------------------------------------

def _set_mem_limit():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))


def _parse_output(stdout, circuit_name):
    """Parse [circuit] lines. With COST_MODE=1 the reported cost is weighted."""
    times, costs = [], []
    prefix = f"[{circuit_name}]"
    circuit_found = False
    circuit_string = ""

    for line in stdout.splitlines():
        words = line.split()
        if not words:
            if circuit_found:
                circuit_string += "\n"
            continue
        if words[0] == prefix:
            try:
                cost = float(words[3])
            except (IndexError, ValueError):
                continue
            if len(words) > 4 and words[4] == "CX:":
                try:
                    t = float(words[10])
                except (IndexError, ValueError):
                    continue
            else:
                try:
                    t = float(words[8])
                except (IndexError, ValueError):
                    continue
            costs.append(cost)
            times.append(t)
        if words[0] == "OPENQASM":
            circuit_found = True
        if circuit_found:
            circuit_string += (line + "\n")

    return times, costs, circuit_string


def run_qalm(filename, circuit_name):
    result = subprocess.run(
        [
            "./build/test_greedy_k_ablation",
            filename,
            circuit_name,
            str(TIMEOUT),
            "2",               # greedy_k
            str(N_POOL),
            str(N_BRANCH),
            "1.5",             # repeat_tolerance
            "0",               # exploration_increase
            "0",               # strictly_reducing_rules
            "1",               # only_do_local_transformations
            "0",               # two_way_rotation_merging
            ECCSET,
            "0",               # fixed_k=0 (advancing)
            "0",               # enqueue_intermediate=0
            str(COST_MODE),    # NEW: cost mode (1 = weighted 1Q + 10*2Q)
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT + 600,
    )
    return _parse_output(result.stdout, circuit_name)


# -- dispatch ------------------------------------------------------------------

def dispatch(args):
    filename, circuit_name = args
    pkl_dir = os.path.join(OUT_DIR, "pkl")
    pkl_path = os.path.join(pkl_dir, f"{circuit_name}_{TIMEOUT}.pkl")
    qasm_dir = os.path.join(OUT_DIR, "qasm")
    qasm_path = os.path.join(qasm_dir, f"{circuit_name}.qasm")

    if os.path.exists(pkl_path):
        print(f"  [weighted] {circuit_name}: cached", flush=True)
        with open(pkl_path, "rb") as f:
            return circuit_name, pickle.load(f)

    try:
        times, costs, qasm_str = run_qalm(filename, circuit_name)
    except Exception:
        traceback.print_exc()
        return circuit_name, ([], [])

    with open(pkl_path, "wb") as f:
        pickle.dump((times, costs), f)
    if qasm_str.strip():
        with open(qasm_path, "w") as f:
            f.write(qasm_str)

    final = int(costs[-1]) if costs else -1
    print(f"  [weighted] {circuit_name}: final weighted_cost={final}", flush=True)
    return circuit_name, (times, costs)


# -- post-processing ----------------------------------------------------------

def count_gates(qasm_path):
    """Return (total_gates, two_q_gates, weighted_cost) for a QASM file."""
    if not os.path.exists(qasm_path):
        return None
    try:
        qc = qiskit.QuantumCircuit.from_qasm_file(qasm_path)
    except Exception:
        return None
    total = 0
    two_q = 0
    for instr, qargs, _ in qc.data:
        if instr.name in ("barrier", "measure"):
            continue
        total += 1
        if len(qargs) >= 2:
            two_q += 1
    weighted = (total - two_q) + two_q * 10
    return total, two_q, weighted


def get_original_stats():
    stats = {}
    for filename, name in CIRCUIT_LIST:
        s = count_gates(filename)
        if s is None:
            continue
        stats[name] = s
    return stats


# -- plotting ------------------------------------------------------------------

def make_plots(results, orig_stats):
    common_times = np.linspace(0, TIMEOUT, 600)

    # Weighted-cost reduction over time (the metric the search optimizes)
    fig, ax = plt.subplots(figsize=(7, 4))
    log_ratios = []
    for cname, (times, costs) in results.items():
        if cname not in orig_stats or not times:
            continue
        orig_w = orig_stats[cname][2]
        pt = ([0.0] + list(times)) if times[0] > 0 else list(times)
        pc = ([float(orig_w)] + list(costs)) if times[0] > 0 else list(costs)
        lr = [np.log(max(c, 1) / orig_w) for c in pc]
        interp = np.interp(common_times, pt, lr, left=lr[0], right=lr[-1])
        log_ratios.append(interp)
    if log_ratios:
        mean_log = np.mean(log_ratios, axis=0)
        geomean_red = (1.0 - np.exp(mean_log)) * 100.0
        ax.plot(common_times, geomean_red, color="tab:blue", linewidth=1.8,
                label="weighted cost (1Q + 10*2Q)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Geo-mean weighted-cost reduction (%)")
    ax.set_title(f"Weighted cost, npool={N_POOL}, nbranch={N_BRANCH} "
                 f"({len(results)} circuits)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"weighted_geomean_{TIMEOUT}s.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def print_summary(results, orig_stats):
    print("\n=== Weighted-cost final values ===")
    print(f"{'circuit':<20} {'orig(1Q/2Q/W)':<22} {'final W':<10} {'W red %':<10}")
    for _, cname in CIRCUIT_LIST:
        s = orig_stats.get(cname)
        data = results.get(cname)
        if s is None or data is None:
            print(f"{cname:<20} {'N/A':<22} {'N/A':<10} {'N/A':<10}")
            continue
        total, two_q, orig_w = s
        one_q = total - two_q
        times, costs = data
        final_w = int(costs[-1]) if costs else -1
        pct = f"{(1 - final_w / orig_w) * 100:.1f}%" if final_w > 0 else "N/A"
        print(f"{cname:<20} {f'{one_q}/{two_q}/{orig_w}':<22} "
              f"{final_w:<10} {pct:<10}")

    # Also report the output QASM 1Q/2Q breakdown.
    print("\n=== Final QASM gate breakdown (1Q / 2Q / total / weighted) ===")
    for _, cname in CIRCUIT_LIST:
        qasm_path = os.path.join(OUT_DIR, "qasm", f"{cname}.qasm")
        s = count_gates(qasm_path)
        if s is None:
            print(f"  {cname}: missing QASM")
            continue
        total, two_q, weighted = s
        one_q = total - two_q
        print(f"  {cname}: {one_q} / {two_q} / {total} / {weighted}")


# -- main ----------------------------------------------------------------------

def main():
    for d in [OUT_DIR, os.path.join(OUT_DIR, "pkl"),
              os.path.join(OUT_DIR, "qasm")]:
        os.makedirs(d, exist_ok=True)

    orig_stats = get_original_stats()

    tasks = list(CIRCUIT_LIST)
    print(f"Launching {len(tasks)} circuits with {N_WORKERS} workers "
          f"(timeout={TIMEOUT}s, mem_limit=8GB, cost_mode={COST_MODE}, "
          f"npool={N_POOL}, nbranch={N_BRANCH})...")

    with multiprocessing.Pool(N_WORKERS, initializer=_set_mem_limit) as pool:
        raw = pool.map(dispatch, tasks)

    results = {cname: (t, c) for cname, (t, c) in raw}

    print_summary(results, orig_stats)
    make_plots(results, orig_stats)
    print("\nDone!")


if __name__ == "__main__":
    main()
