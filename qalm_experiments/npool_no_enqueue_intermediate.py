"""
N_pool comparison experiment WITHOUT intermediate enqueue.

Same as the advancing-k npool experiment but passes enqueue_intermediate=0,
so only the final circuit (after ROQC) is added to the priority queue.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/npool_no_enqueue_intermediate.py
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

OUT_DIR = "npool_no_enqueue_intermediate_results"

PARAM_VALUES = [1, 2, 3, 4, 5]

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
    ("circuit/nam_circs/hwb6.qasm", "hwb6"),
    ("circuit/nam_circs/ham15-low.qasm", "ham15-low"),
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

PLOT_EXCLUDE = {"hwb6", "ham15-low"}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# -- helpers -------------------------------------------------------------------

def _set_mem_limit():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))


def _parse_output(stdout, circuit_name):
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


def run_qalm(filename, circuit_name, timeout, n_pool, n_branch):
    result = subprocess.run(
        [
            "./build/test_greedy_k_ablation",
            filename,
            circuit_name,
            str(timeout),
            "2",              # greedy_k
            str(n_pool),
            str(n_branch),
            "1.5",            # repeat_tolerance
            "0",              # exploration_increase
            "0",              # strictly_reducing_rules
            "1",              # only_do_local_transformations
            "0",              # two_way_rotation_merging
            ECCSET,
            "0",              # fixed_k=0 (advancing)
            "0",              # enqueue_intermediate=0 (NO intermediate enqueue)
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 600,
    )
    return _parse_output(result.stdout, circuit_name)


# -- dispatch ------------------------------------------------------------------

def dispatch(args):
    n_pool, filename, circuit_name = args
    pkl_dir = os.path.join(OUT_DIR, "pkl")
    pkl_path = os.path.join(pkl_dir, f"{circuit_name}_npool_{n_pool}_{TIMEOUT}.pkl")
    qasm_dir = os.path.join(OUT_DIR, "qasm")
    qasm_path = os.path.join(qasm_dir, f"{circuit_name}_npool_{n_pool}.qasm")

    if os.path.exists(pkl_path):
        print(f"  [npool] {circuit_name} npool={n_pool}: cached", flush=True)
        with open(pkl_path, "rb") as f:
            return n_pool, circuit_name, pickle.load(f)

    try:
        times, costs, qasm_str = run_qalm(
            filename, circuit_name, TIMEOUT, n_pool, 1
        )
    except Exception:
        traceback.print_exc()
        return n_pool, circuit_name, ([], [])

    with open(pkl_path, "wb") as f:
        pickle.dump((times, costs), f)
    if qasm_str.strip():
        with open(qasm_path, "w") as f:
            f.write(qasm_str)

    final = int(costs[-1]) if costs else -1
    print(f"  [npool] {circuit_name} npool={n_pool}: {final} gates", flush=True)
    return n_pool, circuit_name, (times, costs)


# -- plotting ------------------------------------------------------------------

def get_original_counts():
    counts = {}
    for filename, name in CIRCUIT_LIST:
        qc = qiskit.QuantumCircuit.from_qasm_file(filename)
        counts[name] = qc.size()
    return counts


def make_aggregate_plots(results, orig_counts):
    plot_circuits = [name for _, name in CIRCUIT_LIST if name not in PLOT_EXCLUDE]
    common_times = np.linspace(0, TIMEOUT, 600)

    # Arithmetic mean reduction
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for pval, color in zip(PARAM_VALUES, COLORS):
        reductions = []
        for cname in plot_circuits:
            orig = orig_counts[cname]
            data = results.get(pval, {}).get(cname)
            if data is None:
                continue
            times, costs = data
            if not times:
                continue
            pt = ([0.0] + list(times)) if times[0] > 0 else list(times)
            pc = ([float(orig)] + list(costs)) if times[0] > 0 else list(costs)
            red = [(orig - c) / orig * 100 for c in pc]
            interp = np.interp(common_times, pt, red, left=red[0], right=red[-1])
            reductions.append(interp)
        if reductions:
            mean_red = np.mean(reductions, axis=0)
            ax1.plot(common_times, mean_red, label=f"N_pool={pval}",
                     color=color, linewidth=1.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Avg. Gate Count Reduction (%)")
    ax1.set_title(f"N_pool comparison, no intermediate enqueue ({len(plot_circuits)} circuits)")
    ax1.legend(fontsize=9)
    ax1.set_ylim(30, 34)
    fig1.tight_layout()
    path1 = os.path.join(OUT_DIR, f"N_pool_arith_mean_{TIMEOUT}s.pdf")
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"Saved -> {path1}")

    # Geometric mean reduction
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for pval, color in zip(PARAM_VALUES, COLORS):
        log_ratios = []
        for cname in plot_circuits:
            orig = orig_counts[cname]
            data = results.get(pval, {}).get(cname)
            if data is None:
                continue
            times, costs = data
            if not times:
                continue
            pt = ([0.0] + list(times)) if times[0] > 0 else list(times)
            pc = ([float(orig)] + list(costs)) if times[0] > 0 else list(costs)
            lr = [np.log(max(c, 1) / orig) for c in pc]
            interp = np.interp(common_times, pt, lr, left=lr[0], right=lr[-1])
            log_ratios.append(interp)
        if log_ratios:
            mean_log = np.mean(log_ratios, axis=0)
            geomean_red = (1.0 - np.exp(mean_log)) * 100.0
            ax2.plot(common_times, geomean_red, label=f"N_pool={pval}",
                     color=color, linewidth=1.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Avg. Gate Count Reduction (%)")
    ax2.set_title(f"N_pool comparison, no intermediate enqueue ({len(plot_circuits)} circuits)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(30, 34)
    fig2.tight_layout()
    path2 = os.path.join(OUT_DIR, f"N_pool_geomean_{TIMEOUT}s.pdf")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Saved -> {path2}")


def print_summary(results, orig_counts):
    print(f"\n=== N_pool final gate counts (no intermediate enqueue) ===")
    for pval in PARAM_VALUES:
        finals = []
        for _, cname in CIRCUIT_LIST:
            data = results.get(pval, {}).get(cname)
            if data is None:
                finals.append(f"{cname}=N/A")
                continue
            times, costs = data
            final = int(costs[-1]) if costs else -1
            orig = orig_counts[cname]
            pct = (
                f"{(1 - final / orig) * 100:.1f}%"
                if final > 0 and orig > 0
                else "N/A"
            )
            finals.append(f"{cname}={final}({pct})")
        print(f"  N_pool={pval}: {', '.join(finals)}")


# -- main ----------------------------------------------------------------------

def main():
    for d in [OUT_DIR, os.path.join(OUT_DIR, "pkl"), os.path.join(OUT_DIR, "qasm")]:
        os.makedirs(d, exist_ok=True)

    orig_counts = get_original_counts()

    tasks = [
        (n_pool, filename, circuit_name)
        for n_pool in PARAM_VALUES
        for filename, circuit_name in CIRCUIT_LIST
    ]

    print(f"Launching {len(tasks)} tasks with {N_WORKERS} workers "
          f"(timeout={TIMEOUT}s, mem_limit=8GB, enqueue_intermediate=OFF)...")
    with multiprocessing.Pool(N_WORKERS, initializer=_set_mem_limit) as pool:
        raw = pool.map(dispatch, tasks)

    results = {}
    for n_pool, cname, (times, costs) in raw:
        results.setdefault(n_pool, {})[cname] = (times, costs)

    print_summary(results, orig_counts)
    make_aggregate_plots(results, orig_counts)
    print("\nDone!")


if __name__ == "__main__":
    main()
