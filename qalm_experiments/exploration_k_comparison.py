"""
Experiment: directly compare varying k (exploration depth) vs varying N_pool vs
varying N_branch, to support the paper's claim that "more exploration steps is
more significant than wider beam search."

All (circuit × config) pairs run in parallel so wall time ≈ timeout, not
timeout × n_circuits.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/exploration_k_comparison.py
"""

import subprocess
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing


# ── circuits ──────────────────────────────────────────────────────────────────
CIRCUIT_LIST = [
    ("circuit/nam_circs/adder_8.qasm",         "adder_8"),
    ("circuit/nam_circs/barenco_tof_3.qasm",   "barenco_tof_3"),
    ("circuit/nam_circs/barenco_tof_4.qasm",   "barenco_tof_4"),
    ("circuit/nam_circs/barenco_tof_5.qasm",   "barenco_tof_5"),
    ("circuit/nam_circs/barenco_tof_10.qasm",  "barenco_tof_10"),
    ("circuit/nam_circs/csla_mux_3.qasm",      "csla_mux_3"),
    ("circuit/nam_circs/csum_mux_9.qasm",      "csum_mux_9"),
    ("circuit/nam_circs/gf2^4_mult.qasm",      "gf2^4_mult"),
    ("circuit/nam_circs/gf2^5_mult.qasm",      "gf2^5_mult"),
    ("circuit/nam_circs/gf2^6_mult.qasm",      "gf2^6_mult"),
    ("circuit/nam_circs/gf2^7_mult.qasm",      "gf2^7_mult"),
    ("circuit/nam_circs/gf2^8_mult.qasm",      "gf2^8_mult"),
    ("circuit/nam_circs/gf2^9_mult.qasm",      "gf2^9_mult"),
    ("circuit/nam_circs/gf2^10_mult.qasm",     "gf2^10_mult"),
    ("circuit/nam_circs/hwb6.qasm",            "hwb6"),
    ("circuit/nam_circs/ham15-low.qasm",       "ham15-low"),
    ("circuit/nam_circs/mod5_4.qasm",          "mod5_4"),
    ("circuit/nam_circs/mod_mult_55.qasm",     "mod_mult_55"),
    ("circuit/nam_circs/mod_red_21.qasm",      "mod_red_21"),
    ("circuit/nam_circs/qcla_adder_10.qasm",   "qcla_adder_10"),
    ("circuit/nam_circs/qcla_com_7.qasm",      "qcla_com_7"),
    ("circuit/nam_circs/qcla_mod_7.qasm",      "qcla_mod_7"),
    ("circuit/nam_circs/rc_adder_6.qasm",      "rc_adder_6"),
    ("circuit/nam_circs/tof_3.qasm",           "tof_3"),
    ("circuit/nam_circs/tof_4.qasm",           "tof_4"),
    ("circuit/nam_circs/tof_5.qasm",           "tof_5"),
]

TIMEOUT = 3600          # seconds; match the paper's one-hour evaluation
ECCSET  = "eccset/Nam_5_3_complete_ECC_set.json"

# ── experiment configurations ─────────────────────────────────────────────────
# (N_pool, N_branch, k, label, group)
#   group: "k_vary" | "pool_vary" | "branch_vary"
CONFIGS = [
    # Vary k, fix N_pool=N_branch=1
    (1, 1, 1, "$k=1$",              "k_vary"),
    (1, 1, 2, "$k=2$",              "k_vary"),
    (1, 1, 3, "$k=3$",              "k_vary"),
    (1, 1, 4, "$k=4$",              "k_vary"),
    (1, 1, 5, "$k=5$",              "k_vary"),
    # Vary N_pool, fix N_branch=1, k=3
    (2, 1, 3, "$N_\\mathrm{pool}=2$",  "pool_vary"),
    (3, 1, 3, "$N_\\mathrm{pool}=3$",  "pool_vary"),
    # Vary N_branch, fix N_pool=1, k=3
    (1, 2, 3, "$N_\\mathrm{branch}=2$","branch_vary"),
    (1, 3, 3, "$N_\\mathrm{branch}=3$","branch_vary"),
]

# fixed flags (all configs share these)
# (repeat_tolerance, exploration_increase, no_increase,
#  only_do_local_transformations, greedy_start, two_way_rm)
FIXED_FLAGS = (1.5, 0, 0, 1, 1, 0)


def run_one(args):
    """Run test_qalm for one (circuit, config) pair and return (times, costs)."""
    circ_path, circ_name, n_pool, n_branch, k = args
    rep_tol, exp_incr, no_incr, local, greedy, two_way = FIXED_FLAGS

    pkl_path = (
        f"k_comparison_results/pkl/"
        f"{circ_name}_{TIMEOUT}_{n_pool}_{n_branch}_{k}.pkl"
    )
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    result = subprocess.run(
        [
            "./build/test_qalm",
            circ_path, circ_name,
            str(TIMEOUT),
            str(n_pool), str(n_branch), str(k),
            str(rep_tol),
            str(int(exp_incr)),
            str(int(no_incr)),
            str(int(local)),
            str(int(greedy)),
            str(int(two_way)),
            ECCSET,
        ],
        capture_output=True,
        text=True,
    )

    times, costs = [], []
    for line in result.stdout.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == f"[{circ_name}]":
            costs.append(float(words[3]))
            times.append(float(words[8]))

    out = (times, costs)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)
    return out


def get_original_gate_count(circ_path):
    import qiskit
    qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
    return qc.size()


def main():
    os.makedirs("k_comparison_results/pkl",     exist_ok=True)
    os.makedirs("k_comparison_results/figures", exist_ok=True)

    # Build flattened task list
    tasks = [
        (circ_path, circ_name, n_pool, n_branch, k)
        for (circ_path, circ_name) in CIRCUIT_LIST
        for (n_pool, n_branch, k, _label, _group) in CONFIGS
    ]

    print(f"Launching {len(tasks)} tasks with up to 128 parallel workers "
          f"(timeout {TIMEOUT}s each) …")
    with multiprocessing.Pool(128) as pool:
        flat_results = pool.map(run_one, tasks)

    # Reshape: results[circ_idx][cfg_idx] = (times, costs)
    n_circs  = len(CIRCUIT_LIST)
    n_cfgs   = len(CONFIGS)
    results  = [
        flat_results[ci * n_cfgs : (ci + 1) * n_cfgs]
        for ci in range(n_circs)
    ]

    # ── aggregate geomean-vs-time curve per config ────────────────────────────
    common_times = np.linspace(0, TIMEOUT, 300)
    log_series   = [[] for _ in CONFIGS]   # per config: list of log-ratio arrays

    for ci, (circ_path, circ_name) in enumerate(CIRCUIT_LIST):
        orig = get_original_gate_count(circ_path)
        for cfg_idx in range(n_cfgs):
            ts, cs = results[ci][cfg_idx]
            if len(ts) < 2:
                continue
            log_rs = [np.log(max(c, 1) / orig) for c in cs]
            interp = np.interp(common_times, ts, log_rs,
                               left=log_rs[0], right=log_rs[-1])
            log_series[cfg_idx].append(interp)

    # ── Figure 1: all-in-one comparison plot ─────────────────────────────────
    group_style = {
        "k_vary":      dict(linestyle="-",  linewidth=1.8),
        "pool_vary":   dict(linestyle="--", linewidth=1.4),
        "branch_vary": dict(linestyle=":",  linewidth=1.4),
    }
    DISTINCT_COLORS = [plt.get_cmap("tab10")(i) for i in range(10)]

    from collections import Counter
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg_idx, (n_pool, n_branch, k, label, group) in enumerate(CONFIGS):
        if not log_series[cfg_idx]:
            continue
        mean_log = np.mean(log_series[cfg_idx], axis=0)
        reduction = (1 - np.exp(mean_log)) * 100
        color     = DISTINCT_COLORS[cfg_idx % len(DISTINCT_COLORS)]
        ax.plot(common_times, reduction,
                label=label, color=color, **group_style[group])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg. gate-count reduction (%)")
    ax.set_ylim(bottom=30)
    ax.set_title(f"Exploration strategy comparison ({TIMEOUT}s timeout, "
                 f"{n_circs} circuits)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_pdf = "k_comparison_results/figures/k_vs_pool_vs_branch.pdf"
    fig.savefig(out_pdf)
    print(f"Saved combined figure → {out_pdf}")
    plt.close(fig)

    # ── Figure 2: three-panel figure for the paper ───────────────────────────
    # Panel (a): k variation; (b): N_pool variation; (c): N_branch variation
    # Each panel shows the k=3, N_pool=1, N_branch=1 baseline in grey for reference.
    baseline_idx = next(
        i for i, (np_, nb, k, _, g) in enumerate(CONFIGS)
        if np_ == 1 and nb == 1 and k == 3
    )
    if log_series[baseline_idx]:
        baseline_reduction = (1 - np.exp(np.mean(log_series[baseline_idx], axis=0))) * 100
    else:
        baseline_reduction = None

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    panel_groups = ["k_vary", "pool_vary", "branch_vary"]
    panel_titles = [
        "Varying $k$ ($N_\\mathrm{pool}=N_\\mathrm{branch}=1$)",
        "Varying $N_\\mathrm{pool}$ ($N_\\mathrm{branch}=1,\\ k=3$)",
        "Varying $N_\\mathrm{branch}$ ($N_\\mathrm{pool}=1,\\ k=3$)",
    ]
    for ax2, group, title in zip(axes, panel_groups, panel_titles):
        # grey baseline (k=3, N_pool=1, N_branch=1) for reference
        baseline_label = {"pool_vary": "$N_\\mathrm{pool}=1$",
                          "branch_vary": "$N_\\mathrm{branch}=1$"}.get(group)
        if baseline_reduction is not None and baseline_label is not None:
            ax2.plot(common_times, baseline_reduction,
                     color="grey", linestyle="-", linewidth=1.2,
                     label=baseline_label, zorder=0)
        cfg_list = [(i, cfg) for i, cfg in enumerate(CONFIGS) if cfg[4] == group]
        for j, (cfg_idx, (n_pool, n_branch, k, label, _)) in enumerate(cfg_list):
            if not log_series[cfg_idx]:
                continue
            reduction = (1 - np.exp(np.mean(log_series[cfg_idx], axis=0))) * 100
            color     = DISTINCT_COLORS[j % len(DISTINCT_COLORS)]
            ax2.plot(common_times, reduction, label=label,
                     color=color, **group_style[group])
        ax2.set_ylim(bottom=30)
        ax2.set_title(title, fontsize=9)
        ax2.set_xlabel("Time (s)")
        if ax2 is axes[0]:
            ax2.set_ylabel("Avg. gate-count reduction (%)")
        ax2.legend(fontsize=8)
    fig2.suptitle(
        f"Impact of exploration parameters ({TIMEOUT}s timeout, {n_circs} circuits)",
        fontsize=10,
    )
    fig2.tight_layout()
    out_pdf2 = "k_comparison_results/figures/exploration_three_panel.pdf"
    fig2.savefig(out_pdf2)
    print(f"Saved three-panel figure → {out_pdf2}")
    plt.close(fig2)

    # ── text summary ──────────────────────────────────────────────────────────
    print("\n=== Final geomean gate-count ratio (lower is better) ===")
    print(f"{'Config':<30} {'GeoMean':>10}")
    print("-" * 42)
    for cfg_idx, (n_pool, n_branch, k, label, group) in enumerate(CONFIGS):
        if not log_series[cfg_idx]:
            print(f"  N_pool={n_pool} N_branch={n_branch} k={k:<2}  (no data)")
            continue
        gm = np.exp(np.mean([arr[-1] for arr in log_series[cfg_idx]]))
        print(f"  N_pool={n_pool} N_branch={n_branch} k={k:<2}  {gm:>10.4f}   ({label})")


if __name__ == "__main__":
    main()
