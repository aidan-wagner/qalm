"""
Experiment: explore higher N_pool / N_branch values and compare ECC 5_3 vs 6_3.

Fixed: k=3, all other FIXED_FLAGS unchanged from k_comparison experiment.
Vary: N_pool (1–5, fix N_branch=1), N_branch (1–5, fix N_pool=1), × {5_3, 6_3}.

ECC tag is encoded in pkl filenames so results from different ECC sets never
collide.

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/higher_params_experiment.py
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

TIMEOUT = 3600
K_FIXED = 3

ECCSETS = {
    "5_3": "eccset/Nam_5_3_complete_ECC_set.json",
    "6_3": "eccset/Nam_6_3_complete_ECC_set.json",
}

# ── experiment configurations ─────────────────────────────────────────────────
# (N_pool, N_branch, ecc_tag, label, group)
#   group: "pool_5_3" | "pool_6_3" | "branch_5_3" | "branch_6_3"
CONFIGS = []

for ecc_tag in ("5_3", "6_3"):
    # Vary N_pool (1–5), fix N_branch=1
    for n_pool in range(1, 6):
        label = f"$N_{{\\mathrm{{pool}}}}={n_pool}$ (ECC {ecc_tag})"
        CONFIGS.append((n_pool, 1, ecc_tag, label, f"pool_{ecc_tag}"))
    # Vary N_branch (1–5), fix N_pool=1
    for n_branch in range(1, 6):
        label = f"$N_{{\\mathrm{{branch}}}}={n_branch}$ (ECC {ecc_tag})"
        CONFIGS.append((1, n_branch, ecc_tag, label, f"branch_{ecc_tag}"))

# fixed flags (same as k_comparison experiment)
# (repeat_tolerance, exploration_increase, no_increase,
#  only_do_local_transformations, greedy_start, two_way_rm)
FIXED_FLAGS = (1.5, 0, 0, 1, 1, 0)

OUT_DIR = "higher_params_results"


def run_one(args):
    """Run test_qalm for one (circuit, config) pair and return (times, costs)."""
    circ_path, circ_name, n_pool, n_branch, ecc_tag = args
    rep_tol, exp_incr, no_incr, local, greedy, two_way = FIXED_FLAGS
    eccset_path = ECCSETS[ecc_tag]

    pkl_path = (
        f"{OUT_DIR}/pkl/"
        f"{circ_name}_{TIMEOUT}_{n_pool}_{n_branch}_{K_FIXED}_{ecc_tag}.pkl"
    )
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    result = subprocess.run(
        [
            "./build/test_qalm",
            circ_path, circ_name,
            str(TIMEOUT),
            str(n_pool), str(n_branch), str(K_FIXED),
            str(rep_tol),
            str(int(exp_incr)),
            str(int(no_incr)),
            str(int(local)),
            str(int(greedy)),
            str(int(two_way)),
            eccset_path,
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
            after_idx = words.index("after")
            times.append(float(words[after_idx + 1]))

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
    os.makedirs(f"{OUT_DIR}/pkl",     exist_ok=True)
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

    tasks = [
        (circ_path, circ_name, n_pool, n_branch, ecc_tag)
        for (circ_path, circ_name) in CIRCUIT_LIST
        for (n_pool, n_branch, ecc_tag, _label, _group) in CONFIGS
    ]

    print(f"Launching {len(tasks)} tasks with up to 128 parallel workers "
          f"(timeout {TIMEOUT}s each) …")
    with multiprocessing.Pool(128) as pool:
        flat_results = pool.map(run_one, tasks)

    n_circs = len(CIRCUIT_LIST)
    n_cfgs  = len(CONFIGS)
    results = [
        flat_results[ci * n_cfgs : (ci + 1) * n_cfgs]
        for ci in range(n_circs)
    ]

    common_times  = np.linspace(0, TIMEOUT, 300)
    ratio_series  = [[] for _ in CONFIGS]

    for ci, (circ_path, _circ_name) in enumerate(CIRCUIT_LIST):
        orig = get_original_gate_count(circ_path)
        for cfg_idx in range(n_cfgs):
            ts, cs = results[ci][cfg_idx]
            if len(ts) < 2:
                continue
            ratios = [max(c, 1) / orig for c in cs]
            interp = np.interp(common_times, ts, ratios,
                               left=ratios[0], right=ratios[-1])
            ratio_series[cfg_idx].append(interp)

    def _agg(series, mode):
        if mode == "arith":
            return np.mean(series, axis=0)
        return np.exp(np.mean(np.log(series), axis=0))

    for mode, ylabel, suffix in [
        ("arith", "Avg. gate-count ratio (vs. original)", ""),
        ("geo",   "Geomean gate-count ratio (vs. original)", "_geomean"),
    ]:
        # Figure 1: pool comparison (5_3 vs 6_3)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, ecc_tag, title in zip(
            axes,
            ("5_3", "6_3"),
            ("Varying $N_{\\mathrm{pool}}$ — ECC 5\\_3", "Varying $N_{\\mathrm{pool}}$ — ECC 6\\_3"),
        ):
            cmap = plt.get_cmap("Blues")
            cfg_list = [
                (i, cfg) for i, cfg in enumerate(CONFIGS)
                if cfg[4] == f"pool_{ecc_tag}"
            ]
            for j, (cfg_idx, (n_pool, n_branch, _ecc, label, _g)) in enumerate(cfg_list):
                if not ratio_series[cfg_idx]:
                    continue
                agg_ratio = _agg(ratio_series[cfg_idx], mode)
                color   = cmap(0.3 + 0.6 * (j + 1) / (len(cfg_list) + 1))
                ax.plot(common_times, agg_ratio, label=f"$N_{{\\mathrm{{pool}}}}={n_pool}$",
                        color=color)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"N_pool sweep ({TIMEOUT}s, {n_circs} circuits)", fontsize=10)
        fig.tight_layout()
        out_pool = f"{OUT_DIR}/figures/pool_sweep{suffix}.pdf"
        fig.savefig(out_pool)
        print(f"Saved → {out_pool}")
        plt.close(fig)

        # Figure 2: branch comparison (5_3 vs 6_3)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, ecc_tag, title in zip(
            axes,
            ("5_3", "6_3"),
            ("Varying $N_{\\mathrm{branch}}$ — ECC 5\\_3", "Varying $N_{\\mathrm{branch}}$ — ECC 6\\_3"),
        ):
            cmap = plt.get_cmap("Oranges")
            cfg_list = [
                (i, cfg) for i, cfg in enumerate(CONFIGS)
                if cfg[4] == f"branch_{ecc_tag}"
            ]
            for j, (cfg_idx, (n_pool, n_branch, _ecc, label, _g)) in enumerate(cfg_list):
                if not ratio_series[cfg_idx]:
                    continue
                agg_ratio = _agg(ratio_series[cfg_idx], mode)
                color   = cmap(0.3 + 0.6 * (j + 1) / (len(cfg_list) + 1))
                ax.plot(common_times, agg_ratio, label=f"$N_{{\\mathrm{{branch}}}}={n_branch}$",
                        color=color)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"N_branch sweep ({TIMEOUT}s, {n_circs} circuits)", fontsize=10)
        fig.tight_layout()
        out_branch = f"{OUT_DIR}/figures/branch_sweep{suffix}.pdf"
        fig.savefig(out_branch)
        print(f"Saved → {out_branch}")
        plt.close(fig)

        # Figure 3: 5_3 vs 6_3 head-to-head
        fig, ax = plt.subplots(figsize=(7, 4))
        for ecc_tag, color in (("5_3", "steelblue"), ("6_3", "darkorange")):
            cfg_idx = next(
                i for i, (np_, nb, et, _, _) in enumerate(CONFIGS)
                if np_ == 1 and nb == 1 and et == ecc_tag
            )
            if ratio_series[cfg_idx]:
                agg_ratio = _agg(ratio_series[cfg_idx], mode)
                ax.plot(common_times, agg_ratio,
                        label=f"ECC {ecc_tag} ($N_{{\\mathrm{{pool}}}}=N_{{\\mathrm{{branch}}}}=1$)",
                        color=color, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ECC 5\\_3 vs 6\\_3 baseline ({TIMEOUT}s, {n_circs} circuits)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_ecc = f"{OUT_DIR}/figures/ecc_comparison{suffix}.pdf"
        fig.savefig(out_ecc)
        print(f"Saved → {out_ecc}")
        plt.close(fig)

    # ── text summary ──────────────────────────────────────────────────────────
    print(f"\n=== Final gate-count ratio (lower is better, k={K_FIXED} fixed) ===")
    print(f"{'Config':<50} {'ArithMean':>10} {'GeoMean':>10}")
    print("-" * 72)
    for cfg_idx, (n_pool, n_branch, ecc_tag, label, group) in enumerate(CONFIGS):
        if not ratio_series[cfg_idx]:
            print(f"  N_pool={n_pool} N_branch={n_branch} ECC={ecc_tag}  (no data)")
            continue
        finals = [arr[-1] for arr in ratio_series[cfg_idx]]
        am = np.mean(finals)
        gm = np.exp(np.mean(np.log(finals)))
        print(f"  N_pool={n_pool} N_branch={n_branch} ECC={ecc_tag}  {am:>10.4f} {gm:>10.4f}")


if __name__ == "__main__":
    main()
