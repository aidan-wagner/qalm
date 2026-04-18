"""
Post-hoc correlation analysis: circuit features vs. best QALM config.

Uses existing pkl results from k_comparison_results/pkl/ to ask:
  - Which hyperparameter config wins per circuit?
  - Do circuit features (size, depth, T-density, etc.) predict the winner?

Run from repo root:
    PYTHONPATH=qalm_experiments python3 qalm_experiments/circuit_feature_analysis.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qiskit
from scipy import stats
from collections import defaultdict

# ── circuits & configs (must match k_comparison_experiment.py) ────────────────
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
PKL_DIR = "k_comparison_results/pkl"
OUT_DIR  = "k_comparison_results/figures"

# (N_pool, N_branch, k, label, group)
CONFIGS = [
    (1, 1, 1, "k=1",       "k_vary"),
    (1, 1, 2, "k=2",       "k_vary"),
    (1, 1, 3, "k=3",       "k_vary"),
    (1, 1, 4, "k=4",       "k_vary"),
    (1, 1, 5, "k=5",       "k_vary"),
    (2, 1, 3, "pool=2",    "pool_vary"),
    (3, 1, 3, "pool=3",    "pool_vary"),
    (1, 2, 3, "branch=2",  "branch_vary"),
    (1, 3, 3, "branch=3",  "branch_vary"),
]


def load_final_cost(circ_name, n_pool, n_branch, k):
    pkl = f"{PKL_DIR}/{circ_name}_{TIMEOUT}_{n_pool}_{n_branch}_{k}.pkl"
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        times, costs = pickle.load(f)
    return costs[-1] if costs else None


def circuit_features(circ_path):
    """Extract scalar features from a QASM circuit."""
    qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
    gate_counts = qc.count_ops()
    n_gates   = qc.size()
    n_qubits  = qc.num_qubits
    depth     = qc.depth()
    n_t       = gate_counts.get("t", 0) + gate_counts.get("tdg", 0)
    n_cnot    = gate_counts.get("cx", 0) + gate_counts.get("cnot", 0)
    n_rz      = gate_counts.get("rz", 0)
    n_h       = gate_counts.get("h", 0)
    t_density = n_t / n_gates if n_gates > 0 else 0
    cnot_density = n_cnot / n_gates if n_gates > 0 else 0
    width_depth_ratio = n_qubits / depth if depth > 0 else 0
    return {
        "n_gates":           n_gates,
        "n_qubits":          n_qubits,
        "depth":             depth,
        "n_t":               n_t,
        "n_cnot":            n_cnot,
        "n_rz":              n_rz,
        "n_h":               n_h,
        "t_density":         t_density,
        "cnot_density":      cnot_density,
        "width_depth_ratio": width_depth_ratio,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── collect data per circuit ──────────────────────────────────────────────
    circuit_data = []   # list of dicts

    for circ_path, circ_name in CIRCUIT_LIST:
        feats = circuit_features(circ_path)
        orig  = feats["n_gates"]

        final_costs = {}
        for n_pool, n_branch, k, label, group in CONFIGS:
            fc = load_final_cost(circ_name, n_pool, n_branch, k)
            final_costs[(n_pool, n_branch, k)] = fc

        # skip circuits with missing data
        if any(v is None for v in final_costs.values()):
            print(f"  [skip] {circ_name}: missing pkl")
            continue

        # reduction ratio per config (lower = better)
        ratios = {cfg: fc / orig for cfg, fc in final_costs.items()}

        # which config(s) achieve the best ratio? (allow ties within 0.1%)
        best_ratio  = min(ratios.values())
        winner_cfgs = [cfg for cfg, r in ratios.items() if r <= best_ratio * 1.001]

        # "winning dimension": what varies in the best config vs baseline (1,1,3)?
        # Represent as continuous scores for correlation:
        #   best_k     = k of the config with lowest ratio
        #   best_pool  = N_pool of the config with lowest ratio
        #   best_branch = N_branch of the config with lowest ratio
        # (using the single argmin for clarity)
        argmin = min(ratios, key=ratios.get)
        best_k, best_pool, best_branch = argmin[2], argmin[0], argmin[1]

        # margin: how much better is the best over the baseline (1,1,3)?
        baseline_ratio = ratios[(1, 1, 3)]
        margin         = baseline_ratio - best_ratio   # positive = best beats baseline

        # per-config improvement over baseline
        improvements = {cfg: baseline_ratio - r for cfg, r in ratios.items()}

        circuit_data.append({
            "name":          circ_name,
            "orig":          orig,
            "ratios":        ratios,
            "improvements":  improvements,
            "best_ratio":    best_ratio,
            "baseline_ratio": baseline_ratio,
            "margin":        margin,
            "best_k":        best_k,
            "best_pool":     best_pool,
            "best_branch":   best_branch,
            "winner_cfgs":   winner_cfgs,
            **feats,
        })

    n = len(circuit_data)
    print(f"\nLoaded {n} circuits.\n")

    # ── text summary: per-circuit winner ─────────────────────────────────────
    print(f"{'Circuit':<22} {'Orig':>5} {'Best':>6} {'Ratio':>6}  Best config     Margin")
    print("-" * 75)
    for d in circuit_data:
        best_cfg_str = f"pool={d['best_pool']} br={d['best_branch']} k={d['best_k']}"
        print(f"  {d['name']:<20} {d['orig']:>5} {d['best_ratio']*d['orig']:>6.0f} "
              f"{d['best_ratio']:>6.4f}  {best_cfg_str:<16} {d['margin']:+.4f}")

    # ── Pearson correlations: features vs. per-config improvement ────────────
    feature_names = [
        "n_gates", "n_qubits", "depth", "n_t", "n_cnot",
        "t_density", "cnot_density", "width_depth_ratio",
    ]
    feature_labels = [
        "Gate count", "Qubits", "Depth", "T-gates", "CNOT count",
        "T-density", "CNOT density", "Width/depth",
    ]

    # For each config (excluding baseline), compute correlation of improvement with each feature
    non_baseline_cfgs = [(np_, nb, k, lbl, grp) for np_, nb, k, lbl, grp in CONFIGS
                         if not (np_ == 1 and nb == 1 and k == 3)]

    print("\n=== Pearson r: circuit feature vs. improvement over baseline (1,1,3) ===")
    print(f"{'Feature':<22}", end="")
    for _, _, _, lbl, _ in non_baseline_cfgs:
        print(f"  {lbl:>9}", end="")
    print()
    print("-" * (22 + 11 * len(non_baseline_cfgs)))

    corr_matrix = np.zeros((len(feature_names), len(non_baseline_cfgs)))
    pval_matrix = np.zeros_like(corr_matrix)

    for fi, fname in enumerate(feature_names):
        feat_vals = np.array([d[fname] for d in circuit_data])
        print(f"  {feature_labels[fi]:<20}", end="")
        for ci, (np_, nb, k, lbl, _) in enumerate(non_baseline_cfgs):
            impr_vals = np.array([d["improvements"][(np_, nb, k)] for d in circuit_data])
            r, p = stats.pearsonr(feat_vals, impr_vals)
            corr_matrix[fi, ci] = r
            pval_matrix[fi, ci] = p
            marker = "*" if p < 0.05 else " "
            print(f"  {r:>+7.3f}{marker} ", end="")
        print()
    print("  (* p < 0.05)")

    # ── Figure 1: correlation heatmap ─────────────────────────────────────────
    cfg_labels = [lbl for _, _, _, lbl, _ in non_baseline_cfgs]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(corr_matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cfg_labels)))
    ax.set_xticklabels(cfg_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=9)
    # annotate cells
    for fi in range(len(feature_names)):
        for ci in range(len(non_baseline_cfgs)):
            r = corr_matrix[fi, ci]
            p = pval_matrix[fi, ci]
            marker = "*" if p < 0.05 else ""
            ax.text(ci, fi, f"{r:+.2f}{marker}", ha="center", va="center",
                    fontsize=7.5, color="black" if abs(r) < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation: circuit feature vs. improvement of config over baseline\n"
                 "(* p < 0.05; improvement = baseline_ratio − config_ratio, higher = better)")
    fig.tight_layout()
    out1 = f"{OUT_DIR}/feature_correlation_heatmap.pdf"
    fig.savefig(out1)
    print(f"\nSaved → {out1}")
    plt.close(fig)

    # ── Figure 2: scatter — n_gates vs. improvement for each non-baseline config ──
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    for ci, (np_, nb, k, lbl, _) in enumerate(non_baseline_cfgs):
        ax = axes[ci]
        x = np.array([d["n_gates"] for d in circuit_data])
        y = np.array([d["improvements"][(np_, nb, k)] for d in circuit_data])
        ax.scatter(x, y, s=30, alpha=0.7)
        for d, xi, yi in zip(circuit_data, x, y):
            ax.annotate(d["name"], (xi, yi), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        # regression line
        m, b, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="red", linewidth=1,
                label=f"r={r:+.2f}{'*' if p<0.05 else ''}")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("Gate count", fontsize=8)
        ax.set_ylabel("Improvement over baseline", fontsize=7)
        ax.legend(fontsize=7)
    # hide unused subplots
    for ax in axes[len(non_baseline_cfgs):]:
        ax.set_visible(False)
    fig.suptitle("Gate count vs. improvement over baseline (1,1,k=3) per config",
                 fontsize=10)
    fig.tight_layout()
    out2 = f"{OUT_DIR}/gate_count_vs_improvement.pdf"
    fig.savefig(out2)
    print(f"Saved → {out2}")
    plt.close(fig)

    # ── Figure 3: winner distribution (which config wins most often) ─────────
    winner_counts = defaultdict(int)
    for d in circuit_data:
        for cfg in d["winner_cfgs"]:
            winner_counts[cfg] += 1

    fig, ax = plt.subplots(figsize=(7, 4))
    cfgs_sorted = [(np_, nb, k, lbl) for np_, nb, k, lbl, _ in CONFIGS]
    counts = [winner_counts.get((np_, nb, k), 0) for np_, nb, k, _ in cfgs_sorted]
    labels = [lbl for _, _, _, lbl in cfgs_sorted]
    bars = ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("# circuits where this config wins (ties allowed)")
    ax.set_title(f"Winning config frequency across {n} circuits")
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha="center", fontsize=8)
    fig.tight_layout()
    out3 = f"{OUT_DIR}/winner_distribution.pdf"
    fig.savefig(out3)
    print(f"Saved → {out3}")
    plt.close(fig)

    # ── Figure 4: margin vs. circuit size (is large-circuit gain significant?) ─
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([d["n_gates"] for d in circuit_data])
    y = np.array([d["margin"] for d in circuit_data])
    ax.scatter(x, y, s=40, alpha=0.8)
    for d, xi, yi in zip(circuit_data, x, y):
        ax.annotate(d["name"], (xi, yi), fontsize=5.5, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    m, b, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="red", linewidth=1.2,
            label=f"r={r:+.2f}{'*' if p<0.05 else ''}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Original gate count")
    ax.set_ylabel("Best config improvement over baseline")
    ax.set_title("Does hyperparameter tuning help more on larger circuits?")
    ax.legend()
    fig.tight_layout()
    out4 = f"{OUT_DIR}/margin_vs_size.pdf"
    fig.savefig(out4)
    print(f"Saved → {out4}")
    plt.close(fig)


if __name__ == "__main__":
    main()
