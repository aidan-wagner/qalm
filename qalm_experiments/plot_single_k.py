"""
Plotting script for single-k ablation study.

Reads pickled results from single_k_results/pkl/ and greedy_k_ablation_results/pkl/.
No experiments are run — this is plot-only.

Usage:
    python3 qalm_experiments/plot_single_k.py
"""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

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
TIMEOUT = 3600
RESULTS_DIR = "single_k_results"
GREEDY_K_RESULTS_DIR = "greedy_k_ablation_results"
REDUCTION_YLIM = (31, 35)  # set to None for auto; only used as a sentinel now


_ORIG_CACHE = {}


def get_original_gate_count(circ_path):
    if circ_path in _ORIG_CACHE:
        return _ORIG_CACHE[circ_path]
    import qiskit
    qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
    _ORIG_CACHE[circ_path] = qc.size()
    return _ORIG_CACHE[circ_path]


def _load_one(circ_name, gk, fk):
    """Load a single pkl; returns (ts, cs, peak_kb) or None if missing."""
    pkl_path = os.path.join(
        RESULTS_DIR, "pkl", f"{circ_name}_gk{gk}_fk{fk}_{TIMEOUT}.pkl",
    )
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    n_circs = len(CIRCUIT_LIST)
    n_fk = len(FIXED_K_VALUES)
    n_gk = len(GREEDY_K_GROUPS)

    COLORS = [plt.get_cmap("tab10")(i) for i in range(10)]
    common_times = np.linspace(0, TIMEOUT, 300)

    # ── Stream-load pkls, extracting small aggregates only.
    # Keep raw (ts, cs) for tof_3 only (needed for the fixed-k vs advancing-k plot).
    # Shapes:
    #   ratio_series[gi][fk_idx] -> list of 300-pt interp arrays
    #   final_costs[gi, ci, fk_idx] -> scalar
    #   peak_mb[gi, ci, fk_idx] -> scalar
    ratio_series = [[[] for _ in FIXED_K_VALUES] for _ in GREEDY_K_GROUPS]
    final_costs = np.full((n_gk, n_circs, n_fk), np.nan)
    peak_mb = np.zeros((n_gk, n_circs, n_fk))
    tof3_raw = {}  # (gi, fk_idx) -> (ts, cs)

    tof3_ci = next(
        (i for i, (_, name) in enumerate(CIRCUIT_LIST) if name == "tof_3"), None
    )

    missing = 0
    for gi, gk in enumerate(GREEDY_K_GROUPS):
        for ci, (circ_path, circ_name) in enumerate(CIRCUIT_LIST):
            orig = get_original_gate_count(circ_path)
            for fk_idx, fk in enumerate(FIXED_K_VALUES):
                loaded = _load_one(circ_name, gk, fk)
                if loaded is None:
                    missing += 1
                    continue
                ts, cs, pk = loaded
                peak_mb[gi, ci, fk_idx] = pk / 1024
                if cs:
                    final_costs[gi, ci, fk_idx] = cs[-1]
                if len(ts) >= 2:
                    ratios = [max(c, 1) / orig for c in cs]
                    interp = np.interp(
                        common_times, ts, ratios,
                        left=ratios[0], right=ratios[-1],
                    )
                    ratio_series[gi][fk_idx].append(interp)
                    if ci == tof3_ci:
                        tof3_raw[(gi, fk_idx)] = (list(ts), list(cs))
                del loaded, ts, cs
    if missing:
        print(f"Warning: {missing} pkl files missing")

    # ── Baseline: advancing-k QALM from the greedy_k ablation, matched to the
    # plot's greedy_k group. For greedy_k=G, QALM advances k starting at G+1.
    baseline_ratios = {gk: [] for gk in GREEDY_K_GROUPS}
    for gk in GREEDY_K_GROUPS:
        for circ_path, circ_name in CIRCUIT_LIST:
            orig = get_original_gate_count(circ_path)
            pkl_path = os.path.join(
                GREEDY_K_RESULTS_DIR, "pkl",
                f"{circ_name}_gk{gk}_{TIMEOUT}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                ts, cs, _peak, _kev = pickle.load(f)
            if len(ts) < 2:
                continue
            ratios = [max(c, 1) / orig for c in cs]
            interp = np.interp(
                common_times, ts, ratios, left=ratios[0], right=ratios[-1],
            )
            baseline_ratios[gk].append(interp)

    # ── one avg reduction plot per greedy_k group ──────────────────────────────
    for gi, gk in enumerate(GREEDY_K_GROUPS):
        arith_ylim = (30, 34) if REDUCTION_YLIM is not None else None
        geo_ylim   = (31, 35) if REDUCTION_YLIM is not None else None

        for mode, ylabel, suffix, ylim in [
            ("arith", "Avg. Gate Count Reduction (%)", "", arith_ylim),
            ("geo",   "Geomean Gate Count Reduction (%)", "_geomean", geo_ylim),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for fk_idx, fk in enumerate(FIXED_K_VALUES):
                if not ratio_series[gi][fk_idx]:
                    continue
                if mode == "arith":
                    agg = np.mean(ratio_series[gi][fk_idx], axis=0)
                else:
                    agg = np.exp(np.mean(np.log(ratio_series[gi][fk_idx]), axis=0))
                reduction = (1 - agg) * 100
                ax.plot(
                    common_times, reduction,
                    label=f"k={fk}",
                    color=COLORS[fk_idx], linewidth=1.8,
                )
            bseries = baseline_ratios.get(gk, [])
            if bseries:
                if mode == "arith":
                    b_agg = np.mean(bseries, axis=0)
                else:
                    b_agg = np.exp(np.mean(np.log(bseries), axis=0))
                b_red = (1 - b_agg) * 100
                k_start = gk + 1
                b_label = f"k={k_start},{k_start+1},{k_start+2},..."
                ax.plot(
                    common_times, b_red,
                    label=b_label,
                    color="black", linewidth=1.8, linestyle="--",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(
                f"Fixed k (greedy_k={gk}, {n_circs} circuits)"
            )
            ax.legend(fontsize=9)
            fig.tight_layout()
            out_pdf = os.path.join(RESULTS_DIR, "figures", f"single_k_gk{gk}{suffix}.pdf")
            fig.savefig(out_pdf)
            print(f"Saved → {out_pdf}")
            plt.close(fig)

    # ── tof_3: fixed-k vs advancing-k (absolute gate count) ──────────────────
    LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for gi, gk in enumerate(GREEDY_K_GROUPS):
        if tof3_ci is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))

        for fk_idx, fk in enumerate(FIXED_K_VALUES):
            entry = tof3_raw.get((gi, fk_idx))
            if entry is None:
                continue
            ts, cs = entry
            if len(ts) < 2:
                continue
            ax.plot(ts, cs, label=f"fixed k={fk}",
                    color=COLORS[fk_idx], linewidth=1.8,
                    linestyle=LINE_STYLES[fk_idx])

        # Overlay advancing-k baseline from greedy_k ablation
        adv_pkl = os.path.join(
            GREEDY_K_RESULTS_DIR, "pkl",
            f"tof_3_gk{gk}_{TIMEOUT}.pkl",
        )
        if os.path.exists(adv_pkl):
            with open(adv_pkl, "rb") as f:
                adv_ts, adv_cs, _, adv_k_events = pickle.load(f)
            if len(adv_ts) >= 2:
                ax.plot(adv_ts, adv_cs, label="advancing k",
                        color="black", linewidth=2.2, linestyle="-")
                for ki, (k_val, k_time) in enumerate(adv_k_events):
                    if ki == 0:
                        continue
                    cost_at_k = np.interp(k_time, adv_ts, adv_cs)
                    ax.plot(k_time, cost_at_k, marker="*", markersize=12,
                            color="black", zorder=5)
                    ax.annotate(f"gk={gk}, k→{k_val}", (k_time, cost_at_k),
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=8, color="black")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gate count")
        ax.set_title(f"tof_3: fixed k vs advancing k (greedy_k={gk})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_pdf = os.path.join(RESULTS_DIR, "figures", f"tof3_gk{gk}.pdf")
        fig.savefig(out_pdf)
        print(f"Saved → {out_pdf}")
        plt.close(fig)

    # ── summary table ─────────────────────────────────────────────────────────
    for gi, gk in enumerate(GREEDY_K_GROUPS):
        print(f"\n=== greedy_k={gk}: final ratio ===")
        print(f"{'k':<6} {'ArithMean':>12} {'GeoMean':>12} {'Avg peak (MB)':>15}")
        print("-" * 48)
        for fk_idx, fk in enumerate(FIXED_K_VALUES):
            final_ratios = []
            peaks = []
            for ci, (circ_path, _) in enumerate(CIRCUIT_LIST):
                final = final_costs[gi, ci, fk_idx]
                if not np.isnan(final):
                    orig = get_original_gate_count(circ_path)
                    final_ratios.append(max(final, 1) / orig)
                    peaks.append(peak_mb[gi, ci, fk_idx])
            if final_ratios:
                am = np.mean(final_ratios)
                gm = np.exp(np.mean(np.log(final_ratios)))
                avg_pk = np.mean(peaks)
                print(f"  k={fk:<3} {am:>12.4f} {gm:>12.4f} {avg_pk:>13.0f} MB")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("circuit,greedy_k,fixed_k,final_cost,original_cost,"
                "reduction_pct,peak_mb\n")
        for gi, gk in enumerate(GREEDY_K_GROUPS):
            for ci, (circ_path, circ_name) in enumerate(CIRCUIT_LIST):
                orig = get_original_gate_count(circ_path)
                for fk_idx, fk in enumerate(FIXED_K_VALUES):
                    final = final_costs[gi, ci, fk_idx]
                    if np.isnan(final):
                        final = orig
                    pk_mb = peak_mb[gi, ci, fk_idx]
                    red = (1 - final / orig) * 100
                    f.write(
                        f"{circ_name},{gk},{fk},{final:.0f},{orig},"
                        f"{red:.2f},{pk_mb:.0f}\n"
                    )
    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
