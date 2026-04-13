"""
Plotting script for greedy_k ablation study.

Reads pickled results from greedy_k_ablation_results/pkl/ and produces figures.
No experiments are run — this is plot-only.

Usage:
    python3 qalm_experiments/plot_greedy_k.py
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

GREEDY_K_VALUES = [0, 1, 2, 3]
TIMEOUT = 3600
RESULTS_DIR = "greedy_k_ablation_results"
REDUCTION_YLIM = (25, 35)  # set to None for auto


_ORIG_CACHE = {}


def get_original_gate_count(circ_path):
    if circ_path in _ORIG_CACHE:
        return _ORIG_CACHE[circ_path]
    import qiskit
    qc = qiskit.QuantumCircuit.from_qasm_file(circ_path)
    _ORIG_CACHE[circ_path] = qc.size()
    return _ORIG_CACHE[circ_path]


def _load_one(circ_name, gk):
    """Load a single pkl; returns (ts, cs, peak_kb, k_events) or None if missing."""
    pkl_path = os.path.join(
        RESULTS_DIR, "pkl", f"{circ_name}_gk{gk}_{TIMEOUT}.pkl",
    )
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    n_circs = len(CIRCUIT_LIST)
    n_gk = len(GREEDY_K_VALUES)

    COLORS = [plt.get_cmap("tab10")(i) for i in range(10)]

    # ── Stream-load pkl files, extracting only the small aggregates needed.
    # For "detail" circuits (tof_3, xor5_254) we keep the raw time series too,
    # since the per-circuit advancement plot needs them.
    common_times = np.linspace(0, TIMEOUT, 300)
    ratio_series = [[] for _ in GREEDY_K_VALUES]
    final_costs = np.full((n_circs, n_gk), np.nan)
    peak_mb_per_gk = [[] for _ in GREEDY_K_VALUES]
    k_events_all = [[[] for _ in range(n_gk)] for _ in range(n_circs)]
    pkl_present = np.zeros((n_circs, n_gk), dtype=bool)

    DETAIL_CIRCUITS = {"tof_3", "xor5_254"}
    detail_raw = {}  # (circ_name, gk_idx) -> (ts, cs, k_events)

    missing = 0
    for ci, (circ_path, circ_name) in enumerate(CIRCUIT_LIST):
        orig = get_original_gate_count(circ_path)
        for gk_idx, gk in enumerate(GREEDY_K_VALUES):
            loaded = _load_one(circ_name, gk)
            if loaded is None:
                missing += 1
                peak_mb_per_gk[gk_idx].append(0)
                continue
            ts, cs, peak_kb, k_events = loaded
            peak_mb_per_gk[gk_idx].append(peak_kb / 1024)
            k_events_all[ci][gk_idx] = k_events
            pkl_present[ci, gk_idx] = True
            if cs:
                final_costs[ci, gk_idx] = cs[-1]
            if len(ts) >= 2:
                ratios = [max(c, 1) / orig for c in cs]
                interp = np.interp(
                    common_times, ts, ratios, left=ratios[0], right=ratios[-1],
                )
                ratio_series[gk_idx].append(interp)
                if circ_name in DETAIL_CIRCUITS:
                    detail_raw[(circ_name, gk_idx)] = (ts, cs, k_events)
            # Explicitly drop large lists to keep memory flat
            del loaded, ts, cs
    if missing:
        print(f"Warning: {missing} pkl files missing")

    # Per-mode ylim overrides (only applied if RESULTS_DIR is the 26-circuit
    # ablation; the 250-circuit GUOQ wrapper sets REDUCTION_YLIM=None).
    arith_ylim = (30, 34) if REDUCTION_YLIM is not None else None
    geo_ylim   = (31, 35) if REDUCTION_YLIM is not None else None

    for mode, ylabel, suffix, ylim in [
        ("arith", "Avg. Gate Count Reduction (%)", "", arith_ylim),
        ("geo",   "Geomean Gate Count Reduction (%)", "_geomean", geo_ylim),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for gk_idx, gk in enumerate(GREEDY_K_VALUES):
            if not ratio_series[gk_idx]:
                continue
            if mode == "arith":
                agg = np.mean(ratio_series[gk_idx], axis=0)
            else:
                agg = np.exp(np.mean(np.log(ratio_series[gk_idx]), axis=0))
            reduction = (1 - agg) * 100
            ax.plot(
                common_times, reduction,
                label=f"greedy_k={gk}",
                color=COLORS[gk_idx], linewidth=1.8,
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(
            f"Effect of greedy_k ({n_circs} circuits)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_pdf = os.path.join(RESULTS_DIR, "figures", f"greedy_k_reduction{suffix}.pdf")
        fig.savefig(out_pdf)
        print(f"Saved → {out_pdf}")
        plt.close(fig)

    # ── per-circuit absolute gate count + k-advancement markers ───────────────
    LINE_STYLES = ["-", "--", "-.", ":"]

    def plot_circuit_advancement(circ_name):
        if circ_name not in DETAIL_CIRCUITS:
            return
        fig_c, ax_c = plt.subplots(figsize=(7, 4))
        for gk_idx, gk in enumerate(GREEDY_K_VALUES):
            entry = detail_raw.get((circ_name, gk_idx))
            if entry is None:
                continue
            ts, cs, k_events = entry
            if len(ts) < 2:
                continue
            ax_c.plot(ts, cs, label=f"greedy_k={gk}",
                      color=COLORS[gk_idx], linewidth=1.8,
                      linestyle=LINE_STYLES[gk_idx % len(LINE_STYLES)])
            # Annotate every k-advancement (skip first); for circuits with many
            # events, only annotate every Nth so the chart stays readable.
            n_events = len(k_events)
            stride = max(1, (n_events - 1) // 10) if n_events > 12 else 1
            for ki, (k_val, k_time) in enumerate(k_events):
                if ki == 0:
                    continue
                cost_at_k = np.interp(k_time, ts, cs)
                ax_c.plot(k_time, cost_at_k, marker="*", markersize=8,
                          color=COLORS[gk_idx], zorder=5)
                if ki == n_events - 1 or ki % stride == 0:
                    ax_c.annotate(f"gk={gk}, k→{k_val}", (k_time, cost_at_k),
                                  textcoords="offset points", xytext=(6, 6),
                                  fontsize=7, color=COLORS[gk_idx])
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("Gate count")
        ax_c.set_title(
            f"{circ_name}: advancing k per greedy_k ({TIMEOUT}s timeout)"
        )
        ax_c.legend(fontsize=9)
        fig_c.tight_layout()
        out = os.path.join(
            RESULTS_DIR, "figures", f"greedy_k_{circ_name}.pdf"
        )
        fig_c.savefig(out)
        print(f"Saved → {out}")
        plt.close(fig_c)

    plot_circuit_advancement("tof_3")
    plot_circuit_advancement("xor5_254")

    # ── Figure 2: peak memory usage per greedy_k ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    data_for_box = [peak_mb_per_gk[gk_idx] for gk_idx in range(n_gk)]
    labels_for_box = [f"gk={gk}" for gk in GREEDY_K_VALUES]
    ax2.boxplot(data_for_box, tick_labels=labels_for_box, notch=False)
    ax2.set_ylabel("Peak RSS (MB)")
    ax2.set_title("Peak memory per greedy_k")
    fig2.tight_layout()
    out_mem = os.path.join(RESULTS_DIR, "figures", "greedy_k_memory.pdf")
    fig2.savefig(out_mem)
    print(f"Saved → {out_mem}")
    plt.close(fig2)

    # ── Figure 3: heatmap (circuit × greedy_k) ───────────────────────────────
    circ_names = [name for _, name in CIRCUIT_LIST]
    final_reduction = np.zeros((n_circs, n_gk))
    for ci, (circ_path, _circ_name) in enumerate(CIRCUIT_LIST):
        orig = get_original_gate_count(circ_path)
        for gk_idx in range(n_gk):
            final = final_costs[ci, gk_idx]
            if np.isnan(final):
                final = orig  # missing data → no reduction
            final_reduction[ci, gk_idx] = (1 - final / orig) * 100

    fig3, ax3 = plt.subplots(figsize=(5, 10))
    im = ax3.imshow(final_reduction, aspect="auto", cmap="YlGn")
    ax3.set_xticks(range(n_gk))
    ax3.set_xticklabels([f"gk={gk}" for gk in GREEDY_K_VALUES])
    ax3.set_yticks(range(n_circs))
    ax3.set_yticklabels(circ_names, fontsize=7)
    for ci in range(n_circs):
        for gk_idx in range(n_gk):
            ax3.text(gk_idx, ci, f"{final_reduction[ci, gk_idx]:.1f}",
                     ha="center", va="center", fontsize=6)
    fig3.colorbar(im, ax=ax3, label="Reduction (%)")
    ax3.set_title("Final gate-count reduction (%)")
    fig3.tight_layout()
    out_heat = os.path.join(RESULTS_DIR, "figures", "greedy_k_heatmap.pdf")
    fig3.savefig(out_heat)
    print(f"Saved → {out_heat}")
    plt.close(fig3)

    # ── Figure 4: per-circuit normalized gate count (gk=0 as baseline) ────────
    # For each circuit, final_cost[gk] / final_cost[gk=0].
    # Sorted by the gk=2 ratio so outliers are easy to spot.
    # (final_costs was populated in the streaming-load pass above)

    # Normalize: gk=0 → 1.0 for each circuit
    gk0_costs = final_costs[:, 0]
    valid = ~np.isnan(gk0_costs) & (gk0_costs > 0)
    norm = np.full_like(final_costs, np.nan)
    for gk_idx in range(n_gk):
        norm[valid, gk_idx] = final_costs[valid, gk_idx] / gk0_costs[valid]

    # Sort by gk=2 ratio (index 2 in GREEDY_K_VALUES)
    gk2_idx = GREEDY_K_VALUES.index(2)
    sort_key = norm[:, gk2_idx].copy()
    sort_key[np.isnan(sort_key)] = 999  # push NaN to the end
    order = np.argsort(sort_key)

    sorted_names = [CIRCUIT_LIST[i][1] for i in order]
    sorted_norm = norm[order]

    fig4, ax4 = plt.subplots(figsize=(max(14, n_circs * 0.12), 5))
    x = np.arange(len(sorted_names))
    for gk_idx, gk in enumerate(GREEDY_K_VALUES):
        if gk == 0:
            continue  # baseline, always 1.0
        vals = sorted_norm[:, gk_idx]
        ax4.plot(x, vals, marker=".", markersize=3, linewidth=0.8,
                 label=f"greedy_k={gk}", color=COLORS[gk_idx], alpha=0.8)
    ax4.axhline(1.0, color="gray", linewidth=0.8, linestyle="--",
                label="greedy_k=0 (baseline)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(sorted_names, rotation=90, fontsize=4, ha="center")
    ax4.set_ylabel("Final gate count / greedy_k=0 gate count")
    ax4.set_title("Per-circuit normalized gate count (sorted by greedy_k=2)")
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    out_norm = os.path.join(RESULTS_DIR, "figures", "greedy_k_normalized.pdf")
    fig4.savefig(out_norm, dpi=200)
    print(f"Saved → {out_norm}")
    plt.close(fig4)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n=== Final gate-count ratio (lower is better) ===")
    print(f"{'greedy_k':<16} {'ArithMean':>12} {'GeoMean':>12} {'Avg peak (MB)':>15}")
    print("-" * 58)
    for gk_idx, gk in enumerate(GREEDY_K_VALUES):
        finals = [arr[-1] for arr in ratio_series[gk_idx]]
        am = np.mean(finals)
        gm = np.exp(np.mean(np.log(finals)))
        avg_peak = (
            np.mean(peak_mb_per_gk[gk_idx]) if peak_mb_per_gk[gk_idx] else float("nan")
        )
        print(f"  greedy_k={gk:<2}   {am:>12.4f} {gm:>12.4f}   {avg_peak:>13.0f} MB")

    # ── k-advancement analysis ────────────────────────────────────────────────
    print("\n=== QALM k-loop advancement ===")
    print(
        f"{'circuit':<22} {'greedy_k':<10} {'k_iters':<10} "
        f"{'max_k':<8} {'k_times'}"
    )
    print("-" * 80)
    for ci, (_, circ_name) in enumerate(CIRCUIT_LIST):
        for gk_idx, gk in enumerate(GREEDY_K_VALUES):
            k_events = k_events_all[ci][gk_idx]
            if not k_events:
                print(f"  {circ_name:<20} gk={gk:<6}  (no k events)")
                continue
            max_k = k_events[-1][0]
            k_times_str = "  ".join(
                f"k={k}@{t:.1f}s" for k, t in k_events
            )
            print(
                f"  {circ_name:<20} gk={gk:<6}  "
                f"iters={len(k_events):<6}  max_k={max_k:<4}  {k_times_str}"
            )

    # ── Average greedy-phase finish time per greedy_k ─────────────────────────
    # For greedy_k=gk, the greedy phase covers levels k=0..gk. It "finishes"
    # at the k_event where k_val == gk+1. Circuits whose search never leaves
    # the greedy range within the timeout are treated as finishing at TIMEOUT.
    # Circuits with no pkl at all are skipped.
    print("\n=== Avg. greedy-phase finish time ===")
    print(f"{'greedy_k':<10} {'avg_finish_s':>14} {'finished/counted':>18}")
    print("-" * 46)
    for gk_idx, gk in enumerate(GREEDY_K_VALUES):
        if gk == 0:
            continue
        finish_times = []
        finished = 0
        for ci in range(n_circs):
            if not pkl_present[ci, gk_idx]:
                continue
            k_events = k_events_all[ci][gk_idx]
            t = None
            for k_val, k_time in k_events:
                if k_val == gk + 1:
                    t = k_time
                    break
            if t is None:
                finish_times.append(float(TIMEOUT))
            else:
                finish_times.append(t)
                finished += 1
        if finish_times:
            avg = float(np.mean(finish_times))
            print(f"  gk={gk:<6} {avg:>14.2f} {finished:>10}/{len(finish_times):<6}")
        else:
            print(f"  gk={gk:<6} {'n/a':>14} {0:>10}/{0:<6}")

    # ── CSV dump ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("circuit,greedy_k,final_cost,original_cost,"
                "reduction_pct,peak_mb,qalm_k_iters,max_qalm_k\n")
        for ci, (circ_path, circ_name) in enumerate(CIRCUIT_LIST):
            orig = get_original_gate_count(circ_path)
            for gk_idx, gk in enumerate(GREEDY_K_VALUES):
                final = final_costs[ci, gk_idx]
                if np.isnan(final):
                    final = orig
                peak_mb = peak_mb_per_gk[gk_idx][ci]
                k_events = k_events_all[ci][gk_idx]
                red = (1 - final / orig) * 100
                k_iters = len(k_events)
                max_qalm_k = k_events[-1][0] if k_events else ""
                f.write(
                    f"{circ_name},{gk},{final:.0f},{orig},"
                    f"{red:.2f},{peak_mb:.0f},"
                    f"{k_iters},{max_qalm_k}\n"
                )
    print(f"\nCSV summary saved → {csv_path}")


if __name__ == "__main__":
    main()
