"""Per-circuit figures of the 248-circuit comparison (paper Sec. 5).

Inputs: result rows in the ``metrics.load_runs`` schema, from
``<results>/runs.csv`` written by ``repro/harness/run.py collect``.  Only suite
S248, ``budget_s=3600``, ``seed=0`` rows are used:

* QALM: config ``a`` (optimizer ``qalm``, greedy_k=2, n_pool=1, n_branch=3,
  fixed_k=0, enqueue=0) or config ``b`` (optimizer ``qalm_old``).  The
  default is ``a``.
* Baselines: GUOQ and QUESO with ``params.opt`` TWO_Q / TOTAL, VOQC, Quartz.

The circuit set is the suite manifest ``repro/inputs/S248.csv``; circuits
without a row are logged and listed as missing (``n`` is then below 248).

Outputs (in ``--out``):

* ``comparison_3x3.pdf``: 3 metric rows (fidelity, 2-qubit and total gate
  reduction) x 6 baseline columns.  In each panel the circuits are sorted by
  the QALM value of that metric (ties by circuit name); x is the rank.
* ``win_rate.png``: per baseline, the share of circuits where QALM >= baseline
  for each metric (``metrics.win_rate``, paired by circuit); the solid part is
  a strict win, the hatched part a tie.
* ``gate_scatter.pdf``: (2-qubit, total) gate counts of the inputs and of the
  QALM and GUOQ (2Q Gate) outputs on log-log axes.  Circuits with a zero
  count cannot be drawn there; they are dropped, logged and counted.
* ``fig_percircuit.json``: the plotted numbers (``build()``'s return value):
  per-column means of both series, win/tie percentages with their n, non-OK
  status counts, missing circuits and dropped scatter points.

All metric definitions come from ``metrics.py``.  Failed runs already count
as the original circuit (collect.py does that).  A baseline without any rows
is drawn as an empty panel or bar group labelled "no data".

The paper's figure scripts are lost, so their look (axes sizes and gaps,
fonts, colours, marker sizes, label offsets) was measured from the paper's
figures and is reproduced explicitly: rendering the paper's own plotted data
with these functions reproduces its PDFs to within 0.02 pt and its PNG pixel
for pixel except for 1-2 px text snapping.  Output is byte-reproducible.

Usage (from the repo root)::

    uv run repro/analysis/fig_percircuit.py --results repro/results \\
        --out repro/analysis/out [--qalm-config a|b]
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter

os.environ["SOURCE_DATE_EPOCH"] = "0"  # reproducible PDF/PNG; before matplotlib

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

log = logging.getLogger("fig_percircuit")

SUITE = "S248"
MANIFEST = os.path.join(os.path.dirname(HERE), "inputs", f"{SUITE}.csv")
BUDGET_S = 3600
SEED = 0
QALM_CONFIGS = {
    "a": (
        "qalm",
        dict(
            greedy_k=2,
            n_pool=1,
            n_branch=3,
            fixed_k=0,
            enqueue=0,
        ),
    ),
    "b": ("qalm_old", {}),
}
# (key, column title, optimizer, params filter)
BASELINES = [
    ("guoq_2q", "GUOQ (2Q Gate)", "guoq", {"opt": "TWO_Q"}),
    ("guoq_total", "GUOQ (Total Gate)", "guoq", {"opt": "TOTAL"}),
    ("queso_2q", "QUESO (2Q Gate)", "queso", {"opt": "TWO_Q"}),
    ("queso_total", "QUESO (Total Gate)", "queso", {"opt": "TOTAL"}),
    ("voqc", "VOQC", "voqc", {}),
    ("quartz", "Quartz", "quartz", {}),
]
# metric names as in metrics.compare / win_rate, and their comparison_3x3 row
METRICS = ["fidelity", "cx", "total"]
ROW_LABELS = {
    "fidelity": "Fidelity",
    "cx": "2-Qubit Gate Reduction (%)",
    "total": "Total Gate Reduction (%)",
}
QALM_COLOR = "#DD8452"
BASE_COLOR = "#4C72B0"
BAR_COLORS = {"fidelity": "#4C72B0", "cx": "#DD8452", "total": "#55A868"}
BAR_LABELS = {
    "fidelity": "Fidelity",
    "cx": "2Q Gate Reduction",
    "total": "Total Gate Reduction",
}
NO_DATA = "no data"
PT = 1 / 72  # inches per point

# comparison_3x3 geometry (points), measured from the paper's PDF.
C3_FIGSIZE = (36, 18)
C3_AX = 357.8983333  # every panel is this square
C3_GAP_X = 63.7
C3_GAP_Y = 42.1982836
C3_LEFT = 115.31  # puts the figure-centred legend where the paper has it
C3_BOTTOM = 60.0
C3_LEGEND_Y = 14.703  # legend anchor above the top row
C3_TITLE_PAD = 12
# A panel whose values go below this (percent) gets its own y-axis, like the
# paper's GUOQ (2Q Gate) total-reduction panel; the rest of the row shares one.
OWN_AXIS_BELOW = -10.0

# gate_scatter geometry (points), measured from the paper's PDF.
GS_FIGSIZE = (10, 5.5)
GS_AXES = [(71.496875, 57.496875), (414.4627286585, 57.496875)]
GS_AX = 304.92
GS_TITLE_PAD = 10

# win_rate: figsize + tight_layout reproduce the paper's PNG geometry exactly.
WR_FIGSIZE = (10, 5)
WR_DPI = 300
WR_BAR_W = 0.22
WR_YLIM = (0, 118)
WR_HATCH_LW = 2.5
WR_EDGE_LW = 0.5  # white separators of the solid bars
WR_TIE_EDGE_LW = 0.8
WR_LABEL_DY = 1.17  # label anchor above the bar top (data units), measured

PDF_META = {"CreationDate": None, "ModDate": None}
PNG_META = {"Software": None}


# ---------- data selection ----------


def by_circuit(rows, optimizer, **params):
    """{circuit: row} of the S248 / 1 h / seed 0 rows of one configuration."""
    sel = metrics.select(
        rows, optimizer, suite=SUITE, budget_s=BUDGET_S, seed=SEED, **params
    )
    out = {}
    for r in sel:
        if r["circuit"] in out:
            raise ValueError(
                f"two {optimizer} {params} rows for {r['circuit']}: "
                f"{out[r['circuit']]['job_id']}, {r['job_id']}"
            )
        out[r["circuit"]] = r
    return out


def value(row, metric):
    """Per-circuit value plotted for a metric (None if undefined)."""
    if metric == "fidelity":
        return metrics.fidelity(row["opt_total"], row["opt_cx"])
    if metric == "cx":
        if row["orig_cx"] == 0:  # metrics.averages skips these too
            return None
        return metrics.reduction_pct(row["orig_cx"], row["opt_cx"])
    return metrics.reduction_pct(row["orig_total"], row["opt_total"])


def status_counts(rows):
    """{status: count} of the non-OK rows."""
    counts = Counter(r["status"] for r in rows if r["status"] != "OK")
    return dict(sorted(counts.items()))


def original_counts(rows):
    """{circuit: (orig_cx, orig_total)} of the suite manifest and all S248 rows.

    The manifest (repro/inputs/S248.csv) defines the circuit set, so a circuit
    without any row (unfinished campaign) is reported missing, not dropped.
    Rows must agree with it and with each other.
    """
    out = {}
    with open(MANIFEST) as f:
        for m in csv.DictReader(f):
            out[m["name"]] = (int(m["cx"]), int(m["total"]))
    for r in rows:
        if r["suite"] != SUITE:
            continue
        k = (r["orig_cx"], r["orig_total"])
        if out.setdefault(r["circuit"], k) != k:
            raise ValueError(f"inconsistent original counts for {r['circuit']}")
    return out


def panel_series(qalm, base, metric):
    """Series of one comparison_3x3 panel, sorted by the QALM value.

    Returns {"qalm": [y], "base_x": [rank], "base": [y], "no_data": bool};
    the baseline series only has the circuits it has a row for.
    """
    order = sorted(
        (v, c) for c, r in qalm.items() if (v := value(r, metric)) is not None
    )
    base_x, base_y = [], []
    for i, (_, c) in enumerate(order):
        if base and c in base:
            base_x.append(i)
            base_y.append(value(base[c], metric))
    return {
        "qalm": [v for v, _ in order],
        "base_x": base_x,
        "base": base_y,
        "no_data": not base,
    }


def gather(rows, qalm_config):
    """Select QALM and baseline rows; return (numbers, panels, qalm rows)."""
    opt, params = QALM_CONFIGS[qalm_config]
    qalm = by_circuit(rows, opt, **params)
    if not qalm:
        raise SystemExit(
            f"no QALM config {qalm_config} rows ({opt} {params}, suite {SUITE}, "
            f"budget {BUDGET_S} s, seed {SEED})"
        )
    universe = set(original_counts(rows))
    missing_q = sorted(universe - set(qalm))
    if missing_q:
        log.warning("QALM has no row for %d circuits: %s", len(missing_q), missing_q)
    res = {
        "suite": SUITE,
        "budget_s": BUDGET_S,
        "qalm_config": qalm_config,
        "qalm": {
            "optimizer": opt,
            "params": params,
            "missing": missing_q,
            "non_ok": status_counts(qalm.values()),
            "means": metrics.averages(list(qalm.values())),
        },
        "baselines": {},
    }
    panels = {m: {} for m in METRICS}
    for key, title, b_opt, b_params in BASELINES:
        base = by_circuit(rows, b_opt, **b_params)
        paired = [base[c] for c in sorted(qalm) if c in base]
        entry = {"title": title, "optimizer": b_opt, "params": b_params}
        res["baselines"][key] = entry
        if not paired:
            log.warning("%s: no rows, drawn as '%s'", title, NO_DATA)
            base = None
            entry.update(no_data=True, n=0)
        else:
            missing = sorted(universe - set(base))
            if missing:
                log.warning(
                    "%s: no row for %d circuits: %s", title, len(missing), missing
                )
            entry.update(
                no_data=False,
                n=len(paired),
                missing=missing,
                non_ok=status_counts(paired),
                means=metrics.averages(paired),
                qalm_means=metrics.averages([qalm[r["circuit"]] for r in paired]),
                win_rate={
                    m: metrics.win_rate(list(qalm.values()), paired, m) for m in METRICS
                },
            )
        for m in METRICS:
            panels[m][key] = panel_series(qalm, base, m)
    return res, panels, qalm


# ---------- comparison_3x3 ----------


def _auto_lim(lo, hi):
    """matplotlib's default 5% autoscale margins."""
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return (lo - pad, hi + pad)


def row_limits(metric, row):
    """{key: (ymin, ymax)} for the panels of one row (paper conventions).

    Fidelity uses (0, 1.05).  Reduction rows share one autoscaled range,
    except panels going below OWN_AXIS_BELOW, which get their own; in the
    total row the shared range starts at 0 when no shared value is negative.
    """
    if metric == "fidelity":
        return {k: (0.0, 1.05) for k in row}
    span = {}
    for k, p in row.items():
        ys = p["qalm"] + p["base"]
        if not p["no_data"] and ys:
            span[k] = (min(ys), max(ys))
    own = {k for k, (lo, _) in span.items() if lo < OWN_AXIS_BELOW}
    shared = [span[k] for k in span if k not in own]
    if not shared:  # only QALM has data: scale to it
        ys = [y for p in row.values() for y in p["qalm"]]
        shared = [(min(ys), max(ys))]
    lo, hi = min(s[0] for s in shared), max(s[1] for s in shared)
    lim = _auto_lim(lo, hi)
    if metric == "total" and lo >= 0:
        lim = (0.0, lim[1])
    return {k: (_auto_lim(*span[k]) if k in own else lim) for k in row}


def _check_inside(name, ys, lim):
    bad = [y for y in ys if not lim[0] <= y <= lim[1]]
    if bad:
        raise ValueError(f"{name}: {len(bad)} values outside y-limits {lim}: {bad[:5]}")


def plot_comparison(panels, path):
    """panels[metric][key] = panel_series(...)."""
    gap_x = C3_GAP_X
    while True:
        fig, overlap = _draw_comparison(panels, gap_x)
        if overlap <= 0:
            break
        plt.close(fig)
        gap_x += overlap + 6
        log.warning(
            "comparison_3x3: widening column gaps to %.1f pt for y labels", gap_x
        )
    fig.savefig(path, bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)


def _draw_comparison(panels, gap_x):
    """Draw the 3x6 grid; return (figure, worst y-label overlap in points)."""
    fig = plt.figure(figsize=C3_FIGSIZE)
    fig_w, fig_h = (v / PT for v in C3_FIGSIZE)
    grid = []
    handles = {}
    for i, metric in enumerate(METRICS):
        lims = row_limits(metric, panels[metric])
        y0 = C3_BOTTOM + (len(METRICS) - 1 - i) * (C3_AX + C3_GAP_Y)
        row = []
        for j, (key, title, _, _) in enumerate(BASELINES):
            x0 = C3_LEFT + j * (C3_AX + gap_x)
            ax = fig.add_axes((x0 / fig_w, y0 / fig_h, C3_AX / fig_w, C3_AX / fig_h))
            p = panels[metric][key]
            if p["no_data"]:
                ax.text(
                    0.5,
                    0.5,
                    NO_DATA,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=28,
                    color="0.5",
                    style="italic",
                )
            else:
                for name, ys in (("QALM", p["qalm"]), ("baseline", p["base"])):
                    _check_inside(f"{metric}/{key} {name}", ys, lims[key])
                handles["Baseline"] = ax.scatter(
                    p["base_x"], p["base"], s=14, color=BASE_COLOR, alpha=0.7, zorder=3
                )
                handles["QALM"] = ax.scatter(
                    range(len(p["qalm"])),
                    p["qalm"],
                    s=14,
                    color=QALM_COLOR,
                    alpha=0.8,
                    zorder=3,
                )
            if metric != "fidelity":
                ax.axhline(0, color="gray", ls="--", lw=0.6)
            ax.set_xlim(-1, len(p["qalm"]))
            ax.set_ylim(*lims[key])
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=24)
            # y tick labels only where the scale differs from the left panel
            labelled = j == 0 or lims[key] != lims[BASELINES[j - 1][0]]
            ax.tick_params(labelleft=labelled)
            if i == 0:
                ax.set_title(title, fontsize=32, fontweight="bold", pad=C3_TITLE_PAD)
            if j == 0:
                ax.set_ylabel(ROW_LABELS[metric], fontsize=28)
            row.append((ax, labelled))
        grid.append(row)
    for name, color, alpha in (
        ("QALM", QALM_COLOR, 0.8),
        ("Baseline", BASE_COLOR, 0.7),
    ):
        if name not in handles:  # no panel with data: proxy marker
            handles[name] = grid[0][0][0].scatter(
                [], [], s=14, color=color, alpha=alpha
            )
    top = C3_BOTTOM + len(METRICS) * C3_AX + (len(METRICS) - 1) * C3_GAP_Y
    fig.legend(
        [handles["QALM"], handles["Baseline"]],
        ["QALM", "Baseline"],
        loc="lower center",
        bbox_to_anchor=(0.5, (top + C3_LEGEND_Y) / fig_h),
        ncol=2,
        fontsize=30,
        frameon=False,
        markerscale=3.5,
    )
    return fig, _ylabel_overlap(fig, grid)


def _ylabel_overlap(fig, grid):
    """Worst overlap (points) of an inner panel's y tick labels with its left panel."""
    renderer = fig.canvas.get_renderer()
    worst = 0.0
    for row in grid:
        for (left, _), (ax, labelled) in zip(row, row[1:]):
            labels = [t for t in ax.get_yticklabels() if t.get_text()]
            if not labelled or not labels:
                continue
            x_min = min(t.get_window_extent(renderer).x0 for t in labels)
            overlap = left.get_window_extent(renderer).x1 - x_min
            worst = max(worst, overlap * 72 / fig.dpi)
    return worst


# ---------- win_rate ----------


def plot_win_rate(win, path):
    """win[key] = {metric: metrics.win_rate(...)} or None (no data)."""
    # The paper's thick hatching (bars and the legend's Tie patch) is rcParams'.
    with plt.rc_context({"hatch.linewidth": WR_HATCH_LW}):
        fig, ax = plt.subplots(figsize=WR_FIGSIZE)
        for i, (key, _, _, _) in enumerate(BASELINES):
            if win.get(key) is None:
                ax.text(
                    i,
                    50,
                    NO_DATA,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="0.5",
                    style="italic",
                )
                continue
            for j, metric in enumerate(METRICS):
                x = i + (j - 1) * WR_BAR_W
                _win_bar(ax, x, win[key][metric], BAR_COLORS[metric])
        ax.axhline(100, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.set_xlim(*_auto_lim(-1.5 * WR_BAR_W, len(BASELINES) - 1 + 1.5 * WR_BAR_W))
        ax.set_ylim(*WR_YLIM)
        ax.set_xticks(range(len(BASELINES)))
        ax.set_xticklabels(
            [t.replace(" (", "\n(") for _, t, _, _ in BASELINES],
            fontsize=12,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("QALM >= Baseline (%)", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        legend = [
            Patch(
                facecolor=BAR_COLORS[m],
                edgecolor="white",
                lw=WR_EDGE_LW,
                label=BAR_LABELS[m],
            )
            for m in METRICS
        ]
        legend.append(
            Patch(
                facecolor="gray", edgecolor="gray", alpha=0.4, hatch="//", label="Tie"
            )
        )
        ax.legend(handles=legend, loc="lower left", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=WR_DPI, bbox_inches="tight", metadata=PNG_META)
        plt.close(fig)


def _win_bar(ax, x, wr, color):
    """One bar: solid strict-win part, hatched tie part, '>=' label on top."""
    ax.bar(x, wr["win"], WR_BAR_W, color=color, edgecolor="white", lw=WR_EDGE_LW)
    if wr["tie"] > 0:
        ax.bar(
            x,
            wr["tie"],
            WR_BAR_W,
            bottom=wr["win"],
            color=color,
            alpha=0.4,
            hatch="//",
            edgecolor=color,
            lw=WR_TIE_EDGE_LW,
        )
    ax.text(
        x,
        wr["win_or_tie"] + WR_LABEL_DY,
        f"{wr['win_or_tie']:.1f}%",
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


# ---------- gate_scatter ----------


def scatter_points(counts, name):
    """(xs, ys, n_dropped) for log axes: the (2q, total) pairs with both > 0."""
    keep = [(cx, tot) for cx, tot in counts if cx > 0 and tot > 0]
    dropped = len(counts) - len(keep)
    if dropped:
        log.warning(
            "gate_scatter %s: %d circuits with a zero count dropped", name, dropped
        )
    return [p[0] for p in keep], [p[1] for p in keep], dropped


def plot_gate_scatter(series, path):
    """series = {"original"|"qalm"|"guoq_2q": [(cx, total)] or None (no data)}.

    Returns ({series: n dropped (zero counts)}, {series: n plotted}).
    """
    pts = {k: (scatter_points(v, k) if v else None) for k, v in series.items()}
    drawn = [p for p in pts.values() if p and p[0]]
    xmax = 1.5 * max(max(p[0]) for p in drawn)  # paper: limits (1, 1.5 x max)
    ymax = 1.5 * max(max(p[1]) for p in drawn)
    fig = plt.figure(figsize=GS_FIGSIZE)
    fig_w, fig_h = (v / PT for v in GS_FIGSIZE)
    rects = [(x / fig_w, y / fig_h, GS_AX / fig_w, GS_AX / fig_h) for x, y in GS_AXES]
    ax_l = fig.add_axes(rects[0])
    ax_r = fig.add_axes(rects[1], sharex=ax_l, sharey=ax_l)
    style = dict(s=34, edgecolors="white", linewidths=0.4, zorder=3)
    ax_l.scatter(
        *pts["original"][:2], color="#7f7f7f", alpha=0.7, label="Original", **style
    )
    for key, label, color in (
        ("guoq_2q", "GUOQ", BASE_COLOR),
        ("qalm", "QALM", QALM_COLOR),
    ):
        if pts[key] is None:
            ax_r.text(
                0.5,
                0.5,
                f"{label}: {NO_DATA}",
                transform=ax_r.transAxes,
                ha="center",
                va="center",
                fontsize=16,
                color="0.5",
                style="italic",
            )
        else:
            ax_r.scatter(*pts[key][:2], color=color, alpha=0.65, label=label, **style)
    for ax, title in ((ax_l, "Original"), (ax_r, "Optimized")):
        ax.plot(
            [1, xmax],
            [2, 2 * xmax],
            "--",
            color="#888888",
            lw=1.5,
            label="total = 2 × 2q",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        ax.set_ylim(1, ymax)
        ax.set_title(title, fontsize=24, fontweight="bold", pad=GS_TITLE_PAD)
        ax.set_xlabel("2-qubit gate count", fontsize=22)
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=16, loc="lower right", framealpha=0.9, markerscale=1.5)
    ax_l.set_ylabel("Total gate count", fontsize=22)
    ax_r.tick_params(labelleft=False)
    fig.savefig(path, bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)
    dropped = {k: (p[2] if p else None) for k, p in pts.items()}
    shown = {k: (len(p[0]) if p else 0) for k, p in pts.items()}
    return dropped, shown


# ---------- driver ----------


def build(rows, out_dir, qalm_config="a"):
    """Write the three figures and fig_percircuit.json; return the numbers."""
    os.makedirs(out_dir, exist_ok=True)
    res, panels, qalm = gather(rows, qalm_config)

    plot_comparison(panels, os.path.join(out_dir, "comparison_3x3.pdf"))
    res["comparison_3x3_points"] = {
        m: {k: [len(p["qalm"]), len(p["base"])] for k, p in panels[m].items()}
        for m in METRICS
    }

    win = {k: e.get("win_rate") for k, e in res["baselines"].items()}
    plot_win_rate(win, os.path.join(out_dir, "win_rate.png"))

    orig = original_counts(rows)
    guoq = by_circuit(rows, "guoq", opt="TWO_Q")
    series = {
        "original": [orig[c] for c in sorted(orig)],
        "qalm": [(r["opt_cx"], r["opt_total"]) for _, r in sorted(qalm.items())],
        "guoq_2q": [(r["opt_cx"], r["opt_total"]) for _, r in sorted(guoq.items())],
    }
    dropped, shown = plot_gate_scatter(
        series, os.path.join(out_dir, "gate_scatter.pdf")
    )
    res["gate_scatter"] = {"dropped_zero_count": dropped, "n_points": shown}

    res["outputs"] = ["comparison_3x3.pdf", "win_rate.png", "gate_scatter.pdf"]
    with open(os.path.join(out_dir, "fig_percircuit.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    return res


def summary_lines(res):
    """Human-readable summary of the plotted numbers."""
    q = res["qalm"]
    out = [
        f"QALM config {res['qalm_config']} ({q['optimizer']}): n={q['means']['n']} "
        f"non-OK={q['non_ok']} means {_fmt_means(q['means'])}"
    ]
    for e in res["baselines"].values():
        if e["no_data"]:
            out.append(f"{e['title']:19s} {NO_DATA}")
            continue
        wr = e["win_rate"]
        out.append(
            f"{e['title']:19s} n={e['n']} non-OK={e['non_ok']} "
            f"means {_fmt_means(e['means'])} (QALM {_fmt_means(e['qalm_means'])}) "
            "QALM>=base win+tie: "
            + " ".join(
                f"{m} {wr[m]['win']:.1f}+{wr[m]['tie']:.1f}={wr[m]['win_or_tie']:.1f}"
                for m in METRICS
            )
        )
    out.append(
        "gate_scatter dropped (zero counts): "
        + str(res["gate_scatter"]["dropped_zero_count"])
    )
    return out


def _fmt_means(m):
    return f"{m['fidelity']:.4f}/{m['cx_red']:.2f}/{m['total_red']:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="repro/results", help="dir with runs.csv")
    ap.add_argument("--out", default="repro/analysis/out", help="output dir")
    ap.add_argument(
        "--qalm-config",
        choices=sorted(QALM_CONFIGS),
        default="a",
        help="QALM configuration (default: a)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(name)s: %(levelname)s: %(message)s"
    )
    rows = metrics.load_runs(args.results)
    res = build(rows, args.out, args.qalm_config)
    for line in summary_lines(res):
        print(line)


if __name__ == "__main__":
    main()
