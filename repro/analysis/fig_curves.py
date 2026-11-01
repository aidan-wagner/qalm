"""Curve figures of the three ablation sections, and the numbers derived from them.

All curves use the 26-circuit suite S26, optimizer ``qalm`` and one time budget
(``--budget``, default 3600 s). Every QALM parameter not named below keeps its
default from ``repro/harness/jobs.py``.

Figures (written to ``--out``), styled like the paper's figures:
    greedy_k_reduction.pdf       E3: greedy_k in {0,1,2,3} (N_pool=1, N_branch=3,
                                 fixed_k=0, enqueue=0, seed 0)
    single_k_gk0.pdf             E4: fixed_k 1..5 with greedy_k=0 (resp. 2), plus
    single_k_gk2.pdf                 the advancing-k run of the same greedy_k
                                 (fixed_k=0) as a black dashed line
    N_pool_arith_mean_<T>s_plain.pdf    E5: N_pool 1..5 (N_branch=3), greedy_k=2,
    N_branch_arith_mean_<T>s_plain.pdf      N_branch 1..5 (N_pool=1), greedy_k=2,
                                 seeds {0,1,2}: the line is the mean over seeds,
                                 the light band spans min..max over seeds
    curves_numbers.json          every number the text quotes about these figures

Inputs (written by ``repro/harness/run.py collect``):
    <results>/runs.csv           one row per job (status, opt_total, greedy_end_s)
    <results>/timeseries.csv.gz  best-so-far change-points per job, with a t=0 row
    <results>/kevents.csv        QALM k-advancement events (cross-checks greedy_end_s)

Curves: each curve is the exact step function ``metrics.mean_reduction_curve``
(mean over circuits of the best-so-far total-gate reduction), drawn with
``drawstyle='steps-post'``. The final value of every curve is checked against
``metrics.averages`` of the runs.csv rows. Time-to-X% uses
``metrics.time_to_reach`` on the same step function.

Timestamp ties: collect.py writes the change-points sorted by (t, total, cx)
and ``metrics.load_timeseries`` sorts the same way, so an improvement logged
at the same timestamp as an earlier point (typically t=0.000, the same
instant as the original-count row) is placed *before* it and would be hidden.
Within one timestamp the best-so-far value is the smallest one, so this
script keeps the smallest (total, cx) per timestamp and reports how often
that happened (``data_issues.same_timestamp``).

Greedy phase (greedy_k figure only): per circuit the runs.csv greedy_end_s,
cross-checked against the first k = greedy_k + 1 event in kevents.csv. A run
that never left the greedy phase within the budget counts as the budget, and
so does a value past the budget (collect.py searches the whole log).

y-range: one range shared by the five figures, chosen from the data. Every
curve and band is fully visible from ``--zoom-from`` seconds on (default
budget/60, i.e. 60 s for one hour); before that, the initial rise from 0 %
(the original circuits at t=0) leaves through the bottom edge, as in the
paper. Bounds are rounded outward to 0.5 % with 15 % headroom above the
highest curve for the legend and a 2 % margin below the lowest one (so no
curve runs along the bottom axis line, which is drawn on top of the
curves). With ``--ylim`` the range is given explicitly;
either way the script fails if any curve would be clipped.

Strict mode (default when --budget is 3600): every configuration and seed
must be present with exactly the 26 circuits of repro/inputs/S26.csv, and any
inconsistency between runs.csv, the time series and the k events is an
error. Without strict mode, absent configurations and seeds are skipped with
a warning, each figure uses the circuits common to all of its curves, and
inconsistencies become warnings (all listed in curves_numbers.json).

Usage (from the repo root):
    uv run repro/analysis/fig_curves.py --results repro/results --out repro/analysis/out
    uv run repro/analysis/fig_curves.py --results repro/runs_smoke/results \\
        --out /tmp/smoke_out --budget 60
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

os.environ["SOURCE_DATE_EPOCH"] = "0"  # reproducible PDFs; set before matplotlib loads

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPRO, "harness"))

import metrics  # noqa: E402
from jobs import DEFAULTS  # noqa: E402

SUITE = "S26"
OPTIMIZER = "qalm"
MANIFEST = os.path.join(REPRO, "inputs", f"{SUITE}.csv")
REPORT_THRESHOLDS = (30.0, 31.0, 32.0, 33.0)
E5_SEEDS = (0, 1, 2)

# Paper look (qalm_experiments/plot_greedy_k.py, plot_single_k.py, npool_nbranch_experiment.py)
FIGSIZE = (7, 4)
LINEWIDTH = 1.8
LEGEND_FONTSIZE = 9
XLABEL = "Time (s)"
YLABEL = "Avg. Gate Count Reduction (%)"
BAND_ALPHA = 0.18
YLIM_STEP = 0.5
YLIM_HEADROOM = 0.15
YLIM_MARGIN = 0.02  # free space below the lowest curve, as a fraction of the data span
PDF_METADATA = {"CreationDate": None, "ModDate": None}


# ---------- what to plot ----------


def _params(**overrides):
    """Full QALM parameter set: the jobs.py defaults with the given overrides."""
    p = dict(DEFAULTS[OPTIMIZER])
    p.update(overrides)
    return p


def _curve(key, label, color, seeds, linestyle="-", **overrides):
    return {
        "key": key,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "seeds": tuple(seeds),
        "overrides": overrides,
        "params": _params(**overrides),
    }


def figure_specs(budget):
    """The five figures: file name, title (``{n}`` = circuit count) and curves."""
    specs = [
        {
            "key": "greedy_k",
            "file": "greedy_k_reduction.pdf",
            "title": "Effect of greedy_k ({n})",
            "curves": [
                _curve(
                    f"greedy_k={gk}",
                    f"greedy_k={gk}",
                    f"C{i}",
                    [0],
                    greedy_k=gk,
                    n_pool=1,
                    n_branch=3,
                    fixed_k=0,
                    enqueue=0,
                )
                for i, gk in enumerate((0, 1, 2, 3))
            ],
        }
    ]
    for gk in (0, 2):
        curves = [
            _curve(
                f"k={fk}",
                f"k={fk}",
                f"C{i}",
                [0],
                greedy_k=gk,
                n_pool=1,
                n_branch=3,
                fixed_k=fk,
                enqueue=0,
            )
            for i, fk in enumerate((1, 2, 3, 4, 5))
        ]
        curves.append(
            _curve(
                "advancing_k",
                f"k={gk + 1},{gk + 2},{gk + 3},...",
                "black",
                [0],
                linestyle="--",
                greedy_k=gk,
                n_pool=1,
                n_branch=3,
                fixed_k=0,
                enqueue=0,
            )
        )
        specs.append(
            {
                "key": f"single_k_gk{gk}",
                "file": f"single_k_gk{gk}.pdf",
                "title": f"Fixed k (greedy_k={gk}, {{n}})",
                "curves": curves,
            }
        )
    for name, other in (("N_pool", {"n_branch": 3}), ("N_branch", {"n_pool": 1})):
        param = name.lower()
        specs.append(
            {
                "key": name,
                "file": f"{name}_arith_mean_{budget}s_plain.pdf",
                "title": f"{name} comparison ({{n}})",
                "curves": [
                    _curve(
                        f"{name}={v}",
                        f"{name}={v}",
                        f"C{i}",
                        E5_SEEDS,
                        greedy_k=2,
                        fixed_k=0,
                        enqueue=0,
                        **{param: v},
                        **other,
                    )
                    for i, v in enumerate((1, 2, 3, 4, 5))
                ],
            }
        )
    return specs


# ---------- loading and consistency checks ----------


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_kevents(results_dir, job_ids):
    """{job_id: [(k, t_s), ...]} sorted by time, exact duplicates removed."""
    ev = defaultdict(set)
    with open(os.path.join(results_dir, "kevents.csv")) as f:
        for r in csv.DictReader(f):
            if r["job_id"] in job_ids:
                ev[r["job_id"]].add((int(r["k"]), float(r["t_s"])))
    return {j: sorted(s, key=lambda e: (e[1], e[0])) for j, s in ev.items()}


class Results:
    """QALM S26 rows of one budget, their cleaned time series and k events."""

    def __init__(self, results_dir, budget, strict):
        self.budget = budget
        self.strict = strict
        self.warnings = []
        self.problems = []
        self.same_t = {"jobs": 0, "timestamps": 0}
        self.greedy_end_capped = 0
        rows = self._dedupe(metrics.load_runs(results_dir))
        self.rows = [
            r
            for r in rows
            if r["optimizer"] == OPTIMIZER
            and r["suite"] == SUITE
            and r["budget_s"] == budget
        ]
        ids = {r["job_id"] for r in self.rows}
        self.raw_series = metrics.load_timeseries(results_dir, ids)
        self.kevents = load_kevents(results_dir, ids)
        self._series = {}

    def warn(self, msg):
        self.warnings.append(msg)
        print("WARNING:", msg, file=sys.stderr)

    def problem(self, msg):
        """A data inconsistency: fatal in strict mode, a recorded warning otherwise."""
        if self.strict:
            raise ValueError(msg + " (strict mode; use --no-strict to continue)")
        self.problems.append(msg)
        self.warn(msg)

    def _dedupe(self, rows):
        by_id, n_dup = {}, 0
        for r in rows:
            prev = by_id.setdefault(r["job_id"], r)
            if prev is r:
                continue
            if prev != r:
                raise ValueError(f"runs.csv: job {r['job_id']} has two different rows")
            n_dup += 1
        if n_dup:
            self.warn(f"runs.csv: ignored {n_dup} exact duplicate rows")
        self.duplicate_rows = n_dup
        return list(by_id.values())

    def unit(self, params, seed):
        """{circuit: row} of one configuration and seed (empty if absent)."""
        sel = metrics.select(
            self.rows,
            OPTIMIZER,
            suite=SUITE,
            budget_s=self.budget,
            seed=seed,
            **params,
        )
        by_c = {}
        for r in sel:
            if r["circuit"] in by_c:
                raise ValueError(
                    f"two runs of {r['circuit']} for {params} seed {seed}: "
                    f"{by_c[r['circuit']]['job_id']}, {r['job_id']}"
                )
            by_c[r["circuit"]] = r
        return by_c

    def series(self, r):
        """Best-so-far change-points [(t, total, cx)] of one run, starting at t=0."""
        jid = r["job_id"]
        if jid in self._series:
            return self._series[jid]
        raw = self.raw_series.get(jid)
        if not raw:
            raise ValueError(f"timeseries.csv.gz has no rows for job {jid}")
        where = f"job {jid} ({r['circuit']}, {r['params']}, seed {r['seed']})"
        if (0.0, r["orig_total"], r["orig_cx"]) not in raw:
            self.problem(f"{where}: no t=0 row with the original counts")
        best = {}
        for t, total, cx in raw:
            best.setdefault(t, set()).add((total, cx))
        ties = sum(len(v) > 1 for v in best.values())
        if ties:
            self.same_t["jobs"] += 1
            self.same_t["timestamps"] += ties
        s = [(t, *min(best[t])) for t in sorted(best)]
        if s[0][0] != 0.0:
            self.problem(f"{where}: series does not start at t=0")
        if s[-1][0] > self.budget:
            self.problem(f"{where}: change-point after the budget")
        if any(b[1] > a[1] for a, b in zip(s, s[1:])):
            self.problem(f"{where}: total gate count increases over time")
        if s[-1][1] != r["opt_total"]:
            self.problem(
                f"{where}: last change-point {s[-1][1]} != opt_total {r['opt_total']} "
                f"(result_source {r['result_source']})"
            )
        self._series[jid] = s
        return s

    def greedy_end(self, r):
        """runs.csv greedy_end_s, cross-checked against kevents.csv."""
        p = r["params"]
        target = p["fixed_k"] if p["fixed_k"] > 0 else p["greedy_k"] + 1
        hits = [
            t
            for k, t in self.kevents.get(r["job_id"], [])
            if k == target and t <= self.budget
        ]
        from_events = min(hits) if hits else float(self.budget)
        value = r.get("greedy_end_s")
        if value in (None, ""):
            self.problem(f"job {r['job_id']} ({r['circuit']}): greedy_end_s is empty")
            return from_events
        value = float(value)
        if value > self.budget:
            # collect.py takes the first event from the whole log, which can lie
            # just past the budget; the greedy phase did not end within the budget.
            self.greedy_end_capped += 1
            value = float(self.budget)
        if abs(value - from_events) > 1e-9:
            self.problem(
                f"job {r['job_id']} ({r['circuit']}): greedy_end_s {value} but the first "
                f"k={target} event is at {from_events}"
            )
        return value


def manifest_circuits():
    with open(MANIFEST) as f:
        return {r["name"] for r in csv.DictReader(f)}


# ---------- curves ----------


def eval_steps(curve, times):
    """Values of a step function [(t, v), ...] (starting at t=0) at the given times."""
    ts = np.array([p[0] for p in curve])
    vs = np.array([p[1] for p in curve])
    return vs[np.searchsorted(ts, times, side="right") - 1]


def _n_label(n):
    return f"{n} circuit" if n == 1 else f"{n} circuits"


def build_figure(res, spec):
    """Collect the runs of one figure and compute its curves (None if nothing to draw)."""
    present = []
    for cs in spec["curves"]:
        units = {}
        for seed in cs["seeds"]:
            by_c = res.unit(cs["params"], seed)
            if by_c:
                units[seed] = by_c
        missing = [s for s in cs["seeds"] if s not in units]
        if missing:
            res.problem(
                f"{spec['key']}: {cs['label']}: no runs for seed(s) {missing}"
                + (f"; using seed(s) {sorted(units)}" if units else "; curve skipped")
            )
        if units:
            present.append((cs, units))
    if not present:
        res.warn(f"{spec['key']}: no curve has any run; {spec['file']} not written")
        return None
    sets = [set(by_c) for _, units in present for by_c in units.values()]
    circuits = set.intersection(*sets)
    union = set.union(*sets)
    if circuits != union:
        res.problem(
            f"{spec['key']}: curves cover different circuits ({len(circuits)} common "
            f"to all; the figure uses only those): {sorted(union - circuits)} missing "
            "from some curve"
        )
    if res.strict:
        expected = manifest_circuits()
        if circuits != expected:
            raise ValueError(
                f"{spec['key']}: {len(circuits)} circuits, expected the "
                f"{len(expected)} of {MANIFEST}; missing {sorted(expected - circuits)}"
            )
    if not circuits:
        res.warn(
            f"{spec['key']}: no circuit is common to all curves; {spec['file']} not written"
        )
        return None
    circuits = sorted(circuits)
    budget = res.budget
    curves = []
    for cs, units in present:
        per_seed = {}
        for seed, by_c in sorted(units.items()):
            rows = [by_c[c] for c in circuits]
            curve = metrics.mean_reduction_curve(
                {c: res.series(by_c[c]) for c in circuits},
                {c: by_c[c]["orig_total"] for c in circuits},
                budget,
            )
            final_rows = metrics.averages(rows)["total_red"]
            if abs(curve[-1][1] - final_rows) > 1e-9:
                res.problem(
                    f"{spec['key']} {cs['label']} seed {seed}: curve ends at "
                    f"{curve[-1][1]} but runs.csv gives {final_rows}"
                )
            per_seed[seed] = {"curve": curve, "rows": rows, "by_c": by_c}
        if len(per_seed) == 1:
            (only,) = per_seed.values()
            line = only["curve"]
            band = None
        else:
            # Mean over (seed, circuit) of equal-sized seeds = mean over seeds of the
            # per-seed mean curves, as one exact step function.
            line = metrics.mean_reduction_curve(
                {
                    (s, c): res.series(d["by_c"][c])
                    for s, d in per_seed.items()
                    for c in circuits
                },
                {
                    (s, c): d["by_c"][c]["orig_total"]
                    for s, d in per_seed.items()
                    for c in circuits
                },
                budget,
            )
            t = np.array([p[0] for p in line])
            vals = np.vstack([eval_steps(d["curve"], t) for d in per_seed.values()])
            if not np.allclose(vals.mean(axis=0), [p[1] for p in line], atol=1e-9):
                raise AssertionError(f"{spec['key']} {cs['label']}: seed mean mismatch")
            band = (t, vals.min(axis=0), vals.max(axis=0))
        curves.append({"spec": cs, "per_seed": per_seed, "line": line, "band": band})
    return {"spec": spec, "circuits": circuits, "curves": curves}


# ---------- y-range and drawing ----------


def plotted_arrays(fig):
    """(name, t, v) of everything drawn: lines and both edges of every band."""
    out = []
    for c in fig["curves"]:
        name = f"{fig['spec']['key']} {c['spec']['label']}"
        out.append(
            (
                name,
                np.array([p[0] for p in c["line"]]),
                np.array([p[1] for p in c["line"]]),
            )
        )
        if c["band"] is not None:
            t, lo, hi = c["band"]
            out += [(name + " band min", t, lo), (name + " band max", t, hi)]
    return out


def choose_ylim(arrays, zoom_from):
    """Shared y-range: everything visible from zoom_from on, 0.5 % steps, legend headroom."""
    lo_data = min(eval_steps(list(zip(t, v)), [zoom_from])[0] for _, t, v in arrays)
    hi_data = max(float(v.max()) for _, t, v in arrays)
    margin = YLIM_MARGIN * (hi_data - lo_data)
    lo = math.floor((lo_data - margin) / YLIM_STEP) * YLIM_STEP
    hi = math.ceil((hi_data + YLIM_HEADROOM * (hi_data - lo)) / YLIM_STEP) * YLIM_STEP
    return (lo, max(hi, lo + YLIM_STEP))


def check_clipping(arrays, ylim, zoom_from):
    """Fail if any value is above the range, or below it at or after zoom_from."""
    lo, hi = ylim
    eps = 1e-9
    bad = []
    for name, t, v in arrays:
        start = np.searchsorted(t, zoom_from, side="right") - 1  # value in effect
        if v.max() > hi + eps:
            bad.append(f"{name}: max {v.max():.4f} > {hi}")
        if v[start:].min() < lo - eps:
            bad.append(f"{name}: {v[start:].min():.4f} < {lo} at t >= {zoom_from} s")
    if bad:
        raise ValueError(
            "curves would be clipped by ylim %s:\n  " % (ylim,) + "\n  ".join(bad)
        )


def draw(fig, path, ylim):
    f, ax = plt.subplots(figsize=FIGSIZE)
    for c in fig["curves"]:
        if c["band"] is not None:
            t, lo, hi = c["band"]
            ax.fill_between(
                t,
                lo,
                hi,
                step="post",
                color=c["spec"]["color"],
                alpha=BAND_ALPHA,
                linewidth=0,
            )
    for c in fig["curves"]:
        ax.plot(
            [p[0] for p in c["line"]],
            [p[1] for p in c["line"]],
            drawstyle="steps-post",
            label=c["spec"]["label"],
            color=c["spec"]["color"],
            linestyle=c["spec"]["linestyle"],
            linewidth=LINEWIDTH,
        )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_ylim(*ylim)
    ax.set_title(fig["spec"]["title"].format(n=_n_label(len(fig["circuits"]))))
    ax.legend(fontsize=LEGEND_FONTSIZE)
    f.tight_layout()
    f.savefig(path, metadata=PDF_METADATA)
    plt.close(f)


# ---------- numbers ----------


def _key(x):
    return f"{x:g}"


def curve_numbers(curve, thresholds, threshold):
    return {
        "final_mean_reduction_pct": curve[-1][1],
        "time_to_reach_s": {
            _key(x): metrics.time_to_reach(curve, x) for x in thresholds
        },
        "time_to_threshold_s": metrics.time_to_reach(curve, threshold),
    }


def greedy_phase(res, by_c, circuits):
    """Greedy-phase end per circuit (runs that never left it count as the budget)."""
    ends = {c: res.greedy_end(by_c[c]) for c in circuits}
    not_finished = sorted(c for c, t in ends.items() if t >= res.budget)
    return {
        "mean_end_s": metrics.mean(ends.values()),
        "n_circuits": len(ends),
        "n_not_finished": len(not_finished),
        "not_finished_circuits": not_finished,
        "end_s_by_circuit": ends,
    }


def figure_numbers(res, fig, thresholds, threshold):
    out = {
        "file": fig["spec"]["file"],
        "n_circuits": len(fig["circuits"]),
        "circuits": fig["circuits"],
        "curves": {},
    }
    drawn = {c["spec"]["key"] for c in fig["curves"]}
    out["absent_curves"] = [
        cs["label"] for cs in fig["spec"]["curves"] if cs["key"] not in drawn
    ]
    for c in fig["curves"]:
        cs = c["spec"]
        per_seed = {}
        for seed, d in c["per_seed"].items():
            n = curve_numbers(d["curve"], thresholds, threshold)
            n["status_counts"] = dict(
                sorted(Counter(r["status"] for r in d["rows"]).items())
            )
            n["non_ok_runs"] = {
                r["circuit"]: r["status"] for r in d["rows"] if r["status"] != "OK"
            }
            if fig["spec"]["key"] == "greedy_k":
                n["greedy_phase"] = greedy_phase(res, d["by_c"], fig["circuits"])
            per_seed[str(seed)] = n
        finals = [d["curve"][-1][1] for d in c["per_seed"].values()]
        entry = {
            "label": cs["label"],
            "params": cs["overrides"],
            "seeds_requested": list(cs["seeds"]),
            "seeds_present": sorted(c["per_seed"]),
            "per_seed": per_seed,
            "final_sd_across_seeds_pct": (
                statistics.stdev(finals) if len(finals) > 1 else None
            ),
            "final_min_across_seeds_pct": min(finals),
            "final_max_across_seeds_pct": max(finals),
        }
        # Final value and time-to-X of the drawn line (= the mean over seeds)
        entry.update(curve_numbers(c["line"], thresholds, threshold))
        out["curves"][cs["key"]] = entry
    return out


def derived_numbers(figs):
    """Comparisons the text makes between curves of a figure."""
    out = {}
    g = figs.get("greedy_k")
    if g:
        base = g["curves"].get("greedy_k=0", {}).get("time_to_threshold_s")
        out["greedy_k_speedup_vs_greedy_k=0_at_threshold"] = {
            k: (
                base / v["time_to_threshold_s"]
                if base and v["time_to_threshold_s"]
                else None
            )
            for k, v in g["curves"].items()
        }
        out["greedy_k_mean_greedy_end_s"] = {
            k: v["per_seed"]["0"]["greedy_phase"]["mean_end_s"]
            for k, v in g["curves"].items()
        }
        out["greedy_k_n_not_finished_greedy"] = {
            k: v["per_seed"]["0"]["greedy_phase"]["n_not_finished"]
            for k, v in g["curves"].items()
        }
    for gk in (0, 2):
        f = figs.get(f"single_k_gk{gk}")
        if not f:
            continue
        fixed = {k: v for k, v in f["curves"].items() if k != "advancing_k"}
        d = {}
        if fixed:
            best = max(fixed, key=lambda k: fixed[k]["final_mean_reduction_pct"])
            d["best_fixed_k"] = best
            d["best_fixed_k_final_pct"] = fixed[best]["final_mean_reduction_pct"]
        if "advancing_k" in f["curves"]:
            adv = f["curves"]["advancing_k"]["final_mean_reduction_pct"]
            d["advancing_k_final_pct"] = adv
            if fixed:
                d["advancing_minus_best_fixed_pct"] = adv - d["best_fixed_k_final_pct"]
        out[f"single_k_gk{gk}"] = d
    for name, default_key in (("N_pool", "N_pool=1"), ("N_branch", "N_branch=3")):
        f = figs.get(name)
        if not f:
            continue
        finals = {k: v["final_mean_reduction_pct"] for k, v in f["curves"].items()}
        out[name] = {
            "mean_final_min_pct": min(finals.values()),
            "mean_final_max_pct": max(finals.values()),
            "mean_final_spread_pct": max(finals.values()) - min(finals.values()),
            "best": max(finals, key=finals.get),
        }
        if default_key in f["curves"]:
            v = f["curves"][default_key]
            out.setdefault(
                "E5_default_config",
                {
                    "params": v["params"],
                    "seeds_present": v["seeds_present"],
                    "final_by_seed_pct": {
                        s: d["final_mean_reduction_pct"]
                        for s, d in v["per_seed"].items()
                    },
                    "final_mean_pct": v["final_mean_reduction_pct"],
                    "final_sd_across_seeds_pct": v["final_sd_across_seeds_pct"],
                    "final_range_across_seeds_pct": v["final_max_across_seeds_pct"]
                    - v["final_min_across_seeds_pct"],
                },
            )
    return out


# ---------- main ----------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="repro/results", help="dir with runs.csv etc.")
    ap.add_argument("--out", default="repro/analysis/out", help="output dir")
    ap.add_argument("--budget", type=int, default=3600, help="budget_s of the runs")
    ap.add_argument(
        "--threshold",
        type=float,
        default=32.0,
        help="mean reduction (%%) for time_to_threshold_s (default 32)",
    )
    ap.add_argument(
        "--zoom-from",
        type=float,
        default=None,
        help="curves must be fully visible from this time on (default budget/60)",
    )
    ap.add_argument(
        "--ylim", type=float, nargs=2, default=None, help="explicit y-range (checked)"
    )
    ap.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="require complete, consistent data (default: on iff --budget is 3600)",
    )
    a = ap.parse_args(argv)
    strict = a.strict if a.strict is not None else a.budget == 3600
    zoom_from = a.zoom_from if a.zoom_from is not None else a.budget / 60
    if not 0 <= zoom_from <= a.budget:
        ap.error("--zoom-from must lie within [0, budget]")
    thresholds = sorted(set(REPORT_THRESHOLDS) | {a.threshold})

    plt.rcdefaults()  # the paper figures used matplotlib's default style
    res = Results(a.results, a.budget, strict)
    figs = []
    specs = figure_specs(a.budget)
    for spec in specs:
        fig = build_figure(res, spec)
        if fig is not None:
            figs.append(fig)
    if not figs:
        raise SystemExit(f"no QALM {SUITE} runs with budget {a.budget} in {a.results}")

    arrays = [x for fig in figs for x in plotted_arrays(fig)]
    ylim = tuple(a.ylim) if a.ylim else choose_ylim(arrays, zoom_from)
    check_clipping(arrays, ylim, zoom_from)

    numbers_figs = {
        fig["spec"]["key"]: figure_numbers(res, fig, thresholds, a.threshold)
        for fig in figs
    }
    skipped = [s["file"] for s in specs if s["key"] not in numbers_figs]
    os.makedirs(a.out, exist_ok=True)
    for fig in figs:
        draw(fig, os.path.join(a.out, fig["spec"]["file"]), ylim)
    numbers = {
        "meta": {
            "script": "repro/analysis/fig_curves.py",
            "suite": SUITE,
            "optimizer": OPTIMIZER,
            "budget_s": a.budget,
            "strict": strict,
            "threshold_pct": a.threshold,
            "thresholds_pct": thresholds,
            "zoom_from_s": zoom_from,
            "ylim": list(ylim),
            "inputs_sha256": {
                n: _sha256(os.path.join(a.results, n))
                for n in ("runs.csv", "timeseries.csv.gz", "kevents.csv")
            },
            "notes": [
                "final_mean_reduction_pct and time_to_* refer to the drawn line "
                "(for E5: the mean over seeds); per_seed holds each seed",
                "final_sd_across_seeds_pct is the sample SD (n-1)",
                "greedy_phase.mean_end_s counts runs that never left the greedy "
                "phase as the budget",
            ],
        },
        "figures": numbers_figs,
        "figures_not_written": skipped,
        "derived": derived_numbers(numbers_figs),
        "data_issues": {
            "duplicate_runs_rows": res.duplicate_rows,
            "same_timestamp": res.same_t,
            "greedy_end_s_after_budget_capped": res.greedy_end_capped,
            "problems": res.problems,
        },
        "warnings": res.warnings,
    }
    path = os.path.join(a.out, "curves_numbers.json")
    with open(path, "w") as f:
        json.dump(numbers, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"y-range {ylim}, fully visible from t = {zoom_from:g} s")
    for key, fn in numbers_figs.items():
        print(f"{fn['file']}: {_n_label(fn['n_circuits'])}")
        for c in fn["curves"].values():
            tt = c["time_to_threshold_s"]
            non_ok = sum(len(d["non_ok_runs"]) for d in c["per_seed"].values())
            print(
                f"  {c['label']:<14} final {c['final_mean_reduction_pct']:.3f} %  "
                f"t({a.threshold:g}%) = {'never' if tt is None else f'{tt:.1f} s'}"
                f"  seeds {c['seeds_present']}"
                + (f"  non-OK runs {non_ok}" if non_ok else "")
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
