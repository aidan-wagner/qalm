"""Compare QALM configuration a (the paper's) with configuration b (the old main run).

Inputs
    ``<results>/runs.csv`` written by ``repro/harness/run.py collect``, loaded
    with ``metrics.load_runs``.  Columns are selected with
    ``tables.select_column``, which checks every row against the suite
    manifests ``repro/inputs/{S248,S26,S26PP}.csv`` (input hash, original
    counts, one row per circuit, no INFRA rows).

Outputs (in ``--out``, default ``repro/analysis/out``)
    compare_qalm_configs.md    the report described below
    compare_qalm_configs.json  every number of the report, unrounded, plus
                               per a-vs-b section the per-circuit counts,
                               differences and comparisons ("per_circuit")

Configurations (``tables.QALM_CONFIGS``), seed 0 unless stated otherwise:
    a  optimizer ``qalm`` with greedy_k=2, n_pool=1, n_branch=3, fixed_k=0,
       enqueue=0: the configuration the paper describes.
    b  optimizer ``qalm_old`` (``test_qalm 1 1 3 1.5 0 0 1 1 0``): the
       configuration of the previous main run.

Report sections
    1-3. a vs b on S248 at 3600 s, S248 at 60 s and, if config b has any S26
         row, S26 at 3600 s.  Per metric (fidelity, CX reduction, total
         reduction; ``metrics.averages`` / ``fig_percircuit.value``): means of
         a and b over the circuits both have, per-circuit differences a - b
         (mean, median, quartiles, IQR), a-better / tie / b-better counts
         (``metrics.compare``: gate counts as integers, fidelity in log space
         with ties |d ln F| <= metrics.FID_TIE) and a two-sided Wilcoxon
         signed-rank test (``scipy.stats.wilcoxon``, zero_method='zsplit', so
         ties stay in the ranking).  The fidelity test uses the differences of
         ln F with the same tie rule, because F underflows to 0 on the largest
         circuits; its mean/median/IQR are given in F.  Positive differences
         mean a is better for all three metrics.  Then the circuits with the
         largest differences in each direction, per metric.
    4.   Against the baselines: tab:avg-comparison numbers
         (``tables.averages_table_data``) and per-circuit win rates
         (``metrics.win_rate``) under a and under b, and every headline
         statement of the paper about them (abstract, introduction, Sec. 5;
         taken from repro/reference/audit_20261030/audit_results.json)
         evaluated under a and under b.  Statements with a crisp condition get
         holds yes/no; qualitative ones ("nearly identical", "nearly every")
         only get their numbers.  "Strongest baseline" is the 1 h baseline
         column with the highest mean of the metric in question.  The same
         for tab:eval_end_to_end (``tables.eval_end_to_end_data``); its
         statements under b are "no data" when config b has no S26 row.
    5.   Noise yardstick: config a on S26 at 3600 s with seeds 0, 1, 2 (E5).
         Per seed the 26-circuit means; their sample SD across seeds (as
         fig_curves.py's final_sd_across_seeds_pct) and range; the
         per-circuit SD across seeds; how many circuits change gate count
         between two seeds.  Each a - b difference of means is divided by the
         SD of the same metric.
    6.   A neutral summary paragraph: the differences, their size relative to
         the yardstick, and the headline statements whose outcome differs
         between a and b.  It makes no recommendation.

Completeness: by default every selection used must hold every circuit of its
suite (S248 a/b at both budgets, the six 1 h S248 baselines, S26 a, the
five S26 baselines, the three E5 seeds and, when config b has any S26 row
(those runs are optional in E2), S26 b); otherwise the script stops and
lists what is missing.  With
``--allow-partial`` it reports on the circuits present: a vs b on the circuits
both have, the headline statements on the circuits common to every column
that has data (statements needing a column without data are "no data"), the
yardstick on the circuits common to the seeds present.  The report states
every coverage.

Usage (from the repo root):
    python3 repro/analysis/compare_qalm_configs.py --results repro/results \\
        --out repro/analysis/out [--allow-partial]
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fig_percircuit  # noqa: E402  (value(), status_counts())
import metrics  # noqa: E402
import tables  # noqa: E402

CONFIGS = ("a", "b")
CONFIG_DESC = {
    "a": "optimizer `qalm`, greedy_k=2, n_pool=1, n_branch=3, fixed_k=0, "
    "enqueue=0 (the configuration the paper describes)",
    "b": "optimizer `qalm_old` (`test_qalm 1 1 3 1.5 0 0 1 1 0`, the previous "
    "main run's configuration)",
}
# (key, suite, budget_s, title); the S26 section exists only if b has S26 rows
SECTIONS = [
    ("S248_3600", "S248", 3600, "S248 (248 circuits), 3600 s"),
    ("S248_60", "S248", 60, "S248 (248 circuits), 60 s"),
    ("S26_3600", "S26", 3600, "S26 (26 circuits), 3600 s"),
]
OPTIONAL_SECTION = "S26_3600"
METRICS = ("fidelity", "cx", "total")  # names of metrics.compare
AVG_KEY = {"fidelity": "fidelity", "cx": "cx_red", "total": "total_red"}
METRIC_LABEL = {
    "fidelity": "Fidelity",
    "cx": "CX reduction (%)",
    "total": "Total reduction (%)",
}
E5_SEEDS = (0, 1, 2)
TOP_N = 5
S248_BASELINES = [c.key for c in tables.averages_columns("a")[2:]]
S26_BASELINES = [c.key for c in tables.e2e_columns("a")[:-1]]
OUT_MD = "compare_qalm_configs.md"
OUT_JSON = "compare_qalm_configs.json"


# ---------- selection ----------


def config_column(config, suite, budget_s):
    """tables.Column of one QALM configuration on one suite and budget."""
    opt, params = tables.QALM_CONFIGS[config]
    return tables.Column(
        f"qalm_{config}", f"QALM ({config})", opt, suite, budget_s, params
    )


class Selections:
    """Named {circuit: row} selections with their coverage of the suite."""

    def __init__(self, rows):
        self.rows = rows
        self.sel = {}

    def add(self, key, col, seed=0, required=True):
        # tables.select_column only takes seed 0: pass the seed-s rows relabelled
        rows = self.rows
        if seed != 0:
            rows = [dict(r, seed=0) for r in rows if r["seed"] == seed]
        by_circ = tables.select_column(rows, col, allow_partial=True)
        self.sel[key] = {
            "column": col,
            "seed": seed,
            "required": required,
            "rows": by_circ,
            "n": len(by_circ),
            "n_suite": len(tables.load_manifest(col.suite)),
        }
        return by_circ

    def __getitem__(self, key):
        return self.sel[key]["rows"]

    def incomplete(self):
        """Required selections that do not hold every circuit of their suite."""
        return [
            f"{k}: {s['n']} of {s['n_suite']} circuits "
            f"({s['column'].optimizer} {s['column'].params} {s['column'].suite} "
            f"{s['column'].budget_s} s seed {s['seed']})"
            for k, s in self.sel.items()
            if s["required"] and s["n"] < s["n_suite"]
        ]

    def coverage(self):
        return {
            k: {
                "optimizer": s["column"].optimizer,
                "params": dict(s["column"].params),
                "suite": s["column"].suite,
                "budget_s": s["column"].budget_s,
                "seed": s["seed"],
                "n": s["n"],
                "n_suite": s["n_suite"],
                "non_ok": fig_percircuit.status_counts(s["rows"].values()),
            }
            for k, s in self.sel.items()
        }


def gather(rows):
    """All selections the report uses; returns (Selections, s26_enabled)."""
    sel = Selections(rows)
    for key, suite, budget, _ in SECTIONS:
        for cfg in CONFIGS:
            sel.add(f"{cfg}_{key}", config_column(cfg, suite, budget))
    # config b on S26 is optional in E2 (priority 3); a and the baselines are not
    s26 = bool(sel[f"b_{OPTIONAL_SECTION}"])
    sel.sel[f"b_{OPTIONAL_SECTION}"]["required"] = s26
    for col in tables.averages_columns("a")[2:]:
        sel.add(f"S248_{col.key}", col)
    for col in tables.e2e_columns("a")[:-1]:
        sel.add(f"S26_{col.key}", col)
    for s in E5_SEEDS:
        sel.add(f"e5_seed{s}", config_column("a", "S26", 3600), seed=s)
    return sel, s26


# ---------- a vs b ----------


def _num(x):
    """Plain float for JSON (None for None/NaN)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank test with zero_method='zsplit'."""
    d = np.asarray(d, dtype=float)
    out = {"n": int(d.size), "n_nonzero": int(np.count_nonzero(d))}
    if d.size == 0:
        return dict(out, statistic=None, pvalue=None, note="no pairs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = stats.wilcoxon(d, zero_method="zsplit")
            stat, p = _num(r.statistic), _num(r.pvalue)
            note = ""
        except ValueError as e:
            stat, p, note = None, None, str(e)
    if caught:
        note = "; ".join([note] * bool(note) + [str(w.message) for w in caught])
    return dict(out, statistic=stat, pvalue=p, note=note)


def _quantiles(d):
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return {
        "mean": _num(np.mean(d)),
        "median": _num(med),
        "q1": _num(q1),
        "q3": _num(q3),
        "iqr": _num(q3 - q1),
        "min": _num(np.min(d)),
        "max": _num(np.max(d)),
    }


def _brief(row):
    return {
        "total": row["opt_total"],
        "cx": row["opt_cx"],
        "status": row["status"],
    }


def compare_metric(A, B, common, metric):
    """Paired statistics of one metric over the common circuits (a - b)."""
    cmp = Counter(metrics.compare(A[c], B[c], metric) for c in common)
    out = {
        "n_compared": len(common),
        "a_better": cmp[1],
        "tie": cmp[0],
        "b_better": cmp[-1],
    }
    vals = []
    for c in common:
        va = fig_percircuit.value(A[c], metric)
        vb = fig_percircuit.value(B[c], metric)
        if va is not None and vb is not None:
            vals.append((c, va, vb))
    out["n"] = len(vals)
    if not vals:
        return dict(out, mean_a=None, mean_b=None, diff=None, wilcoxon=wilcoxon([]))
    d = np.array([va - vb for _, va, vb in vals])
    out["mean_a"] = _num(np.mean([va for _, va, _ in vals]))
    out["mean_b"] = _num(np.mean([vb for _, _, vb in vals]))
    out["diff"] = _quantiles(d)
    if metric == "fidelity":
        dl = []
        for c, _, _ in vals:
            x = metrics.log_fidelity(
                A[c]["opt_total"], A[c]["opt_cx"]
            ) - metrics.log_fidelity(B[c]["opt_total"], B[c]["opt_cx"])
            dl.append(0.0 if abs(x) <= metrics.FID_TIE else x)
        out["diff_lnF"] = _quantiles(np.array(dl))
        out["wilcoxon"] = dict(wilcoxon(dl), on="ln F differences")
    else:
        out["wilcoxon"] = dict(wilcoxon(d), on="per-circuit differences (pp)")
    diffs = [(float(x), c) for x, (c, _, _) in zip(d, vals)]

    def top(pairs):
        return [
            {
                "circuit": c,
                "diff": x,
                "orig_total": A[c]["orig_total"],
                "orig_cx": A[c]["orig_cx"],
                "a": _brief(A[c]),
                "b": _brief(B[c]),
            }
            for x, c in pairs[:TOP_N]
        ]

    # largest first; ties by circuit name
    out["largest_a_better"] = top(sorted((-x, c) for x, c in diffs if x > 0))
    out["largest_b_better"] = top(sorted((x, c) for x, c in diffs if x < 0))
    for e in out["largest_a_better"]:
        e["diff"] = -e["diff"]
    return out


def per_circuit(ra, rb):
    """One paired circuit: counts, a - b per metric (None if undefined), compare."""
    diff = {}
    for m in METRICS:
        va, vb = fig_percircuit.value(ra, m), fig_percircuit.value(rb, m)
        diff[m] = va - vb if va is not None and vb is not None else None
    diff["lnF"] = metrics.log_fidelity(
        ra["opt_total"], ra["opt_cx"]
    ) - metrics.log_fidelity(rb["opt_total"], rb["opt_cx"])
    return {
        "orig_total": ra["orig_total"],
        "orig_cx": ra["orig_cx"],
        "a": _brief(ra),
        "b": _brief(rb),
        "diff": diff,
        "compare": {m: metrics.compare(ra, rb, m) for m in METRICS},
    }


def compare_configs(A, B, suite):
    """a vs b on one suite and budget ({circuit: row} each)."""
    common = sorted(set(A) & set(B))
    out = {
        "suite": suite,
        "n_a": len(A),
        "n_b": len(B),
        "n_paired": len(common),
        "only_a": sorted(set(A) - set(B)),
        "only_b": sorted(set(B) - set(A)),
        "non_ok_a": fig_percircuit.status_counts(A[c] for c in common),
        "non_ok_b": fig_percircuit.status_counts(B[c] for c in common),
        "metrics": {},
    }
    if not common:
        return out
    avg = {
        cfg: metrics.averages([X[c] for c in common]) for cfg, X in (("a", A), ("b", B))
    }
    out["averages"] = avg
    for m in METRICS:
        r = compare_metric(A, B, common, m)
        # the paired means are the tab:avg-comparison definition
        assert math.isclose(r["mean_a"], avg["a"][AVG_KEY[m]], abs_tol=1e-9)
        assert math.isclose(r["mean_b"], avg["b"][AVG_KEY[m]], abs_tol=1e-9)
        out["metrics"][m] = r
    out["per_circuit"] = {c: per_circuit(A[c], B[c]) for c in common}
    if suite == "S26":
        out["geomean_red"] = {
            cfg: metrics.geomean_reduction_pct(
                (X[c]["orig_total"], X[c]["opt_total"]) for c in common
            )
            for cfg, X in (("a", A), ("b", B))
        }
    return out


# ---------- headline statements against the baselines ----------


def _common(sel, keys):
    """Circuits common to every listed selection that has rows; absent keys."""
    present = [k for k in keys if sel[k]]
    common = set.intersection(*(set(sel[k]) for k in present)) if present else set()
    return sorted(common), [k for k in keys if not sel[k]]


def _restrict(rows, suites, circuits):
    keep = set(circuits)
    return [r for r in rows if r["suite"] not in suites or r["circuit"] in keep]


# Headline statements of the paper: id -> (location, paper value, statement
# evaluated here).  Source: repro/reference/audit_20261030/audit_results.json
# (items A1, A2, B).  Ids starting with "s26_" are about tab:eval_end_to_end.
CLAIMS = {
    "intro_1h_total": (
        "texts/1-introduction.tex:123-124",
        "5.97 (vs VOQC)",
        "QALM (1 h) total reduction minus the strongest 1 h baseline's > 0",
    ),
    "intro_1min_total": (
        "texts/1-introduction.tex:124-126",
        "4.98 (vs VOQC)",
        "QALM (1 min) total reduction minus the strongest 1 h baseline's > 0",
    ),
    "eff_drop_lt_1": (
        "texts/5-eval.tex:138-140",
        "<1% (0.3864->0.3747, 24.61->21.18, 52.34->51.35)",
        "1 h -> 1 min drop < 1 (fidelity: relative %; reductions: pp)",
    ),
    "eff_1min_beats_all": (
        "texts/5-eval.tex:140-141",
        "0.3747 > 0.3676; 51.35 > 46.37",
        "QALM (1 min) fidelity and total reduction > those of every 1 h baseline",
    ),
    "guoq2q_cx_similar": (
        "texts/5-eval.tex:116-118",
        "24.61 vs 25.41",
        "CX reduction 'nearly identical' to GUOQ (2Q)'s (value only)",
    ),
    "guoq2q_total_better": (
        "texts/5-eval.tex:118-121",
        "52.34 vs -2.63",
        "total reduction > GUOQ (2Q)'s",
    ),
    "no_negative_total": (
        "texts/5-eval.tex:119-121",
        "QALM none; GUOQ (2Q) many",
        "no QALM circuit has a negative total reduction",
    ),
    "voqc_cx_greater": (
        "texts/5-eval.tex:121-124",
        "CX 24.61 vs 4.24; total 52.34 vs 46.37",
        "CX reduction > VOQC's (total 'similar' is qualitative)",
    ),
    "fid_vs_voqc_guoq2q": (
        "texts/5-eval.tex:124-127",
        "0.3864 vs 0.3428 / 0.3475",
        "fidelity > VOQC's and GUOQ (2Q)'s",
    ),
    "better_all_three": (
        "texts/5-eval.tex:131-134",
        "yes",
        "all three means > those of QUESO (2Q), QUESO (Total), Quartz, " "GUOQ (Total)",
    ),
    "abstract_fid_win_or_tie_guoq2q": (
        "texts/0-abstract.tex:38-40; texts/1-introduction.tex:134-136",
        "83.9%",
        "share of circuits with fidelity >= GUOQ (2Q)'s (value only)",
    ),
    "nearly_every_circuit": (
        "texts/1-introduction.tex:130-134",
        "95.2-99.6% (VOQC 86.5 / 82.0)",
        "'nearly every' circuit: lowest win-or-tie share vs QUESO (both), VOQC, "
        "Quartz, GUOQ (Total) over the 3 metrics (value only)",
    ),
    "majority_fid_wins": (
        "texts/5-eval.tex:105-109",
        "majority",
        "strictly higher fidelity than every baseline on > 50% of circuits",
    ),
    "s26_lowest_all_but_one": (
        "texts/5-eval.tex:155-159",
        "1 (csum_mux_9)",
        "circuits where another column has a lower gate count; 'all but one' "
        "(holds: <= 1)",
    ),
    "s26_geomean_vs_voqc": (
        "texts/5-eval.tex:159-160",
        "7.3",
        "geomean reduction minus VOQC's > 0",
    ),
    "s26_geomean_vs_quartz_nopp": (
        "texts/5-eval.tex:160-162",
        "17.8",
        "geomean reduction minus Quartz w/o preprocessing's > 0",
    ),
    "s26_geomean_vs_queso": (
        "texts/5-eval.tex:163",
        "10.4",
        "geomean reduction minus QUESO's > 0",
    ),
    "s26_geomean_vs_guoq": (
        "texts/5-eval.tex:163",
        "5.1",
        "geomean reduction minus GUOQ's > 0",
    ),
    "s26_geomean_vs_quartz_pp": (
        "texts/5-eval.tex:163-164",
        "4.6",
        "geomean reduction minus Quartz w/ preprocessing's > 0",
    ),
}


def _claim(cid, value=None, holds=None, shown="no data"):
    loc, paper, text = CLAIMS[cid]
    return {
        "id": cid,
        "paper_location": loc,
        "paper_value": paper,
        "statement": text,
        "value": value,
        "holds": holds,
        "shown": shown,
    }


def _fp(x, digits=2):
    return "n/a" if x is None else f"{x:+.{digits}f}"


def s248_claims(cols, per_circ):
    """Headline statements about tab:avg-comparison / win_rate for one config.

    cols: averages_table_data(...)["columns"]; per_circ: {column key:
    {circuit: row}} over the common circuits.  Returns the claims in CLAIMS
    order; a statement needing a column without data is "no data".
    """
    q, q1 = cols["qalm"], cols["qalm_1min"]
    out = {}

    def have(*keys):
        return all(cols[k]["n"] for k in keys)

    def strongest(metric):
        k = max(S248_BASELINES, key=lambda k: cols[k][metric])
        return k, cols[k][metric]

    all_b = have(*S248_BASELINES)
    for cid, qk, label in (
        ("intro_1h_total", "qalm", "1 h"),
        ("intro_1min_total", "qalm_1min", "1 min"),
    ):
        if have(qk) and all_b:
            k, v = strongest("total_red")
            d = cols[qk]["total_red"] - v
            out[cid] = _claim(
                cid, {"diff_pp": d, "strongest": k}, d > 0, f"{_fp(d)} pp vs {k}"
            )
    if have("qalm", "qalm_1min"):
        drop = {
            "fidelity_rel_pct": 100 * (q["fidelity"] - q1["fidelity"]) / q["fidelity"],
            "cx_pp": q["cx_red"] - q1["cx_red"],
            "total_pp": q["total_red"] - q1["total_red"],
        }
        out["eff_drop_lt_1"] = _claim(
            "eff_drop_lt_1",
            drop,
            all(v < 1 for v in drop.values()),
            "fid {:.2f}%, CX {:.2f} pp, total {:.2f} pp".format(*drop.values()),
        )
    if have("qalm_1min") and all_b:
        kf, vf = strongest("fidelity")
        kt, vt = strongest("total_red")
        d = {"fidelity": q1["fidelity"] - vf, "total_pp": q1["total_red"] - vt}
        out["eff_1min_beats_all"] = _claim(
            "eff_1min_beats_all",
            dict(d, strongest_fidelity=kf, strongest_total=kt),
            d["fidelity"] > 0 and d["total_pp"] > 0,
            f"fid {_fp(d['fidelity'], 4)} vs {kf}, "
            f"total {_fp(d['total_pp'])} pp vs {kt}",
        )
    if have("qalm", "guoq_2q"):
        g = cols["guoq_2q"]
        d = q["cx_red"] - g["cx_red"]
        out["guoq2q_cx_similar"] = _claim(
            "guoq2q_cx_similar",
            {"diff_pp": d},
            None,
            f"{q['cx_red']:.2f} vs {g['cx_red']:.2f} ({_fp(d)} pp)",
        )
        d = q["total_red"] - g["total_red"]
        out["guoq2q_total_better"] = _claim(
            "guoq2q_total_better",
            {"diff_pp": d},
            d > 0,
            f"{q['total_red']:.2f} vs {g['total_red']:.2f}",
        )
        neg = {
            k: sum(
                metrics.reduction_pct(r["orig_total"], r["opt_total"]) < 0
                for r in per_circ[k].values()
            )
            for k in ("qalm", "guoq_2q")
        }
        out["no_negative_total"] = _claim(
            "no_negative_total",
            neg,
            neg["qalm"] == 0,
            f"QALM {neg['qalm']}, GUOQ (2Q) {neg['guoq_2q']} of {q['n']}",
        )
    if have("qalm", "voqc"):
        v = cols["voqc"]
        d = {
            "cx_pp": q["cx_red"] - v["cx_red"],
            "total_pp": q["total_red"] - v["total_red"],
        }
        out["voqc_cx_greater"] = _claim(
            "voqc_cx_greater",
            d,
            d["cx_pp"] > 0,
            f"CX {_fp(d['cx_pp'])} pp, total {_fp(d['total_pp'])} pp",
        )
    if have("qalm", "voqc", "guoq_2q"):
        d = {k: q["fidelity"] - cols[k]["fidelity"] for k in ("voqc", "guoq_2q")}
        out["fid_vs_voqc_guoq2q"] = _claim(
            "fid_vs_voqc_guoq2q",
            d,
            all(x > 0 for x in d.values()),
            f"{q['fidelity']:.4f}: {_fp(d['voqc'], 4)} vs VOQC, "
            f"{_fp(d['guoq_2q'], 4)} vs GUOQ (2Q)",
        )
    group = ["queso_2q", "queso_total", "quartz", "guoq_total"]
    if have("qalm", *group):
        fails = [
            f"{k} {AVG_KEY[m]}"
            for k in group
            for m in METRICS
            if not q[AVG_KEY[m]] > cols[k][AVG_KEY[m]]
        ]
        out["better_all_three"] = _claim(
            "better_all_three",
            {"not_better": fails},
            not fails,
            "all 12 better" if not fails else "not better: " + ", ".join(fails),
        )
    # per-circuit statements (win_rate.png)
    qrows = list(per_circ["qalm"].values())

    def win_rate(k, m):
        return metrics.win_rate(qrows, list(per_circ[k].values()), m)

    if qrows and per_circ["guoq_2q"]:
        wr = win_rate("guoq_2q", "fidelity")
        out["abstract_fid_win_or_tie_guoq2q"] = _claim(
            "abstract_fid_win_or_tie_guoq2q",
            wr,
            None,
            f"{wr['win_or_tie']:.1f}% (win {wr['win']:.1f} + tie {wr['tie']:.1f}, "
            f"n={wr['n']})",
        )
    group = ["queso_2q", "queso_total", "voqc", "quartz", "guoq_total"]
    if qrows and all(per_circ[k] for k in group):
        wot = {f"{k} {m}": win_rate(k, m)["win_or_tie"] for k in group for m in METRICS}
        lo = min(wot, key=wot.get)
        out["nearly_every_circuit"] = _claim(
            "nearly_every_circuit", wot, None, f"lowest {wot[lo]:.1f}% ({lo})"
        )
    if qrows and all(per_circ[k] for k in S248_BASELINES):
        win = {k: win_rate(k, "fidelity")["win"] for k in S248_BASELINES}
        lo = min(win, key=win.get)
        out["majority_fid_wins"] = _claim(
            "majority_fid_wins", win, win[lo] > 50, f"lowest {win[lo]:.1f}% ({lo})"
        )
    return [out.get(c) or _claim(c) for c in CLAIMS if not c.startswith("s26_")]


def headline_s248(rows, sel):
    keys = [f"{cfg}_S248_{b}" for cfg in CONFIGS for b in (3600, 60)]
    keys += [f"S248_{k}" for k in S248_BASELINES]
    common, absent = _common(sel, keys)
    rows_h = _restrict(rows, ("S248",), common)
    out = {"n_common": len(common), "absent": absent, "configs": {}}
    keep = set(common)
    for cfg in CONFIGS:
        data = tables.averages_table_data(rows_h, cfg, allow_partial=True)
        keymap = {"qalm": f"{cfg}_S248_3600", "qalm_1min": f"{cfg}_S248_60"}
        keymap.update({k: f"S248_{k}" for k in S248_BASELINES})
        per_circ = {
            k: {c: r for c, r in sel[s].items() if c in keep} for k, s in keymap.items()
        }
        win = {}
        if per_circ["qalm"]:
            win = {
                k: {
                    m: metrics.win_rate(
                        list(per_circ["qalm"].values()), list(per_circ[k].values()), m
                    )
                    for m in METRICS
                }
                for k in S248_BASELINES
                if per_circ[k]
            }
        out["configs"][cfg] = {
            "columns": data["columns"],
            "win_rate": win,
            "claims": s248_claims(data["columns"], per_circ),
        }
    return out


def headline_s26(rows, sel):
    """tab:eval_end_to_end statements under a and b (circuits common to all)."""
    keys = [f"{cfg}_S26_3600" for cfg in CONFIGS] + [f"S26_{k}" for k in S26_BASELINES]
    common, absent = _common(sel, keys)
    rows_h = _restrict(rows, ("S26", "S26PP"), common)
    out = {"n_common": len(common), "absent": absent, "configs": {}}
    for cfg in CONFIGS:
        data = tables.eval_end_to_end_data(rows_h, cfg, allow_partial=True)
        cols = data["columns"]
        claims = {}
        if cols["qalm"]["n"] and all(cols[k]["n"] for k in S26_BASELINES):
            ex = [
                c
                for c in data["circuits"]
                if cols["qalm"]["opt"][c] is not None and "qalm" not in data["best"][c]
            ]
            claims["s26_lowest_all_but_one"] = _claim(
                "s26_lowest_all_but_one",
                {"exceptions": ex},
                len(ex) <= 1,
                f"{len(ex)}" + (f": {', '.join(ex)}" if ex else ""),
            )
        q = cols["qalm"]["geomean_red"]
        for k in S26_BASELINES:
            g = cols[k]["geomean_red"]
            if q is not None and g is not None:
                cid = f"s26_geomean_vs_{k}"
                claims[cid] = _claim(cid, {"diff_pp": q - g}, q > g, f"{_fp(q - g)} pp")
        out["configs"][cfg] = {
            "geomean_red": {k: v["geomean_red"] for k, v in cols.items()},
            "n": {k: v["n"] for k, v in cols.items()},
            "best": data["best"],
            "qalm_vs": data["qalm_vs"],
            "claims": [
                claims.get(c) or _claim(c) for c in CLAIMS if c.startswith("s26_")
            ],
        }
    return out


# ---------- noise yardstick (E5) ----------


def noise_yardstick(sel):
    """Run-to-run variation of config a on S26 at 3600 s across seeds 0-2."""
    by_seed = {s: sel[f"e5_seed{s}"] for s in E5_SEEDS if sel[f"e5_seed{s}"]}
    common = (
        sorted(set.intersection(*(set(v) for v in by_seed.values()))) if by_seed else []
    )
    out = {
        "suite": "S26",
        "budget_s": 3600,
        "config": "a",
        "seeds_present": sorted(by_seed),
        "n_circuits": len(common),
    }
    if not common:
        return out
    out["per_seed"] = {
        str(s): metrics.averages([v[c] for c in common]) for s, v in by_seed.items()
    }
    out["non_ok"] = {
        str(s): fig_percircuit.status_counts(v[c] for c in common)
        for s, v in by_seed.items()
    }
    sd, rng = {}, {}
    for m in METRICS:
        xs = [out["per_seed"][str(s)][AVG_KEY[m]] for s in by_seed]
        sd[m] = statistics.stdev(xs) if len(xs) > 1 else None
        rng[m] = max(xs) - min(xs) if len(xs) > 1 else None
    out["sd_of_mean"] = sd
    out["range_of_mean"] = rng
    if len(by_seed) > 1:
        per_c = [
            statistics.stdev(
                fig_percircuit.value(v[c], "total") for v in by_seed.values()
            )
            for c in common
        ]
        out["per_circuit_sd_total_pp"] = {
            "mean": statistics.mean(per_c),
            "median": statistics.median(per_c),
            "max": max(per_c),
            "n_nonzero": sum(x > 0 for x in per_c),
        }
        out["seed_pairs"] = {}
        for s, t in combinations(sorted(by_seed), 2):
            out["seed_pairs"][f"{s}-{t}"] = pair = {}
            for m in ("cx", "total"):
                cnt = Counter(
                    metrics.compare(by_seed[s][c], by_seed[t][c], m) for c in common
                )
                pair[m] = {
                    "first_better": cnt[1],
                    "tie": cnt[0],
                    "second_better": cnt[-1],
                }
    return out


def relative_to_noise(sections, noise):
    """Each a - b difference of means divided by the yardstick SD."""
    sd = noise.get("sd_of_mean") or {}
    out = {}
    for key, sec in sections.items():
        if sec is None or not sec.get("metrics"):
            continue
        out[key] = {}
        for m in METRICS:
            diff = (
                sec["metrics"][m]["diff"]["mean"] if sec["metrics"][m]["diff"] else None
            )
            s = sd.get(m)
            out[key][m] = {
                "diff_of_means": diff,
                "sd": s,
                "ratio": diff / s if diff is not None and s else None,
            }
    return out


# ---------- driver ----------


def build(rows, allow_partial=False):
    """All numbers of the report (raises SystemExit if incomplete and strict)."""
    sel, s26 = gather(rows)
    problems = sel.incomplete()
    if problems and not allow_partial:
        raise SystemExit(
            "incomplete data (use --allow-partial to report on what is present):\n  "
            + "\n  ".join(problems)
        )
    res = {
        "qalm_configs": {
            cfg: {
                "optimizer": tables.QALM_CONFIGS[cfg][0],
                "params": tables.QALM_CONFIGS[cfg][1],
                "seed": 0,
            }
            for cfg in CONFIGS
        },
        "allow_partial": allow_partial,
        "incomplete": problems,
        "coverage": sel.coverage(),
        "s26_section": s26,
        "sections": {},
    }
    for key, suite, _, _ in SECTIONS:
        if key == OPTIONAL_SECTION and not s26:
            res["sections"][key] = None
            continue
        res["sections"][key] = compare_configs(sel[f"a_{key}"], sel[f"b_{key}"], suite)
    res["headline_S248"] = headline_s248(rows, sel)
    res["headline_S26"] = headline_s26(rows, sel)
    res["noise"] = noise_yardstick(sel)
    res["relative_to_noise"] = relative_to_noise(res["sections"], res["noise"])
    res["summary"] = summary_paragraph(res)
    return res


# ---------- report ----------


def _f(x, digits):
    return "n/a" if x is None else f"{x:.{digits}f}"


def _p(p):
    return "n/a" if p is None else f"{p:.3g}"


DIGITS = {"fidelity": 5, "cx": 3, "total": 3}


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def _status(s):
    return "" if s == "OK" else f" ({s})"


def section_md(title, sec):
    L = [f"## {title}", ""]
    if sec is None:
        return L + ["Config b has no row on this suite; section skipped.", ""]
    L.append(
        f"Coverage: a {sec['n_a']}, b {sec['n_b']}, paired {sec['n_paired']} circuits."
        + (f" Only a: {len(sec['only_a'])}." if sec["only_a"] else "")
        + (f" Only b: {len(sec['only_b'])}." if sec["only_b"] else "")
    )
    L.append(
        f"Non-OK runs among the paired circuits: a {sec['non_ok_a'] or 'none'}, "
        f"b {sec['non_ok_b'] or 'none'}."
    )
    L.append("")
    if not sec["metrics"]:
        return L + ["No circuit has both configurations yet.", ""]
    rows = []
    for m in METRICS:
        r = sec["metrics"][m]
        d, k = r["diff"], DIGITS[m]
        w = r["wilcoxon"]
        rows.append(
            [
                METRIC_LABEL[m],
                r["n"],
                _f(r["mean_a"], k),
                _f(r["mean_b"], k),
                _fp(d["mean"], k),
                _fp(d["median"], k),
                f"[{_fp(d['q1'], k)}, {_fp(d['q3'], k)}]",
                _f(d["iqr"], k),
                f"{r['a_better']} / {r['tie']} / {r['b_better']}",
                _f(w["statistic"], 1),
                _p(w["pvalue"]),
            ]
        )
    L += _table(
        [
            "Metric",
            "n",
            "mean a",
            "mean b",
            "mean diff",
            "median diff",
            "[Q1, Q3] diff",
            "IQR",
            "a better / tie / b better",
            "Wilcoxon W (smaller rank sum)",
            "p",
        ],
        rows,
    )
    dl = sec["metrics"]["fidelity"]["diff_lnF"]
    L += [
        "",
        f"Fidelity in log space (the Wilcoxon input): mean diff ln F "
        f"{_fp(dl['mean'], 4)}, median {_fp(dl['median'], 4)}, IQR {_f(dl['iqr'], 4)}.",
    ]
    if "geomean_red" in sec:
        g = sec["geomean_red"]
        L.append(
            f"Geomean total reduction (tab:eval_end_to_end metric): a {g['a']:.3f}%, "
            f"b {g['b']:.3f}% (a - b {g['a'] - g['b']:+.3f} pp)."
        )
    notes = [
        f"{METRIC_LABEL[m]}: {sec['metrics'][m]['wilcoxon']['note']}"
        for m in METRICS
        if sec["metrics"][m]["wilcoxon"]["note"]
    ]
    if notes:
        L.append("Wilcoxon notes: " + "; ".join(notes) + ".")
    L.append("")
    for m in METRICS:
        r = sec["metrics"][m]
        k = DIGITS[m]
        L += [f"Largest differences, {METRIC_LABEL[m]} (top {TOP_N} each way):", ""]
        rows = []
        for side, lst in (
            ("a better", r["largest_a_better"]),
            ("b better", r["largest_b_better"]),
        ):
            for e in lst:
                rows.append(
                    [
                        side,
                        f"`{e['circuit']}`",
                        _fp(e["diff"], k),
                        f"{e['orig_total']} / {e['orig_cx']}",
                        f"{e['a']['total']} / {e['a']['cx']}{_status(e['a']['status'])}",
                        f"{e['b']['total']} / {e['b']['cx']}{_status(e['b']['status'])}",
                    ]
                )
        if rows:
            L += _table(
                [
                    "",
                    "circuit",
                    "diff (a - b)",
                    "orig total / CX",
                    "a total / CX",
                    "b total / CX",
                ],
                rows,
            )
        else:
            L.append("No circuit differs.")
        L.append("")
    return L


def _holds(h):
    return {True: "yes", False: "**no**", None: "-"}[h]


def headline_md(res):
    L = ["## 4. Against the baselines (headline statements)", ""]
    h = res["headline_S248"]
    L.append(
        f"S248, baselines at 3600 s, seed 0. Circuits common to every column with "
        f"data: {h['n_common']}."
        + (f" Columns without data: {', '.join(h['absent'])}." if h["absent"] else "")
    )
    L.append("")
    rows = []
    seen = set()
    for cfg in CONFIGS:
        for k, c in h["configs"][cfg]["columns"].items():
            name = (
                f"QALM {cfg}" + (" (1 min)" if k == "qalm_1min" else " (1 h)")
                if k.startswith("qalm")
                else k
            )
            if name in seen:
                continue
            seen.add(name)
            rows.append(
                [
                    name,
                    c["n"],
                    _f(c["fidelity"], 5),
                    _f(c["cx_red"], 3),
                    _f(c["total_red"], 3),
                    c["status_counts"] or "",
                ]
            )
    L += _table(
        ["Column", "n", "Fidelity", "CX red. (%)", "Total red. (%)", "non-OK"], rows
    )
    L += [
        "",
        "Per-circuit win rates (QALM vs baseline, `metrics.win_rate`; win + tie = win or tie, %):",
        "",
    ]
    rows = []
    for k in S248_BASELINES:
        for m in METRICS:
            cells = [k, m]
            for cfg in CONFIGS:
                w = h["configs"][cfg]["win_rate"].get(k, {}).get(m)
                cells.append(
                    "n/a"
                    if w is None
                    else f"{w['win']:.1f} + {w['tie']:.1f} = {w['win_or_tie']:.1f}"
                )
            rows.append(cells)
    L += _table(["baseline", "metric", "config a", "config b"], rows)
    L += [
        "",
        "Statements (holds: crisp condition met; '-' = qualitative or no data):",
        "",
    ]
    L += _claims_table(h)
    if res["headline_S26"]:
        h26 = res["headline_S26"]
        L += [
            "",
            f"S26 table (tab:eval_end_to_end), 3600 s. Circuits common to every column with data: "
            f"{h26['n_common']}."
            + (
                f" Columns without data: {', '.join(h26['absent'])}."
                if h26["absent"]
                else ""
            ),
            "",
        ]
        rows = [
            [k, h26["configs"]["a"]["n"][k]]
            + [_f(h26["configs"][cfg]["geomean_red"][k], 3) for cfg in CONFIGS]
            for k in h26["configs"]["a"]["geomean_red"]
        ]
        L += _table(
            ["Column", "n", "geomean red. (%), table with a", "table with b"], rows
        )
        L.append("")
        L += _claims_table(h26)
    L.append("")
    return L


def _claims_table(h):
    a, b = (h["configs"][cfg]["claims"] for cfg in CONFIGS)
    rows = [
        [
            ca["id"],
            ca["statement"],
            ca["paper_location"],
            ca["paper_value"],
            ca["shown"],
            _holds(ca["holds"]),
            cb["shown"],
            _holds(cb["holds"]),
        ]
        for ca, cb in zip(a, b)
    ]
    return _table(
        [
            "id",
            "statement",
            "paper location",
            "paper value",
            "a",
            "holds (a)",
            "b",
            "holds (b)",
        ],
        rows,
    )


def noise_md(res):
    n = res["noise"]
    L = ["## 5. Noise yardstick (E5: config a, S26, 3600 s, seeds 0-2)", ""]
    L.append(
        f"Seeds present: {n['seeds_present']}; circuits common to them: {n['n_circuits']}."
    )
    if not n["n_circuits"]:
        return L + ["", "No E5 data.", ""]
    L.append("")
    rows = [
        [
            s,
            _f(v["fidelity"], 5),
            _f(v["cx_red"], 3),
            _f(v["total_red"], 3),
            n["non_ok"][s] or "",
        ]
        for s, v in n["per_seed"].items()
    ]
    rows.append(
        ["SD across seeds", *(_f(n["sd_of_mean"][m], DIGITS[m]) for m in METRICS), ""]
    )
    rows.append(["range", *(_f(n["range_of_mean"][m], DIGITS[m]) for m in METRICS), ""])
    L += _table(["seed", "Fidelity", "CX red. (%)", "Total red. (%)", "non-OK"], rows)
    if "per_circuit_sd_total_pp" in n:
        p = n["per_circuit_sd_total_pp"]
        L += [
            "",
            f"Per-circuit SD of total reduction across seeds: mean {p['mean']:.3f} pp, "
            f"median {p['median']:.3f} pp, max {p['max']:.3f} pp; "
            f"{p['n_nonzero']} of {n['n_circuits']} circuits vary.",
            "Seed vs seed, per circuit (first better / tie / second better): "
            + "; ".join(
                f"seeds {k}: total {v['total']['first_better']}/{v['total']['tie']}/"
                f"{v['total']['second_better']}, CX {v['cx']['first_better']}/"
                f"{v['cx']['tie']}/{v['cx']['second_better']}"
                for k, v in n["seed_pairs"].items()
            )
            + ".",
        ]
    rel = res["relative_to_noise"]
    if rel:
        L += ["", "a - b difference of means relative to the SD across seeds:", ""]
        rows = [
            [
                key,
                METRIC_LABEL[m],
                _fp(v["diff_of_means"], DIGITS[m]),
                _f(v["sd"], DIGITS[m]),
                _f(v["ratio"], 2),
            ]
            for key, sec in rel.items()
            for m, v in sec.items()
        ]
        L += _table(["section", "metric", "a - b", "SD (E5)", "ratio"], rows)
    L.append("")
    return L


def summary_paragraph(res):
    """Neutral facts: differences, size vs the yardstick, differing statements."""
    parts = []
    if res["incomplete"]:
        parts.append(
            f"PARTIAL DATA: {len(res['incomplete'])} selections do not cover their "
            "suite (listed at the top of the report); every number covers only the "
            "circuits present."
        )
    names = {k: f"{suite} at {budget} s" for k, suite, budget, _ in SECTIONS}
    for key, sec in res["sections"].items():
        if not sec or not sec.get("metrics"):
            continue
        m = sec["metrics"]
        t = m["total"]
        parts.append(
            f"On {names[key]} ({sec['n_paired']} paired circuits), a minus b is "
            f"{m['fidelity']['diff']['mean']:+.4f} in mean fidelity, "
            f"{m['cx']['diff']['mean']:+.2f} pp in mean CX reduction and "
            f"{t['diff']['mean']:+.2f} pp in mean total reduction; by total gate "
            f"count a is better on {t['a_better']}, equal on {t['tie']} and worse "
            f"on {t['b_better']} circuits (Wilcoxon p = {_p(t['wilcoxon']['pvalue'])}; "
            f"CX: {m['cx']['a_better']}/{m['cx']['tie']}/{m['cx']['b_better']}, "
            f"p = {_p(m['cx']['wilcoxon']['pvalue'])})."
        )
    if not parts:
        parts.append("No circuit has rows for both configurations yet.")
    n = res["noise"]
    sd = (n.get("sd_of_mean") or {}).get("total")
    if sd:
        ratios = [
            f"{abs(v['total']['ratio']):.1f} ({names[k]})"
            for k, v in res["relative_to_noise"].items()
            if v["total"]["ratio"] is not None
        ]
        parts.append(
            f"The run-to-run SD of config a's {n['n_circuits']}-circuit mean total "
            f"reduction across seeds {n['seeds_present']} is {sd:.2f} pp (range "
            f"{n['range_of_mean']['total']:.2f} pp), measured on S26 at 3600 s; the "
            "a - b differences in mean total reduction are "
            + (", ".join(ratios) + " times" if ratios else "not available relative to")
            + " that SD in absolute value (Section 5 gives the same ratios for "
            "fidelity and CX reduction)."
        )
    else:
        parts.append(
            "The seed-to-seed SD is not available (fewer than two E5 seeds with data)."
        )
    diff, same = [], 0
    for h in (res["headline_S248"], res["headline_S26"]):
        if not h:
            continue
        for ca, cb in zip(*(h["configs"][cfg]["claims"] for cfg in CONFIGS)):
            if ca["holds"] is None or cb["holds"] is None:
                continue
            if ca["holds"] != cb["holds"]:
                diff.append(f"{ca['id']} ({'a' if ca['holds'] else 'b'} only)")
            else:
                same += 1
    if diff or same:
        parts.append(
            f"Of the headline statements with a crisp condition and data, {same} have "
            "the same outcome under both configurations"
            + (
                f" and {len(diff)} {'differs' if len(diff) == 1 else 'differ'}: "
                f"{', '.join(diff)}."
                if diff
                else "; none differ."
            )
        )
    else:
        parts.append("No headline statement could be evaluated (missing columns).")
    return " ".join(parts)


def render(res, source):
    L = [
        "# QALM configuration a vs b",
        "",
        "Auto-generated by `repro/analysis/compare_qalm_configs.py` — do not edit by hand.",
        "",
        f"Source: `{source['path']}` ({source['n_rows']} result rows, sha256 "
        f"`{source['sha256'][:16]}`).",
        "",
    ]
    if res["incomplete"]:
        L += [
            f"**WARNING: partial data (--allow-partial). {len(res['incomplete'])} "
            "selections do not cover their suite:**",
            "",
            *[f"- {p}" for p in res["incomplete"]],
            "",
        ]
    L += [
        "Configurations (seed 0):",
        "",
        *[f"- **{cfg}**: {CONFIG_DESC[cfg]}" for cfg in CONFIGS],
        "",
        "Conventions: differences are a - b, so a positive difference means a is better "
        "on every metric (higher fidelity, higher reductions; reductions in percentage "
        "points). Means are `metrics.averages` over the paired circuits. Per-circuit "
        "a better / tie / b better uses `metrics.compare` (gate counts as integers; "
        "fidelity in log space, tie if |d ln F| <= 1e-12). Wilcoxon: two-sided "
        "signed-rank test, `scipy.stats.wilcoxon(zero_method='zsplit')` on the "
        "per-circuit differences; for fidelity on the ln F differences (F underflows "
        "to 0 on the largest circuits). Quartiles: numpy linear interpolation; "
        "IQR = Q3 - Q1.",
        "",
    ]
    for i, (key, _, _, title) in enumerate(SECTIONS, 1):
        L += section_md(f"{i}. a vs b: {title}", res["sections"][key])
    L += headline_md(res)
    L += noise_md(res)
    L += ["## 6. Summary", "", res["summary"], ""]
    return "\n".join(L)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="repro/results", help="dir with runs.csv")
    ap.add_argument("--out", default="repro/analysis/out", help="output directory")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="report on the circuits present instead of failing",
    )
    args = ap.parse_args(argv)
    rows = metrics.load_runs(args.results)
    path = os.path.join(args.results, "runs.csv")
    source = {"path": path, "n_rows": len(rows), "sha256": _sha256(path)}
    res = build(rows, args.allow_partial)
    res["source"] = source
    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, OUT_MD)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render(res, source))
    js = os.path.join(args.out, OUT_JSON)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    print(res["summary"])
    print(f"wrote {md}\nwrote {js}")
    return res


if __name__ == "__main__":
    main()
