"""Every data-dependent number of the paper text, as LaTeX macros.

Inputs
    --results DIR   written by ``repro/harness/run.py collect``: runs.csv,
                    timeseries.csv.gz and kevents.csv.
    --events FILE   optional: the scheduler's ``<runs_dir>/events.csv``
                    (concurrency, memory-cap escalations, preemptions, infra
                    re-runs), or the copy ``run.py collect`` writes next to
                    runs.csv.  Only events of the plan's jobs count (``--plan``,
                    else the jobs in runs.csv), so jobs of other plan versions
                    in the same runs dir are left out.  Without it those macros
                    print "??".
    repro/inputs/{S248,S26,S26PP}.csv     suites (circuit counts, gate counts)
    repro/config/campaign.yaml            memory cap tiers and job cores

Outputs (in --out, default repro/analysis/out)
    numbers.tex     one ``\\newcommand{\\num<Name>}{<text>}`` per number the
                    compiled paper prints (abstract, introduction, evaluation,
                    captions), named in CamelCase letters only.  Each expands
                    to the number exactly as printed: fixed decimals as in
                    the paper, percentages without the % sign, a
                    minus as ``\\ensuremath{-}`` (works in text and math).
                    Use ``\\numX{}`` before a space.  A few macros expand to
                    data-derived text (a baseline name, a circuit list, a
                    small count as a word).
    numbers.json    {"meta", "selections", "macros", "problems", "warnings"}:
                    per macro its unrounded value, printed text, formula,
                    inputs (selection keys), n and source function.

Numbers come from the existing generators, never recomputed here:
tables.averages_table_data / eval_end_to_end_data (both tables),
fig_percircuit.gather (win rates), fig_curves (curve finals, time to reach
the threshold, greedy phase, E5 seeds); metrics.py holds every definition.
This module only derives differences (always from unrounded values), counts,
and noise estimates.

QALM is configuration "a" or "b" (``--qalm-config``, see tables.py); the
curve experiments (E3-E5) always use optimizer ``qalm``.  The time-to-X
threshold is ``--threshold`` (default 32 %, as in the paper).

Run-to-run noise: E5 runs every N_pool/N_branch configuration with seeds
{0,1,2}.  The pooled within-configuration SD of the 26-circuit mean final
reduction (\\numSeedSdPooled) and Tukey's HSD for five such means
(studentized range, 95 %; \\numSeedHsd) are the noise reference of the
ablations.

Completeness: every selection must hold all circuits of its suite (and all
seeds), the curve data must be consistent (fig_curves checks) and the
threshold must be reached by the greedy curves; otherwise the script exits
with the full list of problems.  With --allow-partial it continues; a macro
whose inputs are incomplete is marked provisional and printed with a
trailing "?" (a missing value prints "??").

``--budget`` / ``--short-budget`` (default 3600 / 60) select the runs used
as "one hour" and "one minute"; other values are only for smoke tests.

CLI (from the repo root):
    uv run repro/analysis/numbers.py --results repro/results \\
        --out repro/analysis/out [--qalm-config a|b] [--threshold 32] \\
        [--events repro/runs/events.csv] [--plan repro/runs/plan.jsonl] \\
        [--allow-partial]

Note: this file shares its name with the standard library module ``numbers``.
Every script in repro/analysis has this directory on sys.path[0], so an
``import numbers`` (by statistics, numpy, ...) finds this file first.  When
imported under that name, the file loads the real module, puts it into
sys.modules (the import system then returns it) and imports nothing else.
Run it as a script; to call it from Python, load it under another name, e.g.
``importlib.util.spec_from_file_location("paper_numbers", path)``.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)

if __name__ == "numbers":  # shadowing the standard library: hand over to it
    import importlib.machinery
    import importlib.util

    _spec = importlib.machinery.PathFinder.find_spec(
        "numbers", [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]
    )
    _real = importlib.util.module_from_spec(_spec)
    sys.modules["numbers"] = _real
    _spec.loader.exec_module(_real)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from collections import Counter  # noqa: E402

if __name__ != "numbers":
    sys.path.insert(0, HERE)
    import fig_curves  # noqa: E402
    import fig_percircuit  # noqa: E402
    import metrics  # noqa: E402
    import tables  # noqa: E402
    import yaml  # noqa: E402
    from scipy import stats  # noqa: E402

AUTOGEN = "% Auto-generated by repro/analysis/numbers.py -- do not edit by hand."
MISSING_TEXT = "??"
PROVISIONAL_MARK = "?"
NAME_RE = re.compile(r"^num[A-Z][A-Za-z]*$")
GIB = 2**30
ALPHA = 0.05
WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
DIGIT = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

# tables.py column key -> macro suffix
AVG_COLS = {
    "qalm": "Qalm",
    "qalm_1min": "QalmOneMin",
    "guoq_2q": "GuoqTwoQ",
    "guoq_total": "GuoqTotal",
    "queso_2q": "QuesoTwoQ",
    "queso_total": "QuesoTotal",
    "voqc": "Voqc",
    "quartz": "Quartz",
}
BASELINES = ["guoq_2q", "guoq_total", "queso_2q", "queso_total", "voqc", "quartz"]
# Intro: "Compared with QUESO (both modes), VOQC, Quartz, and GUOQ (total mode)"
NEARLY_EVERY = ["queso_2q", "queso_total", "voqc", "quartz", "guoq_total"]
ETE_COLS = {
    "voqc": "Voqc",
    "queso": "Queso",
    "guoq": "Guoq",
    "quartz_nopp": "QuartzNoPP",
    "quartz_pp": "QuartzPP",
    "qalm": "Qalm",
}
# metrics.averages key -> (macro stem, decimals, description)
AVG_METRICS = {
    "fidelity": ("Fid", 4, "mean fidelity 0.999^(total-cx)*0.99^cx"),
    "cx_red": ("CxRed", 2, "mean 2-qubit (CX) gate reduction % (circuits with cx>0)"),
    "total_red": ("TotalRed", 2, "mean total gate reduction %"),
}
WIN_METRICS = {"fidelity": "Fid", "cx": "Cx", "total": "Total"}



# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def fmt_fixed(value, decimals):
    """Fixed-point text; never "-0.00"; a minus sign that works in text and math."""
    s = f"{value:.{decimals}f}"
    if float(s) == 0:
        s = f"{0:.{decimals}f}"
    if s.startswith("-"):
        s = r"\ensuremath{-}" + s[1:]
    return s


def time_decimals(value):
    """Seconds and speedups as the paper prints them: 5.5 s, 82 s, 11x."""
    return 1 if round(value, 1) < 10 else 0


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def _sha(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def _file_sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# the book of macros and selections
# ---------------------------------------------------------------------------


class Book:
    def __init__(self, allow_partial):
        self.allow_partial = allow_partial
        self.selections = {}
        self.macros = {}
        self.warnings = []
        self.errors = []
        self.section = ""

    def warn(self, msg):
        self.warnings.append(msg)
        print("WARNING:", msg, file=sys.stderr)

    def error(self, msg):
        """A completeness/consistency problem: fatal unless --allow-partial."""
        self.errors.append(msg)
        if self.allow_partial:
            self.warn(msg)

    def selection(
        self, key, description, job_ids, n, expected, missing=(), required=True, **extra
    ):
        """Record one input selection; an incomplete required one is an error."""
        complete = n == expected and not extra.get("problems")
        self.selections[key] = {
            "description": description,
            "n": n,
            "expected": expected,
            "complete": complete,
            "job_ids_sha256": _sha(job_ids) if job_ids else None,
            "missing": sorted(missing)[:30],
            **extra,
        }
        if not complete:
            what = (
                f"{n} of {expected}"
                if n != expected
                else "; ".join(extra["problems"][:2])
            )
            msg = f"selection {key} ({description}) is incomplete: {what}"
            (self.error if required else self.warn)(msg)
        return complete

    def complete(self, inputs):
        return all(self.selections[i]["complete"] for i in inputs)

    def add(
        self,
        name,
        value,
        *,
        decimals=None,
        formula,
        inputs,
        source,
        kind="number",
        text=None,
        n=None,
    ):
        """Register one macro.  kind: number (fixed decimals), int, time, text."""
        if not NAME_RE.match(name):
            raise ValueError(f"bad macro name {name!r}")
        if name in self.macros:
            raise ValueError(f"macro {name} defined twice")
        for i in inputs:
            if i not in self.selections:
                raise KeyError(f"{name}: unknown selection {i}")
        if value is None and text is None:
            printed = MISSING_TEXT
        elif text is not None:
            printed = text
        elif kind == "int":
            printed = str(round(value))  # the json keeps the unrounded value
        elif kind == "time":
            decimals = time_decimals(value)
            printed = fmt_fixed(value, decimals)
        else:
            printed = fmt_fixed(value, decimals)
        provisional = not self.complete(inputs) or (value is None and text is None)
        if provisional and printed != MISSING_TEXT:
            printed += PROVISIONAL_MARK
        check_tex(name, printed)
        if n is None and inputs:
            ns = [self.selections[i]["n"] for i in inputs]
            n = ns[0] if len(set(ns)) == 1 else ns
        self.macros[name] = {
            "section": self.section,
            "value": value,
            "printed": printed,
            "decimals": decimals,
            "kind": kind,
            "formula": formula,
            "inputs": list(inputs),
            "n": n,
            "source": source,
            "provisional": provisional,
        }


def check_tex(name, text):
    """A macro body must be safe inside \\newcommand: balanced braces, no specials."""
    depth = 0
    for i, ch in enumerate(text):
        escaped = i > 0 and text[i - 1] == "\\"
        if ch in "%#&$~" and not escaped:
            raise ValueError(f"{name}: unescaped {ch!r} in {text!r}")
        if ch == "{" and not escaped:
            depth += 1
        elif ch == "}" and not escaped:
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        raise ValueError(f"{name}: unbalanced braces in {text!r}")


def _all_known(*xs):
    return all(x is not None for x in xs)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def dedupe(rows, book):
    """Drop exact duplicate runs.csv rows; two different rows for one job are an error."""
    by_id, n_dup = {}, 0
    for r in rows:
        prev = by_id.setdefault(r["job_id"], r)
        if prev is r:
            continue
        if prev != r:
            raise ValueError(f"runs.csv: job {r['job_id']} has two different rows")
        n_dup += 1
    if n_dup:
        book.warn(f"runs.csv: ignored {n_dup} exact duplicate rows")
    return list(by_id.values())


def budget_view(rows, budget, short_budget):
    """Rows as tables.py / fig_percircuit.py select them (3600 s and 60 s).

    The default budgets return the rows unchanged; otherwise (smoke tests) the
    rows with ``budget`` stand in for the one-hour runs and the rows with
    ``short_budget`` for the one-minute runs.
    """
    if (budget, short_budget) == (3600, 60):
        return rows
    out = []
    for r in rows:
        if r["budget_s"] == budget:
            out.append(dict(r, budget_s=3600))
        if r["budget_s"] == short_budget:
            out.append(dict(r, budget_s=60))
    return out


def _sel_desc(col):
    params = ",".join(f"{k}={v}" for k, v in sorted(col.params.items()))
    return (
        f"{col.optimizer}{'(' + params + ')' if params else ''} {col.suite} "
        f"{col.budget_s}s seed {tables.SEED}"
    )


# ---------------------------------------------------------------------------
# S248: averages table, intro margins, one-minute run, win rates
# ---------------------------------------------------------------------------


def s248_numbers(book, rows, qcfg, budget, short_budget):
    book.section = "S248 averages (tab:avg-comparison)"
    avg = tables.averages_table_data(rows, qcfg, allow_partial=True)
    cols = avg["columns"]
    n_circ = avg["n_circuits"]
    sel_rows = {}
    for col in tables.averages_columns(qcfg):
        by_c = tables.select_column(rows, col, allow_partial=True)
        sel_rows[col.key] = by_c
        manifest = tables.load_manifest(col.suite)
        book.selection(
            f"avg:{col.key}",
            _sel_desc(col),
            [r["job_id"] for r in by_c.values()],
            len(by_c),
            n_circ,
            missing=set(manifest) - set(by_c),
            status_counts=cols[col.key]["status_counts"],
        )
    for metric, (stem, dec, desc) in AVG_METRICS.items():
        for key, suffix in AVG_COLS.items():
            book.add(
                f"num{stem}{suffix}",
                cols[key][metric],
                decimals=dec,
                formula=f"{desc} over the circuits of column {key} (metrics.averages)",
                inputs=[f"avg:{key}"],
                source="tables.averages_table_data",
            )
    for key, suffix in AVG_COLS.items():
        book.add(
            f"numFail{suffix}",
            cols[key]["non_ok"] if cols[key]["n"] else None,
            kind="int",
            formula=f"runs of column {key} whose status is not OK "
            f"({cols[key]['status_counts']})",
            inputs=[f"avg:{key}"],
            source="tables.averages_table_data (non_ok)",
        )

    def v(key, metric):
        return cols[key][metric]

    base_tot = {k: v(k, "total_red") for k in BASELINES}
    all_base = [f"avg:{k}" for k in BASELINES]
    strongest = None
    if all(x is not None for x in base_tot.values()):
        strongest = max(base_tot, key=base_tot.get)
    labels = {c.key: c.label for c in tables.averages_columns(qcfg)}
    margin = {}
    for key, suffix in (("qalm", "OneHour"), ("qalm_1min", "OneMin")):
        margin[key] = (
            v(key, "total_red") - base_tot[strongest]
            if strongest and v(key, "total_red") is not None
            else None
        )
        book.add(
            f"numTotalRedMargin{suffix}",
            margin[key],
            decimals=2,
            formula=f"total_red({key}) - max over the six 1-h baselines of total_red "
            "(unrounded; the baseline is \\numStrongestTotalRedBaseline)",
            inputs=[f"avg:{key}"] + all_base,
            source="tables.averages_table_data",
        )
    book.add(
        "numStrongestTotalRedBaseline",
        None,
        kind="text",
        text=labels[strongest] if strongest else None,
        formula="baseline column with the highest mean total gate reduction",
        inputs=all_base,
        source="tables.averages_table_data",
    )
    book.section = "S248 one-minute QALM"
    ones = ["avg:qalm", "avg:qalm_1min"]
    f1h, f1m = v("qalm", "fidelity"), v("qalm_1min", "fidelity")
    drops = {
        "fid_abs": f1h - f1m if _all_known(f1h, f1m) else None,
        "fid_rel_pct": 100 * (f1h - f1m) / f1h if _all_known(f1h, f1m) else None,
        "cx_red_pp": (
            v("qalm", "cx_red") - v("qalm_1min", "cx_red")
            if _all_known(v("qalm", "cx_red"), v("qalm_1min", "cx_red"))
            else None
        ),
        "total_red_pp": (
            v("qalm", "total_red") - v("qalm_1min", "total_red")
            if _all_known(v("qalm", "total_red"), v("qalm_1min", "total_red"))
            else None
        ),
    }
    for name, key, dec, formula in (
        ("numOneMinFidDropRel", "fid_rel_pct", 1, "100*(F(1 h) - F(1 min)) / F(1 h)"),
        ("numOneMinFidDropAbs", "fid_abs", 4, "F(1 h) - F(1 min)"),
        ("numOneMinCxRedDrop", "cx_red_pp", 2, "cx_red(1 h) - cx_red(1 min), points"),
        (
            "numOneMinTotalRedDrop",
            "total_red_pp",
            2,
            "total_red(1 h) - total_red(1 min), points",
        ),
    ):
        book.add(
            name,
            drops[key],
            decimals=dec,
            formula=formula + " (QALM, unrounded table means)",
            inputs=ones,
            source="tables.averages_table_data",
        )
    book.add(
        "numBudgetFactor",
        budget / short_budget,
        kind="int",
        formula=f"one-hour budget / one-minute budget ({budget} s / {short_budget} s)",
        inputs=[],
        source="numbers.py --budget / --short-budget",
    )

    # ---- win rates (fig_percircuit.gather, the numbers on win_rate.png) ----
    book.section = "S248 per-circuit win rates (fig:win_rate)"
    try:
        pc, _, _ = fig_percircuit.gather(rows, qcfg)
    except SystemExit as e:
        book.error(f"fig_percircuit.gather: {e}")
        pc = None
    if pc and v("qalm", "fidelity") is not None:
        got = pc["qalm"]["means"]["fidelity"]
        if abs(got - v("qalm", "fidelity")) > 1e-12:
            raise AssertionError(
                f"fig_percircuit and tables select different QALM rows ({got})"
            )
    win = {}
    for b in BASELINES:
        e = pc["baselines"][b] if pc else {"no_data": True}
        win[b] = None if e.get("no_data") else e["win_rate"]
        for metric, stem in WIN_METRICS.items():
            wr = win[b][metric] if win[b] else None
            ins = ["avg:qalm", f"avg:{b}"]
            book.add(
                f"numWin{stem}{AVG_COLS[b]}",
                wr["win_or_tie"] if wr else None,
                decimals=1,
                formula=f"% of circuits where QALM >= {b} on {metric} "
                "(metrics.win_rate win_or_tie; fidelity ties |d ln F| <= 1e-12)",
                inputs=ins,
                source="fig_percircuit.gather",
                n=wr["n"] if wr else 0,
            )
            book.add(
                f"numStrictWin{stem}{AVG_COLS[b]}",
                wr["win"] if wr else None,
                decimals=1,
                formula=f"% of circuits where QALM > {b} on {metric} (metrics.win_rate win)",
                inputs=ins,
                source="fig_percircuit.gather",
                n=wr["n"] if wr else 0,
            )
    have_all = all(win[b] for b in BASELINES)
    pair_ins = ["avg:qalm"] + all_base
    fid_wot = (
        {b: win[b]["fidelity"]["win_or_tie"] for b in BASELINES} if have_all else {}
    )
    strongest_fid = min(fid_wot, key=fid_wot.get) if fid_wot else None
    book.add(
        "numWinFidStrongest",
        fid_wot.get(strongest_fid),
        decimals=1,
        formula="lowest fidelity win-or-tie % over the six baselines "
        "(the baseline is \\numStrongestFidBaseline)",
        inputs=pair_ins,
        source="fig_percircuit.gather",
    )
    book.add(
        "numStrongestFidBaseline",
        None,
        kind="text",
        text=labels[strongest_fid] if strongest_fid else None,
        formula="baseline against which QALM has the lowest fidelity win-or-tie rate",
        inputs=pair_ins,
        source="fig_percircuit.gather",
    )
    others = (
        {(b, m): win[b][m]["win_or_tie"] for b in NEARLY_EVERY for m in WIN_METRICS}
        if have_all
        else {}
    )
    book.add(
        "numWinMinOthers",
        min(others.values()) if others else None,
        decimals=1,
        formula="lowest win-or-tie % over QUESO (both), VOQC, Quartz, GUOQ (Total) "
        "x {fidelity, cx, total}",
        inputs=["avg:qalm"] + [f"avg:{b}" for b in NEARLY_EVERY],
        source="fig_percircuit.gather",
    )
    strict_fid = {b: win[b]["fidelity"]["win"] for b in BASELINES} if have_all else {}
    book.add(
        "numStrictWinFidMin",
        min(strict_fid.values()) if strict_fid else None,
        decimals=1,
        formula="lowest strict fidelity win % over the six baselines",
        inputs=pair_ins,
        source="fig_percircuit.gather",
    )
    # negative total reduction (GUOQ resynthesis)
    neg = {}
    for key in ("guoq_2q", "qalm"):
        rs = list(sel_rows[key].values())
        neg[key] = sum(r["opt_total"] > r["orig_total"] for r in rs) if rs else None
    book.add(
        "numNegTotalGuoqTwoQ",
        neg["guoq_2q"],
        kind="int",
        formula="circuits where GUOQ (2Q Gate) output has more gates than the input",
        inputs=["avg:guoq_2q"],
        source="tables.select_column",
    )
    n_g = len(sel_rows["guoq_2q"])
    book.add(
        "numNegTotalPctGuoqTwoQ",
        100 * neg["guoq_2q"] / n_g if n_g else None,
        decimals=1,
        formula="100 * numNegTotalGuoqTwoQ / n",
        inputs=["avg:guoq_2q"],
        source="tables.select_column",
    )
    book.add(
        "numNegTotalQalm",
        neg["qalm"],
        kind="int",
        formula="circuits where the QALM output has more gates than the input",
        inputs=["avg:qalm"],
        source="tables.select_column",
    )
    return avg


# ---------------------------------------------------------------------------
# S26 table (tab:eval_end_to_end)
# ---------------------------------------------------------------------------


def ete_numbers(book, rows, qcfg):
    book.section = "S26 table (tab:eval_end_to_end)"
    ete = tables.eval_end_to_end_data(rows, qcfg, allow_partial=True)
    cols = ete["columns"]
    n_circ = ete["n_circuits"]
    for col in tables.e2e_columns(qcfg):
        by_c = tables.select_column(rows, col, allow_partial=True)
        book.selection(
            f"ete:{col.key}",
            _sel_desc(col),
            [r["job_id"] for r in by_c.values()],
            len(by_c),
            n_circ,
            missing=set(ete["circuits"]) - set(by_c),
            status_counts=cols[col.key]["status_counts"],
        )
    for key, suffix in ETE_COLS.items():
        book.add(
            f"numGeoRed{suffix}",
            cols[key]["geomean_red"],
            decimals=1,
            formula=f"100*(1 - geomean(opt/orig)) over the 26 circuits, column {key} "
            "(S26 originals; metrics.geomean_reduction_pct)",
            inputs=[f"ete:{key}"],
            source="tables.eval_end_to_end_data",
        )
        book.add(
            f"numFailETE{suffix}",
            cols[key]["non_ok"] if cols[key]["n"] else None,
            kind="int",
            formula=f"runs of column {key} whose status is not OK "
            f"({cols[key]['status_counts']})",
            inputs=[f"ete:{key}"],
            source="tables.eval_end_to_end_data (non_ok)",
        )
    q = cols["qalm"]["geomean_red"]
    deltas = {}
    for key in ("voqc", "quartz_nopp", "queso", "guoq", "quartz_pp"):
        b = cols[key]["geomean_red"]
        deltas[key] = q - b if _all_known(q, b) else None
        book.add(
            f"numGeoDelta{ETE_COLS[key]}",
            deltas[key],
            decimals=1,
            formula=f"geomean_red(qalm) - geomean_red({key}), unrounded, then rounded",
            inputs=["ete:qalm", f"ete:{key}"],
            source="tables.eval_end_to_end_data",
        )
    all_ete = [f"ete:{k}" for k in ETE_COLS]
    has_q = cols["qalm"]["n"] > 0
    n_best = cols["qalm"]["n_best"] if has_q else None
    not_best = [
        c
        for c in ete["circuits"]
        if "qalm" not in ete["best"][c] and cols["qalm"]["opt"][c] is not None
    ]
    book.add(
        "numETEQalmBest",
        n_best,
        kind="int",
        formula="circuits where QALM has the lowest gate count (ties count)",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (n_best)",
    )
    book.add(
        "numETEQalmNotBest",
        len(not_best) if has_q else None,
        kind="int",
        formula="circuits where QALM is not (co-)lowest (26 - numETEQalmBest)",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (best)",
    )
    book.add(
        "numETEQalmNotBestWord",
        len(not_best) if has_q else None,
        kind="text",
        text=word(len(not_best)) if has_q else None,
        formula="numETEQalmNotBest as a word",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (best)",
    )
    for key, suffix in ETE_COLS.items():
        if key == "qalm":
            continue
        vs = ete["qalm_vs"][key]
        both = has_q and cols[key]["n"] > 0
        book.add(
            f"numETEQalmGeq{suffix}",
            vs["better"] + vs["equal"] if both else None,
            kind="int",
            formula=f"circuits where QALM's gate count <= {key}'s",
            inputs=["ete:qalm", f"ete:{key}"],
            source="tables.eval_end_to_end_data (qalm_vs)",
        )
        book.add(
            f"numETEQalmWorse{suffix}",
            vs["worse"] if both else None,
            kind="int",
            formula=f"circuits where QALM's gate count > {key}'s",
            inputs=["ete:qalm", f"ete:{key}"],
            source="tables.eval_end_to_end_data (qalm_vs)",
        )
    vs_pp = ete["qalm_vs"]["quartz_pp"]
    both_pp = has_q and cols["quartz_pp"]["n"] > 0
    book.add(
        "numETEQalmWorseQuartzPPWord",
        vs_pp["worse"] if both_pp else None,
        kind="text",
        text=word(vs_pp["worse"]) if both_pp else None,
        formula="numETEQalmWorseQuartzPP as a word",
        inputs=["ete:qalm", "ete:quartz_pp"],
        source="tables.eval_end_to_end_data (qalm_vs)",
    )
    book.add(
        "numQuartzSuiteCircuits",
        n_circ,
        kind="int",
        formula="circuits in repro/inputs/S26.csv",
        inputs=["manifest:S26"],
        source="tables.eval_end_to_end_data",
    )
    return ete


# ---------------------------------------------------------------------------
# curves: greedy (E3), single k (E4), N_pool / N_branch (E5)
# ---------------------------------------------------------------------------


def curve_numbers(book, results_dir, budget, threshold):
    book.section = "curve selections"
    res = fig_curves.Results(results_dir, budget, strict=False)
    specs = fig_curves.figure_specs(budget)
    figs = {s["key"]: fig_curves.build_figure(res, s) for s in specs}
    thresholds = sorted(set(fig_curves.REPORT_THRESHOLDS) | {threshold})
    fnum = {
        k: fig_curves.figure_numbers(res, f, thresholds, threshold)
        for k, f in figs.items()
        if f is not None
    }
    derived = fig_curves.derived_numbers(fnum)
    n_manifest = len(fig_curves.manifest_circuits())
    # which jobs belong to which figure (to attribute fig_curves' problems)
    fig_jobs = {}
    for spec in specs:
        fig = figs[spec["key"]]
        circuits = fig["circuits"] if fig else []
        fig_jobs[spec["key"]] = set()
        for cs in spec["curves"]:
            ids = []
            for seed in cs["seeds"]:
                by_c = res.unit(cs["params"], seed)
                ids += [by_c[c]["job_id"] for c in circuits if c in by_c]
            fig_jobs[spec["key"]] |= set(ids)
    fig_problems = {k: [] for k in fig_jobs}
    for p in res.problems:
        hit = [k for k in fig_jobs if p.startswith(k + ":") or p.startswith(k + " ")]
        if not hit:
            ids = set(re.findall(r"\b[0-9a-f]{12}\b", p))
            hit = [k for k, jobs in fig_jobs.items() if ids & jobs]
        for k in hit or list(fig_jobs):
            fig_problems[k].append(p)
    for spec in specs:
        key = spec["key"]
        fig = figs[key]
        circuits = fig["circuits"] if fig else []
        for cs in spec["curves"]:
            ids, present = [], []
            for seed in cs["seeds"]:
                by_c = res.unit(cs["params"], seed)
                if by_c:
                    present.append(seed)
                ids += [by_c[c]["job_id"] for c in circuits if c in by_c]
            over = ",".join(f"{k}={v}" for k, v in cs["overrides"].items())
            n_exp = n_manifest * len(cs["seeds"])
            book.selection(
                f"curve:{key}:{cs['key']}",
                f"qalm({over}) S26 {budget}s seeds {list(cs['seeds'])} "
                f"[{spec['file']}]",
                ids,
                len(ids),
                n_exp,
                missing=set(fig_curves.manifest_circuits()) - set(circuits),
                seeds_present=present,
                problems=fig_problems[key][:20],
            )
    # report any curve-level issues once more for the record
    issues = {
        "same_timestamp": res.same_t,
        "greedy_end_s_after_budget_capped": res.greedy_end_capped,
        "duplicate_runs_rows": res.duplicate_rows,
        "problems": res.problems,
    }
    return fnum, derived, issues


def _curve(fnum, fig, curve, field="final_mean_reduction_pct"):
    try:
        return fnum[fig]["curves"][curve][field]
    except KeyError:
        return None


def e5_noise(fnum):
    """Pooled within-configuration SD of the final mean reduction over E5 seeds."""
    per_cfg = {}
    for fig in ("N_pool", "N_branch"):
        for c in fnum.get(fig, {}).get("curves", {}).values():
            key = json.dumps(c["params"], sort_keys=True)
            finals = [
                d["final_mean_reduction_pct"] for _, d in sorted(c["per_seed"].items())
            ]
            per_cfg[key] = finals
    groups = [f for f in per_cfg.values() if len(f) >= 2]
    if not groups:
        return None
    df = sum(len(f) - 1 for f in groups)
    ss = sum(statistics.variance(f) * (len(f) - 1) for f in groups)
    return {
        "sd_pooled": math.sqrt(ss / df),
        "df": df,
        "n_configs": len(groups),
        "finals_by_config": per_cfg,
    }


def hsd(sd, k, df, n):
    """Tukey's honestly significant difference for k means of n runs each."""
    return stats.studentized_range.ppf(1 - ALPHA, k, df) * sd / math.sqrt(n)


def greedy_numbers(book, fnum, derived, threshold):
    book.section = "greedy mode (fig:greedy_k)"
    sel = {gk: f"curve:greedy_k:greedy_k={gk}" for gk in range(4)}
    book.add(
        "numGreedyThreshold",
        threshold,
        kind="text",
        text=f"{threshold:g}",
        formula="--threshold (mean total reduction %, for time-to-X)",
        inputs=[],
        source="numbers.py --threshold",
        n=None,
    )
    finals = {gk: _curve(fnum, "greedy_k", f"greedy_k={gk}") for gk in range(4)}
    times = {
        gk: _curve(fnum, "greedy_k", f"greedy_k={gk}", "time_to_threshold_s")
        for gk in range(4)
    }
    for gk in range(4):
        book.add(
            f"numGreedyFinalGk{DIGIT[gk]}",
            finals[gk],
            decimals=2,
            formula=f"final mean total reduction % (t = budget), greedy_k={gk}, 26 circuits",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (final_mean_reduction_pct)",
        )
    for gk in range(4):
        t = times[gk]
        fig_ok = fnum.get("greedy_k") and f"greedy_k={gk}" in fnum["greedy_k"]["curves"]
        if fig_ok and t is None:
            book.error(
                f"greedy_k={gk}: the mean reduction never reaches {threshold:g}% "
                f"(final {finals[gk]:.3f}%); choose --threshold <= "
                f"{math.floor(min(x for x in finals.values() if x is not None))}"
            )
        book.add(
            f"numGreedyTimeToThresholdGk{DIGIT[gk]}",
            t,
            kind="time",
            formula=f"first t where the mean reduction step function >= {threshold:g}% "
            f"(metrics.time_to_reach), greedy_k={gk}; seconds",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (time_to_threshold_s)",
        )
    speed = derived.get("greedy_k_speedup_vs_greedy_k=0_at_threshold", {})
    for gk in (1, 2, 3):
        book.add(
            f"numGreedySpeedupGk{DIGIT[gk]}",
            speed.get(f"greedy_k={gk}"),
            kind="time",
            formula=f"time_to_threshold(greedy_k=0) / time_to_threshold(greedy_k={gk})",
            inputs=[sel[0], sel[gk]],
            source="fig_curves.derived_numbers",
        )
    diff = finals[2] - finals[0] if _all_known(finals[2], finals[0]) else None
    book.add(
        "numGreedyFinalDiff",
        diff,
        decimals=2,
        formula="final(greedy_k=2) - final(greedy_k=0), points (unrounded)",
        inputs=[sel[0], sel[2]],
        source="fig_curves.figure_numbers",
    )
    phase = {}
    for gk in (1, 2, 3):
        gp = None
        try:
            gp = fnum["greedy_k"]["curves"][f"greedy_k={gk}"]["per_seed"]["0"][
                "greedy_phase"
            ]
        except KeyError:
            pass
        phase[gk] = gp
        book.add(
            f"numGreedyPhaseMeanGk{DIGIT[gk]}",
            gp["mean_end_s"] if gp else None,
            kind="time",
            formula=f"mean over circuits of the greedy-phase end (first k={gk + 1} "
            "event; runs that never left it count as the budget), seconds",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (greedy_phase.mean_end_s)",
        )
        book.add(
            f"numGreedyNotFinishedGk{DIGIT[gk]}",
            gp["n_not_finished"] if gp else None,
            kind="int",
            formula=f"circuits whose greedy phase (greedy_k={gk}) did not end within "
            "the budget",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (greedy_phase.n_not_finished)",
        )
    gp3 = phase[3]
    book.add(
        "numGreedyNotFinishedGkThreeWord",
        gp3["n_not_finished"] if gp3 else None,
        kind="text",
        text=word(gp3["n_not_finished"]) if gp3 else None,
        formula="numGreedyNotFinishedGkThree as a word",
        inputs=[sel[3]],
        source="fig_curves.figure_numbers",
    )
    # in table order (gf2^9_mult before gf2^10_mult)
    order = {c: i for i, c in enumerate(tables.E2E_CIRCUITS)}
    nf_circuits = (
        sorted(gp3["not_finished_circuits"], key=lambda c: (order.get(c, 99), c))
        if gp3
        else None
    )
    book.add(
        "numGreedyNotFinishedGkThreeList",
        None,
        kind="text",
        text=(
            tables._join_and([tables._texttt(c) for c in nf_circuits]) or "none"
            if gp3
            else None
        ),
        formula="the circuits counted in numGreedyNotFinishedGkThree",
        inputs=[sel[3]],
        source="fig_curves.figure_numbers (greedy_phase.not_finished_circuits)",
    )


def single_k_numbers(book, fnum):
    book.section = "fixed exploration depth (fig:single_k)"
    res = {}
    for gk in (0, 2):
        fig = f"single_k_gk{gk}"
        sel = {fk: f"curve:{fig}:k={fk}" for fk in range(1, 6)}
        adv_sel = f"curve:{fig}:advancing_k"
        finals = {fk: _curve(fnum, fig, f"k={fk}") for fk in range(1, 6)}
        adv = _curve(fnum, fig, "advancing_k")
        for fk in range(1, 6):
            book.add(
                f"numFixedKGk{DIGIT[gk]}K{DIGIT[fk]}",
                finals[fk],
                decimals=2,
                formula=f"final mean total reduction %, greedy_k={gk}, fixed k={fk}",
                inputs=[sel[fk]],
                source="fig_curves.figure_numbers (final_mean_reduction_pct)",
            )
        book.add(
            f"numFixedKGk{DIGIT[gk]}Advancing",
            adv,
            decimals=2,
            formula=f"final mean total reduction %, greedy_k={gk}, advancing k "
            "(same runs as the greedy_k figure)",
            inputs=[adv_sel],
            source="fig_curves.figure_numbers",
        )
        known = None not in finals.values()
        best = max(finals, key=finals.get) if known else None
        book.add(
            f"numFixedKGk{DIGIT[gk]}BestK",
            best,
            kind="int",
            formula=f"fixed k with the highest final reduction, greedy_k={gk}",
            inputs=list(sel.values()),
            source="fig_curves.derived_numbers (best_fixed_k)",
        )
        book.add(
            f"numFixedKGk{DIGIT[gk]}Best",
            finals[best] if known else None,
            decimals=2,
            formula=f"final reduction of the best fixed k, greedy_k={gk}",
            inputs=list(sel.values()),
            source="fig_curves.derived_numbers (best_fixed_k_final_pct)",
        )
        gap = adv - finals[best] if known and adv is not None else None
        book.add(
            f"numFixedKGk{DIGIT[gk]}GapToAdvancing",
            gap,
            decimals=2,
            formula=f"advancing-k final - best fixed-k final, greedy_k={gk} (points)",
            inputs=list(sel.values()) + [adv_sel],
            source="fig_curves.derived_numbers (advancing_minus_best_fixed_pct)",
        )
        res[gk] = {
            "finals": finals,
            "adv": adv,
            "best": best,
            "gap": gap,
            "sel": sel,
            "adv_sel": adv_sel,
        }


def e5_numbers(book, fnum, derived, noise):
    book.section = "exploration strategy (fig:exploration_ablation)"
    seeds = list(fig_curves.E5_SEEDS)
    vals = {}
    for fig, stem in (("N_pool", "Npool"), ("N_branch", "Nbranch")):
        vals[fig] = {}
        for i in range(1, 6):
            key = f"{fig}={i}"
            vals[fig][i] = _curve(fnum, fig, key)
            book.add(
                f"num{stem}{DIGIT[i]}",
                vals[fig][i],
                decimals=2,
                formula=f"final mean total reduction %, {key}, mean over seeds {seeds}",
                inputs=[f"curve:{fig}:{key}"],
                source="fig_curves.figure_numbers (final_mean_reduction_pct)",
            )
    pool_sel = [f"curve:N_pool:N_pool={i}" for i in range(1, 6)]
    br_sel = [f"curve:N_branch:N_branch={i}" for i in range(1, 6)]
    e5_sel = pool_sel + br_sel
    dp = derived.get("N_pool", {})
    for name, key in (
        ("numNpoolMin", "mean_final_min_pct"),
        ("numNpoolMax", "mean_final_max_pct"),
        ("numNpoolSpread", "mean_final_spread_pct"),
    ):
        book.add(
            name,
            dp.get(key),
            decimals=2,
            formula=f"{key} over N_pool in 1..5 (seed-averaged finals)",
            inputs=pool_sel,
            source="fig_curves.derived_numbers",
        )
    book.add(
        "numNbranchSpread",
        derived.get("N_branch", {}).get("mean_final_spread_pct"),
        decimals=2,
        formula="max - min over N_branch in 1..5 (seed-averaged finals)",
        inputs=br_sel,
        source="fig_curves.derived_numbers",
    )
    dflt = derived.get("E5_default_config", {})
    dsel = ["curve:N_branch:N_branch=3"]
    book.add(
        "numSeeds",
        len(seeds),
        kind="int",
        formula="seeds per E5 configuration (fig_curves.E5_SEEDS)",
        inputs=[],
        source="fig_curves.E5_SEEDS",
        n=None,
    )
    book.add(
        "numSeedSdDefault",
        dflt.get("final_sd_across_seeds_pct"),
        decimals=2,
        formula="sample SD over seeds of the final mean reduction, default config "
        "(N_pool=1, N_branch=3)",
        inputs=dsel,
        source="fig_curves.derived_numbers (E5_default_config)",
    )
    book.add(
        "numSeedRangeDefault",
        dflt.get("final_range_across_seeds_pct"),
        decimals=2,
        formula="max - min over seeds of the final mean reduction, default config",
        inputs=dsel,
        source="fig_curves.derived_numbers (E5_default_config)",
    )
    book.add(
        "numSeedSdPooled",
        noise["sd_pooled"] if noise else None,
        decimals=2,
        formula="sqrt(pooled within-config variance over seeds) of the final mean "
        "reduction, over the distinct E5 configurations",
        inputs=e5_sel,
        source="numbers.e5_noise (from fig_curves.figure_numbers per_seed)",
        n=noise["n_configs"] if noise else 0,
    )
    crit = None
    if noise:
        crit = hsd(noise["sd_pooled"], 5, noise["df"], len(seeds))
    book.add(
        "numSeedHsd",
        crit,
        decimals=2,
        formula="Tukey HSD (95%) for 5 means of len(seeds) runs: "
        "q(0.95; 5, df) * SD_pooled / sqrt(seeds)",
        inputs=e5_sel,
        source="numbers.hsd",
    )


# ---------------------------------------------------------------------------
# setup facts
# ---------------------------------------------------------------------------


def _parse_cpus(spec):
    n = 0
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def events_stats(path, job_ids):
    """Concurrency and re-run counts from the events.csv rows of the given jobs."""
    with open(path) as f:
        ev = [e for e in csv.DictReader(f) if e["job_id"] in job_ids]
    by_host = {}
    for e in ev:
        by_host.setdefault(e["host"], []).append(e)
    hosts = {}
    requeue = Counter()
    for host, es in by_host.items():
        es.sort(key=lambda e: float(e["utc"]))
        launches = [float(e["utc"]) for e in es if e["event"] == "launch"]
        if not launches:
            continue
        t0, t_last_launch = launches[0], launches[-1]
        t_end = float(es[-1]["utc"])
        area_all = area_launch = 0.0
        prev_t, prev_n = None, 0
        max_n = 0
        for e in es:
            t, n = float(e["utc"]), int(e["running"])
            if prev_t is not None and t > t0:
                lo = max(prev_t, t0)
                area_all += prev_n * (t - lo)
                if lo < t_last_launch:
                    area_launch += prev_n * (min(t, t_last_launch) - lo)
            prev_t, prev_n = t, n
            max_n = max(max_n, n)
            if e["event"] == "requeue":
                info = json.loads(e["info"]) if e["info"] else {}
                reason = info.get("reason", "?") if isinstance(info, dict) else "?"
                requeue[reason] += 1
        hosts[host] = {
            "max_running": max_n,
            "mean_running": area_all / (t_end - t0) if t_end > t0 else None,
            "mean_running_while_launching": (
                area_launch / (t_last_launch - t0) if t_last_launch > t0 else None
            ),
            "span_h": (t_end - t0) / 3600,
            "n_launch": len(launches),
            "n_finish": sum(e["event"] == "finish" for e in es),
        }
    return {"hosts": hosts, "requeue": dict(requeue), "n_events": len(ev)}


def setup_numbers(book, rows, events_path, job_ids):
    book.section = "setup"
    m248 = tables.load_manifest("S248")
    with open(os.path.join(REPRO, "inputs", "S248.csv")) as f:
        rows248 = list(csv.DictReader(f))
    book.selection("manifest:S248", "repro/inputs/S248.csv", [], len(m248), len(m248))
    book.selection(
        "manifest:S26",
        "repro/inputs/S26.csv",
        [],
        len(tables.load_manifest("S26")),
        len(tables.load_manifest("S26")),
    )
    book.add(
        "numSuiteCircuits",
        len(m248),
        kind="int",
        formula="circuits in repro/inputs/S248.csv (every S248 column must have them all)",
        inputs=["manifest:S248"],
        source="tables.load_manifest",
    )
    src_dirs = {os.path.dirname(r["path"]) for r in rows248}
    removed, n_src = None, None
    # manifest paths are relative to the repository root
    src_dir = os.path.join(os.path.dirname(REPRO), next(iter(src_dirs)))
    if len(src_dirs) == 1 and os.path.isdir(src_dir):
        names = {f[: -len(".qasm")] for f in os.listdir(src_dir) if f.endswith(".qasm")}
        n_src = len(names)
        removed = sorted(names - set(m248))
    else:
        book.warn(f"S248 source directory not readable: {sorted(src_dirs)}")
    book.selection(
        "inputs:S248_source",
        f"*.qasm in {sorted(src_dirs)}",
        [],
        n_src or 0,
        n_src or 0,
        removed=removed,
    )
    book.add(
        "numSuiteSourceCircuits",
        n_src,
        kind="int",
        formula="*.qasm files in the GUOQ benchmark directory S248 is taken from",
        inputs=["inputs:S248_source"],
        source="numbers.setup_numbers",
    )
    book.add(
        "numSuiteRemovedCircuits",
        len(removed) if removed is not None else None,
        kind="int",
        formula=f"source circuits not in S248 ({removed})",
        inputs=["inputs:S248_source", "manifest:S248"],
        source="numbers.setup_numbers",
    )
    tot = [m["total"] for m in m248.values()]
    orders = math.log10(max(tot) / min(tot))
    book.add(
        "numGateCountOrders",
        orders,
        kind="text",
        text=word(math.floor(orders)),
        formula=f"floor(log10(max/min original total)) = floor(log10({max(tot)}/"
        f"{min(tot)})), as a word",
        inputs=["manifest:S248"],
        source="tables.load_manifest",
    )
    # campaign-wide facts from runs.csv
    book.selection(
        "runs:all",
        "every row of runs.csv with a result",
        [r["job_id"] for r in rows],
        len(rows),
        len(rows),
    )
    book.add(
        "numJobs",
        len(rows),
        kind="int",
        formula="runs.csv rows with a result (unique job ids)",
        inputs=["runs:all"],
        source="metrics.load_runs",
    )
    hosts = sorted({r.get("host") or "?" for r in rows})
    book.add(
        "numHosts",
        len(hosts),
        kind="int",
        formula=f"distinct hosts in runs.csv ({hosts})",
        inputs=["runs:all"],
        source="metrics.load_runs",
    )
    peaks = [
        int(r["cgroup_peak_bytes"]) / GIB
        for r in rows
        if str(r.get("cgroup_peak_bytes") or "").strip() not in ("", "None")
    ]
    rss = [
        int(r["peak_rss_kb"]) * 1024 / GIB
        for r in rows
        if str(r.get("peak_rss_kb") or "").strip() not in ("", "None")
    ]
    book.add(
        "numPeakRssGiB",
        max(rss) if rss else None,
        decimals=1,
        formula="max over runs of the peak resident set size (GiB)",
        inputs=["runs:all"],
        source="runs.csv peak_rss_kb",
    )
    book.add(
        "numPeakCgroupGiB",
        max(peaks) if peaks else None,
        decimals=1,
        formula="max over runs of the cgroup memory.peak (GiB; includes page cache)",
        inputs=["runs:all"],
        source="runs.csv cgroup_peak_bytes",
    )
    reruns = sum(
        int(r["attempt"]) > 1 for r in rows if str(r.get("attempt") or "").isdigit()
    )
    book.add(
        "numJobsRerun",
        reruns,
        kind="int",
        formula="runs whose reported attempt is not the first (memory-cap escalation, "
        "preemption or infrastructure failure)",
        inputs=["runs:all"],
        source="runs.csv attempt",
    )
    with open(os.path.join(REPRO, "config", "campaign.yaml")) as f:
        camp = yaml.safe_load(f)
    book.selection("config:campaign", "repro/config/campaign.yaml", [], 1, 1)
    cap = max(camp["mem"]["cap_tiers_gib"])
    book.add(
        "numMemCapGiB",
        cap,
        kind="int",
        formula="largest memory cap tier (GiB): every reported run ran with headroom "
        "below it (smaller tiers escalate)",
        inputs=["config:campaign"],
        source="repro/config/campaign.yaml mem.cap_tiers_gib",
    )
    book.add(
        "numJobCores",
        _parse_cpus(camp["cores"]["jobs"]),
        kind="int",
        formula=f"job cores {camp['cores']['jobs']} (one job per physical core)",
        inputs=["config:campaign"],
        source="repro/config/campaign.yaml cores.jobs",
    )
    # scheduler events (optional)
    ev = events_stats(events_path, job_ids) if events_path else None
    hs = list(ev["hosts"].values()) if ev else []
    book.selection(
        "events",
        f"scheduler events {events_path}",
        [],
        ev["n_events"] if ev else 0,
        ev["n_events"] if ev else 1,
        required=False,
    )
    mean_launch = (
        statistics.mean(h["mean_running_while_launching"] for h in hs) if hs else None
    )
    for name, value, kind, formula in (
        (
            "numMaxConcurrent",
            max(h["max_running"] for h in hs) if hs else None,
            "int",
            "max jobs running at once on one host",
        ),
        (
            "numMeanConcurrent",
            mean_launch,
            "int",
            "time-weighted mean of jobs running on one host while jobs were still being "
            "launched (first to last launch), averaged over hosts; rounded",
        ),
        (
            "numMemEscalations",
            ev["requeue"].get("esc", 0) if ev else None,
            "int",
            "re-runs with the next memory-cap tier (requeue reason esc)",
        ),
        (
            "numPreemptions",
            ev["requeue"].get("pre", 0) if ev else None,
            "int",
            "jobs preempted by the memory watchdog and re-run (reason pre)",
        ),
        (
            "numInfraReruns",
            ev["requeue"].get("infra", 0) if ev else None,
            "int",
            "re-runs after infrastructure failures (reason infra)",
        ),
        (
            "numCampaignHours",
            max(h["span_h"] for h in hs) if hs else None,
            "number",
            "wall-clock hours from the first launch to the last event on a host",
        ),
    ):
        book.add(
            name,
            value,
            kind=kind,
            decimals=0 if kind == "number" else None,
            formula=formula,
            inputs=["events"],
            source="numbers.events_stats",
        )
    return ev


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_tex(book, path, meta):
    lines = [AUTOGEN]
    lines.append(
        f"% results: {meta['results']} (runs.csv sha256 "
        f"{(meta['inputs_sha256']['runs.csv'] or '')[:16]}); QALM config "
        f"{meta['qalm_config']}; threshold {meta['threshold_pct']:g}%"
    )
    if meta["n_provisional"]:
        lines.append(
            f"% PROVISIONAL: built with --allow-partial; {meta['n_provisional']} of "
            f"{meta['n_macros']} macros end in '?' (incomplete inputs), "
            f"{meta['n_missing']} print '{MISSING_TEXT}' (no data)."
        )
    section = None
    for name, m in book.macros.items():
        if m["section"] != section:
            section = m["section"]
            lines += ["", f"% --- {section}"]
        note = m["formula"].replace("\n", " ")
        lines.append(f"\\newcommand{{\\{name}}}{{{m['printed']}}} % {note}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results",
        default=os.path.join(REPRO, "results"),
        help="dir with runs.csv, timeseries.csv.gz, kevents.csv",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="output directory")
    ap.add_argument("--qalm-config", choices=sorted(tables.QALM_CONFIGS), default="a")
    ap.add_argument(
        "--threshold",
        type=float,
        default=32.0,
        help="mean reduction %% for the time-to-X numbers (default 32)",
    )
    ap.add_argument("--events", default=None, help="scheduler events.csv (optional)")
    ap.add_argument(
        "--plan",
        default=None,
        help="plan whose jobs' events count (default: the jobs in runs.csv)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=3600,
        help="budget_s of the 'one hour' runs (other values: smoke tests only)",
    )
    ap.add_argument(
        "--short-budget",
        type=int,
        default=60,
        help="budget_s of the 'one minute' runs (default 60)",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="continue on incomplete selections; mark affected macros provisional",
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    # its per-column "no row for N circuits" lists are in our selection records
    logging.getLogger("fig_percircuit").setLevel(logging.ERROR)

    book = Book(a.allow_partial)
    rows = dedupe(metrics.load_runs(a.results), book)
    view = budget_view(rows, a.budget, a.short_budget)
    if a.plan:
        with open(a.plan) as f:
            job_ids = {json.loads(line)["job_id"] for line in f}
    else:
        with open(os.path.join(a.results, "runs.csv")) as f:
            job_ids = {r["job_id"] for r in csv.DictReader(f)}  # incl. rows load_runs skips
    setup_numbers(book, rows, a.events, job_ids)
    s248_numbers(book, view, a.qalm_config, a.budget, a.short_budget)
    ete_numbers(book, view, a.qalm_config)
    fnum, derived, curve_issues = curve_numbers(book, a.results, a.budget, a.threshold)
    noise = e5_noise(fnum)
    greedy_numbers(book, fnum, derived, a.threshold)
    single_k_numbers(book, fnum)
    e5_numbers(book, fnum, derived, noise)

    if book.errors and not a.allow_partial:
        msg = "\n  ".join(book.errors)
        raise SystemExit(
            f"numbers.py: {len(book.errors)} problem(s); fix the data or use "
            f"--allow-partial:\n  {msg}"
        )
    n_prov = sum(m["provisional"] for m in book.macros.values())
    n_missing = sum(m["printed"] == MISSING_TEXT for m in book.macros.values())
    meta = {
        "script": "repro/analysis/numbers.py",
        "results": a.results,
        "inputs_sha256": {
            n: _file_sha(os.path.join(a.results, n))
            for n in ("runs.csv", "timeseries.csv.gz", "kevents.csv")
        },
        "events": a.events,
        "events_sha256": _file_sha(a.events) if a.events else None,
        "events_plan": a.plan,
        "qalm_config": a.qalm_config,
        "qalm_selection": {
            "optimizer": tables.QALM_CONFIGS[a.qalm_config][0],
            "params": tables.QALM_CONFIGS[a.qalm_config][1],
        },
        "budget_s": a.budget,
        "short_budget_s": a.short_budget,
        "threshold_pct": a.threshold,
        "allow_partial": a.allow_partial,
        "complete": not book.errors,
        "n_macros": len(book.macros),
        "n_provisional": n_prov,
        "n_missing": n_missing,
        "curve_data_issues": curve_issues,
        "noise_e5": noise,
        "conventions": [
            "values are unrounded; differences are taken before rounding",
            "printed: fixed decimals as in the paper; times and speedups with "
            "1 decimal below 10, else 0; a trailing '?' marks a provisional value",
        ],
    }
    os.makedirs(a.out, exist_ok=True)
    out = {
        "meta": meta,
        "selections": book.selections,
        "macros": book.macros,
        "problems": book.errors,
        "warnings": book.warnings,
    }
    jpath = os.path.join(a.out, "numbers.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    tpath = os.path.join(a.out, "numbers.tex")
    write_tex(book, tpath, meta)
    print(f"{len(book.macros)} macros ({n_prov} provisional, {n_missing} without data)")
    print(f"wrote {jpath}\nwrote {tpath}")


if __name__ == "__main__":
    main()
