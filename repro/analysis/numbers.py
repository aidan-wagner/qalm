"""Every data-dependent number and claim of the paper text, as LaTeX macros.

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
    repro/env/*.txt                       host snapshots (CPU model claim)

Outputs (in --out, default repro/analysis/out)
    numbers.tex     one ``\\newcommand{\\num<Name>}{<text>}`` per number the
                    compiled paper prints (abstract, introduction, evaluation,
                    captions), named in CamelCase letters only.  Each expands
                    to the number exactly as printed: fixed decimals as in
                    the current paper, percentages without the % sign, a
                    minus as ``\\ensuremath{-}`` (works in text and math).
                    Use ``\\numX{}`` before a space.  A few macros expand to
                    data-derived text (a baseline name, a circuit list, a
                    small count as a word).
    numbers.json    {"meta", "selections", "macros", "claims", "problems",
                    "warnings"}:
                    per macro its unrounded value, printed text, formula,
                    inputs (selection keys), n, source function and the
                    value it replaces in the current paper; per claim the
                    paper location and wording, the check, its boolean
                    result (null if undetermined) and the numbers behind it.

Numbers come from the existing generators, never recomputed here:
tables.averages_table_data / eval_end_to_end_data (both tables),
fig_percircuit.gather (win rates), fig_curves (curve finals, time to reach
the threshold, greedy phase, E5 seeds); metrics.py holds every definition.
This module only derives differences (always from unrounded values), counts,
noise estimates and the claim checks.

QALM is configuration "a" or "b" (``--qalm-config``, see tables.py); the
curve experiments (E3-E5) always use optimizer ``qalm``.  The time-to-X
threshold is ``--threshold`` (default 32 %, as in the paper).

Run-to-run noise: E5 runs every N_pool/N_branch configuration with seeds
{0,1,2}.  The pooled within-configuration SD of the 26-circuit mean final
reduction (\\numSeedSdPooled) is the noise reference: N_pool "has no
effect" if the spread of the seed-averaged finals is below Tukey's HSD
(studentized range, 95 %); single-run comparisons (greedy, single-k) report
z = difference / (SD * sqrt(2)).

Completeness: every selection must hold all circuits of its suite (and all
seeds), the curve data must be consistent (fig_curves checks) and the
threshold must be reached by the greedy curves; otherwise the script exits
with the full list of problems.  With --allow-partial it continues; a macro
whose inputs are incomplete is marked provisional and printed with a
trailing "?" (a missing value prints "??"), and claims on missing values are
null.

``--budget`` / ``--short-budget`` (default 3600 / 60) select the runs used
as "one hour" and "one minute"; other values are only for smoke tests.

CLI (from the repo root):
    python3 repro/analysis/numbers.py --results repro/results \\
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
import glob  # noqa: E402
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
SEARCH_BASED = ["guoq_2q", "guoq_total", "queso_2q", "queso_total", "quartz"]
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

# Values the macros replace in the current paper (location, printed value).
AT = "figtex/averages_table.tex"
OLD_AVG = {
    "fidelity": (
        f"{AT}:14-15",
        [
            "0.3864",
            "0.3747",
            "0.3475",
            "0.3676",
            "0.3362",
            "0.3554",
            "0.3428",
            "0.3526",
        ],
    ),
    "cx_red": (
        f"{AT}:16-17",
        ["24.61", "21.18", "25.41", "18.83", "13.19", "13.37", "4.24", "12.42"],
    ),
    "total_red": (
        f"{AT}:18-19",
        ["52.34", "51.35", "-2.63", "43.89", "27.40", "39.72", "46.37", "40.73"],
    ),
}
OLD_AVG_TEXT = {  # extra places in the text that print a table cell
    ("cx_red", "qalm"): "texts/5-eval.tex:116,123",
    ("cx_red", "guoq_2q"): "texts/5-eval.tex:117",
    ("total_red", "qalm"): "texts/5-eval.tex:117,122",
    ("total_red", "guoq_2q"): "texts/5-eval.tex:118",
    ("total_red", "voqc"): "texts/5-eval.tex:122",
    ("cx_red", "voqc"): "texts/5-eval.tex:124",
    ("fidelity", "qalm"): "texts/5-eval.tex:124",
    ("fidelity", "voqc"): "texts/5-eval.tex:125",
    ("fidelity", "guoq_2q"): "texts/5-eval.tex:126",
}
OLD_WIN = {  # win-or-tie labels of figure/win_rate.png (fidelity, cx, total)
    "guoq_2q": ("83.9", "54.4", "88.3"),
    "guoq_total": ("95.2", "95.6", "96.0"),
    "queso_2q": ("98.8", "99.6", "98.8"),
    "queso_total": ("98.4", "99.6", "98.0"),
    "voqc": ("86.5", "100.0", "82.0"),
    "quartz": ("97.6", "99.2", "96.8"),
}
ETE = "figtex/eval_end_to_end.tex"
OLD_GEO = {
    "voqc": "27.0",
    "queso": "23.9",
    "guoq": "29.2",
    "quartz_nopp": "16.5",
    "quartz_pp": "29.7",
    "qalm": "34.3",
}
OLD_DELTA = {  # texts/5-eval.tex line, value
    "voqc": ("160", "7.3"),
    "quartz_nopp": ("162", "17.8"),
    "queso": ("163", "10.4"),
    "guoq": ("163", "5.1"),
    "quartz_pp": ("164", "4.6"),
}
EV = "texts/5-eval.tex"


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


def _round(x, nd=6):
    """Rounded copy for the claims' "numbers" (the macros keep full precision)."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


# ---------------------------------------------------------------------------
# the book of macros, selections and claims
# ---------------------------------------------------------------------------


class Book:
    def __init__(self, allow_partial):
        self.allow_partial = allow_partial
        self.selections = {}
        self.macros = {}
        self.claims = []
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
        paper=None,
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
            "paper": (
                None if paper is None else {"location": paper[0], "value": paper[1]}
            ),
        }

    def claim(self, cid, location, wording, check, result, numbers, inputs, note=None):
        for i in inputs:
            if i not in self.selections:
                raise KeyError(f"claim {cid}: unknown selection {i}")
        if result is not None:
            result = bool(result)
        self.claims.append(
            {
                "id": cid,
                "section": self.section,
                "paper_location": location,
                "wording": wording,
                "check": check,
                "result": result,
                "numbers": _round(numbers),
                "inputs": list(inputs),
                "provisional": not self.complete(inputs) or result is None,
                **({"note": note} if note else {}),
            }
        )


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


def _gt(a, b):
    return None if not _all_known(a, b) else a > b


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
        loc, old = OLD_AVG[metric]
        for i, (key, suffix) in enumerate(AVG_COLS.items()):
            where = loc + (
                f"; {OLD_AVG_TEXT[(metric, key)]}"
                if (metric, key) in OLD_AVG_TEXT
                else ""
            )
            book.add(
                f"num{stem}{suffix}",
                cols[key][metric],
                decimals=dec,
                formula=f"{desc} over the circuits of column {key} (metrics.averages)",
                inputs=[f"avg:{key}"],
                source="tables.averages_table_data",
                paper=(where, old[i]),
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
    for key, suffix, where, old in (
        ("qalm", "OneHour", "texts/1-introduction.tex:123", "5.97"),
        ("qalm_1min", "OneMin", "texts/1-introduction.tex:125", "4.98"),
    ):
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
            paper=(where, old),
        )
    book.add(
        "numStrongestTotalRedBaseline",
        None,
        kind="text",
        text=labels[strongest] if strongest else None,
        formula="baseline column with the highest mean total gate reduction",
        inputs=all_base,
        source="tables.averages_table_data",
        paper=("texts/1-introduction.tex:124-125 ('the strongest baseline')", "VOQC"),
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
            paper=(f"{EV}:139-140 ('by less than 1\\%')", "<1"),
        )
    book.add(
        "numBudgetFactor",
        budget / short_budget,
        kind="int",
        formula=f"one-hour budget / one-minute budget ({budget} s / {short_budget} s)",
        inputs=[],
        source="numbers.py --budget / --short-budget",
        paper=(f"{EV}:139", "60"),
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
            old = OLD_WIN[b][list(WIN_METRICS).index(metric)]
            where = "figure/win_rate.png"
            if (b, metric) == ("guoq_2q", "fidelity"):
                where += "; texts/0-abstract.tex:40; texts/1-introduction.tex:136"
            book.add(
                f"numWin{stem}{AVG_COLS[b]}",
                wr["win_or_tie"] if wr else None,
                decimals=1,
                formula=f"% of circuits where QALM >= {b} on {metric} "
                "(metrics.win_rate win_or_tie; fidelity ties |d ln F| <= 1e-12)",
                inputs=ins,
                source="fig_percircuit.gather",
                paper=(where, old),
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
        paper=("texts/0-abstract.tex:40; texts/1-introduction.tex:136", "83.9"),
    )
    book.add(
        "numStrongestFidBaseline",
        None,
        kind="text",
        text=labels[strongest_fid] if strongest_fid else None,
        formula="baseline against which QALM has the lowest fidelity win-or-tie rate",
        inputs=pair_ins,
        source="fig_percircuit.gather",
        paper=("texts/0-abstract.tex:39; figtex/gate_scatter.tex:10-11", "GUOQ (2Q)"),
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
        paper=(
            "texts/1-introduction.tex:133-134; figure/win_rate.png",
            "'nearly every circuit' (figure: lowest 82.0, VOQC total)",
        ),
    )
    strict_fid = {b: win[b]["fidelity"]["win"] for b in BASELINES} if have_all else {}
    book.add(
        "numStrictWinFidMin",
        min(strict_fid.values()) if strict_fid else None,
        decimals=1,
        formula="lowest strict fidelity win % over the six baselines",
        inputs=pair_ins,
        source="fig_percircuit.gather",
        paper=(f"{EV}:107-109 ('on the majority of circuits')", ">50"),
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
        paper=(f"{EV}:119-120 ('on many circuits')", "many"),
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
        paper=(f"{EV}:120", "implied 0 ('avoids this entirely')"),
    )
    s248_claims(book, avg, win, drops, margin, strongest, strongest_fid, neg, n_g)
    return avg


def s248_claims(book, avg, win, drops, margin, strongest, strongest_fid, neg, n_g):
    cols = avg["columns"]

    def v(key, metric):
        return cols[key][metric]

    all_base = [f"avg:{k}" for k in BASELINES]
    have_all = all(win[b] for b in BASELINES)
    pair = ["avg:qalm"] + all_base
    book.section = "claims: abstract / introduction / main results"
    book.claim(
        "abstract_strongest_baseline_fidelity",
        "texts/0-abstract.tex:38-40; texts/1-introduction.tex:134-137",
        "\\optName{} matches or exceeds the fidelity of the strongest baseline on "
        "$83.9\\%$ of these circuits",
        "the baseline the sentence means (GUOQ (2Q Gate)) is still the one with the "
        "lowest fidelity win-or-tie rate; the percentage is \\numWinFidGuoqTwoQ",
        (strongest_fid == "guoq_2q") if strongest_fid else None,
        {
            "fidelity_win_or_tie": (
                {b: win[b]["fidelity"]["win_or_tie"] for b in BASELINES}
                if have_all
                else None
            ),
            "lowest": strongest_fid,
        },
        pair,
    )
    senses = {}
    if have_all and all(v(b, "fidelity") is not None for b in BASELINES):
        senses = {
            "lowest_fidelity_win_or_tie": strongest_fid,
            "highest_mean_fidelity": max(BASELINES, key=lambda b: v(b, "fidelity")),
            "highest_mean_total_red": strongest,
            "highest_mean_cx_red": max(BASELINES, key=lambda b: v(b, "cx_red")),
        }
    book.claim(
        "gate_scatter_strongest_baseline",
        "figtex/gate_scatter.tex:9-11; texts/1-introduction.tex:124-127",
        "\\guoq{} with two-qubit gate optimization mode, which is the strongest "
        "baseline in our evaluation",
        "GUOQ (2Q Gate) is the strongest baseline by per-circuit fidelity (lowest "
        "win-or-tie rate); other senses listed (the intro's 5.97% margin uses the "
        "highest mean total reduction, \\numStrongestTotalRedBaseline)",
        (senses["lowest_fidelity_win_or_tie"] == "guoq_2q") if senses else None,
        senses,
        pair,
    )
    q_tot = v("qalm", "total_red")
    beats_all = (
        all(q_tot > v(b, "total_red") for b in BASELINES)
        if _all_known(q_tot, *[v(b, "total_red") for b in BASELINES])
        else None
    )
    book.claim(
        "intro_outperforms_all_total",
        "texts/0-abstract.tex:34-37; texts/1-introduction.tex:121-122; "
        "texts/7-conclusion.tex:20-22",
        "\\optName{} significantly outperforms all existing rule-based and "
        "search-based optimizers (on total gate count reduction)",
        "QALM (1 h) mean total reduction > every baseline's; margin "
        "\\numTotalRedMarginOneHour points",
        beats_all,
        {"margin_pp": margin["qalm"], "strongest_by_total_red": strongest},
        pair,
    )
    for key, loc, old in (
        ("qalm", "texts/1-introduction.tex:123-125", "5.97"),
        ("qalm_1min", "texts/1-introduction.tex:125-127", "4.98"),
    ):
        book.claim(
            f"intro_margin_positive_{key}",
            loc,
            f"\\optName{{}} achieves {old}\\% more gate count reduction than the "
            "strongest baseline",
            f"total_red({key}) - max baseline total_red > 0 (the strongest baseline "
            "by total reduction must be named, or 'strongest' reworded: the abstract "
            "uses the per-circuit fidelity sense)",
            None if margin[key] is None else margin[key] > 0,
            {
                "margin_pp": margin[key],
                "strongest_by_total_red": strongest,
                "same_as_fidelity_sense": (
                    strongest == strongest_fid
                    if _all_known(strongest, strongest_fid)
                    else None
                ),
            },
            [f"avg:{key}"] + all_base,
        )
    others = (
        {f"{b}/{m}": win[b][m]["win_or_tie"] for b in NEARLY_EVERY for m in WIN_METRICS}
        if have_all
        else {}
    )
    book.claim(
        "intro_nearly_every_circuit",
        "texts/1-introduction.tex:130-135",
        "Compared with \\queso{} (both modes), \\voqc{}, \\quartz{}, and \\guoq{} "
        "(total gate mode), \\optName{} wins or ties on nearly every circuit across "
        "all three metrics",
        "every one of the 15 win-or-tie rates >= 95% ('nearly every'); lowest is "
        "\\numWinMinOthers",
        (min(others.values()) >= 95.0) if others else None,
        {
            "win_or_tie": others,
            "below_95": sorted(k for k, x in others.items() if x < 95.0),
        },
        ["avg:qalm"] + [f"avg:{b}" for b in NEARLY_EVERY],
    )
    gq = (v("qalm", "cx_red"), v("guoq_2q", "cx_red"))
    cx_diff = gq[0] - gq[1] if _all_known(*gq) else None
    book.claim(
        "guoq2q_cx_on_par",
        "texts/1-introduction.tex:135-136; texts/5-eval.tex:115-117",
        "the two-qubit gate reduction is on par with \\guoq{} (2Q mode) / nearly "
        "identical $CX$ reduction ($24.61\\%$ vs. $25.41\\%$)",
        "|cx_red(QALM) - cx_red(GUOQ 2Q)| <= 1 point",
        None if cx_diff is None else abs(cx_diff) <= 1.0,
        {
            "qalm_cx_red": gq[0],
            "guoq_2q_cx_red": gq[1],
            "diff_pp": cx_diff,
            "cx_win_or_tie": (
                win["guoq_2q"]["cx"]["win_or_tie"] if win["guoq_2q"] else None
            ),
        },
        ["avg:qalm", "avg:guoq_2q"],
    )
    tt = (v("qalm", "total_red"), v("guoq_2q", "total_red"))
    tot_diff = tt[0] - tt[1] if _all_known(*tt) else None
    book.claim(
        "guoq2q_total_dramatically_better",
        "texts/1-introduction.tex:136-137; texts/5-eval.tex:117-118",
        "dramatically better total gate reduction ($52.34\\%$ vs. $-2.63\\%$) / "
        "significantly lower total gate count",
        "total_red(QALM) - total_red(GUOQ 2Q) >= 10 points",
        None if tot_diff is None else tot_diff >= 10.0,
        {
            "qalm_total_red": tt[0],
            "guoq_2q_total_red": tt[1],
            "diff_pp": tot_diff,
            "total_win_or_tie": (
                win["guoq_2q"]["total"]["win_or_tie"] if win["guoq_2q"] else None
            ),
        },
        ["avg:qalm", "avg:guoq_2q"],
    )
    book.claim(
        "guoq2q_negative_on_many",
        f"{EV}:118-120",
        "\\guoq{}'s resynthesis ... yielding negative total gate reduction on many "
        "circuits",
        "GUOQ (2Q) has more gates than the input on >= 10% of circuits "
        "(\\numNegTotalGuoqTwoQ circuits)",
        None if neg["guoq_2q"] is None or not n_g else neg["guoq_2q"] >= 0.1 * n_g,
        {"n_negative": neg["guoq_2q"], "n": n_g},
        ["avg:guoq_2q"],
    )
    book.claim(
        "qalm_never_negative",
        f"{EV}:120",
        "whereas \\optName{} avoids this entirely",
        "no QALM (1 h) output has more gates than its input",
        None if neg["qalm"] is None else neg["qalm"] == 0,
        {"n_negative": neg["qalm"]},
        ["avg:qalm"],
    )
    vt = (v("qalm", "total_red"), v("voqc", "total_red"))
    vc = (v("qalm", "cx_red"), v("voqc", "cx_red"))
    book.claim(
        "voqc_similar_total",
        f"{EV}:121-122",
        "Against \\voqc{} ... both achieve similar total gate reductions "
        "($52.34\\%$ vs. $46.37\\%$)",
        "0 <= total_red(QALM) - total_red(VOQC) <= 10 points",
        None if not _all_known(*vt) else 0 <= vt[0] - vt[1] <= 10.0,
        {
            "qalm": vt[0],
            "voqc": vt[1],
            "diff_pp": vt[0] - vt[1] if _all_known(*vt) else None,
        },
        ["avg:qalm", "avg:voqc"],
    )
    book.claim(
        "voqc_much_greater_cx",
        f"{EV}:122-124",
        "\\optName{} yields significantly greater $CX$ reduction ($24.61\\%$ vs. "
        "$4.24\\%$)",
        "cx_red(QALM) - cx_red(VOQC) >= 10 points",
        None if not _all_known(*vc) else vc[0] - vc[1] >= 10.0,
        {
            "qalm": vc[0],
            "voqc": vc[1],
            "diff_pp": vc[0] - vc[1] if _all_known(*vc) else None,
        },
        ["avg:qalm", "avg:voqc"],
    )
    fq, fv, fg = v("qalm", "fidelity"), v("voqc", "fidelity"), v("guoq_2q", "fidelity")
    book.claim(
        "fidelity_beats_voqc_guoq2q",
        f"{EV}:124-127",
        "\\optName{} (average $0.3864$) outperforms both \\voqc{} ($0.3428$) and "
        "\\guoq{} in two-qubit mode ($0.3475$)",
        "mean fidelity QALM > VOQC and > GUOQ (2Q)",
        None if not _all_known(fq, fv, fg) else fq > fv and fq > fg,
        {"qalm": fq, "voqc": fv, "guoq_2q": fg},
        ["avg:qalm", "avg:voqc", "avg:guoq_2q"],
    )
    for metric in ("fidelity", "total_red"):
        others_v = {k: v(k, metric) for k in AVG_COLS if k != "qalm"}
        known = _all_known(v("qalm", metric), *others_v.values())
        book.claim(
            f"table_qalm_best_{metric}",
            "figtex/averages_table.tex:14-19 (bold); texts/5-eval.tex:124-127",
            {
                "fidelity": "Fidelity & \\textbf{0.3864} & ... (QALM column bold: "
                "the highest average fidelity); text: 'as it balances single- and "
                "two-qubit gate reduction most effectively'",
                "total_red": "Total Gate Reduction (\\%) & \\textbf{52.34} & ... "
                "(QALM column bold: the highest total gate reduction)",
            }[metric],
            f"QALM (1 h) mean {metric} >= every other column of the table "
            "(incl. QALM 1 min)",
            (
                None
                if not known
                else all(v("qalm", metric) >= x for x in others_v.values())
            ),
            {
                "qalm": v("qalm", metric),
                "best_other": (max(others_v, key=others_v.get) if known else None),
                "values": others_v,
            },
            [f"avg:{k}" for k in AVG_COLS],
        )
    rest = ["queso_2q", "queso_total", "quartz", "guoq_total"]
    rest_known = all(v(k, m) is not None for k in rest + ["qalm"] for m in AVG_METRICS)
    book.claim(
        "remaining_baselines_all_three",
        f"{EV}:129-132",
        "Against the remaining baselines, \\queso{} (in both optimization modes), "
        "\\quartz{}, and \\guoq{} in total gate optimization mode, \\optName{} "
        "achieves better results on all three metrics",
        "QALM mean > baseline mean for fidelity, cx_red and total_red, for each of "
        "the four",
        (
            all(v("qalm", m) > v(k, m) for k in rest for m in AVG_METRICS)
            if rest_known
            else None
        ),
        {k: {m: v(k, m) for m in AVG_METRICS} for k in rest + ["qalm"]},
        ["avg:qalm"] + [f"avg:{k}" for k in rest],
    )
    strict_fid = {b: win[b]["fidelity"]["win"] for b in BASELINES} if have_all else {}
    book.claim(
        "majority_strict_fidelity",
        f"{EV}:105-109",
        "\\optName{} achieves higher fidelity than all four baselines on the majority "
        "of circuits",
        "strict fidelity win > 50% against each of the six baseline columns",
        (min(strict_fid.values()) > 50.0) if strict_fid else None,
        {"strict_fidelity_win_pct": strict_fid},
        pair,
    )
    book.section = "claims: optimization efficiency (one minute)"
    known = _all_known(*drops.values())
    book.claim(
        "one_minute_degrades_lt_1pct",
        f"{EV}:138-140",
        "reducing the time budget by a factor of $60$ degrades \\optName{}'s average "
        "fidelity and gate reduction by less than $1\\%$",
        "relative fidelity drop < 1% AND cx_red drop < 1 point AND total_red drop "
        "< 1 point",
        (
            drops["fid_rel_pct"] < 1
            and drops["cx_red_pp"] < 1
            and drops["total_red_pp"] < 1
            if known
            else None
        ),
        {
            **drops,
            "each_lt_1": (
                {k: x < 1 for k, x in drops.items() if k != "fid_abs"}
                if known
                else None
            ),
        },
        ["avg:qalm", "avg:qalm_1min"],
    )
    one = {m: v("qalm_1min", m) for m in AVG_METRICS}
    best_base = {
        m: (
            max((v(b, m) for b in BASELINES), default=None)
            if all(v(b, m) is not None for b in BASELINES)
            else None
        )
        for m in AVG_METRICS
    }
    ins = ["avg:qalm_1min"] + all_base
    book.claim(
        "one_minute_beats_every_baseline",
        f"{EV}:140-141",
        "yet it still surpasses the one-hour fidelity and total gate reduction of "
        "every baseline",
        "QALM (1 min) mean fidelity > max baseline fidelity AND total_red > max "
        "baseline total_red",
        (
            one["fidelity"] > best_base["fidelity"]
            and one["total_red"] > best_base["total_red"]
            if _all_known(one["fidelity"], one["total_red"], *best_base.values())
            else None
        ),
        {"qalm_1min": one, "max_baseline": best_base},
        ins,
    )
    search_best = {
        m: (
            max(v(b, m) for b in SEARCH_BASED)
            if all(v(b, m) is not None for b in SEARCH_BASED)
            else None
        )
        for m in AVG_METRICS
    }
    per_metric = {m: _gt(one[m], search_best[m]) for m in AVG_METRICS}
    book.claim(
        "conclusion_one_minute_beats_search_based",
        "texts/7-conclusion.tex:23-24",
        "it surpasses the one-hour results of all search-based baselines within the "
        "first minute",
        "QALM (1 min) mean > max over GUOQ (both), QUESO (both), Quartz, for all three "
        "metrics",
        None if None in per_metric.values() else all(per_metric.values()),
        {"per_metric": per_metric, "qalm_1min": one, "max_search_based": search_best},
        ["avg:qalm_1min"] + [f"avg:{b}" for b in SEARCH_BASED],
    )


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
    labels = {c.key: c.label for c in tables.e2e_columns(qcfg)}
    for key, suffix in ETE_COLS.items():
        book.add(
            f"numGeoRed{suffix}",
            cols[key]["geomean_red"],
            decimals=1,
            formula=f"100*(1 - geomean(opt/orig)) over the 26 circuits, column {key} "
            "(S26 originals; metrics.geomean_reduction_pct)",
            inputs=[f"ete:{key}"],
            source="tables.eval_end_to_end_data",
            paper=(f"{ETE}:37", OLD_GEO[key]),
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
    for key, (line, old) in OLD_DELTA.items():
        b = cols[key]["geomean_red"]
        deltas[key] = q - b if _all_known(q, b) else None
        book.add(
            f"numGeoDelta{ETE_COLS[key]}",
            deltas[key],
            decimals=1,
            formula=f"geomean_red(qalm) - geomean_red({key}), unrounded, then rounded",
            inputs=["ete:qalm", f"ete:{key}"],
            source="tables.eval_end_to_end_data",
            paper=(f"{EV}:{line}", old),
        )
    all_ete = [f"ete:{k}" for k in ETE_COLS]
    has_q = cols["qalm"]["n"] > 0
    n_best = cols["qalm"]["n_best"] if has_q else None
    not_best = [
        c
        for c in ete["circuits"]
        if "qalm" not in ete["best"][c] and cols["qalm"]["opt"][c] is not None
    ]
    complete = book.complete(all_ete)
    book.add(
        "numETEQalmBest",
        n_best,
        kind="int",
        formula="circuits where QALM has the lowest gate count (ties count)",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (n_best)",
        paper=(f"{EV}:156-158", "implied 25 ('all of them except for one')"),
    )
    book.add(
        "numETEQalmNotBest",
        len(not_best) if has_q else None,
        kind="int",
        formula="circuits where QALM is not (co-)lowest (26 - numETEQalmBest)",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (best)",
        paper=(f"{EV}:158", "implied 1 ('except for one circuit')"),
    )
    book.add(
        "numETEQalmNotBestWord",
        len(not_best) if has_q else None,
        kind="text",
        text=word(len(not_best)) if has_q else None,
        formula="numETEQalmNotBest as a word",
        inputs=all_ete,
        source="tables.eval_end_to_end_data (best)",
        paper=(f"{EV}:158", "one"),
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
            paper=(
                (f"{EV}:171-172", "implied 25 ('all but one circuit')")
                if key == "quartz_pp"
                else None
            ),
        )
        book.add(
            f"numETEQalmWorse{suffix}",
            vs["worse"] if both else None,
            kind="int",
            formula=f"circuits where QALM's gate count > {key}'s",
            inputs=["ete:qalm", f"ete:{key}"],
            source="tables.eval_end_to_end_data (qalm_vs)",
            paper=(
                (f"{EV}:171-172", "implied 1 ('all but one circuit')")
                if key == "quartz_pp"
                else None
            ),
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
        paper=(f"{EV}:171-172 ('on all but one circuit')", "one"),
    )
    book.add(
        "numQuartzSuiteCircuits",
        n_circ,
        kind="int",
        formula="circuits in repro/inputs/S26.csv",
        inputs=["manifest:S26"],
        source="tables.eval_end_to_end_data",
        paper=(
            f"{EV}:150,156,331,358,409; {ETE}:3; ablation figure titles",
            "26",
        ),
    )
    # ---- claims ----
    book.section = "claims: Quartz's benchmark (tab:eval_end_to_end)"
    winners = {c: ete["best"][c] for c in not_best}
    book.claim(
        "ete_lowest_all_but_one",
        f"{EV}:155-159",
        "\\optName{} achieves the lowest or equally lowest gate count on all of them "
        "except for one circuit compared to \\quartz with preprocessing",
        "exactly one circuit where QALM is not (co-)lowest, and Quartz w/ "
        "preprocessing is (one of) the lowest there",
        (
            len(not_best) == 1 and "quartz_pp" in winners[not_best[0]]
            if complete
            else None
        ),
        {"not_best": winners, "n_best": n_best},
        all_ete,
    )
    vs_pp = ete["qalm_vs"]["quartz_pp"]
    book.claim(
        "ete_matches_quartz_pp_all_but_one",
        f"{EV}:169-172",
        "\\optName{} still matches or outperforms it [\\quartz{} with preprocessing] "
        "on all but one circuit",
        "QALM gate count > Quartz w/ PP on exactly one circuit",
        vs_pp["worse"] == 1 if book.complete(["ete:qalm", "ete:quartz_pp"]) else None,
        vs_pp,
        ["ete:qalm", "ete:quartz_pp"],
    )
    book.claim(
        "ete_subsumes_preprocessing",
        f"{EV}:174-176",
        "typically subsumes the optimizations found by \\quartz{}'s preprocessing pass",
        "QALM <= Quartz w/ PP on more than half of the 26 circuits",
        (
            vs_pp["better"] + vs_pp["equal"] > n_circ / 2
            if book.complete(["ete:qalm", "ete:quartz_pp"])
            else None
        ),
        vs_pp,
        ["ete:qalm", "ete:quartz_pp"],
    )
    book.claim(
        "ete_outperforms_by_positive",
        f"{EV}:159-164",
        "Compared to the purely rule-based \\voqc{}, \\optName{} reduces gate count by "
        "an additional $7.3\\%$ ...; outperforms \\quartz{} (without preprocessing) by "
        "$17.8\\%$, \\queso{} by $10.4\\%$, \\guoq{} by $5.1\\%$, and \\quartz{} (with "
        "preprocessing) by $4.6\\%$",
        "every geomean delta (QALM - baseline, unrounded) > 0",
        None if None in deltas.values() else all(d > 0 for d in deltas.values()),
        {"deltas_pp": deltas},
        all_ete,
        note="the deltas are differences of geometric-mean reductions (points), not "
        "arithmetic means of per-circuit differences",
    )
    geo = {k: cols[k]["geomean_red"] for k in ETE_COLS}
    book.claim(
        "ete_consistently_better",
        f"{EV}:172-175",
        "yields consistently better circuit quality than either approach alone",
        "QALM has the highest geomean reduction of the six columns",
        (
            all(geo["qalm"] > x for k, x in geo.items() if k != "qalm")
            if None not in geo.values()
            else None
        ),
        {"geomean_red": geo, "labels": labels},
        all_ete,
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


def _noise(diff, sd, n_each=1, df=None):
    """z-score of a difference of two means of n_each runs, given the run-to-run SD."""
    if diff is None or not sd:
        return None
    se = sd * math.sqrt(2 / n_each)
    crit = float(stats.t.ppf(1 - ALPHA / 2, df)) if df else 1.959963984540054
    return {
        "sd_run_to_run": sd,
        "se_diff": se,
        "z": diff / se,
        "critical": crit,
        # bool(): a numpy bool would be written to numbers.json as "True"/"False"
        "resolved": bool(abs(diff / se) > crit),
    }


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


def greedy_numbers(book, fnum, derived, threshold, budget, noise):
    book.section = "greedy mode (fig:greedy_k)"
    sel = {gk: f"curve:greedy_k:greedy_k={gk}" for gk in range(4)}
    allg = list(sel.values())
    book.add(
        "numGreedyThreshold",
        threshold,
        kind="text",
        text=f"{threshold:g}",
        formula="--threshold (mean total reduction %, for time-to-X)",
        inputs=[],
        source="numbers.py --threshold",
        paper=(f"{EV}:324", "32"),
        n=None,
    )
    finals = {gk: _curve(fnum, "greedy_k", f"greedy_k={gk}") for gk in range(4)}
    times = {
        gk: _curve(fnum, "greedy_k", f"greedy_k={gk}", "time_to_threshold_s")
        for gk in range(4)
    }
    old_final = {
        0: (f"{EV}:321", "32.80"),
        2: (f"{EV}:322,335", "33.27"),
        3: (f"{EV}:334", "33.12"),
    }
    old_time = {0: (f"{EV}:323-324", "968"), 1: (f"{EV}:326", "89")}
    old_phase = {
        1: (f"{EV}:329", "5.5"),
        2: (f"{EV}:329-330", "82"),
        3: (f"{EV}:331", "724"),
    }
    for gk in range(4):
        book.add(
            f"numGreedyFinalGk{DIGIT[gk]}",
            finals[gk],
            decimals=2,
            formula=f"final mean total reduction % (t = budget), greedy_k={gk}, 26 circuits",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (final_mean_reduction_pct)",
            paper=old_final.get(gk, ("figure/greedy_k_reduction.pdf", None)),
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
            paper=old_time.get(gk),
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
            paper=((f"{EV}:326", "11") if gk == 1 else None),
        )
    diff = finals[2] - finals[0] if _all_known(finals[2], finals[0]) else None
    book.add(
        "numGreedyFinalDiff",
        diff,
        decimals=2,
        formula="final(greedy_k=2) - final(greedy_k=0), points (unrounded)",
        inputs=[sel[0], sel[2]],
        source="fig_curves.figure_numbers",
        paper=(f"{EV}:320-321 ('by less than 0.5\\%')", "<0.5"),
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
            paper=old_phase.get(gk),
        )
        book.add(
            f"numGreedyNotFinishedGk{DIGIT[gk]}",
            gp["n_not_finished"] if gp else None,
            kind="int",
            formula=f"circuits whose greedy phase (greedy_k={gk}) did not end within "
            "the budget",
            inputs=[sel[gk]],
            source="fig_curves.figure_numbers (greedy_phase.n_not_finished)",
            paper=((f"{EV}:331", "two") if gk == 3 else None),
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
        paper=(f"{EV}:331", "two"),
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
        paper=(f"{EV}:332", "gf2^9_mult and gf2^10_mult"),
    )
    # ---- claims ----
    book.section = "claims: greedy mode"
    sd = noise["sd_pooled"] if noise else None
    df = noise["df"] if noise else None
    book.claim(
        "greedy_final_diff_lt_half",
        f"{EV}:319-322",
        "the final gate count reductions differ by less than 0.5\\% on average "
        "(32.80\\% without greedy mode versus 33.27\\% with greedy mode for $k\\le 2$)",
        "|final(greedy_k=2) - final(greedy_k=0)| < 0.5 points",
        None if diff is None else abs(diff) < 0.5,
        {
            "gk0": finals[0],
            "gk2": finals[2],
            "diff_pp": diff,
            "noise": _noise(diff, sd, 1, df),
        },
        [sel[0], sel[2]],
    )
    book.claim(
        "greedy_gk2_above_gk0",
        f"{EV}:321-322",
        "32.80\\% without greedy mode versus 33.27\\% with greedy mode for $k\\le 2$",
        "final(greedy_k=2) > final(greedy_k=0)",
        _gt(finals[2], finals[0]),
        {"diff_pp": diff, "noise": _noise(diff, sd, 1, df)},
        [sel[0], sel[2]],
    )
    sp1 = speed.get("greedy_k=1")
    book.claim(
        "greedy_k1_reaches_threshold_much_faster",
        f"{EV}:322-326",
        "the speed of convergence varies substantially ... takes 968 seconds to reach "
        "... 32\\% ... Enabling the greedy mode for $k=1$ alone reaches the same "
        "threshold in 89 seconds, an 11$\\times$ speedup",
        f"both greedy_k=0 and greedy_k=1 reach {threshold:g}% within the budget, and "
        "the speedup t(gk0)/t(gk1) >= 2",
        None if not _all_known(times[0], times[1]) else sp1 >= 2,
        {
            "threshold_pct": threshold,
            "time_s": times,
            "speedup": sp1,
            "suggested_max_threshold": (
                math.floor(min(finals.values()))
                if None not in finals.values()
                else None
            ),
        },
        [sel[0], sel[1]],
    )
    ph = {gk: phase[gk]["mean_end_s"] if phase[gk] else None for gk in (1, 2, 3)}
    book.claim(
        "greedy_phase_grows_with_k",
        f"{EV}:328-331",
        "When the greedy mode is only enabled for $k=1$, the greedy phase takes 5.5 "
        "seconds on average. For $k \\le 2$, it takes 82 seconds ... $k \\le 3$ pushes "
        "the greedy phase to 724 seconds",
        "mean greedy-phase end increases: greedy_k=1 < 2 < 3",
        None if None in ph.values() else ph[1] < ph[2] < ph[3],
        {"mean_end_s": ph},
        [sel[1], sel[2], sel[3]],
    )
    nf = gp3["n_not_finished"] if gp3 else None
    book.claim(
        "greedy_k3_some_not_finished",
        f"{EV}:331-333",
        "two of the 26 circuits (\\texttt{gf2\\^{}9\\_mult} and "
        "\\texttt{gf2\\^{}10\\_mult}) fail to finish the greedy phase within the "
        "one-hour budget",
        "at least one circuit does not finish the greedy_k=3 greedy phase within the "
        "budget (count and list are macros)",
        None if nf is None else nf > 0,
        {
            "n_not_finished": nf,
            "circuits": nf_circuits,
            "same_as_paper": (
                set(nf_circuits) == {"gf2^9_mult", "gf2^10_mult"} if gp3 else None
            ),
            "budget_s": budget,
        },
        [sel[3]],
    )
    d32 = finals[3] - finals[2] if _all_known(finals[3], finals[2]) else None
    book.claim(
        "greedy_k3_below_k2",
        f"{EV}:334-336",
        "The resulting final reduction (33.12\\%) is also slightly below the $k \\le 2$ "
        "setting (33.27\\%)",
        "final(greedy_k=3) < final(greedy_k=2)",
        _gt(finals[2], finals[3]),
        {
            "gk2": finals[2],
            "gk3": finals[3],
            "diff_pp": d32,
            "noise": _noise(d32, sd, 1, df),
        },
        [sel[2], sel[3]],
    )
    best = (
        max(range(4), key=lambda g: finals[g]) if None not in finals.values() else None
    )
    second = (
        max((finals[g] for g in range(4) if g != best)) if best is not None else None
    )
    book.claim(
        "greedy_k2_best_of_four",
        f"{EV}:337-339",
        "With a time limit of one hour, enabling greedy mode for $k \\le 2$ strikes a "
        "balance",
        "greedy_k=2 has the highest final reduction of greedy_k in {0,1,2,3}",
        None if best is None else best == 2,
        {
            "finals": finals,
            "best": best,
            "margin_to_second_pp": finals[best] - second if best is not None else None,
            "noise": _noise(
                finals[best] - second if best is not None else None, sd, 1, df
            ),
        },
        allg,
    )


def single_k_numbers(book, fnum, derived, noise):
    book.section = "fixed exploration depth (fig:single_k)"
    sd = noise["sd_pooled"] if noise else None
    df = noise["df"] if noise else None
    old = {
        (0, 1): (f"{EV}:367", "32.83"),
        (0, 5): (f"{EV}:368", "31.46"),
        (0, 2): ("figure/single_k_gk0.pdf", "32.652 (figure only)"),
        (0, 3): ("figure/single_k_gk0.pdf", "32.528 (figure only)"),
        (0, 4): ("figure/single_k_gk0.pdf", "32.100 (figure only)"),
    }
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
                paper=old.get((gk, fk), (f"figure/single_k_gk{gk}.pdf", None)),
            )
        book.add(
            f"numFixedKGk{DIGIT[gk]}Advancing",
            adv,
            decimals=2,
            formula=f"final mean total reduction %, greedy_k={gk}, advancing k "
            "(same runs as the greedy_k figure)",
            inputs=[adv_sel],
            source="fig_curves.figure_numbers",
            paper=((f"{EV}:378", "33.27") if gk == 2 else None),
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
            paper=((f"{EV}:374-375", "3") if gk == 2 else None),
        )
        book.add(
            f"numFixedKGk{DIGIT[gk]}Best",
            finals[best] if known else None,
            decimals=2,
            formula=f"final reduction of the best fixed k, greedy_k={gk}",
            inputs=list(sel.values()),
            source="fig_curves.derived_numbers (best_fixed_k_final_pct)",
            paper=((f"{EV}:375", "33.11") if gk == 2 else None),
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
    # ---- claims ----
    book.section = "claims: fixed exploration depth"
    r0, r2 = res[0], res[2]
    f0 = r0["finals"]
    steps = (
        {f"k{k}-k{k + 1}": f0[k] - f0[k + 1] for k in range(1, 5)}
        if r0["best"] is not None
        else None
    )
    book.claim(
        "fixed_k_gk0_monotone",
        f"{EV}:365-368",
        "Without the greedy mode (\\Cref{fig:single_k_gk0}), the final average "
        "reduction decreases monotonically with $k$, from 32.83\\% at $k=1$ down to "
        "31.46\\% at $k=5$",
        "final(k) > final(k+1) for k = 1..4 (greedy_k=0)",
        None if steps is None else all(x > 0 for x in steps.values()),
        {
            "finals": f0,
            "steps_pp": steps,
            "smallest_step_noise": (
                _noise(min(steps.values()), sd, 1, df) if steps else None
            ),
        },
        list(r0["sel"].values()),
    )
    f2 = r2["finals"]
    second2 = (
        max(f2[k] for k in f2 if k != r2["best"]) if r2["best"] is not None else None
    )
    m2 = f2[r2["best"]] - second2 if second2 is not None else None
    book.claim(
        "fixed_k_gk2_k3_best",
        f"{EV}:372-375",
        "With the greedy mode enabled for $k \\le 2$ and a fixed $k$ after that ..., "
        "$k=3$ becomes the best single fixed depth (33.11\\%)",
        "argmax over fixed k of final(greedy_k=2) is 3",
        None if r2["best"] is None else r2["best"] == 3,
        {
            "finals": f2,
            "best_k": r2["best"],
            "margin_to_second_pp": m2,
            "noise": _noise(m2, sd, 1, df),
        },
        list(r2["sel"].values()),
    )
    book.claim(
        "fixed_k_ordering_shifts",
        f"{EV}:373-374",
        "the ordering shifts",
        "the best fixed k differs between greedy_k=0 and greedy_k=2",
        None if None in (r0["best"], r2["best"]) else r0["best"] != r2["best"],
        {"best_k_gk0": r0["best"], "best_k_gk2": r2["best"]},
        list(r0["sel"].values()) + list(r2["sel"].values()),
    )
    for gk, r in res.items():
        book.claim(
            f"fixed_k_gk{gk}_none_matches_advancing",
            f"{EV}:377-379",
            "Note that no single fixed $k$ matches the advancing-$k$ baseline "
            "(33.27\\%)",
            f"advancing-k final > every fixed-k final (greedy_k={gk})",
            None if r["gap"] is None else r["gap"] > 0,
            {
                "advancing": r["adv"],
                "best_fixed": r["finals"].get(r["best"]) if r["best"] else None,
                "gap_pp": r["gap"],
                "noise": _noise(r["gap"], sd, 1, df),
            },
            list(r["sel"].values()) + [r["adv_sel"]],
            note=(
                "the sentence is in the greedy_k=2 paragraph; greedy_k=0 is listed "
                "because the sentence can be read as general"
                if gk == 0
                else None
            ),
        )


def e5_numbers(book, fnum, derived, noise, finals_gk2):
    book.section = "exploration strategy (fig:exploration_ablation)"
    seeds = list(fig_curves.E5_SEEDS)
    old_nb = {
        1: ("431", "31.33"),
        2: ("433", "32.75"),
        3: ("434", "33.37"),
        4: ("437", "33.30"),
        5: ("438", "33.21"),
    }
    vals = {}
    for fig, stem in (("N_pool", "Npool"), ("N_branch", "Nbranch")):
        vals[fig] = {}
        for i in range(1, 6):
            key = f"{fig}={i}"
            vals[fig][i] = _curve(fnum, fig, key)
            paper = None
            if fig == "N_branch":
                paper = (f"{EV}:{old_nb[i][0]}", old_nb[i][1])
            elif i in (1, 4):
                paper = (
                    f"figure/N_pool_arith_mean_3600s_plain.pdf; {EV}:419 (range end)",
                    {1: "33.16", 4: "33.03"}[i],
                )
            book.add(
                f"num{stem}{DIGIT[i]}",
                vals[fig][i],
                decimals=2,
                formula=f"final mean total reduction %, {key}, mean over seeds {seeds}",
                inputs=[f"curve:{fig}:{key}"],
                source="fig_curves.figure_numbers (final_mean_reduction_pct)",
                paper=paper,
            )
    pool_sel = [f"curve:N_pool:N_pool={i}" for i in range(1, 6)]
    br_sel = [f"curve:N_branch:N_branch={i}" for i in range(1, 6)]
    e5_sel = pool_sel + br_sel
    dp = derived.get("N_pool", {})
    for name, key, old in (
        ("numNpoolMin", "mean_final_min_pct", "33.03"),
        ("numNpoolMax", "mean_final_max_pct", "33.16"),
        ("numNpoolSpread", "mean_final_spread_pct", "0.13"),
    ):
        book.add(
            name,
            dp.get(key),
            decimals=2,
            formula=f"{key} over N_pool in 1..5 (seed-averaged finals)",
            inputs=pool_sel,
            source="fig_curves.derived_numbers",
            paper=(f"{EV}:418-419", old),
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
        paper=(f"{EV}:419-420 ('well within run-to-run variance')", None),
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
    seed0 = None
    try:
        seed0 = fnum["N_branch"]["curves"]["N_branch=3"]["per_seed"]["0"][
            "final_mean_reduction_pct"
        ]
    except KeyError:
        pass
    # ---- claims ----
    book.section = "claims: exploration strategy"
    p, b = vals["N_pool"], vals["N_branch"]
    p_ok = None not in p.values()
    b_ok = None not in b.values()
    spread = dp.get("mean_final_spread_pct")

    def anova(fig):
        groups = []
        for i in range(1, 6):
            c = fnum.get(fig, {}).get("curves", {}).get(f"{fig}={i}")
            if not c:
                return None
            groups.append(
                [d["final_mean_reduction_pct"] for d in c["per_seed"].values()]
            )
        if min(len(g) for g in groups) < 2:
            return None
        return stats.f_oneway(*groups).pvalue

    p_anova, b_anova = anova("N_pool"), anova("N_branch")
    book.claim(
        "npool_no_meaningful_effect",
        f"{EV}:416-420; {EV}:410-411 (caption)",
        "Pool size $N_\\text{pool}$ has no meaningful effect on optimization quality: "
        "... ranges from 33.03\\% to 33.16\\% (a spread of 0.13\\%), well within "
        "run-to-run variance",
        "spread of the seed-averaged N_pool finals < Tukey HSD (\\numSeedHsd) from the "
        "pooled seed SD",
        None if not _all_known(spread, crit) else spread < crit,
        {
            "finals": p,
            "spread_pp": spread,
            "hsd_pp": crit,
            "sd_pooled": noise["sd_pooled"] if noise else None,
            "anova_p": p_anova,
        },
        pool_sel + br_sel,
    )
    bspread = derived.get("N_branch", {}).get("mean_final_spread_pct")
    book.claim(
        "nbranch_material_effect",
        f"{EV}:425-426",
        "Branch factor $N_\\text{branch}$, in contrast, materially affects final "
        "quality",
        "spread of the seed-averaged N_branch finals > Tukey HSD",
        None if not _all_known(bspread, crit) else bspread > crit,
        {"spread_pp": bspread, "hsd_pp": crit, "anova_p": b_anova},
        e5_sel,
    )
    arg = max(b, key=b.get) if b_ok else None
    runner = max(b[i] for i in b if i != arg) if b_ok else None
    book.claim(
        "nbranch_sweet_spot_at_3",
        f"{EV}:411-412 (caption); {EV}:426-427; {EV}:434",
        "$N_\\text{branch}$ exhibits a sweet spot at $N_\\text{branch}=3$ ... peaks at "
        "33.37\\% at $N_\\text{branch}=3$",
        "argmax over N_branch of the seed-averaged final is 3",
        None if arg is None else arg == 3,
        {
            "finals": b,
            "argmax": arg,
            "margin_to_second_pp": b[arg] - runner if arg else None,
            "noise": _noise(
                b[arg] - runner if arg else None,
                noise["sd_pooled"] if noise else None,
                len(seeds),
                noise["df"] if noise else None,
            ),
        },
        br_sel,
    )
    resolved = None
    if b_ok and crit is not None:
        resolved = b[3] - max(b[2], b[4]) > crit
    book.claim(
        "nbranch_sweet_spot_resolved",
        f"{EV}:411-412, 426-427, 435-438",
        "sweet spot at 3 ... slightly degrading quality (33.30\\% and 33.21\\% at "
        "$N_\\text{branch}=4$ and $5$)",
        "final(N_branch=3) exceeds both neighbours (2 and 4) by more than the Tukey "
        "HSD, i.e. the peak is not run-to-run noise",
        resolved,
        {
            "n3_minus_n2": b[3] - b[2] if b_ok else None,
            "n3_minus_n4": b[3] - b[4] if b_ok else None,
            "hsd_pp": crit,
        },
        e5_sel,
    )
    book.claim(
        "nbranch_climbs_to_3",
        f"{EV}:431-434",
        "the average reduction drops to 31.33\\% [at 1] ... quality climbs to 32.75\\% "
        "at $N_\\text{branch}=2$ and peaks at 33.37\\% at $N_\\text{branch}=3$",
        "final(1) < final(2) < final(3), and N_branch=1 is the lowest",
        None if not b_ok else b[1] < b[2] < b[3] and b[1] == min(b.values()),
        {"finals": b},
        br_sel,
    )
    book.claim(
        "nbranch_degrades_after_3",
        f"{EV}:435-438",
        "Beyond this point ... slightly degrading quality (33.30\\% and 33.21\\% at "
        "$N_\\text{branch}=4$ and $5$)",
        "final(3) > final(4) > final(5)",
        None if not b_ok else b[3] > b[4] > b[5],
        {"finals": b},
        br_sel,
    )
    pool_best = max(p.values()) if p_ok else None
    book.claim(
        "default_config_justified",
        f"{EV}:438-440",
        "We therefore use $N_\\text{pool}=1$ and $N_\\text{branch}=3$ as the default "
        "configuration for all other experiments",
        "N_branch=3 is the argmax of N_branch AND N_pool=1 is within Tukey HSD of the "
        "best N_pool",
        (
            arg == 3 and pool_best - p[1] < crit
            if p_ok and b_ok and crit is not None
            else None
        ),
        {"npool1_gap_to_best_pp": pool_best - p[1] if p_ok else None, "hsd_pp": crit},
        e5_sel,
    )
    # the default configuration's seed-0 run is the same job in E2, E3, E4 and E5
    adv2 = finals_gk2
    book.claim(
        "default_config_one_value",
        f"{EV}:322,335,378 vs {EV}:419,434",
        "33.27\\% (greedy, single-k sections) vs 33.37\\% (N_branch=3) vs ~33.16\\% "
        "(N_pool=1): the default configuration should give one value",
        "the greedy_k=2 / advancing-k final equals the seed-0 final of the E5 "
        "default configuration (same runs); E5 prints the mean over seeds",
        None if not _all_known(adv2, seed0) else abs(adv2 - seed0) < 1e-9,
        {
            "greedy_k2_final": adv2,
            "e5_default_seed0_final": seed0,
            "e5_default_mean_over_seeds": b.get(3),
        },
        ["curve:greedy_k:greedy_k=2", "curve:N_branch:N_branch=3"],
        note="E5 values in the text are means over seeds, so \\numNbranchThree and "
        "\\numGreedyFinalGkTwo differ by design; say so in the text",
    )


def config_claims(book, qcfg):
    """Claims about which runs the paper's 'default configuration' refers to."""
    book.section = "claims: configuration and coverage"
    book.claim(
        "qalm_is_described_default",
        f"{EV}:438-440; texts/3-method.tex:122-125",
        "We therefore use $N_\\text{pool}=1$ and $N_\\text{branch}=3$ as the default "
        "configuration for all other experiments",
        "the main-comparison QALM (--qalm-config) is configuration a (greedy k<=2, "
        "N_pool=1, N_branch=3, advancing k, no enqueue), i.e. the one the paper "
        "describes",
        qcfg == "a",
        {
            "qalm_config": qcfg,
            "selection": {
                "optimizer": tables.QALM_CONFIGS[qcfg][0],
                "params": tables.QALM_CONFIGS[qcfg][1],
            },
        },
        ["avg:qalm"],
    )
    a, b = book.selections["ete:qalm"], book.selections["curve:greedy_k:greedy_k=2"]
    book.claim(
        "table_qalm_is_ablation_default",
        f"{EV}:178-179; {EV}:378",
        "The following ablation studies use the same circuits as "
        "\\Cref{tab:eval_end_to_end} (and the advancing-$k$ baseline is the "
        "table's \\optName{})",
        "the QALM column of tab:eval_end_to_end and the greedy_k=2 / advancing-k "
        "curve are the same runs (same job ids)",
        (
            a["job_ids_sha256"] == b["job_ids_sha256"]
            if a["job_ids_sha256"] and b["job_ids_sha256"]
            else None
        ),
        {"ete_qalm": a["job_ids_sha256"], "greedy_k2": b["job_ids_sha256"]},
        ["ete:qalm", "curve:greedy_k:greedy_k=2"],
    )
    s248 = {k: v for k, v in book.selections.items() if k.startswith("avg:")}
    book.claim(
        "every_column_all_248",
        "texts/0-abstract.tex:38; texts/5-eval.tex:26-28; figtex/averages_table.tex:22",
        "In a comprehensive evaluation across 248 circuits ... Per-metric averages "
        "across the benchmark suite",
        "every column of tab:avg-comparison has a result for every S248 circuit",
        all(v["complete"] for v in s248.values()),
        {k: v["n"] for k, v in s248.items()},
        list(s248),
    )
    curves = {k: v for k, v in book.selections.items() if k.startswith("curve:")}
    book.claim(
        "ablations_all_26",
        f"{EV}:178-179, 358, 409-410; figure titles",
        "averaged over 26 benchmark circuits",
        "every ablation curve covers all 26 circuits (and all seeds)",
        all(v["complete"] for v in curves.values()),
        {k: v["n"] for k, v in curves.items()},
        list(curves),
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


def cpu_models():
    out = set()
    for p in sorted(glob.glob(os.path.join(REPRO, "env", "*.txt"))):
        with open(p) as f:
            for line in f:
                if line.startswith("Model name:"):
                    out.add(line.split(":", 1)[1].strip())
    return sorted(out)


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
        paper=(
            "texts/0-abstract.tex:38; texts/1-introduction.tex:121; " f"{EV}:27,40",
            "248",
        ),
    )
    src_dirs = {os.path.dirname(r["path"]) for r in rows248}
    removed, n_src = None, None
    if len(src_dirs) == 1 and os.path.isdir(next(iter(src_dirs))):
        names = {
            f[: -len(".qasm")]
            for f in os.listdir(next(iter(src_dirs)))
            if f.endswith(".qasm")
        }
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
        paper=(f"{EV}:28", "250"),
    )
    book.add(
        "numSuiteRemovedCircuits",
        len(removed) if removed is not None else None,
        kind="int",
        formula=f"source circuits not in S248 ({removed})",
        inputs=["inputs:S248_source", "manifest:S248"],
        source="numbers.setup_numbers",
        paper=(f"{EV}:28-29 (qft_10, qft_16)", "2"),
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
        paper=(f"{EV}:40", "five"),
    )
    book.claim(
        "five_orders_of_magnitude",
        f"{EV}:39-40",
        "The 248 circuits span five orders of magnitude in gate count.",
        "10^5 <= max/min original total gate count < 10^6",
        5 <= orders < 6,
        {"min_total": min(tot), "max_total": max(tot), "log10_ratio": orders},
        ["manifest:S248"],
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
    status = Counter(r["status"] for r in rows)
    mem_limited = sum(status.get(s, 0) for s in ("OOM_CAP", "JVM_OOM"))
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
        paper=(f"{EV}:86", "32"),
    )
    book.add(
        "numJobCores",
        _parse_cpus(camp["cores"]["jobs"]),
        kind="int",
        formula=f"job cores {camp['cores']['jobs']} (one job per physical core)",
        inputs=["config:campaign"],
        source="repro/config/campaign.yaml cores.jobs",
    )
    book.claim(
        "memory_never_limiting",
        f"{EV}:85-86",
        "Each optimizer is given one CPU core, 32 GiB of RAM",
        f"no reported run hit its memory limit (OOM_CAP / JVM_OOM) and the largest "
        f"RSS is below {cap} GiB",
        (mem_limited == 0 and max(rss) < cap) if rss else None,
        {
            "status_counts": dict(sorted(status.items())),
            "max_rss_gib": max(rss) if rss else None,
        },
        ["runs:all", "config:campaign"],
    )
    models = cpu_models()
    book.claim(
        "cpu_model",
        f"{EV}:84-85",
        "All experiments were conducted on servers equipped with AMD EPYC 7763 "
        "processors.",
        "every host snapshot in repro/env reports 'AMD EPYC 7763 64-Core Processor'",
        (models == ["AMD EPYC 7763 64-Core Processor"]) if models else None,
        {"models": models, "hosts_in_runs_csv": hosts},
        ["runs:all"],
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
            paper=(
                (f"{EV}:84-86 (setup paragraph; plan: 'about N concurrent jobs')", None)
                if name == "numMeanConcurrent"
                else None
            ),
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
        if m["paper"] and m["paper"]["value"] is not None:
            note += f" [paper: {m['paper']['value']} at {m['paper']['location']}]"
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
    greedy_numbers(book, fnum, derived, a.threshold, a.budget, noise)
    single_k_numbers(book, fnum, derived, noise)
    e5_numbers(book, fnum, derived, noise, _curve(fnum, "greedy_k", "greedy_k=2"))
    config_claims(book, a.qalm_config)

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
        "n_claims": len(book.claims),
        "n_claims_false": sum(c["result"] is False for c in book.claims),
        "n_claims_undetermined": sum(c["result"] is None for c in book.claims),
        "curve_data_issues": curve_issues,
        "noise_e5": noise,
        "conventions": [
            "values are unrounded; differences are taken before rounding",
            "printed: fixed decimals as in the current paper; times and speedups with "
            "1 decimal below 10, else 0; a trailing '?' marks a provisional value",
            "claims: result null = undetermined (missing data); 'noise' fields give "
            "z = difference / (pooled E5 seed SD * sqrt(2/n_runs_per_mean))",
            "paper locations refer to the paper sources as of the 2026-10-30 audit",
        ],
    }
    os.makedirs(a.out, exist_ok=True)
    out = {
        "meta": meta,
        "selections": book.selections,
        "macros": book.macros,
        "claims": book.claims,
        "problems": book.errors,
        "warnings": book.warnings,
    }
    jpath = os.path.join(a.out, "numbers.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    tpath = os.path.join(a.out, "numbers.tex")
    write_tex(book, tpath, meta)
    false = [c["id"] for c in book.claims if c["result"] is False]
    und = [c["id"] for c in book.claims if c["result"] is None]
    print(
        f"{len(book.macros)} macros ({n_prov} provisional, {n_missing} without data), "
        f"{len(book.claims)} claims ({len(false)} false, {len(und)} undetermined)"
    )
    if false:
        prov = {c["id"] for c in book.claims if c["provisional"]}
        print(
            "claims that do not hold:",
            ", ".join(f"{c}{' (provisional)' if c in prov else ''}" for c in false),
        )
    print(f"wrote {jpath}\nwrote {tpath}")


if __name__ == "__main__":
    main()
