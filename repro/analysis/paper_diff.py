"""Old paper vs. regenerated results: every audited claim and every macro, with a cause.

Inputs
    --old-paper DIR   the paper before the patch (wording of the claims, the
                      old tab:eval_end_to_end cells).
    --numbers FILE    numbers.json written by repro/analysis/numbers.py
                      (macros with unrounded values and the old paper value
                      they replace; claims with their check result).
    --audit FILE      the audit of the old paper (default
                      repro/reference/audit_20261030/audit_results.json); the
                      reviewed ("verified") claim lists of items A1..F are used.
    --tables-dir DIR  where the generated eval_end_to_end.tex is (default: the
                      directory of --numbers), for the per-cell comparison.

Output
    --out FILE        a Markdown report (paper_diff.md) with
                      1. the qualitative claims of numbers.json that no longer
                         hold (false) or cannot be decided (missing data),
                         with the paper's current wording, for rewriting;
                      2. every audit claim (A1.0 ... F.36): old value, new
                         value, delta, cause and a one-line reason;
                      3. every macro of numbers.json: the same columns;
                      4. the cells of tab:eval_end_to_end that changed.

Old value = the value printed in the old paper (audit ``paper_value``,
macro ``paper.value``).  New value = the printed macro text (the unrounded
value is used for the delta).  Delta = new - old in the unit of the value
(percentage points for percentages).

Cause: one of CAUSES, a best-effort assignment from what is known about the
old runs (repro/reference/audit_20261030/qalm_repro_audit.md) and the
re-run's changes (repro/README.md); see ``macro_cause`` for the rules.  A
26-circuit curve value whose change is within 2 SDs of the difference
(pooled E5 seed SD \\numSeedSdPooled times sqrt(2) for a single run, times
sqrt(1 + 1/n_seeds) for an E5 seed mean, the old value being a single run)
is put down to noise.  Unclear cases default to "newly
run" (no old raw data) or "noise".  "unchanged" marks values printed exactly
as before.

CLI (from the repo root):
    python3 repro/analysis/paper_diff.py --old-paper <paper checkout> \\
        --numbers repro/analysis/out/numbers.json --out paper_diff.md
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
DEFAULT_AUDIT = os.path.join(REPRO, "reference", "audit_20261030", "audit_results.json")

NOISE = "noise"
MEMORY = "memory limit 8 GB -> 32 GiB"
CONFIG = "QALM configuration a/b"
HARNESS = "baseline harness fixes"
REGIME = "load/frequency regime"
CONVENTION = "metric/convention fixes"
NEWRUN = "newly run"
CAUSES = {
    NOISE: "run-to-run variation of time-limited searches (E5 measures it)",
    MEMORY: "old runs had an 8 GB limit (QALM 1 h: 3 circuits stopped at 8 GB); now"
    " up to 32 GiB with smaller caps escalated",
    CONFIG: "old 1 h main run used configuration b (N_branch=1, fixed k=3, enqueue),"
    " the old 1 min run and the ablations configuration a",
    HARNESS: "GUOQ resynthesis busy-wait removed, one BQSKit worker, explicit --seed",
    REGIME: "CPU boost off (fixed frequency), strict one-core isolation (SMT sibling"
    " idle)",
    CONVENTION: "244 -> 248 circuits, failures counted, unrounded deltas, exact"
    " step-function curves, fidelity ties in log space, one gate counter",
    NEWRUN: "no raw data of the old run existed; the value is from a new run",
}
UNCHANGED = "unchanged"
WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}

# Columns of the averages table: macro suffix -> rule key
COLS = {
    "Qalm": "qalm",
    "QalmOneMin": "qalm_1min",
    "GuoqTwoQ": "guoq_2q",
    "GuoqTotal": "guoq_total",
    "QuesoTwoQ": "queso_2q",
    "QuesoTotal": "queso_total",
    "Voqc": "voqc",
    "Quartz": "quartz",
}
# tab:eval_end_to_end columns in table order (after the circuit name)
ETE_COLS = ["orig", "voqc", "queso", "guoq", "quartz_nopp", "quartz_pp", "qalm"]
ETE_SUFFIX = {
    "Voqc": "voqc",
    "Queso": "queso",
    "Guoq": "guoq",
    "QuartzNoPP": "quartz_nopp",
    "QuartzPP": "quartz_pp",
    "Qalm": "qalm",
}
E5_PREFIXES = ("numNpool", "numNbranch")
# averages-table column key -> label as latex_plain prints it
COL_LABELS = {
    "guoq_2q": "GUOQ (2Q Gate)",
    "guoq_total": "GUOQ (Total Gate)",
    "queso_2q": "QUESO (2Q Gate)",
    "queso_total": "QUESO (Total Gate)",
    "voqc": "VOQC",
    "quartz": "Quartz",
}
CURVE_PCT = re.compile(
    r"^num(?:GreedyFinalGk\w+|FixedKGk(?:Zero|Two)K\w+|FixedKGk(?:Zero|Two)Advancing"
    r"|FixedKGk(?:Zero|Two)Best|Npool(?:One|Two|Three|Four|Five|Min|Max)"
    r"|Nbranch(?:One|Two|Three|Four|Five))$"
)


# ---------------------------------------------------------------------------
# Causes
# ---------------------------------------------------------------------------
def column_cause(col, ctx, what="column"):
    """Cause of a change in one column of the 248-circuit comparison."""
    cfg = ctx["cfg"]
    if col == "qalm":
        if cfg == "a":
            return CONFIG, "old 1 h QALM run used configuration b; this one uses a"
        return MEMORY, (
            "same configuration b as the old 1 h run, which had an 8 GB limit"
            " (grover_pigeon3, rd73_252, misex1_241 stopped at it)"
        )
    if col == "qalm_1min":
        if cfg == "a":
            return REGIME, (
                "same configuration a as the old 1 min run; 60 s now run on a"
                " fixed-frequency, isolated core"
            )
        return CONFIG, "old 1 min run used configuration a; this one uses b"
    if col in ("guoq_2q", "queso_2q"):
        return HARNESS, (
            "old container spun a busy-waiting resynthesis server on the run's"
            " core; now one BQSKit worker, no busy-wait, explicit --seed"
        )
    if col in ("guoq_total", "queso_total"):
        return NEWRUN, f"no raw data of the old {what} existed"
    if col == "voqc":
        return (
            NEWRUN,
            f"no raw data of the old {what} existed (it covered 244 circuits)",
        )
    if col == "quartz":
        return REGIME, (
            "same tool and inputs (the old raw data reproduced the old value);"
            " boost off and core isolation change how far the 1 h search gets"
        )
    raise KeyError(col)


def win_cause(col, ctx):
    """Cause of a change in a per-circuit win rate of QALM against ``col``."""
    if col in ("guoq_total", "queso_total", "voqc"):
        return NEWRUN, f"the {col} column was newly run"
    if ctx["cfg"] == "a":
        return CONFIG, "the QALM side changed from configuration b to a"
    return column_cause(col, ctx)


def ete_cause(col, ctx):
    """Cause of a change in a tab:eval_end_to_end column (26 circuits)."""
    if col == "orig":
        return CONVENTION, "original counts now from one gate counter over the inputs"
    if col == "voqc":
        return CONVENTION, (
            "VOQC is deterministic; counts now from one gate counter (dropped"
            " rz(0) counts as a reduction for every tool)"
        )
    if col in ("queso", "guoq"):
        return HARNESS, "explicit --seed, no busy-waiting resynthesis server"
    if col == "quartz_nopp":
        return REGIME, "same tool and inputs; boost off and core isolation"
    return NEWRUN, "no raw data of the old column existed"


def macro_cause(name, m, ctx):
    """(cause, reason) for one macro whose printed value changed."""
    rx = re.match(r"^num(Fid|CxRed|TotalRed)(\w+)$", name)
    if rx and rx.group(2) in COLS:
        col = COLS[rx.group(2)]
        if name == "numFidQalm":
            cause, why = column_cause(col, ctx)
            return CONVENTION, (
                "old cell averaged 244 circuits (the old run's 248-circuit mean was"
                f" 0.3799); also: {why}"
            )
        return column_cause(col, ctx)
    rx = re.match(r"^numFail(ETE)?(\w+)$", name)
    if rx:
        return CONVENTION, "failed runs are now counted and disclosed (new number)"
    if name.startswith("numTotalRedMargin"):
        strongest = ctx["macros"].get("numStrongestTotalRedBaseline", {})
        return NEWRUN, (
            "the strongest baseline by total reduction (old: VOQC over 244 circuits,"
            f" no raw data) was newly run; now {latex_plain(strongest.get('printed'))}"
        )
    if name.startswith("numOneMin"):
        return CONFIG, (
            "the old drop compared a configuration-b 1 h run with a configuration-a"
            f" 1 min run; both now use configuration {ctx['cfg']}"
        )
    rx = re.match(r"^num(?:Strict)?Win(Fid|Cx|Total)(\w+)$", name)
    if rx and rx.group(2) in COLS:
        cause, why = win_cause(COLS[rx.group(2)], ctx)
        if rx.group(1) == "Fid":
            why += "; fidelity ties now compared in log space"
        return cause, why
    if name == "numWinFidStrongest":
        strongest = latex_plain(ctx["macros"]["numStrongestFidBaseline"]["printed"])
        col = next((c for c, lab in COL_LABELS.items() if lab == strongest), None)
        if col is None:
            return NEWRUN, f"the strongest baseline is now {strongest}"
        cause, why = win_cause(col, ctx)
        if col == "guoq_2q":
            return cause, why + "; still against GUOQ (2Q Gate)"
        return cause, why + f"; the strongest baseline is now {strongest}"
    if name in ("numWinMinOthers", "numStrictWinFidMin"):
        return NEWRUN, "minimum over columns that include newly run baselines"
    if name.startswith("numNegTotalGuoq"):
        return HARNESS, "GUOQ (2Q) runs without the busy-waiting resynthesis server"
    if name == "numNegTotalQalm":
        return column_cause("qalm", ctx)
    if name.startswith("numStrongest"):
        return NEWRUN, "the ranking now includes the newly run baseline columns"
    rx = re.match(r"^numGeoRed(\w+)$", name)
    if rx:
        col = ETE_SUFFIX[rx.group(1)]
        if col == "qalm":
            return noise_or(name, m, ctx, NEWRUN, "no raw data of the old QALM column")
        return ete_cause(col, ctx)
    rx = re.match(r"^numGeoDelta(\w+)$", name)
    if rx:
        mac = ctx["macros"]
        parts = ["numGeoRedQalm", "numGeoRed" + rx.group(1)]
        if all(name_status(p, mac[p])[0] == UNCHANGED for p in parts):
            return CONVENTION, (
                "both geomeans print as before; the delta is now taken before"
                " rounding"
            )
        return NEWRUN, "the QALM column of the table was newly run"
    if name.startswith("numETE"):
        return NEWRUN, "the QALM column of the table was newly run"
    if CURVE_PCT.match(name):
        return noise_or(name, m, ctx, NEWRUN, "no raw data of the old ablation runs")
    if name.startswith(("numGreedyTimeToThreshold", "numGreedySpeedup")):
        return NEWRUN, (
            "no raw data of the old ablation runs; time-to-X now from the exact"
            " step-function curve"
        )
    if name.startswith(("numGreedy", "numFixedK", "numNpool", "numNbranch")):
        return NEWRUN, "no raw data of the old ablation runs"
    if name.startswith("numSeed"):
        return NEWRUN, "run-to-run noise was not measured before"
    if name in (
        "numMemCapGiB",
        "numPeakRssGiB",
        "numPeakCgroupGiB",
        "numMemEscalations",
    ):
        return MEMORY, "old runs used 8 GB; now up to 32 GiB with escalation"
    if name in (
        "numSuiteCircuits",
        "numSuiteSourceCircuits",
        "numSuiteRemovedCircuits",
        "numGateCountOrders",
        "numQuartzSuiteCircuits",
        "numBudgetFactor",
    ):
        return CONVENTION, "suite and budget definitions (repro/inputs, configs)"
    return NEWRUN, "campaign bookkeeping, new in this run"


def noise_or(name, m, ctx, cause, why):
    """NOISE if a 26-circuit percentage moved by less than 2 SDs of the difference.

    The old value is a single run; the new one a single run (SD of the
    difference sd*sqrt(2)) or, for E5 (N_pool / N_branch), the mean over
    n seeds (sd*sqrt(1 + 1/n)).
    """
    sd = ctx.get("sd")
    old = parse_old(m)
    if sd and old is not None and m.get("value") is not None:
        delta = m["value"] - old
        n_new = (ctx.get("n_seeds") or 1) if name.startswith(E5_PREFIXES) else 1
        z = delta / (sd * math.sqrt(1 + 1 / n_new))
        if abs(z) < 2:
            return NOISE, (
                f"|delta| = {abs(delta):.2f} pp is within run-to-run noise"
                f" (z = {z:+.1f}; SD of a 26-circuit mean = {sd:.2f} pp)"
            )
        return (
            cause,
            f"{why}; |delta| = {abs(delta):.2f} pp exceeds noise (z = {z:+.1f})",
        )
    return cause, why


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def latex_plain(text):
    """Readable form of a macro's printed LaTeX (\\guoq{} -> GUOQ, ...)."""
    if text is None:
        return ""
    t = text.replace("\\ensuremath{-}", "-")
    for cs, word in (
        ("guoq", "GUOQ"),
        ("queso", "QUESO"),
        ("voqc", "VOQC"),
        ("quartz", "Quartz"),
        ("optName", "QALM"),
    ):
        t = t.replace("\\" + cs + "{}", word).replace("\\" + cs, word)
    t = re.sub(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", t)
    return t.replace("\\_", "_").replace("\\^{}", "^").replace("\\%", "%")


def parse_old(m):
    """The old paper value of a macro as a number, or None."""
    paper = m.get("paper") or {}
    v = paper.get("value")
    if v is None:
        return None
    v = str(v).strip()
    hit = re.match(r"^(?:implied\s+)?(-?\d+(?:\.\d+)?)\b", v)
    if hit:
        return float(hit.group(1))
    if v.lower() in WORDS:
        return float(WORDS[v.lower()])
    hit = re.search(r"figure: lowest (-?\d+(?:\.\d+)?)", v)
    if hit:
        return float(hit.group(1))
    return None


def new_number(m):
    t = latex_plain(m.get("printed", "")).rstrip("?")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return float(t)
    if t.lower() in WORDS:
        return float(WORDS[t.lower()])
    return None


def norm_text(t):
    # "GUOQ (2Q)" in the old paper is the column "GUOQ (2Q Gate)"
    return re.sub(r"[^a-z0-9]", "", latex_plain(t).lower()).replace("gate", "")


def name_status(name, m):
    """(status, delta) of one macro: UNCHANGED / 'changed' / 'new' / 'missing'."""
    printed = m.get("printed", "")
    if "??" in printed:
        return "missing", None
    old_raw = (m.get("paper") or {}).get("value")
    if old_raw is None:
        return "new", None
    old = parse_old(m)
    new = new_number(m)
    if old is not None and new is not None:
        threshold_like = str(old_raw).strip()[:1] in "<>"
        if threshold_like:
            return "changed", None
        dec = m.get("decimals")
        same = (
            f"{old:.{dec}f}" == f"{new:.{dec}f}" if isinstance(dec, int) else old == new
        )
        value = m.get("value") if m.get("value") is not None else new
        return (UNCHANGED if same else "changed"), value - old
    if norm_text(str(old_raw)) == norm_text(printed):
        return UNCHANGED, None
    return "changed", None


def fmt_delta(delta, m):
    if delta is None:
        return "—"
    dec = m.get("decimals")
    dec = dec if isinstance(dec, int) else 0
    if m.get("kind") == "int":
        dec = 0
    s = f"{delta:+.{dec}f}"
    return "0" if float(s) == 0 else s.replace("-", "−")


def macro_row(name, m, ctx):
    status, delta = name_status(name, m)
    paper = m.get("paper") or {}
    old = paper.get("value")
    if status == UNCHANGED:
        cause, why = UNCHANGED, "printed as before"
        if name == "numMemCapGiB":
            why = "printed as before, but the old runs actually used 8 GB"
    elif status == "missing":
        cause, why = "—", "no data yet"
    else:
        cause, why = macro_cause(name, m, ctx)
        if status == "new":
            why = "not in the old paper; " + why
        if m.get("provisional") or m.get("printed", "").endswith("?"):
            why += " (provisional: incomplete data)"
    return {
        "name": name,
        "old": "—" if old is None else str(old),
        "where": paper.get("location") or "",
        "new": m.get("printed", ""),
        "delta": fmt_delta(delta, m),
        "status": status,
        "cause": cause,
        "reason": why,
    }


# ---------------------------------------------------------------------------
# Audit claims -> macros / claims.  (key, expected text, macros, claims, cause)
# cause None: from the first changed macro (or UNCHANGED); a (cause, reason)
# tuple: fixed; "table:<col>": the tab:eval_end_to_end column comparison.
# ---------------------------------------------------------------------------
def _avg(stem_suffixes):
    return [f"num{s}" for s in stem_suffixes]


_U = (UNCHANGED, "setup or method statement, not changed by the re-run")
_OUT = (UNCHANGED, "not in the compiled paper (comment block); out of scope")
AUDIT_MAP = {
    "A1.0": ("QALM (1h): mean fidelity", ["numFidQalm"], [], None),
    "A1.1": ("QALM (1h): mean 2Q", ["numCxRedQalm"], [], None),
    "A1.2": ("QALM (1h): mean total", ["numTotalRedQalm"], [], None),
    "A1.3": ("QALM (1 minute): mean fidelity", ["numFidQalmOneMin"], [], None),
    "A1.4": ("QALM (1 minute): mean 2Q", ["numCxRedQalmOneMin"], [], None),
    "A1.5": ("QALM (1 minute): mean total", ["numTotalRedQalmOneMin"], [], None),
    "A1.6": ("GUOQ (2Q Gate mode", ["numFidGuoqTwoQ"], [], None),
    "A1.7": ("GUOQ (2Q Gate): mean 2Q", ["numCxRedGuoqTwoQ"], [], None),
    "A1.8": ("GUOQ (2Q Gate): mean total", ["numTotalRedGuoqTwoQ"], [], None),
    "A1.9": ("GUOQ (Total Gate mode", ["numFidGuoqTotal"], [], None),
    "A1.10": ("GUOQ (Total Gate): mean 2Q", ["numCxRedGuoqTotal"], [], None),
    "A1.11": ("GUOQ (Total Gate): mean total", ["numTotalRedGuoqTotal"], [], None),
    "A1.12": ("QUESO (2Q Gate mode", ["numFidQuesoTwoQ"], [], None),
    "A1.13": ("QUESO (2Q Gate): mean 2Q", ["numCxRedQuesoTwoQ"], [], None),
    "A1.14": ("QUESO (2Q Gate): mean total", ["numTotalRedQuesoTwoQ"], [], None),
    "A1.15": ("QUESO (Total Gate mode", ["numFidQuesoTotal"], [], None),
    "A1.16": ("QUESO (Total Gate): mean 2Q", ["numCxRedQuesoTotal"], [], None),
    "A1.17": ("QUESO (Total Gate): mean total", ["numTotalRedQuesoTotal"], [], None),
    "A1.18": ("VOQC: mean fidelity", ["numFidVoqc"], [], None),
    "A1.19": ("VOQC: mean 2Q", ["numCxRedVoqc"], [], None),
    "A1.20": ("VOQC: mean total", ["numTotalRedVoqc"], [], None),
    "A1.21": ("Quartz (no preprocessing): mean fidelity", ["numFidQuartz"], [], None),
    "A1.22": ("Quartz: mean 2Q", ["numCxRedQuartz"], [], None),
    "A1.23": ("Quartz: mean total", ["numTotalRedQuartz"], [], None),
    "A1.24": (
        "with one hour",
        ["numTotalRedMarginOneHour"],
        ["intro_margin_positive_qalm"],
        None,
    ),
    "A1.25": (
        "only one minute",
        ["numTotalRedMarginOneMin"],
        ["intro_margin_positive_qalm_1min"],
        None,
    ),
    "A1.26": (
        "reducing time 60x",
        ["numOneMinFidDropRel", "numOneMinCxRedDrop", "numOneMinTotalRedDrop"],
        ["one_minute_degrades_lt_1pct"],
        None,
    ),
    "A1.27": (
        "1-minute QALM still surpasses",
        ["numFidQalmOneMin", "numTotalRedQalmOneMin"],
        ["one_minute_beats_every_baseline"],
        None,
    ),
    "A1.28": (
        "nearly identical CX",
        ["numCxRedQalm", "numCxRedGuoqTwoQ"],
        ["guoq2q_cx_on_par"],
        None,
    ),
    "A1.29": (
        "dramatically better total",
        ["numTotalRedQalm", "numTotalRedGuoqTwoQ"],
        [
            "guoq2q_total_dramatically_better",
            "guoq2q_negative_on_many",
            "qalm_never_negative",
        ],
        None,
    ),
    "A1.30": (
        "vs VOQC similar total",
        ["numTotalRedQalm", "numTotalRedVoqc", "numCxRedQalm", "numCxRedVoqc"],
        ["voqc_similar_total", "voqc_much_greater_cx"],
        None,
    ),
    "A1.31": (
        "average fidelity 0.3864 outperforms",
        ["numFidQalm", "numFidVoqc", "numFidGuoqTwoQ"],
        ["fidelity_beats_voqc_guoq2q"],
        None,
    ),
    "A1.32": (
        "better on all three",
        [],
        ["remaining_baselines_all_three"],
        (NEWRUN, "three of the four baselines were newly run"),
    ),
    "A1.33": (
        "majority",
        ["numStrictWinFidMin"],
        ["majority_strict_fidelity"],
        None,
    ),
    "A1.34": (
        "248 circuits = GUOQ's 250",
        ["numSuiteCircuits", "numSuiteSourceCircuits", "numSuiteRemovedCircuits"],
        [],
        None,
    ),
    "A1.35": (
        "QALM configuration for the main comparison",
        [],
        ["qalm_is_described_default"],
        (CONFIG, "the main comparison is now run with the --qalm-config chosen"),
    ),
    "A1.36": (
        "every column is a per-metric average",
        ["numSuiteCircuits", "numMemCapGiB"],
        ["every_column_all_248", "memory_never_limiting"],
        (
            CONVENTION,
            "VOQC column and QALM fidelity cell were over 244 circuits; now 248"
            " (the old runs also used 8 GB, not 32 GiB)",
        ),
    ),
    "A2.0": (
        "matches or exceeds the fidelity of the strongest",
        ["numWinFidStrongest", "numWinFidGuoqTwoQ", "numStrongestFidBaseline"],
        ["abstract_strongest_baseline_fidelity"],
        None,
    ),
    "A2.1": (
        "GUOQ (2Q Gate) on 2Q-gate",
        ["numWinCxGuoqTwoQ"],
        ["guoq2q_cx_on_par"],
        None,
    ),
    "A2.2": ("GUOQ (2Q Gate) on total", ["numWinTotalGuoqTwoQ"], [], None),
    "A2.3": (
        "QUESO (2Q Gate)",
        _avg(f"Win{s}QuesoTwoQ" for s in ("Fid", "Cx", "Total")),
        [],
        None,
    ),
    "A2.4": ("Quartz", _avg(f"Win{s}Quartz" for s in ("Fid", "Cx", "Total")), [], None),
    "A2.5": (
        "GUOQ (Total Gate)",
        _avg(f"Win{s}GuoqTotal" for s in ("Fid", "Cx", "Total")),
        [],
        None,
    ),
    "A2.6": (
        "QUESO (Total Gate)",
        _avg(f"Win{s}QuesoTotal" for s in ("Fid", "Cx", "Total")),
        [],
        None,
    ),
    "A2.7": ("VOQC", _avg(f"Win{s}Voqc" for s in ("Fid", "Cx", "Total")), [], None),
    "A2.8": (
        "Win/tie split",
        _avg(
            f"StrictWin{s}{c}"
            for s in ("Fid", "Cx", "Total")
            for c in (
                "GuoqTwoQ",
                "GuoqTotal",
                "QuesoTwoQ",
                "QuesoTotal",
                "Voqc",
                "Quartz",
            )
        ),
        [],
        (
            NEWRUN,
            "strict-win shares were only drawn (hatching), never printed; see numStrictWin*",
        ),
    ),
    "A2.9": ("nearly every", ["numWinMinOthers"], ["intro_nearly_every_circuit"], None),
    "A2.10": ("majority", ["numStrictWinFidMin"], ["majority_strict_fidelity"], None),
    "A2.11": (
        "negative total gate reduction",
        ["numNegTotalGuoqTwoQ", "numNegTotalQalm"],
        ["guoq2q_negative_on_many", "qalm_never_negative"],
        None,
    ),
    "A2.12": (
        "comparison_3x3 QALM series",
        ["numFidQalm", "numCxRedQalm", "numTotalRedQalm"],
        [],
        None,
    ),
    "A2.13": ("Average QALM fidelity 0.3864", ["numFidQalm"], [], None),
    "A2.14": (
        "Quartz column",
        ["numFidQuartz", "numCxRedQuartz", "numTotalRedQuartz"],
        [],
        None,
    ),
    "A2.15": (
        "GUOQ (2Q Gate) column",
        ["numFidGuoqTwoQ", "numCxRedGuoqTwoQ", "numTotalRedGuoqTwoQ"],
        [],
        None,
    ),
    "A2.16": (
        "QUESO (2Q Gate) column",
        ["numFidQuesoTwoQ", "numCxRedQuesoTwoQ", "numTotalRedQuesoTwoQ"],
        [],
        None,
    ),
    "A2.17": (
        "GUOQ (Total Gate), QUESO (Total Gate) and VOQC columns",
        _avg(
            f"{s}{c}"
            for c in ("GuoqTotal", "QuesoTotal", "Voqc")
            for s in ("Fid", "CxRed", "TotalRed")
        ),
        [],
        None,
    ),
    "A2.18": ("suite size 248", ["numSuiteCircuits"], [], None),
    "A2.19": (
        "five orders of magnitude",
        ["numGateCountOrders"],
        ["five_orders_of_magnitude"],
        None,
    ),
    "A2.20": ("left panel", ["numSuiteCircuits"], [], None),
    "A2.21": ("right panel", ["numSuiteCircuits"], [], None),
    "A2.22": (
        "strongest baseline",
        ["numStrongestFidBaseline"],
        ["gate_scatter_strongest_baseline"],
        None,
    ),
    "A2.23": (
        "VOQC per-circuit panels",
        ["numSuiteCircuits"],
        ["every_column_all_248"],
        (CONVENTION, "the VOQC panels and average covered 244 circuits; now all 248"),
    ),
    "A2.24": ("dashed reference line", [], [], _U),
    "B.0": ("Orig. column", [], [], "table:orig"),
    "B.1": ("VOQC column", [], [], "table:voqc"),
    "B.2": ("QUESO column", [], [], "table:queso"),
    "B.3": ("GUOQ column", [], [], "table:guoq"),
    "B.4": ("Quartz w/o Preprocess column", [], [], "table:quartz_nopp"),
    "B.5": ("Quartz w/ Preprocess column", [], [], "table:quartz_pp"),
    "B.6": ("QALM column", [], [], "table:qalm"),
    "B.7": ("VOQC = 27.0%", ["numGeoRedVoqc"], [], None),
    "B.8": ("QUESO = 23.9%", ["numGeoRedQueso"], [], None),
    "B.9": ("GUOQ = 29.2%", ["numGeoRedGuoq"], [], None),
    "B.10": ("Quartz w/o Preprocess = 16.5%", ["numGeoRedQuartzNoPP"], [], None),
    "B.11": ("Quartz w/ Preprocess = 29.7%", ["numGeoRedQuartzPP"], [], None),
    "B.12": ("QALM = 34.3%", ["numGeoRedQalm"], [], None),
    "B.13": (
        "best result for each circuit is in bold",
        [],
        [],
        (NEWRUN, "bold marks recomputed by tables.py from the new cells"),
    ),
    "B.14": (
        "lowest or equally lowest",
        ["numETEQalmNotBest", "numETEQalmWorseQuartzPP"],
        ["ete_lowest_all_but_one", "ete_matches_quartz_pp_all_but_one"],
        None,
    ),
    "B.15": (
        "additional 7.3%",
        ["numGeoDeltaVoqc"],
        ["ete_outperforms_by_positive"],
        None,
    ),
    "B.16": (
        "Quartz (without preprocessing) by 17.8%",
        ["numGeoDeltaQuartzNoPP"],
        [],
        None,
    ),
    "B.17": ("QUESO by 10.4%", ["numGeoDeltaQueso"], [], None),
    "B.18": ("GUOQ by 5.1%", ["numGeoDeltaGuoq"], [], None),
    "B.19": ("Quartz (with preprocessing) by 4.6%", ["numGeoDeltaQuartzPP"], [], None),
    "B.20": ("absolute percentage", [], ["ete_outperforms_by_positive"], _U),
    "B.21": ("1-hour limit", [], [], _U),
    "B.22": ("26 benchmark circuits", ["numQuartzSuiteCircuits"], [], None),
    "C.0": ("same 26 circuits", ["numQuartzSuiteCircuits"], ["ablations_all_26"], None),
    "C.1": ("greedy_k=0) is", ["numGreedyFinalGkZero"], [], None),
    "C.2": ("greedy_k=2) is", ["numGreedyFinalGkTwo"], [], None),
    "C.3": (
        "less than 0.5%",
        ["numGreedyFinalDiff"],
        ["greedy_final_diff_lt_half"],
        None,
    ),
    "C.4": ("968 s", ["numGreedyTimeToThresholdGkZero"], [], None),
    "C.5": ("89 s", ["numGreedyTimeToThresholdGkOne"], [], None),
    "C.6": (
        "11x speedup",
        ["numGreedySpeedupGkOne"],
        ["greedy_k1_reaches_threshold_much_faster"],
        None,
    ),
    "C.7": ("5.5 s", ["numGreedyPhaseMeanGkOne"], [], None),
    "C.8": ("82 s", ["numGreedyPhaseMeanGkTwo"], [], None),
    "C.9": (
        "724 s",
        ["numGreedyPhaseMeanGkThree"],
        ["greedy_phase_grows_with_k"],
        None,
    ),
    "C.10": (
        "fail to fi",
        ["numGreedyNotFinishedGkThree", "numGreedyNotFinishedGkThreeList"],
        ["greedy_k3_some_not_finished"],
        None,
    ),
    "C.11": ("33.12%", ["numGreedyFinalGkThree"], ["greedy_k3_below_k2"], None),
    "C.12": (
        "best final reduction of the four",
        ["numGreedyFinalGkTwo"],
        ["greedy_k2_best_of_four"],
        (NEWRUN, "no raw data of the old ablation runs"),
    ),
    "C.13": (
        "Experimental setup applies",
        ["numMemCapGiB"],
        ["memory_never_limiting"],
        (
            MEMORY,
            "ablations now run like the main comparison: one isolated core, boost off, up to 32 GiB",
        ),
    ),
    "C.14": ("Caption: each curve", [], [], _U),
    "D.0": ("32.83%", ["numFixedKGkZeroKOne"], [], None),
    "D.1": ("31.46%", ["numFixedKGkZeroKFive"], [], None),
    "D.2": (
        "k=2, 3, 4",
        ["numFixedKGkZeroKTwo", "numFixedKGkZeroKThree", "numFixedKGkZeroKFour"],
        [],
        None,
    ),
    "D.3": (
        "decreases monotonical",
        [],
        ["fixed_k_gk0_monotone"],
        (NEWRUN, "no raw data of the old ablation runs"),
    ),
    "D.4": (
        "k=3 is the best",
        ["numFixedKGkTwoBestK", "numFixedKGkTwoBest"],
        ["fixed_k_gk2_k3_best", "fixed_k_ordering_shifts"],
        None,
    ),
    "D.5": (
        "advancing-k baseline",
        ["numFixedKGkTwoAdvancing"],
        ["fixed_k_gk2_none_matches_advancing", "fixed_k_gk0_none_matches_advancing"],
        None,
    ),
    "D.6": (
        "equals the greedy-section",
        ["numFixedKGkTwoAdvancing", "numGreedyFinalGkTwo"],
        ["default_config_one_value"],
        None,
    ),
    "D.7": (
        "default configuration",
        ["numGreedyFinalGkTwo", "numNbranchThree", "numNpoolOne"],
        ["default_config_one_value"],
        (
            CONVENTION,
            "one seed-0 run is shared by E2-E5; E5 prints the mean over seeds 0-2 (say so in the text)",
        ),
    ),
    "D.8": ("Averaged over 26", ["numQuartzSuiteCircuits"], [], None),
    "D.9": ("1-hour budget, single fixed k", [], [], _U),
    "D.10": ("Explanation", [], [], _U),
    "E.0": (
        "N_pool in {1..5}",
        ["numNpoolMin", "numNpoolMax", "numNpoolSpread"],
        ["npool_no_meaningful_effect"],
        None,
    ),
    "E.1": ("N_pool=1", ["numNpoolOne"], [], None),
    "E.2": ("N_pool=2", ["numNpoolTwo"], [], None),
    "E.3": ("N_pool=3", ["numNpoolThree"], [], None),
    "E.4": ("N_pool=4", ["numNpoolFour"], [], None),
    "E.5": ("N_pool=5", ["numNpoolFive"], [], None),
    "E.6": ("N_branch=1", ["numNbranchOne"], [], None),
    "E.7": ("N_branch=2", ["numNbranchTwo"], [], None),
    "E.8": ("N_branch=3", ["numNbranchThree"], ["nbranch_sweet_spot_at_3"], None),
    "E.9": ("N_branch=4", ["numNbranchFour"], [], None),
    "E.10": ("N_branch=5", ["numNbranchFive"], ["nbranch_degrades_after_3"], None),
    "E.11": (
        "sweet spot at 3",
        ["numSeedHsd", "numSeedSdPooled", "numNpoolSpread"],
        [
            "nbranch_sweet_spot_at_3",
            "nbranch_sweet_spot_resolved",
            "npool_no_meaningful_effect",
        ],
        (
            NEWRUN,
            "run-to-run variance is now measured (3 seeds); the claims test against it",
        ),
    ),
    "E.12": ("single successor", [], [], _U),
    "E.13": ("vary each parameter", [], [], _U),
    "E.14": (
        "default configuration for all",
        [],
        ["default_config_justified", "qalm_is_described_default"],
        (
            CONFIG,
            "the main comparison uses the --qalm-config chosen; E3-E5 use configuration a",
        ),
    ),
    "E.15": (
        "Consistency",
        ["numGreedyFinalGkTwo", "numNbranchThree", "numNpoolOne"],
        ["default_config_one_value"],
        (CONVENTION, "one seed-0 run is shared; E5 prints the mean over seeds 0-2"),
    ),
    "F.0": ("248 circuits compiled by GUOQ", ["numSuiteCircuits"], [], None),
    "F.1": (
        "originally had 250",
        ["numSuiteSourceCircuits", "numSuiteRemovedCircuits"],
        [],
        None,
    ),
    "F.2": ("QAOA, VQE", [], [], _U),
    "F.3": ("Nam gate set", [], [], _U),
    "F.4": (
        "five orders of magnitude",
        ["numGateCountOrders"],
        ["five_orders_of_magnitude"],
        None,
    ),
    "F.5": ("Quartz v0.3.2", [], [], _U),
    "F.6": ("QUESO in two-qubit mode", [], [], (HARNESS, "same flags, plus --seed 0")),
    "F.7": (
        "QUESO in total-gate mode",
        [],
        [],
        (HARNESS, "same flags, plus --seed 0 (column newly run)"),
    ),
    "F.8": ("GUOQ in two-qubit mode", [], [], (HARNESS, "same flags, plus --seed 0")),
    "F.9": (
        "GUOQ in total-gate mode",
        [],
        [],
        (HARNESS, "same flags, plus --seed 0 (column newly run)"),
    ),
    "F.10": (
        "resynthesis only when",
        [],
        [],
        (
            HARNESS,
            "resynthesis server now started only for TWO_Q, one worker, no busy-wait",
        ),
    ),
    "F.11": (
        "Docker environments",
        [],
        [],
        (
            HARNESS,
            "GUOQ, QUESO and VOQC now run in the GUOQ authors' image (setup-docker block)",
        ),
    ),
    "F.12": ("AMD EPYC 7763", [], ["cpu_model"], _U),
    "F.13": (
        "one CPU core",
        [],
        [],
        (
            REGIME,
            "one physical core per run with its SMT sibling idle, CPU boost off",
            "one physical core per run (SMT sibling idle), boost off",
        ),
    ),
    "F.14": (
        "32 GiB of RAM",
        ["numMemCapGiB", "numPeakRssGiB", "numMemEscalations"],
        ["memory_never_limiting"],
        (
            MEMORY,
            "old runs used 8 GB (Quartz partly re-run at 32 GB); now up to 32 GiB",
        ),
    ),
    "F.15": ("one-hour time limit", ["numBudgetFactor"], [], None),
    "F.16": ("without its preprocessing", [], [], _U),
    "F.17": (
        "VOQC is run as a baseline",
        ["numFidVoqc", "numCxRedVoqc", "numTotalRedVoqc"],
        [],
        None,
    ),
    "F.18": ("Fidelity is modeled", [], [], _U),
    "F.19": (
        "constant schedule",
        [],
        ["qalm_is_described_default"],
        (CONFIG, "the old main run used configuration b; the re-run has both a and b"),
    ),
    "F.20": ("explores k steps", [], [], _U),
    "F.21": ("ROQC implements", [], [], _U),
    "F.22": ("Rotation merging", [], [], _U),
    "F.23": ("(5,3)-complete", [], [], _U),
    "F.24": ("recently modified", [], [], _U),
    "F.25": ("2,000 circuits", [], [], _U),
    "F.26": ("visited set", [], [], _U),
    "F.27": (
        "pruning does not affect",
        [],
        [],
        (UNCHANGED, "not re-run: no experiment of the campaign tests it"),
    ),
    "F.28": ("Greedy mode runs for k=1 and k=2", [], [], _U),
    "F.29": ("approximate", [], [], _U),
    "F.30": (
        "5.97%",
        ["numTotalRedMarginOneHour", "numTotalRedMarginOneMin"],
        [],
        None,
    ),
    "F.31": (
        "60x",
        ["numOneMinFidDropRel", "numOneMinCxRedDrop", "numOneMinTotalRedDrop"],
        ["one_minute_degrades_lt_1pct"],
        None,
    ),
    "F.32": (
        "With one minute",
        ["numFidQalmOneMin", "numTotalRedQalmOneMin"],
        ["one_minute_beats_every_baseline", "conclusion_one_minute_beats_search_based"],
        None,
    ),
    "F.33": ("tab:eval_2q", [], [], _OUT),
    "F.34": ("tab:quarl", [], [], _OUT),
    "F.35": ("fig:reduction_over_time", [], [], _OUT),
    "F.36": ("illustrative", [], [], _U),
}


def audit_claims(audit):
    """[(id, item, claim dict)] of the reviewed claim lists, in file order."""
    out = []
    for item in audit:
        claims = (item.get("verified") or {}).get("claims") or item["audit"]["claims"]
        prefix = item["key"].split("-")[0]
        for i, c in enumerate(claims):
            out.append((f"{prefix}.{i}", item["key"], c))
    return out


# ---------------------------------------------------------------------------
# tab:eval_end_to_end
# ---------------------------------------------------------------------------
def read_ete(path):
    """{circuit: {column: int or None}} of an eval_end_to_end.tex file."""
    rows = {}
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            m = re.match(r"^\\texttt\{(.*?)\}\s*&(.*)\\\\\s*$", line.strip())
            if not m:
                continue
            name = m.group(1).replace("\\_", "_").replace("\\^{}", "^")
            cells = [
                re.sub(r"\\textbf\{([^{}]*)\}", r"\1", c).strip()
                for c in m.group(2).split("&")
            ]
            vals = [int(c) if re.fullmatch(r"\d+", c) else None for c in cells]
            rows[name] = dict(zip(ETE_COLS, vals))
    return rows


def ete_compare(old_rows, new_rows):
    """{column: {"changed": [(circuit, old, new)], "n": int, "old_sum", "new_sum"}}."""
    out = {}
    for col in ETE_COLS:
        changed, n, so, sn = [], 0, 0, 0
        for circ, old in old_rows.items():
            o = old.get(col)
            nw = (new_rows.get(circ) or {}).get(col)
            if o is None or nw is None:
                continue
            n += 1
            so += o
            sn += nw
            if o != nw:
                changed.append((circ, o, nw))
        out[col] = {"changed": changed, "n": n, "old_sum": so, "new_sum": sn}
    return out


# ---------------------------------------------------------------------------
# Paper text
# ---------------------------------------------------------------------------
def locations(loc):
    """[(file, first, last)] for every 'file:a-b' / 'file:a,b' in a location."""
    out = []
    for m in re.finditer(r"((?:texts|figtex)/[\w.-]+\.tex):([\d,\- ]+)", str(loc)):
        for rng in m.group(2).split(","):
            rng = rng.strip()
            if not rng:
                continue
            a, _, b = rng.partition("-")
            if a.strip().isdigit():
                out.append(
                    (m.group(1), int(a), int(b) if b.strip().isdigit() else int(a))
                )
    return out


def wording(paper, loc, max_lines=10):
    """The old paper's source lines at a location string (quoted Markdown)."""
    blocks = []
    for f, a, b in locations(loc):
        path = os.path.join(paper, f)
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            lines = fh.read().split("\n")
        seg = lines[a - 1 : min(b, a - 1 + max_lines)]
        blocks.append(f"`{f}:{a}-{b}`\n\n" + "\n".join("> " + s for s in seg))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def md_cell(text, limit=None):
    t = str(text).replace("|", "\\|").replace("\n", " ")
    if limit and len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def claim_state(c):
    r = c.get("result")
    s = {True: "true", False: "false"}.get(r, "undetermined")
    return s + (" (provisional)" if c.get("provisional") else "")


def audit_row(aid, item, ac, numbers, ctx, ete):
    macros = numbers["macros"]
    claims = {c["id"]: c for c in numbers["claims"]}
    expect, names, cids, cause = AUDIT_MAP.get(aid, (None, [], [], None))
    problems = []
    if aid not in AUDIT_MAP:
        problems.append(f"{aid}: not in AUDIT_MAP")
    elif (
        expect
        and expect.lower() not in (ac["claim"] + " " + str(ac["paper_value"])).lower()
    ):
        problems.append(
            f"{aid}: expected text {expect!r} not in the audit claim (audit file changed?)"
        )
    names = [n for n in names if n in macros]
    rows = [macro_row(n, macros[n], ctx) for n in names]
    new_parts = [f"{latex_plain(macros[n]['printed'])}" for n in names[:6]]
    if len(names) > 6:
        new_parts.append(f"… ({len(names)} values)")
    new_parts += [f"`{c}`: {claim_state(claims[c])}" for c in cids if c in claims]
    # delta: from the mapped macros' own old values (numbers.json paper.value)
    deltas = [r["delta"] for r in rows if r["delta"] != "—"]
    if len(deltas) > 4:
        deltas = deltas[:4] + ["…"]
    if isinstance(cause, str) and cause.startswith("table:"):
        col = cause.split(":", 1)[1]
        cmp = ete.get(col)
        if cmp and cmp["n"]:
            k = len(cmp["changed"])
            new_parts = [
                f"{k} of {cmp['n']} cells changed; column sum {cmp['old_sum']} → {cmp['new_sum']}"
            ]
            deltas = [f"{cmp['new_sum'] - cmp['old_sum']:+d} gates (sum)"]
            cause, why = (
                (UNCHANGED, "every cell printed as before")
                if k == 0
                else ete_cause(col, ctx)
            )
        elif ctx.get("ete_new"):
            new_parts, cause, why = ["--"], "—", "no data yet (column empty)"
        else:
            new_parts, cause, why = (
                ["(table not available)"],
                "—",
                "eval_end_to_end.tex not found",
            )
    elif isinstance(cause, tuple):
        if len(cause) == 3:  # (cause, reason, new value text)
            new_parts = [cause[2]] + new_parts
        elif not new_parts and cause[0] != UNCHANGED:
            new_parts = [cause[1]]
        cause, why = cause[:2]
    else:
        changed = [r for r in rows if r["status"] not in (UNCHANGED,)]
        if changed:
            cause, why = changed[0]["cause"], changed[0]["reason"]
        elif rows:
            cause, why = UNCHANGED, "printed as before"
        elif cids and all(claims.get(c, {}).get("result") is True for c in cids):
            cause, why = UNCHANGED, "the statement still holds"
        else:
            cause, why = NEWRUN, "no raw data of the old run existed"
    false = [c for c in cids if c in claims and claims[c].get("result") is not True]
    if false and cause == UNCHANGED:
        why += f"; but {', '.join(false)} no longer holds"
    return {
        "id": aid,
        "item": item,
        "claim": ac["claim"],
        "status": ac.get("status"),
        "where": ac.get("paper_location", ""),
        "old": ac["paper_value"],
        "new": "; ".join(new_parts) if new_parts else "—",
        "delta": " / ".join(deltas) if deltas else "—",
        "cause": cause,
        "reason": why,
        "macros": names,
        "claims": cids,
    }, problems


def build(old_paper, numbers, audit, tables_dir):
    macros = numbers["macros"]
    sd_m = macros.get("numSeedSdPooled") or {}
    ctx = {
        "cfg": numbers["meta"].get("qalm_config"),
        "macros": macros,
        "sd": sd_m.get("value") if "?" not in sd_m.get("printed", "?") else None,
        "n_seeds": (macros.get("numSeeds") or {}).get("value"),
    }
    new_ete = read_ete(os.path.join(tables_dir, "eval_end_to_end.tex"))
    ctx["ete_new"] = bool(new_ete)
    ete = ete_compare(
        read_ete(os.path.join(old_paper, "figtex", "eval_end_to_end.tex")), new_ete
    )
    problems = []
    arows = []
    seen = set()
    for aid, item, ac in audit_claims(audit):
        row, probs = audit_row(aid, item, ac, numbers, ctx, ete)
        arows.append(row)
        problems += probs
        seen.add(aid)
    problems += [
        f"{k}: in AUDIT_MAP but not in the audit file"
        for k in AUDIT_MAP
        if k not in seen
    ]
    mrows = [macro_row(n, m, ctx) for n, m in macros.items()]
    return {
        "ctx": ctx,
        "audit": arows,
        "macros": mrows,
        "ete": ete,
        "problems": problems,
    }


def render(res, numbers, args):
    meta = numbers["meta"]
    claims = numbers["claims"]
    out = []
    w = out.append
    with open(args.numbers, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    w("# QALM paper: old numbers vs. the re-run\n\n")
    w(
        f"Generated by `repro/analysis/paper_diff.py` from `{args.numbers}` (sha256 {sha}),"
        f" old paper `{args.old_paper}`, audit `{os.path.relpath(args.audit, REPRO) if args.audit.startswith(REPRO) else args.audit}`.\n\n"
    )
    w(
        f"QALM configuration **{meta.get('qalm_config')}**; budgets {meta.get('budget_s')} s /"
        f" {meta.get('short_budget_s')} s; time-to-X threshold {meta.get('threshold_pct')}%;"
        f" numbers complete: **{meta.get('complete')}**"
        f" ({meta.get('n_provisional', 0)} provisional, {meta.get('n_missing', 0)} missing).\n\n"
    )
    if not meta.get("complete"):
        w(
            "> **Provisional:** numbers.json was built from incomplete results; '?' marks provisional values and '??' missing ones.\n\n"
        )
    w("## How to read\n\n")
    w(
        "- **old**: the value the old paper prints (audit `paper_value`, or the macro's"
        " `paper.value`); **new**: the macro as printed now; **Δ** = new − old, from the"
        " unrounded new value, in the unit of the value (percentage points for percentages).\n"
        "- **cause**: best-effort assignment, one of the categories below, with a one-line reason."
        " `unchanged` marks values printed exactly as before.\n"
    )
    sd = res["ctx"].get("sd")
    if sd:
        w(
            f"- Noise reference: pooled E5 seed SD of a 26-circuit mean = {sd:.3f} pp; a 26-circuit"
            f" curve value that moved by less than 2·√2·SD = {2 * math.sqrt(2) * sd:.2f} pp"
            " (2·√(1+1/n_seeds)·SD for an E5 seed mean, the old value being a single run) is put down to `noise`.\n"
        )
    w("\n| cause | meaning |\n|---|---|\n")
    for k, v in CAUSES.items():
        w(f"| {k} | {v} |\n")
    # summary
    w("\n## Summary\n\n")
    for label, rows in (("audit claims", res["audit"]), ("macros", res["macros"])):
        counts = {}
        for r in rows:
            counts[r["cause"]] = counts.get(r["cause"], 0) + 1
        w(
            f"- {label} ({len(rows)}): "
            + ", ".join(
                f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
            )
            + "\n"
        )
    changed = [c for c in claims if c.get("result") is not True]
    w(
        f"- qualitative claims that no longer hold or cannot be decided: **{len(changed)}** of {len(claims)} (section 1)\n"
    )
    if res["problems"]:
        w("\n**Mapping problems:**\n\n" + "".join(f"- {p}\n" for p in res["problems"]))
    # 1. claims
    w("\n## 1. Qualitative claims whose truth changed\n\n")
    w(
        "The paper states every one of these as true. Each sentence carries a"
        " `% REPRO-CHECK: <id> is now ...` marker after `paper_patch.py`; rewrite it by hand.\n"
    )
    rev = {}
    for r in res["audit"]:
        for c in r["claims"]:
            rev.setdefault(c, []).append(f"{r['id']} ({r['status']})")
    for c in changed:
        w(f"\n### `{c['id']}`: now {claim_state(c)}\n\n")
        w(f"- where: {c.get('paper_location')}\n")
        w(f"- claim: {c.get('wording')}\n")
        w(f"- check: {c.get('check')}\n")
        w(f"- numbers: `{json.dumps(c.get('numbers'), sort_keys=True)[:600]}`\n")
        if rev.get(c["id"]):
            w(f"- audit of the old paper: {', '.join(rev[c['id']])}\n")
        text = wording(args.old_paper, c.get("paper_location", ""))
        if text:
            w("\nCurrent wording:\n\n" + text + "\n")
    # 2. audit
    w("\n## 2. Audit claims (old paper, items A1..F)\n")
    item = None
    for r in res["audit"]:
        if r["item"] != item:
            item = r["item"]
            w(
                f"\n### {item}\n\n| id | claim | old | new | Δ | cause | reason |\n|---|---|---|---|---|---|---|\n"
            )
        w(
            f"| {r['id']} | {md_cell(r['claim'], 90)} | {md_cell(r['old'], 60)} | {md_cell(r['new'], 90)}"
            f" | {md_cell(r['delta'])} | {r['cause']} | {md_cell(r['reason'], 160)} |\n"
        )
    # 3. macros
    w("\n## 3. Macros of numbers.tex\n")
    order = []
    for r in res["macros"]:
        sec = numbers["macros"][r["name"]].get("section")
        if sec not in order:
            order.append(sec)
    rows = sorted(
        res["macros"],
        key=lambda r: order.index(numbers["macros"][r["name"]].get("section")),
    )
    section = None
    for r in rows:
        sec = numbers["macros"][r["name"]].get("section")
        if sec != section:
            section = sec
            w(
                f"\n### {section}\n\n| macro | old | where (old paper) | new | Δ | cause | reason |\n|---|---|---|---|---|---|---|\n"
            )
        w(
            f"| `\\{r['name']}` | {md_cell(r['old'], 50)} | {md_cell(r['where'], 50)} | {md_cell(latex_plain(r['new']), 60)}"
            f" | {r['delta']} | {r['cause']} | {md_cell(r['reason'], 160)} |\n"
        )
    # 4. table cells
    w("\n## 4. Table tab:eval_end_to_end: changed cells\n\n")
    if not any(v["n"] for v in res["ete"].values()):
        w("(old or new eval_end_to_end.tex not available)\n")
    for col, v in res["ete"].items():
        if not v["n"]:
            continue
        cells = ", ".join(f"{c} {o}→{n}" for c, o, n in v["changed"]) or "none"
        w(f"- **{col}** ({len(v['changed'])} of {v['n']} changed): {cells}\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-paper", required=True, help="the paper before the patch")
    ap.add_argument("--numbers", required=True, help="numbers.json of numbers.py")
    ap.add_argument("--out", required=True, help="Markdown file to write")
    ap.add_argument("--audit", default=DEFAULT_AUDIT, help="audit_results.json")
    ap.add_argument(
        "--tables-dir",
        help="dir with the new eval_end_to_end.tex (default: that of --numbers)",
    )
    args = ap.parse_args()
    with open(args.numbers) as f:
        numbers = json.load(f)
    with open(args.audit) as f:
        audit = json.load(f)
    tables_dir = args.tables_dir or os.path.dirname(os.path.abspath(args.numbers))
    res = build(args.old_paper, numbers, audit, tables_dir)
    with open(args.out, "w") as f:
        f.write(render(res, numbers, args))
    n_changed = sum(c.get("result") is not True for c in numbers["claims"])
    print(
        f"wrote {args.out}: {len(res['audit'])} audit claims, {len(res['macros'])} macros,"
        f" {n_changed} claims no longer true or undetermined"
    )
    for p in res["problems"]:
        print(f"problem: {p}", file=sys.stderr)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
