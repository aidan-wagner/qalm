"""Exact regression of the legacy adapters against known values.

Loads every legacy row with ``legacy.load_legacy()`` and recomputes, with the
shared definitions in ``metrics.py``, the numbers an earlier audit obtained
from the same old data (repro/reference/audit_20261030).  Each check compares
values rounded to the digits shown and prints PASS/FAIL; the exit status is 1
if any check fails.

Checks: per-column averages on the 248-circuit suite (QALM 1 h old config,
QALM 60 s, Quartz, GUOQ-2Q, QUESO-2Q) and their non-OK counts; QALM-1h-old vs
QUESO-2Q win-or-tie rates; the 26-circuit table's per-circuit total gate
counts against reference/eval_end_to_end_table.json (the Dec-2025 QALM column
against the paper table before commit b9d28de) and its geomean reductions; the input counter against the old CSVs' orig columns; and a
runs.csv round trip through ``metrics.load_runs``.

Run from the repo root: ``python3 repro/analysis/validate_legacy.py``
"""

import csv
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPRO, "harness"))

import legacy  # noqa: E402
import metrics  # noqa: E402
from collect import ROW_FIELDS  # noqa: E402
from qasm_count import count_file  # noqa: E402

REF_TABLE = os.path.join(
    REPRO, "reference", "audit_20261030", "eval_end_to_end_table.json"
)
CONFIG_A = dict(greedy_k=2, n_pool=1, n_branch=3, fixed_k=0, enqueue=0)

# Selection of each 248-suite column (metrics.select kwargs) ...
SELECT_248 = {
    "QALM 1h old": dict(optimizer="qalm_old", budget_s=3600),
    "QALM 60s": dict(optimizer="qalm", budget_s=60, seed=0, **CONFIG_A),
    "Quartz": dict(optimizer="quartz", budget_s=3600),
    "GUOQ-2Q": dict(optimizer="guoq", budget_s=3600, opt="TWO_Q"),
    "QUESO-2Q": dict(optimizer="queso", budget_s=3600, opt="TWO_Q"),
}
# ... and its expected (fidelity, cx_red, total_red, number of non-OK runs).
EXPECTED_248 = {
    "QALM 1h old": ("0.37987", "24.6007", "52.3359", 0),
    "QALM 60s": ("0.37470", "21.1843", "51.3484", 0),
    "Quartz": ("0.35258", "12.4179", "40.7268", 20),
    "GUOQ-2Q": ("0.34826", "25.1829", "-3.7089", 0),
    "QUESO-2Q": ("0.33606", "13.1927", "27.3651", 4),
}
WIN_QUESO = {"fidelity": "98.8", "cx": "99.6", "total": "98.8"}
# (label, select kwargs, column of eval_end_to_end_table.json, geomean reduction)
TABLE26 = [
    ("VOQC", dict(optimizer="voqc"), 1, "26.95"),
    ("QUESO", dict(optimizer="queso", opt="TOTAL"), 2, "23.95"),
    ("GUOQ", dict(optimizer="guoq", opt="TOTAL"), 3, "29.21"),
    ("Quartz w/o PP", dict(optimizer="quartz"), 4, "16.53"),
    ("QALM dec2025", dict(optimizer="qalm_dec2025"), None, "34.97"),
]
# QALM column of the 26-circuit table before paper commit b9d28de (the Dec-2025
# run; `git show b9d28de^:figtex/eval_end_to_end.tex`), in repro/inputs/S26.csv order.
OLD_PAPER_QALM = [513, 36, 62, 92, 244, 140, 280, 165, 260, 357, 487, 641, 799]
OLD_PAPER_QALM += [963, 24, 86, 177, 383, 252, 614, 152, 35, 51, 69, 159, 73]

RESULTS = []


def check(name, got, expected):
    ok = got == expected
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {expected}")


def pick(rows, suite, n, **kw):
    sel = metrics.select(rows, suite=suite, **kw)
    circuits = {r["circuit"] for r in sel}
    if len(sel) != n or len(circuits) != n:
        raise AssertionError(
            f"{kw} on {suite}: {len(sel)} rows / {len(circuits)} circuits"
        )
    return sel


def check_averages(rows):
    for label, kw in SELECT_248.items():
        fid, cx, tot, n_bad = EXPECTED_248[label]
        sel = pick(rows, "S248", 248, **kw)
        a = metrics.averages(sel)
        check(f"{label} (248) fidelity", f"{a['fidelity']:.5f}", fid)
        check(f"{label} (248) cx_red", f"{a['cx_red']:.4f}", cx)
        check(f"{label} (248) total_red", f"{a['total_red']:.4f}", tot)
        check(
            f"{label} (248) non-OK runs", sum(r["status"] != "OK" for r in sel), n_bad
        )
    qalm = pick(rows, "S248", 248, **SELECT_248["QALM 1h old"])
    queso = pick(rows, "S248", 248, **SELECT_248["QUESO-2Q"])
    for metric, expected in WIN_QUESO.items():
        w = metrics.win_rate(qalm, queso, metric)
        check(
            f"win-or-tie QALM-1h-old vs QUESO-2Q ({metric})",
            f"{w['win_or_tie']:.1f}",
            expected,
        )


def check_table26(rows):
    with open(REF_TABLE) as f:
        ref = json.load(f)
    for label, kw, col, geo in TABLE26:
        sel = pick(rows, "S26", 26, budget_s=3600, **kw)
        if col is not None:
            same = sum(r["opt_total"] == ref[r["circuit"]][col] for r in sel)
            check(f"26-table {label}: circuits equal to paper column", same, 26)
        else:
            old = dict(zip(legacy.load_manifest("S26"), OLD_PAPER_QALM))
            same = sum(r["opt_total"] == old[r["circuit"]] for r in sel)
            check(f"26-table {label}: circuits equal to pre-b9d28de column", same, 26)
        if label == "VOQC":
            same = sum(r["orig_total"] == ref[r["circuit"]][0] for r in sel)
            check("26-table Orig: circuits equal to paper column", same, 26)
        g = metrics.geomean_reduction_pct(
            (r["orig_total"], r["opt_total"]) for r in sel
        )
        check(f"26-table {label}: geomean reduction", f"{g:.2f}", geo)


def check_counter():
    """Recount every S248 input; compare with the manifest and old CSV orig columns."""
    manifest = legacy.load_manifest("S248")
    ours = {}
    for c, m in manifest.items():
        k = count_file(m["path"])
        ours[c] = (k["total"], k["cx"])
    same = sum(ours[c] == (m["total"], m["cx"]) for c, m in manifest.items())
    check("counter == S248 manifest (orig total, cx)", same, 248)
    for path in (legacy.QALM_OLD_1H_CSV, legacy.QALM_60S_CSV, legacy.QUARTZ_CSV):
        old = legacy.read_old_csv(path)
        same = sum(
            ours[c] == (int(r["orig_total"]), int(r["orig_cx"])) for c, r in old.items()
        )
        check(
            f"counter == orig columns of {os.path.relpath(path, legacy.REPO)}",
            same,
            248,
        )


def check_roundtrip(rows):
    """legacy.write_runs_csv -> metrics.load_runs must give back the same rows."""
    check(
        "row keys == collect.ROW_FIELDS", all(list(r) == ROW_FIELDS for r in rows), True
    )
    with tempfile.TemporaryDirectory() as d:
        legacy.write_runs_csv(rows, d)
        back = metrics.load_runs(d)
    keys = metrics.INT_FIELDS + (
        "job_id",
        "experiments",
        "suite",
        "circuit",
        "input_sha256",
        "optimizer",
        "params",
        "result_source",
        "status",
    )
    same = len(back) == len(rows) and all(
        all(a[k] == b[k] for k in keys) and all(type(a[k]) is type(b[k]) for k in keys)
        for a, b in zip(rows, back)
    )
    check(
        f"runs.csv round trip through metrics.load_runs ({len(rows)} rows)", same, True
    )
    check("job_id unique", len({r["job_id"] for r in rows}), len(rows))


def main():
    rows = legacy.load_legacy()
    check_averages(rows)
    check_table26(rows)
    check_counter()
    check_roundtrip(rows)
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
