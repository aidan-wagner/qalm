"""All metric definitions used for the paper, in one place.

Inputs are result rows as written by repro/harness/run.py collect
(repro/results/runs.csv): every row has orig_total, orig_cx, opt_total, opt_cx.
"""
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPRO, "harness"))
from jobs import full_params  # noqa: E402

F1Q, F2Q = 0.999, 0.99
FID_TIE = 1e-12


def fidelity(total, cx):
    """F = f1q^(#1q) * f2q^(#2q); every two-qubit gate in the Nam set is a cx."""
    return F1Q ** (total - cx) * F2Q ** cx


def log_fidelity(total, cx):
    """ln F; used for comparisons because F underflows to 0 for the largest circuits."""
    return (total - cx) * math.log(F1Q) + cx * math.log(F2Q)


def reduction_pct(orig, opt):
    return 100.0 * (orig - opt) / orig


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def geomean_reduction_pct(pairs):
    """26-circuit table: 1 - geomean(opt/orig), in percent."""
    pairs = list(pairs)
    return 100.0 * (1 - math.exp(sum(math.log(opt / orig) for orig, opt in pairs) / len(pairs)))


def averages(rows):
    """Table avg-comparison cells: arithmetic means over circuits."""
    return {
        "n": len(rows),
        "fidelity": mean(fidelity(r["opt_total"], r["opt_cx"]) for r in rows),
        "cx_red": mean(reduction_pct(r["orig_cx"], r["opt_cx"]) for r in rows if r["orig_cx"] > 0),
        "total_red": mean(reduction_pct(r["orig_total"], r["opt_total"]) for r in rows),
    }


def compare(a, b, metric):
    """+1 if a better, 0 tie, -1 worse, for one circuit (a, b are rows of the same circuit)."""
    if metric == "fidelity":
        # compare in log space: |d ln F| <= FID_TIE is the same relative tie rule, without underflow
        la, lb = log_fidelity(a["opt_total"], a["opt_cx"]), log_fidelity(b["opt_total"], b["opt_cx"])
        if abs(la - lb) <= FID_TIE:
            return 0
        return 1 if la > lb else -1
    key = {"cx": "opt_cx", "total": "opt_total"}[metric]
    return (a[key] < b[key]) - (a[key] > b[key])


def win_rate(qalm_rows, base_rows, metric):
    """Percent of circuits where QALM wins and where it ties (paired by circuit name)."""
    base = {r["circuit"]: r for r in base_rows}
    pairs = [(q, base[q["circuit"]]) for q in qalm_rows if q["circuit"] in base]
    res = [compare(q, b, metric) for q, b in pairs]
    n = len(res)
    return {"n": n, "win": 100.0 * res.count(1) / n, "tie": 100.0 * res.count(0) / n,
            "win_or_tie": 100.0 * (n - res.count(-1)) / n}


# ---------- curves (step functions) ----------

def step_value(series, t):
    """Best-so-far value of a change-point series [(t, total, cx), ...] (sorted, starts at t=0)."""
    v = series[0]
    for p in series:
        if p[0] <= t:
            v = p
        else:
            break
    return v


def mean_reduction_curve(series_by_circuit, orig_by_circuit, budget):
    """Exact mean-reduction step function over circuits: list of (t, mean_total_red_pct).

    Each circuit's series is a best-so-far step function starting at (0, orig).
    The mean only changes at change-points, so the returned list is exact.
    """
    times = sorted({p[0] for s in series_by_circuit.values() for p in s if p[0] <= budget} | {0.0, budget})
    n = len(series_by_circuit)
    cur = {c: s[0][1] for c, s in series_by_circuit.items()}
    idx = {c: 0 for c in series_by_circuit}
    out = []
    for t in times:
        for c, s in series_by_circuit.items():
            while idx[c] + 1 < len(s) and s[idx[c] + 1][0] <= t:
                idx[c] += 1
                cur[c] = s[idx[c]][1]
        out.append((t, sum(reduction_pct(orig_by_circuit[c], cur[c]) for c in cur) / n))
    return out


def time_to_reach(curve, threshold):
    """First time the mean-reduction step function reaches threshold (None if never)."""
    for t, v in curve:
        if v >= threshold:
            return t
    return None


# ---------- loading ----------

INT_FIELDS = ("budget_s", "seed", "orig_total", "orig_cx", "opt_total", "opt_cx")


def load_runs(results_dir):
    rows = []
    with open(os.path.join(results_dir, "runs.csv")) as f:
        for r in csv.DictReader(f):
            if r["opt_total"] == "":  # INFRA / COLLECT_ERROR rows carry no result
                print(f"load_runs: skipping {r['job_id']} ({r['optimizer']} {r['circuit']}): status {r['status']}")
                continue
            for k in INT_FIELDS:
                r[k] = int(r[k])
            # rows of plans older than a parameter get the value those runs implicitly used
            r["params"] = full_params(r["optimizer"], json.loads(r["params"]))
            r["experiments"] = r["experiments"].split(";")
            rows.append(r)
    return rows


def load_timeseries(results_dir, job_ids=None):
    ts = defaultdict(list)
    with gzip.open(os.path.join(results_dir, "timeseries.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            if job_ids is None or r["job_id"] in job_ids:
                ts[r["job_id"]].append((float(r["t_s"]), int(r["total"]), int(r["cx"])))
    for s in ts.values():
        s.sort(key=lambda p: p[0])  # stable: equal timestamps keep log order (last one wins)
    return ts


def select(rows, optimizer, suite=None, budget_s=None, seed=None, **params):
    out = []
    for r in rows:
        if r["optimizer"] != optimizer:
            continue
        if suite is not None and r["suite"] != suite:
            continue
        if budget_s is not None and r["budget_s"] != budget_s:
            continue
        if seed is not None and r["seed"] != seed:
            continue
        if any(r["params"].get(k) != v for k, v in params.items()):
            continue
        out.append(r)
    return out
