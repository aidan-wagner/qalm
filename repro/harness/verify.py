"""Equivalence check of optimized outputs against their inputs.

For circuits with at most --max-qubits qubits, input and output are applied to the
same random states with a small numpy statevector simulator and compared via
|<psi_in|psi_out>| (global phase ignored). Verdicts:
  pass            overlap >= 1 - 1e-9 (exact up to floating point)
  pass_rounding   QALM/Quartz/VOQC only, overlap >= 1 - 1e-6: they print rotation angles with 6
                  decimals (QALM also round-trips every circuit through that text format when
                  calling ROQC), so arbitrary-angle circuits carry ~1e-6 rad angle rounding
                  (the 7 VOQC outputs below 1 - 1e-9 in the interim check all had >= 1 - 1.5e-8)
  approx          GUOQ TWO_Q (approximate resynthesis, epsilon 1e-8), overlap >= 1 - 1e-6
  FAIL            anything else
  trivial         the reported result is the input circuit (nothing to check)
  unverifiable    no output circuit was kept (e.g. a native run killed before it printed one)
  skipped         more than --max-qubits qubits
A verdict is prefixed "final_only:" when the checked circuit is the run's final circuit but
the reported counts come from an earlier point of its log (killed or improved after T).
Usage (repo root):
  python3 repro/harness/verify.py --plan PLAN [--runs RUNS_DIR] [--results DIR] [--max-qubits 16]
                                  [--workers 16] [--out CSV]
"""
import argparse
import csv
import json
import math
import multiprocessing
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs as jobsmod  # noqa: E402

_COMMENT = re.compile(r"//[^\n]*")
_GATE_DEF = re.compile(r"\bgate\b[^{]*\{[^}]*\}", re.S)
_STMT = re.compile(r"^([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*(.*)$", re.S)
_QARG = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")
_SAFE = {"pi": math.pi, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "__builtins__": {}}

S2 = 1 / math.sqrt(2)
FIXED = {
    "h": np.array([[S2, S2], [S2, -S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.diag([1, -1]).astype(complex),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
    "id": np.eye(2, dtype=complex),
}


def one_qubit_matrix(op, params):
    if op in FIXED:
        return FIXED[op]
    th = params[0]
    if op == "rz":
        return np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)])
    if op in ("u1", "p"):
        return np.diag([1, np.exp(1j * th)])
    if op == "rx":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])
    if op == "ry":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    raise ValueError(f"unsupported gate {op}")


def parse(path):
    """-> (n_qubits, [(op, params, [qubit indices])]) with one flat index space over all qregs."""
    text = _GATE_DEF.sub("", _COMMENT.sub("", open(path).read()))
    offsets, n, ops = {}, 0, []
    for stmt in text.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        m = _STMT.match(stmt)
        op = m.group(1).lower()
        if op == "qreg":
            name, size = _QARG.search(m.group(3)).groups()
            offsets[name] = n
            n += int(size)
            continue
        if op in ("openqasm", "include", "creg", "barrier", "measure", "reset"):
            continue
        params = [float(eval(p, _SAFE)) for p in m.group(2).split(",")] if m.group(2) else []
        qubits = [offsets[r] + int(i) for r, i in _QARG.findall(m.group(3))]
        ops.append((op, params, qubits))
    return n, ops


def simulate(n, ops, state):
    psi = state.reshape([2] * n)
    for op, params, q in ops:
        if op == "cx":
            c, t = q
            psi = np.ascontiguousarray(psi)
            idx = [slice(None)] * n
            idx[c] = 1
            tt = t - (1 if t > c else 0)  # target axis after the control axis is indexed away
            psi[tuple(idx)] = np.flip(psi[tuple(idx)], axis=tt).copy()
        else:
            u = one_qubit_matrix(op, params)
            psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [q[0]])), 0, q[0])
    return psi.reshape(-1)


def check(task):
    job, out_path = task
    try:
        n_in, ops_in = parse(job["input_path"])
        n_out, ops_out = parse(out_path)
        n = max(n_in, n_out)
        rng = np.random.default_rng(int(job["job_id"], 16) % 2**32)
        worst = 1.0
        for _ in range(2):
            v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
            v /= np.linalg.norm(v)
            a = simulate(n, ops_in, v.copy())
            b = simulate(n, ops_out, v.copy())
            worst = min(worst, abs(np.vdot(a, b)))
        approx = job["optimizer"] == "guoq" and job["params"].get("opt") == "TWO_Q"
        rounding = job["optimizer"] in ("qalm", "qalm_old", "quartz", "voqc")
        if worst >= 1 - 1e-9 and not approx:
            verdict = "pass"
        elif worst >= 1 - 1e-6 and (approx or rounding):
            verdict = "approx" if approx else "pass_rounding"
        else:
            verdict = "FAIL"
    except Exception as e:  # report, never hide
        worst, verdict = None, f"ERROR {e!r}"[:200]
    return {"job_id": job["job_id"], "optimizer": job["optimizer"], "circuit": job["circuit"],
            "n_qubits": job["n_qubits"], "overlap": worst, "verdict": verdict}


def output_path(job, jobdir):
    if job["kind"] == "native":
        return os.path.join(jobdir, "final.qasm")
    res = json.load(open(os.path.join(jobdir, "out", "result.json")))
    return os.path.join(jobdir, "out", res.get("output_qasm", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--runs", default=jobsmod.load_campaign()["runs_dir"])
    p.add_argument("--results", default=os.path.join(jobsmod.REPRO, "results"),
                   help="dir with runs.csv from run.py collect (to know which circuit was reported)")
    p.add_argument("--max-qubits", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out")
    a = p.parse_args()
    reported = {}
    runs_csv = os.path.join(a.results, "runs.csv")
    if os.path.exists(runs_csv):
        reported = {r["job_id"]: r for r in csv.DictReader(open(runs_csv))}
    tasks, rows, final_only = [], [], set()

    def note(job, verdict):
        rows.append({"job_id": job["job_id"], "optimizer": job["optimizer"], "circuit": job["circuit"],
                     "n_qubits": job["n_qubits"], "overlap": None, "verdict": verdict})

    for line in open(a.plan):
        job = json.loads(line)
        d = os.path.join(a.runs, "jobs", job["job_id"])
        if not os.path.exists(os.path.join(d, "done.json")):
            continue
        rep = reported.get(job["job_id"])
        out = output_path(job, d)
        if rep and rep["result_source"] == "original":
            note(job, "trivial")
        elif job["n_qubits"] > a.max_qubits:
            note(job, "skipped")
        elif not os.path.exists(out):
            note(job, "unverifiable")
        else:
            if rep and (rep["final_total"], rep["final_cx"]) != (rep["opt_total"], rep["opt_cx"]):
                final_only.add(job["job_id"])
            tasks.append((job, out))
    with multiprocessing.Pool(a.workers) as pool:
        for r in pool.imap_unordered(check, tasks, chunksize=1):
            if r["job_id"] in final_only:
                r["verdict"] = "final_only:" + r["verdict"]
            rows.append(r)
            if r["verdict"].split(":")[-1] not in ("pass", "pass_rounding", "approx"):
                print(r["verdict"], r["job_id"], r["optimizer"], r["circuit"], r["overlap"], flush=True)
    counts, by_opt = {}, {}
    for r in rows:
        k = "ERROR" if "ERROR" in r["verdict"] else r["verdict"]
        counts[k] = counts.get(k, 0) + 1
        by_opt.setdefault(r["optimizer"], {}).setdefault(k, 0)
        by_opt[r["optimizer"]][k] += 1
    print("verify:", counts)
    for o, c in sorted(by_opt.items()):
        print(f"  {o}: {c}")
    if a.out:
        rows.sort(key=lambda r: r["job_id"])
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    bad = sum(v for k, v in counts.items() if k.endswith("FAIL") or k == "ERROR")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
