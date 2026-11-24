#!/usr/bin/env python3
"""Equivalence check of the larger circuits (more than 16 qubits) with MQT QCEC.

Complements verify.py, whose statevector simulation stops at 16 qubits. For every finished
job of a plan whose circuit has at least --min-qubits qubits (default 17: exactly the runs
verify.py reports as "skipped"), the input circuit and the reported output circuit are
compared with QCEC's decision-diagram *alternating* checker, which decides equivalence for all
inputs (not for a few sample states). QCEC's simulation checker (it can only say "probably
equivalent", and once gave a false "probably equivalent" for mod_mult_55), its ZX checker and
its construction checker are switched off. Written and tested against mqt.qcec 3.12.0.

Method (measured with mqt.qcec 3.12.0 on circuits with known answers): every check uses QCEC's
default numerical tolerance (2.3e-13). A looser tolerance was tried and rejected: at 1e-5 it
accepted an output with one angle off by 1e-3 rad (distance 5e-4, above the threshold), and at
1e-7 it rejected a correct one. QCEC's HSF checker supports at most 63 split operations, far
fewer than these circuits need.
Rounded angles: QALM, Quartz and VOQC print rotation angles with 6 decimals. At the default
tolerance such slightly-off angles keep the decision diagrams from collapsing, and the check does
not finish even on small circuits (tof_10, 19 qubits). Most large circuits are Clifford+T: every
input angle is a multiple of pi/4, so every correct output angle is one too. When the input is
Clifford+T, every output angle within SNAP_TOL (1e-5 rad) of a multiple of pi/4 is set to that
exact multiple (in memory; no file is written), all other angles stay as printed, and this circuit
is checked exactly. Otherwise (arbitrary-angle inputs) the output is checked as printed.

Verdicts (column "verdict"), with the same meaning as in verify.py:
  pass             proved equivalent (or up to a global phase), as printed
  pass_rounding    proved equivalent after snapping printed angles to multiples of pi/4 (the note
                   gives how many and the largest correction), or, for QALM/Quartz/VOQC on
                   arbitrary-angle circuits, proved closer than the distance threshold below.
                   For other tools the note says to check by hand (they are not expected to round)
  approx           GUOQ TWO_Q only (approximate resynthesis by design): proved closer than the
                   distance threshold
  FAIL             proved not equivalent (beyond the threshold, where one applies)
  unchecked:<why>  no proof either way: timeout, memory, killed, crash, no_information, ...
                   Never counts as passed.
  ERROR            the check itself failed (e.g. a circuit QCEC cannot parse); see "note"
  trivial          the reported result is the input circuit (nothing to check)
  unverifiable     no output circuit was kept
A verdict is prefixed "final_only:" when the checked circuit is the run's final circuit but the
reported counts come from an earlier point of its log (as in verify.py).

Tolerance: --rounding-tol is verify.py's (max 1 - |<psi|U^dag V|psi>| for random states, default
1e-8). QCEC's approximate check bounds the projective Hilbert-Schmidt distance
sqrt(1 - |Tr(U V^dag)/2^n|^2) instead; for small deviations 1 - overlap ~ distance^2 / 2, so the
equivalent distance threshold is sqrt(2 * rounding_tol) (1.41e-4 for 1e-8). Measured with 3.12.0:
the approximate check matches true distances from ~1e-5 up, and accepts anything below ~1e-6
(the decision-diagram package's numerical floor), so thresholds of 1e-5 or more are meaningful.

Resources: every check runs in its own process on one core (QCEC's parallel mode is off), with a
wall-clock limit (--timeout for the whole check; the process is killed 120 s after that) and an
address-space limit (--mem-gib). Check processes set oom_score_adj=1000, so a memory shortage
kills a check, never this supervisor. --workers checks run at once; the default is every CPU this
process may use, so start it with the CPUs it should use (e.g. systemd-run -p AllowedCPUs=...).
The largest checks start first.

Progress: one line per finished check ([done/total], verdict, seconds, job) and a status line
every 60 s (running checks, the longest-running one, verdict counts).

Files: written only inside --out-dir, which must be new or empty (or, with --resume, a previous
out-dir of this script): run_info_<UTC>.json (settings of each start), results.jsonl (one line
per finished check, written immediately, so an interrupted run continues with --resume),
verify_qcec.csv (all results, rewritten at the end of each start) and, for --self-test,
selftest/*.qasm (deliberately broken outputs) and selftest/summary.json. Nothing else is created
or modified.

Usage (repo root; run with the Python of a separate venv that has mqt.qcec; nothing else needed):
  python repro/harness/verify_qcec.py --show-config
  python repro/harness/verify_qcec.py --plan repro/runs/plan.jsonl --out-dir DIR --self-test auto
  python repro/harness/verify_qcec.py --plan repro/runs/plan.jsonl --out-dir DIR [--limit N] [--job-ids A,B]
        [--min-qubits 17] [--timeout 3600] [--mem-gib 16] [--workers N] [--rounding-tol 1e-8]
        [--resume [--retry-unchecked]]
"""
import argparse
import csv
import json
import math
import os
import re
import resource
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACCEPT = {"equivalent", "equivalent_up_to_global_phase"}  # checked: a pure global phase gives the latter
ROUNDING = {"qalm", "qalm_old", "quartz", "voqc"}  # print angles with 6 decimals (as in verify.py)
MUTATIONS = ("drop_cx", "swap_cx", "shift_rz")
CSV_FIELDS = ["job_id", "optimizer", "mode", "suite", "circuit", "n_qubits", "size_est", "verdict",
              "equivalence", "check_s", "note"]
PRINT_LOCK = threading.Lock()


def log(*args):
    with PRINT_LOCK:
        print(*args, flush=True)


def distance_threshold(rounding_tol):
    return math.sqrt(2 * rounding_tol)


# ---------------------------------------------------------------- one check (child process)

SNAP_TOL = 1e-5  # rad; 6-decimal printing errs by at most 5e-7 (radians) or 1.6e-6 (pi*<6 decimals>)
_ROT = re.compile(r"(?m)^([ \t]*)(rz|u1|p)(\s*\()([^()]*)(\))")
_SAFE = {"pi": math.pi, "__builtins__": {}}
QUARTER = math.pi / 4


def _angles(text):
    """-> [(match, value or None)] for every rz/u1/p gate of a QASM text."""
    res = []
    for m in _ROT.finditer(text):
        try:
            res.append((m, float(eval(m.group(4), _SAFE))))
        except Exception:
            res.append((m, None))
    return res


def is_clifford_t(text):
    """Every rotation angle is a multiple of pi/4 (up to float noise)."""
    return all(v is not None and abs(v - round(v / QUARTER) * QUARTER) < 1e-12 for _, v in _angles(text))


def snap_quarter(text):
    """-> (text with every angle within SNAP_TOL of a multiple of pi/4 set exactly to it, number of
    angles changed, largest change, number of angles left as printed because they are not that
    close to a multiple of pi/4 or cannot be evaluated)."""
    pieces, last, n, worst, far = [], 0, 0, 0.0, 0
    for m, v in _angles(text):
        if v is None:
            far += 1
            continue
        k = round(v / QUARTER)
        d = abs(v - k * QUARTER)
        if d > SNAP_TOL:
            far += 1
        elif d > 1e-12:
            pieces.append(text[last:m.start(4)] + f"pi*{k / 4!r}")
            last = m.end(4)
            n += 1
            worst = max(worst, d)
    pieces.append(text[last:])
    return "".join(pieces), n, worst, far


def make_config(timeout, approx_threshold=None):
    from mqt.qcec.pyqcec import Configuration
    cfg = Configuration()
    ex = cfg.execution
    ex.run_alternating_checker = True
    ex.run_construction_checker = False
    ex.run_simulation_checker = False
    ex.run_zx_checker = False
    ex.run_hsf_checker = False
    ex.parallel = False
    ex.nthreads = 1
    ex.timeout = float(timeout)
    if approx_threshold is not None:
        cfg.functionality.check_approximate_equivalence = True
        cfg.functionality.approximate_checking_threshold = float(approx_threshold)
    return cfg


def run_one(task):
    cap = int(task["mem_gib"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    t0 = time.time()
    deadline = t0 + task["timeout"]
    out = {"verdict": None, "equivalence": None, "note": ""}
    try:
        from mqt import qcec

        from mqt.core.ir import QuantumComputation

        def left():
            return deadline - time.time()

        def check(second, threshold=None):
            cfg = make_config(max(1.0, left()), threshold)
            return qcec.verify(task["input"], second, configuration=cfg).equivalence.name

        def unresolved(e):
            return "unchecked:timeout" if e == "no_information" and left() < 5 else f"unchecked:{e}"

        d_max = distance_threshold(task["rounding_tol"])
        with open(task["input"]) as f:
            in_text = f.read()
        with open(task["output"]) as f:
            out_text = f.read()
        snapped, n_snap, worst, far = (None, 0, 0.0, 0)
        if not task["approx"] and is_clifford_t(in_text):
            snapped, n_snap, worst, far = snap_quarter(out_text)
        if n_snap > 0:  # Clifford+T input, rounded printed angles
            e = check(QuantumComputation.from_qasm_str(snapped))
            out["equivalence"] = f"snapped:{e}"
            out["note"] = (f"{n_snap} printed angles set to the nearest multiple of pi/4 "
                           f"(largest change {worst:.1e} rad)"
                           + (f"; {far} angles left as printed (not within {SNAP_TOL:g} rad of one)" if far else ""))
            if e in ACCEPT:
                out["verdict"] = "pass_rounding"
                if task["optimizer"] not in ROUNDING:
                    out["note"] += f"; {task['optimizer']} is not expected to round angles: check by hand"
            else:
                out["verdict"] = "FAIL" if e == "not_equivalent" else unresolved(e)
        elif task["approx"]:  # GUOQ TWO_Q
            e = check(task["output"], d_max)
            out["equivalence"] = f"approx:{e}"
            out["verdict"] = "approx" if e in ACCEPT else "FAIL" if e == "not_equivalent" else unresolved(e)
        else:
            e = check(task["output"])
            out["equivalence"] = e
            if e in ACCEPT:
                out["verdict"] = "pass"
            elif e != "not_equivalent":
                out["verdict"] = unresolved(e)
            elif task["optimizer"] in ROUNDING:  # arbitrary angles, printed rounded: within the threshold?
                e2 = check(task["output"], d_max)
                out["equivalence"] = f"{e}; approx:{e2}"
                out["verdict"] = ("pass_rounding" if e2 in ACCEPT else "FAIL" if e2 == "not_equivalent"
                                  else unresolved(e2))
            else:
                out["verdict"] = "FAIL"
    except MemoryError:
        out["verdict"], out["note"] = "unchecked:memory", f"address-space limit {task['mem_gib']:g} GiB"
    except Exception as e:  # report, never hide
        out["verdict"], out["note"] = "ERROR", repr(e)[:300]
    out["check_s"] = round(time.time() - t0, 2)
    print("RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------- supervisor side

RUNNING, RUNNING_LOCK = {}, threading.Lock()


def run_task(task, a):
    """Run one check in a child process and return its result dict (never raises)."""
    child = {**task, "timeout": a.timeout, "mem_gib": a.mem_gib, "rounding_tol": a.rounding_tol}
    cmd = [sys.executable, os.path.abspath(__file__), "--one", json.dumps(child)]
    t0 = time.time()
    with RUNNING_LOCK:
        RUNNING[task["job_id"] + task.get("variant", "")] = (t0, task)
    try:
        # one BLAS thread per check: numpy's OpenBLAS (loaded by QCEC) otherwise starts one thread per
        # visible CPU in every check process, each reserving memory against the --mem-gib limit
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout + 120, env=env)
        except subprocess.TimeoutExpired:
            return {"verdict": "unchecked:timeout", "equivalence": None,
                    "check_s": round(time.time() - t0, 2), "note": "killed after --timeout + 120 s"}
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if lines:
            return json.loads(lines[-1][len("RESULT "):])
        if p.returncode == -9:
            why, verdict = "killed by signal 9 (probably out of memory)", "unchecked:killed"
        elif p.returncode < 0:
            why, verdict = f"killed by signal {-p.returncode}", "unchecked:killed"
        else:
            why, verdict = f"exit code {p.returncode}", "unchecked:crash"
        err = p.stderr.strip()
        if any(s in err for s in ("MemoryError", "bad_alloc", "Memory allocation", "Cannot allocate memory")):
            verdict = "unchecked:memory"
        return {"verdict": verdict, "equivalence": None, "check_s": round(time.time() - t0, 2),
                "note": f"{why}; stderr: {err[-250:]}"}
    finally:
        with RUNNING_LOCK:
            RUNNING.pop(task["job_id"] + task.get("variant", ""), None)


def output_path(job, jobdir):  # as verify.output_path
    if job["kind"] == "native":
        return os.path.join(jobdir, "final.qasm")
    with open(os.path.join(jobdir, "out", "result.json")) as f:
        res = json.load(f)
    return os.path.join(jobdir, "out", res.get("output_qasm", ""))


_SIZE = {}


def size_est(path):
    """Number of ';' in a QASM file: a cheap gate-count estimate (cached per path)."""
    if path not in _SIZE:
        with open(path, errors="replace") as f:
            _SIZE[path] = f.read().count(";")
    return _SIZE[path]


def select(a, job_ids, min_qubits):
    """-> (tasks to check, rows that need no check). Same choice of circuit as verify.py."""
    rc = os.path.join(a.results, "runs.csv")
    if not os.path.exists(rc):
        sys.exit(f"no {rc}: run 'run.py collect' first (needed for trivial / final_only)")
    with open(rc, newline="") as f:
        rep = {r["job_id"]: r for r in csv.DictReader(f)}
    only = set(job_ids) if job_ids else None
    tasks, fixed, finished, seen = [], [], 0, set()
    with open(a.plan) as f:
        for line in f:
            job = json.loads(line)
            if only is not None and job["job_id"] not in only:
                continue
            d = os.path.join(a.runs, "jobs", job["job_id"])
            if not os.path.exists(os.path.join(d, "done.json")):
                continue
            finished += 1
            seen.add(job["job_id"])
            if only is None and job["n_qubits"] < min_qubits:
                continue
            base = {"job_id": job["job_id"], "optimizer": job["optimizer"],
                    "mode": job.get("params", {}).get("opt", ""), "suite": job.get("suite", ""),
                    "circuit": job["circuit"], "n_qubits": job["n_qubits"]}
            r = rep.get(job["job_id"])
            if r and r["result_source"] == "original":
                fixed.append({**base, "verdict": "trivial"})
                continue
            out = output_path(job, d)
            if not os.path.exists(out):
                fixed.append({**base, "verdict": "unverifiable"})
                continue
            inp = job["input_path"] if os.path.isabs(job["input_path"]) else os.path.join(REPO, job["input_path"])
            final_only = bool(r and (r["final_total"], r["final_cx"]) != (r["opt_total"], r["opt_cx"]))
            tasks.append({**base, "input": inp, "output": out, "final_only": final_only,
                          "approx": job["optimizer"] == "guoq" and base["mode"] == "TWO_Q",
                          "size_est": size_est(inp) + size_est(out)})
    if finished == 0:
        sys.exit(f"no finished jobs of {a.plan} found in {a.runs}/jobs: wrong --runs?")
    if only is not None and only - seen:
        sys.exit(f"job ids not found or not finished: {sorted(only - seen)}")
    return tasks, fixed


def spread(tasks, n):
    """n tasks evenly spread over the size range (smallest to largest)."""
    s = sorted(tasks, key=lambda t: t["size_est"])
    if n >= len(s):
        return s
    if n == 1:
        return [s[len(s) // 2]]
    idx = sorted({round(i * (len(s) - 1) / (n - 1)) for i in range(n)})
    return [s[i] for i in idx]


def load_jsonl(path):
    last = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    last[r["job_id"]] = r
    return last


def prepare_out_dir(a):
    od = os.path.abspath(a.out_dir)
    if a.resume:
        if not os.path.exists(os.path.join(od, "results.jsonl")):
            sys.exit(f"--resume: no {od}/results.jsonl")
    elif os.path.exists(od) and os.listdir(od):
        sys.exit(f"--out-dir {od} exists and is not empty; use a new directory (or --resume)")
    os.makedirs(od, exist_ok=True)
    return od


def qcec_version():
    from mqt import qcec
    make_config(1.0)  # fail early if this QCEC version lacks an option the checks use
    make_config(1.0, 1e-4)
    from mqt.core.ir import QuantumComputation  # noqa: F401  (used for snapped circuits)
    return getattr(qcec, "__version__", "?")


def write_run_info(od, a, version, n_tasks):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    info = {"started_utc": stamp, "host": socket.gethostname(), "qcec_version": version,
            "distance_threshold": distance_threshold(a.rounding_tol), "snap_tolerance_rad": SNAP_TOL,
            "argv": sys.argv,
            "args": {k: v for k, v in vars(a).items() if k != "one"}, "n_checks": n_tasks,
            "cpus": sorted(os.sched_getaffinity(0))}
    with open(os.path.join(od, f"run_info_{stamp}.json"), "x") as f:
        json.dump(info, f, indent=1)


def fmt_elapsed(s):
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def heartbeat(stop, t_start, total, done_ref, counts, every):
    while not stop.wait(every):
        with RUNNING_LOCK:
            running = list(RUNNING.values())
        snap = dict(counts)
        longest = max(running, key=lambda x: time.time() - x[0], default=None)
        lg = (f"; longest running: {longest[1]['optimizer']} {longest[1]['circuit']} "
              f"({longest[1]['n_qubits']}q) {(time.time() - longest[0]) / 60:.0f} min") if longest else ""
        log(f"-- {fmt_elapsed(time.time() - t_start)}  done {done_ref[0]}/{total}, running {len(running)}{lg}"
            f"  | {dict(sorted(snap.items()))}")


def run_all(a, od):
    tasks, fixed = select(a, a.job_ids.split(",") if a.job_ids else None, a.min_qubits)
    if a.limit:
        tasks = spread(tasks, a.limit)
    jsonl = os.path.join(od, "results.jsonl")
    prev = load_jsonl(jsonl) if a.resume else {}

    def redo(jid):
        r = prev.get(jid)
        return r is None or (a.retry_unchecked and ("unchecked" in r["verdict"] or "ERROR" in r["verdict"]))

    todo = sorted((t for t in tasks if redo(t["job_id"])), key=lambda t: -t["size_est"])
    version = qcec_version()
    write_run_info(od, a, version, len(todo))
    log(f"mqt.qcec {version}; plan {a.plan}; runs {a.runs}; out {od}")
    log(f"{len(tasks)} circuit pairs selected ({len(tasks) - len(todo)} already done), {len(fixed)} need no check; "
        f"checking {len(todo)} with {a.workers} workers, timeout {a.timeout:.0f} s and {a.mem_gib:g} GiB each, "
        f"distance threshold {distance_threshold(a.rounding_tol):.2e}, snap tolerance {SNAP_TOL:g} rad")
    counts, done = {}, [0]
    t_start = time.time()
    with open(jsonl, "a") as out:
        def record(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            os.fsync(out.fileno())

        for row in fixed:
            if row["job_id"] not in prev:
                record({**row, "equivalence": None, "check_s": None, "note": ""})
        stop = threading.Event()
        hb = threading.Thread(target=heartbeat, args=(stop, t_start, len(todo), done, counts, a.status_every),
                              daemon=True)
        hb.start()
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(a.workers, len(todo) or 1))) as ex:
                futs = {ex.submit(run_task, t, a): t for t in todo}
                for f in as_completed(futs):
                    t = futs[f]
                    r = f.result()
                    if t["final_only"]:
                        r["verdict"] = "final_only:" + r["verdict"]
                    row = {k: t[k] for k in ("job_id", "optimizer", "mode", "suite", "circuit", "n_qubits", "size_est")}
                    row.update({"verdict": r["verdict"], "equivalence": r.get("equivalence"),
                                "check_s": r.get("check_s"), "note": r.get("note", "")})
                    record(row)
                    done[0] += 1
                    key = "ERROR" if "ERROR" in r["verdict"] else r["verdict"].split("final_only:")[-1]
                    counts[key] = counts.get(key, 0) + 1
                    log(f"[{done[0]:>4}/{len(todo)}] {fmt_elapsed(time.time() - t_start)}  {r['verdict']:30s} "
                        f"{r.get('check_s') or 0:8.1f}s  {t['optimizer']:9s} {t['mode']:6s} {t['circuit']} "
                        f"({t['n_qubits']}q, ~{t['size_est']} gates)  {t['job_id']}"
                        + (f"  [{r['note'][:150]}]" if r.get("note") else ""))
        finally:
            stop.set()
    return finish(od, jsonl)


def finish(od, jsonl):
    rows = sorted(load_jsonl(jsonl).values(), key=lambda r: r["job_id"])
    with open(os.path.join(od, "verify_qcec.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts, by = {}, {}
    for r in rows:
        k = "ERROR" if "ERROR" in r["verdict"] else r["verdict"]
        counts[k] = counts.get(k, 0) + 1
        g = by.setdefault(f"{r['optimizer']} {r['mode']}".strip(), {})
        g[k] = g.get(k, 0) + 1
    log(f"verify_qcec: {len(rows)} rows -> {os.path.join(od, 'verify_qcec.csv')}")
    log(f"  all: {dict(sorted(counts.items()))}")
    for name, c in sorted(by.items()):
        log(f"  {name}: {dict(sorted(c.items()))}")
    bad = sum(v for k, v in counts.items() if k.endswith("FAIL") or k == "ERROR")
    return 1 if bad else 0


# ---------------------------------------------------------------- self-test

_GATE = re.compile(r"(?m)^[ \t]*(cx|rz)\b[^;]*;")


def mutate(text, kind):
    """A deliberately broken copy of a QASM text (None if the circuit has no gate to break)."""
    ms = list(_GATE.finditer(text))
    cx = [m for m in ms if m.group(1) == "cx"]
    rz = [m for m in ms if m.group(1) == "rz"]
    if kind in ("drop_cx", "swap_cx"):
        if not cx:
            return None
        m = cx[len(cx) // 2]
        if kind == "drop_cx":
            new = ""
        else:
            mm = re.match(r"([ \t]*cx\s+)([^,;]+?)\s*,\s*([^;]+?)\s*;", m.group(0))
            if not mm:
                return None
            new = f"{mm.group(1)}{mm.group(3)}, {mm.group(2)};"
    else:  # shift_rz: one rotation angle off by 0.01 rad (distance 5e-3, far above the threshold)
        if not rz:
            return None
        m = rz[len(rz) // 2]
        new = re.sub(r"rz\s*\(", "rz(0.01 + ", m.group(0), count=1)
    return text[:m.start()] + new + text[m.end():]


def pick_auto(tasks):
    """A few representative jobs: QALM <= 16 and > 16 qubits, GUOQ TWO_Q <= 16 and > 16, VOQC > 16."""
    picks = []
    for want in (lambda t: t["optimizer"] == "qalm" and t["n_qubits"] <= 16,
                 lambda t: t["optimizer"] == "qalm" and t["n_qubits"] > 16,
                 lambda t: t["approx"] and t["n_qubits"] <= 16,
                 lambda t: t["approx"] and t["n_qubits"] > 16,
                 lambda t: t["optimizer"] == "voqc" and t["n_qubits"] > 16):
        c = sorted((t for t in tasks if want(t) and t["size_est"] > 200 and not t["final_only"]),
                   key=lambda t: t["size_est"])
        if c and c[0] not in picks:
            picks.append(c[0])
    return picks


def expected_original(t):
    if t["approx"]:
        return {"approx"}
    return {"pass", "pass_rounding"} if t["optimizer"] in ROUNDING else {"pass"}


def self_test(a, od):
    if a.self_test == "auto":
        tasks = pick_auto(select(a, None, 0)[0])
    else:
        tasks = select(a, a.self_test.split(","), 0)[0]
    if not tasks:
        sys.exit("self-test: no jobs to test")
    version = qcec_version()
    write_run_info(od, a, version, len(tasks))
    sd = os.path.join(od, "selftest")
    os.makedirs(sd)
    checks = []
    for t in tasks:
        checks.append(({**t, "variant": ":original"}, "original", expected_original(t)))
        with open(t["output"]) as f:
            text = f.read()
        for kind in MUTATIONS:
            new = mutate(text, kind)
            if new is None:
                continue
            path = os.path.join(sd, f"{t['job_id']}_{kind}.qasm")
            with open(path, "x") as f:  # "x": never overwrite anything
                f.write(new)
            checks.append(({**t, "output": path, "final_only": False, "variant": ":" + kind}, kind, {"FAIL"}))
    log(f"mqt.qcec {version}; distance threshold {distance_threshold(a.rounding_tol):.2e}")
    log(f"self-test: {len(checks)} checks on {len(tasks)} jobs (each original must pass, each broken copy must FAIL)")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(a.workers, len(checks)))) as ex:
        futs = {ex.submit(run_task, t, a): (t, kind, want) for t, kind, want in checks}
        for f in as_completed(futs):
            t, kind, want = futs[f]
            r = f.result()
            ok = r["verdict"] in want
            results.append({"job_id": t["job_id"], "optimizer": t["optimizer"], "mode": t["mode"],
                            "circuit": t["circuit"], "n_qubits": t["n_qubits"], "check": kind,
                            "expected": sorted(want), **r, "ok": ok})
            log(f"  {'ok ' if ok else 'BAD'}  {kind:9s} expected {'/'.join(sorted(want)):19s} got {r['verdict']:24s} "
                f"{r.get('check_s') or 0:7.1f}s  {t['optimizer']} {t['mode']} {t['circuit']} "
                f"({t['n_qubits']}q, ~{t['size_est']} gates) {t['job_id']}"
                + (f"  [{r['note'][:150]}]" if r.get("note") else ""))
    with open(os.path.join(sd, "summary.json"), "x") as f:
        json.dump(results, f, indent=1)
    bad = sum(not r["ok"] for r in results)
    log("self-test PASSED" if bad == 0 else f"self-test FAILED: {bad} of {len(results)} checks unexpected")
    return 0 if bad == 0 else 1


def show_config():
    from mqt import qcec
    print("mqt.qcec", getattr(qcec, "__version__", "?"))
    cfg = make_config(3600.0, 1.41e-4)
    for sec in ("execution", "functionality", "simulation", "optimizations", "application", "parameterized"):
        o = getattr(cfg, sec, None)
        if o is None:
            continue
        for k in sorted(dir(o)):
            if k.startswith("_"):
                continue
            try:
                v = getattr(o, k)
            except Exception as e:
                v = f"<{e!r}>"
            if not callable(v):
                print(f"  {sec}.{k} = {v!r}")
    print("(settings of an approximate check; an exact check differs only in check_approximate_equivalence = False)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--one", help=argparse.SUPPRESS)
    p.add_argument("--show-config", action="store_true", help="print the QCEC version and settings, then exit")
    p.add_argument("--plan")
    p.add_argument("--runs", default=os.path.join(REPO, "repro", "runs"))
    p.add_argument("--results", default=os.path.join(REPO, "repro", "results"),
                   help="dir with runs.csv (to know which circuit each run reported)")
    p.add_argument("--out-dir", help="new (or empty) directory for every file this script writes")
    p.add_argument("--min-qubits", type=int, default=17)
    p.add_argument("--timeout", type=float, default=3600, help="seconds per check (default 3600)")
    p.add_argument("--mem-gib", type=float, default=16, help="address-space limit per check (default 16)")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="checks at once (default: CPUs this process may use)")
    p.add_argument("--rounding-tol", type=float, default=1e-8,
                   help="verify.py's tolerance (1 - overlap); QCEC distance threshold = sqrt(2 * tol)")
    p.add_argument("--job-ids", help="comma-separated job ids (any qubit count)")
    p.add_argument("--limit", type=int, help="only N checks, evenly spread over circuit size (a pilot run)")
    p.add_argument("--resume", action="store_true", help="continue in an existing --out-dir")
    p.add_argument("--retry-unchecked", action="store_true", help="with --resume: redo unchecked / ERROR checks")
    p.add_argument("--self-test", metavar="auto|JOB_IDS",
                   help="check originals and deliberately broken copies of a few outputs")
    p.add_argument("--status-every", type=float, default=60, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.one:
        run_one(json.loads(a.one))
        return 0
    if a.show_config:
        show_config()
        return 0
    if not a.plan or not a.out_dir:
        p.error("--plan and --out-dir are required")
    od = prepare_out_dir(a)
    if a.self_test:
        if a.resume:
            p.error("--self-test needs a new --out-dir")
        return self_test(a, od)
    return run_all(a, od)


if __name__ == "__main__":
    sys.exit(main())
