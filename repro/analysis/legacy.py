"""Adapters that turn OLD result data on this machine into runs.csv-style rows.

The new table/figure generators read rows produced by
``metrics.load_runs(results_dir)``.  ``load_legacy()`` returns rows in exactly
that schema (every ``collect.ROW_FIELDS`` key; ints for counts, ``params`` as a
dict, ``experiments`` as a list), built from pre-campaign data, so the
generators can be exercised and regression-tested before the new campaign
finishes.  Every qasm file is counted with ``repro/harness/qasm_count.py``.

Sources (all read-only):

1. ``qalm_old`` 3600 s, S248 -- ``guoq_benchmark_results/results_3600s.csv``
   (rows with source=guoq; old BEST_CONFIG run of benchmark_guoq.py@b10f6ec).
2. ``qalm`` config a, 60 s, S248 --
   ``guoq_benchmark_prev_qalm_results/results_60s.csv`` (source=guoq).
3. ``quartz`` 3600 s, S248 -- ``original_benchmark_results/results_3600s.csv``.
   OOM_AFTER_GREEDY rows already hold the post-greedy counts (status OOM_CAP).
4. ``guoq`` / ``queso`` with opt=TWO_Q, 3600 s, S248 -- top-level
   ``latest_sol_*.qasm`` of each circuit in the April 2026 docker run
   (nested ``results_<c>/`` dirs hold a later, different run and are ignored).
   A missing output (4 QUESO runs) counts as the original circuit (status
   NO_RESULT).  Every other run is OK and uses its latest_sol, exactly like
   collect.py (exit 0/124 -> OK, final qasm).  That includes the runs whose
   ``results_*.json`` has no ``best_*`` fields (8 GUOQ, 74 QUESO): run_guoq.py
   fills those only from a ``rules_applied_to_best`` line, i.e. on an
   improvement, and these runs ran the full hour with an empty stderr, so they
   never improved rather than failed.  Their latest_sol is the tool's re-emitted
   input, identical to it except where the tool dropped trivial rotations
   (e.g. ground_state_estimation_10: 371940 instead of 390180 gates).
5. 26-circuit table, S26 -- ``all_bench_results_hour/fresh_results_3600``:
   ``voqc``, ``queso``/``guoq`` (opt=TOTAL), ``quartz`` (= Quartz w/o
   preprocessing) and ``qalm_dec2025`` (the pre-April QALM column of the old
   paper).  No raw data exists for Quartz w/ preprocessing (S26PP).

Original counts always come from the input manifests (``repro/inputs``), i.e.
from our counter; for sources 1-3 they equal the CSV's orig columns (checked by
validate_legacy.py).  Conventions that differ from a new-campaign row:
``seed`` is 0 everywhere (old GUOQ/QUESO runs used random seeds),
``result_source`` is ``legacy_csv`` / ``final_qasm`` / ``original``, and all
resource/log columns are None.

CLI: ``python3 repro/analysis/legacy.py --out DIR`` writes ``DIR/runs.csv`` in
the collect.py format, so ``metrics.load_runs(DIR)`` returns the same rows.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
REPO = os.path.dirname(REPRO)
sys.path.insert(0, os.path.join(REPRO, "harness"))

from collect import ROW_FIELDS  # noqa: E402
from jobs import DEFAULTS  # noqa: E402
from qasm_count import count_file  # noqa: E402

QALM_OLD_1H_CSV = os.path.join(REPO, "guoq_benchmark_results", "results_3600s.csv")
QALM_60S_CSV = os.path.join(REPO, "guoq_benchmark_prev_qalm_results", "results_60s.csv")
QUARTZ_CSV = os.path.join(REPO, "original_benchmark_results", "results_3600s.csv")
DOCKER_248 = (
    "/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44"
)
NAM26 = os.path.join(
    REPO, "all_bench_results_hour", "fresh_results_3600", "qalm_bench", "nam"
)

QALM_DEC2025_STEM = "3600_1_1_2_1.5_0_0_1_1_1_Nam_5_3_complete_ECC_set.json"
# Fields of the old comparison_experiments.py result-file name (timeout first).
QALM_DEC2025_PARAMS = {
    "initial_pool_size": 1,
    "exploration_pool_size": 1,
    "exploration_steps": 2,
    "repeat_tolerance": 1.5,
    "exploration_increase": 0,
    "no_increase": 0,
    "only_do_local_transformations": 1,
    "greedy_start": 1,
    "two_way_rm": 1,
    "eccset": "Nam_5_3_complete_ECC_set.json",
}

E1, E2 = "E1_main", "E2_quartz26"
CSV_STATUS = {"OK": "OK", "OOM_AFTER_GREEDY": "OOM_CAP"}
GUOQ_TAGS = {"guoq": "none_1", "queso": "none_none"}


def load_manifest(suite):
    """{circuit: manifest row} of repro/inputs/<suite>.csv (counts as ints)."""
    with open(os.path.join(REPRO, "inputs", f"{suite}.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_qubits", "total", "cx"):
            r[k] = int(r[k])
    return {r["name"]: r for r in rows}


def read_old_csv(path):
    """{circuit: row} of an old benchmark CSV, source=guoq rows of the S248 suite."""
    suite = load_manifest("S248")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "guoq"]
    out = {r["circuit"]: r for r in rows if r["circuit"] in suite}
    missing = set(suite) - set(out)
    if missing:
        raise ValueError(f"{path}: no row for {sorted(missing)}")
    return out


def _make_row(suite, man, circuit, optimizer, params, budget_s, opt, status, source):
    m = man[circuit]
    key = {
        "legacy": True,
        "optimizer": optimizer,
        "params": params,
        "circuit": circuit,
        "input_sha256": m["sha256"],
        "budget_s": budget_s,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
    row = dict.fromkeys(ROW_FIELDS)
    row.update(
        job_id="legacy-" + digest[:12],
        experiments=[E2 if suite == "S26" else E1],
        suite=suite,
        circuit=circuit,
        input_sha256=m["sha256"],
        optimizer=optimizer,
        budget_s=int(budget_s),
        seed=0,
        params=dict(params),
        orig_total=m["total"],
        orig_cx=m["cx"],
        opt_total=int(opt[0]),
        opt_cx=int(opt[1]),
        result_source=source,
        status=status,
    )
    return row


def _from_old_csv(path, optimizer, params, budget_s):
    man = load_manifest("S248")
    rows = []
    for c, r in sorted(read_old_csv(path).items()):
        status = CSV_STATUS[r["status"]]
        opt = (r["opt_total"], r["opt_cx"])
        rows.append(
            _make_row(
                "S248", man, c, optimizer, params, budget_s, opt, status, "legacy_csv"
            )
        )
    return rows


def _qasm_or_original(man, circuit, qasm):
    """(opt counts, status, result_source) of an output file; missing = original."""
    if not os.path.exists(qasm):
        m = man[circuit]
        return (m["total"], m["cx"]), "NO_RESULT", "original"
    k = count_file(qasm)
    return (k["total"], k["cx"]), "OK", "final_qasm"


def qalm_old_1h():
    """Source 1: old-config QALM, 1 h, S248."""
    return _from_old_csv(QALM_OLD_1H_CSV, "qalm_old", DEFAULTS["qalm_old"], 3600)


def qalm_60s():
    """Source 2: QALM config a, 60 s, S248."""
    return _from_old_csv(QALM_60S_CSV, "qalm", DEFAULTS["qalm"], 60)


def quartz_248():
    """Source 3: Quartz, 1 h, S248."""
    return _from_old_csv(QUARTZ_CSV, "quartz", DEFAULTS["quartz"], 3600)


def guoq_queso_248(optimizer):
    """Source 4: GUOQ or QUESO with -opt TWO_Q, 1 h, S248 (April 2026 docker run)."""
    tag = GUOQ_TAGS[optimizer]
    man = load_manifest("S248")
    rows = []
    for c in sorted(man):
        d = os.path.join(DOCKER_248, c, optimizer)
        opt, status, source = _qasm_or_original(
            man, c, os.path.join(d, f"latest_sol_{tag}_{c}.qasm")
        )
        params = {"opt": "TWO_Q"}
        rows.append(
            _make_row("S248", man, c, optimizer, params, 3600, opt, status, source)
        )
    return rows


def table26():
    """Source 5: the 26-circuit table runs (S26), all 1 h."""
    man = load_manifest("S26")
    paths = {
        ("voqc", None): "voqc/results_{c}/optimized_none_none_{c}.qasm",
        ("queso", "TOTAL"): "queso/results_{c}/latest_sol_none_none_{c}.qasm",
        ("guoq", "TOTAL"): "guoq/results_{c}/latest_sol_none_1_{c}.qasm",
        ("quartz", None): "qalm/{c}/quartz_result.qasm",
        ("qalm_dec2025", None): "qalm/{c}/" + QALM_DEC2025_STEM + "_result.qasm",
    }
    rows = []
    for (optimizer, mode), pattern in paths.items():
        if optimizer == "qalm_dec2025":
            params = QALM_DEC2025_PARAMS
        elif mode is None:
            params = DEFAULTS[optimizer]
        else:
            params = {"opt": mode}
        for c in man:
            qasm = os.path.join(NAM26, pattern.format(c=c))
            opt, status, source = _qasm_or_original(man, c, qasm)
            rows.append(
                _make_row("S26", man, c, optimizer, params, 3600, opt, status, source)
            )
    return rows


def load_legacy():
    """All legacy rows (sources 1-5), in the metrics.load_runs schema."""
    return (
        qalm_old_1h()
        + qalm_60s()
        + quartz_248()
        + guoq_queso_248("guoq")
        + guoq_queso_248("queso")
        + table26()
    )


def write_runs_csv(rows, out_dir):
    """Write rows as <out_dir>/runs.csv in the collect.py format."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["experiments"] = ";".join(r["experiments"])
            out["params"] = json.dumps(r["params"], sort_keys=True)
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory to write runs.csv into")
    args = ap.parse_args()
    rows = load_legacy()
    write_runs_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {os.path.join(args.out, 'runs.csv')}")


if __name__ == "__main__":
    main()
