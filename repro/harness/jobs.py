"""Expand experiment YAML files into a deduplicated list of jobs.

A job is fully described by (optimizer, full parameter set, circuit name,
input sha256, budget, seed); its id is the first 12 hex digits of the sha1 of
that canonical description, so the same run shared by several experiments
(e.g. the default QALM config in E2/E3/E5) is executed once. The circuit name
is part of the key because the GUOQ suite contains 5 pairs of byte-identical
circuits (e.g. 9symml_195 / sym9_193) that the paper counts separately.

Native jobs also carry the campaign's `native_build` tag (when it is set), which
is part of their key: re-running the native optimizers on a new build gives new
ids while docker jobs keep theirs.
"""
import csv
import hashlib
import itertools
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)

# Full default parameters per optimizer. Every job records all of them.
DEFAULTS = {
    "qalm": {"greedy_k": 2, "n_pool": 1, "n_branch": 3, "repeat_tol": 1.5, "exp_increase": 0,
             "strictly_reducing": 0, "local_only": 1, "two_way": 0, "fixed_k": 0,
             "enqueue": 0, "cost_mode": 0},
    # BEST_CONFIG = (1, 1, 3, 1.5, 0, 0, 1, 1, 0) of benchmark_guoq.py@b10f6ec, via test_qalm
    "qalm_old": {"n_pool": 1, "n_branch": 1, "k": 3, "repeat_tol": 1.5, "exp_increase": 0,
                 "strictly_reducing": 0, "local_only": 1, "preprocess": 1, "two_way": 0},
    "quartz": {},
    "guoq": {"opt": "TWO_Q"},
    "queso": {"opt": "TWO_Q"},
    "voqc": {},
}
KIND = {"qalm": "native", "qalm_old": "native", "quartz": "native",
        "guoq": "docker", "queso": "docker", "voqc": "docker"}
# Optimizers whose result does not depend on a seed get seed 0 only.
SEEDED = {"qalm", "guoq", "queso"}
# Runtime library of the native binaries when campaign.yaml has no `lib`.
DEFAULT_LIB = "build/libquartz_runtime.so"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_campaign(path=None):
    cfg = load_yaml(path or os.path.join(REPRO, "config", "campaign.yaml"))
    # repo_root defaults to this checkout; a relative runs_dir is relative to it
    cfg["repo_root"] = os.path.abspath(cfg.get("repo_root") or os.path.dirname(REPRO))
    if os.environ.get("REPRO_RUNS_DIR"):  # e.g. a separate runs dir for a test plan
        cfg["runs_dir"] = os.environ["REPRO_RUNS_DIR"]
    cfg["runs_dir"] = os.path.join(cfg["repo_root"], cfg.get("runs_dir", "repro/runs"))
    return cfg


def load_suite(name):
    """Manifest rows of a suite; their paths (relative to the repository root) are made absolute."""
    with open(os.path.join(REPRO, "inputs", f"{name}.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = os.path.join(os.path.dirname(REPRO), r["path"])
    return rows


def lib_path(campaign):
    return campaign.get("lib", DEFAULT_LIB)


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _expand_params(params):
    keys = sorted(params)
    for combo in itertools.product(*(_as_list(params[k]) for k in keys)):
        yield dict(zip(keys, combo))


def argv_for(job, campaign):
    """Command line for a native job (run with cwd = repo root)."""
    opt, p, c = job["optimizer"], job["params"], job["circuit"]
    T = str(job["budget_s"])
    bins, ecc = campaign["binaries"], campaign["ecc"]
    if opt == "qalm":
        return [bins["qalm"], job["input_path"], c, T, str(p["greedy_k"]), str(p["n_pool"]),
                str(p["n_branch"]), str(p["repeat_tol"]), str(p["exp_increase"]),
                str(p["strictly_reducing"]), str(p["local_only"]), str(p["two_way"]), ecc,
                str(p["fixed_k"]), str(p["enqueue"]), str(p["cost_mode"]), str(job["seed"])]
    if opt == "qalm_old":
        return [bins["qalm_old"], job["input_path"], c, T, str(p["n_pool"]), str(p["n_branch"]),
                str(p["k"]), str(p["repeat_tol"]), str(p["exp_increase"]),
                str(p["strictly_reducing"]), str(p["local_only"]), str(p["preprocess"]),
                str(p["two_way"]), ecc]
    if opt == "quartz":
        return [bins["quartz"], job["input_path"], c, T]
    raise ValueError(f"{opt} is not a native optimizer")


def docker_args_for(job):
    """Arguments passed to repro/baselines/entry.py inside the container."""
    mode = job["params"].get("opt", "-")
    return [job["optimizer"], mode, "/in/" + os.path.basename(job["input_path"]),
            str(job["budget_s"]), str(job["seed"])]


def job_id(job):
    key = {k: job[k] for k in ("optimizer", "params", "circuit", "input_sha256", "budget_s", "seed")}
    if "build" in job:
        key["build"] = job["build"]
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def _select_circuits(path, suite, names):
    """Suite rows, restricted to `names` (a `circuits:` list of the config) if given."""
    rows = load_suite(suite)
    if names is None:
        return rows
    wanted = set(names)
    unknown = wanted - {c["name"] for c in rows}
    if unknown:
        raise ValueError(f"{path}: circuits {sorted(unknown)} are not in suite {suite}")
    return [c for c in rows if c["name"] in wanted]


def expand(config_paths, campaign=None):
    """Jobs of the experiment files. Optional per file or per run: `circuits:` (names in the suite)."""
    build = (campaign or {}).get("native_build")
    jobs = {}
    for path in config_paths:
        cfg = load_yaml(path)
        exp = cfg["experiment"]
        seeds = cfg.get("seeds", [0])
        for run in cfg["runs"]:
            opt = run["optimizer"]
            suite = run.get("suite", cfg["suite"])
            circuits = _select_circuits(path, suite, run.get("circuits", cfg.get("circuits")))
            for budget in _as_list(run["budget_s"]):
                for params in _expand_params(run.get("params", {})):
                    full = dict(DEFAULTS[opt])
                    unknown = set(params) - set(full)
                    if unknown:
                        raise ValueError(f"{path}: unknown params {unknown} for {opt}")
                    full.update(params)
                    for seed in (seeds if opt in SEEDED else [0]):
                        for c in circuits:
                            job = {"optimizer": opt, "kind": KIND[opt], "params": full,
                                   "suite": suite, "circuit": c["name"],
                                   "input_path": c["path"], "input_sha256": c["sha256"],
                                   "orig_total": int(c["total"]), "orig_cx": int(c["cx"]),
                                   "n_qubits": int(c["n_qubits"]), "budget_s": int(budget),
                                   "seed": seed, "priority": run.get("priority", 1)}
                            if build and KIND[opt] == "native":
                                job["build"] = build
                            jid = job_id(job)
                            if jid in jobs:
                                jobs[jid]["experiments"] = sorted(
                                    set(jobs[jid]["experiments"]) | {exp})
                                jobs[jid]["priority"] = min(jobs[jid]["priority"], job["priority"])
                            else:
                                job["job_id"] = jid
                                job["experiments"] = [exp]
                                jobs[jid] = job
    return list(jobs.values())
