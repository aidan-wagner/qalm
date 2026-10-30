"""Turn a finished job directory into one result row (+ time series, k events).

Result rule (same for every tool): the result is the best solution the tool
had reported at or before the budget T. For QALM/Quartz that is the last
progress change-point with t <= T (the final qasm is used to check the log);
for GUOQ/QUESO it is latest_sol_*.qasm (the process is killed at T); for VOQC
it is the output circuit. Runs with nothing reported count as the original
circuit. Infrastructure failures (global/slice OOM, missing result.json,
preemption) are returned as status INFRA and must be re-run, never reported.
"""
import json
import os

from parsers import parse_guoq, parse_native, parse_voqc

ROW_FIELDS = [
    "job_id", "experiments", "suite", "circuit", "input_sha256", "optimizer", "budget_s", "seed",
    "params", "orig_total", "orig_cx", "opt_total", "opt_cx", "result_source", "status",
    "exit_code", "oom_events", "oom_kill_events", "log_final_consistent", "improved_after_budget",
    "final_total", "final_cx", "greedy_end_s", "last_k", "n_changepoints", "peak_rss_kb",
    "cgroup_peak_bytes", "cpu_user_s", "cpu_sys_s", "wall_s", "host", "core", "attempt",
]


def _events(cg):
    ev = {}
    text = (cg or {}).get("memory.events") or ""
    for line in text.splitlines():
        k, v = line.split()
        ev[k] = int(v)
    return ev


def _greedy_end(job, kevents):
    """Time at which QALM's greedy phase ended: first [QALM_K_START] with k = greedy_k + 1."""
    if job["optimizer"] != "qalm":
        return None
    p = job["params"]
    if p["fixed_k"] > 0:
        target = p["fixed_k"]
    else:
        target = p["greedy_k"] + 1
    for k, t in kevents:
        if k == target:
            return t
    return float(job["budget_s"])  # never reached (same convention as plot_greedy_k.py)


def summarize(job, jobdir, launch):
    T = float(job["budget_s"])
    res_path = os.path.join(jobdir, "result.json")
    if job["kind"] == "docker":
        res_path = os.path.join(jobdir, "out", "result.json")
    row = {k: job.get(k) for k in ("job_id", "suite", "circuit", "input_sha256", "optimizer",
                                   "budget_s", "seed", "orig_total", "orig_cx")}
    row["experiments"] = ";".join(job["experiments"])
    row["params"] = json.dumps(job["params"], sort_keys=True)
    row["host"], row["core"], row["attempt"] = launch.get("host"), launch.get("core"), launch.get("attempt")
    if not os.path.exists(res_path):
        row["status"] = "INFRA"
        return row, [], []
    with open(res_path) as f:
        res = json.load(f)
    cg = res.get("cgroup", {})
    ev = _events(cg)
    row["oom_events"], row["oom_kill_events"] = ev.get("oom", 0), ev.get("oom_kill", 0)
    row["cgroup_peak_bytes"] = int(cg["memory.peak"]) if cg.get("memory.peak") else None
    # killed by a global or slice-level OOM (not this job's own cap): an infrastructure failure
    infra_oom = bool(row["oom_kill_events"]) and not row["oom_events"]

    if job["kind"] == "native":
        row["exit_code"] = res["exit_code"]
        row["peak_rss_kb"], row["cpu_user_s"], row["cpu_sys_s"], row["wall_s"] = (
            res["ru_maxrss_kb"], res["utime_s"], res["stime_s"], res["wall_s"])
        parsed = parse_native(os.path.join(jobdir, "stdout.log"), job["circuit"])
        final_qasm = os.path.join(jobdir, "final.qasm")
        if parsed["extra"].get("final_text") and not os.path.exists(final_qasm):
            with open(final_qasm, "w") as f:  # kept for equivalence checks (gitignored)
                f.write(parsed["extra"]["final_text"])
        # did the program's own clock pass the budget before it died? (from the flushed log)
        passed_T = (parsed["extra"]["timeout_lines"] > 0
                    or any(p[0] >= T for p in parsed["series"])
                    or any(t >= T for _, t in parsed["kevents"])
                    or any(g[1] >= T for g in parsed["greedy"]))
        if infra_oom:
            status = "INFRA"
        elif row["oom_events"]:
            # an OOM after T does not change the result (best reported at or before T)
            status = "OOM_AFTER_T" if passed_T else "OOM_CAP"
        elif res["wall_kill"]:
            status = "WALL_KILL"
        elif res["exit_code"] == 0:
            status = "OK"
        else:
            status = "CRASH"
    else:
        run = res.get("run", {})
        row["exit_code"] = run.get("exit_code")
        row["peak_rss_kb"], row["cpu_user_s"], row["cpu_sys_s"], row["wall_s"] = (
            run.get("ru_maxrss_kb"), run.get("utime_s"), run.get("stime_s"), run.get("wall_s"))
        out_dir = os.path.join(jobdir, "out")
        if job["optimizer"] == "voqc":
            parsed = parse_voqc(out_dir, res)
        else:
            parsed = parse_guoq(out_dir, res)
        # `timeout -k` sends SIGKILL if the tool ignores SIGTERM at the budget: still a timeout
        timed_out = row["exit_code"] == 124 or (row["exit_code"] in (-9, 137) and (row["wall_s"] or 0) >= T)
        if infra_oom or parsed["extra"].get("final_truncated") or parsed["extra"].get("final_parse_error"):
            status = "INFRA"  # global/slice OOM, or an output cut mid-write: re-run
        elif res.get("entry_error"):
            status = "CRASH"
        elif row["oom_events"] or (launch.get("docker_oom_killed") and not ev):
            status = "OOM_CAP"
        elif parsed["extra"].get("java_oom"):
            status = "JVM_OOM"
        elif job["optimizer"] == "voqc":
            status = "OK" if row["exit_code"] == 0 else ("TIMEOUT" if timed_out else "CRASH")
        else:
            status = "OK" if row["exit_code"] == 0 or timed_out else "CRASH"
    row["status"] = status

    series = parsed["series"]
    within = [p for p in series if p[0] <= T]
    final = parsed["final"]
    row["final_total"] = final["total"] if final else None
    row["final_cx"] = final["cx"] if final else None
    row["n_changepoints"] = len(series)
    row["improved_after_budget"] = len(within) < len(series)
    row["log_final_consistent"] = (bool(series) and final is not None
                                   and (series[-1][1], series[-1][2]) == (final["total"], final["cx"]))
    if job["optimizer"] in ("qalm", "qalm_old", "quartz"):
        if within:
            row["opt_total"], row["opt_cx"], row["result_source"] = within[-1][1], within[-1][2], "log_le_budget"
        elif final is not None and not series:
            row["opt_total"], row["opt_cx"], row["result_source"] = final["total"], final["cx"], "final_qasm"
        else:
            row["opt_total"], row["opt_cx"], row["result_source"] = job["orig_total"], job["orig_cx"], "original"
    else:
        if final is not None:
            row["opt_total"], row["opt_cx"], row["result_source"] = final["total"], final["cx"], "final_qasm"
        elif within:
            row["opt_total"], row["opt_cx"], row["result_source"] = within[-1][1], within[-1][2], "log_le_budget"
        else:
            row["opt_total"], row["opt_cx"], row["result_source"] = job["orig_total"], job["orig_cx"], "original"
    row["greedy_end_s"] = _greedy_end(job, parsed["kevents"])
    row["last_k"] = parsed["kevents"][-1][0] if parsed["kevents"] else None
    ts = [(job["job_id"], 0.0, job["orig_total"], job["orig_cx"])] + [
        (job["job_id"], t, a, b) for (t, a, b) in within]
    ks = [(job["job_id"], k, t) for (k, t) in parsed["kevents"] if t <= T]
    return row, ts, ks
