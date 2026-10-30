"""Campaign CLI. Run from the repo root.

  python3 repro/harness/run.py plan [--configs repro/config/E*.yaml] [--experiments E3_greedy ...]
                                    [--circuits a,b] [--budget-override 60] [--shard i/n] -o PLAN
  python3 repro/harness/run.py launch --plan PLAN [--max-running N]     (usually via launch_sched.sh)
  python3 repro/harness/run.py status
  python3 repro/harness/run.py drain | resume | stop
  python3 repro/harness/run.py collect [--plan PLAN] [--out repro/results]
"""
import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs as jobsmod  # noqa: E402
from collect import ROW_FIELDS, summarize  # noqa: E402
from sched import Priors, Scheduler, order_plan  # noqa: E402

CFG = jobsmod.load_campaign()
RUNS = CFG["runs_dir"]


def cmd_plan(a):
    configs = a.configs or sorted(glob.glob(os.path.join(jobsmod.REPRO, "config", "E*.yaml")))
    jobs = jobsmod.expand(configs)
    if a.experiments:
        jobs = [j for j in jobs if set(j["experiments"]) & set(a.experiments)]
    if a.circuits:
        keep = set(a.circuits.split(","))
        jobs = [j for j in jobs if j["circuit"] in keep]
    if a.optimizers:
        keep = set(a.optimizers.split(","))
        jobs = [j for j in jobs if j["optimizer"] in keep]
    if a.budget_override:
        # smoke tests only: shorter budget -> different job ids, never mixed with campaign results
        dedup = {}
        for j in jobs:
            j["budget_s"] = a.budget_override
            j["job_id"] = jobsmod.job_id(j)
            dedup.setdefault(j["job_id"], j)
        jobs = list(dedup.values())
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        jobs = [j for j in jobs
                if int(hashlib.sha1(j["circuit"].encode()).hexdigest(), 16) % n == i]
    priors = Priors(os.path.join(jobsmod.REPRO, "config", "mem_priors.csv"))
    jobs = order_plan(jobs, CFG, priors)
    with open(a.output, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    hours = sum(j["budget_s"] for j in jobs if j["optimizer"] != "voqc") / 3600
    print(f"{len(jobs)} jobs, <= {hours:.0f} core-hours -> {a.output}")


def cmd_launch(a):
    def log(msg):
        print(time.strftime("%Y-%m-%dT%H:%M:%S"), msg, flush=True)
    Scheduler(a.plan, max_running=a.max_running, log=log).run()


def cmd_flag(name, present):
    path = os.path.join(RUNS, name)
    if present:
        open(path, "w").close()
    elif os.path.exists(path):
        os.remove(path)
    print(f"{name} {'set' if present else 'cleared'}")


def cmd_status(a):
    st = os.path.join(RUNS, "status.json")
    if os.path.exists(st):
        print(json.load(open(st)))
    counts = {}
    for d in glob.glob(os.path.join(RUNS, "jobs", "*", "done.json")):
        s = json.load(open(d))["status"]
        counts[s] = counts.get(s, 0) + 1
    infra = len(glob.glob(os.path.join(RUNS, "jobs", "*.infra*")))
    esc = len(glob.glob(os.path.join(RUNS, "jobs", "*.esc*")))
    pre = len(glob.glob(os.path.join(RUNS, "jobs", "*.pre*")))
    print("finished:", counts, "| re-runs: infra", infra, "memory-cap escalations", esc, "preemptions", pre)


def cmd_collect(a):
    plan = [json.loads(l) for l in open(a.plan)]
    os.makedirs(a.out, exist_ok=True)
    rows, ts, ks = [], [], []
    for job in plan:
        d = os.path.join(RUNS, "jobs", job["job_id"])
        if not os.path.exists(os.path.join(d, "done.json")):
            continue
        launch = json.load(open(os.path.join(d, "launch.json")))
        try:
            row, t, k = summarize(job, d, launch)
        except Exception as e:  # keep going; the row shows the problem
            print(f"collect error {job['job_id']}: {e!r}")
            row, t, k = {f: job.get(f) for f in ("job_id", "suite", "circuit", "optimizer", "budget_s", "seed")}, [], []
            row["status"] = "COLLECT_ERROR"
        rows.append(row)
        ts.extend(t)
        ks.extend(k)
        for log in (os.path.join(d, "stdout.log"), os.path.join(d, "out", "tool.out")):
            if os.path.exists(log) and a.gzip_logs:
                with open(log, "rb") as src, gzip.open(log + ".gz", "wb") as dst:
                    dst.writelines(src)
                os.remove(log)
    rows.sort(key=lambda r: r["job_id"])
    with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with gzip.open(os.path.join(a.out, "timeseries.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "t_s", "total", "cx"])
        w.writerows(sorted(ts, key=lambda r: (r[0], r[1])))  # stable: keep log order at equal t
    with open(os.path.join(a.out, "kevents.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "k", "t_s"])
        w.writerows(sorted(ks, key=lambda r: (r[0], r[2])))
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"collected {len(rows)}/{len(plan)} jobs -> {a.out}: {statuses}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--configs", nargs="*")
    s.add_argument("--experiments", nargs="*")
    s.add_argument("--circuits")
    s.add_argument("--optimizers")
    s.add_argument("--budget-override", type=int)
    s.add_argument("--shard")
    s.add_argument("-o", "--output", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--plan", required=True)
    s.add_argument("--max-running", type=int)
    sub.add_parser("status")
    sub.add_parser("drain")
    sub.add_parser("resume")
    sub.add_parser("stop")
    s = sub.add_parser("collect")
    s.add_argument("--plan", required=True)
    s.add_argument("--out", default=os.path.join(jobsmod.REPRO, "results"))
    s.add_argument("--gzip-logs", action="store_true")
    a = p.parse_args()
    if a.cmd == "plan":
        cmd_plan(a)
    elif a.cmd == "launch":
        cmd_launch(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "drain":
        cmd_flag("DRAIN", True)
    elif a.cmd == "resume":
        cmd_flag("DRAIN", False)
        cmd_flag("STOP", False)
    elif a.cmd == "stop":
        cmd_flag("STOP", True)
    elif a.cmd == "collect":
        cmd_collect(a)


if __name__ == "__main__":
    main()
