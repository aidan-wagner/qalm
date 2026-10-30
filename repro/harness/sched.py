"""Memory-aware scheduler: one job per physical core, survives restarts.

State lives in <runs_dir>/jobs/<job_id>/:
  job.json      what to run (written at launch)
  launch.json   where/when it was launched (core, attempt, memory cap, CPU mode, unit or container)
  result.json   written by jobwrap.py (native) or out/result.json (docker)
  done.json     written here after reaping; its presence marks the job finished
A job that must be re-run is moved to jobs/<job_id>.<reason><n> and re-queued at the
head: reason "infra" (missing/unusable result, global OOM), "pre" (preempted by the
memory watchdog) or "esc" (a native job reached a memory cap below the maximum and is
re-run with the next larger cap). After 3 infra attempts a job is given up (INFRA_GAVE_UP).

Memory policy:
  * every job gets a hard cap; native jobs start at the smallest tier that covers
    1.5x their expected peak and escalate (re-run) if they reach it (or reach 90% of it
    in RSS); containers get the maximum tier. A run that stays well below its cap runs as
    it would under the maximum cap.
  * admission is time-aware and NUMA-aware: each running job's usage is projected forward
    (growing at least as fast as its expected peak spread over its budget, or its measured
    slope, never above its cap); a job is admitted only if the projected sum stays within
    the budget at every future job-end time, both machine-wide and on its NUMA node. Part
    of the far-future budget is held back so that start times spread out instead of
    starting in lockstep waves.
  * a watchdog preempts the newest job if MemAvailable still gets too low.
Safety: listings that fail are treated as "unknown" (nothing is reaped), a job is only
considered finished once its cgroup is gone, cores used by any live cgroup in the slice
are never reused, and only one scheduler may own a runs dir.
"""
import csv
import fcntl
import hashlib
import json
import os
import random
import socket
import subprocess
import time
import traceback

import jobs as jobsmod
from collect import summarize

REPO = jobsmod.load_campaign()["repo_root"]
GIB = 2**30
SLICE_CG = "/sys/fs/cgroup/qalmrepro.slice"
WANT_CPU_MODE = {"boost": "0", "governors": ["performance"]}
MAX_INFRA_ATTEMPTS = 3


def parse_cpuset(spec):
    out = []
    for part in spec.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable")


def sh(cmd, timeout=120):
    """Run a command; never raises. A timeout is reported as returncode -1."""
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, -1, "", f"timeout after {timeout}s")


def docker(*args, timeout=60):
    # the timeout runs inside sudo so that it can kill a hung docker CLI
    return sh(["sudo", "-n", "timeout", "-k", "10", str(timeout), "docker", *args], timeout=timeout + 30)


def write_json(path, obj, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_mode(cores):
    with open("/sys/devices/system/cpu/cpufreq/boost") as f:
        boost = f.read().strip()
    govs = set()
    for c in cores:
        with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor") as f:
            govs.add(f.read().strip())
    return {"boost": boost, "governors": sorted(govs)}


def is_short(job):
    return job["budget_s"] <= 300


def is_quick(job):
    """Jobs that finish within minutes: launched back to back."""
    return is_short(job) or job["optimizer"] == "voqc"


class Priors:
    """Expected peak memory (GiB) of a job before it starts."""

    def __init__(self, path):
        self.table = {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                self.table[(r["family"], r["circuit"])] = float(r["prior_gib"])
        self.learned = {}

    @staticmethod
    def family(job):
        return {"qalm": "qalm", "qalm_old": "qalm", "quartz": "quartz"}.get(job["optimizer"], job["optimizer"])

    def get(self, job):
        fam = self.family(job)
        if is_short(job) and fam == "qalm" and ("qalm_short", job["circuit"]) in self.table:
            # a previous 60 s run of this circuit: use its own peak (times the scale, for tests)
            base = max(self.table[("qalm_short", job["circuit"])], self.learned.get(("qalm_short", job["circuit"]), 0.0))
            return min(base * float(os.environ.get("REPRO_PRIOR_SCALE", "1")), 32.0)
        key = (fam, job["circuit"])
        base = max(self.table.get(key, 0.0), self.learned.get(key, 0.0))
        if base == 0.0:
            if fam in ("guoq", "queso"):
                base = 6.0 if job["orig_total"] < 100_000 else 21.0   # JVM is started with -Xmx20g
            elif fam == "voqc":
                base = 2.0 if job["orig_total"] < 100_000 else 4.0  # observed < 1 GB on the largest
            else:
                base = 4.0
        if is_short(job):
            base = min(3.0, max(0.5, 0.4 * base))  # 60 s runs peaked at 2.3 GB in earlier runs
        return min(base * float(os.environ.get("REPRO_PRIOR_SCALE", "1")), 32.0)  # scale: tests only

    def learn(self, job, peak_bytes):
        if not peak_bytes:
            return
        fam = self.family(job)
        if is_short(job):
            if fam != "qalm":
                return
            fam = "qalm_short"
        key = (fam, job["circuit"])
        self.learned[key] = max(self.learned.get(key, 0.0), 1.1 * peak_bytes / GIB)

    def dump(self):
        return {f"{f}|{c}": v for (f, c), v in self.learned.items()}

    def load(self, d):
        self.learned = {tuple(k.split("|", 1)): v for k, v in d.items()}


class Scheduler:
    def __init__(self, plan_path, max_running=None, log=print):
        self.cfg = jobsmod.load_campaign()
        self.runs = self.cfg["runs_dir"]
        self.jobs_dir = os.path.join(self.runs, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        self.plan = [json.loads(l) for l in open(plan_path)]
        self.by_id = {j["job_id"]: j for j in self.plan}
        self.cores = parse_cpuset(self.cfg["cores"]["jobs"])
        self.split = self.cfg["numa_of_core"]["split"]
        self.max_running = max_running or len(self.cores)
        self.mem = self.cfg["mem"]
        tiers = os.environ.get("REPRO_CAP_TIERS_GIB")  # tests only
        self.tiers = [float(t) for t in tiers.split(",")] if tiers else [float(t) for t in self.mem["cap_tiers_gib"]]
        self.max_cap = self.tiers[-1]
        self.horizon = max(j["budget_s"] for j in self.plan)
        self.priors = Priors(os.path.join(REPO, "repro", "config", "mem_priors.csv"))
        self.learn_path = os.path.join(self.runs, "learned_priors.json")
        if os.path.exists(self.learn_path):
            self.priors.load(json.load(open(self.learn_path)))
        self.host = socket.gethostname()
        self.log = log
        self.running = {}     # job_id -> dict(core, kind, name, start, prior, cap, samples)
        self.queue = []
        self.last_launch = 0.0
        self.events_path = os.path.join(self.runs, "events.csv")
        self.esc_path = os.path.join(self.runs, "escalations.json")
        self.min_cap = json.load(open(self.esc_path)) if os.path.exists(self.esc_path) else {}
        self.head_wait_since = None
        self.allow_cpu_mode = os.environ.get("REPRO_ALLOW_CPU_MODE") == "1"  # smoke tests only
        self.cpu_mode_ok = True
        self.binary_hashes = {}

    # ---------- bookkeeping ----------
    def jobdir(self, jid):
        return os.path.join(self.jobs_dir, jid)

    def event(self, kind, jid="", **kw):
        new = not os.path.exists(self.events_path)
        with open(self.events_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["utc", "host", "event", "job_id", "core", "running", "mem_available_gib", "info"])
            w.writerow([f"{time.time():.1f}", self.host, kind, jid, kw.get("core", ""), len(self.running),
                        f"{mem_available_gib():.1f}", json.dumps(kw.get("info", ""))])

    def flag(self, name):
        return os.path.exists(os.path.join(self.runs, name))

    def numa(self, core):
        return 0 if core < self.split else 1

    def unit(self, jid):
        return f"qalmrepro-{jid}"

    def cap_for(self, job, prior):
        """Smallest cap tier covering 1.5x the expected peak (containers: maximum tier)."""
        if job["kind"] == "docker":
            return self.max_cap
        floor = max(1.5 * prior, self.min_cap.get(job["job_id"], 0.0))
        return next((t for t in self.tiers if t >= floor), self.max_cap)

    def past_attempts(self, jid, reason=None):
        prefix = jid + "." + (reason or "")
        return sum(1 for n in os.listdir(self.jobs_dir) if n.startswith(prefix))

    # ---------- startup ----------
    def preflight(self):
        self._lock = open(os.path.join(self.runs, "sched.lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another scheduler already owns {self.runs}")
        ecc = os.path.join(REPO, self.cfg["ecc"])
        md5 = hashlib.md5(open(ecc, "rb").read()).hexdigest()
        assert md5 == self.cfg["ecc_md5"], f"ECC md5 mismatch {md5}"
        assert not os.path.exists(os.path.join(REPO, "benchmark-logs")), "benchmark-logs/ must not exist"
        assert not os.path.exists(os.path.join(REPO, "libquartz_runtime.so")), \
            "repo-root libquartz_runtime.so would shadow build/repro (RUNPATH order)"
        recorded = {}
        in_block = False
        for line in open(os.path.join(REPO, "repro", "env", "build_info.txt")):
            if line.startswith("## binaries"):
                in_block = True
                continue
            if in_block and line.startswith("##"):
                break
            if in_block and line.strip():
                h, p = line.split()
                recorded[p] = h
        for p in list(self.cfg["binaries"].values()) + ["build/repro/libquartz_runtime.so"]:
            h = sha256(os.path.join(REPO, p))
            assert recorded.get(p) == h, f"{p} differs from repro/env/build_info.txt (rebuild?)"
            self.binary_hashes[p] = h
        assert all("nam-benchmarks" not in j["input_path"] for j in self.plan)
        assert len(self.by_id) == len(self.plan), "duplicate job ids in plan"
        out = sh(["systemctl", "show", "qalmrepro.slice", "-p", "ActiveState", "-p", "MemoryMax",
                  "-p", "AllowedCPUs"]).stdout
        p = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        assert p.get("ActiveState") == "active", "qalmrepro.slice not active (use launch_sched.sh)"
        assert p.get("MemoryMax") == str(self.mem["slice_cap_gib"] * GIB), f"slice MemoryMax {p.get('MemoryMax')}"
        assert set(parse_cpuset(p.get("AllowedCPUs", ""))) == set(self.cores), \
            f"slice AllowedCPUs {p.get('AllowedCPUs')} != job cores"
        if not self.allow_cpu_mode:
            mode = cpu_mode(self.cores)
            assert mode == WANT_CPU_MODE, f"CPU mode {mode}; run repro/env/set_cpu_mode.sh fixed"
            hk = set(parse_cpuset(self.cfg["cores"]["housekeeping"]))
            for u in ("user.slice", "system.slice", "init.scope"):
                eff = sh(["systemctl", "show", u, "-p", "EffectiveCPUs", "--value"]).stdout.strip()
                assert set(parse_cpuset(eff)) <= hk, f"{u} may run on job cores ({eff}); use launch_sched.sh"
        else:
            self.log("WARNING: REPRO_ALLOW_CPU_MODE=1 (CPU mode and isolation not enforced; smoke tests only)")
        if os.path.exists(os.path.join(self.runs, "STOP")):
            os.remove(os.path.join(self.runs, "STOP"))

    def active_units(self):
        """Set of active qalmrepro units, or None if systemd could not be asked."""
        r = sh(["systemctl", "list-units", "qalmrepro-*", "--all", "--no-legend", "--no-pager",
                "--plain", "--state=active,activating,deactivating"], timeout=60)
        return set(r.stdout.split()) if r.returncode == 0 else None

    def docker_states(self):
        """{container name: state}, or None if docker could not be asked."""
        r = docker("ps", "-a", "--filter", "name=qalmrepro-", "--format", "{{.Names}} {{.State}}")
        if r.returncode != 0:
            return None
        return dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)

    def slice_busy_cores(self):
        """Cores of every live cgroup in the slice (jobs of any plan, or leftovers)."""
        busy = set()
        try:
            names = os.listdir(SLICE_CG)
        except OSError:
            return busy
        for n in names:
            try:
                with open(os.path.join(SLICE_CG, n, "cpuset.cpus.effective")) as f:
                    spec = f.read().strip()
            except OSError:
                continue
            cores = parse_cpuset(spec)
            if len(cores) <= 2:  # a job cgroup (not an intermediate cgroup spanning the slice)
                busy.update(cores)
        return busy

    def recover(self):
        """Re-attach to jobs still running from a previous scheduler instance."""
        active_units = self.active_units()
        containers = self.docker_states()
        if active_units is None or containers is None:
            raise SystemExit("cannot list units/containers; refusing to guess what is running")
        for jid in self.by_id:
            d = self.jobdir(jid)
            if os.path.exists(os.path.join(d, "done.json")) or not os.path.exists(os.path.join(d, "launch.json")):
                continue
            try:
                launch = json.load(open(os.path.join(d, "launch.json")))
            except json.JSONDecodeError:
                os.rename(d, f"{d}.infra{self.past_attempts(jid, 'infra') + 1}")
                continue
            alive = (f"{self.unit(jid)}.service" in active_units) if launch["kind"] == "native" \
                else containers.get(launch["name"]) == "running"
            self.running[jid] = {"core": launch["core"], "kind": launch["kind"], "name": launch["name"],
                                 "start": launch["start_unix"], "prior": launch["prior_gib"],
                                 "cap": launch.get("cap_gib", self.max_cap), "samples": []}
            self.log(f"recovered {jid} on core {launch['core']} alive={alive}")
        adopted_units = {f"{i['name']}.service" for i in self.running.values() if i["kind"] == "native"}
        adopted_ctrs = {i["name"] for i in self.running.values() if i["kind"] == "docker"}
        stray = sorted(u for u in active_units if u.startswith("qalmrepro-") and u.endswith(".service")
                       and u != "qalmrepro-sched.service" and u not in adopted_units)
        stray += sorted(n for n, st in containers.items()
                        if st in ("running", "created", "restarting") and n not in adopted_ctrs)
        if stray:
            raise SystemExit(f"jobs not belonging to this plan/runs dir are running: {stray[:10]}")
        done = {jid for jid in self.by_id if os.path.exists(os.path.join(self.jobdir(jid), "done.json"))}
        self.queue = [j for j in self.plan if j["job_id"] not in done and j["job_id"] not in self.running]

    # ---------- launching ----------
    def wall_limit(self, job):
        m = self.cfg["wall_margin_s"]
        return job["budget_s"] + (m["short"] if is_short(job) else m["default"])

    def launch(self, job, core, prior, cap, mode):
        jid = job["job_id"]
        d = self.jobdir(jid)
        attempt = 1 + self.past_attempts(jid)
        os.makedirs(d, exist_ok=True)
        for stale in ("result.json", "done.json"):  # never mix files from another attempt
            if os.path.exists(os.path.join(d, stale)):
                os.rename(os.path.join(d, stale), os.path.join(d, f"{stale}.stale{int(time.time())}"))
        spec = dict(job, repo_root=REPO, wall_limit_s=self.wall_limit(job))
        cap_str = f"{int(cap * 1024)}M"
        if job["kind"] == "native":
            # a unit killed moments ago may still be loaded in "failed" state
            sh(["sudo", "-n", "systemctl", "reset-failed", f"{self.unit(jid)}.service"])
            spec["argv"] = jobsmod.argv_for(job, self.cfg)
            write_json(os.path.join(d, "job.json"), spec, indent=1)
            cmd = ["sudo", "-n", "systemd-run", f"--unit={self.unit(jid)}", "--slice=qalmrepro.slice",
                   "--uid=cc", "--gid=cc", "--collect", "--no-block", "--quiet",
                   f"--working-directory={REPO}",
                   "-p", f"AllowedCPUs={core}", "-p", f"MemoryMax={cap_str}",
                   "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "-p", "OOMScoreAdjust=500",
                   "-p", "LimitCORE=0", "-p", f"RuntimeMaxSec={spec['wall_limit_s'] + 600}",
                   "-p", f"StandardOutput=append:{d}/wrap.log", "-p", f"StandardError=append:{d}/wrap.log",
                   "/usr/bin/python3", "-I", os.path.join(REPO, "repro/harness/jobwrap.py"), d]
            name = self.unit(jid)
        else:
            spec["docker_args"] = jobsmod.docker_args_for(job)
            write_json(os.path.join(d, "job.json"), spec, indent=1)
            os.makedirs(os.path.join(d, "out"), exist_ok=True)
            name = f"qalmrepro-{jid}"
            docker("rm", "-f", name)
            cmd = ["sudo", "-n", "timeout", "-k", "10", "300", "docker", "run", "-d", "--name", name,
                   "--cgroup-parent=qalmrepro.slice",
                   f"--cpuset-cpus={core}", f"--memory={cap_str.lower()}",
                   f"--memory-swap={cap_str.lower()}", "--network=none", "--ulimit", "core=0",
                   "--oom-score-adj=500",
                   "-v", f"{os.path.dirname(job['input_path'])}:/in:ro",
                   "-v", f"{os.path.join(REPO, 'repro/baselines')}:/repro:ro",
                   "-v", f"{d}/out:/out", "-w", "/out", self.cfg["docker_image"],
                   "/opt/conda/bin/python", "/repro/entry.py", *spec["docker_args"]]
        now = time.time()
        launch = {"job_id": jid, "kind": job["kind"], "name": name, "core": core, "numa": self.numa(core),
                  "host": self.host, "attempt": attempt, "start_unix": now, "prior_gib": prior,
                  "cap_gib": cap, "cpu_mode": mode, "binaries": self.binary_hashes,
                  "docker_image": self.cfg["docker_image"], "cmd": cmd}
        write_json(os.path.join(d, "launch.json"), launch, indent=1)
        r = sh(cmd, timeout=330)
        if r.returncode != 0:
            # never delete the directory: only drop this attempt's launch records; the core
            # stays blocked while anything of this job is still alive (slice_busy_cores)
            self.log(f"launch failed {jid}: {r.stderr.strip()[:300]}")
            for name_ in ("launch.json", "job.json"):
                if os.path.exists(os.path.join(d, name_)):
                    os.rename(os.path.join(d, name_), os.path.join(d, f"{name_}.failed{int(now)}"))
            self.event("launch_failed", jid, core=core, info=r.stderr.strip()[:300])
            return False
        self.running[jid] = {"core": core, "kind": job["kind"], "name": name, "start": now,
                             "prior": prior, "cap": cap, "samples": []}
        self.last_launch = now
        self.event("launch", jid, core=core, info={"prior_gib": prior, "cap_gib": cap, "attempt": attempt})
        return True

    # ---------- monitoring ----------
    def cgroup_dir(self, info):
        if info["kind"] == "native":
            return f"{SLICE_CG}/{info['name']}.service"
        if "cid" not in info:
            r = docker("inspect", "-f", "{{.Id}}", info["name"])
            if r.returncode != 0:
                return None
            info["cid"] = r.stdout.strip()
        return f"{SLICE_CG}/docker-{info['cid']}.scope"

    def sample_memory(self):
        now = time.time()
        for info in self.running.values():
            cg = self.cgroup_dir(info)
            try:
                with open(os.path.join(cg, "memory.current")) as f:
                    info["samples"].append((now, int(f.read()) / GIB))
            except (OSError, TypeError):
                continue
            info["samples"] = [s for s in info["samples"] if now - s[0] <= 600]

    def reap(self):
        units = self.active_units()
        if units is None:
            self.log("systemctl list-units failed; not reaping this round")
            return
        containers = None
        if any(i["kind"] == "docker" for i in self.running.values()):
            containers = self.docker_states()
            if containers is None:
                self.log("docker ps failed; not reaping containers this round")
        for jid in list(self.running):
            info = self.running[jid]
            if time.time() - info["start"] < 15:
                continue  # give systemd/docker time to register a just-launched job
            if info["kind"] == "native":
                if f"{info['name']}.service" in units or os.path.exists(f"{SLICE_CG}/{info['name']}.service"):
                    continue  # still alive (the cgroup check needs no D-Bus)
                docker_info = {}
            else:
                if containers is None:
                    continue
                if containers.get(info["name"]) in ("running", "created", "restarting"):
                    continue
                if info["name"] not in containers:
                    docker_info = {}  # container vanished: summarize decides (no result -> INFRA)
                else:
                    r = docker("inspect", "-f",
                               "{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}} "
                               "{{.State.StartedAt}} {{.State.FinishedAt}}", info["name"])
                    parts = r.stdout.split()
                    if r.returncode != 0 or len(parts) < 5 or parts[0] != "false":
                        continue  # not sure yet; look again next round
                    docker_info = {"docker_oom_killed": parts[1] == "true", "docker_exit": int(parts[2]),
                                   "docker_started": parts[3], "docker_finished": parts[4]}
                    docker("rm", info["name"])
            self.finish(jid, docker_info)

    def requeue(self, jid, reason, info):
        d = self.jobdir(jid)
        n = self.past_attempts(jid, reason) + 1
        os.rename(d, f"{d}.{reason}{n}")
        self.queue.insert(0, self.by_id[jid])
        self.event("requeue", jid, core=info["core"], info={"reason": reason, "n": n})
        self.log(f"{reason} {jid} (#{n}) -> requeued")

    def done(self, jid, status, info, extra=None):
        rec = {"status": status, "finished_unix": time.time()}
        rec.update(extra or {})
        write_json(os.path.join(self.jobdir(jid), "done.json"), rec)
        self.event("finish", jid, core=info["core"], info=rec)

    def finish(self, jid, docker_info, preempted=False):
        info = self.running.pop(jid)
        job = self.by_id[jid]
        d = self.jobdir(jid)
        try:
            launch = json.load(open(os.path.join(d, "launch.json")))
            launch.update(docker_info)
            write_json(os.path.join(d, "launch.json"), launch, indent=1)
            if preempted:
                self.requeue(jid, "pre", info)
                return
            row, _, _ = summarize(job, d, launch)
        except Exception:
            self.log(f"collect error for {jid}:\n{traceback.format_exc()}")
            self.done(jid, "COLLECT_ERROR", info)
            return
        # learn from RSS: the cgroup peak also counts page cache (e.g. the job's own log file)
        rss_bytes = (row.get("peak_rss_kb") or 0) * 1024
        self.priors.learn(job, rss_bytes)
        write_json(self.learn_path, self.priors.dump())
        status = row["status"]
        if status == "INFRA":
            if self.past_attempts(jid, "infra") + 1 >= MAX_INFRA_ATTEMPTS:
                self.log(f"giving up on {jid} after {MAX_INFRA_ATTEMPTS} infra attempts")
                self.done(jid, "INFRA_GAVE_UP", info)
            else:
                self.requeue(jid, "infra", info)
            return
        cap = launch.get("cap_gib", self.max_cap)
        # Escalate if the run reached its cap before its budget ran out, or came close enough
        # (RSS >= 90% of the cap) that cgroup reclaim may have slowed it: every reported run below
        # the maximum cap then ran with headroom, i.e. as it would have under the maximum cap.
        near_cap = rss_bytes >= self.mem["escalate_rss_fraction"] * cap * GIB
        if (status == "OOM_CAP" or (near_cap and status != "OOM_AFTER_T")) and cap < self.max_cap:
            nxt = next((t for t in self.tiers if t > cap), self.max_cap)
            self.min_cap[jid] = nxt
            write_json(self.esc_path, self.min_cap)
            self.requeue(jid, "esc", info)
            return
        self.done(jid, status, info, {"cap_gib": cap, "rss_gib": round(rss_bytes / GIB, 3),
                                      "cgroup_peak_gib": round((row.get("cgroup_peak_bytes") or 0) / GIB, 3)})

    def kill(self, jid):
        info = self.running[jid]
        if info["kind"] == "native":
            r = sh(["sudo", "-n", "systemctl", "kill", "-s", "KILL", f"{info['name']}.service"])
        else:
            r = docker("kill", info["name"])
        return r.returncode == 0

    def wait_gone(self, info, seconds=30):
        cg = self.cgroup_dir(info)
        for _ in range(seconds):
            if cg is None or not os.path.exists(cg):
                return True
            time.sleep(1)
        return False

    def enforce_wall_limits(self):
        # Native jobs enforce their own wall limit in jobwrap.py; containers are killed here.
        now = time.time()
        for jid, info in list(self.running.items()):
            if info["kind"] == "docker" and now - info["start"] > self.wall_limit(self.by_id[jid]) + 120:
                self.log(f"wall limit: killing container {jid}")
                self.kill(jid)

    def watchdog(self):
        avail = mem_available_gib()
        if avail < self.mem["preempt_below_gib"] and self.running:
            newest = max(self.running, key=lambda j: self.running[j]["start"])
            info = self.running[newest]
            self.log(f"MemAvailable {avail:.1f} GiB: preempting newest job {newest}")
            if self.kill(newest) and self.wait_gone(info):
                peak = max((s[1] for s in info["samples"]), default=0.0)
                self.priors.learn(self.by_id[newest], peak * GIB)
                if info["kind"] == "docker":
                    docker("rm", "-f", info["name"])
                self.finish(newest, {}, preempted=True)
            else:
                self.log(f"could not confirm that {newest} was killed; will retry")
        return avail

    # ---------- admission ----------
    def end_time(self, job, start, now):
        end = start + job["budget_s"] + 60
        if now + 60 > end:  # already past its budget: may hold (and grow) memory until its wall limit
            end = start + self.wall_limit(job) + 120
        return max(end, now + 60)

    @staticmethod
    def usage(g, tau):
        """Projected usage (GiB) at time tau of a job described by growth(): the larger of
        its expected ramp (reaching its prior at the end of its budget, then flat) and its
        measured usage extrapolated at the measured slope; never above its cap."""
        cur, slope, now, target, start, budget, cap, end = g
        ramp = 0.5 + target * min(1.0, max(0.0, tau - start) / budget)
        return min(cap, max(ramp, cur + slope * (tau - now)))

    def growth(self, job, info, now):
        """Parameters of a running job's projected usage (see usage())."""
        s = info["samples"]
        cur = s[-1][1] if s else 0.0
        slope = 0.0
        if len(s) >= 2 and s[-1][0] - s[0][0] >= 60:
            slope = max(0.0, (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]))
        # after half the budget (at most 10 minutes), trust the measurement more than the prior:
        # expect at most 1.5x the current usage extrapolated linearly to the end of the budget
        target = info["prior"]
        elapsed = now - info["start"]
        if s and elapsed >= min(600, job["budget_s"] / 2):
            target = min(target, max(cur, 1.5 * cur * job["budget_s"] / elapsed))
        return (cur, slope, now, target, info["start"], job["budget_s"], info["cap"],
                self.end_time(job, info["start"], now))

    def profile(self, now, node=None):
        """Projected total usage of the running jobs (optionally on one NUMA node) at each end time."""
        jobs = [self.growth(self.by_id[jid], info, now) for jid, info in self.running.items()
                if node is None or self.numa(info["core"]) == node]
        points = sorted({now} | {g[7] - 1 for g in jobs if g[7] - 1 > now})
        m = self.mem["margin"]
        totals = [sum(self.usage(g, tau) * m for g in jobs if g[7] > tau) for tau in points]
        return jobs, points, totals

    def fits(self, job, prior, cap, now, jobs, points, totals, budget):
        end = self.end_time(job, now, now)
        new = (0.0, 0.0, now, prior, now, job["budget_s"], cap, end)
        m = self.mem["margin"]
        for tau, tot in zip(points, totals):
            # hold back part of the far-future budget for jobs admitted later, so that start
            # times spread out instead of forming lockstep waves (tightens, never loosens)
            u = min(1.0, (tau - now) / self.horizon)
            if tau < end and tot + self.usage(new, tau) * m > budget * (1 - 0.5 * u * u):
                return False
        tau = end - 1  # the candidate's own end, where it is largest
        tot = sum(self.usage(g, tau) * m for g in jobs if g[7] > tau)
        return tot + self.usage(new, tau) * m <= budget

    def pick(self, free_cores):
        """Next queued job that fits; returns (index, core, prior, cap) or None."""
        now = time.time()
        glob = self.profile(now)
        per = {n: self.profile(now, n) for n in (0, 1)}
        node_order = sorted((0, 1), key=lambda n: max(per[n][2], default=0.0))
        head_starving = self.head_wait_since is not None and now - self.head_wait_since > 1800
        for i, job in enumerate(self.queue[:400]):
            if job["job_id"] in self.running:
                continue
            prior = self.priors.get(job)
            cap = self.cap_for(job, prior)
            chosen = None
            if self.fits(job, prior, cap, now, *glob, self.mem["budget_gib"]):
                for n in node_order:
                    cands = [c for c in free_cores if self.numa(c) == n]
                    if cands and self.fits(job, prior, cap, now, *per[n], self.mem["node_budget_gib"]):
                        chosen = cands[0]
                        break
            if chosen is not None:
                if i == 0:
                    self.head_wait_since = None
                return i, chosen, prior, cap
            if i == 0 and self.head_wait_since is None:
                self.head_wait_since = now
            if head_starving:
                return None  # stop skipping ahead until the head job gets in
        return None

    # ---------- main loop ----------
    def run(self, poll_s=2.0):
        self.preflight()
        self.recover()
        self.log(f"plan {len(self.plan)} jobs, queued {len(self.queue)}, running {len(self.running)}, "
                 f"cap tiers {self.tiers}, {len(self.cores)} job cores")
        last_sample = last_mode = 0.0
        mode = cpu_mode(self.cores)
        while self.queue or self.running:
            self.reap()
            if time.time() - last_sample >= 30:
                self.sample_memory()
                last_sample = time.time()
            if time.time() - last_mode >= 60:
                mode = cpu_mode(self.cores)
                ok = self.allow_cpu_mode or mode == WANT_CPU_MODE
                if ok != self.cpu_mode_ok:
                    self.event("cpu_mode", info=mode)
                    self.log(f"CPU mode {'restored' if ok else 'CHANGED'}: {mode}")
                self.cpu_mode_ok = ok
                last_mode = time.time()
            self.enforce_wall_limits()
            avail = self.watchdog()
            draining = self.flag("DRAIN") or self.flag("STOP") or not self.cpu_mode_ok
            for _ in range(16):  # bounded burst so reaping and the watchdog still run often
                if draining or avail < self.mem["pause_below_gib"] or not self.queue \
                        or len(self.running) >= self.max_running:
                    break
                busy = {i["core"] for i in self.running.values()} | self.slice_busy_cores()
                free = [c for c in self.cores if c not in busy]
                choice = self.pick(free) if free else None
                if not choice:
                    break
                i, core, prior, cap = choice
                job = self.queue[i]
                gap = self.cfg["stagger_s"]["short" if is_quick(job) else "long"]
                wait = gap - (time.time() - self.last_launch)
                if wait > 0:
                    if wait > poll_s:
                        break
                    time.sleep(wait)
                self.queue.pop(i)
                if not self.launch(job, core, prior, cap, mode):
                    self.queue.append(job)
                    break
            self.write_status(avail, draining)
            if self.flag("STOP") and not self.running:
                break
            time.sleep(poll_s)
        self.log("scheduler finished")

    def write_status(self, avail, draining):
        now = time.time()
        _, _, totals = self.profile(now)
        st = {"utc": now, "running": len(self.running), "queued": len(self.queue),
              "mem_available_gib": round(avail, 1), "draining": draining, "cpu_mode_ok": self.cpu_mode_ok,
              "projected_peak_gib": round(max(totals, default=0.0), 1),
              "current_gib": round(sum(i["samples"][-1][1] for i in self.running.values() if i["samples"]), 1)}
        write_json(os.path.join(self.runs, "status.json"), st)


def order_plan(jobs, cfg, priors):
    """Launch order, quickest results first; seeded shuffle inside each class.

    class 0: short jobs (60 s budgets) and VOQC (seconds per circuit)
    class 1: the 26-circuit experiments (E2-E5; low memory, complete figures early)
    class 2: 1 h jobs on the 248-circuit suite
    priority-3 (optional) jobs go last.
    """
    rng = random.Random(cfg["order_seed"])

    def cls(j):
        if j["priority"] > 1:
            return 3
        if is_quick(j):
            return 0
        return 1 if j["suite"] in ("S26", "S26PP") else 2

    out = []
    for c in range(4):
        part = sorted((j for j in jobs if cls(j) == c), key=lambda j: j["job_id"])
        rng.shuffle(part)
        out.extend(part)
    return out
