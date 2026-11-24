#!/usr/bin/env python3
"""Watch for CPU and memory use that could disturb the campaign's jobs; read-only, stdlib only.

Every --interval seconds it reads /proc once and logs (one line each, UTC):
  WARN onjob    a process outside qalmrepro.slice that may run on a job core (its allowed CPUs
                include job cores) and used CPU in the interval: direct contention with a job
  WARN sibling  an SMT sibling of a job core was busy (it should be idle; shares the job's core)
  WARN busy     a process outside qalmrepro.slice used more than --proc-pct of one CPU (on the
                housekeeping CPUs it cannot take a job's core, but it adds load to the machine)
  WARN mem      a process outside qalmrepro.slice holds more than --proc-mem-gib resident memory
                (logged once, and again each time it has doubled since it was last logged)
  WARN lowmem   MemAvailable fell below --low-mem-gib (the scheduler stops launching below 24
                GiB and preempts jobs below 10); INFO memok when it is back above
  HOUR          hourly summary: warnings, largest CPU values, lowest MemAvailable, and the
                largest anonymous memory of user.slice / system.slice / init.scope
Kernel threads are left out of the per-process checks (the sibling check still sees their load).
Under systemd with WatchdogSec= it reports "WATCHDOG=1" after every scan, so a hung watcher is
killed and (with Restart=on-failure) restarted; each start writes a new START line.

Usage (job cores and their siblings are read from qalmrepro.slice and /sys):
  python3 -I cpuwatch.py --log repro/runs/cpuwatch.log [--interval 30] [--proc-pct 20]
      [--sibling-pct 10] [--onjob-pct 1] [--proc-mem-gib 4] [--low-mem-gib 40]
      [--job-cores 2-63,66-127]
"""
import argparse
import os
import socket
import subprocess
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
GIB = 2**30
SLICE = "/qalmrepro.slice"
OUTSIDE = ("user.slice", "system.slice", "init.scope")  # top-level cgroups of everything else
PF_KTHREAD = 0x00200000


def parse_cpus(spec):
    out = set()
    for part in spec.replace(" ", ",").split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def ranges(cpus):
    cpus, out = sorted(cpus), []
    for c in cpus:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in out)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def cpu_times():
    """{cpu: (busy_ticks, total_ticks)} from /proc/stat."""
    out = {}
    for line in read("/proc/stat").splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:9]]
            total = sum(v)
            out[int(f[0][3:])] = (total - v[3] - v[4], total)  # minus idle, iowait
    return out


def mem_available_gib():
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def outside_anon_gib():
    """{cgroup: anonymous memory in GiB} of the top-level cgroups outside the campaign."""
    out = {}
    for cg in OUTSIDE:
        s = read(f"/sys/fs/cgroup/{cg}/memory.stat")
        if s:
            out[cg] = next((int(l.split()[1]) for l in s.splitlines() if l.startswith("anon ")), 0) / GIB
    return out


def processes():
    """{pid: (start, ticks, name, user_id, is_kthread, rss_bytes)} for every process."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        s = read(f"/proc/{d}/stat")
        if not s:
            continue
        name = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()  # f[0] is field 3 (state)
        flags, utime, stime, start = int(f[6]), int(f[11]), int(f[12]), int(f[19])
        rss = int(f[21]) * PAGE  # field 24
        try:
            uid = os.stat(f"/proc/{d}").st_uid
        except OSError:
            continue
        out[int(d)] = (start, utime + stime, name, uid, bool(flags & PF_KTHREAD), rss)
    return out


def cgroup(pid):
    """The process's cgroup path, or None if it can no longer be read (the process exited)."""
    c = read(f"/proc/{pid}/cgroup")
    if c is None:
        return None
    return next((l[3:] for l in c.splitlines() if l.startswith("0::")), c.strip())


def allowed(pid):
    for line in (read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return parse_cpus(line.split(":", 1)[1].strip())
    return set()


def user(uid):
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (KeyError, ImportError):
        return str(uid)


def sd_notify(msg):
    """Send a systemd notification (e.g. the watchdog keep-alive); a no-op outside systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
    except OSError:
        pass


def new_hour(now):
    return {"start": now, "warn": 0, "max_proc": (0.0, ""), "max_sib": (0.0, ""),
            "min_avail": float("inf"), "max_anon": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--proc-pct", type=float, default=20)
    ap.add_argument("--sibling-pct", type=float, default=10)
    ap.add_argument("--onjob-pct", type=float, default=1)
    ap.add_argument("--proc-mem-gib", type=float, default=4)
    ap.add_argument("--low-mem-gib", type=float, default=40)
    ap.add_argument("--job-cores")
    a = ap.parse_args()

    spec = a.job_cores or subprocess.run(
        ["systemctl", "show", "qalmrepro.slice", "-p", "AllowedCPUs", "--value"],
        capture_output=True, text=True).stdout.strip()
    jobs = parse_cpus(spec)
    if not jobs:
        sys.exit("no job cores (is qalmrepro.slice active? or pass --job-cores)")
    sib = set()
    for c in jobs:
        s = read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        sib |= parse_cpus(s.strip()) - {c} if s else set()
    sib -= jobs

    log = open(a.log, "a", buffering=1)

    def say(msg):
        log.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")

    say(f"START pid {os.getpid()}; job cores {ranges(jobs)}; idle siblings {ranges(sib) or 'none'}; "
        f"interval {a.interval}s; thresholds busy {a.proc_pct}% sibling {a.sibling_pct}% onjob {a.onjob_pct}% "
        f"proc-mem {a.proc_mem_gib} GiB low-mem {a.low_mem_gib} GiB; MemAvailable {mem_available_gib():.1f} GiB")
    prev_p, prev_c, prev_t = processes(), cpu_times(), time.monotonic()
    hour = new_hour(prev_t)
    seen_onjob = set()
    mem_logged = {}  # (pid, start) -> rss when last logged
    low = False
    sd_notify("WATCHDOG=1")
    while True:
        time.sleep(a.interval)
        now_p, now_c, now_t = processes(), cpu_times(), time.monotonic()
        dt = now_t - prev_t
        # SMT siblings of job cores
        for c in sorted(sib):
            if c in now_c and c in prev_c:
                busy = now_c[c][0] - prev_c[c][0]
                total = now_c[c][1] - prev_c[c][1]
                pct = 100.0 * busy / total if total else 0.0
                if pct > hour["max_sib"][0]:
                    hour["max_sib"] = (pct, f"cpu{c}")
                if pct > a.sibling_pct:
                    hour["warn"] += 1
                    say(f"WARN sibling cpu{c} busy {pct:.0f}%")
        # machine-wide free memory
        avail = mem_available_gib()
        hour["min_avail"] = min(hour["min_avail"], avail)
        if avail < a.low_mem_gib and not low:
            low = True
            hour["warn"] += 1
            say(f"WARN lowmem MemAvailable {avail:.1f} GiB")
        elif avail >= a.low_mem_gib and low:
            low = False
            say(f"INFO memok MemAvailable {avail:.1f} GiB")
        for cg, gib in outside_anon_gib().items():
            hour["max_anon"][cg] = max(hour["max_anon"].get(cg, 0.0), gib)
        # processes outside the campaign slice
        for pid, (start, ticks, name, uid, kthread, rss) in now_p.items():
            if kthread:
                continue
            old = prev_p.get(pid)
            # a process new in this interval (or a reused pid) counts all its CPU time
            used = ticks - old[1] if old and old[0] == start else ticks
            pct = 100.0 * used / CLK / dt
            big = rss > a.proc_mem_gib * GIB
            if pct < min(a.onjob_pct, a.proc_pct) and not big:
                continue
            cg = cgroup(pid)
            # exited since its stat was read (e.g. a job ending now): its cgroup is unknown
            if cg is None or cg.startswith(SLICE):
                continue
            who = f"pid {pid} {name!r} user {user(uid)} cgroup {cg}"
            if big:
                last = mem_logged.get((pid, start))
                if last is None or rss >= 2 * last:
                    mem_logged[(pid, start)] = rss
                    hour["warn"] += 1
                    say(f"WARN mem {who} resident {rss / GIB:.1f} GiB; MemAvailable {avail:.1f} GiB")
            if pct > hour["max_proc"][0]:
                hour["max_proc"] = (pct, who)
            on = allowed(pid) & jobs
            if on and pct >= a.onjob_pct and (pid, start) not in seen_onjob:
                seen_onjob.add((pid, start))  # once per process: its allowed CPUs do not change often
                hour["warn"] += 1
                say(f"WARN onjob {who} cpu {pct:.0f}% may run on job cores {ranges(on)}")
            elif pct > a.proc_pct:
                hour["warn"] += 1
                say(f"WARN busy {who} cpu {pct:.0f}%")
        alive = {(pid, v[0]) for pid, v in now_p.items()}
        mem_logged = {k: v for k, v in mem_logged.items() if k in alive}
        prev_p, prev_c, prev_t = now_p, now_c, now_t
        if now_t - hour["start"] >= 3600:
            anon = ", ".join(f"{cg} {g:.1f}" for cg, g in sorted(hour["max_anon"].items())) or "-"
            say(f"HOUR warnings {hour['warn']}; max outside-slice process {hour['max_proc'][0]:.0f}% "
                f"({hour['max_proc'][1] or '-'}); max sibling {hour['max_sib'][0]:.0f}% ({hour['max_sib'][1] or '-'}); "
                f"min MemAvailable {hour['min_avail']:.1f} GiB; max anon GiB outside slice: {anon}")
            hour = new_hour(now_t)
        sd_notify("WATCHDOG=1")  # one complete scan done


if __name__ == "__main__":
    main()
