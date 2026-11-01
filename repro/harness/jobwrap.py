"""Run one native job inside its own systemd transient unit.

Launched by the scheduler as
  sudo systemd-run --unit=qalmrepro-<id> --slice=qalmrepro.slice --uid=<uid> ... \
      /usr/bin/python3 -I repro/harness/jobwrap.py <jobdir>
It starts the optimizer binary (stdout/stderr to files), enforces the wall
limit, and finally records rusage plus its own cgroup's memory.peak,
memory.events and cpu.stat into <jobdir>/result.json. Standard library only.
The runtime library hashed into result.json is job.json's "lib" (the campaign's
`lib`), or else the libquartz_runtime.so next to the binary (where its RUNPATH
points; job.json files written before "lib" existed).
"""
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def own_cgroup_dir():
    with open("/proc/self/cgroup") as f:
        for line in f:
            if line.startswith("0::"):
                return "/sys/fs/cgroup" + line.strip()[3:]
    return None


def read_cgroup(cg):
    out = {}
    for name in ("memory.peak", "memory.events", "cpu.stat", "memory.max", "cpuset.cpus.effective"):
        try:
            with open(os.path.join(cg, name)) as f:
                out[name] = f.read().strip()
        except (OSError, TypeError):
            out[name] = None
    return out


def main():
    jobdir = sys.argv[1]
    with open(os.path.join(jobdir, "job.json")) as f:
        job = json.load(f)
    env = dict(os.environ, OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1", LANG="C.UTF-8")
    lib = job.get("lib") or os.path.join(os.path.dirname(job["argv"][0]), "libquartz_runtime.so")
    result = {"job_id": job["job_id"], "host": socket.gethostname(), "argv": job["argv"],
              "wall_limit_s": job["wall_limit_s"], "started_utc": time.time(),
              "binary_sha256": sha256(os.path.join(job["repo_root"], job["argv"][0])),
              "lib": lib, "lib_sha256": sha256(os.path.join(job["repo_root"], lib))}
    wall_kill = False
    with open(os.path.join(jobdir, "stdout.log"), "wb") as out, \
            open(os.path.join(jobdir, "stderr.log"), "wb") as err:
        t0 = time.monotonic()
        proc = subprocess.Popen(job["argv"], cwd=job["repo_root"], stdout=out, stderr=err, env=env)
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - t0 > job["wall_limit_s"] and not wall_kill:
                wall_kill = True
                os.kill(proc.pid, signal.SIGKILL)
            time.sleep(1.0)
        wall = time.monotonic() - t0
    result.update({
        "exit_code": os.waitstatus_to_exitcode(status),
        "wall_kill": wall_kill,
        "wall_s": wall,
        "ru_maxrss_kb": ru.ru_maxrss,
        "utime_s": ru.ru_utime,
        "stime_s": ru.ru_stime,
        "finished_utc": time.time(),
        "cgroup": read_cgroup(own_cgroup_dir()),
    })
    tmp = os.path.join(jobdir, "result.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, os.path.join(jobdir, "result.json"))


if __name__ == "__main__":
    main()
