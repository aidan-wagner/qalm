"""Container entry point for the author-provided baselines (GUOQ, QUESO, VOQC).

Runs inside amxu/guoq (pinned by digest) with /in (inputs, ro), /repro (this
directory, ro) and /out (job output) mounted. Mirrors the authors' own harness
(/home/tool_invoke_defs.sh + /home/run_guoq.py + /home/run_voqc.py, copies in
upstream/) with these deliberate differences:
  * the resynthesis server is resynth_1core.py (no busy-wait, one BQSKit worker),
    and it is started only for GUOQ TWO_Q (TOTAL mode never resynthesizes);
  * --seed is passed explicitly;
  * resource usage is recorded in /out/result.json.

usage: entry.py {guoq,queso,voqc} {TWO_Q,TOTAL,-} <circuit.qasm> <budget_s> <seed>
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

JAR = "/home/GUOQ-1.0-jar-with-dependencies.jar"
GUOQ_PY = "/opt/conda/envs/guoq/bin/python"
VOQC_PY = "/opt/conda/envs/voqc/bin/python"
# owner of the output files on the host (the scheduler passes its own uid and gid)
HOST_UID = int(os.environ.get("HOST_UID", 1000))
HOST_GID = int(os.environ.get("HOST_GID", 1000))


def tool_args(tool, mode, seed):
    if tool == "guoq":
        return f"-g NAM -opt {mode} --rules-dir /home/queso_rules/ --resynth-weight 180 --seed {seed}"
    if tool == "queso":
        return (f"-g NAM -opt {mode} --rules-dir /home/queso_rules/ -search BEAM -temp 0 "
                f"-q 8000 -resynth NONE --seed {seed}")
    raise ValueError(tool)


def write_args_file(path, args_str, circuit_file, job, out_dir):
    # Same layout as run_guoq.write_args_file
    with open(path, "w") as f:
        for token in args_str.split(" "):
            f.write(f"{token}\n")
        f.write(f"-job\n{job}\n")
        f.write(f"-out\n{out_dir}\n")
        f.write("--verbosity\n2\n")
        f.write(f"{circuit_file}\n")


def read_cgroup():
    out = {}
    for name in ("memory.peak", "memory.events", "cpu.stat", "memory.max", "cpuset.cpus.effective"):
        try:
            with open(f"/sys/fs/cgroup/{name}") as f:
                out[name] = f.read().strip()
        except OSError:
            out[name] = None
    return out


def run_and_wait(cmd, cwd, stdout, stderr, env=None):
    with open(stdout, "w") as out, open(stderr, "w") as err:
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, env=env)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.time() - t0
    return {
        "cmd": cmd,
        "exit_code": os.waitstatus_to_exitcode(status),
        "wall_s": wall,
        "ru_maxrss_kb": ru.ru_maxrss,
        "utime_s": ru.ru_utime,
        "stime_s": ru.ru_stime,
    }


def start_resynth_server(log_path):
    env = dict(os.environ, PATH=f"/opt/conda/envs/guoq/bin:{os.environ['PATH']}")
    log = open(log_path, "w")
    proc = subprocess.Popen([GUOQ_PY, "/repro/resynth_1core.py", "--bqskit"], cwd="/home",
                            stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        with open(log_path) as f:
            if "Resynthesis server is ready" in f.read():
                break
        if proc.poll() is not None:
            raise RuntimeError("resynthesis server exited before becoming ready")
        time.sleep(0.5)
    else:
        raise RuntimeError("resynthesis server not ready after 600 s")
    time.sleep(10)  # same pause as tool_invoke_defs.sh
    return proc


def stop_server(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    tool, mode, circuit, budget, seed = sys.argv[1:6]
    budget = int(budget)
    name = os.path.basename(circuit)[:-len(".qasm")]
    out_dir = "/out"
    with open(JAR, "rb") as f:
        jar_sha = hashlib.sha256(f.read()).hexdigest()
    result = {"tool": tool, "mode": mode, "circuit": circuit, "budget_s": budget,
              "seed": int(seed), "image_jar_sha256": jar_sha, "started_utc": time.time()}
    server = None
    try:
        if tool in ("guoq", "queso"):
            args_str = tool_args(tool, mode, seed)
            job = "none_1" if tool == "guoq" else "none_none"
            args_path = f"{out_dir}/args_{job}.txt"
            write_args_file(args_path, args_str, circuit, job, out_dir)
            if tool == "guoq" and mode == "TWO_Q":
                server = start_resynth_server(f"{out_dir}/resynth_log.txt")
            cmd = ["timeout", "-k", "60", str(budget), "java", "-ea", "-Xmx20g", "-cp", JAR,
                   "qoptimizer.Optimizer", f"@{args_path}"]
            result["run"] = run_and_wait(cmd, out_dir, f"{out_dir}/tool.out", f"{out_dir}/tool.err")
            result["output_qasm"] = f"latest_sol_{job}_{name}.qasm"
        elif tool == "voqc":
            env = dict(os.environ, PYTHONPATH="/home")
            cmd = ["timeout", "-k", "60", str(budget), VOQC_PY, "/home/run_voqc.py",
                   "--circuit_file", circuit, "--gate_set", "nam"]
            result["run"] = run_and_wait(cmd, out_dir, f"{out_dir}/tool.out", f"{out_dir}/tool.err", env)
            result["output_qasm"] = f"results_{name}/optimized_none_none_{name}.qasm"
        else:
            raise ValueError(tool)
    except Exception as e:  # recorded; the host-side collector decides the status
        result["entry_error"] = repr(e)
    finally:
        if server is not None:
            stop_server(server)
        result["finished_utc"] = time.time()
        result["cgroup"] = read_cgroup()
        with open(f"{out_dir}/result.json.tmp", "w") as f:
            json.dump(result, f, indent=1)
        os.replace(f"{out_dir}/result.json.tmp", f"{out_dir}/result.json")
        for root, dirs, files in os.walk(out_dir):
            for p in dirs + files:
                os.chown(os.path.join(root, p), HOST_UID, HOST_GID)
        os.chown(out_dir, HOST_UID, HOST_GID)


if __name__ == "__main__":
    main()
