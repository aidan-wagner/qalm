import subprocess
import os
import argparse
import json
import random

# TODO: make able to call from top level dir


def write_args_file(args_file, args_str, circuit_file, cluster, process, out_dir):
    args_split = args_str.split(" ")
    with open(args_file, "w") as f:
        for token in args_split:
            f.write(f"{token}\n")
        f.write(f"-job\n{cluster}_{process}\n")
        f.write(f"-out\n{out_dir}\n")
        f.write(f"--verbosity\n{2}\n")
        f.write(f"{circuit_file}\n")


def run(args: dict):
    cluster = args["cluster"]
    process = args["process"]
    jar = args["jar"]
    circuit_file = args["circuit_file"]
    timeout = args["timeout"]
    with open(args["guoq_cli_args"], "r") as f:
        args["guoq_cli_args"] = f.read().replace("%", ",").replace("!", " ")

    circuit_id = circuit_file[circuit_file.rfind("/") + 1 :].replace(".qasm", "")

    out_dir = f"results_{circuit_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{out_dir}/guoq_log_{cluster}_{process}.out"
    err_name = out_name.replace(".out", ".err")

    args_file = f"args_{cluster}_{process}.txt"
    args_file_path = f"{out_dir}/{args_file}"
    write_args_file(
        args_file_path, args["guoq_cli_args"], circuit_file, cluster, process, out_dir
    )

    with open(out_name, "w") as out:
        with open(err_name, "w") as err:
            command = f"timeout {timeout} java -ea -Xmx20g -cp {jar} qoptimizer.Optimizer @{args_file_path}"
            command_list = command.split(" ")
            proc = subprocess.Popen(
                command_list,
                stdout=out,
                stderr=err,
            )
            proc.wait()

    results = {}
    original_counts = {}
    config = None
    with open(out_name, "r") as f:
        log = f.readlines()
        for line in log:
            if "job_info" in line:
                config = json.loads(line)
                break
        for line in log:
            if "original_total" in line:
                original_counts = json.loads(line)
                break
        log.reverse()
        for line in log:
            if "rules_applied_to_best" in line:
                results = json.loads(line)
                results.pop("rules_applied_to_best", None)
                results.pop("all_rules_applied", None)
                break

    errors = None
    with open(err_name, "r") as f:
        contents = f.read()
        if "Exception" in contents or "Error" in contents:
            errors = "error in ours"
    resynth_errors = None
    with open(f"{out_dir}/resynth_log_{cluster}_{process}.txt") as f:
        contents = f.read()
        if "Exception" in contents or "Error" in contents:
            resynth_errors = "error in resynth"

    args.update(results)
    args.update(original_counts)
    args.update(
        {
            "circuit_id": circuit_id,
            "out_dir": out_dir,
            "command": command,
            "args_file": args_file,
            "guoq_config": config,
            "error": errors,
            "resynth_errors": resynth_errors,
        }
    )

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GUOQ", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--cluster", type=str, default="none")
    parser.add_argument("--process", type=str, default="none")
    parser.add_argument("--jar", type=str, default="GUOQ-1.0-jar-with-dependencies.jar")
    parser.add_argument("--timeout", type=int, default=60 * 60)
    parser.add_argument("--circuit_file", type=str, required=True)
    parser.add_argument("--guoq_cli_args", type=str)

    args = parser.parse_args()

    result = run(vars(args))

    with open(
        f"{result['out_dir']}/results_{args.cluster}_{args.process}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=4)