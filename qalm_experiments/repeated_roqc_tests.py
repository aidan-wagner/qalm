import argparse
import subprocess
import os

def run_experiments(circuit_list_file):
    circuit_files = []
    with open(circuit_list_file, 'r') as f:
        for circuit_name in f:
            circuit_files.append(circuit_name)

    static_circuit_files = circuit_files.copy()

    results = {}

    for circuit_file in circuit_files:
        res = subprocess.run(["./roqc/target/release/roqc", circuit_file.strip()], capture_output = True, text = True)
        final_length = 0
        for line in res.stdout.splitlines():
            if "Optimized length: " in line:
                final_length = int(line.strip().split()[-1])
        results[circuit_file] = final_length
    
    # Repeat
    for circuit_file in circuit_files:
        res = subprocess.run(["./roqc/target/release/roqc", circuit_file.strip() + ".roqc"], capture_output = True, text = True)
        final_length = 0
        for line in res.stdout.splitlines():
            if "Optimized length: " in line:
                final_length = int(line.strip().split()[-1])
                if results[circuit_file] == final_length:
                    circuit_files.remove(circuit_file)
                else:
                    print(f"Found additional improvement for {circuit_file}:", results[circuit_file], "->", final_length)
                    results[circuit_file] = final_length

    if not os.path.exists("../fresh_results/qalm/nam/repeated_roqc"):
        os.makedirs("../fresh_results/qalm/nam/repeated_roqc")

    for circuit_file in static_circuit_files:
        os.rename(circuit_file.strip() + ".roqc", "../fresh_results/qalm_bench/nam/repeated_roqc/" + circuit_file.strip().split("/")[-1] + ".roqc")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()

    print(args.filename)

    run_experiments(args.filename)

