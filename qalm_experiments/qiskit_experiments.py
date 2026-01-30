import argparse
import subprocess
import os

def run_experiments(circuit_list_file):
    circuit_files = []
    with open(circuit_list_file, 'r') as f:
        for circuit_name in f:
            circuit_files.append(circuit_name)

    print(circuit_files)

    for circuit_file in circuit_files:
        res = subprocess.run(["python", "baselines/qiskit_compiler/run_qiskit.py","-f", circuit_file.strip(), "-o", circuit_file.strip() + ".qiskit"], capture_output = True, text = True)

    if not os.path.exists("../fresh_results/qalm/nam/qiskit"):
        os.makedirs("../fresh_results/qalm/nam/qiskit")

    for circuit_file in circuit_files:
        os.rename(circuit_file.strip() + ".qiskit", "../fresh_results/qalm_bench/nam/qiskit/" + circuit_file.strip().split("/")[-1] + ".qiskit")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()

    run_experiments(args.filename)

