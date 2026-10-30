from pyvoqc.qiskit.voqc_pass import voqc_pass_manager
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from time import time_ns
import argparse
import os
import json
from common import GATE_SET_DICT

# Doesn't support sx or rxx

TRANSPILE_METHODS = {"nam": ["optimize_nam"], "ibm": ["optimize_nam", "optimize_ibm"]}


def run(args: dict):
    cluster = args["cluster"]
    process = args["process"]
    input_file = args["circuit_file"]
    gate_set = args["gate_set"]

    circuit_id = input_file[input_file.rfind("/") + 1 :].replace(".qasm", "")

    out_dir = f"results_{circuit_id}"
    os.makedirs(out_dir, exist_ok=True)

    result = {}
    circuit = QuantumCircuit.from_qasm_file(input_file)
    result["original_total"] = circuit.size()
    result["original_2q"] = circuit.num_nonlocal_gates()

    vpm = voqc_pass_manager(
        post_opts=TRANSPILE_METHODS[gate_set] if gate_set in TRANSPILE_METHODS else []
    )

    start = time_ns()

    compiled = vpm.run(circuit)

    end = time_ns()
    result["optimized_total"] = compiled.size()
    result["optimized_2q"] = compiled.num_nonlocal_gates()
    result["total_time"] = (end - start) / 1000000000

    output_file = f"{out_dir}/optimized_{cluster}_{process}_{circuit_id}.qasm"
    dump(compiled, output_file)

    result.update(args)
    result.update(
        {
            "circuit_id": circuit_id,
            "out_dir": out_dir,
            "method": "voqc",
        }
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VOQC", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--cluster", type=str, default="none")
    parser.add_argument("--process", type=str, default="none")
    parser.add_argument("--circuit_file", type=str, required=True)
    parser.add_argument("--gate_set", type=str, required=True)

    args = parser.parse_args()

    result = run(vars(args))

    with open(
        f"{result['out_dir']}/results_{args.cluster}_{args.process}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=4)
