import json
import csv
from qiskit import QuantumCircuit

def process_results():
    circuit_list = []

    total_results = {}

    total_results["qalm_1_results"] = []
    total_results["qalm_5_results"] = []
    total_results["qalm_50_results"] = []
    total_results["qalm_100_results"] = []
    total_results["qalm_0_results"] = []
    total_results["voqc_results"] = []
    total_results["qiskit_results"] = []
    total_results["guoq_results"] = []
    total_results["queso_results"] = []
    total_results["repeated_roqc_results"] = []

    with open("qalm_circuits_test.txt", "r") as f:
        circuit_list = [line.split("/")[-1].strip().split(".")[0] for line in f.readlines()]

    results_file = "qalm_bench_results.csv"

    original_lengths = []

    for circuit in circuit_list:
        original_lengths.append(str(QuantumCircuit.from_qasm_file(f"qalm/circuit/nam_circs/{circuit}.qasm").size()))

        for res in parse_qalm(circuit):
            total_results["qalm_" + res[0] + "_results"].append(res[1])

        r_queso = parse_queso(circuit)
        r_guoq = parse_guoq(circuit)
        r_qiskit = parse_qiskit(circuit)
        r_voqc = parse_voqc(circuit)
        r_repeated_roqc = parse_repeated_roqc(circuit)

        total_results["qiskit_results"].append(r_qiskit)
        total_results["guoq_results"].append(r_guoq)
        total_results["queso_results"].append(r_queso)
        total_results["voqc_results"].append(r_voqc)
        total_results["repeated_roqc_results"].append(r_repeated_roqc)

    opt_order = ["qalm_1", "qalm_5","qalm_50", "qalm_100", "qalm_0", "voqc", "qiskit", "guoq", "queso", "repeated_roqc"]

    fields = ["Optimizer Name"] + circuit_list
    all_data = []

    # Put original gate count in:
    original_lengths = ["initial"] + original_lengths

    for opt_method in opt_order:
        opt_data = [opt_method]
        for result in total_results[opt_method + "_results"]:
            opt_data.append(result)
        all_data.append(opt_data)

    with open(results_file, 'w') as r_out:
        csv_writer = csv.writer(r_out)
        csv_writer.writerow(fields)
        csv_writer.writerow(original_lengths)
        csv_writer.writerows(all_data)


# These results come from own experiments
def parse_qalm(circuit_name):
    intervals = [0, 1, 5, 50, 100]
    for interval in intervals:
        c = QuantumCircuit.from_qasm_file(f"fresh_results/qalm_bench/nam/qalm/{circuit_name}_interval_{interval}_timeout_60_result_circuit.qasm")
        yield (str(interval), str(c.size()))

# These results come from guoq artifact code
# TODO: Check that parameters are correct: none_none vs none_1 also optimization goal
def parse_guoq(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/guoq/results_{circuit_name}/results_none_1.json"
    with open(result_file, "r") as f:
        result_dict = json.load(f)
    return result_dict["best_circuit_size"]

def parse_queso(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/queso/results_{circuit_name}/results_none_none.json"
    with open(result_file, "r") as f:
        result_dict = json.load(f)
    return result_dict["best_circuit_size"]

def parse_voqc(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/voqc/results_{circuit_name}/results_none_none.json"
    with open(result_file, "r") as f:
        result_dict = json.load(f)
    return result_dict["optimized_total"]

def parse_qiskit(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/qiskit/{circuit_name}.qasm.qiskit"
    c = QuantumCircuit.from_qasm_file(result_file)
    return c.size()

def parse_repeated_roqc(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/repeated_roqc/{circuit_name}.qasm.roqc"
    c = QuantumCircuit.from_qasm_file(result_file)
    return c.size()

if __name__ == '__main__' :
    process_results()
