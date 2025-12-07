import json
import csv
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import os

def process_results():
    circuit_list = []

    total_results = {}

    total_2q = {}

    # total_results["voqc_results"] = []
    # total_results["qiskit_results"] = []
    total_results["guoq_results"] = []
    # total_results["queso_results"] = []
    total_2q["guoq_results"] = []
    # total_2q["queso_results"] = []
    # total_results["repeated_roqc_results"] = []

    with open("../qalm/qalm_experiments/qalm_circuits_full.txt", "r") as f:
        circuit_list = [line.split("/")[-1].strip().split(".")[0] for line in f.readlines()]

    results_file = "10_min_qalm_bench_results.csv"
    results_file_2q = "10_min_qalm_bench_results_2q.csv"

    original_lengths = []
    original_2q = []

    for circuit in circuit_list:
        print(circuit)
        original_lengths.append(str(QuantumCircuit.from_qasm_file(f"../qalm/circuit/nam_circs/{circuit}.qasm").size()))
        original_2q.append(str(QuantumCircuit.from_qasm_file(f"../qalm/circuit/nam_circs/{circuit}.qasm").count_ops()["cx"]))


        for res in parse_qalm(circuit):
            if ((res[0] + "_results") not in total_results.keys()):
                total_results[res[0] + "_results"] = []
                total_2q[res[0] + "_results"] = []
            total_results[res[0] + "_results"].append(res[1])
            total_2q[res[0] + "_results"].append(res[2])

        # r_queso = parse_queso(circuit)
        r_guoq = parse_guoq(circuit)
        # r_qiskit = parse_qiskit(circuit)
        # r_voqc = parse_voqc(circuit)
        # r_repeated_roqc = parse_repeated_roqc(circuit)

        # total_results["qiskit_results"].append(r_qiskit)
        total_results["guoq_results"].append(r_guoq[0])
        # total_results["queso_results"].append(r_queso[0])
        total_2q["guoq_results"].append(r_guoq[1])
        # total_2q["queso_results"].append(r_queso[1])
        # total_results["voqc_results"].append(r_voqc)
        # total_results["repeated_roqc_results"].append(r_repeated_roqc)

    opt_order = total_results.keys()

    fields = ["Optimizer Name"] + circuit_list
    all_data = []

    # Put original gate count in:
    original_lengths = ["initial"] + original_lengths
    original_2q = ["initial"] + original_2q
    print(original_lengths)
    print(original_2q)

    for opt_method in opt_order:
        opt_data = [opt_method]
        for result in total_results[opt_method]:
            opt_data.append(result)
        all_data.append(opt_data)

    with open(results_file, 'w') as r_out:
        csv_writer = csv.writer(r_out)
        csv_writer.writerow(fields)
        csv_writer.writerow(original_lengths)
        csv_writer.writerows(all_data)

    all_data = []

    for opt_method in opt_order:
        opt_data = [opt_method]
        for result in total_2q[opt_method]:
            opt_data.append(result)
        all_data.append(opt_data)

    with open(results_file_2q, 'w') as r_out:
        csv_writer = csv.writer(r_out)
        csv_writer.writerow(fields)
        csv_writer.writerow(original_2q)
        csv_writer.writerows(all_data)

    # graph_guoq_queso_comparison(circuit_list, total_results, original_lengths[1:])
    graph_guoq_queso_comparison_2q(circuit_list, total_2q, original_lengths[1:])

def graph_best_comparisons(circuit_list, total_results, initial_lengths):
    # (circuit_name best_qalm, best_other)
    circuit_bests = []
    qalm_args_list = []
    benchmark_list = []

    for a in total_results.keys():
        if "600" in a:
            qalm_args_list.append(a)
        elif "initial" != a:
            benchmark_list.append(a)

    for circuit_index in range(len(circuit_list)):

        # Find best qalm result
        best_qalm_args = qalm_args_list[0]
        for args in qalm_args_list:
            if total_results[args][circuit_index] < total_results[best_qalm_args][circuit_index]:
                best_qalm_args = args

        # Find best benchmark result
        best_bench = benchmark_list[0]
        for bench in benchmark_list:
            if total_results[bench][circuit_index] < total_results[best_bench][circuit_index]:
                best_bench = bench

        circuit_bests.append((circuit_list[circuit_index], float(total_results[best_bench][circuit_index]) /  float(total_results[best_qalm_args][circuit_index])))

    circuit_bests.sort(key= lambda result: result[1])

    # plt.plot(circuit_bests)
    plt.plot(list(zip(*circuit_bests))[0], list(zip(*circuit_bests))[1], label="QALM gate count divided by best other method gate count")
    plt.plot(list(zip(*circuit_bests))[0], [1] * len(circuit_bests), label="1.00")
    plt.xticks(rotation=90)
    plt.title("Difference in circuit length reduction")
    plt.tight_layout()
    plt.legend()
    plt.show()


def graph_guoq_queso_comparison(circuit_list, total_results, initial_lengths):
    experiment_wins = {}
    for experiment in total_results.keys():
        if "600" not in experiment:
            continue

        experiment_wins[experiment] = 0

        for circuit_index in range(len(circuit_list)):
            if total_results[experiment][circuit_index] < total_results["queso_results"][circuit_index]:
                experiment_wins[experiment] = experiment_wins[experiment] + 1
            if total_results[experiment][circuit_index] < total_results["guoq_results"][circuit_index]:
                experiment_wins[experiment] = experiment_wins[experiment] + 1

    best_experiment = ""
    best_wins = -1

    for experiment in total_results.keys():
        if "600" not in experiment:
            continue

        if experiment_wins[experiment] > best_wins:
            best_experiment = experiment
            best_wins = experiment_wins[experiment]
    print("Best experiment is: ", best_experiment)

    queso_comp = [(circuit_list[i], float(total_results["queso_results"][i]) / float(total_results[best_experiment][i])) for i in range(len(circuit_list))]
    guoq_comp = [(circuit_list[i], float(total_results["guoq_results"][i]) / float(total_results[best_experiment][i])) for i in range(len(circuit_list))]

    queso_comp.sort(key= lambda result: result[1])
    guoq_comp.sort(key= lambda result: result[1])

    # plt.plot(circuit_bests)
    plt.plot(list(zip(*queso_comp))[0], list(zip(*queso_comp))[1], label="QALM gate count divided by QUESO gate count")
    plt.plot(list(zip(*queso_comp))[0], [1] * len(queso_comp), label="1.00")
    plt.xticks(rotation=90)
    plt.title("Difference in circuit length reduction")
    plt.tight_layout()
    plt.legend()
    plt.savefig("queso_comp.pdf", format="pdf")

    plt.clf()

    plt.plot(list(zip(*guoq_comp))[0], list(zip(*guoq_comp))[1], label="QALM gate count divided by GUOQ gate count")
    plt.plot(list(zip(*guoq_comp))[0], [1] * len(guoq_comp), label="1.00")
    plt.xticks(rotation=90)
    plt.title("Difference in circuit length reduction")
    plt.tight_layout()
    plt.legend()
    plt.savefig("guoq_comp.pdf", format="pdf")

    plt.clf()

def graph_guoq_queso_comparison_2q(circuit_list, total_results, initial_lengths):
    experiment_wins = {}
    for experiment in total_results.keys():
        if "600" not in experiment:
            continue

        experiment_wins[experiment] = 0

        for circuit_index in range(len(circuit_list)):
            if total_results[experiment][circuit_index] < total_results["guoq_results"][circuit_index]:
                experiment_wins[experiment] = experiment_wins[experiment] + 1
            if total_results[experiment][circuit_index] < total_results["guoq_results"][circuit_index]:
                experiment_wins[experiment] = experiment_wins[experiment] + 1

    best_experiment = ""
    best_wins = -1

    for experiment in total_results.keys():
        if "600" not in experiment:
            continue

        if experiment_wins[experiment] > best_wins:
            best_experiment = experiment
            best_wins = experiment_wins[experiment]
    print("Best experiment is: ", best_experiment)

    queso_comp = [(circuit_list[i], float(total_results["guoq_results"][i]) / float(total_results[best_experiment][i])) for i in range(len(circuit_list))]
    guoq_comp = [(circuit_list[i], float(total_results["guoq_results"][i]) / float(total_results[best_experiment][i])) for i in range(len(circuit_list))]

    queso_comp.sort(key= lambda result: result[1])
    guoq_comp.sort(key= lambda result: result[1])

    # plt.plot(circuit_bests)
    plt.plot(list(zip(*queso_comp))[0], list(zip(*queso_comp))[1], label="QALM 2q gate count divided by QUESO 2q gate count")
    plt.plot(list(zip(*queso_comp))[0], [1] * len(queso_comp), label="1.00")
    plt.xticks(rotation=90)
    plt.title("Difference in 2q gate reduction")
    plt.tight_layout()
    plt.legend()
    plt.savefig("queso_comp_2q.pdf", format="pdf")

    plt.clf()

    plt.plot(list(zip(*guoq_comp))[0], list(zip(*guoq_comp))[1], label="QALM 2q gate count divided by GUOQ 2q gate count")
    plt.plot(list(zip(*guoq_comp))[0], [1] * len(guoq_comp), label="1.00")
    plt.xticks(rotation=90)
    plt.title("Difference in 2q gate reduction")
    plt.tight_layout()
    plt.legend()
    plt.savefig("guoq_comp_2q.pdf", format="pdf")

    plt.clf()


# These results come from own experiments
def parse_qalm(circuit_name):
    prefixes = [file[:-12] for file in os.listdir("fresh_results/qalm_bench/nam/qalm/adder_8")]
    for prefix in prefixes:
        c = QuantumCircuit.from_qasm_file(f"fresh_results/qalm_bench/nam/qalm/{circuit_name}/{prefix}_result.qasm")
        yield (prefix, str(c.size()), str(c.count_ops()["cx"]))

# These results come from guoq artifact code
# TODO: Check that parameters are correct: none_none vs none_1 also optimization goal
def parse_guoq(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/guoq/results_{circuit_name}/results_none_1.json"
    with open(result_file, "r") as f:
        result_dict = json.load(f)
    return (result_dict["best_circuit_size"], result_dict["best_size_2q"])

def parse_queso(circuit_name):
    result_file = f"fresh_results/qalm_bench/nam/queso/results_{circuit_name}/results_none_none.json"
    with open(result_file, "r") as f:
        result_dict = json.load(f)
    return (result_dict["best_circuit_size"], result_dict["best_size_2q"])


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
