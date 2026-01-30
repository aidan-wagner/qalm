import subprocess
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import numpy as np
from enum import Enum
import qiskit
import os
import json

class OptimizationType(Enum):
    roqc_interval = 1
    qalm = 2
    quartz = 3

import equiv_verification

def get_qalm_sampling(result, sample_times):
    times = result[0]
    counts = result[1]

    sampled_counts = []

    for t in sample_times:
        closest_time = 0
        for t_index in range(len(times)):
            if times[t_index] > t:
                assert(t_index != 0)
                sampled_counts.append(counts[t_index - 1])
                break

    while len(sampled_counts) != len(sample_times):
        sampled_counts.append(sampled_counts[-1])

    return sampled_counts

def get_qalm_avg(results_dict, circuit_list, exp_args, qalm_sample_times, initial_counts):
    all_samples = {}
    for circuit in circuit_list:
        all_samples[circuit[1]] = []
    for circuit in circuit_list:
        for arg_index in range(len(exp_args)):
            circuit_name = circuit[1]

            sample = get_qalm_sampling(results_dict[circuit_name][arg_index], qalm_sample_times)

            all_samples[circuit_name].append(sample)

    running_sums = [[0.0 for i in range(len(qalm_sample_times))] for j in range(len(exp_args))]
    for circuit in circuit_list:
        circuit_name = circuit[1]
        for s_index in range(len(running_sums)):
            running_sums[s_index] = [running_sums[s_index][i] + (all_samples[circuit_name][s_index][i]/initial_counts[circuit_name]) for i in range(len(qalm_sample_times))]
    
    for s_index in range(len(running_sums)):
        running_sums[s_index] = [v/float(len(circuit_list)) for v in running_sums[s_index]]

    print(len(running_sums[0]))

    return running_sums

def get_bench_avg(bench_dict, circuit_list, bench_sample_times, initial_counts):
    running_sums = [[0.0 for i in range(len(bench_sample_times))] for j in range(3)]
    for circuit in circuit_list:
        circuit_name = circuit[1]
        for s_index in range(len(running_sums)):
            running_sums[s_index] = [running_sums[s_index][i] + (bench_dict[circuit_name][s_index][1][i]/initial_counts[circuit_name]) for i in range(len(bench_sample_times))]
    
    for s_index in range(len(running_sums)):
        running_sums[s_index] = [v/float(len(circuit_list)) for v in running_sums[s_index]]

    return running_sums

def plot_avg_graph(bench_results, qalm_results, circuit_list, exp_args, graph_labels, bench_labels, initial_counts):
    qalm_sample_times = range(1*60, 61*60, 1*60)
    bench_timeouts = [0, 60, 180, 900, 1800, 2700, 3600]
    qalm_avgs = get_qalm_avg(qalm_results, circuit_list, exp_args, qalm_sample_times, initial_counts)
    bench_avgs = get_bench_avg(bench_results, circuit_list, bench_timeouts, initial_counts)

    fig = plt.figure()
    ax = fig.add_subplot()

    # for i in range(len(exp_args)):
    #     ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in qalm_avgs[i]], label = graph_labels[i])
    ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in qalm_avgs[2]], label = "QALM")
    ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in qalm_avgs[1]], label = "Quartz")
    ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in qalm_avgs[0]], label = "Quartz - with preprocessing")

    for i in range(3):
        # new_time, new_gates = zip(*sorted(zip(bench_results[i][0], bench_results[i][1])))
        new_time, new_gates = zip(*sorted(zip([float(t) for t in bench_timeouts], bench_avgs[i])))
        ax.plot(new_time, [1.0-x for x in new_gates], label = bench_labels[i])

    ax.set_xlabel("Time (s)", fontsize="large")
    ax.set_ylabel("Gate Count Reduction Percentage", fontsize="large")

    ax.set_title(f"Average Gate Count Reduction Across All Circuits", fontsize="large")

    ax.set_xticks(range(0, 350, 50))
    ax.set_xlim(0, 300)
    y_ticks = [float(x)/10.0 for x in range(0, 5, 1)]
    ax.set_yticks(y_ticks, labels=[f"{y * 100}%" for y in y_ticks])

    fig.legend(bbox_to_anchor=(.9,0.4), loc="upper right", fontsize="medium")

    fig.savefig(f"avg_reduction.pdf", format="pdf",  dpi=500)
    fig.clf()

    fig = plt.figure()
    ax = fig.add_subplot()

    I_ablation = [qalm_avgs[2], qalm_avgs[3], qalm_avgs[4]]
    # I_ablation_labels = [graph_labels[1], graph_labels[2], graph_labels[3]]
    I_ablation_labels = ["$N_{pool}=1$", "$N_{pool}=2$", "$N_{pool}=3$"]

    for i in range(len(I_ablation)):
        ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in I_ablation[i]], label = I_ablation_labels[i])

    ax.set_xlabel("Time (s)", fontsize="large")
    ax.set_ylabel("Gate Count Reduction Percentage", fontsize="large")

    ax.set_title("Impact of $N_{pool}$ on Average Gate Count Reduction", fontsize="large")

    ax.set_xticks(range(0, 4200, 600))
    y_ticks = [float(x)/100.0 for x in range(33, 38, 1)]
    ax.set_yticks(y_ticks, labels=[f"{y * 100}%" for y in y_ticks])
    ax.set_ylim(.33)

    fig.legend(bbox_to_anchor=(.9,0.4), loc="upper right", fontsize="large")

    fig.savefig(f"i_testing.pdf", format="pdf", dpi=500)
    fig.clf()

    fig = plt.figure()
    ax = fig.add_subplot()

    G_ablation = [qalm_avgs[2], qalm_avgs[5], qalm_avgs[6]]
    # G_ablation_labels = [graph_labels[1], graph_labels[4], graph_labels[5]]
    G_ablation_labels = ["$N_{branch}=1$", "$N_{branch}=2$", "$N_{branch}=3$"]

    for i in range(len(G_ablation)):
        ax.plot([0.0] + list(qalm_sample_times), [0] + [1.0-x for x in G_ablation[i]], label = G_ablation_labels[i])

    ax.set_xlabel("Time (s)", fontsize="large")
    ax.set_ylabel("Gate Count Reduction Percentage", fontsize="large")

    ax.set_title("Impact of $N_{branch}$ on Average Gate Count Reduction", fontsize="large")

    ax.set_xticks(range(0, 4200, 600))
    y_ticks = [float(x)/100.0 for x in range(33, 38, 1)]
    ax.set_yticks(y_ticks, labels=[f"{y * 100}%" for y in y_ticks])
    ax.set_ylim(.33)

    fig.legend(bbox_to_anchor=(.9,0.4), loc="upper right", fontsize="large")

    fig.savefig(f"g_testing.pdf", format="pdf", dpi=500)
    fig.clf()








def graph_results():

    if not os.path.isdir("pickled_results"):
        raise Exception("Pickled results not found")


    circuit_list = [("circuit/nam_circs/adder_8.qasm", "adder_8"),
                    ("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                    ("circuit/nam_circs/barenco_tof_4.qasm", "barenco_tof_4"),
                    ("circuit/nam_circs/barenco_tof_5.qasm", "barenco_tof_5"),
                    ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
                    ("circuit/nam_circs/csla_mux_3.qasm", "csla_mux_3"),
                    ("circuit/nam_circs/csum_mux_9.qasm", "csum_mux_9"),
                    ("circuit/nam_circs/gf2^4_mult.qasm", "gf2^4_mult"),
                    ("circuit/nam_circs/gf2^5_mult.qasm", "gf2^5_mult"),
                    ("circuit/nam_circs/gf2^6_mult.qasm", "gf2^6_mult"),
                    ("circuit/nam_circs/gf2^7_mult.qasm", "gf2^7_mult"),
                    ("circuit/nam_circs/gf2^8_mult.qasm", "gf2^8_mult"),
                    ("circuit/nam_circs/gf2^9_mult.qasm", "gf2^9_mult"),
                    ("circuit/nam_circs/gf2^10_mult.qasm", "gf2^10_mult"),
                    ("circuit/nam_circs/mod5_4.qasm", "mod5_4"),
                    ("circuit/nam_circs/mod_mult_55.qasm", "mod_mult_55"),
                    ("circuit/nam_circs/mod_red_21.qasm", "mod_red_21"),
                    ("circuit/nam_circs/qcla_adder_10.qasm", "qcla_adder_10"),
                    ("circuit/nam_circs/qcla_com_7.qasm", "qcla_com_7"),
                    ("circuit/nam_circs/qcla_mod_7.qasm", "qcla_mod_7"),
                    ("circuit/nam_circs/rc_adder_6.qasm", "rc_adder_6"),
                    ("circuit/nam_circs/tof_3.qasm", "tof_3"),
                    ("circuit/nam_circs/tof_4.qasm", "tof_4"),
                    ("circuit/nam_circs/tof_5.qasm", "tof_5"),
                    ("circuit/nam_circs/tof_10.qasm", "tof_10"),
                    ("circuit/nam_circs/vbe_adder_3.qasm", "vbe_adder_3"),
    ]

    with open("pickled_results/combined_raw_results.pkl", 'rb') as f:
        raw_results = pickle.load(f)

    with open("pickled_results/combined_exp_params.pkl", 'rb') as f:
        experiments = pickle.load(f)

    with open("pickled_results/combined_graph_labels.pkl", 'rb') as f:
        graph_labels = pickle.load(f)

    raw_results = raw_results[:len(experiments)*13] + raw_results[len(experiments)*15:]

    bench_labels = ["QUESO", "GUOQ", "VOQC"]

    # Circuit Name mapped to list of (list of times, list of lengths at these times
    bench_dict = {}
    bench_timeouts = ["60", "180", "900", "1800", "2700", "3600"]

    original_counts = {}

    for circuit in circuit_list:
        original_qc = qiskit.QuantumCircuit.from_qasm_file(("../qalm/" + circuit[0]).replace('nam-benchmarks', 'nam_circs'))
        original_gate_count = original_qc.size()

        original_counts[circuit[1]] = float(original_gate_count)

        bench_dict[circuit[1]] = []
        queso_times = [0.0]
        queso_gates = [float(original_gate_count)]
        for bench_timeout in bench_timeouts:
            with open(f"fresh_results_{bench_timeout}/qalm_bench/nam/queso/results_{circuit[1]}/results_none_none.json", "r") as f:
                result_dict = json.load(f)
                queso_times.append(float(result_dict["seconds_elapsed"]))
                queso_gates.append(float(result_dict["best_circuit_size"]))
        bench_dict[circuit[1]].append((queso_times, queso_gates))

        guoq_times = [0.0]
        guoq_gates = [float(original_gate_count)]
        for bench_timeout in bench_timeouts:
            with open(f"fresh_results_{bench_timeout}/qalm_bench/nam/guoq/results_{circuit[1]}/results_none_1.json", "r") as f:
                result_dict = json.load(f)
                guoq_times.append(float(result_dict["seconds_elapsed"]))
                guoq_gates.append(float(result_dict["best_circuit_size"]))
        bench_dict[circuit[1]].append((guoq_times, guoq_gates))

        voqc_times = [0.0]
        voqc_gates = [float(original_gate_count)]
        for bench_timeout in bench_timeouts:
            with open(f"fresh_results_{bench_timeout}/qalm_bench/nam/voqc/results_{circuit[1]}/results_none_none.json", "r") as f:
                result_dict = json.load(f)
                voqc_times.append(float(result_dict["total_time"]))
                voqc_gates.append(float(result_dict["optimized_total"]))
        bench_dict[circuit[1]].append((voqc_times, voqc_gates))


    result_dict = {}
    for circuit in circuit_list:
        result_dict[circuit[1]] = []

    print(len(experiments))
    print(len(circuit_list))
    print(len(graph_labels))

    for circuit_index in range(len(circuit_list)):
        for exp_index in range(len(experiments)):
            result_dict[circuit_list[circuit_index][1]].append(raw_results[exp_index + circuit_index*len(experiments)])

    for circuit in circuit_list:
        bench_results = bench_dict[circuit[1]]
        results = result_dict[circuit[1]]
        original_qc = qiskit.QuantumCircuit.from_qasm_file(("../qalm/" + circuit[0]).replace('nam-benchmarks', 'nam_circs'))
        original_gate_count = original_qc.size()

        fig = plt.figure()
        ax = fig.add_subplot()
        colormap = plt.get_cmap("gist_rainbow")
        ax.set_prop_cycle(color=[colormap(1.*i/(len(experiments)+3)) for i in range(len(experiments) + 3)])


        for i in range(len(results)):
            ax.plot(results[i][0], results[i][1], label = graph_labels[i], linestyle=['-', '--', ':'][i % 3])
        for i in range(3):
            # new_time, new_gates = zip(*sorted(zip(bench_results[i][0], bench_results[i][1])))
            new_time, new_gates = zip(*sorted(zip([0] + [float(t) for t in bench_timeouts], bench_results[i][1])))
            ax.plot(new_time, new_gates, label = bench_labels[i], linestyle=['-', '--', ':'][i % 3])


        # ax.plot(voqc_result[0], voqc_result[1], label = "Voqc")


        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gate Count")

        ax.set_title(f"Optimization Experiments - {circuit[1]} - ECC set (5,3)")


        fig.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize="x-small")
        fig.tight_layout()
        #plt.show()
        fig.savefig(f"long_result_{circuit[1]}_seconds.png", dpi=500)
        fig.clf()

    plot_avg_graph(bench_dict, result_dict, circuit_list, experiments, graph_labels, bench_labels, original_counts)



if __name__ == '__main__':
    graph_results()
