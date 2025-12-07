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

import equiv_verification

def graph_results():

    if not os.path.isdir("pickled_results"):
        raise Exception("Pickled results not found")


    circuit_list = [("circuit/nam_circs/adder_8.qasm", "adder_8"),
                    ("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                    ("circuit/nam_circs/barenco_tof_4.qasm", "barenco_tof_4"),
                    # ("circuit/nam_circs/barenco_tof_5.qasm", "barenco_tof_5"),
                    # ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
                    # ("circuit/nam_circs/csla_mux_3.qasm", "csla_mux_3"),
                    # ("circuit/nam_circs/csum_mux_9.qasm", "csum_mux_9"),
                    # ("circuit/nam_circs/gf2^4_mult.qasm", "gf2^4_mult"),
                    # ("circuit/nam_circs/gf2^5_mult.qasm", "gf2^5_mult"),
                    # ("circuit/nam_circs/gf2^6_mult.qasm", "gf2^6_mult"),
                    # ("circuit/nam_circs/gf2^7_mult.qasm", "gf2^7_mult"),
                    # ("circuit/nam_circs/gf2^8_mult.qasm", "gf2^8_mult"),
                    # ("circuit/nam_circs/gf2^9_mult.qasm", "gf2^9_mult"),
                    # ("circuit/nam_circs/gf2^10_mult.qasm", "gf2^10_mult"),
                    # ("circuit/nam_circs/mod5_4.qasm", "mod5_4"),
                    # ("circuit/nam_circs/mod_mult_55.qasm", "mod_mult_55"),
                    # ("circuit/nam_circs/mod_red_21.qasm", "mod_red_21"),
                    # ("circuit/nam_circs/qcla_adder_10.qasm", "qcla_adder_10"),
                    # ("circuit/nam_circs/qcla_com_7.qasm", "qcla_com_7"),
                    # ("circuit/nam_circs/qcla_mod_7.qasm", "qcla_mod_7"),
                    # ("circuit/nam_circs/rc_adder_6.qasm", "rc_adder_6"),
                    # ("circuit/nam_circs/tof_3.qasm", "tof_3"),
                    # ("circuit/nam_circs/tof_4.qasm", "tof_4"),
                    # ("circuit/nam_circs/tof_5.qasm", "tof_5"),
                    # ("circuit/nam_circs/tof_10.qasm", "tof_10"),
                    # ("circuit/nam_circs/vbe_adder_3.qasm", "vbe_adder_3"),
    ]

    with open("pickled_results/raw_results.pkl", 'rb') as f:
        raw_results = pickle.load(f)

    with open("pickled_results/exp_params.pkl", 'rb') as f:
        experiments = pickle.load(f)

    with open("pickled_results/graph_labels.pkl", 'rb') as f:
        graph_labels = pickle.load(f)

    bench_labels = ["QUESO", "GUOQ", "VOQC"]

    # Circuit Name mapped to list of (list of times, list of lengths at these times
    bench_dict = {}
    bench_timeouts = ["900", "1800", "2700", "3600"]

    for circuit in circuit_list:
        original_qc = qiskit.QuantumCircuit.from_qasm_file(("../qalm/" + circuit[0]).replace('nam-benchmarks', 'nam_circs'))
        original_gate_count = original_qc.size()

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
            new_time, new_gates = zip(*sorted(zip(bench_results[i][0], bench_results[i][1])))
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



if __name__ == '__main__':
    graph_results()
