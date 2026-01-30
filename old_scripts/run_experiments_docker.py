import subprocess
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import numpy as np
import os
from qiskit import QuantumCircuit

import equiv_verification

def tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    roqc_interval = arguments[3]
    return (roqc_interval, run_quartz(filename, circuit_name, timeout, roqc_interval))

def run_experiments():

    if not os.path.exists("../fresh_results/qualm_bench/nam/qualm"):
        os.makedirs("../fresh_results/qualm_bench/nam/qualm")
    if not os.path.exists("final_circuits"):
        os.makedirs("final_circuits")
    if not os.path.exists("result_figures"):
        os.makedirs("result_figures")
    if not os.path.exists("saved_timelines"):
        os.makedirs("saved_timelines")


    timeout = 60
    validate = False
    roqc_intervals = [0, 1, 5, 50, 100]
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
                    ];
    circuit_list = [("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                    ("circuit/nam_circs/adder_8.qasm", "adder_8")]

    full_results = []

    for circuit in circuit_list:
        print(f"Running experiments for {circuit[1]}")

        qiskit_circuit = QuantumCircuit.from_qasm_file(circuit[0])

        original_gate_count = qiskit_circuit.size()

        results = {}
        arguments = [(circuit[0], circuit[1], timeout, roqc_interval) for roqc_interval in roqc_intervals]

        with multiprocessing.Pool(5) as pool:
            initial_results = pool.map(tester, arguments)


        for result in initial_results:
            results[result[0]] = result[1]

        if validate:
            filenames = [f"final_circuits/{circuit[1]}_interval_{roqc_interval}_timeout_{timeout}_result.qasm" for roqc_interval in roqc_intervals]

            for filename in filenames:
                assert(equiv_verification.monte_carlo_compare_by_filename(circuit[0], filename))
                print(f"====Test Passed====")

        with open(f"saved_timelines/{circuit[1]}_timeout_{timeout}_timeline.txt", 'w') as f:
            f.truncate(0)
            for interval in roqc_intervals:
                f.write(f"interval: {interval}, results: {results[interval]}\n")


        plt.plot(results[0][0], results[0][1], label = "Quartz - No ROQC")
        plt.plot(results[1][0], results[1][1], label = "Quartz - ROQC Interval = 1")
        plt.plot(results[5][0], results[5][1], label = "Quartz - ROQC Interval = 5")
        plt.plot(results[50][0], results[50][1], label = "Quartz - ROQC Interval = 50")
        plt.plot(results[100][0], results[100][1], label = "Quartz - ROQC Interval = 100")

        plt.xlabel("Time (s)")
        plt.ylabel("Gate Count")

        plt.title(f"Optimization with different ROQC intervals - {circuit[1]} - ECC set (5,3)")

        plt.legend()
        #plt.show()
        plt.savefig(f"result_figures/progress_{circuit[1]}_timeout_{timeout}_seconds.png")
        plt.clf()


        final_gate_counts = [original_gate_count,
                             results[0][1][-1],
                             results[1][1][-1],
                             results[5][1][-1],
                             results[50][1][-1],
                             results[100][1][-1]]

        full_results.append(final_gate_counts)

        with open(f"../fresh_results/qualm_bench/nam/qualm/results_{circuit[1]}.txt", 'w') as f:
            f.write(f"qualm_1 {original_gate_count}/{final_gate_counts[2]}\n")
            f.write(f"qualm_5 {original_gate_count}/{final_gate_counts[3]}\n")
            f.write(f"qualm_50 {original_gate_count}/{final_gate_counts[4]}\n")
            f.write(f"qualm_100 {original_gate_count}/{final_gate_counts[5]}\n")
            f.write(f"quartz {original_gate_count}/{final_gate_counts[0]}\n")


    bar_width = 0.15
    x_axis = np.arange(len(circuit_list))
    x_axis_2 = [x + bar_width for x in x_axis]
    x_axis_3 = [x + bar_width for x in x_axis_2]
    x_axis_4 = [x + bar_width for x in x_axis_3]
    x_axis_5 = [x + bar_width for x in x_axis_4]

    

    plt.bar(x_axis, [res[1]/res[0] for res in full_results], width=bar_width, label="Quartz - No ROQC")
    plt.bar(x_axis_2, [res[2]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 1")
    plt.bar(x_axis_3, [res[3]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 5")
    plt.bar(x_axis_4, [res[4]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 50")
    plt.bar(x_axis_5, [res[5]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 100")

    plt.xlabel("Circuit")
    plt.ylabel("Final Gate Count Percentage of Original")

    plt.xticks(x_axis + 3*bar_width, [circuit[1] for circuit in circuit_list])

    plt.legend()
    plt.savefig(f"result_figures/bar_comparison_timeout_{timeout}.png")
    # plt.show()

def run_quartz(filename, circuit_name, timeout, roqc_interval):
    result = subprocess.run(["./build_docker/test_optimize", f"{filename}", f"{circuit_name}", f"{timeout}", f"{roqc_interval}"], capture_output = True, text=True)
    result_lines = result.stdout.splitlines()
    costs = []
    times = []
    circuit_found = False
    circuit_string = ""
    for line in result_lines:
        words = line.split()
        if words[0] == f"[{circuit_name}]":
            best_cost = float(words[3])
            time = float(words[8])
            costs.append(best_cost)
            times.append(time)

        if words[0] == "OPENQASM":
            circuit_found = True
        if circuit_found:
            circuit_string += (line + '\n')

    final_results = (times, costs)

    with open(f"final_circuits/{circuit_name}_interval_{roqc_interval}_timeout_{timeout}_result.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    return final_results

def run_voqc(qiskit_circuit):
    # vpm = voqc_pass_manager(post_opts=["optimize_nam"])

    # optimized_circuit = vpm.run(qiskit_circuit)

    # return optimzed_circuit.size()

    return 0

if __name__ == '__main__':
    run_experiments()
