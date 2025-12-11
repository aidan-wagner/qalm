import subprocess
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import numpy as np
from enum import Enum
import qiskit
import os

import equiv_verification

class OptimizationType(Enum):
    roqc_interval = 1
    qalm = 2
    quartz = 3

def tester(arguments):
    opt_type = arguments[0]
    if opt_type == OptimizationType.roqc_interval:
        return roqc_interval_tester(arguments[1])
    elif opt_type == OptimizationType.qalm:
        return qalm_tester(arguments[1])
    elif opt_type == OptimizationType.quartz:
        return quartz_tester(arguments[1])
    else:
        print("whoops")

def quartz_tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    return run_original(filename, circuit_name, timeout)

def roqc_interval_tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    roqc_interval = arguments[3]
    greedy_start = arguments[4]
    two_way_rm = arguments[5]
    eccset = "eccset/RZ_RX_RY_H_CX_4_3_complete_ECC_set.json" if "../.." in circuit_name else arguments[6]
    return run_interval(filename, circuit_name, timeout, roqc_interval, greedy_start, two_way_rm, eccset)

def qalm_tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    intial_pool_size = arguments[3]
    exploration_pool = arguments[4]
    exploration_steps = arguments[5]
    repeat_tolerance = arguments[6]
    exploration_increase = arguments[7]
    no_increase = arguments[8]
    only_do_local_transformations = arguments[9]
    greedy_start = arguments[10]
    two_way_rm = arguments[11]
    eccset = "eccset/RZ_RX_RY_H_CX_4_3_complete_ECC_set.json" if "../.." in circuit_name else arguments[12]
    return run_qalm(filename, circuit_name, timeout, intial_pool_size, exploration_pool, exploration_steps, repeat_tolerance, exploration_increase, no_increase, only_do_local_transformations, greedy_start, two_way_rm, eccset)

def run_experiments():

    if os.path.isdir("comparison_results"):
        raise Exception("comparison_results found in current directory. These must be removed")

    os.mkdir("comparison_results")

    if not os.path.isdir("pickled_results"):
        os.mkdir("pickled_results")


    roqc_time = 0.0
    explore_time = 0.0
    pool_gen_time = 0.0

    timeout = 60 * 60
    circuit_list = [("circuit/nam_rm_circs/adder_8.qasm", "adder_8"),
                    ("circuit/nam_rm_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                    ("circuit/nam_rm_circs/barenco_tof_4.qasm", "barenco_tof_4"),
                    ("circuit/nam_rm_circs/barenco_tof_5.qasm", "barenco_tof_5"),
                    ("circuit/nam_rm_circs/barenco_tof_10.qasm", "barenco_tof_10"),
                    ("circuit/nam_rm_circs/csla_mux_3.qasm", "csla_mux_3"),
                    ("circuit/nam_rm_circs/csum_mux_9.qasm", "csum_mux_9"),
                    ("circuit/nam_rm_circs/gf2^4_mult.qasm", "gf2^4_mult"),
                    ("circuit/nam_rm_circs/gf2^5_mult.qasm", "gf2^5_mult"),
                    ("circuit/nam_rm_circs/gf2^6_mult.qasm", "gf2^6_mult"),
                    ("circuit/nam_rm_circs/gf2^7_mult.qasm", "gf2^7_mult"),
                    ("circuit/nam_rm_circs/gf2^8_mult.qasm", "gf2^8_mult"),
                    ("circuit/nam_rm_circs/gf2^9_mult.qasm", "gf2^9_mult"),
                    ("circuit/nam_rm_circs/gf2^10_mult.qasm", "gf2^10_mult"),
                    ("circuit/nam_rm_circs/hwb6.qasm", "hwb6"),
                    ("circuit/nam_rm_circs/ham15-low.qasm", "ham15-low"),
                    ("circuit/nam_rm_circs/mod5_4.qasm", "mod5_4"),
                    ("circuit/nam_rm_circs/mod_mult_55.qasm", "mod_mult_55"),
                    ("circuit/nam_rm_circs/mod_red_21.qasm", "mod_red_21"),
                    ("circuit/nam_rm_circs/qcla_adder_10.qasm", "qcla_adder_10"),
                    ("circuit/nam_rm_circs/qcla_com_7.qasm", "qcla_com_7"),
                    ("circuit/nam_rm_circs/qcla_mod_7.qasm", "qcla_mod_7"),
                    ("circuit/nam_rm_circs/rc_adder_6.qasm", "rc_adder_6"),
                    ("circuit/nam_rm_circs/tof_3.qasm", "tof_3"),
                    ("circuit/nam_rm_circs/tof_4.qasm", "tof_4"),
                    ("circuit/nam_rm_circs/tof_5.qasm", "tof_5"),
                    ("circuit/nam_rm_circs/tof_10.qasm", "tof_10"),
                    ("circuit/nam_rm_circs/vbe_adder_3.qasm", "vbe_adder_3"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/ucc/qcnn_N10_4layers_basis_rz_rx_ry_h_cx.qasm", "qcnn_10"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/ucc/qcnn_N100_7layers_basis_rz_rx_ry_h_cx.qasm", "qcnn_100"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qaoa_barabasi_albert_N10_3reps_basis_rz_rx_ry_cx.qasm", "qaoa_10"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qaoa_barabasi_albert_N100_3reps_basis_rz_rx_ry_cx.qasm", "qaoa_100"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qft_N010_basis_rz_rx_ry_cx.qasm", "qft_10"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qft_N100_basis_rz_rx_ry_cx.qasm", "qft_100"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qv_N010_12345_basis_rz_rx_ry_cx.qasm", "qv_10"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/qv_N100_12345_basis_rz_rx_ry_cx.qasm", "qv_100"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/square_heisenberg_N9_basis_rz_rx_ry_cx.qasm", "heisenberg_9"),
                    # ("../../PycharmProjects/ucc-bench/benchmarks/circuits/benchpress/square_heisenberg_N100_basis_rz_rx_ry_cx.qasm", "heisenberg_100"),
                    ]

    experiments = [
        (OptimizationType.quartz, (0,)),
        # (OptimizationType.qalm, (1, 1, 2, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")),
        # (OptimizationType.qalm, (2, 1, 2, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")),
        # (OptimizationType.qalm, (3, 1, 2, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")),
        # (OptimizationType.qalm, (1, 2, 2, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")),
        # (OptimizationType.qalm, (1, 3, 2, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")),
    ]

    graph_labels = [
        "Quartz - with preprocessing",
        # "I1, G1, E2, ECC_5_3, all_xfers, local, greedy, two_way, 5_3",
        # "I2, G1, E2, ECC_5_3, all_xfers, local, greedy, two_way, 5_3",
        # "I3, G1, E2, ECC_5_3, all_xfers, local, greedy, two_way, 5_3",
        # "I1, G2, E2, ECC_5_3, all_xfers, local, greedy, two_way, 5_3",
        # "I1, G3, E2, ECC_5_3, all_xfers, local, greedy, two_way, 5_3",
    ]

    # voqc_avg = 0
    exp_avgs = [0] * len(experiments)
    arguments = []

    # The order here REALLY matters
    # Multiprocessing pool maintains order of results and this
    # is used in parsing of results
    for circuit in circuit_list:
        print(f"Running experiments for {circuit[1]}")

        os.mkdir("comparison_results/" + circuit[1])

        arguments += [(experiment[0], (circuit[0], circuit[1], timeout) + experiment[1]) for experiment in experiments]

    with multiprocessing.Pool(32) as pool:
        raw_results = pool.map(tester, arguments)

    with open("pickled_results/combined_raw_results.pkl", 'wb') as f:
        f.truncate(0)
        pickle.dump(raw_results, f)
    with open("pickled_results/combined_exp_params.pkl", 'wb') as f:
        f.truncate(0)
        pickle.dump(experiments, f)
    with open("pickled_results/combined_graph_labels.pkl", 'wb') as f:
        f.truncate(0)
        pickle.dump(graph_labels, f)



    result_dict = {}
    for circuit in circuit_list:
        result_dict[circuit[1]] = []

    assert(len(raw_results) == len(experiments) * len(circuit_list))

    for circuit_index in range(len(circuit_list)):
        for exp_index in range(len(experiments)):
            result_dict[circuit_list[circuit_index][1]].append(raw_results[exp_index + circuit_index*len(experiments)])

    for circuit in circuit_list:
        results = result_dict[circuit[1]]
        original_qc = qiskit.QuantumCircuit.from_qasm_file(circuit[0].replace('nam-benchmarks', 'nam_circs'))
        original_gate_count = original_qc.size()

        fig = plt.figure()
        ax = fig.add_subplot()
        colormap = plt.get_cmap("gist_rainbow")
        ax.set_prop_cycle(color=[colormap(1.*i/len(experiments)) for i in range(len(experiments))])


        for i in range(len(results)):
            ax.plot(results[i][0], results[i][1], label = graph_labels[i], linestyle=['-', '--', ':'][i % 3])
            exp_avgs[i] += float(results[i][1][-1]) / float(original_gate_count) if len(results[i][1]) > 0 else 1

        # ax.plot(voqc_result[0], voqc_result[1], label = "Voqc")


        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gate Count")

        ax.set_title(f"Optimization Experiments - {circuit[1]} - ECC set (6,3)")


        fig.legend(bbox_to_anchor=(1.1, 1), loc="upper right", fontsize="x-small")
        fig.tight_layout()
        #plt.show()
        fig.savefig(f"comparison_results/result_figure_{circuit[1]}_{timeout}_seconds.png", dpi=500)
        fig.clf()


        # final_gate_counts = [original_gate_count,
        #                      results[0][1][-1],
        #                      results[1][1][-1],
        #                      results[5][1][-1],
        #                      # results[50][1][-1],
        #                      # results[100][1][-1],
        #                      voqc_result[1][-1]]

        # full_results.append(final_gate_counts)

        
        # with open(f"fresh_results/qualm_bench/nam/qualm/results_{circuit[1]}.txt", 'w') as f:
        #     f.write(f"qualm_1 {original_gate_count}/{final_gate_counts[2]}\n")
        #     f.write(f"qualm_5 {original_gate_count}/{final_gate_counts[3]}\n")
        #     f.write(f"qualm_50 {original_gate_count}/{final_gate_counts[4]}\n")
        #     f.write(f"qualm_100 {original_gate_count}/{final_gate_counts[5]}\n")
        #     f.write(f"quartz {original_gate_count}/{final_gate_counts[0]}\n")
        #     f.write(f"voqc {original_gate_count}/{final_gate_counts[6]}\n")


    # bar_width = 0.1
    # x_axis = np.arange(len(circuit_list))
    # x_axis_2 = [x + bar_width for x in x_axis]
    # x_axis_3 = [x + bar_width for x in x_axis_2]
    # x_axis_4 = [x + bar_width for x in x_axis_3]
    # x_axis_5 = [x + bar_width for x in x_axis_4]
    # x_axis_6 = [x + bar_width for x in x_axis_5]

    

    # plt.bar(x_axis, [res[1]/res[0] for res in full_results], width=bar_width, label="Quartz - No ROQC")
    # plt.bar(x_axis_2, [res[2]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 1")
    # plt.bar(x_axis_3, [res[3]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 5")
    # plt.bar(x_axis_4, [res[4]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 50")
    # plt.bar(x_axis_5, [res[5]/res[0] for res in full_results], width=bar_width, label="Quartz - ROQC Interval = 100")
    # plt.bar(x_axis_6, [res[6]/res[0] for res in full_results], width=bar_width, label="VOQC - Single Run")

    # plt.xlabel("Circuit")
    # plt.ylabel("Final Gate Count Percentage of Original")

    # plt.xticks(x_axis + 3*bar_width, [circuit[1] for circuit in circuit_list])

    # plt.legend()
    # plt.show()

    # voqc_avg /= len(circuit_list)
    exp_avgs_final = [exp_avg / len(circuit_list) for exp_avg in exp_avgs]

    # print("VOQC Average: ", voqc_avg)
    for i in range(len(exp_avgs_final)):
        print(graph_labels[i], "Average: ", exp_avgs_final[i])

    total_runs = len(circuit_list) * (len(experiments) - 5) # dont want to include roqc interval experiments

    print("Percentage of time spent in ROQC phase:", roqc_time/total_runs)
    print("Percentage of time spent in Explore phase:", explore_time/total_runs)
    print("Percentage of time spent in Pool Gen phase:", pool_gen_time/total_runs)

def run_interval(filename, circuit_name, timeout, roqc_interval, greedy_start, two_way_rm, eccset):
    ecc_name = eccset.split("/")[-1]
    result = subprocess.run(["./build/test_optimize", f"{filename}", f"{circuit_name}", f"{timeout}", f"{roqc_interval}", f"{greedy_start}", f"{two_way_rm}", f"{eccset}"], capture_output = True, text=True)
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

    with open(f"comparison_results/{circuit_name}/interval_{roqc_interval}_{greedy_start}_{two_way_rm}_{ecc_name}_result.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    return final_results

def run_original(filename, circuit_name, timeout):
    result = subprocess.run(["./build/test_original", f"{filename}", f"{circuit_name}", f"{timeout}"], capture_output = True, text=True)
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

    with open(f"comparison_results/{circuit_name}/quartz_result.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    return final_results

def run_qalm(filename, circuit_name, timeout, initial_pool_size, exploration_pool_size, exploration_steps, repeat_tolerance, exploration_increase, no_increase, only_do_local_transformations, greedy_start, two_way_rm, eccset):
    # Interval doesn't matter for test_qalm
    ecc_name = eccset.split("/")[-1]
    result = subprocess.run(["./build/test_qalm", f"{filename}", f"{circuit_name}", f"{timeout}", f"{initial_pool_size}", f"{exploration_pool_size}", f"{exploration_steps}", f"{repeat_tolerance}", f"{exploration_increase}", f"{no_increase}", f"{only_do_local_transformations}", f"{greedy_start}", f"{two_way_rm}", f"{eccset}"], capture_output = True, text=True)
    result_lines = result.stdout.splitlines()
    costs = []
    times = []
    circuit_found = False
    circuit_string = ""

    circuit_roqc_time = 0.0
    circuit_explore_time = 0.0
    circuit_pool_gen_time = 0.0


    for line in result_lines:
        words = line.split()
        if words[0] == f"[{circuit_name}]":
            best_cost = float(words[3])
            time = float(words[8])
            costs.append(best_cost)
            times.append(time)

        if words[0] == "ROQC" and words[1] == "time:":
            if "e" not in words[2]:
                circuit_roqc_time = float(words[2])
        if words[0] == "Explore" and words[1] == "time:":
            if "e" not in words[2]:
                circuit_explore_time = float(words[2])
        if words[0] == "Pool" and words[1] == "Gen" and words[2] == "time:":
            if "e" not in words[3]:
                circuit_pool_gen_time = float(words[3])

        if words[0] == "OPENQASM":
            circuit_found = True
        if circuit_found:
            circuit_string += (line + '\n')

    final_results = (times, costs)

    with open(f"pickled_results/{circuit_name}_{timeout}_{initial_pool_size}_{exploration_pool_size}_{exploration_steps}_{repeat_tolerance}_{exploration_increase}_{no_increase}_{only_do_local_transformations}_{greedy_start}_{two_way_rm}_{ecc_name}_time_benchmark.pkl", 'wb') as f:
        f.truncate(0)
        pickle.dump((circuit_roqc_time, circuit_explore_time, circuit_pool_gen_time), f)


    with open(f"comparison_results/{circuit_name}/{timeout}_{initial_pool_size}_{exploration_pool_size}_{exploration_steps}_{repeat_tolerance}_{exploration_increase}_{no_increase}_{only_do_local_transformations}_{greedy_start}_{two_way_rm}_{ecc_name}_result.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    return final_results

def run_voqc(filename):
    result = subprocess.run(["../roqc/voqc_exec/voqc_exec_linux", "-f", filename, "-o", "comparison_results/voqc_comp_result.qasm"], capture_output=True, text=True)

    original_gate_count = 0
    new_gate_count = 0
    opt_time = 0.0

    for line in result.stdout.splitlines():
        words = line.split()
        if words[0] == "After":
            new_gate_count = int(words[5])
        if words[0] == "Optimization":
            opt_time = float(words[2])
        if words[0] == "Input":
            original_gate_count = int(words[3])

    return ([0, opt_time], [original_gate_count, new_gate_count])


if __name__ == '__main__':
    run_experiments()
