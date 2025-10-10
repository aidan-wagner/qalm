import subprocess
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import numpy as np

import equiv_verification

def tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    roqc_interval = arguments[3]
    return (roqc_interval, run_quartz(filename, circuit_name, timeout, roqc_interval))

def run_experiments():

    timeout = 60 * 5
    validate = False
    # roqc_intervals = [0, 1, 5, 50, 100]
    roqc_intervals = [0, 1, 5]
    # circuit_list = [("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3")]
    circuit_list = [("circuit/nam_circs/adder_8.qasm", "adder_8"),
                    # ("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                    # ("circuit/nam_circs/barenco_tof_4.qasm", "barenco_tof_4"),
                    # ("circuit/nam_circs/barenco_tof_5.qasm", "barenco_tof_5"),
                    ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
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
                    ];

    full_results = []

    for circuit in circuit_list:
        print(f"Running experiments for {circuit[1]}")

        results = {}
        arguments = [(circuit[0], circuit[1], timeout, roqc_interval) for roqc_interval in roqc_intervals]

        try:
            with open(f"pickled_results/{circuit[1]}_{timeout}_results.pkl", "rb") as f:
                results = pickle.load(f)
        except:
            with multiprocessing.Pool(8) as pool:
                initial_results = pool.map(tester, arguments)


                for result in initial_results:
                    results[result[0]] = result[1]

        if validate:
            filenames = [f"{circuit[1]}_interval_{roqc_interval}_result.qasm" for roqc_interval in roqc_intervals]

            for filename in filenames:
                assert(equiv_verification.monte_carlo_compare_by_filename(circuit[0], filename))
                print(f"====Test Passed====")

        voqc_result = run_voqc(circuit[0])

        qalm_results_10_10 = run_qalm(circuit[0], circuit[1], timeout, 10, 10, 1.5, 0, 0)

        qalm_results_10_10_decr = run_qalm(circuit[0], circuit[1], timeout, 10, 10, 1.5, 0, 1)

        qalm_results_50_50 = run_qalm(circuit[0], circuit[1], timeout, 50, 50, 1.5, 0, 0)

        qalm_results_10_10_exp_increase = run_qalm(circuit[0], circuit[1], timeout, 10, 10, 1.5, 0, 0)

        original_gate_count = voqc_result[1][0]


        # with open(f"pickled_results/{circuit[1]}_{timeout}_results.pkl", 'wb') as f:
        #    f.truncate(0)
        #    pickle.dump(results, f)


        plt.plot(results[0][0], results[0][1], label = "Quartz - No ROQC")
        plt.plot(results[1][0], results[1][1], label = "Quartz - ROQC Interval = 1")
        plt.plot(results[5][0], results[5][1], label = "Quartz - ROQC Interval = 5")
        # plt.plot(results[50][0], results[50][1], label = "Quartz - ROQC Interval = 50")
        # plt.plot(results[100][0], results[100][1], label = "Quartz - ROQC Interval = 100")
        plt.plot(voqc_result[0], voqc_result[1], label = "VOQC - Single Run") 
        plt.plot(qalm_results_10_10[0], qalm_results_10_10[1], label = "QALM 10 10")
        plt.plot(qalm_results_10_10_decr[0], qalm_results_10_10_decr[1], label = "QALM 10 10 - strictly decrease")
        plt.plot(qalm_results_50_50[0], qalm_results_50_50[1], label = "QALM 50 50")
        plt.plot(qalm_results_10_10_exp_increase[0], qalm_results_10_10_exp_increase[1], label = "QALM 10 10 - increase explore")

        plt.xlabel("Time (s)")
        plt.ylabel("Gate Count")

        plt.title(f"Optimization with different ROQC intervals - {circuit[1]} - ECC set (5,3)")

        plt.legend()
        #plt.show()
        plt.savefig(f"comparison_results/result_figure_{circuit[1]}_{timeout}_seconds.png")
        plt.clf()


        final_gate_counts = [original_gate_count,
                             results[0][1][-1],
                             results[1][1][-1],
                             results[5][1][-1],
                             # results[50][1][-1],
                             # results[100][1][-1],
                             voqc_result[1][-1]]

        full_results.append(final_gate_counts)

        
        # with open(f"fresh_results/qualm_bench/nam/qualm/results_{circuit[1]}.txt", 'w') as f:
        #     f.write(f"qualm_1 {original_gate_count}/{final_gate_counts[2]}\n")
        #     f.write(f"qualm_5 {original_gate_count}/{final_gate_counts[3]}\n")
        #     f.write(f"qualm_50 {original_gate_count}/{final_gate_counts[4]}\n")
        #     f.write(f"qualm_100 {original_gate_count}/{final_gate_counts[5]}\n")
        #     f.write(f"quartz {original_gate_count}/{final_gate_counts[0]}\n")
        #     f.write(f"voqc {original_gate_count}/{final_gate_counts[6]}\n")


    bar_width = 0.1
    x_axis = np.arange(len(circuit_list))
    x_axis_2 = [x + bar_width for x in x_axis]
    x_axis_3 = [x + bar_width for x in x_axis_2]
    x_axis_4 = [x + bar_width for x in x_axis_3]
    x_axis_5 = [x + bar_width for x in x_axis_4]
    x_axis_6 = [x + bar_width for x in x_axis_5]

    

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

def run_quartz(filename, circuit_name, timeout, roqc_interval):
    result = subprocess.run(["./build_non_conda/test_optimize", f"{filename}", f"{circuit_name}", f"{timeout}", f"{roqc_interval}"], capture_output = True, text=True)
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

    with open(f"comparison_results/{circuit_name}_interval_{roqc_interval}_result.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    return final_results

def run_qalm(filename, circuit_name, timeout, exploration_pool_size, exploration_steps, repeat_tolerance, exploration_increase, strictly_increasing):
    # Interval doesn't matter for test_qalm
    result = subprocess.run(["./build_non_conda/test_qalm", f"{filename}", f"{circuit_name}", f"{timeout}", f"{exploration_pool_size}", f"{exploration_steps}", f"{repeat_tolerance}", f"{exploration_increase}"], capture_output = True, text=True)
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

    with open(f"comparison_results/{circuit_name}_qalm_result.qasm", 'w') as f:
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
