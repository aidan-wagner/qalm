import subprocess
import multiprocessing
import pickle
import os
import tqdm

import equiv_verification

def tester(arguments):
    filename = arguments[0]
    circuit_name = arguments[1]
    timeout = arguments[2]
    roqc_interval = arguments[3]
    run_quartz(filename, circuit_name, timeout, roqc_interval)

def run_experiments():

    if not os.path.exists("../fresh_results/qalm_bench/nam/qalm"):
        os.makedirs("../fresh_results/qalm_bench/nam/qalm")

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

    full_args = []

    for circuit in circuit_list:
        full_args = full_args + [(circuit[0], circuit[1], timeout, roqc_interval) for roqc_interval in roqc_intervals]

    with multiprocessing.Pool(12) as pool:
        pool.map(tester, full_args)

    if validate:
        for circuit in circuit_list:
            filenames = [f"../fresh_results/qalm_bench/nam/qalm/{circuit[1]}_interval_{roqc_interval}_timeout_{timeout}_result_circuit.qasm" for roqc_interval in roqc_intervals]

            for filename in filenames:
                assert(equiv_verification.monte_carlo_compare_by_filename(circuit[0], filename))
                print(f"====Test Passed====")


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

        # At this point, we will have no more messages about the time and costs, and we can begin saving the circuit
        if words[0] == "OPENQASM":
            circuit_found = True
        if circuit_found:
            circuit_string += (line + '\n')

    final_results = (times, costs)

    # Write optimized circuit
    with open(f"../fresh_results/qalm_bench/nam/qalm/{circuit_name}_interval_{roqc_interval}_timeout_{timeout}_result_circuit.qasm", 'w') as f:
        f.truncate(0)
        f.write(circuit_string)

    # Write time/cost results
    with open(f"../fresh_results/qalm_bench/nam/qalm/{circuit_name}_interval_{roqc_interval}_timeout_{timeout}_output.txt", 'w') as f:
        f.truncate(0)
        f.write(str(final_results))

    print(f"Optimization for {circuit_name} with interval {roqc_interval} done!")

if __name__ == '__main__':
    run_experiments()
