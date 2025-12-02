import multiprocessing
from equiv_verification import monte_carlo_compare_by_filename
import os
import signal

def timeout_handler(signum, frame):
    print("Timeout occured")

def tester(arguments):

    signal.signal(signal.SIGALRM, timeout_handler)

    circuit_name = arguments[1][1]
    new_circuit_path = arguments[0]
    original_circuit_path = "~/Quantum/qalm/" + arguments[1][0]
    signal.alarm(60*10)
    try:
        if monte_carlo_compare_by_filename(original_circuit_path, new_circuit_path):
            print(f"Test Passed for {circuit_name} on {new_circuit_path}")
        else:
            print(f"!!!!!!!!!!!!!!!! FAILURE for {circuit_name} on {new_circuit_path} !!!!!!!!!!!!!!!!!!!!!")
    except:
        pass

def validate_results():
    circuit_list = [("circuit/nam_circs/adder_8.qasm", "adder_8"),
                ("circuit/nam_circs/barenco_tof_3.qasm", "barenco_tof_3"),
                ("circuit/nam_circs/barenco_tof_4.qasm", "barenco_tof_4"),
                ("circuit/nam_circs/barenco_tof_5.qasm", "barenco_tof_5"),
                ("circuit/nam_circs/barenco_tof_10.qasm", "barenco_tof_10"),
                ("circuit/nam_circs/csla_mux_3.qasm", "csla_mux_3"),
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


    arguments = []
    circuit_prefix = "comparison_results/"
    for circuit in circuit_list:
        all_files = os.listdir(circuit_prefix + circuit[1])
        for file in all_files:
            if file[-5:] == ".qasm" and "600_1_1_2_1.5_0_0_1_1_0_Nam_5_3" in file:
                arguments.append((circuit_prefix + circuit[1] + "/" + file, circuit))

    with multiprocessing.Pool(4) as pool:
        pool.map(tester, arguments)

if __name__ == '__main__':
    validate_results()
