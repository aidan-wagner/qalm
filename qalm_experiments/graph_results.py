import matplotlib.pyplot as plt
import csv
import numpy as np

def graph_results():
    results_file = "qalm_bench_results.csv"
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

    with open(results_file, "r") as f:
        csv_reader = csv.reader(f)
        fields = next(csv_reader)
        results = {}
        for row in csv_reader:
            results[row[0]] = row[1:]


    bar_width = 0.1
    x_axis_1 = np.arange(len(fields) - 1)
    x_axis_2 = [x + bar_width for x in x_axis_1]
    x_axis_3 = [x + bar_width for x in x_axis_2]
    x_axis_4 = [x + bar_width for x in x_axis_3]
    x_axis_5 = [x + bar_width for x in x_axis_4]
    x_axis_6 = [x + bar_width for x in x_axis_5]
    x_axis_7 = [x + bar_width for x in x_axis_6]
    x_axis_8 = [x + bar_width for x in x_axis_7]
    x_axis_9 = [x + bar_width for x in x_axis_8]

    plt.bar(x_axis_1, [int(results["qalm_0"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Quartz - No ROQC")
    plt.bar(x_axis_2, [int(results["qalm_1"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Quartz - ROQC Interval 1")
    plt.bar(x_axis_3, [int(results["qalm_5"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Quartz - ROQC Interval 5")
    plt.bar(x_axis_4, [int(results["qalm_50"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Quartz - ROQC Interval 50")
    plt.bar(x_axis_5, [int(results["qalm_100"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Quartz - ROQC Interval 100")
    plt.bar(x_axis_6, [int(results["guoq"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="GUOQ")
    plt.bar(x_axis_7, [int(results["queso"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="QUESO")
    plt.bar(x_axis_8, [int(results["qiskit"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="Qiskit")
    plt.bar(x_axis_9, [int(results["voqc"][i])/int(results["initial"][i]) for i in range(len(fields) - 1)], width=bar_width, label="VOQC")

    plt.xlabel("Circuit")
    plt.ylabel("Final Gate Count Percentage of Original")

    plt.xticks(x_axis_1 + 3*bar_width, [circuit[1] for circuit in circuit_list])

    plt.legend()
    plt.savefig("qalm_bench_results.png")

if __name__ == '__main__':
    graph_results()
