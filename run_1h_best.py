#!/usr/bin/env python3
"""Run best config (I3,G3,E3,two_way,5_3) on nam_rm_circs for 1h, all circuits in parallel."""
import os, sys, pickle, multiprocessing, numpy as np
import qiskit

sys.path.insert(0, "qalm_experiments")
from comparison_experiments import run_qalm

TIMEOUT = 3600
# I3, G3, E3, repeat_tolerance=1.5, exploration_increase=0, no_increase=0,
# only_do_local=1, greedy_start=1, two_way_rm=1, eccset=Nam_5_3
BEST_CONFIG = (3, 3, 3, 1.5, 0, 0, 1, 1, 1, "eccset/Nam_5_3_complete_ECC_set.json")

CIRCUIT_LIST = [
    ("circuit/nam_rm_circs/adder_8.qasm",        "adder_8_rm"),
    ("circuit/nam_rm_circs/barenco_tof_3.qasm",  "barenco_tof_3_rm"),
    ("circuit/nam_rm_circs/barenco_tof_4.qasm",  "barenco_tof_4_rm"),
    ("circuit/nam_rm_circs/barenco_tof_5.qasm",  "barenco_tof_5_rm"),
    ("circuit/nam_rm_circs/barenco_tof_10.qasm", "barenco_tof_10_rm"),
    ("circuit/nam_rm_circs/csla_mux_3.qasm",     "csla_mux_3_rm"),
    ("circuit/nam_rm_circs/csum_mux_9.qasm",     "csum_mux_9_rm"),
    ("circuit/nam_rm_circs/gf2^4_mult.qasm",     "gf2^4_mult_rm"),
    ("circuit/nam_rm_circs/gf2^5_mult.qasm",     "gf2^5_mult_rm"),
    ("circuit/nam_rm_circs/gf2^6_mult.qasm",     "gf2^6_mult_rm"),
    ("circuit/nam_rm_circs/gf2^7_mult.qasm",     "gf2^7_mult_rm"),
    ("circuit/nam_rm_circs/gf2^8_mult.qasm",     "gf2^8_mult_rm"),
    ("circuit/nam_rm_circs/gf2^9_mult.qasm",     "gf2^9_mult_rm"),
    ("circuit/nam_rm_circs/gf2^10_mult.qasm",    "gf2^10_mult_rm"),
    ("circuit/nam_rm_circs/mod5_4.qasm",         "mod5_4_rm"),
    ("circuit/nam_rm_circs/mod_mult_55.qasm",    "mod_mult_55_rm"),
    ("circuit/nam_rm_circs/mod_red_21.qasm",     "mod_red_21_rm"),
    ("circuit/nam_rm_circs/qcla_adder_10.qasm",  "qcla_adder_10_rm"),
    ("circuit/nam_rm_circs/qcla_com_7.qasm",     "qcla_com_7_rm"),
    ("circuit/nam_rm_circs/qcla_mod_7.qasm",     "qcla_mod_7_rm"),
    ("circuit/nam_rm_circs/rc_adder_6.qasm",     "rc_adder_6_rm"),
    ("circuit/nam_rm_circs/tof_3.qasm",          "tof_3_rm"),
    ("circuit/nam_rm_circs/tof_4.qasm",          "tof_4_rm"),
    ("circuit/nam_rm_circs/tof_5.qasm",          "tof_5_rm"),
    ("circuit/nam_rm_circs/tof_10.qasm",         "tof_10_rm"),
    ("circuit/nam_rm_circs/vbe_adder_3.qasm",    "vbe_adder_3_rm"),
]


def run_one(args):
    circuit_path, circuit_name = args
    pkl_path = f"pickled_results/{circuit_name}_{TIMEOUT}_results.pkl"
    if os.path.exists(pkl_path):
        print(f"[cached]  {circuit_name}", flush=True)
        with open(pkl_path, "rb") as f:
            return circuit_name, pickle.load(f)[0]
    print(f"[start]   {circuit_name}", flush=True)
    result = run_qalm(circuit_path, circuit_name, TIMEOUT, *BEST_CONFIG)
    with open(pkl_path, "wb") as f:
        pickle.dump([result], f)
    final = result[1][-1] if result[1] else "?"
    print(f"[done]    {circuit_name}  final={final}", flush=True)
    return circuit_name, result


if __name__ == "__main__":
    os.makedirs("pickled_results", exist_ok=True)
    with multiprocessing.Pool(len(CIRCUIT_LIST)) as pool:
        results = pool.map(run_one, CIRCUIT_LIST)

    # Compute avg & geomean reduction vs nam_circs originals
    ratio_sum = 0.0
    log_sum = 0.0
    print(f"\n{'Circuit':<22} {'Orig':>6} {'Final':>6} {'Reduction':>10}")
    print("-" * 48)
    for circuit_name, (times, counts) in results:
        base_name = circuit_name.replace("_rm", "")
        orig_path = f"circuit/nam_circs/{base_name}.qasm"
        orig = qiskit.QuantumCircuit.from_qasm_file(orig_path).size()
        final = counts[-1] if counts else orig
        red = (orig - final) / orig
        ratio_sum += final / orig
        log_sum += np.log(final / orig)
        print(f"{circuit_name:<22} {orig:>6} {final:>6} {red:>9.1%}")

    n = len(CIRCUIT_LIST)
    avg_ratio = ratio_sum / n
    geo_ratio = np.exp(log_sum / n)
    print(f"\nAvg. reduction (vs nam_circs):     {1 - avg_ratio:.1%}")
    print(f"Geomean reduction (vs nam_circs):  {1 - geo_ratio:.1%}")
