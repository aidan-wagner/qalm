"""Build the input manifests S248, S26, S26PP (name, path, sha256, n_qubits, total, cx).

Run from the repo root: python3 -I repro/inputs/make_manifests.py
"""
import csv
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from qasm_count import count_file  # noqa: E402

S248_DIR = os.path.expanduser("~/guoq/benchmarks/nam_rz")
S248_EXCLUDE = {"qft_10", "qft_16"}  # exact identity circuits
S26 = ["adder_8", "barenco_tof_3", "barenco_tof_4", "barenco_tof_5", "barenco_tof_10",
       "csla_mux_3", "csum_mux_9", "gf2^4_mult", "gf2^5_mult", "gf2^6_mult", "gf2^7_mult",
       "gf2^8_mult", "gf2^9_mult", "gf2^10_mult", "mod5_4", "mod_mult_55", "mod_red_21",
       "qcla_adder_10", "qcla_com_7", "qcla_mod_7", "rc_adder_6", "tof_3", "tof_4", "tof_5",
       "tof_10", "vbe_adder_3"]


def row(name, path):
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    c = count_file(path)
    return {"name": name, "path": os.path.abspath(path), "sha256": sha,
            "n_qubits": c["n_qubits"], "total": c["total"], "cx": c["cx"]}


def write(fname, rows):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "path", "sha256", "n_qubits", "total", "cx"])
        w.writeheader()
        w.writerows(rows)
    print(f"{fname}: {len(rows)} rows")


def main():
    s248 = [row(os.path.basename(p)[:-5], p)
            for p in sorted(glob.glob(os.path.join(S248_DIR, "*.qasm")))
            if os.path.basename(p)[:-5] not in S248_EXCLUDE]
    write("S248.csv", s248)
    write("S26.csv", [row(n, f"circuit/nam_circs/{n}.qasm") for n in S26])
    write("S26PP.csv", [row(n, f"circuit/nam_rm_circs/{n}.qasm") for n in S26])


if __name__ == "__main__":
    main()
