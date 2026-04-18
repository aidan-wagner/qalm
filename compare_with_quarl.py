import os
import math

circuits = [
    "tof_3",
    "barenco_tof_3",
    "mod5_4",
    "tof_4",
    "barenco_tof_4",
    "tof_5",
    "mod_mult_55",
    "vbe_adder_3",
    "barenco_tof_5",
    "csla_mux_3",
    "rc_adder_6",
    "gf2^4_mult",
    "tof_10",
    "mod_red_21",
    "hwb6",
    "gf2^5_mult",
    "csum_mux_9",
    "barenco_tof_10",
    "qcla_com_7",
    "ham15-low",
    "gf2^6_mult",
    "qcla_adder_10",
    "gf2^7_mult",
    "gf2^8_mult",
    "qcla_mod_7",
    "adder_8",
]

# Original column
original_counts = [
    45, 58, 63, 75, 114, 105, 119, 150, 170, 170, 200, 225,
    255, 278, 259, 347, 420, 450, 443, 443, 495, 521, 669,
    883, 884, 900
]

# Quarl w/ RM
quarl_rm = [
    33, 35, 24, 51, 62, 69, 84, 71, 90, 141, 134, 160, 169,
    179, 193, 257, 224, 236, 258, 323, 352, 380, 484, 648,
    629, 560
]

# Quarl w/o RM
quarl_wo = [
    33, 36, 24, 51, 62, 69, 84, 71, 96, 141, 148, 163, 169,
    177, 196, 266, 282, 274, 277, 322, 386, 391, 530, 755,
    664, 669
]

base_dir = (
    "20251128_exploration_increase_add_1_every_100_with_vanilla4_experiments_with_"
    "local_qalm_poolgen_and_exploration_together/comparison_results"
)

def latex_escape(s):
    repl = {
        "_": r"\_",
        "^": r"\^{}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "~": r"\~{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def count_gates(path):
    if not os.path.isfile(path):
        return None
    count = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#"):
                continue
            if (s.startswith("OPENQASM") or s.startswith("include")
                    or s.startswith("qreg") or s.startswith("creg")):
                continue
            count += 1
    return count

rows = []
reductions = []

for circ, orig, qrm, qwo in zip(circuits, original_counts, quarl_rm, quarl_wo):
    qasm_path = os.path.join(
        base_dir,
        circ,
        "600_3_3_2_1.5_0_0_1_1_result.qasm"
    )
    gc = count_gates(qasm_path)

    # skip entire row if missing
    if gc is None:
        continue

    # determine minimum among optimizer columns (not including Original)
    optimizer_vals = [qrm, qwo, gc]
    m = min(optimizer_vals)

    def fmt(v):
        return f"\\textbf{{{v}}}" if v == m else f"{v}"

    rows.append((latex_escape(circ), orig, fmt(qrm), fmt(qwo), fmt(gc)))

    reductions.append((qrm / orig, qwo / orig, gc / orig))

def avg_mean(xs):
    return 1 - sum(xs) / len(xs)

def geo_mean(xs):
    return 1 - math.exp(sum(math.log(x) for x in xs) / len(xs))

am_qrm = avg_mean([r[0] for r in reductions])
am_qwo = avg_mean([r[1] for r in reductions])
am_gc  = avg_mean([r[2] for r in reductions])
gm_qrm = geo_mean([r[0] for r in reductions])
gm_qwo = geo_mean([r[1] for r in reductions])
gm_gc  = geo_mean([r[2] for r in reductions])

# LaTeX table
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("Circuit & Original & Quarl w/ R.M. & Quarl w/o R.M. & This Work \\\\")
print("\\hline")

for circ, orig, qrm_fmt, qwo_fmt, gc_fmt in rows:
    print(f"{circ} & {orig} & {qrm_fmt} & {qwo_fmt} & {gc_fmt} \\\\")

print("\\hline")
print(f"Avg. Reduction & - & {am_qrm:.3f} & {am_qwo:.3f} & {am_gc:.3f} \\\\")
print(f"Geo. Mean Reduction & - & {gm_qrm:.3f} & {gm_qwo:.3f} & {gm_gc:.3f} \\\\")
print("\\hline")
print("\\end{tabular}")
