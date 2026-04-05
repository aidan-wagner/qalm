"""Plot time vs geomean gate-count reduction for nam, nam_rm (26-circuit subset), and guoq."""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
TIMEOUT  = int(sys.argv[1]) if len(sys.argv) > 1 else 600
CSV_PATH = f"guoq_benchmark_results/results_{TIMEOUT}s.csv"
OUT_PATH = f"guoq_benchmark_results/time_vs_geomean_reduction_{TIMEOUT}s.pdf"

def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_time_cols(rows):
    if not rows:
        return []
    sample = rows[0]
    result = []
    for key in sample:
        if key.startswith("total_t"):
            try:
                t = int(key[len("total_t"):])
                result.append((key, t))
            except ValueError:
                pass
    return sorted(result, key=lambda x: x[1])

def geomean(values):
    if not values:
        return float("nan")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))

def compute_series(rows, source_filter, circuit_filter=None, orig_lookup=None):
    """Return (times, geomean_reductions).

    orig_lookup: optional dict {circuit -> orig_total} to use as denominator
    instead of each row's own orig_total.
    """
    subset = [r for r in rows if r["source"] == source_filter and r["status"] == "OK"]
    if circuit_filter is not None:
        subset = [r for r in subset if r["circuit"] in circuit_filter]
    if orig_lookup is not None:
        subset = [r for r in subset if r["circuit"] in orig_lookup]

    times = []
    geomeans = []
    for col, t in TIME_COLS:
        reductions = []
        for r in subset:
            orig = float(orig_lookup[r["circuit"]]) if orig_lookup else float(r["orig_total"])
            opt_t = float(r[col])
            if orig > 0:
                reductions.append(opt_t / orig)
        if reductions:
            times.append(t)
            geomeans.append((1 - geomean([max(0.0, red) + 1e-9 for red in reductions])) * 100.0)
    return times, geomeans

rows = read_csv(CSV_PATH)
TIME_COLS = get_time_cols(rows)

# The 26 nam circuits that appear in both nam and nam_rm
nam_circuits = {r["circuit"] for r in rows if r["source"] == "nam"}
nam_rm_circuits = {r["circuit"] for r in rows if r["source"] == "nam_rm"}
nam_rm_26 = nam_circuits & nam_rm_circuits  # intersection = 26-circuit subset

# orig_total from the nam rows (true original gate counts)
nam_orig_lookup = {r["circuit"]: r["orig_total"] for r in rows if r["source"] == "nam" and r["status"] == "OK"}

print(f"nam circuits: {len(nam_circuits)}")
print(f"nam_rm circuits: {len(nam_rm_circuits)}")
print(f"nam_rm 26-circuit subset: {len(nam_rm_26)}")
print(f"guoq circuits: {sum(1 for r in rows if r['source'] == 'guoq' and r['status'] == 'OK')}")

nam_t, nam_g = compute_series(rows, "nam")
nam_rm_t, nam_rm_g = compute_series(rows, "nam_rm", circuit_filter=nam_rm_26)
nam_rm_vs_orig_t, nam_rm_vs_orig_g = compute_series(
    rows, "nam_rm", circuit_filter=nam_rm_26, orig_lookup=nam_orig_lookup
)
guoq_t, guoq_g = compute_series(rows, "guoq")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(nam_t, nam_g, marker="o", label="Nam", linewidth=2)
ax.plot(nam_rm_t, nam_rm_g, marker="s", label="Nam_rm vs. after rm", linewidth=2)
ax.plot(nam_rm_vs_orig_t, nam_rm_vs_orig_g, marker="D", label="Nam_rm vs. orig", linewidth=2)
ax.plot(guoq_t, guoq_g, marker="^", label="GUOQ", linewidth=2)

ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Geomean gate-count reduction (%)", fontsize=12)
ax.set_title(f"Gate-count reduction vs. time ({TIMEOUT} s)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale("linear")
ax.set_xticks([round(TIMEOUT * i / 6) for i in range(7)])

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")

# Print values for reference
print("\nnam:")
for t, g in zip(nam_t, nam_g): print(f"  t={t}s: {g:.2f}%")
print("nam_rm vs. after rm (26):")
for t, g in zip(nam_rm_t, nam_rm_g): print(f"  t={t}s: {g:.2f}%")
print("nam_rm vs. orig (26):")
for t, g in zip(nam_rm_vs_orig_t, nam_rm_vs_orig_g): print(f"  t={t}s: {g:.2f}%")
print("guoq:")
for t, g in zip(guoq_t, guoq_g): print(f"  t={t}s: {g:.2f}%")
