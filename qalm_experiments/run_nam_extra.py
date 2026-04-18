"""Run tof_10 and vbe_adder_3 under nam source, then insert after nam rows in the 3600s CSV."""

import csv
import multiprocessing
import os
import sys

# Reuse everything from benchmark_guoq
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_guoq as bq

EXTRA = [
    ("circuit/nam_circs/tof_10.qasm",      "tof_10",      "nam"),
    ("circuit/nam_circs/vbe_adder_3.qasm", "vbe_adder_3", "nam"),
]

def main():
    os.makedirs(f"{bq.OUT_DIR}/pkl", exist_ok=True)

    print(f"Running {len(EXTRA)} extra nam circuits with timeout={bq.TIMEOUT}s")
    with multiprocessing.Pool(len(EXTRA)) as pool:
        results = pool.map(bq.run_one, EXTRA)

    # Read existing CSV
    csv_path = f"{bq.OUT_DIR}/results_{bq.TIMEOUT}s.csv"
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Build new rows for the two extra circuits
    new_rows = []
    for (circ_path, circ_name, source), result in zip(EXTRA, results):
        times, totals, cxs, peak_kb, status = result
        orig_total, orig_cx = bq._orig_counts(circ_path)
        opt_total = int(totals[-1]) if totals else -1
        opt_cx    = int(cxs[-1])    if cxs    else -1

        def pct(opt, orig):
            return f"{(1 - opt/orig)*100:.2f}" if orig > 0 and opt >= 0 else "nan"

        chk_vals = []
        for t in bq.CHECKPOINTS:
            chk_vals.append(bq._interp_at(times, totals, t) or "")
            chk_vals.append(bq._interp_at(times, cxs, t) or "")

        new_rows.append(
            [source, circ_name,
             orig_total, orig_cx,
             opt_total, opt_cx,
             pct(opt_total, orig_total), pct(opt_cx, orig_cx),
             f"{peak_kb/1024:.1f}", status]
            + chk_vals
        )
        print(f"[{source}] {circ_name}: opt={opt_total}, peak={peak_kb/1024:.0f} MB, status={status}")

    # Find insertion point: after the last nam row
    last_nam_idx = max(i for i, r in enumerate(rows) if r and r[0] == "nam")
    rows = rows[:last_nam_idx + 1] + new_rows + rows[last_nam_idx + 1:]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nInserted after row {last_nam_idx} in {csv_path}")

if __name__ == "__main__":
    main()
