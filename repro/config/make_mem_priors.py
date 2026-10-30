"""Memory priors (GiB) per (family, circuit) from earlier runs on this machine.

Families: qalm (both QALM configs), quartz. Values are historical peaks; runs
that hit the old 8 GB cap are censored and get CENSORED_GIB. The scheduler
also learns online from finished jobs, so these only need to be roughly right.
Run from the repo root: python3 repro/config/make_mem_priors.py
"""
import csv
import os
from collections import defaultdict

CENSORED_MB = 7900
CENSORED_GIB = 16.0
QUARTZ_OOM_GIB = 32.0

prior = defaultdict(float)


def bump(family, circuit, gib):
    prior[(family, circuit)] = max(prior[(family, circuit)], round(gib, 2))


def mb_to_prior(mb):
    mb = float(mb)
    return CENSORED_GIB if mb >= CENSORED_MB else mb / 1024


for path in ["guoq_benchmark_results/results_3600s.csv", "guoq_benchmark_advk_results/results_3600s.csv",
             "guoq_benchmark_weighted_cost_results/results_3600s.csv",
             "guoq_benchmark_prev_qalm_results/results_60s.csv"]:
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["peak_rss_mb"]:
                bump("qalm", r["circuit"], mb_to_prior(r["peak_rss_mb"]))
# 60 s runs have their own (much smaller) peaks
for r in csv.DictReader(open("guoq_benchmark_prev_qalm_results/results_60s.csv")):
    if r["peak_rss_mb"]:
        bump("qalm_short", r["circuit"], max(0.25, float(r["peak_rss_mb"]) / 1024))
for path in ["greedy_k_guoq_benchmark_results/summary.csv", "single_k_guoq_benchmark_results/summary.csv",
             "greedy_k_ablation_results/summary.csv", "single_k_results/summary.csv"]:
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["peak_mb"]:
                bump("qalm", r["circuit"], mb_to_prior(r["peak_mb"]))
for r in csv.DictReader(open("original_benchmark_results/results_3600s.csv")):
    gib = QUARTZ_OOM_GIB if r["status"].startswith("OOM") else float(r["peak_rss_mb"] or 0) / 1024
    bump("quartz", r["circuit"], gib)

out = "repro/config/mem_priors.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "circuit", "prior_gib"])
    for (fam, c), g in sorted(prior.items()):
        w.writerow([fam, c, g])
print(f"wrote {out}: {len(prior)} rows; >=12 GiB: {sum(1 for g in prior.values() if g >= 12)}")
