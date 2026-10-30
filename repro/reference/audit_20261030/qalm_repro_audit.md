# QALM Paper: Reproducibility Audit

**Do the paper's results have reproducible data on this machine?**

- **Paper:** *QALM: Escaping Local Minima via Interleaved Exploration and Exploitation in Quantum Circuit Optimization*
  - Repo: `/home/cc/seach_and_rule_based_opt`, git HEAD `043ff38`
- **Machine audited:** Chameleon host `qalm` (2× AMD EPYC 7763, 250 GiB RAM)
- **Date:** 2026-10-30
- **Method:**
  - Every empirical claim in the compiled paper was checked: tables, figures, and numbers in the text.
  - Material inside `\begin{comment}` was excluded because it is not compiled.
  - Seven auditors each took one group of results. They located the raw per-circuit data and the generating scripts, then recomputed each number.
  - Seven independent verifiers then re-derived every "reproduced" verdict with their own scripts. They also searched again for anything marked "missing", using content grep, `git log --all`, `git fsck`, pickles and transcripts.
  - For figures with no surviving data, the plotted values were extracted from the PDF vector paths, with axes calibrated on the tick marks, so that the text could be checked against the figure.
  - The headline findings below were also spot-checked by hand.

**Status legend**

| Status | Meaning |
|---|---|
| ✅ REPRODUCED | Raw per-circuit data is on this machine, and recomputing from it gives the paper's value (up to rounding). |
| ⚠️ MINOR_DISCREPANCY | Raw data from the same configuration exists, but it comes from a different run. The value is close but not identical. |
| ❌ CONTRADICTED | The data, or the paper's own numbers, contradict the claim. |
| ◐ DATA_ONLY | Related data exists, but this specific number cannot be derived from it. |
| ✖ NOT_FOUND | No raw data for this result exists anywhere on this machine. |
| – N/A | Non-empirical, or checked against code/config only. |

---

## 1. Bottom line

Only part of the paper can be reproduced from this machine.

**What reproduces exactly from raw data:**
- On the 26-circuit Quartz suite (Table `eval_end_to_end`): the Quartz baseline and the baseline columns Orig., VOQC, QUESO, GUOQ, and Quartz without preprocessing.
- On the 248-circuit suite (Table `avg-comparison`): the "QALM (1 minute)" column and the QALM 1-hour total gate reduction.
- Most of the experimental-setup statements.

**Data that is absent:**
- **All four ablation sections**: Greedy Mode, Exploration Depth, and Exploration Strategy (the N_pool and N_branch sweeps).
- **The current QALM column of the 26-circuit table.**
- **Three of the seven baseline configurations on the 248-circuit suite**: GUOQ total-gate mode, QUESO total-gate mode, and VOQC.

The ablation figures and texts are consistent with each other, but no per-circuit data behind them survives here. For the two baselines that do have raw 248-circuit data (GUOQ two-qubit mode and QUESO two-qubit mode), the runs on disk use the same configuration but are not the runs that were plotted. The QALM 1-hour data is in the same position.

The audit also found **several problems in the paper itself** that do not depend on missing data (Section 3). The most important are:
- QALM's 1-hour fidelity, 0.3864, is averaged over the wrong set of circuits.
- The 32 GiB memory limit stated in the paper is not what was used.
- The configuration the paper calls the default is not the one used for the main 1-hour run.
- The claim of less than 1% degradation with a 60× smaller time budget does not hold.

---

## 2. Status by result

| Paper result | Status | Summary |
|---|---|---|
| **Table `avg-comparison`** (8 configs × 3 metrics) | ✅ 7 · ⚠️ ~5 · ✖ 9 · ❌ 1 | ✅ QALM‑1min ×3 and Quartz ×3 are exact; QALM‑1h total 52.34 reproduces.<br>⚠️ Near misses: QALM‑1h 2Q (24.60 vs 24.61), GUOQ‑2Q fidelity, and QUESO‑2Q ×3.<br>✖ GUOQ‑Total, QUESO‑Total and VOQC have no data.<br>❌ QALM‑1h fidelity is wrong (§3.1).<br>❌ GUOQ‑2Q 2Q and total reductions (25.41 and −2.63) don't match the on-disk rerun (25.18 and −3.71). |
| **Fig. `comparison_3x3`** | Partial | ✅ Only the Quartz column matches the raw data per circuit (248/248).<br>⚠️ The QALM, GUOQ‑2Q and QUESO‑2Q series come from runs that are not on disk.<br>✖ Three columns have no data at all.<br>✖ The plotting script is missing. |
| **Fig. `win_rate`** | Partial | ✅ The QUESO‑2Q bars are exact.<br>⚠️ The Quartz bars are within 1 circuit.<br>◐ The GUOQ‑2Q bars are off by 1–5 circuits.<br>✖ The GUOQ‑Total, QUESO‑Total and VOQC bars have no data.<br>✖ The plotting script is missing. |
| **Abstract and intro: 83.9% fidelity win/tie vs GUOQ‑2Q** | ◐ | The figure itself gives 208/248 = 83.9%. The on-disk GUOQ‑2Q rerun with the same configuration gives 212/248 = **85.5%**. |
| **Fig. `gate_scatter`** | Partial | ✅ The left panel (original counts) matches 248/248.<br>⚠️ In the right panel, 168/248 QALM points and only 18/248 GUOQ points match the data on disk.<br>✖ The plotting script is missing. |
| **"Five orders of magnitude" / 248 circuits** | ✅ | Gate counts run from 5 to 768,096 (10^5.19). qft_10 and qft_16 were verified to be exact identities by statevector simulation. |
| **Table `eval_end_to_end`** (26 circuits) | ✅ 130 cells · ◐ 26 · ✖ 26 | ✅ The Orig., VOQC, QUESO, GUOQ and Quartz w/o PP columns and their four geomeans reproduce 26/26.<br>◐ Quartz w/ PP can only be read off the old per-circuit plots, where 25/26 match.<br>✖ **The current QALM column has no data.** |
| **Text: +7.3 / 17.8 / 10.4 / 5.1 / 4.6%** | ◐ | Each is a difference of rounded geomeans. The baseline sides reproduce, but the QALM side has no data. |
| **Fig. `greedy_k` and Greedy Mode text** (32.80, 33.27, 968 s, 89 s, 11×, 5.5 s, 82 s, 724 s, 33.12) | ✖ | The script exists, but its input directory is missing. The figure and the text agree with each other. |
| **Figs. `single_k_gk0` and `single_k_gk2`** (32.83, 31.46, 33.11) | ✖ | Same situation as the greedy ablation. |
| **Figs. `npool` and `nbranch`** (33.03–33.16; 31.33, 32.75, 33.37, 33.30, 33.21) | ✖ | Same situation. The on-disk `*_advk_results` are a different experiment (enqueue on) with a different ranking: N_branch=1 is best there. |
| **Setup** (benchmark, flags, timeouts, fidelity model, CPU) | ✅ mostly | ✅ Matches the scripts and logs.<br>❌ **The 32 GiB memory limit** does not.<br>✖ No GUOQ/QUESO `-opt TOTAL` or VOQC runs exist on the 248-circuit suite. |

---

## 3. Problems in the paper itself

These hold independently of whether the missing data can be recovered.

### 3.1 QALM 1-hour fidelity 0.3864 is averaged over 244 circuits, not 248 (Table `avg-comparison` and §5.2 text)

- **Over 248 circuits, the value is 0.3799.** That comes from the raw CSV `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`. The QALM series in the paper's own figure gives 0.3802.
- **0.3864 is the QALM mean in the VOQC panel of `comparison_3x3`.** That panel has only 244 points. It drops the four largest circuits, `hwb9_119`, `urf4_187`, `urf3_155` and `ground_state_estimation_10`, whose fidelities are about 0. The raw CSV restricted to the same 244 circuits gives 0.3861.
- **The other two QALM cells are 248-circuit means:** 24.61 for 2Q reduction and 52.34 for total reduction. The likely cause is that the generator took the maximum of the per-panel QALM means. Older figure versions (git `a199895`, `8958634`, `f99df52`) drew a separate QALM mean line in each panel. The generator script is missing, so this cannot be confirmed.
- **The ranking does not change**, because 0.3799 is still higher than GUOQ‑Total's 0.3676. The text comparisons "0.3864 vs 0.3428 / 0.3475" need updating.

### 3.2 The VOQC column is a 244-circuit average, and this is not disclosed

The VOQC numbers in Table `avg-comparison` (0.3428, 4.24, 46.37) and its `win_rate` bars cover 244 circuits. The first auto-generated caption, in git `5c636b2`, said "VOQC: 244". That note was removed in `df83847`.

The intro's margins inherit this mismatch. They compare QALM on 248 circuits against VOQC on 244:
- **1 hour:** "+5.97%" becomes 5.80 points on matched circuits.
- **1 minute:** "+4.98%" becomes 4.79 points.

### 3.3 "32 GiB of RAM per optimizer" is not what was used

| Optimizer / run | Memory limit actually used |
|---|---|
| QALM 1 h (248 circuits) | 8 GB: `MEM_LIMIT_GB = 8` in `benchmark_guoq.py@b10f6ec`; the log header says `mem_limit=8GB` |
| QALM 1 minute | 8 GB |
| Ablations (`ablation_greedy_k.py`) | 8 GB |
| GUOQ / QUESO | `docker --memory 8g` (in both copies of `run_docker_tests.py`) |
| Quartz | 8 GB, except that its memory-heavy circuits were **rerun at 32 GB** |

The 8 GB cap affected QALM's results directly. Three QALM circuits (`grover_pigeon3`, `rd73_252`, `misex1_241`) were killed at about 8 GB before the hour was up. `ham15-high` peaked at 7.8 GB. The paper's figure plots exactly these capped values.

`rerun_oom_circuits.py` exists to rerun them at 32 GB, but the CSV rows are unchanged since `b10f6ec`.

### 3.4 The configuration called "default for all other experiments" is not what the main run used

The paper says the default is N_pool=1, N_branch=3, advancing k, with greedy mode for k≤2 (`5-eval.tex` around line 392).

Both on-disk 1-hour, 248-circuit QALM runs that reproduce 52.34 used N_branch=1. The main one, which is also the only one with CX counts:
- was produced by `benchmark_guoq.py@b10f6ec` with `BEST_CONFIG = (1, 1, 3, 1.5, 0, 0, 1, 1, 0)`;
- uses **N_branch=1** and **a fixed k=3**, so k never advances;
- always enqueued intermediate circuits, because the `enqueue_intermediate` option did not exist yet.

Other evidence:
- The other run, `greedy_k_guoq_benchmark_results` at greedy_k=2 (52.34 total, no CX counts), used `ablation_greedy_k.py@a128476` with `N_BRANCH = 1` and intermediates enqueued.
- The 2026-04-13 version of the paper (`3dadb8b`) said "N_branch=1 as the default"; the text was later changed to 3 without rerunning the experiment.
- The project memory note (`project_paper.md`) says the main search uses a fixed `exploration_steps`.

The paper's stated default *is* used by the 1-minute run and by the greedy-k, single-k, N_pool and N_branch ablations.

### 3.5 "Reducing the time budget 60× degrades fidelity and gate reduction by less than 1%" (§5.3)

| Metric | Using the paper's own table | Using raw data |
|---|---|---|
| Fidelity | 0.3864 → 0.3747: −0.0117 absolute, −3.0% relative | 0.3799 → 0.3747: −1.4% relative |
| 2Q reduction | 24.61 → 21.18: **−3.43 points** | about the same |
| Total reduction | 52.34 → 51.35: −0.99 points | about the same |

Only total reduction, read in absolute percentage points, stays under 1. The two runs also differ in configuration (see §3.4).

The conclusion goes further and says QALM surpasses all baselines' one-hour results within the first minute. That is false for CX reduction: GUOQ‑2Q's one-hour 25.41% is higher than QALM‑1min's 21.18%.

### 3.6 The same default configuration produces three different numbers

N_pool=1, N_branch=3, greedy k≤2, no intermediate enqueue appears in three places:

| Where it appears | Value |
|---|---|
| Greedy-mode and single-k figures | 33.27% |
| N_branch=3 point of the N_branch figure | 33.37% |
| N_pool=1 point of the N_pool figure | 33.16% |

The scripts pass byte-identical binary arguments, so these are three independent, non-deterministic runs. They also differed in setup: 32 vs 52 parallel workers, with and without a memory guard.

The 0.21-point spread is **larger than the 0.08-point gap between N_branch=3 and N_branch=4** that the "sweet spot at N_branch=3" claim rests on. The paper reports no estimate of run-to-run variance.

### 3.7 Minor issues

- **"+7.3% vs VOQC" should be 7.4.** All five deltas are differences of *rounded* geomean reductions. Unrounded, the VOQC delta is 34.339 − 26.953 = 7.385. The text says "on average"; the arithmetic means of the per-circuit differences would be 6.94, 18.29, 10.62, 5.02 and 4.46.
- **"Wins or ties on nearly every circuit"** overstates the VOQC case. By the paper's own figure, QALM wins or ties on 82.0% of circuits for total reduction and 86.5% for fidelity.
- **"No single fixed k matches advancing‑k"** holds only in the greedy k≤2 panel. In the no-greedy panel, fixed k=1 (32.83) ends slightly above advancing‑k (32.80).
- **GUOQ‑2Q is called "the strongest baseline"**, but by average fidelity GUOQ‑Total is the strongest (0.3676 vs 0.3475). GUOQ‑2Q is simply the baseline against which QALM has its lowest win rate.
- **The two-qubit-mode baseline values differ from the on-disk reruns.** For GUOQ‑2Q (25.41 / −2.63) and QUESO‑2Q, the on-disk runs use the identical configuration but are different stochastic runs. They give 25.18 / −3.71 and the 85.5% win rate in §2. This shows the plotted runs are not on this machine. It does not show the paper misreported anything.
- **Two circuit-count conventions are mixed in `gate_scatter`.** Its left panel uses GUOQ-parsed original counts, which are lower than the QASM counts for 21 circuits. The reduction percentages elsewhere use QALM-side counts.
- **The `averages_table.tex` header describes "dashed reference lines"** in the 3×N figure. In the current figure the dashed lines are only y=0. Average lines exist only in the older version `a199895`.
- **Figure titles saying "26 circuits"** come from `len(CIRCUIT_LIST)`, not from the number of result files actually loaded. They do not show that all runs finished.
- **The code and the algorithm description differ:**
  - Each branch applies **k+1** transformations before ROQC. One comes from pool generation, then k more from the exploration loop.
  - Candidates above 1.05× the pass-start cost are dropped, which the paper does not mention.
  - The visited set is a `std::set<size_t>` of hashes, not a hash table.
- **"Quartz v0.3.2"** is actually the QALM fork's `./build/test_original` binary. That is v0.3.2 plus local changes; for example, `8c370dd` adds a timeout to the greedy pass.
- **The claim that "preliminary experiments" showed queue pruning (2000 → 1000) doesn't matter** has no experiment on disk. The only related artifact is a single-circuit queue probe on `xor5_254`.

---

## 4. What *is* reproducible, and where

| Result | Raw data | Recomputed |
|---|---|---|
| QALM 1 h, 248 circuits | `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv` (rows with `source=guoq`, excluding `qft_10` and `qft_16`) | fidelity 0.3799 · 2Q 24.60% · total **52.34%** |
| QALM 1 min, 248 circuits | `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv` | **0.3747 · 21.18% · 51.35%** (exact) |
| Quartz w/o preprocessing, 248 circuits | `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv` (and `qasm/`) | **0.3526 · 12.42% · 40.73%**; per-circuit match with the figure is 248/248 |
| GUOQ‑2Q, 248 circuits (same config, *not* the plotted run) | `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/` | 0.3483 · 25.18% · −3.71% |
| QUESO‑2Q, 248 circuits (same config, *not* the plotted run) | same directory, `<circuit>/queso/` (outputs missing for `urf5_280`, `vqe_4`, `vqe_uccsd_n6`, `xor5_254`) | 0.3361 · 13.19% · 27.37%; the win-rate bars (98.8 / 99.6 / 98.8) are exact |
| 26-circuit table, Orig. | `/home/cc/mingkuan/qalm/circuit/nam_circs/<c>.qasm` | 26/26 |
| 26-circuit table, VOQC / QUESO / GUOQ | `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/{voqc,queso,guoq}/results_<c>/` | 26/26 for each; geomeans 26.95 / 23.95 / 29.21 |
| 26-circuit table, Quartz w/o PP | `…/fresh_results_3600/qalm_bench/nam/qalm/<c>/quartz_result.qasm` | 26/26; geomean 16.53 |
| 26-circuit table, Quartz w/ PP (plots only) | `/home/cc/mingkuan/qalm/all_bench_results_hour/long_result_<c>_seconds.png` (inputs in `circuit/nam_rm_circs/`) | 25/26 read from pixels. The `gf2^10_mult` panel actually shows `ham15-low`, because of a slicing bug in `bench_with_qalm_time.py`. |
| 26-circuit table, *old* QALM column (before `b9d28de`, geomean 35.0%) | `…/fresh_results_3600/qalm_bench/nam/qalm/<c>/3600_1_1_2_*_result.qasm` | 26/26 |
| Setup: suite, gate set, flags, timeouts | `/home/cc/guoq/benchmarks/nam_rz` (same files as `awagner/.../all_circuits`), plus the args and JSON files in the docker results | The two-qubit-mode flags match exactly; the 1 h timeouts are confirmed. |

Notes for anyone rerunning:
- **Geomean convention.** The 26-circuit table's geomean reduction is `1 − geomean(opt/orig)`.
- **Fidelity convention.** The 248-circuit table uses an arithmetic mean of `0.99^cx · 0.999^(total−cx)`.
- **One baseline directory is unusable.** `/home/cc/queso_guoq_tests/qalm_benchmarks/docker_results` is a failed run: 224 of 250 runs crashed with `NoSuchFileException`.
- **`compute_metrics.py` is not the paper's generator.** It is in `/home/cc/mingkuan/qalm/guoq_benchmark_results/` and was written in June 2026, after the table.

---

## 5. What is missing, and where it probably is

### A. Ablation and 26-circuit QALM outputs: on the machine that ran the Apr 13–18 experiments

This qalm checkout was fast-forwarded by `git pull` on 2026-06-16. No file on this machine has an mtime between 2026-04-13 and 2026-04-20. `**/*.pkl` is gitignored, and the backup commit `1011727` ("result backup for asplos", Apr 18) omitted the following directories:

| Missing directory (under `/home/cc/mingkuan/qalm/`) | Paper content that depends on it | Script that produces it |
|---|---|---|
| `greedy_k_ablation_no_enqueue_results/` (`pkl/<c>_gk{0..3}_3600.pkl`, `summary.csv`, `qasm/`, and the plot's stdout with greedy-phase times) | Fig. `greedy_k`; every number in the Greedy Mode section; the dashed advancing‑k baselines in both single-k figures | `ablation_greedy_k.py`, `plot_greedy_k.py` |
| `single_k_no_enqueue_results/` (`pkl/<c>_gk{0,2}_fk{1..5}_3600.pkl`) | Figs. `single_k_gk0` and `single_k_gk2`; the numbers 32.83, 31.46 and 33.11 | `ablation_single_k.py`, `plot_single_k.py` |
| `npool_no_enqueue_intermediate_results/` | Fig. `npool_comparison`; the 33.03–33.16% range | `npool_no_enqueue_intermediate.py` |
| `nbranch_no_enqueue_intermediate_results/` | Fig. `nbranch_comparison`. Also **the current QALM column of Table `eval_end_to_end`**: its arithmetic mean, 33.3733%, equals the end of the N_branch=3 curve to 4 decimals | `nbranch_no_enqueue_intermediate.py` |

Two more pieces are missing:
- No code computes the "time to reach 32%" figures (968 s and 89 s). Those values were read off the curves.
- The binary `build/test_greedy_k_ablation` is not built on this machine.

### B. The exact 248-circuit runs that were plotted

The per-circuit data behind `comparison_3x3`, `win_rate` and `gate_scatter` is missing for:
- **QALM 1 h:** the on-disk CSV matches only about 65% of the plotted points.
- **GUOQ‑2Q.**
- **QUESO‑2Q.**

The on-disk runs use the same configuration but are different runs.

### C. Baselines never run on the 248-circuit suite (nowhere on this machine)

- GUOQ `-opt TOTAL --resynth-weight 180`
- QUESO `-opt TOTAL -search BEAM -temp 0 -q 8000 -resynth NONE`
- **VOQC.** `run_voqc` is commented out in every `run_qalm_bench_tools.sh`, and no `docker_results_*` directory has a `voqc/` subdirectory.

These flags appear only in the 26-circuit runs in `all_bench_results_hour/`. Without these runs the following cannot be recomputed:
- 9 cells of Table `avg-comparison`
- 3 columns of `comparison_3x3`
- 9 bars of `win_rate`
- the intro's +5.97% and +4.98%, since VOQC is the strongest total-reduction baseline

### D. Scripts

| Missing script | Notes |
|---|---|
| `gen_averages_table.py` | Named in the header of `figtex/averages_table.tex`. Absent from disk and from every git history. |
| Plotting scripts for `figure/comparison_3x3.pdf`, `figure/win_rate.png` and `figure/gate_scatter.pdf` | The current `comparison_3x3.pdf` was committed by Pengyu Liu in `31fb9f6`. The earlier versions arrived via Overleaf. |

### E. Other

- **Raw Quartz-with-preprocessing outputs for the 26-circuit table:** `pickled_results/combined_*.pkl` from the Dec 2025 run. They were gitignored; only the plots derived from them were committed.
- **`tool_invoke_defs.sh`:** exists only inside the docker image `amxu/guoq:1`.
- **`full_total_q_test.sh`:** referenced in a commented-out Dockerfile line; absent.
- **Outputs of the 32 GB OOM reruns for QALM** (`rerun_oom_circuits.py`).
- **Not searchable without sudo:** `/var/lib/docker`, `/var/lib/containerd`, and `/home/ccadmin`.

---

## 6. Suggested next steps

1. **Recover the four `*_no_enqueue*` result directories in §5A** from the machine where `d579c3f`, `c7bae31` and `b9d28de` were run, and commit their `summary.csv` and `qasm/` (or archive the pkl files).
2. **Locate the baseline runs:** GUOQ‑Total, QUESO‑Total and VOQC on the 248-circuit suite, plus the plotted GUOQ‑2Q and QUESO‑2Q runs. Commit `gen_averages_table.py` and the three plotting scripts next to the data.
3. **Fix the paper issues**, independent of recovery:
   - QALM 1 h fidelity: change it to 0.3799/0.3802, or state the 244-circuit basis.
   - Disclose that the VOQC column covers 244 circuits.
   - Correct the memory limit to 8 GB, or rerun with 32 GB.
   - State the configuration actually used for the main 1 h run, or rerun it with the stated default.
   - Reword the "<1% degradation" claim.
   - Report the run-to-run variance before claiming a sweet spot at N_branch=3.
   - Fix 7.3 → 7.4, or say that the deltas are differences of geomeans.

---

## Appendix: per-claim audit results

These are the verifiers' final verdicts, generated from the workflow output. The full notes for each claim are in `/home/cc/qalm_repro_audit_claims.json`.


### A1. Table avg-comparison and derived headline numbers (248 circuits)

Verdict counts: ❌ CONTRADICTED ×8, ⚠️ MINOR_DISCREPANCY ×5, ✅ REPRODUCED ×8, ✖ NOT_FOUND ×10, ◐ DATA_ONLY_NOT_RECOMPUTABLE ×5, – NOT_APPLICABLE ×1

1. ❌ **CONTRADICTED**: Table tab:avg-comparison, QALM (1h): mean fidelity (0.999^n1q\*0.99^n2q) over the suite  
   *Paper:* 0.3864  
   *Recomputed:* CSV over 248: 0.37987. Paper figure's QALM series over 248: 0.38015 (my own PDF parser), the same in the current figure and the 5c636b2 version. 0.3864 equals only the QALM series in the VOQC panel (244 points, mean 0.38639). The CSV restricted to those 244 circuits gives 0.38610  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`; `git 5c636b2:figure/comparison_3x3.pdf (the figure version committed with the first table)`

2. ⚠️ **MINOR_DISCREPANCY**: Table, QALM (1h): mean 2Q (CX) gate reduction %  
   *Paper:* 24.61  
   *Recomputed:* 24.6007 over 248 from the CSV (rounds to 24.60). The figure's QALM series gives 24.6059  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

3. ✅ **REPRODUCED**: Table, QALM (1h): mean total gate reduction %  
   *Paper:* 52.34  
   *Recomputed:* 52.3359 over 248 from the 1h CSV (rounds to 52.34). Independently, the 248-circuit greedy_k=2 run in greedy_k_guoq_benchmark_results gives 52.3435, which also rounds to 52.34. [Editor's correction: that run was made by ablation_greedy_k_guoq_benchmark.py at a128476 with N_BRANCH=1, before enqueue_intermediate existed. It is not the paper's stated default config, contrary to the verifier's note.] The figure's QALM series gives 52.3402  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/greedy_k_guoq_benchmark_results/summary.csv`

4. ✅ **REPRODUCED**: Table, QALM (1 minute): mean fidelity  
   *Paper:* 0.3747  
   *Recomputed:* 0.37470 over 248 (all 250 rows OK; orig counts match the inputs)  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

5. ✅ **REPRODUCED**: Table, QALM (1 minute): mean 2Q gate reduction %  
   *Paper:* 21.18  
   *Recomputed:* 21.1843 over 248  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

6. ✅ **REPRODUCED**: Table, QALM (1 minute): mean total gate reduction %  
   *Paper:* 51.35  
   *Recomputed:* 51.3484 over 248  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

7. ⚠️ **MINOR_DISCREPANCY**: Table, GUOQ (2Q Gate mode, -opt TWO_Q --resynth-weight 180): mean fidelity  
   *Paper:* 0.3475  
   *Recomputed:* 0.34826 over 248 (my own count of the output QASMs; matches the JSON best_\* wherever present). The figure's GUOQ-2Q series gives 0.34748  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/latest_sol_none_1_<circuit>.qasm`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/guoq_log_none_1.out`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits/`

8. ❌ **CONTRADICTED**: Table, GUOQ (2Q Gate): mean 2Q gate reduction %  
   *Paper:* 25.41  
   *Recomputed:* 25.1829 over 248 from the only on-disk GUOQ TWO_Q run on the suite. The figure series gives 25.412  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/`

9. ❌ **CONTRADICTED**: Table, GUOQ (2Q Gate): mean total gate reduction %  
   *Paper:* -2.63  
   *Recomputed:* -3.7089 over 248 on disk. The figure series gives -2.6316  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/`

10. ✖ **NOT_FOUND**: Table, GUOQ (Total Gate mode, -opt TOTAL): mean fidelity  
   *Paper:* 0.3676  
   *Recomputed:* No raw data. The figure-extracted mean over 248 is 0.36756, which matches the table  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

11. ✖ **NOT_FOUND**: Table, GUOQ (Total Gate): mean 2Q gate reduction %  
   *Paper:* 18.83  
   *Recomputed:* No raw data. The figure-extracted mean is 18.834  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

12. ✖ **NOT_FOUND**: Table, GUOQ (Total Gate): mean total gate reduction %  
   *Paper:* 43.89  
   *Recomputed:* No raw data. The figure-extracted mean is 43.891  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

13. ⚠️ **MINOR_DISCREPANCY**: Table, QUESO (2Q Gate mode: -opt TWO_Q -search BEAM -temp 0 -q 8000 -resynth NONE): mean fidelity  
   *Paper:* 0.3362  
   *Recomputed:* 0.33606 over 248, with the 4 missing circuits set to unoptimized (0.33685 over the 244 present). The figure gives 0.33621  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/latest_sol_none_none_<circuit>.qasm`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/guoq_log_none_none.out`

14. ⚠️ **MINOR_DISCREPANCY**: Table, QUESO (2Q Gate): mean 2Q gate reduction %  
   *Paper:* 13.19  
   *Recomputed:* 13.1927 over 248 with the 4 missing set to 0% (13.4089 over the 244 present). The figure gives 13.1865  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/`

15. ⚠️ **MINOR_DISCREPANCY**: Table, QUESO (2Q Gate): mean total gate reduction %  
   *Paper:* 27.40  
   *Recomputed:* 27.3651 over 248 with the 4 missing set to 0%. The figure gives 27.3959  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/`

16. ✖ **NOT_FOUND**: Table, QUESO (Total Gate mode, -opt TOTAL): mean fidelity  
   *Paper:* 0.3554  
   *Recomputed:* No raw data. The figure-extracted mean is 0.35541  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

17. ✖ **NOT_FOUND**: Table, QUESO (Total Gate): mean 2Q gate reduction %  
   *Paper:* 13.37  
   *Recomputed:* No raw data. The figure-extracted mean is 13.368  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

18. ✖ **NOT_FOUND**: Table, QUESO (Total Gate): mean total gate reduction %  
   *Paper:* 39.72  
   *Recomputed:* No raw data. The figure-extracted mean is 39.716  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

19. ✖ **NOT_FOUND**: Table, VOQC: mean fidelity  
   *Paper:* 0.3428  
   *Recomputed:* No raw data. The figure-extracted mean over 244 points is 0.34284  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

20. ✖ **NOT_FOUND**: Table, VOQC: mean 2Q gate reduction %  
   *Paper:* 4.24  
   *Recomputed:* No raw data. The figure-extracted mean over 244 is 4.2429  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

21. ✖ **NOT_FOUND**: Table, VOQC: mean total gate reduction %  
   *Paper:* 46.37  
   *Recomputed:* No raw data. The figure-extracted mean over 244 is 46.3749  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf (plotted points only)`

22. ✅ **REPRODUCED**: Table, Quartz (no preprocessing): mean fidelity  
   *Paper:* 0.3526  
   *Recomputed:* 0.35258 over 248  
   *Evidence:* `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/qasm/`; `/home/cc/mingkuan/qalm/original_benchmark_results/qasm_greedy/`

23. ✅ **REPRODUCED**: Table, Quartz: mean 2Q gate reduction %  
   *Paper:* 12.42  
   *Recomputed:* 12.4179 over 248  
   *Evidence:* `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

24. ✅ **REPRODUCED**: Table, Quartz: mean total gate reduction %  
   *Paper:* 40.73  
   *Recomputed:* 40.7268 over 248  
   *Evidence:* `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

25. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Intro: with one hour, QALM achieves 5.97% more gate count reduction than the strongest baseline  
   *Paper:* 5.97% (= 52.34 - 46.37, VOQC)  
   *Recomputed:* 52.34-46.37=5.97 (arithmetic correct). The QALM side reproduces; the VOQC side exists only as plotted points. On the matched 244 circuits the gap is 52.17-46.37=5.80  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

26. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Intro: with only one minute, QALM still achieves 4.98% more reduction than the strongest baseline  
   *Paper:* 4.98% (= 51.35 - 46.37)  
   *Recomputed:* 51.35-46.37=4.98 (arithmetic correct). QALM-1min reproduces (51.3484); VOQC has no raw data. On the 244-circuit VOQC subset, QALM-1min is 51.164, giving a 4.79 gap  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

27. ❌ **CONTRADICTED**: Eval Optimization Efficiency: reducing time 60x degrades QALM's average fidelity and gate reduction by less than 1%  
   *Paper:* &lt;1% degradation (table: 0.3864-&gt;0.3747, 24.61-&gt;21.18, 52.34-&gt;51.35)  
   *Recomputed:* From the table: fidelity -0.0117 (3.0% relative), 2Q reduction -3.43 pp, total reduction -0.99 pp. On a consistent 248-circuit basis from raw data: fidelity 0.3799-&gt;0.3747 (-0.0052, -1.4% relative), 2Q -3.42 pp, total -0.99 pp  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

28. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Eval Optimization Efficiency: the 1-minute QALM still surpasses the one-hour fidelity and total gate reduction of every baseline  
   *Paper:* 0.3747 &gt; max baseline fidelity 0.3676; 51.35 &gt; max baseline total 46.37  
   *Recomputed:* Against raw data, QALM-1min (0.3747/51.35) beats Quartz (0.3526/40.73), QUESO-2Q (0.3361/27.37) and GUOQ-2Q (0.3483/-3.71). GUOQ-Total, QUESO-Total and VOQC exist only as figure means, which also support the claim  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/`

29. ❌ **CONTRADICTED**: Main Results: QALM achieves nearly identical CX reduction to GUOQ-2Q (24.61% vs 25.41%)  
   *Paper:* 24.61 vs 25.41  
   *Recomputed:* QALM 24.60 (CSV) vs GUOQ-2Q 25.18 (on-disk same-config run)  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/`

30. ❌ **CONTRADICTED**: Main Results: QALM dramatically better total gate reduction than GUOQ-2Q (52.34% vs -2.63%); GUOQ resynthesis yields negative total reduction on many circuits  
   *Paper:* 52.34 vs -2.63  
   *Recomputed:* QALM 52.34 (reproduced) vs GUOQ-2Q -3.71 on disk. 115 of 248 negative on disk; 124 in the figure  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/`

31. ✖ **NOT_FOUND**: Main Results: vs VOQC similar total reduction (52.34% vs 46.37%) but much greater CX reduction (24.61% vs 4.24%)  
   *Paper:* 52.34 vs 46.37; 24.61 vs 4.24  
   *Recomputed:* QALM side: 52.34 / 24.60 (248). VOQC side: figure only, 46.37 / 4.24 over 244. QALM on the same 244 circuits: 52.17 / 24.46  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

32. ❌ **CONTRADICTED**: Main Results: QALM average fidelity 0.3864 outperforms VOQC (0.3428) and GUOQ-2Q (0.3475)  
   *Paper:* 0.3864 / 0.3428 / 0.3475  
   *Recomputed:* QALM 0.3799 over 248 (CSV) or 0.3802 (figure). 0.3864 is QALM's mean over VOQC's 244 circuits. GUOQ-2Q on disk: 0.3483. VOQC: figure only, 0.3428 over 244  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

33. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Main Results: vs QUESO (both modes), Quartz and GUOQ total-mode, QALM is better on all three metrics  
   *Paper:* QALM 0.3864/24.61/52.34 vs QUESO-2Q 0.3362/13.19/27.40, QUESO-Total 0.3554/13.37/39.72, Quartz 0.3526/12.42/40.73, GUOQ-Total 0.3676/18.83/43.89  
   *Recomputed:* Raw data: QALM (0.3799/24.60/52.34) beats Quartz (0.3526/12.42/40.73) and QUESO-2Q (0.3361/13.19/27.37) on all three. QUESO-Total and GUOQ-Total exist only as figure means, all below QALM  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/`

34. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Main Results (added): QALM achieves higher fidelity than all four baselines on the majority of circuits  
   *Paper:* majority of circuits  
   *Recomputed:* Figure pairs, QALM strictly greater per panel: GUOQ2Q 66.1%, GUOQTotal 62.9%, QUESO2Q 76.2%, QUESOTotal 72.6%, VOQC 69.3%, Quartz 71.4%. Raw: QALM-1h CSV beats Quartz on 72.2%, on-disk GUOQ2Q on 68.5% and on-disk QUESO2Q on 75.8%  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/`

35. ✅ **REPRODUCED**: Setup: the suite has 248 circuits = GUOQ's 250 circuits minus qft_10 and qft_16 (identity circuits)  
   *Paper:* 248  
   *Recomputed:* 250 input QASMs; 250-2 = 248. All three QALM/Quartz CSVs have 250 guoq rows whose orig counts match the inputs  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits/`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/qasm/guoq_qft_10_3600.qasm`

36. – **NOT_APPLICABLE**: Setup (added): QALM configuration for the main comparison is the stated default (N_pool=1, N_branch=3, greedy k&lt;=2)  
   *Paper:* N_pool=1, N_branch=3, greedy k&lt;=2  
   *Recomputed:* The 1h CSV used for the table's QALM cells runs test_qalm with (1,1,3,...,greedy=1), i.e. N_branch=1, k=3. The 1-minute run uses (1,3,2): the default. A 1h greedy_k=2 run (greedy_k_guoq_benchmark_results) records only total counts (52.34 total) [Editor's correction: it used N_BRANCH=1 (a128476), so it is not the stated default either]  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/greedy_k_guoq_benchmark_results/summary.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

37. ❌ **CONTRADICTED**: Table caption / setup: every column is a per-metric average across the same benchmark suite (248 circuits); each optimizer given one CPU core, 32 GiB RAM and one hour per circuit  
   *Paper:* all columns over the 248-circuit suite; 32 GiB; 1 h  
   *Recomputed:* Figure point counts are 248 in every panel except VOQC (244). The VOQC column and the QALM fidelity cell are averaged over 244. Memory limits: - QALM 1h: 8 GB. grover_pigeon3, rd73_252 and misex1_241 were killed at 8 GB before 3600 s, and the figure plots exactly those 8 GB values. - QALM 1min: 8 GB, but not binding. - Quartz: 8 GB, with 26 memory-heavy circuits rerun at 32 GB. - On-disk GUOQ/QUESO: docker --cpus=1 --memory 8g.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/run_3600s.log`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv` …


### A2. Figures comparison_3x3, win_rate, gate_scatter; 83.9% claim

Verdict counts: ◐ DATA_ONLY_NOT_RECOMPUTABLE ×7, ⚠️ MINOR_DISCREPANCY ×4, ✅ REPRODUCED ×6, ✖ NOT_FOUND ×4, ❌ CONTRADICTED ×1, – NOT_APPLICABLE ×3

1. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM matches or exceeds the fidelity of the strongest baseline (GUOQ, 2Q-gate mode) on 83.9% of the 248 circuits  
   *Paper:* 83.9% (=208/248)  
   *Recomputed:* On-disk raw data gives 212/248 = 85.5%. That is 80.6% strict wins plus 4.8% ties with float compare, or 82.7% + 2.8% with exact log compare. The 8 failed GUOQ runs count as the original circuit; their latest_sol qasm files equal the input circuits. Other treatments: skipping failures gives 204/240 = 85.0%; counting failures as QALM losses gives 204/248 = 82.3%. No treatment gives 208. Data extracted from comparison_3x3.pdf gives exactly 208/248 = 83.9%.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

2. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Win-rate bar: QALM &gt;= GUOQ (2Q Gate) on 2Q-gate (CX) reduction (described in the text as 'on par')  
   *Paper:* 54.4% (solid part measured from pixels: 45.5%)  
   *Recomputed:* Raw: 130/248 = 52.4% (win 45.6%, tie 6.9%). Skipping failures gives 50.8%. Figure-embedded: 135/248 = 54.4% (win 45.6%).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`

3. ⚠️ **MINOR_DISCREPANCY**: Win-rate bar: QALM &gt;= GUOQ (2Q Gate) on total gate reduction  
   *Paper:* 88.3% (solid part about 85.5%)  
   *Recomputed:* Raw: 218/248 = 87.9% (win 85.1%, tie 2.8%). Skipping failures gives 87.5%. Figure-embedded: 219/248 = 88.3% (win 85.5%).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`

4. ✅ **REPRODUCED**: Win-rate bars: QALM &gt;= QUESO (2Q Gate) on fidelity / CX reduction / total reduction  
   *Paper:* 98.8% / 99.6% / 98.8% (solid about 89.1 / 78.2 / 90.7)  
   *Recomputed:* Raw: fidelity 245/248 = 98.8%, CX 247/248 = 99.6%, total 245/248 = 98.8%. Strict wins: 87.9 / 77.4 / 89.9 with float compare. The 78 QUESO failures (74 with null best size, 4 with no output: urf5_280, vqe_4, vqe_uccsd_n6, xor5_254) count as the original circuit. Skipping them gives 98.2 / 99.4 / 98.2 (n=170).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/results_none_none.json`

5. ⚠️ **MINOR_DISCREPANCY**: Win-rate bars: QALM &gt;= Quartz (no preprocessing) on fidelity / CX reduction / total reduction  
   *Paper:* 97.6% / 99.2% / 96.8% (solid about 84.3 / 76.2 / 85.5)  
   *Recomputed:* Raw: fidelity 242/248 = 97.6% (exact match), CX 246/248 = 99.2% (exact match), total 241/248 = 97.2% (paper 96.8% = 240). Strict wins: 83.9 / 75.8 / 85.5. Figure-embedded: fidelity 243 at PDF resolution, CX 246, total 240.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

6. ✖ **NOT_FOUND**: Win-rate bars: QALM &gt;= GUOQ (Total Gate) on fidelity / CX reduction / total reduction  
   *Paper:* 95.2% / 95.6% / 96.0% (solid about 76.2 / 65.3 / 79.8)  
   *Recomputed:* No raw data for the 248-circuit suite. Figure-embedded data gives 236/248 = 95.2%, 237/248 = 95.6% and 238/248 = 96.0%, consistent with the PNG.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

7. ✖ **NOT_FOUND**: Win-rate bars: QALM &gt;= QUESO (Total Gate) on fidelity / CX reduction / total reduction  
   *Paper:* 98.4% / 99.6% / 98.0% (solid about 85.9 / 78.2 / 87.4)  
   *Recomputed:* No raw data for the suite. Figure-embedded: 244/248 = 98.4%, 247/248 = 99.6%, 243/248 = 98.0%, consistent with the PNG.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso`

8. ✖ **NOT_FOUND**: Win-rate bars: QALM &gt;= VOQC on fidelity / CX reduction / total reduction (VOQC column uses 244 circuits)  
   *Paper:* 86.5% / 100.0% / 82.0% (solid about 81.6 / 84.0 / 76.6), n=244  
   *Recomputed:* No raw VOQC data for the suite. Figure-embedded (244 points): fidelity 213/244 = 87.3% at 1e-9 tolerance (PNG says 86.5% = 211), CX 244/244 = 100%, total 200/244 = 82.0%.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/voqc`; `/home/cc/fresh_results_2_8_26/rq1/nam/voqc`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

9. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Win/tie split in win_rate.png (solid = QALM strictly better, hatched = tie)  
   *Paper:* Solid heights I measured from pixels: Fidelity 79.0/76.2/89.1/85.9/81.6/84.3; CX 45.5/65.3/78.2/78.2/84.0/76.2; Total 85.5/79.8/90.7/87.4/76.6/85.5 (GUOQ2Q, GUOQTot, QUESO2Q, QUESOTot, VOQC, Quartz)  
   *Recomputed:* Strict wins from figure-embedded data: CX 45.6/65.3/78.2/78.2/84.0/76.2 and Total 85.5/79.8/90.7/87.5/76.6/85.5, matching the PNG. Fidelity strict wins cannot be resolved from the PDF (only 61.7-76.2% resolvable). Raw on-disk strict wins (float compare): GUOQ2Q fid/CX/total 80.6/45.6/85.1, QUESO2Q 87.9/77.4/89.9, Quartz 83.9/75.8/85.5.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/win_rate.png`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

10. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Against QUESO (both modes), VOQC, Quartz and GUOQ (Total), QALM wins or ties on nearly every circuit across all three metrics  
   *Paper:* 'nearly every circuit' (PNG: 95.2-100% except VOQC 86.5% fidelity and 82.0% total)  
   *Recomputed:* Raw data supports it for QUESO 2Q (98.8/99.6/98.8) and Quartz (97.6/99.2/97.2). There is no raw data for QUESO Total, GUOQ Total or VOQC.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

11. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM achieves higher fidelity than all four baselines on the majority of circuits  
   *Paper:* majority (&gt;50%) strict fidelity wins vs every baseline  
   *Recomputed:* Raw strict fidelity wins (float / exact-log compare): GUOQ2Q 80.6/82.7%, QUESO2Q 87.9/89.9%, Quartz 83.9/85.9%. From the figure, every configuration has at least 61.7% strict wins even at a 1e-4 tolerance.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

12. ✅ **REPRODUCED**: GUOQ's resynthesis (2Q mode) yields negative total gate reduction on many circuits, whereas QALM avoids this entirely  
   *Paper:* many circuits negative for GUOQ 2Q; none for QALM  
   *Recomputed:* Raw: the GUOQ 2Q total is above the original on 115 of 240 successful runs, with both QALM-side and GUOQ-side originals. QALM is never above the original (0/248; minimum reduction 0.0%; no CX increase either). Figure: GUOQ2Q is negative on 124/248 points (threshold -1e-3%; minimum -224.1%); QALM minimum is 0.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

13. ⚠️ **MINOR_DISCREPANCY**: comparison_3x3 QALM series (one-hour QALM per-circuit fidelity / CX% / total%)  
   *Paper:* figure means (248): fidelity 0.3802, CX 24.6059%, total 52.3402% (paper prints 24.61 and 52.34)  
   *Recomputed:* results_3600s.csv (source=guoq, excluding qft_10 and qft_16, n=248; all rows OK): fidelity 0.3799, CX 24.6007% (rounds to 24.60, paper 24.61), total 52.3359% (52.34). compute_metrics.py's 'guoq (no qft)' line prints the same values. Per circuit, only 139/215 identified totals and 64/115 identified CX values equal the plotted QALM values.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_advk_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_weighted_cost_results/results_3600s.csv`

14. ❌ **CONTRADICTED**: Average QALM fidelity 0.3864 (Table avg-comparison and eval text) over the benchmark suite  
   *Paper:* 0.3864  
   *Recomputed:* 248-circuit arithmetic mean: 0.3799 from raw data, 0.3802 from the figure. 244-circuit subset without hwb9_119, urf4_187, urf3_155 and ground_state_estimation_10: 0.3861 from raw data, exactly 0.3864 from the figure's VOQC panel.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`

15. ✅ **REPRODUCED**: comparison_3x3 Quartz column (Quartz without preprocessing) per-circuit values; Table means 0.3526 / 12.42 / 40.73  
   *Paper:* means fidelity 0.3526, CX 12.42%, total 40.73%  
   *Recomputed:* The plotted blue Quartz values match the csv-derived values 248/248 in each of the fidelity, CX% and total% rows. Means: 0.3526 / 12.4179 / 40.7268.  
   *Evidence:* `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/qasm`; `/home/cc/mingkuan/qalm/original_benchmark_results/qasm_greedy`

16. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: comparison_3x3 GUOQ (2Q Gate) column per-circuit values; Table means 0.3475 / 25.41 / -2.63  
   *Paper:* means fidelity 0.3475, CX 25.41%, total -2.63%  
   *Recomputed:* The on-disk TWO_Q run, with failures counted as the original circuit, gives fidelity 0.3483, CX 25.18% and total -3.73%. Per circuit, excluding failures, only 3/207 identified totals and 10/113 identified CX values equal the plotted values; the multiset match is 25/248 for totals.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`

17. ⚠️ **MINOR_DISCREPANCY**: comparison_3x3 QUESO (2Q Gate) column per-circuit values; Table means 0.3362 / 13.19 / 27.40  
   *Paper:* means fidelity 0.3362, CX 13.19%, total 27.40%  
   *Recomputed:* On-disk run (78 failures counted as 0% reduction): fidelity 0.3361, CX 13.19%, total 27.34%. Per circuit, failures included: 118/215 totals and 63/115 CX match. Excluding failures: 57/153 and 51/102.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/queso/results_none_none.json`

18. ✖ **NOT_FOUND**: comparison_3x3 GUOQ (Total Gate), QUESO (Total Gate) and VOQC columns; Table means 0.3676/18.83/43.89, 0.3554/13.37/39.72, 0.3428/4.24/46.37  
   *Paper:* as listed (VOQC n=244)  
   *Recomputed:* Cannot be recomputed from raw data. My own extraction of the figure-embedded means reproduces the table: GUOQTot 0.3676/18.8343/43.8913, QUESOTot 0.3554/13.3683/39.7159, VOQC (244) 0.3428/4.2429/46.3749.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam`

19. ✅ **REPRODUCED**: Per-circuit comparison: circuits sorted by QALM metric; suite size 248 (VOQC 244)  
   *Paper:* 248 circuits (250 minus identity circuits qft_10 and qft_16), sorted by QALM value  
   *Recomputed:* The PDF has 36 marker series (/M0-/M35): 30 with 248 points and 6 (VOQC) with 244. All 18 QALM series are monotonically non-decreasing. nam_rz and all_circuits contain the same 250 names, minus qft_10 and qft_16 = 248.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

20. ✅ **REPRODUCED**: The 248 circuits span five orders of magnitude in gate count  
   *Paper:* five orders of magnitude  
   *Recomputed:* Total gates range from 5 (graycode6_47) to 768096 (urf4_187), a factor of 10^5.19. 2Q gates range from 5 to 224028.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/graycode6_47.qasm`; `/home/cc/guoq/benchmarks/nam_rz/urf4_187.qasm`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

21. ✅ **REPRODUCED**: gate_scatter left panel: original (2Q, total) gate counts of the 248 circuits  
   *Paper:* 248 points  
   *Recomputed:* 248/248 points equal the GUOQ-reported (original_2q, original_total), which is also present in the failed-run jsons. 227/248 equal the qasm/csv counts.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`

22. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: gate_scatter right panel: optimized (2Q, total) counts for QALM and GUOQ (2Q mode)  
   *Paper:* 248 QALM + 248 GUOQ points  
   *Recomputed:* QALM points: 168/248 equal results_3600s.csv (cx, total). GUOQ points: only 18/248 equal the on-disk GUOQ TWO_Q run (19 if failures use GUOQ-parsed originals).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/<circuit>/guoq/results_none_1.json`

23. – **NOT_APPLICABLE**: GUOQ in 2Q-gate mode is the strongest baseline in our evaluation  
   *Paper:* GUOQ (2Q) = strongest  
   *Recomputed:* By the paper's own table, GUOQ 2Q has the best CX reduction (25.41) but not the best fidelity (0.3475 vs GUOQ Total 0.3676) or total reduction (-2.63).  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figtex/averages_table.tex`

24. – **NOT_APPLICABLE**: (added) VOQC per-circuit panels and average cover the benchmark suite  
   *Paper:* implicitly 248 circuits  
   *Recomputed:* The VOQC panels have 244 points. The 4 missing QALM values identify the 4 largest circuits exactly: hwb9_119, urf4_187, urf3_155 and ground_state_estimation_10.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/comparison_3x3.pdf`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

25. – **NOT_APPLICABLE**: (added) gate_scatter dashed reference line marks total = 2 x 2q  
   *Paper:* dashed line total = 2 x 2q  
   *Recomputed:* In log-log space the dashed path runs from (0, 0.301) to (5.526, 5.827), a constant offset of log10(2) = 0.3010.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/gate_scatter.pdf`


### B. Table eval_end_to_end (26 Quartz circuits)

Verdict counts: ✅ REPRODUCED ×9, ◐ DATA_ONLY_NOT_RECOMPUTABLE ×9, ✖ NOT_FOUND ×2, – NOT_APPLICABLE ×3

1. ✅ **REPRODUCED**: Orig. column: original total gate counts of the 26 Quartz benchmark circuits (Nam gate set)  
   *Paper:* adder_8 900, barenco_tof_3 58, ..., vbe_adder_3 150 (26 values)  
   *Recomputed:* 26/26 identical. Gate types over all 26: cx 4612, rz 5075, h 691, x 103.  
   *Evidence:* `/home/cc/mingkuan/qalm/circuit/nam_circs/<c>.qasm`

2. ✅ **REPRODUCED**: VOQC column: total gate count after VOQC on each of the 26 circuits  
   *Paper:* 596, 40, 72, 104, 264, 158, 280, 186, 287, 401, 543, 706, 879, 1065, 51, 92, 184, 416, 269, 678, 152, 35, 55, 75, 175, 89  
   *Recomputed:* 26/26 identical  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/voqc/results_<c>/optimized_none_none_<c>.qasm`

3. ✅ **REPRODUCED**: QUESO column: total gate count after QUESO, 1 hour  
   *Paper:* 645, 38, 68, 98, 248, 148, 392, 195, 306, 426, 603, 815, 996, 1225, 33, 100, 198, 452, 331, 791, 176, 35, 55, 75, 175, 83  
   *Recomputed:* 26/26 identical  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso/results_<c>/latest_sol_none_none_<c>.qasm`; `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso/results_<c>/args_none_none.txt`

4. ✅ **REPRODUCED**: GUOQ column: total gate count after GUOQ, 1 hour  
   *Paper:* 578, 39, 72, 102, 262, 145, 346, 169, 271, 376, 500, 673, 824, 1046, 26, 101, 198, 395, 289, 657, 180, 35, 55, 75, 180, 80  
   *Recomputed:* 26/26 identical  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq/results_<c>/latest_sol_none_1_<c>.qasm`; `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq/results_<c>/args_none_1.txt`

5. ✅ **REPRODUCED**: Quartz w/o Preprocess column: total gate count after Quartz (no preprocessing), 1 hour  
   *Paper:* 897, 38, 90, 134, 420, 146, 400, 199, 316, 462, 648, 880, 1079, 1327, 24, 98, 215, 488, 418, 878, 180, 35, 55, 76, 177, 114  
   *Recomputed:* 26/26 identical  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/qalm/<c>/quartz_result.qasm`

6. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Quartz w/ Preprocess column: total gate count after Quartz on preprocessed circuits, 1 hour (preprocessing time excluded)  
   *Paper:* 621, 38, 72, 104, 264, 148, 279, 173, 276, 383, 524, 695, 861, 1040, 24, 93, 202, 401, 288, 646, 152, 35, 55, 75, 175, 89  
   *Recomputed:* No raw qasm/pkl output exists. Reading the red 'Quartz - with preprocessing' curve end from the per-circuit plots, calibrated against 3-8 known curves per plot, matches the table on 18/26 circuits to within 0.3 gates: adder_8 620.99, barenco_tof_4 72.0, barenco_tof_5 103.99, barenco_tof_10 264.1, csla_mux_3 147.92, csum_mux_9 279.03, gf2^4 173.0, gf2^5 276.03, gf2^6 382.99, gf2^7 523.94, gf2^8 694.9, gf2^9 860.96, mod_mult_55 93.01, mod_red_21 201.97, qcla_adder_10 400.98, qcla_com_7 287.88, qcla_mod_7 645.7, vbe_adder_3 89.01. On 7 circuits the red curve ends hidden under later-drawn curves at the same y-value, which is consistent with the table (barenco_tof_3, mod5_4, rc_adder_6, tof_3, tof_4, tof_5, tof_10; checked visually for tof_10). gf2^10_mult (1040) has no support.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/long_result_<c>_seconds.png (26 per-circuit time-series plots, committed in qalm 3afaf8e)`; `/home/cc/mingkuan/qalm/all_bench_results_hour/avg_reduction.png`; `/home/cc/mingkuan/qalm/circuit/nam_rm_circs/<c>.qasm (inputs)`

7. ✖ **NOT_FOUND**: QALM column: total gate count after QALM, 1 hour (values replaced 2026-04-15)  
   *Paper:* 512, 38, 62, 94, 236, 144, 280, 167, 265, 370, 500, 660, 805, 972, 24, 86, 175, 385, 252, 622, 152, 35, 51, 71, 165, 71  
   *Recomputed:* No per-circuit source on this machine. Best match among 646 qasm groups is 13/26 (comparison_results 600 s configs). CSVs, including all total_tXX time snapshots: at most 15/26 (guoq_benchmark_results nam total_t900). pickled_results finals: at most 13/26. The column's arithmetic-mean reduction (33.3733%) equals the end of the N_branch=3 curve in the paper's N_branch_arith_mean_3600s_plain.pdf (33.3733, read from the PDF vector path) to 4 decimals.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf (aggregate only)`

8. ✅ **REPRODUCED**: Geo. Mean Reduction, VOQC = 27.0%  
   *Paper:* 27.0%  
   *Recomputed:* 26.9533%  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/voqc/results_<c>/optimized_none_none_<c>.qasm`; `/home/cc/mingkuan/qalm/circuit/nam_circs/<c>.qasm`

9. ✅ **REPRODUCED**: Geo. Mean Reduction, QUESO = 23.9%  
   *Paper:* 23.9%  
   *Recomputed:* 23.949962%  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso/results_<c>/latest_sol_none_none_<c>.qasm`

10. ✅ **REPRODUCED**: Geo. Mean Reduction, GUOQ = 29.2%  
   *Paper:* 29.2%  
   *Recomputed:* 29.2096%  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq/results_<c>/latest_sol_none_1_<c>.qasm`

11. ✅ **REPRODUCED**: Geo. Mean Reduction, Quartz w/o Preprocess = 16.5%  
   *Paper:* 16.5%  
   *Recomputed:* 16.5277%  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/qalm/<c>/quartz_result.qasm`

12. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Geo. Mean Reduction, Quartz w/ Preprocess = 29.7%  
   *Paper:* 29.7%  
   *Recomputed:* 29.7063% from the printed column. 25 of the 26 inputs are supported by the plots (18 read directly, 7 consistent but hidden). The gf2^10_mult value (1040) is unsupported. The geomean stays at 29.7 for any gf2^10 value in about 1022-1059.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/long_result_<c>_seconds.png`

13. ✖ **NOT_FOUND**: Geo. Mean Reduction, QALM = 34.3%  
   *Paper:* 34.3%  
   *Recomputed:* 34.3386% from the printed column (arithmetic mean 33.3733, which equals the end of the N_branch=3 figure curve). Other 1 h QALM runs on disk give similar geomeans, e.g. nbranch_comparison_3600s_advk nbranch_3 34.07, npool_1 34.27, Dec-2025 run 34.97.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf (aggregate, arithmetic mean only)`

14. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: Caption: the best result for each circuit is in bold  
   *Paper:* bolding per row  
   *Recomputed:* Bolding equals the row minimum, ties included, in all 26 rows  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figtex/eval_end_to_end.tex`

15. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM achieves the lowest or equally lowest gate count on all 26 circuits except one, where Quartz with preprocessing wins (csum_mux_9); 'matches or outperforms it on all but one circuit'  
   *Paper:* only csum_mux_9 (QALM 280 vs Quartz w/ PP 279)  
   *Recomputed:* In the printed table, csum_mux_9 is the only row where any baseline is below QALM (Quartz w/ PP 279 &lt; 280)  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/long_result_csum_mux_9_seconds.png`; `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/{voqc,queso,guoq,qalm}`

16. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM reduces gate count by an additional 7.3% on average vs VOQC  
   *Paper:* 7.3%  
   *Recomputed:* 34.3 - 27.0 = 7.3 from rounded geomeans. Unrounded 34.339 - 26.953 = 7.385, which would print as 7.4. The arithmetic mean of per-circuit (VOQC - QALM)/orig is 6.94.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/voqc`

17. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM outperforms Quartz (without preprocessing) by 17.8%  
   *Paper:* 17.8%  
   *Recomputed:* 34.3 - 16.5 = 17.8; unrounded 17.811; arithmetic mean of per-circuit differences 18.29  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/qalm/<c>/quartz_result.qasm`

18. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM outperforms QUESO by 10.4%  
   *Paper:* 10.4%  
   *Recomputed:* 34.3 - 23.9 = 10.4; unrounded 10.389; arithmetic mean of per-circuit differences 10.62  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso`

19. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM outperforms GUOQ by 5.1%  
   *Paper:* 5.1%  
   *Recomputed:* 34.3 - 29.2 = 5.1; unrounded 5.129; arithmetic mean of per-circuit differences 5.02  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq`

20. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: QALM outperforms Quartz (with preprocessing) by 4.6%  
   *Paper:* 4.6%  
   *Recomputed:* 34.3 - 29.7 = 4.6; unrounded 4.632; arithmetic mean of per-circuit differences 4.46 (from the printed table)  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/long_result_<c>_seconds.png`

21. – **NOT_APPLICABLE**: Wording: the differences are 'on average' and are 'absolute percentages (relative to the original gate count)'  
   *Paper:* 7.3/17.8/10.4/5.1/4.6 described as average absolute percentage points  
   *Recomputed:* All five equal (rounded QALM geomean reduction) minus (rounded baseline geomean reduction). Arithmetic averages of per-circuit absolute differences would be 6.94/18.29/10.62/5.02/4.46.  

22. – **NOT_APPLICABLE**: Setup: each tool gets a 1-hour limit; Quartz preprocessing runs separately and is not counted in the hour  
   *Paper:* 1-hour limit; Quartz preprocessing time excluded  
   *Recomputed:* Limits are 1 h in config. The GUOQ start timestamps chain at exactly 2h00m18-20s per circuit in 14 cases (GUOQ 1 h + QUESO 1 h). For 7 circuits the gap to the next circuit is shorter: adder_8 65 min, barenco_tof_3 70, barenco_tof_4 79.5, gf2^6 90, gf2^7 69, gf2^9 72, gf2^10 73.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_{600,900,1800,2700,3600}/qalm_bench/nam/guoq/results_<c>/resynth_log_none_1.txt (start timestamps)`; `/home/cc/mingkuan/qalm/circuit/nam_rm_circs/`; `/home/cc/.claude/history.jsonl`

23. – **NOT_APPLICABLE**: The table uses the 26 benchmark circuits used by Quartz  
   *Paper:* 26 circuits  
   *Recomputed:* 26 rows, identical to qalm_circuits_full.txt; nam_circs additionally holds hwb6 and ham15-low  
   *Evidence:* `/home/cc/mingkuan/qalm/circuit/nam_circs/`; `/home/cc/mingkuan/qalm/qalm_experiments/qalm_circuits_full.txt`


### C. Greedy Mode (Fig. greedy_k)

Verdict counts: – NOT_APPLICABLE ×3, ✖ NOT_FOUND ×11, ◐ DATA_ONLY_NOT_RECOMPUTABLE ×1

1. – **NOT_APPLICABLE**: The greedy-mode ablation uses the same 26 circuits as Table eval_end_to_end, with a 1-hour budget per run  
   *Paper:* 26 Quartz/Nam circuits, 3600 s  
   *Recomputed:* Checked against config, independently. CIRCUIT_LIST in ablation_greedy_k.py has 26 circuit/nam_circs entries, the same set as the 26 \texttt rows of eval_end_to_end.tex. The gate-line count of each nam_circs qasm equals the table's Orig. column for all 26. original_cost in greedy_k_ablation_results/summary.csv also equals Orig. for all 26. TIMEOUT defaults to 3600. In test_greedy_k_ablation.cpp (at 05e8667, line 150), `start` is taken before the greedy passes, so greedy time counts toward the 1-hour budget.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figtex/eval_end_to_end.tex`; `/home/cc/mingkuan/qalm/circuit/nam_circs/`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`

2. ✖ **NOT_FOUND**: Final average gate-count reduction without greedy mode (greedy_k=0) is 32.80%  
   *Paper:* 32.80%  
   *Recomputed:* No raw data. My own parse of the paper figure's vector paths (axes box 54.02/41.96/439.18x219.16; xlim -180..3780; ylim 30..34) puts the gk=0 curve end at 32.8031%. That figure is an aggregate artifact, not raw data. The only on-disk 26-circuit greedy data is the superseded run in greedy_k_ablation_results/summary.csv, which gives gk=0 = 33.338%.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`

3. ✖ **NOT_FOUND**: Final average gate-count reduction with greedy mode for k&lt;=2 (greedy_k=2) is 33.27%  
   *Paper:* 33.27%  
   *Recomputed:* From the paper figure path, gk=2 ends at 33.2664%. The superseded on-disk run gives gk=2 = 33.440%. The QALM column of Table eval_end_to_end averages 33.373%, a different run, so 33.27% cannot be derived from the table.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`; `/home/cc/seach_and_rule_based_opt/figtex/eval_end_to_end.tex`

4. ✖ **NOT_FOUND**: Final reductions with and without greedy mode differ by less than 0.5% on average  
   *Paper:* &lt;0.5% (33.27 - 32.80 = 0.47 pp)  
   *Recomputed:* From the figure paths: 33.2664 - 32.8031 = 0.463 pp, which is also the maximum spread across all four curves (finals 32.80, 32.83, 33.27, 33.12). The superseded on-disk run has a spread of 0.17 pp.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`

5. ✖ **NOT_FOUND**: Without greedy mode, QALM takes 968 s to reach an average gate-count reduction of 32%  
   *Paper:* 968 s  
   *Recomputed:* From the paper figure's gk=0 path, the linear-interpolated 32% crossing is at 965.5 s. That lies in the 963.2-975.3 s cell of the plot's 300-point grid, so it is consistent with 968 s. The gk=0 advancing-k baseline in single_k_gk0.pdf is point-identical and gives the same 965.5 s. The superseded run's figure gives 453.2 s; the dbc9014 draft said 448.9 s.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`

6. ✖ **NOT_FOUND**: Greedy mode for k=1 alone reaches the 32% threshold in 89 s  
   *Paper:* 89 s  
   *Recomputed:* From the figure's gk=1 path: 88.4 s, which is in grid cell 84.3-96.3 s and consistent with 89 s. The superseded run gives 242.4 s; the draft said 242.5 s.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`

7. ✖ **NOT_FOUND**: k=1 greedy mode gives an 11x speedup to reach 32% (968 s vs 89 s)  
   *Paper:* 11x  
   *Recomputed:* 968/89 = 10.88, which rounds to 11. From the figure paths, 965.5/88.4 = 10.92. The superseded run gives about 1.9x.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`

8. ✖ **NOT_FOUND**: With greedy mode only for k=1, the greedy phase takes 5.5 s on average  
   *Paper:* 5.5 s  
   *Recomputed:* Cannot be computed. No on-disk dataset has per-circuit greedy-phase timestamps for the 26 circuits. summary.csv has only qalm_k_iters and max_qalm_k. The nam_termination\*, k_comparison, higher_params and single_k results have no greedy timing either.  
   *Evidence:* `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`

9. ✖ **NOT_FOUND**: With greedy mode for k&lt;=2, the greedy phase takes 82 s on average  
   *Paper:* 82 s  
   *Recomputed:* Cannot be computed (see the k=1 claim). The dbc9014 draft's value, from the old run, was 80.4 s.  

10. ✖ **NOT_FOUND**: Extending greedy mode to k&lt;=3 pushes the greedy phase to 724 s on average  
   *Paper:* 724 s  
   *Recomputed:* Cannot be computed. Under the script's rule (unfinished = 3600 s), 724 s implies the other 24 circuits average (724\*26 - 7200)/24 = 484 s. The dbc9014 draft's value, from the old run, was 704.5 s.  

11. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: With k&lt;=3, two of the 26 circuits (gf2^9_mult, gf2^10_mult) fail to finish the greedy phase within the one-hour budget  
   *Paper:* 2 circuits: gf2^9_mult, gf2^10_mult  
   *Recomputed:* Superseded 26-circuit run, greedy_k=3: exactly 2 of 26 rows have qalm_k_iters=0 and an empty max_qalm_k, namely gf2^9_mult (final 805) and gf2^10_mult (final 972). All other circuits have max_qalm_k=4. In the separate old 248-circuit GUOQ-suite run, the same two are again the only members of the 26-circuit Quartz set with qalm_k_iters=0 at gk=3 (gf2^9_mult 812, gf2^10_mult 993). This corroborates the claim, but not for the paper's run.  
   *Evidence:* `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`; `/home/cc/mingkuan/qalm/greedy_k_guoq_benchmark_results/summary.csv`

12. ✖ **NOT_FOUND**: Final reduction at k&lt;=3 is 33.12%, slightly below the k&lt;=2 setting (33.27%)  
   *Paper:* 33.12% (vs 33.27%)  
   *Recomputed:* From the figure's gk=3 path: 33.1180%, below gk=2 at 33.2664%. This matches the text. The superseded on-disk run gives gk=3 = 33.274% and gk=2 = 33.440%.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv`

13. ✖ **NOT_FOUND**: (Added) The k&lt;=2 setting gives the best final reduction of the four settings ('strikes a balance'); figure curves not quantified in the text: gk=1 final value, and gk=2/gk=3 time-to-32%  
   *Paper:* Plotted only: gk=2 is the highest final curve; gk=1 ends at ~32.8%; gk=2 and gk=3 cross 32% within the first few hundred seconds  
   *Recomputed:* From the figure paths: - Final values: gk0 32.8031, gk1 32.8253, gk2 33.2664 (highest), gk3 33.1180. - Time-to-32%: gk2 203.6 s, gk3 122.6 s. - Values at t=1000 s: 32.09, 32.65, 32.69, 32.73. The earlier text (bdeda50) said gk2 needs '199 seconds to 32%', which agrees with 203.6 s on the 12 s grid. That sentence was removed from the final text.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`

14. – **NOT_APPLICABLE**: (Added) Experimental setup applies to these ablation runs: each optimizer gets one CPU core, 32 GiB of RAM and a one-hour limit  
   *Paper:* 1 core, 32 GiB RAM, 1 h per run  
   *Recomputed:* Config check of ablation_greedy_k.py at HEAD (identical in the relevant part to d579c3f/05e8667, which drove the paper run): - MEM_LIMIT_GB = 8, enforced with RLIMIT_AS and an RSS monitor that SIGKILLs over 8 GB. - N_WORKERS = 52 parallel processes. - A memory guard kills the largest child when MemAvailable drops below 16 GB, and --rerun-killed reruns those. - TIMEOUT = 3600 (wall kill at 3720 s). This does not match the stated 32 GiB per run. The 1-hour limit does match.  

15. – **NOT_APPLICABLE**: Caption: each curve runs QALM with greedy mode (cost-reducing only) up to depth k&lt;=greedy_k, then unrestricted search beyond  
   *Paper:* methodology statement  
   *Recomputed:* The code is consistent with the caption. At 05e8667, test_greedy_k_ablation.cpp runs, in order: - greedy_optimize_with_roqc (k=1) - greedy_optimize_with_local_search (k=2) - greedy_optimize_with_deeper_local_search (k=3) It stops at greedy_k, then runs advancing optimize_qalm starting at k=greedy_k+1, logging [QALM_K_START]. ablation_greedy_k.py passes fixed_k=0 and enqueue_intermediate=0, with N_pool=1 and N_branch=3.  


### D. Exploration Depth (Figs. single_k_gk0 / gk2)

Verdict counts: ✖ NOT_FOUND ×8, – NOT_APPLICABLE ×3

1. ✖ **NOT_FOUND**: Without greedy mode (greedy_k=0), fixed k=1 reaches 32.83% final average gate-count reduction (26 circuits, 1 h)  
   *Paper:* 32.83%  
   *Recomputed:* Paper figure vector endpoint (my own parser) = 32.830. The older on-disk run (single_k_results) gives 33.149. The 26-circuit subset of single_k_guoq_benchmark_results gives 24.628.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf (vector data only, a derived artifact)`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv (older enqueue_intermediate=1 run, not the paper's run)`; `/home/cc/mingkuan/qalm/single_k_guoq_benchmark_results/summary.csv (248-circuit GUOQ nam_rz run; includes the same 26 names but is a different experiment)`

2. ✖ **NOT_FOUND**: Without greedy mode, fixed k=5 reaches 31.46% final average reduction  
   *Paper:* 31.46%  
   *Recomputed:* Paper figure vector endpoint = 31.463. Older on-disk enqueue run = 33.024. GUOQ-run 26-subset = 24.441.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv (older enqueue run)`

3. ✖ **NOT_FOUND**: Without greedy mode, k=2, 3, 4 final average reductions (figure only, not printed in the text)  
   *Paper:* not printed; figure endpoints k=2 32.652, k=3 32.528, k=4 32.100  
   *Recomputed:* Paper figure vectors: k=2 32.652, k=3 32.528, k=4 32.100. Older on-disk run: k=2 33.152, k=3 33.183, k=4 33.111.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv (older enqueue run)`

4. ✖ **NOT_FOUND**: Without greedy mode, the final average reduction decreases monotonically with k  
   *Paper:* monotonic, 32.83 down to 31.46  
   *Recomputed:* Paper figure: 32.830 &gt; 32.652 &gt; 32.528 &gt; 32.100 &gt; 31.463 (monotonic). Older on-disk enqueue run: 33.149, 33.152, 33.183, 33.111, 33.024 (not monotonic; k=3 best).  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv (older enqueue run)`

5. ✖ **NOT_FOUND**: With greedy mode for k&lt;=2 then a fixed k, k=3 is the best fixed depth at 33.11% (all five values)  
   *Paper:* k=3 best at 33.11%  
   *Recomputed:* Paper figure vectors: k=1 32.726, k=2 33.039, k=3 33.112 (best), k=4 32.935, k=5 32.767. Older on-disk run: 33.169, 33.229, 33.381, 33.382 (k=4 best), 33.319. GUOQ-run 26-subset: 18.04, 15.08, 15.13, 14.97, 14.32 (including fallbacks for missing runs).  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv (older enqueue run)`; `/home/cc/mingkuan/qalm/single_k_guoq_benchmark_results/summary.csv (different experiment; 8-12 of the 26 gk2 runs missing)`

6. ✖ **NOT_FOUND**: No single fixed k matches the advancing-k baseline (33.27%)  
   *Paper:* 33.27%  
   *Recomputed:* Paper dashed gk2 endpoint = 33.266 (rounds to 33.27), which is above the best fixed k (33.112). In the gk0 panel the dashed advancing-k curve ends at 32.803, BELOW fixed k=1 (32.830). Older on-disk run: advancing gk2 33.440 and gk0 33.338, each above all fixed k in its group.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv (older enqueue run)`

7. ✖ **NOT_FOUND**: Cross-check: the 33.27% advancing-k baseline equals the greedy-section k&lt;=2 run (and 32.80% equals greedy_k=0)  
   *Paper:* 33.27% (both sections); 32.80% for greedy_k=0  
   *Recomputed:* greedy_k_reduction.pdf endpoints: gk0 32.803, gk1 32.825, gk2 33.266, gk3 33.118. The single_k dashed polylines are point-for-point identical to the greedy_k_reduction gk0 curve (116 points) and gk2 curve (105 points). Older on-disk greedy_k_ablation_results gives 33.338 / 33.394 / 33.440 / 33.274.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk0.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf`; `/home/cc/mingkuan/qalm/greedy_k_ablation_results/summary.csv (older enqueue run)`

8. ✖ **NOT_FOUND**: Cross-check: the default configuration (greedy k&lt;=2, N_pool=1, N_branch=3, no enqueue) appears as 33.27% here, 33.37% at N_branch=3, and 33.16% at N_pool=1  
   *Paper:* 33.27% vs 33.37% vs 33.16%  
   *Recomputed:* From figure vectors (independently re-extracted): N_branch=1..5 gives 31.333 / 32.751 / 33.373 / 33.295 / 33.207; N_pool=1..5 gives 33.162 / 33.158 / 33.042 / 33.032 / 33.116; single-k/greedy gk2 baseline 33.266.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf`

9. – **NOT_APPLICABLE**: Averaged over 26 benchmark circuits  
   *Paper:* 26 circuits  
   *Recomputed:* CIRCUIT_LIST has 26 nam_circs. The older summary.csv covers exactly these 26 for each of the 10 (gk, fk) configs, and greedy_k_ablation_results/summary.csv has the same 26 names.  
   *Evidence:* `/home/cc/mingkuan/qalm/circuit/nam_circs/`; `/home/cc/mingkuan/qalm/single_k_results/summary.csv`

10. – **NOT_APPLICABLE**: 1-hour budget, single fixed k in {1..5}, no depth advancement; greedy_k groups 0 and 2  
   *Paper:* 3600 s, k in {1,...,5}  
   *Recomputed:* ablation_single_k.py: TIMEOUT=3600, FIXED_K_VALUES=[1..5], GREEDY_K_GROUPS=[0,2], N_POOL=1, N_BRANCH=3, argv[13]=fixed_k, argv[14]='0' (no enqueue), N_WORKERS=52, MIN_FREE_GB=16. test_greedy_k_ablation.cpp:194-212: fixed_k&gt;0 makes a single optimize_qalm call at depth k with the full timeout. All paper curves end at x=3600.  
   *Evidence:* `/home/cc/mingkuan/qalm/src/test/test_greedy_k_ablation.cpp`

11. – **NOT_APPLICABLE**: Explanation: a deeper fixed k costs more per iteration, so fewer iterations fit in the budget  
   *Paper:* qualitative  
   *Recomputed:* n/a  


### E. Exploration Strategy (Figs. N_pool / N_branch)

Verdict counts: ✖ NOT_FOUND ×13, – NOT_APPLICABLE ×2, ❌ CONTRADICTED ×1

1. ✖ **NOT_FOUND**: Across N_pool in {1..5} (N_branch=3 fixed, 26 circuits, 1h, greedy k&lt;=2), the final average reduction ranges from 33.03% to 33.16% (a spread of 0.13%)  
   *Paper:* 33.03%..33.16%, spread 0.13%  
   *Recomputed:* From the paper figure's vector paths only (my own extractor): N_pool=1..5 end at 33.1618 / 33.1576 / 33.0415 / 33.0325 / 33.1159, so min 33.03, max 33.16, spread 0.129. This matches the text. The on-disk advk qasm recomputes to 33.3772 / 33.2800 / 33.3327 / 33.2611 / 33.2975 (range 33.26-33.38), which does not match.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`; `/home/cc/mingkuan/qalm/npool_comparison_3600s_advk_results/qasm/ (different experiment: N_branch=1, enqueue on)`

2. ✖ **NOT_FOUND**: N_pool=1 (N_branch=3) final average reduction (shown in the figure; the text gives only the range)  
   *Paper:* ~33.16% (upper end of range)  
   *Recomputed:* 33.1618 (figure vector data only). The on-disk advk npool_1 run (N_branch=1, enqueue on) gives 33.3772.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`

3. ✖ **NOT_FOUND**: N_pool=2 final average reduction (figure)  
   *Paper:* within 33.03-33.16% (not printed individually)  
   *Recomputed:* 33.1576 (figure vector data). The advk on-disk npool_2 (different config) gives 33.2800.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`

4. ✖ **NOT_FOUND**: N_pool=3 final average reduction (figure)  
   *Paper:* within 33.03-33.16%  
   *Recomputed:* 33.0415 (figure vector data). The advk on-disk npool_3 gives 33.3327.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`

5. ✖ **NOT_FOUND**: N_pool=4 final average reduction (figure; the minimum of the range)  
   *Paper:* 33.03%  
   *Recomputed:* 33.0325 (figure vector data). The advk on-disk npool_4 gives 33.2611.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`

6. ✖ **NOT_FOUND**: N_pool=5 final average reduction (figure)  
   *Paper:* within 33.03-33.16%  
   *Recomputed:* 33.1159 (figure vector data). The advk on-disk npool_5 gives 33.2975.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`

7. ✖ **NOT_FOUND**: N_branch=1 (N_pool=1): average reduction drops to 31.33%  
   *Paper:* 31.33%  
   *Recomputed:* The figure's vector data gives 31.3334, matching the text. The curve is flat from about 1170 s onward. Recomputed from the on-disk advk qasm (26 circuits, enqueue on): 33.4299, clearly different.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`; `/home/cc/mingkuan/qalm/nbranch_comparison_3600s_advk_results/qasm/*_nbranch_1.qasm (different experiment)`

8. ✖ **NOT_FOUND**: N_branch=2: 32.75%  
   *Paper:* 32.75%  
   *Recomputed:* 32.7507 (figure vector data; matches the text). The advk on-disk nbranch_2 gives 33.1260.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`

9. ✖ **NOT_FOUND**: N_branch=3: peaks at 33.37%  
   *Paper:* 33.37%  
   *Recomputed:* 33.3733 (figure vector data; matches the text). The advk on-disk nbranch_3 gives 33.1867.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`

10. ✖ **NOT_FOUND**: N_branch=4: 33.30%  
   *Paper:* 33.30%  
   *Recomputed:* 33.2953 (figure vector data; rounds to 33.30, matches). The advk on-disk nbranch_4 gives 33.3887.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`

11. ✖ **NOT_FOUND**: N_branch=5: 33.21%  
   *Paper:* 33.21%  
   *Recomputed:* 33.2067 (figure vector data; matches). The advk on-disk nbranch_5 gives 33.3213.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`

12. ✖ **NOT_FOUND**: N_branch exhibits a sweet spot at 3 (it beats 4 and 5); the N_pool spread is 'well within run-to-run variance'  
   *Paper:* 3 &gt; 4 &gt; 5 (33.37 / 33.30 / 33.21); N_pool spread 0.13 within variance  
   *Recomputed:* The identical no-enqueue config NP1/NB3/gk2 appears three times in the paper's own figures: 33.1618, 33.3733 and 33.2664 (spread 0.21pp). Gaps between N_branch settings: 3 vs 4 = 0.078pp, 3 vs 5 = 0.167pp. In the on-disk advk data, the two identical NP1/NB1 runs give 33.3772 vs 33.4299 (0.053pp); 9 of 28 circuits differ by 1-5 gates (barenco_tof_10 240/238, qcla_mod_7 615/620, etc.).  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/mingkuan/qalm/npool_comparison_3600s_advk_results/qasm` …

13. – **NOT_APPLICABLE**: At N_branch=1 every expansion keeps a single successor, so the search queue holds at most one circuit at any time  
   *Paper:* queue size &lt;= 1  
   *Recomputed:* Code check: true only with enqueue_intermediate=0 (and N_pool=1)  

14. – **NOT_APPLICABLE**: Experimental setup: vary each parameter over {1..5} with the other at its default (N_pool=1, N_branch=3); greedy mode k&lt;=2; 26 circuits; one-hour timeout; greedy passes do not use N_pool/N_branch  
   *Paper:* N_pool sweep at N_branch=3; N_branch sweep at N_pool=1; gk&lt;=2; 26 circuits; 3600s  
   *Recomputed:* The no-enqueue scripts match the stated setup. The on-disk advk script does not: its N_pool sweep uses N_branch=1.  

15. ❌ **CONTRADICTED**: N_pool=1 and N_branch=3 are used as the default configuration for all other experiments  
   *Paper:* N_pool=1, N_branch=3 everywhere else  
   *Recomputed:* Recomputed from raw counts, source=guoq, 248 circuits excluding qft_10/qft_16: - guoq_benchmark_results/results_3600s.csv: total 52.3359%, CX 24.6007%. The paper's main 1-hour QALM column is 52.34 / 24.61, so total matches and 2Q is off by 0.01pp. - Other runs on disk: advk N_branch=1 52.3547 / 24.5668; weighted-cost N_branch=3 52.2363 / 24.0754. - The 1-minute column (prev_qalm, N_branch=3, no-enqueue) gives 51.3484 / 21.1843, matching 51.35 / 21.18.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_advk_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_weighted_cost_results/results_3600s.csv` …

16. ✖ **NOT_FOUND**: Consistency: the default config (N_pool=1, N_branch=3, gk&lt;=2 advancing, no-enqueue) appears as N_pool=1 in fig:npool_comparison, as N_branch=3 in fig:nbranch_comparison and as the advancing-k k&lt;=2 value (33.27%) in the greedy/single-k sections; these should agree  
   *Paper:* 33.27% (greedy/single-k sections) vs 33.37% (N_branch=3) vs ~33.16% (N_pool=1)  
   *Recomputed:* From the paper's own figure vectors: - N_pool=1 curve: 33.1618 - N_branch=3 curve: 33.3733 - greedy_k=2 curve in figure/greedy_k_reduction.pdf: 33.2664; the same curve is reused as the advancing baseline in figure/single_k_gk2.pdf (33.2664) This is a 0.21pp spread for one config. The on-disk greedy_k_ablation_results gk2 (older N_branch=1, enqueue-on run) gives 33.4397, which matches none of them.  
   *Evidence:* `/home/cc/seach_and_rule_based_opt/figure/N_pool_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/N_branch_arith_mean_3600s_plain.pdf`; `/home/cc/seach_and_rule_based_opt/figure/greedy_k_reduction.pdf`; `/home/cc/seach_and_rule_based_opt/figure/single_k_gk2.pdf` …


### F. Experimental setup and remaining statements

Verdict counts: ✅ REPRODUCED ×11, – NOT_APPLICABLE ×16, ✖ NOT_FOUND ×4, ❌ CONTRADICTED ×3, ⚠️ MINOR_DISCREPANCY ×1, ◐ DATA_ONLY_NOT_RECOMPUTABLE ×2

1. ✅ **REPRODUCED**: The benchmark suite has 248 circuits compiled by GUOQ (the guoq-benchmarks suite).  
   *Paper:* 248 circuits  
   *Recomputed:* I diffed the md5 lists myself: the 250 qasm files in /home/cc/guoq/benchmarks/nam_rz and all_circuits are identical. The QALM 1h, QALM 1-min and Quartz CSVs each have exactly the same 250 'guoq' circuit names, which become 248 after removing qft_10 and qft_16.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits/`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv` …

2. ✅ **REPRODUCED**: The suite originally had 250 circuits. qft_10 and qft_16 were removed because they are identity circuits.  
   *Paper:* 250 -&gt; 248; qft_10, qft_16 = identity  
   *Recomputed:* My own numpy statevector simulator (16 qubits) gives |&lt;psi|U|psi&gt;| = 1.0000000000 with phase of order 1e-18 on 3 random states for both qft_10 (280 gates) and qft_16 (640 gates), so U = I. The control qft_5 gives about 0.26.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits/qft_10.qasm`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/all_circuits/qft_16.qasm`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

3. ✅ **REPRODUCED**: The suite includes QAOA, VQE, TFIM, Heisenberg, XY, adders, GF multipliers, modular arithmetic, sqrt, Grover, Shor, QFT and QPE circuits.  
   *Paper:* categories listed  
   *Recomputed:* Name matches among the 248: qaoa 11, vqe 6, TFIM 13, heisenberg 11, XY 2, adder 7+, gf2^n_mult 7, mod\* (several), sqrt8_260, grover_pigeon3, shor_15_7, qft 7, qpe 2. Every listed category is present.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`

4. ✅ **REPRODUCED**: Gate count is measured in the Nam gate set (X, Rz, H, CX).  
   *Paper:* Nam gate set  
   *Recomputed:* Gate histogram over the 250 inputs: rz 3,034,095; cx 1,484,354; h 539,417; x 130. No other gate types. orig_total and orig_cx in the QALM 1h CSV match a direct qasm count for 250/250 circuits.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`

5. ✅ **REPRODUCED**: The 248 circuits span five orders of magnitude in gate count.  
   *Paper:* five orders of magnitude  
   *Recomputed:* Minimum is graycode6_47 with 5 gates; maximum is urf4_187 with 768,096. The ratio is 1.54e5, or 10^5.19.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`

6. – **NOT_APPLICABLE**: Quartz v0.3.2 is the baseline used.  
   *Paper:* Quartz v0.3.2  
   *Recomputed:* Commit 9bb64f4 'Update to Quartz v0.3.2 (#4)' (2026-01-29) in /home/cc/mingkuan/qalm is an ancestor of b10f6ec (git merge-base --is-ancestor). There is no v0.3.2 tag and no separate Quartz install.  
   *Evidence:* `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

7. ✅ **REPRODUCED**: QUESO in two-qubit mode runs with: -g NAM -opt TWO_Q --rules-dir /home/queso_rules/ -search BEAM -temp 0 -q 8000 -resynth NONE, consistent with the authors' recommendations.  
   *Paper:* exact flag string  
   *Recomputed:* Ignoring -job, -out and --verbosity, all 246/246 QUESO args files in docker_results_05-13-26_03-05-44 equal the paper's flag string exactly.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/queso/args_none_none.txt`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/queso/results_none_none.json`

8. ✖ **NOT_FOUND**: QUESO in total-gate mode runs with: -g NAM -opt TOTAL --rules-dir /home/queso_rules/ -search BEAM -temp 0 -q 8000 -resynth NONE, on the 248-circuit suite.  
   *Paper:* exact flag string  
   *Recomputed:* None of the 4,756 args_\*.txt files on the filesystem has TOTAL mode for a 248/250-suite circuit. TOTAL appears only in 26 Nam-circuit runs per time budget (all_bench_results_hour/fresh_results_{30,60,180,600,900,1800,2700,3600}).  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/queso/`

9. ✅ **REPRODUCED**: GUOQ in two-qubit mode runs with: -g NAM -opt TWO_Q --rules-dir /home/queso_rules/ --resynth-weight 180.  
   *Paper:* exact flag string  
   *Recomputed:* 250/250 args files in docker_results_05-13-26_03-05-44 match exactly, and the JSON guoq_cli_args field agrees.  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/guoq/args_none_1.txt`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/guoq/results_none_1.json`

10. ✖ **NOT_FOUND**: GUOQ in total-gate mode runs with: -g NAM -opt TOTAL --rules-dir /home/queso_rules/ --resynth-weight 180, on the 248-circuit suite.  
   *Paper:* exact flag string  
   *Recomputed:* The exact string appears only in 26 Nam-circuit runs per time budget. No GUOQ TOTAL run exists for the 248-suite.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq/`

11. ✅ **REPRODUCED**: GUOQ performs circuit resynthesis only when optimizing for two-qubit gates.  
   *Paper:* resynth only in TWO_Q  
   *Recomputed:* In Optimizer.java (lines 2444-2451), the default resynthesis is BQSKIT for TWO_Q and FIDELITY and NONE for TOTAL. In the logs, 245/250 GUOQ TWO_Q resynth logs contain 'POST /bqskit', against 0/26 for GUOQ TOTAL Nam runs and 0/246 for QUESO TWO_Q (which uses -resynth NONE).  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/guoq/resynth_log_none_1.txt`; `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_3600/qalm_bench/nam/guoq/*/resynth_log_none_1.txt`

12. – **NOT_APPLICABLE**: GUOQ and QUESO run inside the Docker environments their authors provide.  
   *Paper:* authors' docker  
   *Recomputed:* The Dockerfile starts FROM amxu/guoq:1 and copies the benchmark files and run scripts in. It also copies updated_resynth.py over /home/resynth.py; md5 e8690ab1... differs from /home/cc/guoq/resynth.py (3832618...).  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/Dockerfile`; `/home/cc/queso_guoq_tests/qalm_benchmarks/Dockerfile`

13. – **NOT_APPLICABLE**: All experiments ran on servers with AMD EPYC 7763 processors.  
   *Paper:* AMD EPYC 7763  
   *Recomputed:* lscpu on this host (hostname qalm): AMD EPYC 7763 64-Core Processor, 2 sockets x 64 cores x 2 threads (256 logical CPUs), 250 GiB RAM.  

14. – **NOT_APPLICABLE**: Each optimizer gets one CPU core.  
   *Paper:* 1 CPU core  
   *Recomputed:* GUOQ/QUESO: docker --cpus=1 (a CPU-time quota, not pinning) with Pool(25) concurrent containers. QALM/Quartz: no taskset, affinity or cgroup limit, with Pool(32) workers (16 for the 1-min run).  

15. ❌ **CONTRADICTED**: Each optimizer gets 32 GiB of RAM.  
   *Paper:* 32 GiB per run  
   *Recomputed:* QALM 1h (b10f6ec) used MEM_LIMIT_GB=8 (RLIMIT_AS plus an RSS monitor); the log header says 'mem_limit=8GB'. QALM 1-min also used 8 GB. GUOQ/QUESO used docker --memory 8g with a -Xmx20g JVM (both copies of run_docker_tests.py). Only Quartz got 32 GB reruns.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/run_3600s.log`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/0410184_169/guoq/results_none_1.json` …

16. ✅ **REPRODUCED**: Each run has a one-hour time limit (one minute for the 'QALM (1 minute)' setting).  
   *Paper:* 3600 s / 60 s  
   *Recomputed:* QALM 1h: TIMEOUT=3600, and opt_total/opt_cx equal the t3600 snapshot in 304/306 rows. The other 2 are qft_10/qft_16, whose cx snapshot is empty because cx=0. GUOQ/QUESO TWO_Q: JSON timeout=3600 with 'timeout 3600 java' in 250/250 and 246/246 runs; max seconds_elapsed 3589.1 and 3592.3. Quartz: TIMEOUT=3600. QALM 1-min: TIMEOUT=60.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/run_3600s.log`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/*/{guoq,queso}/results_none_*.json`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

17. ✅ **REPRODUCED**: Quartz ran without its preprocessing pass on the 248-circuit suite.  
   *Paper:* no preprocessing  
   *Recomputed:* test_original.cpp contains no toffoli/ccz preprocessing, only Quartz greedy_optimize followed by optimize_original. The script feeds ~/guoq/benchmarks/nam_rz, whose 250 files contain only rz/cx/h/x (no ccz). The resulting CSV recomputes to 40.73 / 12.42 / 0.3526 over 248, matching the paper's Quartz column.  
   *Evidence:* `/home/cc/guoq/benchmarks/nam_rz/`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`

18. ✖ **NOT_FOUND**: VOQC is run as a baseline on the 248-circuit suite.  
   *Paper:* VOQC baseline on 248 circuits (46.37% / 4.24% / 0.3428)  
   *Recomputed:* No VOQC output exists for any 248/250-suite circuit. VOQC data exists only for Nam circuits (21-26 per dir).  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_*/qalm_bench/nam/voqc`; `/home/cc/fresh_results_2_8_26/rq1/nam/voqc`; `/home/cc/mingkuan/qalm/2q_fresh_results/qalm_bench/nam/voqc`; `/home/cc/old_awagner/ram_test_results/rq1/nam/voqc`

19. ✅ **REPRODUCED**: Fidelity is modeled as F = f1q^n1q \* f2q^n2q with f1q = 0.999 and f2q = 0.99.  
   *Paper:* f1q=0.999, f2q=0.99  
   *Recomputed:* With mean over 248 of 0.99^cx \* 0.999^(total-cx), my own script gives QALM 1-min 0.3747 (paper 0.3747) and Quartz 0.3526 (paper 0.3526). QUESO-2Q from latest_sol qasm, with 4 missing circuits set to their original counts, gives 0.3361 (paper 0.3362).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/`

20. ❌ **CONTRADICTED**: The main search loop uses a constant schedule N_pool=1, N_branch=3, with k advancing from 1 in passes (greedy for k&lt;=2), and this is the default for all other experiments.  
   *Paper:* N_pool=1, N_branch=3, advancing k, used for all other experiments  
   *Recomputed:* Only guoq_benchmark_results reproduces the paper's 1h total reduction (52.34). It was produced at b10f6ec by ./build/test_qalm with BEST_CONFIG=(1,1,3,1.5,0,0,1,1,0): N_pool=1, N_branch=1, fixed exploration_steps=3 (exploration_increase=0, so k never advances), greedy k=1,2 preprocessing, and intermediate circuits always enqueued (candidates.push in both pool-gen and the explore loop at b10f6ec). The other 1h 248-suite runs do not match: advk (N_branch=1, advancing, enqueue=1) gives 52.35, and weighted_cost (N_branch=3, advancing, no-enq, but 1Q+10\*2Q cost) gives 52.24.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_advk_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_weighted_cost_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

21. ⚠️ **MINOR_DISCREPANCY**: Each branch explores k steps and then runs one exploitation (ROQC) step (Alg. 1 / Alg. 2).  
   *Paper:* k transformations then Exploit  
   *Recomputed:* In Graph::optimize_qalm (HEAD line 3118 onward; also at b10f6ec), each branch gets 1 random transformation during pool generation (while found_circuits.size() &lt; exploration_pool_size) and then exploration_steps more (for i &lt; exploration_steps) before ROQC, so k+1 in total. test_greedy_k_ablation starts the advancing loop at exploration_steps = greedy_k+1 = 3, so the pass labeled k=3 applies 4 transformations.  

22. – **NOT_APPLICABLE**: ROQC implements all VOQC routines: Not propagation, Hadamard reduction, single-qubit cancellation, two-qubit cancellation and rotation merging.  
   *Paper:* 5 VOQC routines  
   *Recomputed:* opt_routine_0 through 4 implement the five routines; the default order is [0,1,3,2,3,1,2,4,3,2,3].  

23. – **NOT_APPLICABLE**: Rotation merging is extended with PhasePoly's tagging technique, so Rz gates sharing a tag can merge.  
   *Paper:* tag-based rotation merging  
   *Recomputed:* merge_rotations assigns a new virtual qubit (tag) after each Hadamard.  

24. – **NOT_APPLICABLE**: The search uses the (5,3)-complete ECC set from Quartz.  
   *Paper:* (5,3)-complete ECC set  
   *Recomputed:* ECCSET='eccset/Nam_5_3_complete_ECC_set.json' in the main 1h, 1-min and ablation scripts.  
   *Evidence:* `/home/cc/mingkuan/qalm/eccset/Nam_5_3_complete_ECC_set.json`

25. – **NOT_APPLICABLE**: Exploration biases transformation choice toward recently modified regions.  
   *Paper:* local transformations  
   *Recomputed:* only_do_local_transformations=1 in every run config; after each step, all_nodes is set to xfer-&gt;dstOps mapOps.  

26. – **NOT_APPLICABLE**: When the priority queue holds more than 2,000 circuits, it is pruned to the top 1,000.  
   *Paper:* &gt;2000 -&gt; 1000  
   *Recomputed:* kMaxNumCandidates=2000 and kShrinkToNumCandidates=1000 (HEAD line 3200; b10f6ec line 2954), with shrink_candidates() called after a post-ROQC push. The probe log shows the queue reaching 2000 on xor5_254.  
   *Evidence:* `/home/cc/mingkuan/qalm/xor5_254_queue_probe/fk10_np1_nb3.stdout`

27. – **NOT_APPLICABLE**: The visited set V is a hash table that stores only hash values, not whole circuits.  
   *Paper:* hash-only visited set  
   *Recomputed:* 'std::set&lt;size_t&gt; hashmap' stores graph-&gt;hash() values.  

28. ✖ **NOT_FOUND**: Preliminary experiments suggested the queue pruning does not affect results.  
   *Paper:* pruning has no effect  
   *Recomputed:* No experiment varies or disables kMaxNumCandidates/kShrinkToNumCandidates.  
   *Evidence:* `/home/cc/mingkuan/qalm/xor5_254_queue_probe/fk10_np1_nb3.stdout`

29. – **NOT_APPLICABLE**: Greedy mode runs for k=1 and k=2: one pass per transformation, wrap-around position scanning, accepting only cost-reducing results after ROQC.  
   *Paper:* greedy k&lt;=2 algorithm  
   *Recomputed:* greedy_optimize_with_roqc loops over xfers with a (last_node_id + node_diff) % size scan and accepts after run_roqc if cost drops. greedy_optimize_with_local_search covers k=2. Both were called with preprocess=1 in the main 1h run (b10f6ec test_qalm.cpp), and with greedy_k=2 in the 1-min run and ablations.  

30. – **NOT_APPLICABLE**: GUOQ's resynthesis is approximate and does not preserve circuit semantics exactly.  
   *Paper:* approximate resynthesis  
   *Recomputed:* guoq_config in the TWO_Q results records epsilon=1e-8 with BQSKit resynthesis (bqskitOptLevel 3).  
   *Evidence:* `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/0410184_169/guoq/results_none_1.json`

31. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: In one hour QALM achieves 5.97% more gate reduction than the strongest baseline, and 4.98% more when given one minute.  
   *Paper:* 5.97%, 4.98%  
   *Recomputed:* The QALM side recomputes from raw data: 52.34 (1h) and 51.35 (1 min). Subtracting the table's VOQC 46.37 gives 5.97 and 4.98.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

32. ❌ **CONTRADICTED**: Cutting the budget 60x degrades QALM's average fidelity and gate reduction by less than 1%.  
   *Paper:* &lt; 1% degradation  
   *Recomputed:* From raw data, 1h to 1min: total 52.336 -&gt; 51.348 (-0.99 pp, -1.89% relative); CX 24.601 -&gt; 21.184 (-3.42 pp, -13.9% relative); fidelity 0.3799 -&gt; 0.3747 (-0.0052, -1.36% relative). From the paper's table: fidelity 0.3864 -&gt; 0.3747 (-1.17 pp, -3.03% relative); 2Q 24.61 -&gt; 21.18 (-3.43 pp).  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_results/results_3600s.csv`; `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`

33. ◐ **DATA_ONLY_NOT_RECOMPUTABLE**: With one minute QALM still beats every baseline's one-hour fidelity and total gate reduction. The conclusion says it surpasses all search-based baselines' one-hour results within the first minute.  
   *Paper:* 1-min QALM &gt; all 1-h baselines (fidelity, total)  
   *Recomputed:* QALM 1-min is 0.3747 / 51.35 from raw data. Against the baselines with raw data it wins: Quartz 0.3526/40.73, GUOQ-2Q about 0.348/-3.7 (from latest_sol), QUESO-2Q about 0.336/27.4. The binding baselines (GUOQ-Total fidelity 0.3676, VOQC total 46.37) have no raw data.  
   *Evidence:* `/home/cc/mingkuan/qalm/guoq_benchmark_prev_qalm_results/results_60s.csv`; `/home/cc/mingkuan/qalm/original_benchmark_results/results_3600s.csv`; `/home/cc/awagner/fidelity_tests/qalm_benchmarks/docker_results_05-13-26_03-05-44/`

34. – **NOT_APPLICABLE**: Table tab:eval_2q (CX count after 10 minutes) is commented out of the compiled paper.  
   *Paper:* not in compiled paper  
   *Recomputed:* Not audited because it is not in the compiled paper.  
   *Evidence:* `/home/cc/mingkuan/qalm/2q_fresh_results/qalm_bench/nam/`; `/home/cc/fresh_results_2_8_26/rq1/nam/`

35. – **NOT_APPLICABLE**: Table tab:quarl (comparison with Quarl) is commented out of the compiled paper.  
   *Paper:* not in compiled paper  
   *Recomputed:* Not audited because it is not in the compiled paper.  

36. – **NOT_APPLICABLE**: Figure fig:reduction_over_time (figure/avg_reduction.pdf) is commented out of the compiled paper.  
   *Paper:* not in compiled paper  
   *Recomputed:* Not audited because it is not in the compiled paper.  
   *Evidence:* `/home/cc/mingkuan/qalm/all_bench_results_hour/fresh_results_{30,60,180,600,900,1800,2700,3600}/qalm_bench/nam/`

37. – **NOT_APPLICABLE**: The illustrative figures and algorithm boxes (Fig. fig1, fig:search_advantage, fig:HadRule, fig:RZandCNOTRule, Alg. 1-3) are not empirical.  
   *Paper:* illustrative  
   *Recomputed:* Viewed both PDFs. fig1 is a schematic gate-count landscape with points A/B/C. search_advantage is a 3-qubit CX rewrite example (3 -&gt; 4 -&gt; 2 gates). Neither contains measured data.  

