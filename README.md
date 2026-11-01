# QALM artifact

Code, inputs, raw results and analysis scripts for the paper
*QALM: Escaping Local Minima via Interleaved Exploration and Exploitation in Quantum Circuit Optimization*.

QALM is built on the [Quartz](https://github.com/quantum-compiler/quartz) circuit optimizer
(C++, this repository is a fork of it) and on ROQC, a rule-based rotation optimizer written in
Rust (`roqc/`, linked into the Quartz library through [corrosion](https://github.com/corrosion-rs/corrosion)).
Every table, figure and number in the paper's evaluation is computed from `repro/results/`
by the scripts in `repro/analysis/`; the experiments themselves are run by the harness in
`repro/harness/`.

## Contents

| Path | What |
|---|---|
| `src/quartz/` | Quartz library; the QALM search is `Graph::optimize_qalm` and the greedy passes are `Graph::greedy_optimize_with_roqc` / `greedy_optimize_with_local_search` in `src/quartz/tasograph/tasograph.cpp` |
| `src/test/test_greedy_k_ablation.cpp` | QALM driver used for every QALM result in the paper |
| `src/test/test_original.cpp` | Quartz baseline driver (greedy pass, then Quartz's backtracking search) |
| `src/test/test_qalm.cpp` | QALM configuration (b) (supplementary, not in the paper) |
| `src/test/gen_ecc_set.cpp` | regenerates the ECC set `eccset/Nam_5_3_complete_ECC_set.json` (shipped) |
| `roqc/` | ROQC (Rust) |
| `circuit/nam_circs/`, `circuit/nam_rm_circs/` | the 26-circuit benchmark (S26), and the same circuits after rotation merging (S26PP, input of "Quartz with preprocessing") |
| `benchmarks/guoq_nam_rz/` | the 248-circuit benchmark (S248); not in git, downloaded by `repro/inputs/fetch_guoq_benchmarks.sh` |
| `repro/` | experiment harness, configurations, results and analysis; see [`repro/README.md`](repro/README.md) |

## Requirements

- Linux x86-64 (the paper's runs: Ubuntu 24.04, 2x AMD EPYC 7763, 251 GiB RAM).
- Build: CMake >= 3.16, GCC with OpenMP (paper: GCC 13.3.0, CMake 3.28.3), Python 3 with
  development headers and `pybind11`, and `rustup` (the Rust toolchain is pinned to 1.94.1 in
  `roqc/rust-toolchain.toml`). CMake downloads corrosion v0.5 at configure time.
- Analysis and harness: Python 3 with `numpy`, `scipy`, `matplotlib`, `pyyaml`
  (`repro/requirements.txt`), installed for the system `/usr/bin/python3`, which the scheduler runs.
- Full re-run only: `systemd` and passwordless `sudo` (the scheduler runs every job in its own
  transient unit with a CPU and memory limit), and Docker for the baselines GUOQ, QUESO and VOQC
  (the GUOQ authors' artifact image `amxu/guoq:1`, pinned by digest; see
  [`repro/baselines/README.md`](repro/baselines/README.md), also for getting it from Zenodo).

On Ubuntu 24.04 (as root or with `sudo`):

```shell
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl ca-certificates python3-dev \
  python3-pybind11 pybind11-dev python3-yaml python3-numpy python3-scipy python3-matplotlib
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
. "$HOME/.cargo/env" && rustup toolchain install 1.94.1
# full re-run only:
sudo apt-get install -y docker.io && sudo systemctl enable --now docker
```

## Build

```shell
git clone -b artifact https://github.com/aidan-wagner/qalm.git && cd qalm
cmake -S . -B build -DRust_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu -DPython_EXECUTABLE=/usr/bin/python3
RUSTUP_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu cmake --build build -j \
  --target test_greedy_k_ablation test_qalm test_original
```

This produces `build/test_greedy_k_ablation`, `build/test_qalm`, `build/test_original` and
`build/libquartz_runtime.so`. `repro/env/build_info_paper.txt` records the exact toolchain and
the hashes of the binaries that produced the paper's results.

## Quick test (a few minutes, no harness)

Optimize one circuit of S26 with QALM for 60 seconds, in the paper's configuration
(greedy pass with k <= 2, then the search with N_pool = 1, N_branch = 3 and advancing k):

```shell
build/test_greedy_k_ablation circuit/nam_circs/barenco_tof_4.qasm barenco_tof_4 60 \
  2 1 3 1.5 0 0 1 0 eccset/Nam_5_3_complete_ECC_set.json 0 0 0 0
```

Arguments: input, name, time budget (s), greedy k, N_pool, N_branch, repeat tolerance,
k-increase flag (0), strictly-reducing rules (0), local transformations only (1), two-way rotation
merging (0), ECC set, fixed k (0 = advancing k, the paper's default; 1-5 for Sec. 5.6), enqueue
intermediate circuits (0), cost mode (0 = total gate count), seed. The log shows one progress
line per improvement (`[name] Best cost: ... CX: ...`) and ends with the best circuit in OpenQASM.
barenco_tof_4 has 114 gates (48 CX); QALM reaches about 70 (32 CX) within the minute. The Quartz baseline is `build/test_original <input> <name> <budget>`.

## Regenerate the paper's tables, figures and numbers (minutes)

The collected results of all 3,676 runs are in `repro/results/` (`runs.csv`: one row per run;
`timeseries.csv.gz`: every progress line; `kevents.csv`: start time and k of every QALM search pass).

```shell
repro/inputs/fetch_guoq_benchmarks.sh   # optional: only the suite-size number needs it
repro/analysis/make_all.sh              # writes repro/analysis/out/
```

| Paper | Output (`repro/analysis/out/`) | Experiment |
|---|---|---|
| Table 1 (avg-comparison) | `a/averages_table.tex` | E1 |
| Fig. 2 (win rate) | `a/win_rate.png` | E1 |
| Fig. 6 (gate scatter) | `a/gate_scatter.pdf` | E1 |
| Fig. 7 (comparison 3x3) | `a/comparison_3x3.pdf` | E1 |
| Table 2 (eval_end_to_end, Sec. 5.4) | `a/eval_end_to_end.tex` | E2 |
| Fig. 8 (greedy k, Sec. 5.5) | `curves/greedy_k_reduction.pdf` | E3 |
| Fig. 9 (single k, Sec. 5.6) | `curves/single_k_gk0.pdf`, `curves/single_k_gk2.pdf` | E4 |
| Fig. 10 (N_pool / N_branch, Sec. 5.7) | `curves/N_pool_arith_mean_3600s_plain.pdf`, `curves/N_branch_arith_mean_3600s_plain.pdf` | E5 |
| numbers in the text | `numbers/numbers.tex` (one macro per number), `numbers/numbers.json` (how each was computed) | all |

`out/b/` holds the same tables and S248 figures for configuration (b), which is not in the paper.
The committed `repro/analysis/out/` was produced this way.

## Re-run the experiments (about 2,900 core-hours)

See [`repro/README.md`](repro/README.md). In short: the harness expands `repro/config/E*.yaml`
into 3,676 jobs (one core and at most 32 GiB each, 1 hour budget except the 60-second QALM runs),
runs them on the cores listed in `repro/config/campaign.yaml` and collects the results into the
same format as `repro/results/`. On the paper's machine (124 job slots) the campaign took about
a day. Results are not bit-reproducible, because the budgets are wall-clock time; Sec. 5.7 reports the
run-to-run spread over three seeds.

## Quartz

This repository is a fork of Quartz. Quartz:

* Mingkuan Xu, Zikun Li, Oded Padon, Sina Lin, Jessica Pointing, Auguste Hirth,
  Henry Ma, Jens Palsberg, Alex Aiken, Umut A. Acar, and Zhihao Jia.
  [Quartz: Superoptimization of Quantum Circuits](https://arxiv.org/abs/2204.09033).
  In Proceedings of the Conference on Programming Language Design and Implementation (PLDI), June 2022.

## License

Apache License 2.0 (`LICENSE`), as Quartz.
