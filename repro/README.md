# QALM paper re-run (`repro/`)

Everything needed to re-run every experiment of the QALM paper from scratch,
on one or more identical machines, and to regenerate every table, figure and
number in the paper from the raw results. Started 2026-10-30 because about
half of the paper's original results had no surviving raw data
(see `reference/audit_20261030/qalm_repro_audit.md`).

## Layout

| Path | What |
|---|---|
| `config/campaign.yaml` | machine/resource policy (cores, memory caps, binaries, ECC set, docker image digest) |
| `config/E*.yaml` | the experiments (one file per paper section) |
| `config/mem_priors.csv` | expected peak memory per circuit, from earlier runs (`make_mem_priors.py`) |
| `inputs/S248.csv`, `S26.csv`, `S26PP.csv` | input circuits with sha256 and gate counts (`make_manifests.py`) |
| `harness/` | job expansion, scheduler, per-job wrapper, log parsers, result collection, equivalence check |
| `baselines/` | container entry point for GUOQ/QUESO/VOQC, the patched resynthesis server, and the authors' original scripts (`upstream/`) |
| `analysis/` | metric definitions (`metrics.py`) and the generators for every paper table/figure/number |
| `results/` | committed results: `runs.csv` (one row per job), `timeseries.csv.gz`, `kevents.csv`, `ARTIFACTS.sha256` |
| `env/` | host snapshots, build info, roqc source manifest, CPU-mode script |
| `reference/` | frozen copy of the audit of the previous paper version |
| `runs/` | raw per-job directories (gitignored; archived to the Chameleon object store container `qalm-repro`) |

## Experiments

| Config | Paper content | Suite | Jobs |
|---|---|---|---|
| `E1_main` | Table avg-comparison, Figs comparison_3x3 / win_rate / gate_scatter | S248 | QALM (a) and (b) at 1 h and 60 s, Quartz, GUOQ TWO_Q/TOTAL, QUESO TWO_Q/TOTAL, VOQC |
| `E2_quartz26` | Table eval_end_to_end | S26 (+ S26PP for Quartz w/ preprocessing) | VOQC, QUESO TOTAL, GUOQ TOTAL, Quartz, QALM (a), QALM (b) |
| `E3_greedy` | Fig greedy_k | S26 | greedy_k 0..3 |
| `E4_single_k` | Figs single_k_gk0/gk2 | S26 | greedy_k {0,2} x fixed k 1..5 |
| `E5_explore` | Figs N_pool / N_branch | S26 | 9 configs x seeds {0,1,2} |

QALM (a) is the configuration the paper describes (`N_pool=1, N_branch=3`, greedy k<=2,
advancing k, no intermediate enqueue). QALM (b) is the configuration the previous
main run actually used (`build/test_qalm 1 1 3 1.5 0 0 1 1 0`). Identical runs shared by
several experiments are executed once (job id = hash of optimizer, parameters, circuit,
input hash, budget, seed).

## Resource policy (identical for every run)

- One physical core per job: logical CPUs 2-63 and 66-127 (124 job slots); their SMT siblings stay
  idle. Every other process (user sessions, system services, the docker daemon, the scheduler) is
  confined to the housekeeping CPUs 0,1,64,65 (+ siblings) by `harness/launch_sched.sh`.
- Memory: at most 32 GiB per job (cgroup `MemoryMax`, docker `--memory`), no swap. To run as many
  jobs in parallel as possible, native jobs start with the smallest cap in {2,4,8,16,32} GiB that
  covers 1.5x their expected peak, and are re-run with the next cap if they reach it (or reach 90%
  of it in RSS) before their budget ends. Every reported run therefore ran with headroom, i.e. as
  it would have under 32 GiB. Containers always get 32 GiB.
- CPU boost off, governor `performance` (`env/set_cpu_mode.sh fixed`); the scheduler refuses to
  launch otherwise and records the mode with every job.
- QALM/Quartz run natively (`OMP_NUM_THREADS=RAYON_NUM_THREADS=1`); GUOQ/QUESO/VOQC run in the
  authors' image `amxu/guoq@sha256:310905ea...` (see `config/campaign.yaml`).
- Seed 0 for every canonical run; GUOQ/QUESO get `--seed` explicitly. Time budgets are
  wall-clock, so runs are not bit-replayable; E5 quantifies run-to-run variance.

Differences from the authors' baseline harness (`baselines/upstream/tool_invoke_defs.sh`):
the resynthesis server `baselines/resynth_1core.py` does not busy-wait (the original spins
a core with `while True: continue`, which under a one-core limit would take CPU from GUOQ),
creates one BQSKit compiler with one worker (upstream GUOQ commit 8c4c3a5 made the same
"create once" fix), and is only started for GUOQ TWO_Q (TOTAL mode never resynthesizes).

## Result rule

The result of a run is the best circuit the tool had reported at or before its budget:
QALM/Quartz: last progress line with t <= T (the final circuit is checked against the log);
GUOQ/QUESO: `latest_sol_*.qasm` (the JVM is killed at T); VOQC: its output. A run that
reported nothing counts as the original circuit. Statuses (`OOM_CAP`, `WALL_KILL`,
`JVM_OOM`, `CRASH`, `TIMEOUT`) are kept in `results/runs.csv` and disclosed in the paper.
Infrastructure failures (preemption, global OOM, missing or truncated output) are re-run, never
reported (`INFRA_GAVE_UP` after 3 attempts). `OOM_AFTER_T`: the job reached its cap only after its
budget, so its result (best at or before T) is unaffected.

Gate counts are always taken from the circuit a tool reports, with one counter
(`harness/qasm_count.py`). Some tools drop trivial rotations (e.g. `rz(0)`) when they write a
circuit out; that counts as a reduction for whichever tool does it (decided 2026-10-30).

Equivalence (`harness/verify.py`, circuits up to 16 qubits): QALM and Quartz print rotation angles
with 6 decimals, and QALM also passes every circuit to ROQC through that text format, so on
arbitrary-angle circuits their outputs are equivalent only up to ~1e-6 rad angle rounding
(state overlap >= 1 - 1e-6, verdict `pass_rounding`). The binaries are evaluated unchanged
(decided 2026-10-30). GUOQ TWO_Q is approximate by design (`approx`).

## Running

```shell
cd /home/cc/mingkuan/qalm
# one-time per boot
sudo cp repro/env/qalmrepro.slice /run/systemd/system/ && sudo systemctl daemon-reload
repro/env/set_cpu_mode.sh fixed
# build (see env/build_info.txt for the exact configuration used)
cmake -S . -B build/repro -DRust_TOOLCHAIN=stable-x86_64-unknown-linux-gnu -DPython_EXECUTABLE=/usr/bin/python3
RUSTUP_TOOLCHAIN=stable cmake --build build/repro --target test_greedy_k_ablation test_qalm test_original -j 48
sudo docker pull amxu/guoq@sha256:310905ead964978676921d191819ce21f9ccdd06e5709a6b6b5dbec4777780e9
# plan and launch (the scheduler is a system unit and survives logout)
python3 repro/harness/run.py plan -o repro/runs/plan.jsonl      # add --shard i/n per node
repro/harness/launch_sched.sh repro/runs/plan.jsonl
python3 repro/harness/run.py status                              # drain / resume / stop also available
# hold queued jobs without stopping anything: repro/runs/HOLD is a JSON list of rules such as
# [{"optimizer": ["qalm", "qalm_old"]}]; it is re-read on change, delete it to release
# afterwards
python3 repro/harness/run.py collect --plan repro/runs/plan.jsonl --gzip-logs
python3 repro/harness/verify.py --plan repro/runs/plan.jsonl
```
