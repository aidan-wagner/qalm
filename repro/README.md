# Experiments (`repro/`)

The harness that ran every experiment of the paper, the collected results, and the scripts
that turn them into the paper's tables, figures and numbers.

## Layout

| Path | What |
|---|---|
| `config/campaign.yaml` | machine and resource policy: job cores, memory caps, binaries, runtime library (`lib`, required: `env/record_build.sh` reads it), ECC set, docker image digest, native build tag |
| `config/E*.yaml` | the experiments, one file per part of the evaluation (below) |
| `config/mem_priors.csv` | expected peak memory per circuit and optimizer, used only to schedule jobs |
| `inputs/S248.csv`, `S26.csv` | the input circuits with sha256 and gate counts (`make_manifests.py`); paths are relative to the repository root |
| `inputs/fetch_guoq_benchmarks.sh` | downloads S248 (GUOQ's `benchmarks/nam_rz` at commit 8c4c3a5) and checks it |
| `harness/` | job expansion (`jobs.py`), scheduler (`sched.py`, `run.py`, `launch_sched.sh`), per-job wrapper (`jobwrap.py`), log parsers, result collection (`collect.py`), gate counter (`qasm_count.py`), equivalence check (`verify.py`) |
| `baselines/` | entry point of the GUOQ/QUESO/VOQC container (`entry.py`), the single-core resynthesis server for GUOQ, and the authors' original scripts (`upstream/`); the image and how each baseline is run: `baselines/README.md` |
| `results/` | collected results of the paper's 3,676 runs (below) |
| `analysis/` | metric definitions (`metrics.py`), table/figure/number generators, `make_all.sh`, and their output `out/` |
| `env/` | build provenance (`build_info_paper.txt`), host snapshots of the paper's machine, CPU-mode script, systemd slice |
| `runs/` | raw per-job directories of a run (not in git) |

## Experiments

| Config | Paper | Suite | Runs |
|---|---|---|---|
| `E1_main` | Table 1, Figs. 2, 6, 7 | S248 | QALM (a) at 1 h and 60 s, Quartz, GUOQ (TWO_Q, TOTAL), QUESO (TWO_Q, TOTAL), VOQC; QALM (b) at 1 h and 60 s |
| `E2_quartz26` | Table 2 (Sec. 5.4) | S26 | VOQC, QUESO TOTAL, GUOQ TOTAL, Quartz, QALM (a), QALM (b) |
| `E3_greedy` | Fig. 8 (Sec. 5.5) | S26 | QALM with greedy k 0..3 |
| `E4_single_k` | Fig. 9 (Sec. 5.6) | S26 | QALM with greedy k in {0, 2} and fixed k 1..5 |
| `E5_explore` | Fig. 10 (Sec. 5.7) | S26 | QALM with N_pool 1..5 and N_branch 1..5, seeds 0, 1, 2 |

- **QALM (a)** is the configuration of the paper: `build/test_greedy_k_ablation` with greedy
  k <= 2, N_pool = 1, N_branch = 3 and advancing k (a pass ends when its queue is empty, then
  k + 1), intermediate circuits not enqueued.
- **QALM (b)** is supplementary and not in the paper: `build/test_qalm` with N_pool = 1,
  N_branch = 1, fixed k = 3 and intermediate circuits enqueued. Leave it out with
  `run.py plan --optimizers qalm,quartz,guoq,queso,voqc` (saves about 280 core-hours).
- Quartz with preprocessing (Quartz on the rotation-merged inputs in `circuit/nam_rm_circs`) is
  not run: 4 of those files (`mod_mult_55`, `rc_adder_6`, `csum_mux_9`, `qcla_mod_7`) are not
  equivalent to their `circuit/nam_circs` originals (checked with MQT QCEC), and `verify.py`
  compares an output only with the circuit the job was given, so it cannot catch that.
- Runs shared by several experiments run once. A job's id is a hash of optimizer, parameters,
  circuit, input hash, budget, seed and (native jobs) the build tag `native_build`.
- Two provenance limits for the containers: (1) a docker job's id does not include the image, so
  after a change of `docker_image` in `config/campaign.yaml` the scheduler treats existing
  GUOQ/QUESO/VOQC results as done; use a fresh runs directory then. Each run records the image
  digest it ran in its `launch.json`. (2) `run.py collect` refuses to mix native binaries or
  libraries within one optimizer, but does not compare the GUOQ jar: each container run records
  its hash (`image_jar_sha256` in `out/result.json`), and a mix has to be checked by hand.

## Resource policy

- One physical core per job: logical CPUs 2-63 and 66-127 of the paper's machine (124 job
  slots), whose SMT siblings stay idle. Every other process (user sessions, system services, the
  docker daemon, the scheduler) is confined to the housekeeping CPUs by `harness/launch_sched.sh`.
- Memory: at most 32 GiB per job (cgroup `MemoryMax`, docker `--memory`), no swap. Native jobs
  start with the smallest cap in {2, 4, 8, 16, 32} GiB that covers 1.5x their expected peak and are
  re-run with the next cap if they reach it (or 90% of it in RSS) at any time, also after their
  budget, so every reported run ran as it would have under 32 GiB. Containers always get 32 GiB.
- CPU boost off and governor `performance` (`env/set_cpu_mode.sh fixed`); the scheduler refuses
  to start otherwise and records the mode with every job. **Warning:** the recorded mode is the
  one at the job's launch (`cpu_mode` in its `launch.json`). The scheduler re-checks the mode
  every 60 s and stops launching while it is wrong (a `cpu_mode` event in `events.csv`), but
  jobs already running keep running and are reported normally. If `events.csv` has a
  `cpu_mode` event, re-run the jobs that were running during it by hand.
- QALM and Quartz run natively with `OMP_NUM_THREADS=RAYON_NUM_THREADS=1`; GUOQ, QUESO and VOQC
  run in the authors' image `amxu/guoq@sha256:310905ea...`.
- Seed 0 for every run except E5; GUOQ and QUESO get `--seed` explicitly. Budgets are wall-clock
  time, so runs are not bit-reproducible; E5 measures the run-to-run spread.

Differences from the authors' baseline harness (`baselines/upstream/tool_invoke_defs.sh`): the
resynthesis server `baselines/resynth_1core.py` does not busy-wait (the original spins a core
with `while True: continue`, which under a one-core limit would take CPU time from GUOQ), creates
one BQSKit compiler with one worker (upstream GUOQ commit 8c4c3a5 made the same "create once"
fix), and is started only for GUOQ TWO_Q (TOTAL mode never resynthesizes).

## Result rule

The result of a run is the best circuit the tool had reported at or before its budget:
for QALM and Quartz, the last progress line with t <= T (the final circuit is checked against the log);
for GUOQ and QUESO, `latest_sol_*.qasm` (the JVM is killed at T); for VOQC, its output. A run that
reported nothing counts as the original circuit. Gate counts of GUOQ, QUESO and VOQC come only
from the circuit they wrote, counted by `harness/qasm_count.py` (also used for the inputs). QALM
and Quartz print their counts with every progress line (their own count of the circuit at that
point); `qasm_count.py` counts their final circuit, and `log_final_consistent` in `runs.csv`
records whether the last progress line agrees with it (see below). A tool that drops trivial
rotations such as `rz(0)` when it writes a circuit gets the reduction.

Statuses in `results/runs.csv`: `OK`; `OOM_CAP` (reached 32 GiB; 2 Quartz runs), `JVM_OOM`
(13 GUOQ/QUESO runs), `TIMEOUT` (7 VOQC runs) count with their last reported circuit;
`OOM_AFTER_T` reached 32 GiB only after the budget, so its result is unaffected (below 32 GiB
it is re-run with the next cap, like any run that reached its cap). Infrastructure
failures (preemption, host OOM, missing or truncated output) are re-run and never reported.

Equivalence (`harness/verify.py`): inputs and outputs are applied to the same random states by a
numpy statevector simulator. QALM, Quartz and VOQC print rotation angles with 6 decimals (and
QALM passes every circuit to ROQC in that text format), so on arbitrary-angle circuits their
outputs are equivalent up to ~1e-6 rad of angle rounding (verdict `pass_rounding`); GUOQ TWO_Q
is approximate by design (`approx`). `results/verify_q16.csv` checks every result on circuits
with at most 16 qubits (229 of the 248 S248 circuits): none fails (one Quartz run that hit its
memory cap before printing a circuit is `unverifiable`; its result is a point of its log).

## Results (`results/`)

| File | What |
|---|---|
| `runs.csv` | one row per run: job id, experiments, suite, circuit, optimizer, parameters, budget, seed, original and reported gate counts, status, memory, CPU time, host and core, binary and library hashes |
| `timeseries.csv.gz` | every improvement of every run (job id, time, total, CX), as the tool logged it (see below) |
| `kevents.csv` | start time and k of every QALM search pass |
| `events.csv` | scheduler events (launch, finish, requeue after an infrastructure failure) |
| `verify_q16.csv` | equivalence verdicts (circuits with at most 16 qubits) |

`runs.csv`'s `log_final_consistent` is `True` when the run's last logged counts equal the
counts of its final circuit. `False` means either that they differ or that there is nothing
to compare: no progress line in the log (VOQC never writes one; GUOQ or QUESO runs that never
improved) or no final circuit (e.g. a native run killed before printing it). Only rows that
have both `n_changepoints > 0` and a `final_total` can be real disagreements.

`timeseries.csv.gz` starts every run at t=0 with the input's counts from the manifest, followed
by the tool's own progress lines with their counts. For GUOQ and QUESO these are the tools' own
counts, which do not include `rz(0)` gates (dropped when reading the circuit), so their last point
can differ from the result in `runs.csv`, which is always counted from the written circuit (e.g.
`ground_state_estimation_10`: 371,940 gates in `runs.csv`, no improvement point in the time
series). VOQC logs nothing, so its series is only the t=0 point. `runs.csv` is authoritative;
the analysis uses the time series only for the QALM curves (E3-E5).

Regenerate everything in `analysis/out/` with `analysis/make_all.sh` (see the top-level README
for which output is which table or figure).

## Running the experiments

The commands below were used for the paper. They need systemd, passwordless `sudo` and Docker.
`config/campaign.yaml` describes the paper's machine (2x AMD EPYC 7763, 128 physical cores,
251 GiB). On another machine adapt, in `config/campaign.yaml`:
- `cores.jobs`: one logical CPU per physical core that may run a job (`lscpu -e` lists the cores;
  leave the SMT siblings out), and `cores.housekeeping`: the remaining CPUs (incl. siblings);
- `numa_of_core`: `{split: N}` with N the first CPU of NUMA node 1 (or a value above every CPU
  on a single-node machine). This assumes at most two nodes, each with contiguous CPU numbers
  among the job CPUs (check `lscpu | grep NUMA`; e.g. `node0 0-63,128-191` is fine). On
  machines that number CPUs alternately across nodes (`node0 0,2,4,...`, common on Intel servers)
  the per-node memory budget is applied to the wrong CPUs: this only affects scheduling
  (possibly preemptions and re-runs), not results;
- `mem`: `slice_cap_gib` and `budget_gib` a little below the machine's RAM, `node_budget_gib`
  about `budget_gib` / number of NUMA nodes; keep `cap_tiers_gib` (the per-job cap of 32 GiB is
  part of the setup);

and in `env/qalmrepro.slice` the same CPUs (`AllowedCPUs` = `cores.jobs`) and
`MemoryMax` = `slice_cap_gib` (the scheduler checks both). Machines without CPU frequency control
(many VMs): skip `set_cpu_mode.sh` and set `REPRO_ALLOW_CPU_MODE=1` for `launch_sched.sh`, which
also leaves other processes unconfined; the runs are then less comparable to the paper's.
`set_cpu_mode.sh restore` does not restore the previous setting: it sets the paper machine's
defaults (boost on, governor `schedutil`). On machines with other defaults (e.g. Intel
`intel_pstate`, which offers only `performance` and `powersave`) it stops after re-enabling
boost; set the governor by hand or reboot (the settings of `fixed` do not survive a reboot).

```shell
# build as in the top-level README, then record the build; the scheduler refuses to start if the
# binaries no longer match this record
repro/env/record_build.sh
repro/inputs/fetch_guoq_benchmarks.sh
sudo docker pull amxu/guoq@sha256:310905ead964978676921d191819ce21f9ccdd06e5709a6b6b5dbec4777780e9
repro/env/set_cpu_mode.sh fixed                       # once per boot; "restore" undoes it

# plan and launch; the scheduler is a systemd unit and survives logout
uv run repro/harness/run.py plan -o repro/runs/plan.jsonl          # add --shard i/n to split over n machines
repro/harness/launch_sched.sh repro/runs/plan.jsonl
uv run repro/harness/run.py status                                  # also: drain, resume, stop
repro/harness/monitor.sh                                             # optional: wait for anomalies

# afterwards: collect the results (overwrites repro/results/) and check equivalence
uv run repro/harness/run.py collect --plan repro/runs/plan.jsonl --gzip-logs
uv run repro/harness/verify.py --plan repro/runs/plan.jsonl --max-qubits 16 --out repro/results/verify_q16.csv
repro/analysis/make_all.sh

# give the machine back (launch_sched.sh confined everything else to the housekeeping CPUs;
# the settings are runtime-only, so a reboot also undoes them)
for u in user.slice system.slice init.scope; do
  sudo systemctl set-property --runtime $u AllowedCPUs=0-$(( $(nproc --all) - 1 ))
done
repro/env/set_cpu_mode.sh restore
```

Smaller runs: `run.py plan` accepts `--experiments E3_greedy`, `--circuits tof_3,barenco_tof_4`,
`--optimizers qalm,quartz` and `--budget-override 60` (shorter budget), e.g.
a smoke test of a few minutes. A job's id depends on its budget, so overridden jobs get new
ids, except jobs whose budget already equals the override (with `--budget-override 60`, the
60 s QALM runs of E1): those keep their campaign ids. Run smoke plans in their own runs
directory, as below, never in `repro/runs`, or the scheduler treats those jobs as done and
`collect` reports the campaign's results for them:

```shell
uv run repro/harness/run.py plan --circuits tof_3,barenco_tof_3 --budget-override 60 \
  -o /tmp/plan_smoke.jsonl                                          # 108 jobs, every experiment
REPRO_ALLOW_CPU_MODE=1 repro/harness/launch_sched.sh /tmp/plan_smoke.jsonl $PWD/repro/runs_smoke
export REPRO_RUNS_DIR=$PWD/repro/runs_smoke                         # runs dir must be absolute
uv run repro/harness/run.py status                                 # done after ~3 minutes
uv run repro/harness/run.py collect --plan /tmp/plan_smoke.jsonl --out /tmp/smoke_results
uv run repro/harness/verify.py --plan /tmp/plan_smoke.jsonl --results /tmp/smoke_results
```

Notes on the `run.py plan` options:
- The filters are not checked: a misspelt name (`--experiments E3_gredy`, `--circuits tof3`) or
  an out-of-range shard (`--shard 2/2`) gives an empty plan; the only sign is "0 jobs" in the
  output. `--circuits` selects by name in every suite (S248 and S26 share 26 names).
- `--shard i/n` splits by circuit name, so all jobs of a circuit run on the same machine. The
  shards are not balanced: the S26 circuits carry about 50 jobs each, the others about 10 (with
  n = 2: about 1,600 and 1,280 core-hours). Results of different machines are only comparable
  if the machines are identical (same CPU, same `campaign.yaml` settings).
- A plan stores absolute input paths (`input_path`) of the checkout that made it. Make each
  plan, shards included, in the checkout that runs it (or in one at the same path); a plan
  moved to another location points to the wrong circuit files.
- `--native-experiments E4_single_k,...` keeps every docker job but native jobs only of the
  listed experiments. It is meant for re-running native jobs after a new build in the same runs
  directory, where the docker jobs are already done and are skipped; in a fresh runs directory
  it runs all docker jobs too.
- `--campaign FILE` plans against another campaign file (e.g. one with a new `native_build`
  tag) without replacing `config/campaign.yaml`. `launch` always reads
  `config/campaign.yaml` and refuses a plan whose native build tag differs from it.

`runs/HOLD` (a JSON list of rules such as `[{"kind": "docker"}]`) keeps matching queued jobs
from being launched without stopping anything; write it atomically (tmp file + `mv`) and set it
to `[]` to release them.
