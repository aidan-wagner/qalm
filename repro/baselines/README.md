# Baselines (GUOQ, QUESO, VOQC)

The baselines run in the Docker image of the GUOQ authors' artifact, unchanged, so that every
baseline runs exactly as its authors ran it. This directory contains only what the harness
mounts into the container (`/repro`, read-only).

## The image

| | |
|---|---|
| Image | `amxu/guoq:1`, used by digest `amxu/guoq@sha256:310905ead964978676921d191819ce21f9ccdd06e5709a6b6b5dbec4777780e9` (`docker_image` in `repro/config/campaign.yaml`) |
| Source | artifact of *Optimizing Quantum Circuits, Fast and Slow* (GUOQ, ASPLOS 2025), Zenodo [10.5281/zenodo.14055562](https://doi.org/10.5281/zenodo.14055562), 2024-11-08 |
| Size | 22.5 GB (5.8 GB compressed) |
| Used inside | `/home/GUOQ-1.0-jar-with-dependencies.jar` (md5 `8fe405b8f0dd61a48e7248d8bd0681b6`, the jar of the Zenodo record), `/home/queso_rules/`, OpenJDK 21.0.4; conda env `guoq` (BQSKit 1.1.1, Qiskit 0.46.0) for GUOQ's resynthesis server; conda env `voqc` (pyvoqc 0.1.1) with `/home/run_voqc.py` |

Get it with either:

```shell
sudo docker pull amxu/guoq@sha256:310905ead964978676921d191819ce21f9ccdd06e5709a6b6b5dbec4777780e9
# or, if Docker Hub no longer has it, from the Zenodo record (same image):
curl -L -o guoq_docker_image.tar.gz https://zenodo.org/records/14057840/files/guoq_docker_image.tar.gz
sudo docker load --input guoq_docker_image.tar.gz     # md5 904e5ebcd5d1a113ebc5f49bb52d8619
# a loaded image has no registry digest: then set docker_image: "amxu/guoq:1" in campaign.yaml
```

The authors built the image interactively; no Dockerfile reproduces it. (The `Dockerfile` in the
GUOQ repository builds a later image on top of `amxu/queso:20`, not this one.)

## How each baseline is run

`entry.py` (the container's entry point, run by the harness as
`/opt/conda/bin/python /repro/entry.py <tool> <mode> /in/<circuit>.qasm <budget_s> <seed>`)
reproduces the authors' harness (`upstream/tool_invoke_defs.sh`, `upstream/run_guoq.py`,
`upstream/run_voqc.py`, copied from `/home` of the image):

| Baseline | Command inside the container |
|---|---|
| GUOQ (TWO_Q, TOTAL) | `java -Xmx20g -cp GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g NAM -opt <mode> --rules-dir /home/queso_rules/ --resynth-weight 180 --seed <seed>`; TWO_Q also starts the resynthesis server |
| QUESO (TWO_Q, TOTAL) | the same jar with QUESO's search: `-g NAM -opt <mode> --rules-dir /home/queso_rules/ -search BEAM -temp 0 -q 8000 -resynth NONE --seed <seed>` |
| VOQC | `/home/run_voqc.py --circuit_file <circuit> --gate_set nam` in the `voqc` env |

Every tool runs under `timeout <budget>`; GUOQ and QUESO write `latest_sol_*.qasm` whenever they
improve, and the result of a run is the last one written before the budget (see `repro/README.md`).
`entry.py` also writes `result.json` (exit code, wall/CPU time, peak memory, cgroup counters).

Differences from the authors' harness, all needed to run one job per core:

- `resynth_1core.py` replaces the image's `/home/resynth.py` (`resynth_1core.diff`): the original
  busy-waits a core (`while True: continue`), which under a one-core limit takes CPU time from GUOQ;
  it now creates one BQSKit compiler with one worker (upstream GUOQ commit 8c4c3a5 made the same
  "create once" change); and it is started only for GUOQ TWO_Q (TOTAL never resynthesizes).
- `--seed` is passed explicitly.

## Running one baseline by hand

```shell
mkdir -p /tmp/bl && sudo docker run --rm --cpuset-cpus=2 --memory=32g --network=none \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -v $PWD/circuit/nam_circs:/in:ro -v $PWD/repro/baselines:/repro:ro -v /tmp/bl:/out -w /out \
  amxu/guoq@sha256:310905ead964978676921d191819ce21f9ccdd06e5709a6b6b5dbec4777780e9 \
  /opt/conda/bin/python /repro/entry.py guoq TOTAL /in/tof_3.qasm 60 0
ls /tmp/bl    # latest_sol_none_1_tof_3.qasm, tool.out, result.json
```
