# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QALM** is a quantum circuit optimizer that combines **Quartz** (a C++ quantum circuit superoptimizer) with **ROQC** (a Rust-based rotation optimization pass). The key innovation is interleaving Quartz's backtracking search with ROQC's rotation cancellation/merging during optimization.

## Build

### Prerequisites

```shell
# Conda environment (Python 3.11, Cython, pybind11, z3-solver, qiskit, sympy)
conda env create --name quartz --file env.yml
conda activate quartz

# Arb library (required for C++ build)
sudo apt install libflint-arb-dev=1:2.22.1-1

# Rust (for ROQC)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build C++ (Quartz)

```shell
mkdir build && cd build
cmake ..
make
cd ..
```

Binaries land in `build/` (e.g., `build/test_optimize`, `build/test_qalm`).

### Build ROQC (Rust)

```shell
cd roqc
cargo build --release
```

### Build Python bindings (optional)

```shell
cd src/python
python setup.py build_ext --inplace install
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Build in Docker

```shell
sudo docker build -t qalm_testing .
sudo docker run -it qalm_testing
# Inside container:
mkdir build_docker && cd build_docker && cmake .. && make && cd ..
```

## Running Tests

### C++ tests (individual executables in `build/`)

```shell
./build/test_optimize       # verify basic optimization works
./build/test_qalm           # QALM-specific tests
./build/test_nam            # Nam gate set tests
./build/test_rotation_merging
```

### Python tests

```shell
cd src/python
python -m pytest
```

### ROQC tests

```shell
cd roqc
cargo test
python qalm_circuits_test.py
```

## Running Experiments

All experiment scripts must be run from the repo root. The primary experiment script is `qalm_experiments/comparison_experiments.py`, which requires `PYTHONPATH=qalm_experiments` since it imports `equiv_verification` from that directory.

```shell
# Main QALM parameter-sweep experiments (creates comparison_results/ and pickled_results/)
mkdir -p comparison_results pickled_results
PYTHONPATH=qalm_experiments python3 qalm_experiments/comparison_experiments.py
```

The script runs 30 QALM configurations × 28 circuits with a 600s timeout using 32 parallel workers. Results are pickled per-circuit in `pickled_results/` (so reruns skip completed circuits) and output QASM files go to `comparison_results/{circuit_name}/`. Plots are saved as `comparison_results/result_figure_{circuit}_{timeout}_seconds.png`.

To customize: edit `timeout`, `circuit_list`, and `experiments` at the top of `run_experiments()`.

```shell
# ROQC benchmarks
python qalm_experiments/repeated_roqc_tests.py qalm_experiments/qalm_circuits_test.txt   # test set
python qalm_experiments/repeated_roqc_tests.py qalm_experiments/qalm_circuits_full.txt   # full set
```

## Code Formatting

Pre-commit hooks enforce formatting on `git commit`. Install once:

```shell
pip install pre-commit clang-format
pre-commit install
```

Run manually: `pre-commit run --all-files`

- C++: clang-format (Google style, see `.clang-format`)
- Python: black + isort

## Architecture

### Two-phase workflow

1. **ECC Generation** (`src/quartz/generator/` + `src/python/verifier/verifier.py`): Generate candidate circuits, verify equivalence using Z3 SMT solver, produce Equivalent Circuit Class (ECC) sets stored in `eccset/`.

2. **Circuit Optimization** (`src/quartz/tasograph/tasograph.cpp`): Load an ECC set, parse input QASM, run cost-based backtracking search applying `GraphXfer` transformations. QALM interleaves this with ROQC rotation optimization at configurable intervals.

### Key components

| Component | Location | Role |
|-----------|----------|------|
| `Graph` | `src/quartz/tasograph/tasograph.h` | DAG circuit representation; `Graph::optimize` is the main search entry point |
| `GraphXfer` | `src/quartz/tasograph/substitution.h` | A circuit transformation rule derived from an ECC |
| `CircuitSeq` | `src/quartz/circuitseq/circuitseq.h` | Sequential gate-list circuit representation |
| `Context` | `src/quartz/context/context.h` | Gate set definition for a target quantum processor |
| `EquivalenceSet` | `src/quartz/dataset/equivalence_set.h` | Collection of verified ECCs used by the optimizer |
| `QASMParser` | `src/quartz/parser/qasm_parser.h` | Parses OpenQASM 2.0/3.0 into `CircuitSeq` |
| Z3 verifier | `src/python/verifier/verifier.py` | `equivalent()` / `find_equivalences()` using Z3 |
| ROQC | `roqc/roqc/src/` | Rust: rotation merging, gate propagation/cancellation |

### Data flow

```
Input QASM
  → QASMParser → CircuitSeq
  → Graph (DAG)
  → Graph::optimize (backtracking + GraphXfer from ECC set)
      ↕ (at intervals) ROQC rotation pass
  → Output optimized QASM
```

### Gate sets

Supported: **Nam** (`{rz, h, cnot, x, t, tdg}`), **IBM** (`{u1, u2, u3, cx}`), **Rigetti** (`{rx, rz, cz}`). ECC sets for each are in `eccset/` and `example_eccsets/`.

### Experiment/RL code

`experiment/` contains deprecated and active PPO-based RL approaches for learning circuit optimization policies. These are separate from the main Quartz optimizer and use PyTorch.

## Paper

The QALM paper draft is in `seach_and_rule_based_opt/` (note the directory name has a typo — "seach" instead of "search"). The paper is titled **"QALM: Synergizing Exploration and Exploitation in Quantum Circuit Optimization"** and targets OSDI (submission #2205). It uses the USENIX 2020-09 LaTeX template.

Structure:
- `main.tex` — top-level document
- `texts/` — one file per section (`0-abstract.tex` through `8-ack.tex`; `3-method-old.tex` is a superseded draft)
- `figtex/` — figure/algorithm `.tex` includes
- `figure/` — PDF/PNG figures
- `ref.bib` — bibliography
