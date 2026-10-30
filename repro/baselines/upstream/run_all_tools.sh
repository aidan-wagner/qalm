#!/bin/bash

benchmark=$1
timeout=$2
num_guoq_trials=$3

source tool_invoke_defs.sh

############################## Make fresh results dirs ##############################
mkdir -p fresh_results
mkdir -p fresh_results/rq1
mkdir -p fresh_results/rq1/ibm_old
mkdir -p fresh_results/rq1/ibm_old/guoq
mkdir -p fresh_results/rq1/ibm_old/tket
mkdir -p fresh_results/rq1/ibm_old/voqc
mkdir -p fresh_results/rq1/ibm_old/qiskit
mkdir -p fresh_results/rq1/ibm_old/bqskit
mkdir -p fresh_results/rq1/ibm_old/queso
mkdir -p fresh_results/rq1/ibm_old/quartz

mkdir -p fresh_results/rq1/ibm_new
mkdir -p fresh_results/rq1/ibm_new/guoq
mkdir -p fresh_results/rq1/ibm_new/tket
mkdir -p fresh_results/rq1/ibm_new/qiskit
mkdir -p fresh_results/rq1/ibm_new/bqskit
mkdir -p fresh_results/rq1/ibm_new/queso
mkdir -p fresh_results/rq1/ibm_new/quartz
mkdir -p fresh_results/rq1/ibm_new/quarl-1
mkdir -p fresh_results/rq1/ibm_new/quarl-2
mkdir -p fresh_results/rq1/ibm_new/quarl-3

mkdir -p fresh_results/rq1/ion
mkdir -p fresh_results/rq1/ion/guoq
mkdir -p fresh_results/rq1/ion/qiskit
mkdir -p fresh_results/rq1/ion/bqskit
mkdir -p fresh_results/rq1/ion/queso

mkdir -p fresh_results/rq1/nam
mkdir -p fresh_results/rq1/nam/guoq
mkdir -p fresh_results/rq1/nam/tket
mkdir -p fresh_results/rq1/nam/voqc
mkdir -p fresh_results/rq1/nam/qiskit
mkdir -p fresh_results/rq1/nam/bqskit
mkdir -p fresh_results/rq1/nam/queso
mkdir -p fresh_results/rq1/nam/quartz
mkdir -p fresh_results/rq1/nam/quarl-1
mkdir -p fresh_results/rq1/nam/quarl-2
mkdir -p fresh_results/rq1/nam/quarl-3

mkdir -p fresh_results/rq2
mkdir -p fresh_results/rq2/guoq
mkdir -p fresh_results/rq2/guoq-rewrite
mkdir -p fresh_results/rq2/guoq-resynth

mkdir -p fresh_results/rq3
mkdir -p fresh_results/rq3/guoq-rewrite-resynth
mkdir -p fresh_results/rq3/guoq-resynth-rewrite
mkdir -p fresh_results/rq3/guoq-beam

mkdir -p fresh_results/rq4
mkdir -p fresh_results/rq4/guoq
mkdir -p fresh_results/rq4/guoq-rewrite
mkdir -p fresh_results/rq4/guoq-resynth
mkdir -p fresh_results/rq4/guoq-after-pyzx
mkdir -p fresh_results/rq4/queso
mkdir -p fresh_results/rq4/qiskit
mkdir -p fresh_results/rq4/synthetiq
mkdir -p fresh_results/rq4/bqskit
mkdir -p fresh_results/rq4/pyzx

############################## RQ1 IBMO (IBMQ20) ##############################
BENCHMARK_DIR="benchmarks/ibm"
echo "Running RQ1 IBMQ20..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMO!--fidelity!40!--rules-dir!/home/queso_rules/!--resynth-weight!10"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq1/ibm_old/guoq

run_tket $BENCHMARK_DIR/$benchmark $timeout "ibm"
mv results_* fresh_results/rq1/ibm_old/tket

run_voqc $BENCHMARK_DIR/$benchmark $timeout "ibm"
mv results_* fresh_results/rq1/ibm_old/voqc

run_qiskit $BENCHMARK_DIR/$benchmark $timeout "ibm"
mv results_* fresh_results/rq1/ibm_old/qiskit

run_bqskit $BENCHMARK_DIR/$benchmark $timeout "ibm"
mv results_* fresh_results/rq1/ibm_old/bqskit

run_queso $BENCHMARK_DIR/$benchmark $timeout "-g!IBMO!-opt!TOTAL!--rules-dir!/home/queso_rules/!-search!BEAM!-temp!0!-q!8000!-resynth!NONE"
mv results_* fresh_results/rq1/ibm_old/queso

run_quartz ${benchmark%.qasm} $timeout "ibm" "ibm_old_bfs" "False"
mv results_* fresh_results/rq1/ibm_old/quartz

############################## RQ1 IBMN (IBM-Eagle) ##############################
BENCHMARK_DIR="benchmarks/ibmnew"
echo "Running RQ1 IBM-Eagle..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMN!--fidelity!40!--rules-dir!/home/queso_rules/!--resynth-weight!130"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq1/ibm_new/guoq

run_tket $BENCHMARK_DIR/$benchmark $timeout "ibm_new"
mv results_* fresh_results/rq1/ibm_new/tket 

run_qiskit $BENCHMARK_DIR/$benchmark $timeout "ibm_new"
mv results_* fresh_results/rq1/ibm_new/qiskit

run_bqskit $BENCHMARK_DIR/$benchmark $timeout "ibm_new"
mv results_* fresh_results/rq1/ibm_new/bqskit

run_queso $BENCHMARK_DIR/$benchmark $timeout "-g!IBMN!-opt!TOTAL!--rules-dir!/home/queso_rules/!-search!BEAM!-temp!0!-q!8000!-resynth!NONE"
mv results_* fresh_results/rq1/ibm_new/queso

run_quartz ${benchmark%.qasm} $timeout "ibm_new" "ibm_bfs" "False"
mv results_* fresh_results/rq1/ibm_new/quartz

# run_quarl
# mv results_* fresh_results/rq1/ibm_new/quarl

############################## RQ1 Ion ##############################
BENCHMARK_DIR="benchmarks/ion"
echo "Running RQ1 IonQ..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!ION!--fidelity!20!--rules-dir!/home/queso_rules/!--resynth-weight!22"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq1/ion/guoq

run_qiskit $BENCHMARK_DIR/$benchmark $timeout "ion"
mv results_* fresh_results/rq1/ion/qiskit

run_bqskit $BENCHMARK_DIR/$benchmark $timeout "ion"
mv results_* fresh_results/rq1/ion/bqskit

run_queso $BENCHMARK_DIR/$benchmark $timeout "-g!ION!-opt!TOTAL!--rules-dir!/home/queso_rules/!-search!BEAM!-temp!0!-q!8000!-resynth!NONE"
mv results_* fresh_results/rq1/ion/queso

############################## RQ1 Nam ##############################
BENCHMARK_DIR="benchmarks/nam_rz"
echo "Running RQ1 Nam..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!NAM!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!180"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq1/nam/guoq

run_tket $BENCHMARK_DIR/$benchmark $timeout "nam"
mv results_* fresh_results/rq1/nam/tket

run_voqc $BENCHMARK_DIR/$benchmark $timeout "nam"
mv results_* fresh_results/rq1/nam/voqc

run_qiskit $BENCHMARK_DIR/$benchmark $timeout "nam"
mv results_* fresh_results/rq1/nam/qiskit

run_bqskit $BENCHMARK_DIR/$benchmark $timeout "nam"
mv results_* fresh_results/rq1/nam/bqskit

run_queso $BENCHMARK_DIR/$benchmark $timeout "-g!NAM!-opt!TOTAL!--rules-dir!/home/queso_rules/!-search!BEAM!-temp!0!-q!8000!-resynth!NONE"
mv results_* fresh_results/rq1/nam/queso

run_quartz ${benchmark%.qasm} $timeout "nam" "nam2_bfs" "True"
mv results_* fresh_results/rq1/nam/quartz

# run_quarl
# mv results_* fresh_results/rq1/nam/quarl

############################## RQ2 ##############################
BENCHMARK_DIR="benchmarks/ibm"
echo "Running RQ2..."

for i in $(seq 1 $num_guoq_trials); do
    # used for RQ3 baseline as well
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMO!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!10"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq2/guoq

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Rewrite Only trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMO!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!10!-resynth!NONE"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq2/guoq-rewrite

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Resynth Only trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMO!-opt!TWO_Q!--resynth-weight!10!-r!/home/queso_rules/empty.txt!-sr!/home/queso_rules/empty.txt"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq2/guoq-resynth


############################# RQ3 ##############################
BENCHMARK_DIR="benchmarks/ibm"
echo "Running RQ3..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Beam trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!IBMO!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!10!-search!BEAM!-temp!0!-q!8000"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i "--bqskit"
done
mv results_* fresh_results/rq3/guoq-beam

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Rewrite Resynth trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args1="-g!IBMO!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!10!-resynth!NONE"
    guoq_args2="-g!IBMO!-opt!TWO_Q!--resynth-weight!10!-r!/home/queso_rules/empty.txt!-sr!/home/queso_rules/empty.txt"
    run_guoq_seq $BENCHMARK_DIR/$benchmark $timeout $guoq_args1 $guoq_args2 $i
done
mv results_* fresh_results/rq3/guoq-rewrite-resynth

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Resynth Rewrite trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args1="-g!IBMO!-opt!TWO_Q!--resynth-weight!10!-r!/home/queso_rules/empty.txt!-sr!/home/queso_rules/empty.txt"
    guoq_args2="-g!IBMO!-opt!TWO_Q!--rules-dir!/home/queso_rules/!--resynth-weight!10!-resynth!NONE"
    run_guoq_seq $BENCHMARK_DIR/$benchmark $timeout $guoq_args1 $guoq_args2 $i
done
mv results_* fresh_results/rq3/guoq-resynth-rewrite

############################# RQ4 ##############################
BENCHMARK_DIR="benchmarks/nam_t_tdg"
echo "Running RQ4..."

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!CLIFFORDT!-opt!FT!--rules-dir!/home/queso_rules/!--resynth-weight!622"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i ""
done
mv results_* fresh_results/rq4/guoq

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Rewrite Only trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!CLIFFORDT!-opt!FT!--rules-dir!/home/queso_rules/!--resynth-weight!622!-resynth!NONE"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i ""
done
mv results_* fresh_results/rq4/guoq-rewrite

for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ Resynth Only trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!CLIFFORDT!-opt!FT!--resynth-weight!622!-r!/home/queso_rules/empty.txt!-sr!/home/queso_rules/empty.txt"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i ""
done
mv results_* fresh_results/rq4/guoq-resynth

run_qiskit $BENCHMARK_DIR/$benchmark $timeout "nam_t_tdg"
mv results_* fresh_results/rq4/qiskit

run_bqskit $BENCHMARK_DIR/$benchmark $timeout "nam_t_tdg"
mv results_* fresh_results/rq4/bqskit

run_synthetiq $BENCHMARK_DIR/$benchmark $timeout
mv results_* fresh_results/rq4/synthetiq

run_queso $BENCHMARK_DIR/$benchmark $timeout "-g!CLIFFORDT!-opt!TOTAL!--rules-dir!/home/queso_rules/!-search!BEAM!-temp!0!-q!8000!-resynth!NONE"
mv results_* fresh_results/rq4/queso

run_pyzx $BENCHMARK_DIR/$benchmark $timeout
mv results_* fresh_results/rq4/pyzx

# GUOQ on output of PyZX
BENCHMARK_DIR="benchmarks/after_pyzx"
for i in $(seq 1 $num_guoq_trials); do
    echo "Running GUOQ on PyZX output trial $i out of $num_guoq_trials on $benchmark..."
    guoq_args="-g!CLIFFORDT!-opt!FT!--rules-dir!/home/queso_rules/!--resynth-weight!622"
    run_guoq $BENCHMARK_DIR/$benchmark $timeout $guoq_args $i ""
done
mv results_* fresh_results/rq4/guoq-after-pyzx
