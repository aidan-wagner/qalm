#!/bin/bash

run_guoq () {
    ulimit -v 33554432 # 32GB
    echo -n $3 > temp_args.txt
    out_dir=results_$(basename "$1" .qasm)
    mkdir -p results_$(basename "$1" .qasm)

    conda run --live-stream -n guoq python resynth.py $5 --bqskit_auto_workers &> $out_dir/resynth_log_none_${4}.txt &
    resynth_pid=$!
    tail -f $out_dir/resynth_log_none_${4}.txt | while read line; do
        if [[ "$line" == *"Resynthesis server is ready"* ]]; then
            break
        fi
    done
    sleep 10
    (conda run --live-stream -n guoq python run_guoq.py \
        --circuit_file $1 \
        --timeout $2 \
        --process $4 \
        --guoq_cli_args temp_args.txt) || echo 'ran out of memory or time'
    pkill -P $resynth_pid
    pkill -f resynth.py
    rm temp_args.txt
} 

run_guoq_seq () {
    ulimit -v 33554432 # 32GB
    echo -n $3 > temp_args1.txt
    echo -n $4 > temp_args2.txt
    out_dir=results_$(basename "$1" .qasm)
    mkdir -p results_$(basename "$1" .qasm)
    local benchmark=$(basename "$1" .qasm)
    touch $out_dir/resynth_log_none_${5}_first.txt

    conda run --live-stream -n guoq python resynth.py --bqskit --bqskit_auto_workers &> $out_dir/resynth_log_none_${5}_second.txt &
    resynth_pid=$!
    tail -f $out_dir/resynth_log_none_${5}_second.txt | while read line; do
        if [[ "$line" == *"Resynthesis server is ready"* ]]; then
            break
        fi
    done
    sleep 10
    (conda run --live-stream -n guoq python run_guoq.py \
        --circuit_file $1 \
        --timeout $(($2 / 2)) \
        --process ${5}_first \
        --guoq_cli_args temp_args1.txt) || echo 'ran out of memory or time'

    cp results_$benchmark/latest_sol_*${5}_first_${benchmark}.qasm results_$benchmark/${benchmark}.qasm 

    (conda run --live-stream -n guoq python run_guoq.py \
        --circuit_file results_$benchmark/${benchmark}.qasm \
        --timeout $(($2 / 2)) \
        --process ${5}_second \
        --guoq_cli_args temp_args2.txt) || echo 'ran out of memory or time'

    pkill -P $resynth_pid
    pkill -f resynth.py
    rm temp_args1.txt
    rm temp_args2.txt
    rm results_$benchmark/*first*json
}

run_queso () {
    echo "Running QUESO on $1..."
    ulimit -v 33554432 # 32GB
    echo -n $3 > temp_args.txt
    out_dir=results_$(basename "$1" .qasm)
    mkdir -p results_$(basename "$1" .qasm)
    touch $out_dir/resynth_log_none_none.txt
    (conda run --live-stream -n guoq python run_guoq.py \
        --circuit_file $1 \
        --timeout $2 \
        --guoq_cli_args temp_args.txt) || echo 'ran out of memory or time'
}

run_quartz () {
    echo "Running Quartz on $1..."
    ulimit -v 33554432 # 32GB
    export ROTATION_MERGE_ON=$5
    cd /home/quartz/experiment/ppo-new
    (conda run --live-stream -n quartz timeout $2 python scripts/$4.py $1 results_quartz_${3}_${1}) || echo 'ran out of memory or time'
    cd /home
    mv /home/quartz/experiment/ppo-new/results_quartz_${3}_${1} .
}

run_qiskit () {
    echo "Running Qiskit on $1..."
    ulimit -v 33554432 # 32GB
    (conda run --live-stream -n qiskit timeout $2 python run_qiskit.py \
        --circuit_file $1 \
        --gate_set $3 \
        --optimization_level 3) || echo 'ran out of memory or time'
} 

run_bqskit () {
    echo "Running BQSKit on $1..."
    ulimit -v 33554432 # 32GB
    (conda run --live-stream -n bqskit timeout $2 python run_bqskit.py \
        --circuit_file $1 \
        --gate_set $3 \
        --optimization_level 4 \
        --error_threshold 1e-08 \
        --block_size 3) || echo 'ran out of memory or time'
} 

run_tket () {
    echo "Running TKET on $1..."
    ulimit -v 33554432 # 32GB
    mkdir -p tket_config
    export XDG_CONFIG_HOME=tket_config
    (conda run --live-stream -n tket timeout $2 python run_tket.py \
        --circuit_file $1 \
        --gate_set $3) || echo 'ran out of memory or time'
} 

run_voqc () {
    echo "Running VOQC on $1..."
    ulimit -v 33554432 # 32GB
    (conda run --live-stream -n voqc timeout $2 python run_voqc.py \
        --circuit_file $1 \
        --gate_set $3) || echo 'ran out of memory or time'
} 

run_synthetiq () {
    echo "Running Synthetiq on $1..."
    ulimit -v 33554432 # 32GB
    (conda run --live-stream -n guoq timeout $2 python run_synthetiq.py \
        --circuit_file $1 \
        --error_threshold 1e-08 \
        --block_size 3 \
        --num_circuits 100 \
        --threads 8 \
        --path_to_synthetiq /home/lib/synthetiq) || echo 'ran out of memory or time'
} 

run_pyzx () {
    echo "Running PyZX on $1..."
    ulimit -v 33554432 # 32GB
    (conda run --live-stream -n pyzx timeout $2 python run_pyzx.py \
        --circuit_file $1) || echo 'ran out of memory or time'
} 