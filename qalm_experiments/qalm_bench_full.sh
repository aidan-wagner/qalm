#!/bin/bash

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 600 1
done < qalm_circuits_full.txt

# while read line; do
#     echo $line
#     ./run_qalm_bench.sh $line 900 1
# done < qalm_circuits_full.txt
# 
# while read line; do
#     echo $line
#     ./run_qalm_bench.sh $line 1800 1
# done < qalm_circuits_full.txt
# 
# while read line; do
#     echo $line
#     ./run_qalm_bench.sh $line 2700 1
# done < qalm_circuits_full.txt
# 
# while read line; do
#     echo $line
#     ./run_qalm_bench.sh $line 3600 1
# done < qalm_circuits_full.txt
