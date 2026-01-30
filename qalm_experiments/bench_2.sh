#!/bin/bash

CIRC_FILE="circuits_2.txt"

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 600 1
done < $CIRC_FILE

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 900 1
done < $CIRC_FILE

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 1800 1
done < $CIRC_FILE

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 2700 1
done < $CIRC_FILE

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 3600 1
done < $CIRC_FILE
