#!/bin/bash

CIRC_FILE="circuits_small.txt"

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 3600 1
done < $CIRC_FILE
