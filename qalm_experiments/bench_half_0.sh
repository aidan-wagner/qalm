#!/bin/bash

while read line; do
    echo $line
    ./run_qalm_bench.sh $line 3600 1
done < circuits_half_0.txt
