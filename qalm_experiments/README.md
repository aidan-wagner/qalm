# Running Qalm Experiments

## Starting Docker Container

Ensure that you are in the qalm base directory

Build and Run the docker container:

`sudo docker build -t qalm_testing .`

`sudo docker run -it .`

### Install Rust:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Enter 1 for default install

### Install Other Dependencies

`pip install numpy qiskit matplotlib`

### Build Qalm

`cd qalm`

`mkdir build_docker`

`cd build_docker`

`cmake ..`

`make`

`cd ..`

## Running Qalm Benchmarks

Set the desired runtime in qalm_experiments/run_qalm_experiments.py
Select which circuit list you would like to run in qalm_experiments/run_qalm_experiments.py

`python qalm_experiments/run_qalm_experiments.py`

## Running Repeated ROQC Benchmarks

Run full benchmarks:

`python qalm_experiments/repeated_roqc_tests.py ../qalm_circuits_full.txt`

Run test benchmarks:

`python qalm_experiments/repeated_roqc_tests.py ../qalm_circuits_test.txt`

## Running Other Benchamrks

Begin in the home directory

To run small test:

`./qalm_bench_test.sh`

To run full test:

`./qalm_bench_full.sh`

To alter the timeout for either, edit line timeout in the script. Default value is 60 seconds

## Generating Result CSV

Ensure that you have run Qalm and Other benchmarks
Begin in home directory

Make sure the circuit list file in compile_results.py matches the tests you ran (test vs. full)

Compile the results:

`python compile_results.py`

After this, save the file qalm_bench_results.csv to your local machine

You can also save the entire fresh_results directory for use later

## Generating Result Plots

To generate the result plot, run:

`python graph_results.py`

In the same directory as the qalm_bench_results.csv file, and the result graph will be saved as qalm_bench_results.png
