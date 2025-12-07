FROM amxu/guoq:1

WORKDIR /home

RUN apt-get update && apt-get install git

RUN rm -r /home/quartz

# RUN git clone https://github.com/aidan-wagner/qoalm.git

# COPY qualm_circuits.txt start_qualm_bench.sh run_on_qualm_circuits.sh /home/

# COPY testing_qualm_light.txt testing_qualm_bench.sh /home/

COPY . /home/qalm

RUN mv qalm/qalm_experiments/run_qalm_bench.sh /home/
RUN mv qalm/qalm_experiments/qalm_bench_test.sh /home/
RUN mv qalm/qalm_experiments/qalm_bench_full.sh /home/
RUN mv qalm/qalm_experiments/qalm_circuits_test.txt /home/
RUN mv qalm/qalm_experiments/qalm_circuits_full.txt /home/
RUN mv qalm/qalm_experiments/compile_results.py /home/

RUN mv qalm/qalm_experiments/circuits_0.txt /home/
RUN mv qalm/qalm_experiments/circuits_1.txt /home/
RUN mv qalm/qalm_experiments/circuits_2.txt /home/
RUN mv qalm/qalm_experiments/circuits_3.txt /home/
RUN mv qalm/qalm_experiments/circuits_4.txt /home/
RUN mv qalm/qalm_experiments/circuits_5.txt /home/
RUN mv qalm/qalm_experiments/circuits_6.txt /home/
RUN mv qalm/qalm_experiments/circuits_small.txt /home/

RUN mv qalm/qalm_experiments/bench_0.sh /home/
RUN mv qalm/qalm_experiments/bench_1.sh /home/
RUN mv qalm/qalm_experiments/bench_2.sh /home/
RUN mv qalm/qalm_experiments/bench_3.sh /home/
RUN mv qalm/qalm_experiments/bench_4.sh /home/
RUN mv qalm/qalm_experiments/bench_5.sh /home/
RUN mv qalm/qalm_experiments/bench_6.sh /home/
RUN mv qalm/qalm_experiments/bench_small.sh /home/

# Enable Executables
RUN chmod +x run_qalm_bench.sh
RUN chmod +x qalm_bench_test.sh
RUN chmod +x qalm_bench_full.sh

RUN chmod +x bench_0.sh
RUN chmod +x bench_1.sh
RUN chmod +x bench_2.sh
RUN chmod +x bench_3.sh
RUN chmod +x bench_4.sh
RUN chmod +x bench_5.sh
RUN chmod +x bench_6.sh
RUN chmod +x bench_small.sh

# Install Rust - Doesn't work
# RUN curl -y --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python Dependencies
RUN pip install matplotlib qiskit
RUN pip install pybind11[global]
