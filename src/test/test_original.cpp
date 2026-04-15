#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

#include <chrono>
#include <iostream>

using namespace quartz;

int main(int argc, char **argv) {
  std::string input_fn =
      kQuartzRootPath.string() + "/circuit/nam_circs/barenco_tof_3.qasm";
  std::string circuit_name = "barenco_tof_3";
  std::string output_fn;
  std::string eqset_fn =
      kQuartzRootPath.string() + "/eccset/Nam_5_3_complete_ECC_set.json";

  if (argc >= 2) {
    assert(argv[1] != nullptr);
    input_fn = std::string(argv[1]);
    if (argc >= 3) {
      assert(argv[2] != nullptr);
      circuit_name = std::string(argv[2]);
    }
  }

  assert(argv[3] != nullptr);

  std::size_t timeout = std::stoi(argv[3]);

  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              /*num_qubits=*/3, &param_info);

  EquivalenceSet eqs;
  // Load ECC set from file
  if (!eqs.load_json(&ctx, eqset_fn,
                     /*from_verifier=*/false)) {
    // generate ECC set
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
        kQuartzRootPath.string() + "/eccset/Nam_3_3_", true, false, 3, 2, 3);
    if (!eqs.load_json(&ctx, eqset_fn,
                       /*from_verifier=*/false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
  }

  // Get xfer from the equivalent set
  std::vector<GraphXfer *> xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  auto graph = Graph::from_qasm_file(&ctx, input_fn);
  assert(graph);

  auto start = std::chrono::steady_clock::now();
  auto graph_preprocessed = graph->greedy_optimize(
      graph->context, eqs, /*print_message=*/true, /*cost_function=*/nullptr,
      /*store_all_steps_file_prefix=*/"", (double)timeout, start);
  auto greedy_end = std::chrono::steady_clock::now();
  double greedy_seconds =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(greedy_end -
                                                                    start)
              .count() /
      1000.0;

  // Emit the post-greedy intermediate result immediately so that if
  // optimize_original later OOMs, the benchmark driver still has a valid
  // circuit + timing to report.
  std::cout << "Post-greedy time: " << greedy_seconds << " seconds"
            << std::endl;
  std::cout << "Post-greedy graph:" << std::endl;
  std::cout << graph_preprocessed->to_qasm();
  std::cout << "End post-greedy graph." << std::endl;
  std::cout.flush();

  auto graph_optimized = graph_preprocessed->optimize_original(
      xfers, graph_preprocessed->gate_count() * 1.05, circuit_name, "", true,
      nullptr, (double)timeout, "", /*continue_storing_all_steps=*/false,
      start);
  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
