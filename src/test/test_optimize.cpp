#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"
#include "utils/utils.h"

using namespace quartz;

int main(int argc, char **argv) {
  std::string input_fn =
      kQuartzRootPath.string() + "/circuit/nam_circs/adder_8.qasm";

  std::cout << kQuartzRootPath.string() << " " << input_fn << std::endl;
  std::string circuit_name = "adder_8";
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
    assert(argv[4] != nullptr);

    std::size_t timeout = std::stoi(argv[3]);
    std::size_t roqc_interval = std::stoi(argv[4]);
    bool preprocess = std::stoi(argv[5]);

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
        kQuartzRootPath.string() + "/eccset/Nam_5_3_", true, false, 3, 2, 5);
    if (!eqs.load_json(&ctx, eqset_fn,
                       /*from_verifier=*/false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
  }

  // Get xfer from the equivalent set
  std::vector<GraphXfer *> xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  auto seq = CircuitSeq::from_qasm_file(&ctx, input_fn);
  auto graph = std::make_shared<Graph>(&ctx, seq.get());
  std::cout << "got" << std::endl;
  assert(graph);

  if (preprocess) {
    graph = graph->greedy_optimize(&ctx, eqs, true, nullptr, kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name + "_timeout_" + std::to_string(timeout) + "_roqc_interval_" + std::to_string(roqc_interval) + "_");
  }

  auto graph_optimized = graph->optimize(xfers, graph->gate_count() * 1.05,
                                         circuit_name, "", true, nullptr, timeout, kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name + "_timeout_" + std::to_string(timeout) + "_roqc_interval_" + std::to_string(roqc_interval) + "_", false, roqc_interval);
  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
