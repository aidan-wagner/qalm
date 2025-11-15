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
  std::size_t initial_pool_size = std::stoi(argv[4]);
  std::size_t exploration_pool_size = std::stoi(argv[5]);
  std::size_t exploration_steps = std::stoi(argv[6]);
  float repeat_tolerance = std::stof(argv[7]);
  bool exploration_increase = std::stoi(argv[8]);
  bool strictly_reducing_rules = std::stoi(argv[9]);
  bool only_keep_distant_circuits = std::stoi(argv[10]);

  bool preprocess = std::stoi(argv[11]);


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

  std::vector<GraphXfer *> xfers;

  if (strictly_reducing_rules) {

    auto cost_function = [](Graph *graph) { return graph->total_cost(); };

    // Get xfers that strictly reduce the cost from the ECC set
    auto eccs = eqs.get_all_equivalence_sets();
    for (const auto &ecc : eccs) {
      const int ecc_size = (int)ecc.size();
      std::vector<Graph> graphs;
      std::vector<int> graph_cost;
      graphs.reserve(ecc_size);
      graph_cost.reserve(ecc_size);
      for (auto &circuit : ecc) {
        graphs.emplace_back(&ctx, circuit);
        graph_cost.emplace_back(cost_function(&graphs.back()));
      }
      int representative_id =
          (int)(std::min_element(graph_cost.begin(), graph_cost.end()) -
                graph_cost.begin());
      for (int i = 0; i < ecc_size; i++) {
        if (graph_cost[i] != graph_cost[representative_id]) {
          auto xfer = GraphXfer::create_GraphXfer(&ctx, ecc[i],
                                                  ecc[representative_id], true);
          if (xfer != nullptr) {
            xfers.push_back(xfer);
          }
        } else if (i != representative_id) {
          auto xfer =
              GraphXfer::create_GraphXfer(&ctx, ecc[i], ecc[representative_id], true);
          if (xfer != nullptr) {
            xfers.push_back(xfer);
          }
          xfer = GraphXfer::create_GraphXfer(&ctx, ecc[representative_id], ecc[i], true);
          if (xfer != nullptr) {
            xfers.push_back(xfer);
          }
        }
      }
    }
    std::cout << "Number of xfers that reduce or maintain cost: "
              << xfers.size() << std::endl;
  }
  else {

  // Get xfer from the equivalent set
    std::cout << "Trying to load eccset" << std::endl;
    xfers =
        GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
    std::cout << "number of xfers: " << xfers.size() << std::endl;
  }

  auto graph = Graph::from_qasm_file(&ctx, input_fn);
  std::cout << "Recieved circuit from qasm file" << std::endl;
  assert(graph);

  if (preprocess) {
    graph = graph->greedy_optimize(&ctx, eqs, true, nullptr, kQuartzRootPath.string()
                                              + "/benchmark-logs/" + circuit_name
                                              + "_timeout_" + std::to_string(timeout)
                                              + "_init_pool_" + std::to_string(initial_pool_size)
                                              + "_exp_pool_" + std::to_string(exploration_pool_size)
                                              + "_exp_steps_" + std::to_string(exploration_steps) 
                                              + "_exp_increase_" + std::to_string(exploration_increase)
                                              + "_no_increase_" + std::to_string(strictly_reducing_rules)
                                              + "_only_keep_distant_circuits_" + std::to_string(only_keep_distant_circuits) + "_");
  }


  // auto graph_optimized = graph->optimize_qalm(xfers,
  //                                             graph->gate_count() * 1.05,
  //                                             circuit_name,
  //                                             "",
  //                                             false,
  //                                             nullptr,
  //                                             timeout,
  //                                             kQuartzRootPath.string()
  //                                             + "/benchmark-logs/" + circuit_name
  //                                             + "_timeout_" + std::to_string(timeout)
  //                                             + "_init_pool_" + std::to_string(initial_pool_size)
  //                                             + "_exp_pool_" + std::to_string(exploration_pool_size)
  //                                             + "_exp_steps_" + std::to_string(exploration_steps) 
  //                                             + "_exp_increase_" + std::to_string(exploration_increase)
  //                                             + "_no_increase_" + std::to_string(strictly_reducing_rules)
  //                                             + "_only_keep_distant_circuits_" + std::to_string(only_keep_distant_circuits) + "_",
  //                                             true,
  //                                             initial_pool_size,
  //                                             exploration_pool_size,
  //                                             exploration_steps,
  //                                             repeat_tolerance,
  //                                             exploration_increase,
  //                                             only_keep_distant_circuits);
  auto graph_optimized = graph->optimize_qalm(xfers,
                                              graph->gate_count() * 1.05,
                                              circuit_name,
                                              "",
                                              true,
                                              nullptr,
                                              timeout,
                                              "",
                                              true,
                                              initial_pool_size,
                                              exploration_pool_size,
                                              exploration_steps,
                                              repeat_tolerance,
                                              exploration_increase,
                                              only_keep_distant_circuits);
  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
