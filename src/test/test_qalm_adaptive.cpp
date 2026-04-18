#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"
#include "utils/utils.h"

using namespace quartz;

// Usage:
//   ./build/test_qalm_adaptive <circuit> <name> <timeout>
//       <base_pool> <base_branch> <base_k>
//       <repeat_tolerance>
//       <strictly_reducing_rules> <only_do_local_transformations>
//       <greedy_start> <two_way_rm>
//       <eccset>
//       <stall_window_s> <max_pool> <max_branch> <max_k>
//
// Mirrors test_qalm.cpp except:
//   - no exploration_increase arg (superseded by adaptive logic)
//   - four additional args at the end: stall_window_s, max_pool, max_branch, max_k

int main(int argc, char **argv) {
  std::string input_fn =
      kQuartzRootPath.string() + "/circuit/nam_circs/adder_8.qasm";
  std::string circuit_name = "adder_8";
  std::string eqset_fn     = kQuartzRootPath.string() + "/";
  eqset_fn = eqset_fn + argv[12];

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

  std::size_t timeout              = std::stoi(argv[3]);
  std::size_t initial_pool_size    = std::stoi(argv[4]);
  std::size_t exploration_pool_size = std::stoi(argv[5]);
  std::size_t exploration_steps    = std::stoi(argv[6]);
  float       repeat_tolerance     = std::stof(argv[7]);
  bool strictly_reducing_rules         = std::stoi(argv[8]);
  bool only_do_local_transformations   = std::stoi(argv[9]);
  bool preprocess                      = std::stoi(argv[10]);
  bool two_way_rotation_merging        = std::stoi(argv[11]);
  // argv[12] is eccset (read above); adaptive params start at argv[13]
  double stall_window_seconds = std::stod(argv[13]);
  std::size_t max_pool_size   = std::stoi(argv[14]);
  std::size_t max_branch_size = std::stoi(argv[15]);
  std::size_t max_steps       = std::stoi(argv[16]);

  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::rx,
               GateType::ry, GateType::add},
              /*num_qubits=*/3, &param_info);

  EquivalenceSet eqs;
  if (!eqs.load_json(&ctx, eqset_fn, /*from_verifier=*/false)) {
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
        kQuartzRootPath.string() + "/eccset/Nam_5_3_", true, false, 3, 2, 5);
    if (!eqs.load_json(&ctx, eqset_fn, /*from_verifier=*/false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
  }

  std::vector<GraphXfer *> xfers;

  if (strictly_reducing_rules) {
    auto cost_function = [](Graph *graph) { return graph->total_cost(); };
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
          if (xfer != nullptr) xfers.push_back(xfer);
        } else if (i != representative_id) {
          auto xfer = GraphXfer::create_GraphXfer(&ctx, ecc[i],
                                                  ecc[representative_id], true);
          if (xfer != nullptr) xfers.push_back(xfer);
          xfer = GraphXfer::create_GraphXfer(&ctx, ecc[representative_id],
                                             ecc[i], true);
          if (xfer != nullptr) xfers.push_back(xfer);
        }
      }
    }
    std::cout << "Number of xfers that reduce or maintain cost: "
              << xfers.size() << std::endl;
  } else {
    std::cout << "Trying to load eccset" << std::endl;
    xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
    std::cout << "number of xfers: " << xfers.size() << std::endl;
  }

  std::shared_ptr<Graph> graph;
  std::shared_ptr<Graph> g;
  std::unique_ptr<Context> src_ctx;
  std::unique_ptr<Context> union_ctx;
  if (input_fn.find(std::string("nam-benchmarks")) != input_fn.npos) {
    src_ctx = std::make_unique<Context>(
        Context({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                 GateType::input_qubit, GateType::input_param},
                &param_info));
    union_ctx =
        std::make_unique<Context>(union_contexts(src_ctx.get(), &ctx));
    auto xfer_pair =
        GraphXfer::ccz_cx_rz_xfer(src_ctx.get(), &ctx, union_ctx.get());
    QASMParser qasm_parser(src_ctx.get());
    CircuitSeq *dag = nullptr;
    if (!qasm_parser.load_qasm(input_fn, dag)) {
      std::cout << "Parser failed" << std::endl;
    }
    g     = std::make_shared<Graph>(src_ctx.get(), dag);
    graph = g->toffoli_flip_greedy(GateType::rz, xfer_pair.first,
                                   xfer_pair.second);
  } else {
    graph = Graph::from_qasm_file(&ctx, input_fn);
  }
  std::cout << "Received circuit from qasm file" << std::endl;
  assert(graph);

  auto start = std::chrono::steady_clock::now();
  if (preprocess) {
    const auto &greedy_xfers =
        strictly_reducing_rules ? GraphXfer::get_all_xfers_from_eqs(&ctx, eqs)
                                : xfers;
    graph = graph->greedy_optimize_with_roqc(
        &ctx, greedy_xfers, circuit_name, "", true, nullptr, timeout,
        kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name +
            "_adaptive_",
        start);
    graph = graph->greedy_optimize_with_local_search(
        &ctx, greedy_xfers, circuit_name, "", true, nullptr, timeout,
        kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name +
            "_adaptive_",
        true, start);
  }

  auto graph_optimized = graph->optimize_qalm_adaptive(
      xfers, graph->gate_count() * 1.05, circuit_name, "", true, nullptr,
      timeout, "", preprocess, start,
      initial_pool_size, exploration_pool_size, exploration_steps,
      repeat_tolerance, only_do_local_transformations, two_way_rotation_merging,
      stall_window_seconds, max_pool_size, max_branch_size, max_steps);

  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
