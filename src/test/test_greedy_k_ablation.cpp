/**
 * test_greedy_k_ablation.cpp
 *
 * Ablation study binary for greedy_k.
 *
 * Arguments:
 *   argv[1]  input QASM file path
 *   argv[2]  circuit name (for logging)
 *   argv[3]  timeout (seconds)
 *   argv[4]  greedy_k  (0..3)
 *             0 = no greedy preprocessing; QALM starts with exploration_steps=1
 *             1 = greedy(k=1) [greedy_optimize_with_roqc], then QALM k=2
 *             2 = greedy(k=1..2), then QALM k=3
 *             3 = greedy(k=1..3), then QALM k=4
 *   argv[5]  initial_pool_size   (N_pool for QALM)
 *   argv[6]  exploration_pool_size (N_branch for QALM)
 *   argv[7]  repeat_tolerance
 *   argv[8]  exploration_increase  (0 or 1)
 *   argv[9]  strictly_reducing_rules / no_increase  (0 or 1)
 *   argv[10] only_do_local_transformations  (0 or 1)
 *   argv[11] two_way_rotation_merging  (0 or 1)
 *   argv[12] ECC set path (relative to repo root)
 *   argv[13] (optional) fixed_k — if >0, run QALM with this fixed
 *            exploration depth (no k-advancement).  0 or omitted = advancing.
 *   argv[14] (optional) enqueue_intermediate — 0 to skip enqueuing
 *            intermediate exploration results (only enqueue after ROQC).
 *            1 or omitted = enqueue all (default).
 */

#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"
#include "utils/utils.h"

using namespace quartz;

int main(int argc, char **argv) {
  assert(argc >= 13);

  std::string input_fn = std::string(argv[1]);
  std::string circuit_name = std::string(argv[2]);
  std::size_t timeout = std::stoi(argv[3]);
  int greedy_k = std::stoi(argv[4]);
  std::size_t initial_pool_size = std::stoi(argv[5]);
  std::size_t exploration_pool_size = std::stoi(argv[6]);
  float repeat_tolerance = std::stof(argv[7]);
  bool exploration_increase = std::stoi(argv[8]);
  bool strictly_reducing_rules = std::stoi(argv[9]);
  bool only_do_local_transformations = std::stoi(argv[10]);
  bool two_way_rotation_merging = std::stoi(argv[11]);
  std::string eqset_fn =
      kQuartzRootPath.string() + "/" + std::string(argv[12]);

  // Optional: fixed exploration k (0 = advancing, >0 = fixed at this depth)
  int fixed_k = 0;
  if (argc >= 14) {
    fixed_k = std::stoi(argv[13]);
  }

  // Optional: enqueue intermediate exploration results (default true)
  bool enqueue_intermediate = true;
  if (argc >= 15) {
    enqueue_intermediate = std::stoi(argv[14]);
  }

  // QALM starts exploration at greedy_k+1 (unless fixed_k overrides)
  std::size_t exploration_steps = (std::size_t)(greedy_k + 1);

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

  // Build xfers
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
  } else {
    xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
  }
  std::cout << "Number of xfers: " << xfers.size() << std::endl;

  // Load graph
  std::shared_ptr<Graph> graph;
  std::shared_ptr<Graph> g;
  std::unique_ptr<Context> src_ctx;
  std::unique_ptr<Context> union_ctx;
  if (input_fn.find(std::string("nam-benchmarks")) != input_fn.npos) {
    src_ctx = std::make_unique<Context>(
        Context({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                 GateType::input_qubit, GateType::input_param},
                &param_info));
    union_ctx = std::make_unique<Context>(union_contexts(src_ctx.get(), &ctx));
    auto xfer_pair =
        GraphXfer::ccz_cx_rz_xfer(src_ctx.get(), &ctx, union_ctx.get());
    QASMParser qasm_parser(src_ctx.get());
    CircuitSeq *dag = nullptr;
    if (!qasm_parser.load_qasm(input_fn, dag)) {
      std::cout << "Parser failed" << std::endl;
    }
    g = std::make_shared<Graph>(src_ctx.get(), dag);
    graph =
        g->toffoli_flip_greedy(GateType::rz, xfer_pair.first, xfer_pair.second);
  } else {
    graph = Graph::from_qasm_file(&ctx, input_fn);
  }
  assert(graph);

  auto start = std::chrono::steady_clock::now();

  // All greedy phases use all xfers (not strictly reducing)
  const auto &greedy_xfers =
      strictly_reducing_rules ? GraphXfer::get_all_xfers_from_eqs(&ctx, eqs)
                              : xfers;

  // Run greedy phases 1 through greedy_k
  if (greedy_k >= 1) {
    std::cout << "Running greedy(k=1): greedy_optimize_with_roqc" << std::endl;
    graph = graph->greedy_optimize_with_roqc(
        &ctx, greedy_xfers, circuit_name, "", /*print_message=*/true,
        nullptr, timeout, "", start);
  }
  if (greedy_k >= 2) {
    std::cout << "Running greedy(k=2): greedy_optimize_with_local_search"
              << std::endl;
    graph = graph->greedy_optimize_with_local_search(
        &ctx, greedy_xfers, circuit_name, "", /*print_message=*/true,
        nullptr, timeout, "", /*continue_storing_all_steps=*/false, start);
  }
  // greedy_k >= 3 is not yet implemented
  // (greedy_optimize_with_deeper_local_search does not exist)

  // QALM phase.
  // fixed_k > 0  → run once with that exploration depth, then stop.
  // fixed_k == 0 → advancing mode: start at greedy_k+1, increment each iter.
  if (fixed_k > 0) {
    std::size_t k = (std::size_t)fixed_k;
    printf("[%s] [QALM_K_START] k=%zu at %.3f seconds.\n",
           circuit_name.c_str(), k,
           (double)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start)
                   .count() /
               1000.0);
    fflush(stdout);
    graph = graph->optimize_qalm(
        xfers, graph->gate_count() * 1.05, circuit_name, "",
        /*print_message=*/true, nullptr, timeout, "",
        /*continue_storing_all_steps=*/false, start, initial_pool_size,
        exploration_pool_size, k, repeat_tolerance, exploration_increase,
        only_do_local_transformations, two_way_rotation_merging,
        enqueue_intermediate);
  } else {
    std::size_t k = exploration_steps;
    while (true) {
      auto now = std::chrono::steady_clock::now();
      double elapsed =
          (double)std::chrono::duration_cast<std::chrono::milliseconds>(
              now - start)
              .count() /
          1000.0;
      if (elapsed >= (double)timeout) {
        break;
      }
      printf("[%s] [QALM_K_START] k=%zu at %.3f seconds.\n",
             circuit_name.c_str(), k, elapsed);
      fflush(stdout);

      auto new_graph = graph->optimize_qalm(
          xfers, graph->gate_count() * 1.05, circuit_name, "",
          /*print_message=*/true, nullptr, timeout, "",
          /*continue_storing_all_steps=*/false, start, initial_pool_size,
          exploration_pool_size, k, repeat_tolerance, exploration_increase,
          only_do_local_transformations, two_way_rotation_merging,
          enqueue_intermediate);
      graph = new_graph;
      k++;
    }
  }

  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph->to_qasm();
  return 0;
}
