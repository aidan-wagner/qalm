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
  std::string eqset_fn = kQuartzRootPath.string() + "/";
  eqset_fn = eqset_fn + argv[7];

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
  int roqc_interval = std::stoi(argv[4]);
  bool preprocess = std::stoi(argv[5]);
  bool two_way_rotation_merging = std::stoi(argv[6]);

  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::rx,
               GateType::ry, GateType::add},
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
    // Load qasm file
    QASMParser qasm_parser(src_ctx.get());
    CircuitSeq *dag = nullptr;
    if (!qasm_parser.load_qasm(input_fn, dag)) {
      std::cout << "Parser failed" << std::endl;
    }
    g = std::make_shared<Graph>(src_ctx.get(), dag);

    // Greedy toffoli flip
    graph =
        g->toffoli_flip_greedy(GateType::rz, xfer_pair.first, xfer_pair.second);
  } else {
    graph = Graph::from_qasm_file(&ctx, input_fn);
  }
  std::cout << "Circuit retrieved from qasm file" << std::endl;
  assert(graph);

  auto start = std::chrono::steady_clock::now();
  if (preprocess) {
    graph = graph->greedy_optimize_with_roqc(
        &ctx, xfers, circuit_name, "", true, nullptr, timeout,
        kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name +
            "_timeout_" + std::to_string(timeout) + "_roqc_interval_" +
            std::to_string(roqc_interval) + "_",
        start);
    graph = graph->greedy_optimize_with_local_search(
        &ctx, xfers, circuit_name, "", true, nullptr, timeout,
        kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name +
            "_timeout_" + std::to_string(timeout) + "_roqc_interval_" +
            std::to_string(roqc_interval) + "_",
        true, start);
  }

  auto graph_optimized = graph->optimize(
      xfers, graph->gate_count() * 1.05, circuit_name, "", true, nullptr,
      timeout,
      kQuartzRootPath.string() + "/benchmark-logs/" + circuit_name +
          "_timeout_" + std::to_string(timeout) + "_roqc_interval_" +
          std::to_string(roqc_interval) + "_",
      preprocess, start, roqc_interval, two_way_rotation_merging);
  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
