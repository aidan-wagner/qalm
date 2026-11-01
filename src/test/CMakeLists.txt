include_directories(${CMAKE_INCLUDE_PATH})
include_directories("..")
link_directories("../..")

# ECC set generator (eccset/Nam_5_3_complete_ECC_set.json is shipped; this regenerates it)
add_executable(gen_ecc_set gen_ecc_set.cpp)
# QALM: greedy mode, then the advancing-k (or fixed-k) search (configuration a and the ablations)
add_executable(test_greedy_k_ablation test_greedy_k_ablation.cpp)
# QALM configuration b: greedy k <= 2, then a single fixed-k search pass
add_executable(test_qalm test_qalm.cpp)
# Quartz baseline (greedy pass, then Quartz's backtracking search)
add_executable(test_original test_original.cpp)
