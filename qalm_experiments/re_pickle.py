import pickle
import sys
from enum import Enum

class OptimizationType(Enum):
    roqc_interval = 1
    qalm = 2
    quartz = 3


def merge(pickle_folder_1, pickle_folder_2, circuit_count):
    # Graph_labels
    with open(pickle_folder_1 + "/graph_labels.pkl", "rb") as gl1:
        graph_labels_1 = pickle.load(gl1)
    with open(pickle_folder_2 + "/graph_labels.pkl", "rb") as gl2:
        graph_labels_2 = pickle.load(gl2)

    graph_lables_full = graph_labels_1 + graph_labels_2

    with open ("combined_graph_labels.pkl", "wb") as gl_full:
        pickle.dump(graph_lables_full, gl_full)

    # Experiment Parameters
    with open(pickle_folder_1 + "/exp_params.pkl", "rb") as ep1:
        exp_params_1 = pickle.load(ep1)
    with open(pickle_folder_2 + "/exp_params.pkl", "rb") as ep2:
        exp_params_2 = pickle.load(ep2)

    exp_params_full = exp_params_1 + exp_params_2

    with open ("combined_exp_params.pkl", "wb") as ep_full:
        pickle.dump(exp_params_full, ep_full)

    #Results
    with open(pickle_folder_1 + "/raw_results.pkl", "rb") as rr1:
        raw_results_1 = pickle.load(rr1)
    with open(pickle_folder_2 + "/raw_results.pkl", "rb") as rr2:
        raw_results_2 = pickle.load(rr2)

    raw_results_full = []

    for c_index in range(circuit_count):
        for i in range(len(exp_params_1)):
            raw_results_full.append(raw_results_1[i + (c_index*len(exp_params_1))])
        for j in range(len(exp_params_2)):
            raw_results_full.append(raw_results_2[j + (c_index*len(exp_params_2))])

    with open ("combined_raw_results.pkl", "wb") as rr_full:
        pickle.dump(raw_results_full, rr_full)

if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2], 28)
