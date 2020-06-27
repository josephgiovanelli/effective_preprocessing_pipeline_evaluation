from __future__ import print_function

import itertools

from results_extraction_utils import get_filtered_datasets, load_results, merge_results, \
    save_comparison, save_summary, plot_comparison
from utils import create_directory

def main():
    # configure environment
    input_auto, input_algorithm, result_path = "../results/evaluation3/preprocessing_algorithm", "../results/evaluation3/algorithm", "../results/evaluation3"
    result_path = create_directory(result_path, "summary")
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]

    auto_results = load_results(input_auto, filtered_data_sets, algorithm_comparison = True)
    algorithm_results = load_results(input_algorithm, filtered_data_sets, algorithm_comparison = True)

    comparison, summary = merge_results(auto_results, algorithm_results, 'algorithm', filtered_data_sets)
    #print(comparison)
    save_comparison(comparison, create_directory(result_path, "pipeline_algorithm_vs_algorithm"))
    save_summary(summary, create_directory(result_path, "pipeline_algorithm_vs_algorithm"))
    plot_comparison(comparison, create_directory(result_path, "pipeline_algorithm_vs_algorithm"))

main()