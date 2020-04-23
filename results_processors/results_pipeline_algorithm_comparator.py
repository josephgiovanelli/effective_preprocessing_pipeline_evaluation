from __future__ import print_function

import argparse
import itertools

from results_processors.results_extraction_utils import get_filtered_datasets, load_results, merge_results, \
    save_comparison, save_summary, merge_all_results
from results_processors.utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-iau", "--input_auto", nargs="?", type=str, required=True, help="path of first input")
    parser.add_argument("-ial", "--input_algorithm", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_auto, args.input_algorithm, args.output

def main():
    # configure environment
    input_auto, input_algorithm, result_path = parse_args()
    result_path = create_directory(result_path, "summary")
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    print(filtered_data_sets)

    auto_results = load_results(input_auto, filtered_data_sets, algorithm_comparison = True)
    algorithm_results = load_results(input_algorithm, filtered_data_sets, algorithm_comparison = True)

    comparison, summary = merge_results(auto_results, algorithm_results, 'algorithm', filtered_data_sets, algorithm_comparison = True)
    save_comparison(comparison, create_directory(result_path, "pipeline_algorithm_vs_algorithm"))
    save_summary(summary, create_directory(result_path, "pipeline_algorithm_vs_algorithm"))

main()