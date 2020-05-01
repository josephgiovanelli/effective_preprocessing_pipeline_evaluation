from __future__ import print_function

import argparse
import itertools

from results_processors.results_extraction_utils import get_filtered_datasets, load_results_pipelines, load_results_auto, \
     get_winners_accuracy, save_comparison
from results_processors.utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-ip", "--input_pipelines", nargs="?", type=str, required=True, help="path of the first input")
    parser.add_argument("-ia", "--input_auto", nargs="?", type=str, required=True, help="path of the second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_pipelines, args.input_auto, args.output

def main():
    # configure environment
    input_pipelines, input_auto, result_path = parse_args()
    result_path = create_directory(result_path, "pipelines_comparison")
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results_pipelines = load_results_pipelines(input_pipelines, filtered_data_sets)
    results_pipelines = get_winners_accuracy(results_pipelines)
    results_auto = load_results_auto(input_auto, filtered_data_sets)
    #print(results_pipelines)
    #print(results_auto)

    save_comparison(results_pipelines, results_auto, result_path)


main()