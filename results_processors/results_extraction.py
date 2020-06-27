from __future__ import print_function

import argparse
import itertools

from results_extraction_utils import get_filtered_datasets, load_results_pipelines, declare_winners, \
    summarize_winners, save_summary
from utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of the input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output

def main():
    # configure environment
    input, result_path = "../results/evaluation1", "../results"
    result_path = create_directory(create_directory(result_path, "summary"), "evaluation1")
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results = load_results_pipelines(input, filtered_data_sets)
    #print(results)

    winners = declare_winners(results)
    #print(winners)

    summary = summarize_winners(winners)
    #print(summary)

    save_summary(summary, result_path)


main()