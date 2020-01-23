from __future__ import print_function

import argparse

from results_processors.results_extraction_utils import get_filtered_datasets, load_results, merge_results, \
    save_comparison, save_summary
from results_processors.utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-ia", "--input_auto", nargs="?", type=str, required=True, help="path of first input")
    parser.add_argument("-iq", "--input_quemy", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_auto, args.input_quemy, args.output

def main():
    # configure environment
    input_auto, input_quemy, result_path = parse_args()
    result_path = create_directory(result_path, "summary")
    filtered_data_sets = get_filtered_datasets()


    auto_results = load_results(input_auto, filtered_data_sets)
    quemy_results = load_results(input_quemy, filtered_data_sets)

    print(auto_results)
    print(quemy_results)

    comparison, summary = merge_results(auto_results, quemy_results)

    save_comparison(comparison, result_path)
    save_summary(summary, result_path)

main()