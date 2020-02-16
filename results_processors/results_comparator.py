from __future__ import print_function

import argparse

from results_processors.results_extraction_utils import get_filtered_datasets, load_results, merge_results, \
    save_comparison, save_summary, merge_all_results
from results_processors.utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-ia", "--input_auto", nargs="?", type=str, required=True, help="path of first input")
    parser.add_argument("-iq", "--input_quemy", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-ip", "--input_pseudo_exhaustive", nargs="?", type=str, required=True, help="path of third input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_auto, args.input_quemy, args.input_pseudo_exhaustive, args.output

def main():
    # configure environment
    input_auto, input_quemy, input_pseudo_exhaustive, result_path = parse_args()
    result_path = create_directory(result_path, "summary")
    filtered_data_sets = get_filtered_datasets()


    auto_results = load_results(input_auto, filtered_data_sets)
    quemy_results = load_results(input_quemy, filtered_data_sets)
    pseudo_exhaustive_results = load_results(input_pseudo_exhaustive, filtered_data_sets)

    print('quemy_auto')
    print()

    comparison, summary = merge_results(quemy_results, auto_results, 'quemy', 'auto')
    save_comparison(comparison, create_directory(result_path, "quemy_auto"))
    save_summary(summary, create_directory(result_path, "quemy_auto"))

    print()
    print()
    print('auto_pseudoexhaustive')
    print()

    comparison, summary = merge_results(auto_results, pseudo_exhaustive_results, 'auto', 'pseudo_exhaustive')
    save_comparison(comparison, create_directory(result_path, "auto_pseudoexhaustive"))
    save_summary(summary, create_directory(result_path, "auto_pseudoexhaustive"))

    print()
    print()
    print('all_together')
    print()

    comparison, summary = merge_all_results(quemy_results, auto_results, pseudo_exhaustive_results, 'quemy', 'auto', 'pseudo_exhaustive')
    save_comparison(comparison, create_directory(result_path, "all_together"))
    save_summary(summary, create_directory(result_path, "all_together"))

main()