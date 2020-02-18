from __future__ import print_function

import argparse

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
    filtered_data_sets = ['rf_11', 'rf_46', 'knn_40984', 'rf_50', 'rf_1475', 'rf_31', 'rf_1480', 'rf_1068', 'rf_3',
                          'nb_1067', 'knn_46', 'nb_1461', 'knn_11', 'knn_1485', 'nb_16', 'knn_50', 'rf_1053', 'nb_1590',
                          'rf_41027', 'nb_1494', 'rf_40499', 'nb_1050', 'nb_40984', 'rf_1049', 'knn_31', 'knn_1050',
                          'rf_37', 'rf_40979', 'rf_1486', 'knn_1590', 'nb_11', 'knn_16', 'knn_1494', 'nb_50', 'nb_1489',
                          'rf_40975', 'nb_40983', 'nb_31', 'knn_40668', 'knn_1461', 'knn_40994', 'knn_6', 'knn_40701',
                          'knn_40982', 'nb_1485', 'rf_40670', 'nb_3', 'nb_40668', 'knn_40983', 'rf_16', 'rf_1464',
                          'knn_1067', 'rf_4534', 'nb_40982', 'nb_40701', 'nb_40994', 'knn_37', 'rf_1468', 'rf_1487',
                          'rf_4538', 'rf_1501', 'nb_1464', 'nb_1063', 'knn_14', 'knn_40975', 'knn_1480', 'knn_1068',
                          'nb_44', 'rf_40668', 'nb_469', 'rf_40994', 'rf_182', 'rf_40982', 'rf_23', 'knn_1510',
                          'rf_23517', 'knn_40670', 'nb_4534', 'knn_458', 'nb_29', 'nb_1468', 'nb_1487', 'rf_54',
                          'knn_182', 'knn_1475', 'knn_18', 'knn_40979', 'rf_458', 'rf_15', 'knn_22', 'nb_4538',
                          'nb_40979', 'knn_1049', 'knn_23', 'nb_32', 'rf_1489', 'rf_14', 'knn_1462', 'knn_300',
                          'nb_1486', 'nb_28', 'nb_307', 'nb_40975', 'rf_22', 'rf_40983', 'knn_1053', 'rf_300', 'rf_18',
                          'rf_1485', 'knn_54', 'nb_12', 'nb_40670', 'nb_151', 'rf_6', 'knn_1497', 'knn_15', 'nb_6',
                          'knn_307', 'rf_1461', 'knn_151', 'nb_1053', 'rf_29', 'nb_23', 'knn_32', 'nb_1462', 'rf_1494',
                          'nb_41027', 'nb_54', 'rf_151', 'knn_12', 'rf_1590', 'nb_15', 'knn_28', 'rf_307', 'nb_40499',
                          'rf_40984', 'rf_1050', 'nb_1049', 'nb_300', 'nb_182', 'rf_469', 'knn_41027', 'knn_4538',
                          'rf_32', 'knn_29', 'knn_1501', 'knn_40499', 'nb_1475', 'nb_14', 'knn_1468', 'knn_1487',
                          'knn_44', 'nb_22', 'nb_1510', 'knn_4534', 'knn_1063', 'rf_12', 'knn_1464', 'rf_1067', 'nb_18',
                          'knn_3', 'knn_469', 'nb_1068', 'nb_458', 'nb_1480']


    auto_results = load_results(input_auto, filtered_data_sets)
    algorithm_results = load_results(input_algorithm, filtered_data_sets)

    comparison, summary = merge_results(auto_results, algorithm_results, 'algorithm', filtered_data_sets)
    save_comparison(comparison, create_directory(result_path, "algorithm_auto"))
    save_summary(summary, create_directory(result_path, "algorithm_auto"))

main()