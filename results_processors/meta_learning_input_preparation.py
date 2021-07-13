from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import create_directory

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def main():
    # configure environment
    input, result_path = "../results/summary/evaluation3", "../results/summary/evaluation3"
    input_auto = "../results/evaluation3/preprocessing_algorithm"
    result_path = create_directory(result_path, "extension")
    algorithms = ["knn", "nb", "rf"]

    ## Meta-features Loading
    all_classification = pd.read_csv('meta_features/extended_meta_features_all_classification.csv')
    openml_cc_18 = pd.read_csv('meta_features/extended_meta_features_openml_cc_18.csv')
    study_1 = pd.read_csv('meta_features/extended_meta_features_study_1.csv')
    all_classification = all_classification[all_classification['ID'].isin(diff(all_classification['ID'], openml_cc_18['ID']))]
    meta_features = pd.concat([openml_cc_18, all_classification, study_1], ignore_index=True, sort=True)

    baseline_results = {}
    ## Results Loading
    for algorithm in algorithms:
        baseline_results[algorithm] = pd.read_csv(os.path.join(input, algorithm + '.csv'))
        baseline_results[algorithm].rename(columns={"dataset": "ID"}, inplace=True)

    results_map = {}
    for algorithm in algorithms:
        results_map[algorithm] = pd.DataFrame()

        df = pd.read_csv(os.path.join("..", "results", "summary", "evaluation3", algorithm + ".csv"))
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    pipeline = data['context']['best_config']['pipeline']

                    encode_flag = pipeline["encode"][0].split("_",1)[1]
                    features_flag = pipeline["features"][0].split("_",1)[1]
                    impute_flag = pipeline["impute"][0].split("_",1)[1]
                    try:
                        normalize_flag = pipeline["normalize"][0].split("_",1)[1]
                    except:
                        normalize_flag = "NoneType"
                    try:
                        discretize_flag = pipeline["discretize"][0].split("_",1)[1]
                    except:
                        discretize_flag = "NoneType"
                    rebalance_flag = pipeline["rebalance"][0].split("_",1)[1]

                    results_map[algorithm] = results_map[algorithm].append(pd.DataFrame({
                        "ID": [dataset], 
                        "baseline": [baseline_results[algorithm].loc[baseline_results[algorithm]["ID"] == dataset, "baseline"].iloc[0]],
                        "encode": [encode_flag],
                        "features": [features_flag],
                        "impute": [impute_flag],
                        "normalize": [normalize_flag],
                        "discretize": [discretize_flag],
                        "rebalance": [rebalance_flag]
                        }), ignore_index=True)

    for algorithm in algorithms:
        results_map[algorithm] = pd.merge(results_map[algorithm], meta_features, on="ID")

    ## Data Saving
    meta_learning_input_path = create_directory(result_path, "operator_meta_learning")
    meta_learning_input_path = create_directory(meta_learning_input_path, "input")
    #for algorithm in algorithms:
    #    results_map[algorithm].to_csv(os.path.join(meta_learning_input_path, algorithm + '_raw.csv'), index=False)

    data = copy.deepcopy(results_map)

    ## Data Preparation
    benchmarking_features = ['CfsSubsetEval_NaiveBayesErrRate', 'NaiveBayesErrRate', 'RandomTreeDepth2AUC', 'RandomTreeDepth3Kappa', 'REPTreeDepth2AUC', 'J48.001.ErrRate', 'CfsSubsetEval_DecisionStumpKappa', 'J48.0001.ErrRate', 'J48.00001.Kappa', 'RandomTreeDepth2ErrRate', 'REPTreeDepth2ErrRate', 'CfsSubsetEval_NaiveBayesKappa', 'RandomTreeDepth2Kappa', 'RandomTreeDepth3ErrRate', 'REPTreeDepth1AUC', 'REPTreeDepth1ErrRate', 'REPTreeDepth2Kappa', 'NaiveBayesKappa', 'CfsSubsetEval_NaiveBayesAUC', 'REPTreeDepth3Kappa', 'kNN1NAUC', 'J48.001.Kappa', 'DecisionStumpErrRate', 'DecisionStumpAUC', 'J48.00001.AUC', 'REPTreeDepth1Kappa', 'RandomTreeDepth3AUC', 'REPTreeDepth3AUC', 'J48.0001.Kappa', 'CfsSubsetEval_DecisionStumpErrRate', 'J48.00001.ErrRate', 'RandomTreeDepth1Kappa', 'RandomTreeDepth1AUC', 'CfsSubsetEval_kNN1NErrRate', 'DecisionStumpKappa', 'kNN1NErrRate', 'kNN1NKappa', 'J48.001.AUC', 'CfsSubsetEval_kNN1NAUC', 'NaiveBayesAUC', 'CfsSubsetEval_DecisionStumpAUC', 'RandomTreeDepth1ErrRate', 'CfsSubsetEval_kNN1NKappa', 'REPTreeDepth3ErrRate', 'J48.0001.AUC']
    manual_fs_data = data.copy()
    #for algorithm in algorithms:
    #    manual_fs_data[algorithm].drop(benchmarking_features, axis=1, inplace=True)

    ## Data Saving
    #for algorithm in algorithms:
    #    manual_fs_data[algorithm].to_csv(os.path.join(meta_learning_input_path, 'manual_fs_' + algorithm + '.csv'), index=False)

    ## Data Preparation
    for algorithm in algorithms:
        data[algorithm]["algorithm"] = algorithm
        manual_fs_data[algorithm]["algorithm"] = algorithm
    union = pd.concat([data["knn"], data["nb"], data["rf"]], ignore_index=True)
    manual_fs_union = pd.concat([manual_fs_data["knn"], manual_fs_data["nb"], manual_fs_data["rf"]], ignore_index=True)

    ## Data Saving
    #union.to_csv(os.path.join(meta_learning_input_path, 'union' + '.csv'), index=False)
    manual_fs_union.to_csv(os.path.join(meta_learning_input_path, 'data' + '.csv'), index=False)
    
main()