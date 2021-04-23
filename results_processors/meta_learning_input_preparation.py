from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools
import copy

import pandas as pd

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
    
    ## Results Loading
    df = {}
    for algorithm in algorithms:
        df[algorithm] = pd.read_csv(os.path.join(input, algorithm + '.csv'))
        df[algorithm].rename(columns={"dataset": "ID"}, inplace=True)

    ## Data Preparation
    for algorithm in algorithms:
        df[algorithm] = pd.merge(df[algorithm], meta_features, on="ID")

    ## Data Saving
    meta_learning_input_path = create_directory(result_path, "meta_learning")
    meta_learning_input_path = create_directory(meta_learning_input_path, "input")
    for algorithm in algorithms:
        df[algorithm].to_csv(os.path.join(meta_learning_input_path, algorithm + '_raw.csv'), index=False)

    for pa_impact in [True, False]:
        data = copy.deepcopy(df)

        ## Data Preparation
        for algorithm in algorithms:
            if pa_impact:
                data[algorithm]["pa_impact"] = data[algorithm]["pipeline_algorithm"] - data[algorithm]["baseline"]
                data[algorithm].drop(['pa_percentage'], axis=1, inplace=True)
            data[algorithm].drop(['pipeline_algorithm', 'algorithm', 'a_score', 'pa_score', 'a_percentage'], axis=1, inplace=True)

        ## Data Saving
        for algorithm in algorithms:
            data[algorithm].to_csv(os.path.join(meta_learning_input_path, algorithm + ('_pa_impact' if pa_impact else '') + '.csv'), index=False)

        ## Data Preparation
        benchmarking_features = ['CfsSubsetEval_NaiveBayesErrRate', 'NaiveBayesErrRate', 'RandomTreeDepth2AUC', 'RandomTreeDepth3Kappa', 'REPTreeDepth2AUC', 'J48.001.ErrRate', 'CfsSubsetEval_DecisionStumpKappa', 'J48.0001.ErrRate', 'J48.00001.Kappa', 'RandomTreeDepth2ErrRate', 'REPTreeDepth2ErrRate', 'CfsSubsetEval_NaiveBayesKappa', 'RandomTreeDepth2Kappa', 'RandomTreeDepth3ErrRate', 'REPTreeDepth1AUC', 'REPTreeDepth1ErrRate', 'REPTreeDepth2Kappa', 'NaiveBayesKappa', 'CfsSubsetEval_NaiveBayesAUC', 'REPTreeDepth3Kappa', 'kNN1NAUC', 'J48.001.Kappa', 'DecisionStumpErrRate', 'DecisionStumpAUC', 'J48.00001.AUC', 'REPTreeDepth1Kappa', 'RandomTreeDepth3AUC', 'REPTreeDepth3AUC', 'J48.0001.Kappa', 'CfsSubsetEval_DecisionStumpErrRate', 'J48.00001.ErrRate', 'RandomTreeDepth1Kappa', 'RandomTreeDepth1AUC', 'CfsSubsetEval_kNN1NErrRate', 'DecisionStumpKappa', 'kNN1NErrRate', 'kNN1NKappa', 'J48.001.AUC', 'CfsSubsetEval_kNN1NAUC', 'NaiveBayesAUC', 'CfsSubsetEval_DecisionStumpAUC', 'RandomTreeDepth1ErrRate', 'CfsSubsetEval_kNN1NKappa', 'REPTreeDepth3ErrRate', 'J48.0001.AUC']
        manual_fs_data = data.copy()
        for algorithm in algorithms:
            manual_fs_data[algorithm].drop(benchmarking_features, axis=1, inplace=True)

        ## Data Saving
        for algorithm in algorithms:
            manual_fs_data[algorithm].to_csv(os.path.join(meta_learning_input_path, 'manual_fs_' + algorithm + ('_pa_impact' if pa_impact else '') + '.csv'), index=False)

        ## Data Preparation
        for algorithm in algorithms:
            data[algorithm]["algorithm"] = algorithm
            manual_fs_data[algorithm]["algorithm"] = algorithm
        union = pd.concat([data["knn"], data["nb"], data["rf"]], ignore_index=True)
        manual_fs_union = pd.concat([manual_fs_data["knn"], manual_fs_data["nb"], manual_fs_data["rf"]], ignore_index=True)

        ## Data Saving
        union.to_csv(os.path.join(meta_learning_input_path, 'union' + ('_pa_impact' if pa_impact else '') + '.csv'), index=False)
        manual_fs_union.to_csv(os.path.join(meta_learning_input_path, 'manual_fs_union' + ('_pa_impact' if pa_impact else '') + '.csv'), index=False)
    
main()