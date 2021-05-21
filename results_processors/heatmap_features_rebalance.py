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

    baseline_results = {}
    ## Results Loading
    for algorithm in algorithms:
        baseline_results[algorithm] = pd.read_csv(os.path.join(input, algorithm + '.csv'))
        baseline_results[algorithm].rename(columns={"dataset": "ID"}, inplace=True)

    results_map = {}
    
    for algorithm in algorithms:
        results_map[algorithm] = pd.DataFrame({
                        "FR": ["NoneType", "PCA", "SelectKBest", "FeatureUnion"], 
                        "NoneType": [0, 0, 0, 0],
                        "NearMiss": [0, 0, 0, 0],
                        "SMOTE": [0, 0, 0, 0]
                        })
        results_map[algorithm] = results_map[algorithm].set_index('FR')
        print(results_map[algorithm])
    for algorithm in algorithms:

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

                    features_flag = pipeline["features"][0].split("_",1)[1]
                    rebalance_flag = pipeline["rebalance"][0].split("_",1)[1]

                    results_map[algorithm].loc[features_flag, rebalance_flag] += 1

    ## Data Saving
    meta_learning_input_path = create_directory(result_path, "insight")
    for algorithm in algorithms:
        results_map[algorithm].to_csv(os.path.join(meta_learning_input_path, 'heatmap_features_rebalance_' + algorithm + '.csv'), index=True)
    
main()