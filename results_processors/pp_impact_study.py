from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools

import pandas as pd

from utils import create_directory

def main():
    # configure environment
    input, result_path = "../results/summary/evaluation3", "../results/summary/evaluation3"
    input_auto = "../results/evaluation3/preprocessing_algorithm"
    result_path = create_directory(result_path, "extension")

    results_map = pd.DataFrame()
    for algorithm in ["knn", "nb", "rf"]:

        df = pd.read_csv(os.path.join("..", "results", "summary", "evaluation3", algorithm + ".csv"))
        df = df[(df["pa_percentage"] <= 0.5)]
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    pipeline = data['context']['best_config']['pipeline']

                    encode_flag = "None" in pipeline["encode"][0]
                    features_flag = "None" in pipeline["features"][0]
                    impute_flag = "None" in pipeline["impute"][0]
                    try:
                        normalize_flag = "None" in pipeline["normalize"][0]
                    except:
                        normalize_flag = True
                    try:
                        discretize_flag = "None" in pipeline["discretize"][0]
                    except:
                        discretize_flag = True
                    rebalance_flag = "None" in pipeline["rebalance"][0]

                    results_map = results_map.append(pd.DataFrame({
                        "algorithm": [algorithm],
                        "dataset": [dataset],
                        "encode_flag": [encode_flag],
                        "features_flag": [features_flag],
                        "impute_flag": [impute_flag],
                        "normalize_flag": [normalize_flag],
                        "discretize_flag": [discretize_flag],
                        "rebalance_flag": [rebalance_flag],
                        "all_none": [encode_flag and features_flag and impute_flag and normalize_flag and discretize_flag and rebalance_flag]
                    }), ignore_index=True)

        results_map.to_csv(os.path.join(result_path, "pp_impact_study.csv"), index=False)

main()