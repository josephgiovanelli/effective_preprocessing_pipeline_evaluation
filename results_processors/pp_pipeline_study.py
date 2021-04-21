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
        #df = df[(df["pa_percentage"] == 0.5)]
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

                    encode_flag = 1 if "None" not in pipeline["encode"][0] else 0
                    features_flag = 1 if "None" not in pipeline["features"][0] else 0
                    impute_flag = 1 if "None" not in pipeline["impute"][0] else 0
                    try:
                        normalize_flag = 1 if "None" not in pipeline["normalize"][0] else 0
                    except:
                        normalize_flag = 0
                    try:
                        discretize_flag = 1 if "None" not in pipeline["discretize"][0] else 0
                    except:
                        discretize_flag = 0
                    rebalance_flag = 1 if "None" not in pipeline["rebalance"][0] else 0

                    results_map = results_map.append(pd.DataFrame({
                        "algorithm": [algorithm],
                        "dataset": [dataset],
                        "encode": [encode_flag],
                        "features": [features_flag],
                        "impute": [impute_flag],
                        "normalize": [normalize_flag],
                        "discretize": [discretize_flag],
                        "rebalance": [rebalance_flag]
                    }), ignore_index=True)

        results_map.to_csv(os.path.join(result_path, "pp_pipeline_study.csv"), index=False)

        result = results_map.groupby(['algorithm']).sum()
        result = result.reset_index()
        result = result.drop(['dataset'], axis=1)
        result.to_csv(os.path.join(result_path, "pp_pipeline_study_grouped.csv"), index=False)


main()