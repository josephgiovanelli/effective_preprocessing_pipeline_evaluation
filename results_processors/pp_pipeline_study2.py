from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools

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

    results_map = pd.DataFrame()
    for algorithm in ["knn", "nb", "rf"]:

        df = pd.read_csv(os.path.join("..", "results", "summary", "evaluation3", algorithm + ".csv"))
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
                    pipeline_conf = data['pipeline']
                    pipeline_conf = ''.join([a[0] for a in pipeline_conf]).upper()

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
                    if encode_flag or not(encode_flag):
                        pipeline_conf = pipeline_conf.replace("E", "")
                    if features_flag:
                        pipeline_conf = pipeline_conf.replace("F", "")
                    if impute_flag or not(impute_flag):
                        pipeline_conf = pipeline_conf.replace("I", "")
                    if normalize_flag:
                        pipeline_conf = pipeline_conf.replace("N", "")
                    if discretize_flag:
                        pipeline_conf = pipeline_conf.replace("D", "")
                    if rebalance_flag:
                        pipeline_conf = pipeline_conf.replace("R", "")

                    
                    results_map = results_map.append(pd.DataFrame({"algorithm": [algorithm], "pipeline": [pipeline_conf]}), ignore_index=True)

    results_map.to_csv(os.path.join(result_path, "pp_pipeline_study2.csv"), index=False)
    results_map = results_map.pivot_table(index='algorithm', columns='pipeline', aggfunc=len, fill_value=0)
    results_map["sum"] = results_map.sum(axis=1)
    results_map = results_map.div(results_map["sum"], axis=0)
    results_map = results_map.drop(['sum'], axis=1)
    results_map = results_map.reset_index()
    results_map.to_csv(os.path.join(result_path, "pp_pipeline_study2_pivoted.csv"), index=False)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    results_map.plot(ax=axes, kind='bar', x='algorithm')
    axes.get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axes.transAxes)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(h_pad=3.0, w_pad=4.0)
    fig.savefig(os.path.join(result_path, "pp_pipeline_study2.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')


main()