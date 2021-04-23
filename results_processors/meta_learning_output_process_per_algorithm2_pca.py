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

    ## Meta-features Loading
    meta_features = {}
    for algorithm in algorithms:
        meta_features[algorithm] = pd.read_csv(os.path.join(input, "extension", "meta_learning", "output", algorithm + "_pca.csv"))
        meta_features[algorithm] = meta_features[algorithm].drop(['baseline'], axis=1)
    
    ## Results Loading
    mf_data = {}
    for algorithm in algorithms:
        mf_data[algorithm] = pd.read_csv(os.path.join(input, algorithm + '.csv'))
        mf_data[algorithm].rename(columns={"dataset": "ID"}, inplace=True)

    ## Data Preparation
    for algorithm in algorithms:
        mf_data[algorithm] = pd.merge(mf_data[algorithm], meta_features[algorithm], on="ID")

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

                    dataset_meta_features = mf_data[algorithm][(mf_data[algorithm]['ID'] == dataset)].iloc[0]
                    if algorithm == "knn":
                        if dataset_meta_features["baseline"] <= 79.22:
                            if dataset_meta_features["PC5"] <= 2758.098:
                                if dataset_meta_features["PC4"] <= -625.574:
                                    leaf = "knn_4"
                                else:
                                    if dataset_meta_features["PC2"] <= 9923.364:
                                        leaf = "knn_6"
                                    else:
                                        leaf = "knn_7"
                            else:
                                leaf = "knn_8"
                        else:
                            if dataset_meta_features["PC4"] <= -499.952:
                                leaf = "knn_10"
                            else:
                                leaf = "knn_11"
                    elif algorithm == "nb":
                        if dataset_meta_features["baseline"] <= 92.44:
                            if dataset_meta_features["PC5"] <= 1185.705:
                                leaf = "nb_3"
                            else:
                                if dataset_meta_features["PC2"] <= 9955.667:
                                    leaf = "nb_5"
                                else:
                                    leaf = "nb_6"
                        else:
                            leaf = "nb_7"
                    else:
                        if dataset_meta_features["PC3"] <= -36140.67:
                            leaf = "rf_2"
                        else:
                            if dataset_meta_features["PC4"] <= -522.044:
                                leaf = "rf_4"
                            else:
                                if dataset_meta_features["PC2"] <= 9902.933:
                                    if dataset_meta_features["PC2"] <= 9542.96:
                                        leaf = "rf_7"
                                    else:
                                        leaf = "rf_8"
                                else:
                                    leaf = "rf_9"
                    results_map = results_map.append(pd.DataFrame({"leaf": [leaf], "pipeline": [pipeline_conf]}), ignore_index=True)

    results_map.to_csv(os.path.join(result_path, "meta_learning_output_process2_per_algorithm_pca.csv"), index=False)
    results_map = results_map.pivot_table(index='leaf', columns='pipeline', aggfunc=len, fill_value=0)
    results_map["sum"] = results_map.sum(axis=1)
    results_map = results_map.div(results_map["sum"], axis=0)
    results_map = results_map.drop(['sum'], axis=1)
    results_map = results_map.reset_index()
    results_map.to_csv(os.path.join(result_path, "meta_learning_output_process2_per_algorithm_pca_pivoted.csv"), index=False)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    results_map[(results_map["leaf"].str.startswith('knn'))].plot(ax=axes[0], kind='bar', x='leaf')
    results_map[(results_map["leaf"].str.startswith('nb'))].plot(ax=axes[1], kind='bar', x='leaf')
    results_map[(results_map["leaf"].str.startswith('rf'))].plot(ax=axes[2], kind='bar', x='leaf')
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axes[0].transAxes)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(h_pad=3.0, w_pad=4.0)
    fig.savefig(os.path.join(result_path, "meta_learning_output_process2_per_algorithm_pca.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')


main()