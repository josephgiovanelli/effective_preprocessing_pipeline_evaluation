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

    results_map = pd.DataFrame({
        "leaf": ["knn_4", "knn_6", "knn_7", "knn_8", "knn_10", "knn_11", "nb_3", "nb_5", "nb_6", "nb_7", "rf_2", "rf_4", "rf_7", "rf_8", "rf_9"],
        #"encode": np.zeros(15),
        "features": np.zeros(15),
        #"impute": np.zeros(15),
        "normalize": np.zeros(15),
        "discretize": np.zeros(15),
        "rebalance": np.zeros(15)
        })
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

                    #print(mf_data)
                    dataset_meta_features = mf_data[algorithm][(mf_data[algorithm]['ID'] == dataset)].iloc[0]
                    #print(dataset_meta_features)
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
                    #results_map.loc[results_map["leaf"] == leaf, "encode"] = results_map.loc[results_map["leaf"] == leaf, "encode"] + encode_flag
                    results_map.loc[results_map["leaf"] == leaf, "features"] = results_map.loc[results_map["leaf"] == leaf, "features"] + features_flag
                    #results_map.loc[results_map["leaf"] == leaf, "impute"] = results_map.loc[results_map["leaf"] == leaf, "impute"] + impute_flag
                    results_map.loc[results_map["leaf"] == leaf, "normalize"] = results_map.loc[results_map["leaf"] == leaf, "normalize"] + normalize_flag
                    results_map.loc[results_map["leaf"] == leaf, "discretize"] = results_map.loc[results_map["leaf"] == leaf, "discretize"] + discretize_flag
                    results_map.loc[results_map["leaf"] == leaf, "rebalance"] = results_map.loc[results_map["leaf"] == leaf, "rebalance"] + rebalance_flag

    results_map.to_csv(os.path.join(result_path, "meta_learning_output_process_per_algorithm_pca.csv"), index=False)

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
    fig.savefig(os.path.join(result_path, "meta_learning_output_process_per_algorithm_pca.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')

main()