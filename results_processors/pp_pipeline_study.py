from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import create_directory

def main():
    # configure environment
    input, result_path = "../results/summary/evaluation3", "../results/summary/evaluation3"
    input_auto = "../results/evaluation3/preprocessing_algorithm"
    result_path = create_directory(result_path, "extension")

    discretize_count, normalize_count = {}, {}
    results_map = pd.DataFrame()
    for algorithm in ["nb", "knn", "rf"]:
        discretize_count[algorithm], normalize_count[algorithm] = 0, 0
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
                        normalize_count[algorithm] += 1
                    except:
                        normalize_flag = 0
                    try:
                        discretize_flag = 1 if "None" not in pipeline["discretize"][0] else 0
                        discretize_count[algorithm] += 1
                    except:
                        discretize_flag = 0
                    rebalance_flag = 1 if "None" not in pipeline["rebalance"][0] else 0

                    results_map = results_map.append(pd.DataFrame({
                        "algorithm": [algorithm],
                        "dataset": [dataset],
                        "E": [encode_flag],
                        "N": [normalize_flag],
                        "D": [discretize_flag],
                        "I": [impute_flag],
                        "R": [rebalance_flag],
                        "F": [features_flag]
                    }), ignore_index=True)

    results_map.to_csv(os.path.join(result_path, "pp_pipeline_study.csv"), index=False)

    result = results_map.groupby(['algorithm']).sum()
    result = result.reset_index()
    result = result.drop(['dataset'], axis=1)
    result = result.set_index(["algorithm"])
    result = result.reindex(["nb", "knn", "rf"])

    for algorithm in ["nb", "knn", "rf"]:
        result.loc[algorithm,'N'] /= normalize_count[algorithm]
        result.loc[algorithm,'D'] /= discretize_count[algorithm]
    result = result.reset_index()
    result['E'] = result['E'] / 80
    result['I'] = result['I'] / 80
    result['R'] = result['R'] / 80
    result['F'] = result['F'] / 80
    print(normalize_count, discretize_count)
    result.to_csv(os.path.join(result_path, "pp_pipeline_study_grouped.csv"), index=False)
    result = result.rename(columns={
        'E' : r'$E$',
        'I' : r'$I$',
        'N' : r'$N$',
        'D' : r'$D$',
        'F' : r'$F$',
        'R' : r'$R$',
    })
    labels = [x.upper() for x in result["algorithm"]]
    colors = ['mediumpurple', 'greenyellow', 'lightcoral', 'gold', 'mediumturquoise', 'orange']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    SMALL_SIZE = 8
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 21

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    rects = {}
    columns = list(result.columns)[1:]
    for column in range(len(columns)):
        ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, result[columns[column]], width, color=colors[column], label=columns[column])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    
    
    ax.set_yticks(np.linspace(0, 1, 11))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(result_path, "pp_pipeline_study_grouped.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')

main()