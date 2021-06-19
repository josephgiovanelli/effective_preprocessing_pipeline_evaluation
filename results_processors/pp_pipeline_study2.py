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
    results_map = results_map.set_index(["algorithm"])
    results_map = results_map.reindex(["nb", "knn", "rf"])
    results_map = results_map.reset_index()
    results_map.to_csv(os.path.join(result_path, "pp_pipeline_study2_pivoted.csv"), index=False)
    results_map = results_map.rename(columns={
        'DF': r'$D \to F$',
        'RF': r'$R \to F$',
        'D': r'$D$',
        'N': r'$N$',
        'R': r'$R$',
        'DFR': r'$D \to F \to R$',
        'DR': r'$D \to R$',
        'DRF': r'$D \to R \to F$',
        'NFR': r'$N \to F \to R$',
        'NRF': r'$N \to R \to F$',
        'RD': r'$R \to D$',
        'DF': r'$D \to F$',
        'DR': r'$D \to R$',
        'FR': r'$F \to R$',
        'NF': r'$N \to F$',
        'NR': r'$N \to R$',
        'RDF': r'$R \to D \to F$',
    })
    print(results_map)
    labels = [x.upper() for x in results_map["algorithm"]]
    patterns = ['/', '\\', '-\\', '-', '+', 'x', 'o', '//', '\\\\', 'O.', '--', '++', 'xx', 'OO', '\\|']
    colors= ['skyblue', 'orange', 'lightgreen','tomato', 'mediumorchid', 'xkcd:medium brown', 'xkcd:pale pink', 'xkcd:greyish', 'xkcd:aquamarine', 'xkcd:dodger blue', 'xkcd:sun yellow', 'xkcd:flat green', 'xkcd:red orange', 'xkcd:lighter purple', 'xkcd:silver']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
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
    columns = list(results_map.columns)[2:]
    for column in range(len(columns)):
        ax.bar((x * width * 20) + (width * (column)), results_map[columns[column]], width, label=columns[column], color=colors[column], hatch=patterns[column])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x*2.5+0.88)
    ax.set_xticklabels(labels)
    #ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 5, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(result_path, "pp_pipeline_study2.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')

    '''
    fig, axes = plt.subplots(nrows=1, ncols=1)
    results_map.plot(ax=axes, kind='bar', x='algorithm')
    axes.get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axes.transAxes)
    fig.set_size_inches(10, 5, forward=True)
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, "pp_pipeline_study2.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')
    '''

main()