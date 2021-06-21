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
                    pipeline = data['context']['best_config']['pipeline']
                    for transformation in ["encode", "features", "impute", "normalize", "discretize", "rebalance"]:
                        try:
                            results_map = results_map.append(pd.DataFrame({
                                "algorithm": [algorithm],
                                "dataset": [dataset],
                                "transformation": [pipeline[transformation][0].split("_")[0]],
                                "operator": [pipeline[transformation][0].split("_")[1]]
                            }), ignore_index=True)
                        except:
                            pass


    results_map.to_csv(os.path.join(result_path, "pp_pipeline_operator_study.csv"), index=False)

    result = results_map.groupby(['algorithm', 'transformation', 'operator']).count()
    result_sum = result.groupby(['algorithm', 'transformation']).sum()
    for algorithm in result.index.get_level_values('algorithm').unique():
        for transformation in result.index.get_level_values('transformation').unique():
            for operator in result.index.get_level_values('operator').unique():
                try:
                    result.loc[algorithm, transformation, operator] /= result_sum.loc[algorithm, transformation]
                except:
                    pass
    result = result.reset_index()
    result = result[result['operator'] != 'NoneType']
    result = result.set_index(['transformation', 'operator', 'algorithm'])
    print(result)
    
    labels = ["NB", "KNN", "RF"]
    colors = ['mediumpurple', 'xkcd:dark grass green', 'xkcd:kermit green', 'xkcd:lime green', 'xkcd:light pea green', 'xkcd:dark coral', 'xkcd:salmon', 'xkcd:sun yellow', 'xkcd:straw', 'xkcd:aqua green', 'xkcd:light aquamarine', 'xkcd:pumpkin orange', 'xkcd:apricot', 'xkcd:light peach']
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
    column = -1
    i = 0
    cumulative = False
    last_transformation, last_operator = '', ''
    for transformation in ['encode', 'normalize', 'discretize', 'impute', 'rebalance', 'features']:
        for operator in result.index.get_level_values('operator').unique():
            try:
                result.loc[transformation, operator]
                flag = True
            except:
                flag = False
            if flag:
                curr_bar = result.loc[transformation, operator].to_numpy().flatten()
                curr_bar = [curr_bar[1], curr_bar[0], curr_bar[2]]
                if transformation != last_transformation or last_transformation == '':
                    column += 1
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width, color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                else:
                    if not(cumulative):
                        last_bar = result.loc[last_transformation, last_operator].to_numpy().flatten()
                        last_bar = [last_bar[1], last_bar[0], last_bar[2]]
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width, bottom=last_bar, color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                if transformation == last_transformation:
                    cumulative = True
                    last_bar = [curr_bar[0] + last_bar[0], curr_bar[1] + last_bar[1], curr_bar[2] + last_bar[2]]
                else:
                    cumulative = False
                last_transformation, last_operator = transformation, operator
                i += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Frequency')
    
    ax.set_yticks(np.linspace(0, 1, 11))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 3, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(result_path, "pp_pipeline_study_grouped.pdf"), bbox_extra_artists=(lgd,text), bbox_inches='tight')
    
main()