from __future__ import print_function

import argparse
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt


from results_extraction_utils import get_filtered_datasets, load_results_pipelines, load_results_auto, \
     get_winners_accuracy, save_comparison
from utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-ip", "--input_pipelines", nargs="?", type=str, required=True, help="path of the first input")
    parser.add_argument("-ia", "--input_auto", nargs="?", type=str, required=True, help="path of the second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_pipelines, args.input_auto, args.output

def main():
    # configure environment
    input_pipelines, input_auto, result_path = "../results/evaluation1", "../results/evaluation3/preprocessing_algorithm", "../results"
    result_path = create_directory(create_directory(result_path, "summary"), "evaluation2")
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results_pipelines = load_results_pipelines(input_pipelines, filtered_data_sets)
    #print(results_pipelines)
    results_auto = load_results_auto(input_auto, filtered_data_sets)
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            if results_auto[algorithm][dataset][1] == 0:
                evaluation_3 = pd.read_csv("../results/summary/evaluation3/" + algorithm + ".csv")
                evaluation_3 = evaluation_3.set_index(['dataset'])
                results_auto[algorithm][dataset] = (results_auto[algorithm][dataset][0], evaluation_3.loc[int(dataset)]["baseline"])
    impacts = pd.DataFrame()
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            try:
                for elem in results_pipelines[algorithm][dataset]:
                    impacts = impacts.append({'algorithm': algorithm, 'dataset': dataset, 'index': int(elem['index']), 'impact': elem['accuracy']-results_auto[algorithm][dataset][1]}, ignore_index=True)
            except:
                print(dataset)
    print(impacts)
    knn = impacts[impacts["algorithm"] == "knn"]
    nb = impacts[impacts["algorithm"] == "nb"]
    rf = impacts[impacts["algorithm"] == "rf"]
    knn = knn.pivot(index = 'dataset', columns='index', values = 'impact')
    nb = nb.pivot(index = 'dataset', columns='index', values = 'impact')
    rf = rf.pivot(index = 'dataset', columns='index', values = 'impact')
    knnBoxplot = plt.figure()
    knnBp = knn.boxplot(column=list(range(24)))
    knnBoxplot.savefig(os.path.join(result_path, "knn_box_plot.pdf"), format="pdf")
    nbBoxplot = plt.figure()
    nbBp = nb.boxplot(column=list(range(24)))
    nbBoxplot.savefig(os.path.join(result_path, "nb_box_plot.pdf"), format="pdf")
    rfBoxplot = plt.figure()
    rfBp = rf.boxplot(column=list(range(24)))
    rfBoxplot.savefig(os.path.join(result_path, "rf_box_plot.pdf"), format="pdf")
    #save_comparison(results_pipelines, results_auto, result_path)


main()