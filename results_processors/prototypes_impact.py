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
    #print(results_auto)
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            if results_auto[algorithm][dataset][1] == 0:
                evaluation_3 = pd.read_csv("../results/summary/evaluation3/" + algorithm + ".csv")
                evaluation_3 = evaluation_3.set_index(['dataset'])
                results_auto[algorithm][dataset] = (results_auto[algorithm][dataset][0], evaluation_3.loc[int(dataset)]["baseline"])
    #print(results_auto)
    impacts = pd.DataFrame()
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            try:
                for elem in results_pipelines[algorithm][dataset]:
                    impacts = impacts.append({'algorithm': algorithm, 'dataset': dataset, 'index': int(elem['index']) + 1, 'impact': elem['accuracy']-results_auto[algorithm][dataset][1]}, ignore_index=True)
            except:
                print(dataset)
    #print(impacts)
    knn = impacts[impacts["algorithm"] == "knn"]
    nb = impacts[impacts["algorithm"] == "nb"]
    rf = impacts[impacts["algorithm"] == "rf"]
    knn = knn.pivot(index = 'dataset', columns='index', values = 'impact')
    nb = nb.pivot(index = 'dataset', columns='index', values = 'impact')
    rf = rf.pivot(index = 'dataset', columns='index', values = 'impact')
    data = [nb, knn, rf]
    fig, ax = plt.subplots(1, 3, sharey= True, constrained_layout=True)
    for i in range(len(data)):
        ax[i].boxplot(data[i], showfliers=False)
        ax[i].set_title("NB" if i == 0 else ("KNN" if i == 1 else "RF"))
        ax[i].set_xlabel('Prototypes IDs')
        ax[i].set_xticklabels(range(1,25), fontsize=8)
    fig.text(0.0, 0.5, 'Impact in respect to the baseline', va='center', rotation='vertical')
    #plt.boxplot(knn, showfliers=False)
    #plt.xticks(range(1,25))
    #plt.ylabel('Mean of the p-values among the\n4 folds of cross-validation', labelpad=15.0)
    #plt.yticks([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    #plt.ylim(-40.0, 40.0)
    #fig.tight_layout()
    fig.set_size_inches(12, 3)
    fig.savefig(os.path.join(result_path, "prototypes_impact.pdf"))
    #plt.clf()
    '''
    knnBoxplot = plt.figure()
    knnBp = knn.boxplot(column=list(range(1,25)), showfliers=False, grid=False)
    knnBoxplot.savefig(os.path.join(result_path, "knn_box_plot.pdf"), format="pdf")
    nbBoxplot = plt.figure()
    nbBp = nb.boxplot(column=list(range(1,25)), showfliers=False, grid=False)
    nbBoxplot.savefig(os.path.join(result_path, "nb_box_plot.pdf"), format="pdf")
    rfBoxplot = plt.figure()
    rfBp = rf.boxplot(column=list(range(1,25)), showfliers=False, grid=False)
    rfBoxplot.savefig(os.path.join(result_path, "rf_box_plot.pdf"), format="pdf")
    '''
    #save_comparison(results_pipelines, results_auto, result_path)


main()