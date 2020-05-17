import collections
import os
import json
import string

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from commons import benchmark_suite, algorithms


def get_filtered_datasets():
    df = pd.read_csv('meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def load_results_pipelines(input_path, filtered_data_sets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']

    for acronym in filtered_data_sets:
        algorithm = acronym.split('_')[0]
        data_set = acronym.split('_')[1]

        if acronym in results:
            if not(algorithm in results_map.keys()):
                results_map[algorithm] = {}
            results_map[algorithm][data_set] = []
            with open(os.path.join(input_path, acronym + '.json')) as json_file:
                data = json.load(json_file)
                for i in range(0, 24):
                    index = data['pipelines'][i]['index']
                    accuracy = data['pipelines'][i]['accuracy']
                    results_map[algorithm][data_set].append({'index': index, 'accuracy': accuracy})

    return results_map

def load_results_auto(input_path, filtered_data_sets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']

    for acronym in filtered_data_sets:
        algorithm = acronym.split('_')[0]
        data_set = acronym.split('_')[1]

        if not (algorithm in results_map.keys()):
            results_map[algorithm] = {}
        if acronym in results:
            with open(os.path.join(input_path, acronym + '.json')) as json_file:
                data = json.load(json_file)
                accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                baseline = data['context']['baseline_score'] // 0.0001 / 100
        else:
            accuracy = 0
            baseline = 0

        results_map[algorithm][data_set] = (accuracy, baseline)

    return results_map

def declare_winners(results_map):

    winners_map = {}
    for algorithm, value in results_map.items():
        winners_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]['accuracy'] != 0:
                        index_max_accuracy = i
                else:
                    if results_map[algorithm][dataset][index_max_accuracy]['accuracy'] < results_map[algorithm][dataset][i]['accuracy']:
                        index_max_accuracy = i
            winners_map[algorithm][dataset] = index_max_accuracy

    return winners_map

def get_winners_accuracy(results_map):

    accuracy_map = {}
    for algorithm, value in results_map.items():
        accuracy_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]['accuracy'] != 0:
                        index_max_accuracy = i
                else:
                    if results_map[algorithm][dataset][index_max_accuracy]['accuracy'] < results_map[algorithm][dataset][i]['accuracy']:
                        index_max_accuracy = i
            accuracy_map[algorithm][dataset] = results_map[algorithm][dataset][index_max_accuracy]['accuracy']

    return accuracy_map

def summarize_winners(winners_map):

    summary_map = {}
    for algorithm, value in winners_map.items():
        summary_map[algorithm] = {}
        for i in range(-1, 24):
            summary_map[algorithm][i] = 0

        for _, winner in value.items():
            summary_map[algorithm][winner] += 1

    return summary_map

def save_summary(summary_map, result_path):
    if os.path.exists('summary.csv'):
        os.remove('summary.csv')
    total = {}
    for algorithm, value in summary_map.items():
        pipelines = []
        winners = []
        with open(os.path.join(result_path, '{}.csv'.format(algorithm)), 'w') as out:
            out.write('pipeline,winners\n')
            for pipeline, winner in value.items():
                out.write(str(pipeline) + ',' + str(winner) + '\n')
                pipelines.append(str(pipeline))
                winners.append(int(winner))
                total[str(pipeline)] = winner if not(str(pipeline) in total.keys()) else total[str(pipeline)] + winner

        plt.rcdefaults()
        plt.bar(pipelines, winners)

        plt.xlabel('Prototype IDs')
        plt.ylabel('Number of data-sets for which a prototype\nachieved the best performance')
        plt.yticks(np.arange(0, 10, 1))
        plt.title('Comparison of the goodness of the prototypes for {}'.format(algorithm.upper()))

        fig = plt.gcf()
        fig.set_size_inches(10, 5, forward=True)
        fig.savefig(os.path.join(result_path, '{}.pdf'.format(algorithm)))

        plt.clf()

    plt.rcdefaults()
    plt.bar(total.keys(), total.values())

    plt.xlabel('Prototype IDs')
    plt.ylabel('Number of times for which a prototype\nachieved the best performance')
    plt.yticks(np.arange(0, 20, 2))
    plt.title('Comparison of the goodness of the prototypes (all data-sets and algorithms together)')

    fig = plt.gcf()
    fig.set_size_inches(10, 5, forward=True)
    fig.savefig(os.path.join(result_path, 'summary.pdf'))

    plt.clf()



def save_comparison(results_pipelines, results_auto, result_path):
    if os.path.exists('comparison.csv'):
        os.remove('comparison.csv')
    for algorithm, value in results_pipelines.items():
        plot_results = {}
        with open(os.path.join(result_path, '{}.csv'.format(algorithm)), 'w') as out:
            out.write('dataset,exhaustive,pseudo-exhaustive,baseline,score\n')
            for dataset, accuracy in value.items():
                score = 0 if (accuracy - results_auto[algorithm][dataset][1]) == 0 else (results_auto[algorithm][dataset][0] - results_auto[algorithm][dataset][1]) / (accuracy - results_auto[algorithm][dataset][1])
                out.write(str(dataset) + ',' + str(accuracy) + ',' + str(results_auto[algorithm][dataset][0]) + ',' +
                          str(results_auto[algorithm][dataset][1]) + ',' + str(score) + '\n')
                plot_results[dataset] = score

        plt.rcdefaults()
        plt.bar(plot_results.keys(), plot_results.values())

        plt.axhline(y=1.0, color='#aaaaaa', linestyle='--')

        plt.xlabel('Dataset IDs')
        plt.xticks(fontsize=6, rotation=90)
        plt.ylabel('Score')
        plt.ylim(0.0, 1.1)
        plt.title('Evaluation of the prototype building through the proposed precedence for {}'.format(algorithm.upper()))

        fig = plt.gcf()
        fig.set_size_inches(10, 5, forward=True)
        fig.savefig(os.path.join(result_path, '{}.pdf'.format(algorithm)))

        plt.clf()