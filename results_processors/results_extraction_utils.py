import collections
import os
import json
import openml
import pandas as pd

from os import listdir
from os.path import isfile, join

from commons import benchmark_suite, algorithms


def get_filtered_datasets():
    df = pd.read_csv('meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def load_results(input_path, filtered_data_sets, algorithm_comparison = False):
    #exceptions = ['nb_37', 'rf_1510', 'rf_1497', 'knn_1489', 'rf_1462', 'nb_46', 'knn_23517', 'rf_1063', 'nb_23517',
    #              'rf_40701', 'nb_1501', 'rf_44', 'nb_1497', 'knn_1486', 'rf_28']
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for acronym in filtered_data_sets:
        if acronym in results:
            with open(os.path.join(input_path, acronym + '.json')) as json_file:
                data = json.load(json_file)
                accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                if not algorithm_comparison:
                    pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
                    prototype = str(data['pipeline']).replace(' ', '')
                    discretize = 'not_in_prototype' \
                        if not('discretize' in prototype) \
                        else ('not_in_pipeline'
                    if data['context']['best_config']['pipeline']['discretize'][0] == 'discretize_NoneType'
                    else 'in_pipeline')
                num_iterations = data['context']['iteration'] + 1
                best_iteration = data['context']['best_config']['iteration'] + 1
                baseline_score = data['context']['baseline_score'] // 0.0001 / 100
        else:
            accuracy = 0
            pipeline = ''
            num_iterations = 0
            best_iteration = 0
            baseline_score = 0

        results_map[acronym] = {}
        results_map[acronym]['accuracy'] = accuracy
        results_map[acronym]['baseline_score'] = baseline_score
        results_map[acronym]['num_iterations'] = num_iterations
        results_map[acronym]['best_iteration'] = best_iteration
        if not algorithm_comparison:
            results_map[acronym]['pipeline'] = pipeline
            results_map[acronym]['prototype'] = prototype
            results_map[acronym]['discretize'] = discretize

    return results_map

def merge_results(auto_results, other_results, other_label, filtered_data_sets):
    auto_label = 'pipeline_algorithm'
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {auto_label: 0, other_label: 0, 'draw': 0}
        comparison[acronym] = {}

    for key in filtered_data_sets:
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]

        baseline_score = auto_results[key]['baseline_score']
        if auto_results[key]['baseline_score'] != other_results[key]['baseline_score']:
            print('Different baseline scores: ' + str(key) + ' ' + str(auto_results[key]['baseline_score']) + ' ' + str(other_results[key]['baseline_score']))
            baseline_score = auto_results[key]['baseline_score'] if auto_results[key]['baseline_score'] > other_results[key]['baseline_score'] else other_results[key]['baseline_score']

        comparison[acronym][data_set] = {auto_label: auto_results[key]['accuracy'],
                                         other_label: other_results[key]['accuracy'],
                                         'baseline': baseline_score}

        max_element = max([comparison[acronym][data_set][auto_label], comparison[acronym][data_set][other_label], comparison[acronym][data_set]['baseline']])
        min_element = min([comparison[acronym][data_set][auto_label], comparison[acronym][data_set][other_label], comparison[acronym][data_set]['baseline']])

        if max_element != min_element:
            other_score = (comparison[acronym][data_set][other_label] - min_element) / (max_element - min_element)
            auto_score = (comparison[acronym][data_set][auto_label] - min_element) / (max_element - min_element)
        else:
            other_score = 0
            auto_score = 0

        comparison[acronym][data_set]['a_score'] = other_score
        comparison[acronym][data_set]['pa_score'] = auto_score

        if max_element != min_element:
            comparison[acronym][data_set]['a_percentage'] = other_score / (other_score + auto_score)
            comparison[acronym][data_set]['pa_percentage'] = auto_score / (other_score + auto_score)
        else:
            comparison[acronym][data_set]['a_percentage'] = 0.5
            comparison[acronym][data_set]['pa_percentage'] = 0.5


        winner, loser = (other_label, auto_label) \
            if comparison[acronym][data_set][other_label] > comparison[acronym][data_set][auto_label] \
            else ((auto_label, other_label)
            if comparison[acronym][data_set][other_label] < comparison[acronym][data_set][auto_label]
            else ('draw', 'draw'))
        summary[acronym][winner] += 1

        if winner == 'draw':
            winner, loser = auto_label, other_label

    new_summary = {auto_label: 0, other_label: 0, 'draw': 0}
    for algorithm, results in summary.items():
        for category, result in summary[algorithm].items():
            new_summary[category] += summary[algorithm][category]

    summary['summary'] = new_summary


    return comparison, summary

def save_comparison(comparison, result_path):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:
            keys = comparison[acronym][list(comparison[acronym].keys())[0]].keys()
            header = ','.join(keys)
            out.write('dataset,' + header + '\n')
            for dataset, results in comparison[acronym].items():
                result_string = ','.join(values_to_string(results.values()))
                out.write(dataset + ',' + result_string + '\n')

def save_summary(summary, result_path):
    if os.path.exists('summary.csv'):
        os.remove('summary.csv')
    with open(os.path.join(result_path, 'summary.csv'), 'w') as out:
        keys = summary[list(summary.keys())[0]].keys()
        header = ','.join(keys)
        out.write(',' + header + '\n')
        for algorithm, results in summary.items():
            result_string = ','.join([str(elem) for elem in results.values()])
            out.write(algorithm + ',' + result_string + '\n')

def plot_comparison(comparison, result_path):
    import matplotlib.pyplot as plt
    import numpy as np

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()

        keys = []
        a_percentages = []
        pa_percentages = []
        for key, value in comparison[acronym].items():
            #keys.append(openml.datasets.get_dataset(key).name)
            keys.append(key)
            a_percentages.append(comparison[acronym][key]['a_percentage'] // 0.0001 / 100)
            pa_percentages.append(comparison[acronym][key]['pa_percentage'] // 0.0001 / 100)
        print(a_percentages)
        print(pa_percentages)

        data = {'dataset': keys, 'a_percentages': a_percentages, 'pa_percentages': pa_percentages}
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by=['pa_percentages', 'a_percentages'])

        plt.rcdefaults()

        plt.bar(df['dataset'].tolist(), df['pa_percentages'].tolist(), label='pipeline_algorithm')
        plt.bar(df['dataset'].tolist(), df['a_percentages'].tolist(), bottom=df['pa_percentages'].tolist(), label='algorithm')

        plt.axhline(y=50, color='#aaaaaa', linestyle='--')

        plt.xlabel('Data-set IDs')
        #plt.xticks(fontsize=3, rotation=90)
        plt.xticks(fontsize=6, rotation=90)
        plt.ylabel('Normalized improvement percentage')
        plt.yticks(ticks=np.linspace(0, 100, 11), labels=['{}%'.format(x) for x in np.linspace(0, 100, 11)])
        plt.title('Comparison of approaches improvements for {}'.format(algorithm))
        plt.legend(loc="upper left")

        fig = plt.gcf()
        fig.set_size_inches(10, 5, forward=True)
        fig.savefig(os.path.join(result_path, '{}.pdf'.format(acronym)))

        plt.clf()









