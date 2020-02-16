import collections
import os
import json

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

def load_results(input_path, filtered_datasets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for algorithm in algorithms:
        for dataset in filtered_datasets:
            acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
            acronym += '_' + str(dataset)
            if acronym in results:
                with open(os.path.join(input_path, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
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
            results_map[acronym]['pipeline'] = pipeline

    return results_map

def merge_results(first_results, second_results, first_label, second_label):
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {first_label: 0, second_label: 0, 'draw': 0}
        comparison[acronym] = {}

    for key, value in first_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]

        if first_results[key]['baseline_score'] != second_results[key]['baseline_score']:
            print('Different baseline scores: ' + str(key) + ' ' + str(first_results[key]['baseline_score']) + ' ' + str(second_results[key]['baseline_score']))

        comparison[acronym][data_set] = {first_label: first_results[key]['accuracy'],
                                         second_label: second_results[key]['accuracy'],
                                         'baseline': first_results[key]['baseline_score']}
        winner, loser = (first_label, second_label) if comparison[acronym][data_set][first_label] > comparison[acronym][data_set][second_label] else ((second_label, first_label) if comparison[acronym][data_set][first_label] < comparison[acronym][data_set][second_label] else ('draw', 'draw'))
        summary[acronym][winner] += 1

        if winner == 'draw':
            winner, loser = first_label, second_label

        if comparison[acronym][data_set][loser] == comparison[acronym][data_set]['baseline']:
            comparison[acronym][data_set]['score'] = 'nan'
        else:
            score = (comparison[acronym][data_set][winner] - comparison[acronym][data_set]['baseline']) / (comparison[acronym][data_set][loser] - comparison[acronym][data_set]['baseline'])

            if winner == first_label:
                comparison[acronym][data_set]['score'] = 1 - score
            else:
                comparison[acronym][data_set]['score'] = score - 1

    new_summary = {first_label: 0, second_label: 0, 'draw': 0}
    for algorithm, results in summary.items():
        for category, result in summary[algorithm].items():
            new_summary[category] += summary[algorithm][category]

    summary['summary'] = new_summary

    return comparison, summary

def merge_all_results(first_results, second_results, third_results, first_label, second_label, third_label):
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {
            first_label: 0,
            second_label: 0,
            third_label: 0,
            first_label[0] + second_label[0]: 0,
            first_label[0] + third_label[0]: 0,
            second_label[0] + third_label[0]: 0,
            'draw': 0}
        comparison[acronym] = {}

    for key, value in first_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]

        if first_results[key]['baseline_score'] != second_results[key]['baseline_score'] or \
                first_results[key]['baseline_score'] != third_results[key]['baseline_score'] or \
                second_results[key]['baseline_score'] != third_results[key]['baseline_score']:
            print('Different baseline scores: ' + str(key) + ' ' + str(first_results[key]['baseline_score']) + ' ' + str(second_results[key]['baseline_score']) + ' ' + str(third_results[key]['baseline_score']))

        comparison[acronym][data_set] = {first_label: first_results[key]['accuracy'],
                                         second_label: second_results[key]['accuracy'],
                                         third_label: third_results[key]['accuracy'],
                                         'baseline': first_results[key]['baseline_score']}
        winner = first_label \
            if comparison[acronym][data_set][first_label] > comparison[acronym][data_set][second_label] \
               and comparison[acronym][data_set][first_label] > comparison[acronym][data_set][third_label] \
            else (second_label
        if comparison[acronym][data_set][second_label] > comparison[acronym][data_set][first_label]
           and comparison[acronym][data_set][second_label] > comparison[acronym][data_set][third_label]
        else (third_label
        if comparison[acronym][data_set][third_label] > comparison[acronym][data_set][first_label]
           and comparison[acronym][data_set][third_label] > comparison[acronym][data_set][second_label]
        else (first_label[0] + second_label[0]
        if comparison[acronym][data_set][first_label] == comparison[acronym][data_set][second_label]
           and comparison[acronym][data_set][first_label] > comparison[acronym][data_set][third_label]
        else (first_label[0] + third_label[0]
        if comparison[acronym][data_set][first_label] == comparison[acronym][data_set][third_label]
           and comparison[acronym][data_set][first_label] > comparison[acronym][data_set][second_label]
        else (second_label[0] + third_label[0]
        if comparison[acronym][data_set][second_label] == comparison[acronym][data_set][third_label]
           and comparison[acronym][data_set][second_label] > comparison[acronym][data_set][first_label]
        else 'draw')))))
        summary[acronym][winner] += 1

    new_summary = {
            first_label: 0,
            second_label: 0,
            third_label: 0,
            first_label[0] + second_label[0]: 0,
            first_label[0] + third_label[0]: 0,
            second_label[0] + third_label[0]: 0,
            'draw': 0}
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






