import os
import json
import pandas as pd

from os import listdir
from os.path import isfile, join

from matplotlib import gridspec

algorithms = ['NaiveBayes', 'KNearestNeighbors', 'RandomForest']
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]

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
            #print('Different baseline scores: ' + str(key) + ' ' + str(auto_results[key]['baseline_score']) + ' ' + str(other_results[key]['baseline_score']))
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

    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()

    SMALL_SIZE = 8
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 22

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


    for i in range(0, 3):
        algorithm = algorithms[i]
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()

        keys = []
        a_percentages = []
        pa_percentages = []
        for key, value in comparison[acronym].items():
            #keys.append(openml.datasets.get_dataset(key).name)
            keys.append(key)
            a_percentages.append(comparison[acronym][key]['a_percentage'] // 0.0001 / 100)
            pa_percentages.append(comparison[acronym][key]['pa_percentage'] // 0.0001 / 100)
        #print(a_percentages)
        #print(pa_percentages)

        data = {'dataset': keys, 'a_percentages': a_percentages, 'pa_percentages': pa_percentages}
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by=['a_percentages', 'pa_percentages'])

        if i == 0:
            plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        if i == 1:
            plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        if i == 2:
            plt.subplot2grid((4, 4), (2, 1), colspan=2, rowspan=2)

        plt.bar(df['dataset'].tolist(), df['pa_percentages'].tolist(), label='Pre-processing and hyper-parameter optimization', color = (1.0, 0.5, 0.15, 1.0))
        plt.bar(df['dataset'].tolist(), df['a_percentages'].tolist(), bottom=df['pa_percentages'].tolist(), label='Hyper-parameter optimization', color = (0.15, 0.5, 0.7, 1.0))

        plt.axhline(y=50, color='#aaaaaa', linestyle='--')

        plt.xlabel('Data sets')
        plt.ylabel('Normalized improvement\npercentage')
        plt.yticks(ticks=np.linspace(0, 100, 11), labels=['{}%'.format(x) for x in np.linspace(0, 100, 11)])
        plt.xticks(ticks=[])
        plt.title('Approaches comparison for {}'.format(algorithm))



    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    text = fig.text(-0.2, 20.15, "")
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(result_path, 'evaluation.pdf'), bbox_extra_artists=(lgd, text), bbox_inches='tight')


    plt.clf()









