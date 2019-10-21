from __future__ import print_function

import argparse
import collections
import os
import json
import pandas as pd

from os import listdir
from os.path import isfile, join

algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors', 'SVM', 'NeuralNet']
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]

def mergeDict(dict1, dict2):
    """ Merge dictionaries and keep values of common keys in list"""
    dict3 = {**dict2, **dict1}
    for key, value in dict3.items():
        if key in dict2 and key in dict1:
            dict3[key] = [value, dict2[key]]

    return dict3

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--first_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-ii", "--second_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    input_paths = [args.first_input, args.second_input]
    result_path = args.output
    pipeline = args.pipeline
    print(pipeline)
    return input_paths, result_path, pipeline

def get_filtered_datasets():
    df = pd.read_csv("openml/meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def load_results(input_paths, filtered_datasets, comparison):
    for path in input_paths:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']
        comparison[path] = {}
        for algorithm in algorithms:
            for dataset in filtered_datasets:
                acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
                acronym += '_' + str(dataset)
                if acronym in results:
                    with open(os.path.join(path, acronym + '.json')) as json_file:
                        data = json.load(json_file)
                        accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                        pipeline = str(data['context']['best_config']['pipeline']).replace(" ", "").replace(",", " ")
                        num_iterations = data['context']['iteration']
                        best_iteration = data['context']['best_config']['iteration']
                        comparison[path][acronym] = (accuracy, pipeline, num_iterations, best_iteration)
                else:
                    accuracy = 0
                    pipeline = ""
                    num_iterations = 0
                    best_iteration = 0
                    comparison[path][acronym] = (accuracy, pipeline, num_iterations, best_iteration)

    return collections.OrderedDict(sorted(mergeDict(comparison[input_paths[0]], comparison[input_paths[1]]).items()))

def create_output_files(result_path, partial_results):
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        partial_results[acronym] = {'conf1': 0, 'draws': 0, 'conf2': 0, 'valid': 0}
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write("dataset,name,dimensions,conf1,pipeline1,num_iterations1,best_iteration1,conf2,pipeline2,"
                      "num_iterations2,best_iteration2,valid\n")
    return partial_results

def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {"pipeline1": [], "pipeline2": []}
    for step in scheme:
        if pipeline1 != "":
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(" ", ","))
            pipelines["pipeline1"].append(raw_pipeline1[step][0].split("_")[1])
        if pipeline2 != "":
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(" ", ","))
            pipelines["pipeline2"].append(raw_pipeline2[step][0].split("_")[1])
    return pipelines


def check_validity(pipelines, result):
    if pipelines["pipeline1"] == [] or pipelines["pipeline1"] == []:
        return False
    else:
        if pipelines["pipeline1"].__contains__('NoneType') and pipelines["pipeline2"].__contains__('NoneType'):
            return result == 0
        if pipelines["pipeline1"].__contains__('NoneType') and not(pipelines["pipeline2"].__contains__('NoneType')):
            return result == 0 or result == 2
        if not(pipelines["pipeline1"].__contains__('NoneType')) and pipelines["pipeline2"].__contains__('NoneType'):
            return result == 0 or result == 1
    return True




def aggregate_results(filtered_datasets, results, partial_results, result_path, pipeline):
    df = pd.read_csv("openml/meta-features.csv")
    df = df.loc[df['did'].isin(filtered_datasets)]
    for key, value in results.items():
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]
        name = df.loc[df['did'] == int(dataset)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(dataset)][['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])
        conf1 = value[0][0]
        conf2 = value[1][0]
        pipelines = compose_pipeline(value[0][1], value[1][1], pipeline)
        result = -1
        if conf1 > conf2:
            result = 1
        elif conf1 == conf2:
            result = 0
        elif conf1 < conf2:
            result = 2
        if result == -1:
            raise ValueError('A very bad thing happened.')
        if acronym == "knn" and result == 0 and abs(conf1 - conf2) < 1:
            result = 0
        valid = check_validity(pipelines, result)
        if valid:
            if result == 0:
                partial_results[acronym]['draws'] += 1
            elif result == 1:
                partial_results[acronym]['conf1'] += 1
            elif result == 2:
                partial_results[acronym]['conf2'] += 1
            partial_results[acronym]['valid'] += 1

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write(dataset + "," + name + "," + dimensions + "," + str(conf1) + "," + str(value[0][1]) + "," +
                      str(value[0][2]) + "," + str(value[0][3]) + "," + str(conf2) + "," + str(value[1][1]) +
                      "," + str(value[1][2]) +  "," + str(value[1][3]) +  "," + str(valid) + "\n")

    complete_results = {'conf1': sum(x['conf1'] for x in partial_results.values()),
                     'draws': sum(x['draws'] for x in partial_results.values()),
                     'conf2': sum(x['conf2'] for x in partial_results.values()),
                     'valid': sum(x['valid'] for x in partial_results.values())}
    return partial_results, complete_results

def write_summary(result_path, partial_results, complete_results):
    with open(os.path.join(result_path, 'results.csv'), "a") as out:
        out.write("algorithm,conf1,draws,conf2,valid\n")
        for key, value in partial_results.items():
            row = key
            for k, v in value.items():
                row += "," + str(v)
            row += "\n"
            out.write(row)
        row = "summary"
        for key, value in complete_results.items():
            row += "," + str(value)
        out.write(row)

def main():
    input_paths, result_path, pipeline = parse_args()
    filtered_datasets = get_filtered_datasets()
    results = load_results(input_paths, filtered_datasets, {})
    partial_results = create_output_files(result_path, {})
    partial_results, complete_results = aggregate_results(filtered_datasets, results, partial_results, result_path, pipeline)
    write_summary(result_path, partial_results, complete_results)

main()









