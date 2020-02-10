import json
import os
from functools import reduce
import shutil

import yaml
from six import iteritems
import subprocess
import datetime

from prettytable import PrettyTable
from tqdm import tqdm

import argparse
import h2o
import pandas as pd

from experiment.utils import scenarios as scenarios_util
from results_processors.utils import create_directory
from auto_pipeline_builder import extract_metafeatures, predict_order, build_pipeline, UndefinedOrders, DefinedOrders, \
    check_existence, load_metafeatures, pseudo_exhaustive_pipelines

parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-p", "--pipeline", nargs="?", type=str, required=True, help="kind of pipeline to execute")
parser.add_argument("-r", "--result_path", nargs="?", type=str, required=True, help="path where put the results")
args = parser.parse_args()

SCENARIO_PATH = './scenarios/'
RESULT_PATH = args.result_path
GLOBAL_SEED = 42

def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

# Gather list of scenarios
scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
result_list = [p for p in os.listdir(RESULT_PATH) if '.json' in p]
scenarios = {}

# Determine which one have no result files
for scenario in scenario_list:
    base_scenario = scenario.split('.yaml')[0]
    if scenario not in scenarios:
        scenarios[scenario] = {'results': None, 'path': scenario}
    for result in result_list:
        base_result = result.split('.json')[0]
        print(base_scenario)
        print(base_result)
        if base_result.__eq__(base_scenario):
            scenarios[scenario]['results'] = result
            #date = base_result.split(base_scenario + '_')[-1].replace('_', ' ')
            #scenarios[scenario]['results_date'] = date

# Calculate total amount of time
total_runtime = 0
for path, scenario in iteritems(scenarios):
    with open(os.path.join(SCENARIO_PATH, path), 'r') as f:
        details = None
        try:
            details = yaml.safe_load(f)
        except Exception:
            details = None
            scenario['status'] = 'Invalid YAML'
        if details is not None:
            try:
                runtime = details['setup']['runtime']
                scenario['status'] = 'Ok'
                scenario['runtime'] = runtime
                if scenario['results'] is None:
                    total_runtime += runtime
            except:
                scenario['status'] = 'No runtime info'

# Display list of scenario to be run
invalid_scenarios = {k:v for k,v in iteritems(scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'RUNTIME',  'STATUS', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row([v['path'], str(v['runtime']) + 's', v['status'], v['results']])

to_run = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'RUNTIME', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['runtime']) + 's', v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print('TOTAL RUNTIME: {} ({}s)'.format(datetime.timedelta(seconds=total_runtime), total_runtime))
print

print("The total runtime is {}.".format(datetime.timedelta(seconds=total_runtime)))
print

import psutil


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

with tqdm(total=total_runtime) as pbar:
    h2o.init()
    for info in to_run.values():
        base_scenario = info['path'].split('.yaml')[0]
        output = base_scenario.split('_')[0]
        pbar.set_description("Running scenario {}\n\r".format(info['path']))
        print()

        current_scenario = scenarios_util.load(os.path.join(SCENARIO_PATH, info['path']))
        config = scenarios_util.to_config(current_scenario)

        if check_existence(current_scenario['setup']['dataset'], config['algorithm']):
            if args.pipeline == "auto":
                meta_features = load_metafeatures(current_scenario['setup']['dataset'])

                features_rebalance_order = predict_order(current_scenario['setup']['dataset'], meta_features, config['algorithm'], UndefinedOrders.features_rebalance)
                discretize_rebalance_order = predict_order(current_scenario['setup']['dataset'], meta_features, config['algorithm'], UndefinedOrders.discretize_rebalance)
                pipelines = build_pipeline(features_rebalance_order, discretize_rebalance_order)

                print(pipelines)

                result_path = create_directory(RESULT_PATH, "auto")

                results = []
                for i in range(0, len(pipelines)):
                    pipeline = pipelines[i]
                    cmd = 'python3 ./main.py -s {} -c control.seed={} -p {} -r {} -f {}'.format(
                        os.path.join(SCENARIO_PATH, info['path']),
                        GLOBAL_SEED,
                        pipeline,
                        result_path,
                        len(pipelines))
                    with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario + "_" + str(i))), "a") as log_out:
                        with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario + "_" + str(i))), "a") as log_err:
                            max_time = 3000 / len(pipelines)
                            try:
                                process = subprocess.Popen(cmd, shell=True, stdout=log_out, stderr=log_err)
                                process.wait(timeout=max_time)
                            except:
                                kill(process.pid)
                                print("\n\n" + base_scenario + " does not finish in " + str(max_time) + "\n\n")

                    try:
                        os.rename(os.path.join(result_path, '{}.json'.format(base_scenario)),
                                  os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(i))))

                        with open(os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(i)))) as json_file:
                            data = json.load(json_file)
                            accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                            results.append(accuracy)
                    except:
                        accuracy = 0
                        results.append(accuracy)
                    print(results)

                try:
                    max_i = 0
                    for i in range(1, len(pipelines)):
                        if results[i] > results[max_i]:
                            max_i = i

                    src_dir = os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(max_i)))
                    dst_dir = os.path.join(result_path, '{}.json'.format(base_scenario))
                    shutil.copy(src_dir, dst_dir)
                except:
                    with open(os.path.join(result_path, '{}.txt'.format(base_scenario)), "a") as log_out:
                        print("no available result")


            elif args.pipeline == "quemy":
                pipeline = "impute rebalance normalize features"
                #pipeline = "impute encode rebalance normalize features"

                result_path = create_directory(RESULT_PATH, "quemy")

                cmd = 'python3 ./main.py -s {} -c control.seed={} -p {} -r {} -f {}'.format(
                    os.path.join(SCENARIO_PATH, info['path']),
                    GLOBAL_SEED,
                    pipeline,
                    result_path,
                    1)
                with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario)), "a") as log_out:
                    with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario)), "a") as log_err:
                        max_time = 3000
                        try:
                            process = subprocess.Popen(cmd, shell=True, stdout=log_out, stderr=log_err)
                            process.wait(timeout=max_time)
                        except:
                            kill(process.pid)
                            print("\n\n" + base_scenario + " does not finish in " + str(max_time) + "\n\n")
            elif args.pipeline == "pseudo-exhaustive":

                pipelines = pseudo_exhaustive_pipelines()

                result_path = create_directory(RESULT_PATH, "pseudo-exhaustive")

                results = []
                for i in range(0, len(pipelines)):
                    pipeline = pipelines[i]
                    cmd = 'python3 ./main.py -s {} -c control.seed={} -p {} -r {} -f {}'.format(
                        os.path.join(SCENARIO_PATH, info['path']),
                        GLOBAL_SEED,
                        pipeline,
                        result_path,
                        len(pipelines))
                    with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario + "_" + str(i))),
                              "a") as log_out:
                        with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario + "_" + str(i))),
                                  "a") as log_err:
                            max_time = 3000 / len(pipelines)
                            try:
                                process = subprocess.Popen(cmd, shell=True, stdout=log_out, stderr=log_err)
                                process.wait(timeout=max_time)
                            except:
                                kill(process.pid)
                                print("\n\n" + base_scenario + " does not finish in " + str(max_time) + "\n\n")

                    try:
                        os.rename(os.path.join(result_path, '{}.json'.format(base_scenario)),
                                  os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(i))))

                        with open(
                                os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(i)))) as json_file:
                            data = json.load(json_file)
                            accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                            results.append(accuracy)
                    except:
                        accuracy = 0
                        results.append(accuracy)
                    print(results)

                try:
                    max_i = 0
                    for i in range(1, len(pipelines)):
                        if results[i] > results[max_i]:
                            max_i = i

                    src_dir = os.path.join(result_path, '{}.json'.format(base_scenario + "_" + str(max_i)))
                    dst_dir = os.path.join(result_path, '{}.json'.format(base_scenario))
                    shutil.copy(src_dir, dst_dir)
                except:
                    with open(os.path.join(result_path, '{}.txt'.format(base_scenario)), "a") as log_out:
                        print("no available result")
        else:
            print("not available in all the meta-learners")

        pbar.update(info['runtime'])
