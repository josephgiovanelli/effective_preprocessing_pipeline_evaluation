import os
import copy
import re
from collections import OrderedDict

import openml
import pandas as pd

from commons import benchmark_suite, algorithms

SCENARIO_PATH = './scenarios/'

policies = ['split']


policies_config = {
    'iterative': {
        'step_algorithm': 15,
        'step_pipeline': 15,
        'reset_trial': False
    },
    'split': {
        'step_pipeline': 30
    },
    'adaptive': {
        'initial_step_time': 15,
        'reset_trial': False,
        'reset_trials_after': 2
    },
    'joint': {}
}

base = OrderedDict([
    ('title', 'Random Forest on Wine with Iterative policy'),
    ('setup', {
        'policy': 'iterative',
        'runtime': 200,
        'algorithm': 'RandomForest',
        'dataset': 'wine'
    }),
    ('control', {
        'seed': 42
    }),
    ('policy', {})
])

def __write_scenario(path, scenario):
    try:
        print('   -> {}'.format(path))
        with open(path, 'w') as f:
            for k,v in scenario.items():
                if isinstance(v, str):
                    f.write('{}: {}\n'.format(k, v))
                else:
                    f.write('{}:\n'.format(k))
                    for i,j in v.items():
                        f.write('  {}: {}\n'.format(i,j))
    except Exception as e:
        print(e)

def get_filtered_datasets():
    df = pd.read_csv("results_processors/meta_features/simple-meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

for id in get_filtered_datasets():
    print('# DATASET: {}'.format(id))
    for algorithm in algorithms:
        print('## ALGORITHM: {}'.format(algorithm))
        for policy in policies:
            scenario = copy.deepcopy(base)
            scenario['setup']['dataset'] = id
            scenario['setup']['algorithm'] = algorithm
            scenario['setup']['policy'] = policy
            scenario['policy'] = copy.deepcopy(policies_config[policy])
            a = re.sub(r"(\w)([A-Z])", r"\1 \2", algorithm)
            b = ''.join([c for c in algorithm if c.isupper()]).lower()
            scenario['title'] = '{} on dataset n {} with {} policy'.format(
                a,
                id,
                policy.title()
            )
            '''
            if policy == 'split':
                runtime = scenario['setup']['runtime']
                step = policies_config['split']['step_pipeline']
                ranges = [i for i in range(0, runtime+step, step)]
                for r in ranges:
                    scenario['policy']['step_pipeline'] = r
                    path = os.path.join('./scenarios', '{}_{}_{}_{}.yaml'.format(b, task_id, policy, r))
                    __write_scenario(path, scenario)
            else:
                path = os.path.join('./scenarios', '{}_{}_{}.yaml'.format(b, task_id, policy))
                __write_scenario(path, scenario)
            '''
            runtime = scenario['setup']['runtime']
            step = policies_config['split']['step_pipeline']
            scenario['policy']['step_pipeline'] = runtime
            path = os.path.join(SCENARIO_PATH, '{}_{}.yaml'.format(b, id))
            __write_scenario(path, scenario)

