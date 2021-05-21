from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import create_directory

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def main():
    # configure environment
    input_path = ".."
    input_path = os.path.join(input_path, "results")
    input_path = os.path.join(input_path, "summary")
    input_path = os.path.join(input_path, "evaluation3")
    input_path = os.path.join(input_path, "extension")
    input_path = os.path.join(input_path, "operator_meta_learning")
    input_path = os.path.join(input_path, "input")

    ## Meta-features Loading
    data = pd.read_csv(os.path.join(input_path, "manual_fs_union.csv"))
    data['InterQuartileRangeOfNumericAtts'] = data['Quartile3MeansOfNumericAtts'] - data['Quartile1MeansOfNumericAtts']
    data2 = data[(data["ClassEntropy"] <= 0.863) & (data["Quartile1MeansOfNumericAtts"] > 9.329)]
    #print(data2)
    print(data2['InterQuartileRangeOfNumericAtts'].drop_duplicates())
    data = data[(data["ClassEntropy"] <= 0.863) & (data["Quartile1MeansOfNumericAtts"] <= 9.329) & (data["features"] == "SelectKBest")]
    #data.to_csv(os.path.join(input_path, 'data' + '.csv'), index=False)
    #print(data)
    print(data['InterQuartileRangeOfNumericAtts'].drop_duplicates())


main()