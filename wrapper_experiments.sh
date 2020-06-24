#!/bin/bash

python scenario_generator.py -path -mode algorithm
python scenario_generator.py -path -mode preprocessing_algorithm
python experiments_launcher.py -mode algorithm
python experiments_launcher.py -mode preprocessing_algorithm
