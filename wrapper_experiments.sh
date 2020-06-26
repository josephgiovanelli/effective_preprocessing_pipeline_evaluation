#!/bin/bash

python scenario_generator.py -mode algorithm
python scenario_generator.py -mode preprocessing_algorithm
python experiments_launcher.py -mode algorithm
python experiments_launcher.py -mode preprocessing_algorithm
cd results_processors
python results_pipeline_algorithm_comparator.py