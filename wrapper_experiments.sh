#!/bin/bash

python scenario_generator.py -mode algorithm
python scenario_generator.py -mode preprocessing_algorithm
python experiments_launcher.py -mode algorithm
python experiments_launcher.py -mode preprocessing_algorithm
cd results_processors
python results_comparator.py
python pp_pipeline_study.py
python pp_pipeline_study2.py
python meta_learning_input_preparation.py