#!/bin/bash


python scenario_generator.py
python experiments_launcher.py
cd results_processors
python results_extraction.py
python results_comparison.py
