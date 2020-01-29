#!/bin/bash

python3 experiments_launcher.py -p auto -r results
python3 experiments_launcher.py -p quemy -r results
python3 experiments_launcher.py -p pseudo-exhaustive -r results
python3 results_processors/auto_quemy_comparator.py -ia ../results/auto -iq ../results/quemy -ip ../results/pseudo-exhaustive -o ../results