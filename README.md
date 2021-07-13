# Dependencies

The dependencies can be found in the ```requirements.txt``` file

# Reproducing the experiments

There are two branches related to the DOLAP 2021 paper:
- ```evaluation_1_2```
- ```evaluation_3```

Run the ```wrapper_experiments.sh``` of these branches to get the results about the related experiments.
The script will create a folder ```results``` where you can find the outcome with some graphs. The folder is not in tracking, hence you can switch between branches, run the script, and have all the results in that folder.

Remember: the branch ```evaluation_1_2``` need the results of the ```evaluation_3``` branch to be executed.

For the extension for the Information Systems journal we created two new branches: ```extension_evaluation_1_2``` and ```extension_evaluation_3```. 
Run the ```wrapper_experiments.sh``` of these branches to get the updated results.
Once you got such results, the meta-learner can be run.
Since it is implemented in R, it has to be run separately.
The script is in ```results_processor/meta_learner/meta_learner.R```.

# Pipeline construction

This repository contains just the evaluation experiments of our approach to build effective pre-processing pipelines. Meanwhile, the [effective_preprocessing_pipeline_construction](https://github.com/josephgiovanelli/effective_preprocessing_pipeline_construction) repository contains the experiments performed to discover the dependencies between transformations and, hence, to develop our approach.
