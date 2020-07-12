# Dependencies

The dependencies can be found in the ```requirements.txt``` file

# Reproducing the experiments

There are two branches:
- ```evaluation_1_2```
- ```evaluation_3```

Run the ```wrapper_experiments.sh``` of these branches to get the results about the related experiments.
The script will create a folder ```results``` where you can find the outcome with some graphs. The folder is not in tracking, hence you can switch between branches, run the script, and have all the results in that folder.

Remember: the branch ```evaluation_1_2``` need the results of the ```evaluation_3``` branch to be executed.

# Pipeline construction

This repository contains just the evaluation experiments of our approach to build effective pre-processing pipelines. Meanwhile, the [effective_preprocessing_pipeline_construction](https://github.com/josephgiovanelli/effective_preprocessing_pipeline_construction) repository contains the experiments performed to discover the dependencies between transformations and, hence, to develop our approach.
