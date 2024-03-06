# Source code of the paper "ReClean: Reinforcement Learning for Automated Data Cleaning in ML Pipelines," at Proceddings of the DBML'24 (ICDE Workshop), May 2024.


## Scripts

#### error_injection/inject_errors.py
- Inject errors on a given dataset using *error-generator* library, with given methods and their corresponding error rates as function arguments.

#### detections_and_features/ed2-feature_extraction.py
- Get detections and features from dirty datasets using ED2. Intended feature extraction method can be specified as function argument.

#### rdcl/Cleaners.py
- The class including cleaners used in cleaner inventory of the RDCL framework. If new cleaners are added to the class, RDCL class also must be updated by specifying the indices of new cleaners.

#### rdcl/DataPreprocessing.py
- The class that apply preprocessing steps on dirty and validation datasets after cleaning is done, including options for normalization, dropping of nans, one-hot encoding of categorical features, balancing of datasets through upsampling and downsampling, dropping of duplicates, and so on.

#### rdcl/RDCL.py
- The main class of the project that merges everything each component of the framework. The framework has the following steps respectively;
    - sampling a batch and their corresponding detections and feature vectors, 
    - execution of selected cleaners using ```Cleaners.py``` to clean the dirty batch, 
    - preprocessing the cleaned datasets using ```DataPreprocessing.py```, 
    - getting reward -a performance metric- using predictor network trained on the cleaned dataset.

#### example/RDCL-run-evaluate.py
- Example script for loading framework parameters and running an experiment for RDCL pipeline and then, getting the performance results of the RDCL and baseline cleaners.




## Extending the Project
The project can be extended by running on new datasets, new cleaners, new predictor models, or even new tasks, e.g. clustering, by defining the proper performance metrics.

- In case of a new dataset; ```inject_errors.py``` and ```ed2-feature_extraction.py``` can be run consecutively on the dataset to generate a dirty version of it and their corresponding detections and feature vectors by ED2. An external error detector and a feature extractor can also be used instead of ED2.

- New cleaners should be implemented in ```Cleaners.py```, and then added to the ```clean_errors()``` method of RDCL class in ```RDCL.py``` in the same format as other cleaners.

- Currently, the pipeline accepts LogisticRegression, LinearRegression and MLP network implemented in Tensorflow as predictor models. Any new models can be implemented in ```RDCL.py```. Almost all essential performance metrics for classification and regression tasks are available in the pipeline already. In case a new one is intended to be added, e.g. Silhouette score, it can be incorporated in the last part of ```train_rl_cleaner()``` of RDCL class.

- Loss function also can be tweaked in ```loss_fn()``` method of RDCL class.



## Installation
Create a conda environment with required packages for the project:
```
conda rdcl_env create -f environment.yml
conda activate rdcl_env
```

Run the following command in the project directory;

on Windows and Linux:
```
python setup.py install
```

on Mac:
```
python -m pip install .
```
