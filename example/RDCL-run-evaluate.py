from sklearn import preprocessing, linear_model, metrics
import pickle, copy
import numpy as np
import pandas as pd
import tensorflow as tf

# import RDCL 
from RDCL import RDCL
from DataPreprocessing import DataPreprocessing

# dataset
dataset = 'nasa'
error_percentage = 10 # error percentage in dirty dataset


# detections and features
# ActiveClean, word2vec, BoostClean, Metadata, ValueCorrelation
detection = 'ActiveClean'
input_features = 'ActiveClean'


problem  = 'reg' # NASA, Wine Quality, CCPP, Retail Sales
# problem  = 'clf' # Smartfactory, WDBC

# Normalization of datasets for training predictor
norm = 'minmax'
# norm = 'standard'


# load error df and feature vectors
error_file_name = 'errors_df.pkl'
feature_vec_file_name = 'feature_matrix.pkl'

with open(f'./datasets/{dataset}/{error_percentage}%/{detection}/{error_file_name}', 'rb') as f:
    error_status_df = pickle.load(f)

with open(f'./datasets/{dataset}/{error_percentage}%/{input_features}/{feature_vec_file_name}', 'rb') as f:
    feature_vectors = pickle.load(f)


# preprocessing of feature vectors
feature_vectors = feature_vectors.toarray()
feature_vectors = preprocessing.StandardScaler().fit_transform(feature_vectors)

# drop full zero columns from the end
c = feature_vectors.any(axis=0)
feature_vectors = feature_vectors[:, :np.max(np.where(c == True)) + 1:]




# Load data splits
dataset_dictionary = {
    "name": 'nasa',
    "path": f"./datasets/{dataset}/nasa_{error_percentage}%.csv",
    "clean_path": f"./datasets/{dataset}/nasa_clean_train.csv"
}
valid_path = f'./datasets/{dataset}/{dataset}_valid.csv'


dirty = pd.read_csv(dataset_dictionary['path'])
clean = pd.read_csv(dataset_dictionary['clean_path'])
valid = pd.read_csv(valid_path)



# Dataset features
target_feature = 'sound_pressure_level'
features = valid.columns.tolist()
numerical_features = copy.deepcopy(features)
numerical_features.remove(target_feature)
categorical_features = []

label_idx = dirty.columns.tolist().index(target_feature) # comment this line if any errors are injected in the output column

# create detection dictionary from error df
if not isinstance(error_status_df, dict):
    detection_dict = dict()
    for col in range(error_status_df.shape[1]):
        for row in range(error_status_df.shape[0]):
            if error_status_df.iat[row, col]:
                if col != label_idx:
                    detection_dict[(row, col)] = 'error'
else:
    detection_dict = error_status_df




# DataPreprocessing parameters
dataloader_params = {'train_dataset': None,
                     'valid_dataset':None,
                     'balance_dataset':False,
                     'drop_duplicates':False,
                     'labels_to_keep':[],
                     'numerical_features':numerical_features,
                     'categorical_features':categorical_features,
                     'target_column':target_feature,
                     'problem':('classification' if problem=='clf' else 'regression'),
                     'normalization':norm}

# List of cleaners included in the cleaner inventory
cleaner_list = ['mean_mode_imputer', 'median_mode_imputer',
                             'knn_missForest_imputer_5', 'knn_missForest_imputer_20', 'knn_missForest_imputer_50',
                             'em_missForest_imputer_50', 'em_missForest_imputer_100',
                             'bayesianRidge_missForest_imputer',
                             'missForest_imputer_50', 'missForest_imputer_100', 'missForest_imputer_200']


# All parameters that RDCL takes
rdcl_params = {'dirty_dataset':dirty,
               'valid_dataset':valid,
               'cleaner_selector_neurons':[128, 128, 256, 256],
               'activation':tf.nn.relu,
               'n_epochs':2, # epochs for cleaner RL training
               'batch_size':2000,
               'cleaner_list':cleaner_list,
               'predictor_algorithm':('linear' if problem=='reg' else 'logistic'),
               'predictor_model':(linear_model.LinearRegression() if problem=='reg' else linear_model.LogisticRegression()),
               'n_iterations':None,  # epochs for Predictor (MLP) training
               'eval_metric':('rmse' if problem=='reg' else 'roc auc'),
               'baseline_window':10,
               'dataloader_params':dataloader_params,
               'detection_dictionary':detection_dict,
               'feature_vectors':feature_vectors,
               'exp_threshold':0.8
               }



# Initialize the framework
rl = RDCL(rdcl_params)

# Run RDCL
rl.train_rl_cleaner()



# Evaluation
preds = rl.cleaner_nn.predict(rl.feature_vectors)

dirty_row_idx = np.unique([i for i, j in detection_dict.keys()]).tolist()  # rows containing errors
clean_row_idx = list(set(dirty_row_idx).symmetric_difference(dirty))

cleaners_chosen = np.argmax(preds, axis=1)

print('chosen cleaners: ', np.unique(cleaners_chosen, return_counts=1))
print('chosen cleaners: ', np.unique(cleaners_chosen[dirty_row_idx], return_counts=1))




# Store results
scores = []

# Execute chosen cleaners on dirty dataset
repaired_dataset = pd.DataFrame(np.empty_like(rl.dirty_dataset), columns=dirty.columns)
for cl_idx in set(cleaners_chosen):
    # repaired indices for respective cleaners
    repaired_indices = np.where(cleaners_chosen == cl_idx)[0].tolist()
    cleaned_dataset = rl.cleaned_datasets[cl_idx]
    repaired_dataset.iloc[repaired_indices, :] = cleaned_dataset.iloc[repaired_indices, :]


if problem=='clf':
    params = {'train_dataset': repaired_dataset, 'valid_dataset':valid.copy(),
              'balance_dataset':False,
              'drop_duplicates':False,
              'labels_to_keep':[],
              'features_to_drop':[],
              'numerical_features':numerical_features,
              'categorical_features':categorical_features,
              'target_column':target_feature,
              'problem':'classification',
              'normalization':norm}
else:
    params = {'train_dataset': repaired_dataset, 'valid_dataset':valid.copy(),
              'balance_dataset':False,
              'drop_duplicates':False,
              'labels_to_keep':[],
              'features_to_drop':[],
              'numerical_features':numerical_features,
              'categorical_features':categorical_features,
              'target_column':target_feature,
              'problem':'regression',
              'normalization':norm}


dl = DataPreprocessing(params)
x_train_r, y_train_r, x_val_r, y_val_r = dl.preprocess_data()
x_train_r, y_train_r, x_val_r, y_val_r = x_train_r.values, y_train_r.values, x_val_r.values, y_val_r.values


if problem=='reg':
    lr = linear_model.LinearRegression().fit(x_train_r, y_train_r)
    y_pred_valid_r = lr.predict(x_val_r)
    rmse_score = metrics.mean_squared_error(y_val_r, y_pred_valid_r, squared=False)
    print(f'rmse: {rmse_score}')
    scores.append(rmse_score)

else:
    lr = linear_model.LogisticRegression().fit(x_train_r, y_train_r)
    y_pred_valid_r = lr.predict_proba(x_val_r)
    roc_score = metrics.roc_auc_score(y_val_r, y_pred_valid_r[:, 1])
    print(f'roc auc: {roc_score}')
    scores.append(roc_score)




# Results: execute all cleaners in cleaner inventory on dirty data
for cl_idx in range(len(rl.cleaner_list)):
    if problem=='clf':
        params = {'train_dataset': rl.cleaned_datasets[cl_idx].copy(), 'valid_dataset':valid.copy(),
                  'balance_dataset':False,
                  'drop_duplicates':False,
                  'labels_to_keep':[],
                  'features_to_drop':[],
                  'numerical_features':numerical_features,
                  'categorical_features':categorical_features,
                  'target_column':target_feature,
                  'problem':'classification',
                  'normalization':norm}
    else:
        params = {'train_dataset': rl.cleaned_datasets[cl_idx].copy(), 'valid_dataset': valid.copy(),
                  'balance_dataset': False,
                  'drop_duplicates': False,
                  'labels_to_keep': [],
                  'features_to_drop': [],
                  'numerical_features': numerical_features,
                  'categorical_features': categorical_features,
                  'target_column': target_feature,
                  'problem': 'regression',
                  'normalization': norm}

    dl = DataPreprocessing(params)
    x_train_r, y_train_r, x_val_r, y_val_r = dl.preprocess_data()
    x_train_r, y_train_r, x_val_r, y_val_r = x_train_r.values, y_train_r.values, x_val_r.values, y_val_r.values

    if problem=='reg':
        lr = linear_model.LinearRegression().fit(x_train_r, y_train_r)
        y_pred_valid_r = lr.predict(x_val_r)
        score = metrics.r2_score(y_val_r, y_pred_valid_r)
        rmse = metrics.mean_squared_error(y_val_r, y_pred_valid_r, squared=False)
        scores.append(rmse)
    else:
        lr = linear_model.LogisticRegression().fit(x_train_r, y_train_r)
        y_pred_valid_r = lr.predict_proba(x_val_r)
        auc_score = metrics.roc_auc_score(y_val_r, y_pred_valid_r[:, 1])
        scores.append(auc_score)

    print(f'cleaner: {rl.cleaner_list[cl_idx]}')
    print(f'rmse: {rmse}')
