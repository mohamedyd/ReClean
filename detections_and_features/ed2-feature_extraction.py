from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run_multi
from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run
import multiprocessing as mp
from ml.datasets.specificDataset import SpecificDataset
from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier

import os
import sys
import pandas as pd
import numpy as np




def run_ed2(clean_df_path, dirty_df_path, label_cutoff, method):
    """
    Extract features and detections by ED2.

    Arguments:
        clean_df_path (str): path to clean dataframe
        dirty_df_path (str): path to dirty dataframe
        label_cutoff (list): number of maximum labeling by user/clean set that is referred to during active learning.
        method (str): feature extraction method: ActiveClean, word2vec etc.

    Returns:
        all_error_statusDF (pd.DataFrame): dataframe with True/False flags to indicate erroneous cells
        feature_matrix (csr matrix): matrix including row-wise feature vectors
    """

    # ed default settings
    parameters = {'use_word2vec_only': False,
                  'w2v_size': 100}  # char unigrams + meta data + correlation + word2vec
    # parameters={'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100} #char unigrams + meta data + correlation + word2vec

    df_dirty = pd.read_csv(dirty_df_path,
                           dtype=str,
                           header="infer",
                           encoding="utf-8",
                           keep_default_na=False,
                           low_memory=False
                           )

    df_clean = pd.read_csv(clean_df_path_df_path,
                           dtype=str,
                           header="infer",
                           encoding="utf-8",
                           keep_default_na=False,
                           low_memory=False
                           )

    # classifiers = [XGBoostClassifier, LinearSVMClassifier, NaiveBayesClassifier]
    classifier = XGBoostClassifier

    data = SpecificDataset('', df_dirty, df_clean)
    my_dict = parameters.copy()
    my_dict['dataSet'] = data
    my_dict['classifier_model'] = classifier
    my_dict['checkN'] = 1
    my_dict['label_threshold'] = label_cutoff

    if method == 'ActiveClean':
        my_dict['use_active_clean'] = True
    elif method== 'word2vec':
        my_dict['use_word2vec'] = True
    elif method == 'ValueCorrelation':
        my_dict['use_cond_prob'] = True
    elif method =='Metadata':
        my_dict['use_metadata'] = True
    elif method =='BoostClean':
        my_dict['use_boostclean_metadata'] = True

    # runs experiment checkN rounds
    return_dict, all_error_status, feature_matrix = run(**my_dict)

    # get dataframe which cell is true if cell is an detected error and else false
    all_error_statusDF = pd.DataFrame(all_error_status)
    return all_error_statusDF, feature_matrix


if __name__ == '__main__':
    clean_df_path = ''
    dirty_df_path = ''
    label_cutoff = 1000
    method = 'ActiveClean'

    error_df, feature_matrix = run_ed2(clean_df_path, dirty_df_path, label_cutoff, method)
