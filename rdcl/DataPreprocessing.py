import copy
import numpy as np
import pandas as pd
from imblearn import over_sampling, under_sampling
from sklearn import metrics, preprocessing




class DataPreprocessing:
    def __init__(self, params: dict):

        self.checks(params)  # check the types of parameters

        # datasets
        self.train_dataset = params['train_dataset']
        self.valid_dataset = params['valid_dataset']

        # flag for dropping duplicates
        self.drop_duplicates = params['drop_duplicates']

        self.problem = params['problem'] # classification or regression

        # scaler: 'standard' for z-score normalization, 'minmax' for min-max normalization
        self.scaler = (preprocessing.StandardScaler() if params['normalization'] == 'standard' else preprocessing.MinMaxScaler())

        self.balance_dataset = params['balance_dataset']  # oversampling flag in case of classification

        self.numerical_features = params['numerical_features']  # numerical features

        self.number_of_numerical_cols = len(self.numerical_features)  # used in the normalization of oversampled data

        self.categorical_features = params['categorical_features']  # categorical features

        self.target_column = params['target_column']  # output variable

    def preprocess_data(self):
        """
        Applies a generic preprocessing on dirty and validation datasets.
        """

        # drop nans
        self.train_dataset.dropna(axis=0, inplace=True)

        # drop labels from numerical features if included
        if self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)

        # train (cleaned) dataset
        x_train = self.train_dataset.drop([self.target_column], axis=1)  # train data
        y_train = self.train_dataset[self.target_column]  # train labels

        # validation (clean) dataset
        x_val = self.valid_dataset.drop([self.target_column], axis=1)  # validation data
        y_val = self.valid_dataset[self.target_column]  # validation labels

        # normalization
        if len(self.numerical_features):
            x_val[self.numerical_features] = self.scaler.fit_transform(x_val[self.numerical_features])


        if len(self.categorical_features):
            # make categorical features object type
            x_train.loc[:, self.categorical_features] = x_train.loc[:, self.categorical_features].astype('object')
            x_val.loc[:, self.categorical_features] = x_val.loc[:, self.categorical_features].astype('object')

            # one-hot encoding of categorical features
            x_train = pd.get_dummies(x_train)
            x_val = pd.get_dummies(x_val)

        # upsampling to balance the labels (in case of classification)
        if self.problem == 'classification' and self.balance_dataset:
            x_train, y_train = self.upsampling(x_train, y_train)

        # drop duplicates
        if self.drop_duplicates:
            dirty = pd.concat([x_train, y_train], axis=1)
            clean = pd.concat([x_val, y_val], axis=1)
            dirty.drop_duplicates(inplace=True)
            clean.drop_duplicates(inplace=True)

            x_train = dirty.drop([self.target_column], axis=1)  # train data
            y_train = dirty[self.target_column]  # train labels

            x_val = clean.drop([self.target_column], axis=1)  # validation data
            y_val = clean[self.target_column]  # validation labels

        return x_train, y_train, x_val, y_val

    def upsampling(self, x_train, y_train):
        """
        Upsampling of imbalanced datasets, followed by downsampling to remove redundantly close samples.
        """

        if len(self.categorical_features):
            # SMOTENC if any categorical features exists
            x_upsampled, y_upsampled = over_sampling.SMOTENC(categorical_features=[*range(self.number_of_numerical_cols,
                                                                                          x_train.shape[-1])],
                                                             sampling_strategy='minority').fit_resample(x_train,
                                                                                                        y_train)
        else:
            x_upsampled, y_upsampled = over_sampling.SMOTE(sampling_strategy='minority').fit_resample(x_train,
                                                                                                      y_train)
        # downsampling by Tomek Links
        x_downsampled, y_downsampled = under_sampling.TomekLinks().fit_resample(x_upsampled, y_upsampled)

        # standardization of upsampled data
        for i in range(self.number_of_numerical_cols):
            x_downsampled.iloc[:, i] = self.scaler.fit_transform(x_downsampled.iloc[:, i].values.reshape(-1, 1))

        return x_downsampled, y_downsampled


    @staticmethod
    def checks(params):
        if not isinstance(params['train_dataset'], pd.DataFrame):
            raise TypeError('dirty_dataset must be an instance of pandas.Dataframe!')

        if not isinstance(params['valid_dataset'], pd.DataFrame):
            raise TypeError('"clean_dataset" must be an instance of pandas.Dataframe!')

        if not isinstance(params['problem'], str):
            raise TypeError('"problem" must be an instance of "str"!')

        if not isinstance(params['normalization'], str):
            raise TypeError('"normalization" must be an instance of "str"!')

        if not isinstance(params['balance_dataset'], bool):
            raise TypeError('"balance_dataset" must be an instance of "bool"!')

        if not isinstance(params['drop_duplicates'], bool):
            raise TypeError('"drop_duplicates" must be an instance of "bool"!')

        if not isinstance(params['numerical_features'], list):
            raise TypeError('"numerical_features" must be an instance of list!')

        if not isinstance(params['categorical_features'], list):
            raise TypeError('"categorical_features" must be an instance of list!')

        if not isinstance(params['target_column'], str):
            raise TypeError('"target_column" must be an instance of "str"!')

        return

