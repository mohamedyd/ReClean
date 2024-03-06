import os
import copy
import h5py
import warnings
import numpy as np
import pandas as pd
import pickle
import time
from collections import deque
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from imblearn import over_sampling, under_sampling

from sklearn import impute
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics, preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from missingpy import MissForest
from impyute.imputation.cs import em as impEM

from Cleaners import Cleaners
from DataPreprocessing import DataPreprocessing

warnings.filterwarnings("ignore")



class RDCL:
    def __init__(self, rdcl_params: dict):

        self.dirty_dataset = rdcl_params['dirty_dataset']
        self.valid_dataset = rdcl_params['valid_dataset']

        self.numerical_features = rdcl_params['dataloader_params']['numerical_features']
        self.categorical_features = rdcl_params['dataloader_params']['categorical_features']
        self.target_column = rdcl_params['dataloader_params']['target_column']

        self.feature_vectors = rdcl_params['feature_vectors']
        self.detection_dictionary = rdcl_params['detection_dictionary']

        self.dataloader_params = rdcl_params['dataloader_params']

        self.cleaner_list = rdcl_params['cleaner_list']


        # network parameters
        self.cleaner_rl_model_neurons = rdcl_params['cleaner_selector_neurons']  # ordered list of neurons for cleaner-selector network
        self.activation = rdcl_params['activation']  # activation function for cleaner selector
        self.predictor = rdcl_params['predictor_algorithm']  # predictor algorithm: 'linear', 'logistic' or 'mlp'
        self.predictor_model = rdcl_params['predictor_model'] # LogisticRegression(), LinearRegression(), or a tensorflow MLP model

        self.problem = rdcl_params['dataloader_params']['problem']  # classification / regression
        self.normalization = rdcl_params['dataloader_params']['normalization']
        self.scaler = (preprocessing.StandardScaler() if self.normalization == 'standard' else preprocessing.MinMaxScaler())

        # One-hot encoding in case of multiclass
        # if self.problem == 'classification':
        #     self.n_class = len(np.unique(self.dirty_dataset[self.target_column].values))
        #
        #     self.y_train_onehot = np.eye(len(self.y_train))[self.y_train.astype(int), :self.n_class].squeeze()
        #     y_val_onehot = np.eye(len(y_val))[y_val.astype(int), :self.n_class].squeeze()
        # else:
        #     self.y_train_onehot = None
        #     y_val_onehot = None

        # performance evaluation metric: roc-auc, mse etc.
        self.eval_metric = rdcl_params['eval_metric']

        # deque to store rewards for moving average window baseline
        self.baseline_rewards = deque(maxlen=rdcl_params['baseline_window'])

        # Training parameters
        self.n_epochs = rdcl_params['n_epochs']  # outer epochs for training of RDCL framework
        self.batch_size = rdcl_params['batch_size']
        self.n_iterations = rdcl_params['n_iterations']  # epochs for training Predictor (MLP)

        self.threshold = rdcl_params['exp_threshold']
        self.epsilon = 1e-8 # avoids zero log

        # List of pre-cleaned datasets
        self.cleaned_datasets = []


        # Prepare dirty data for cleaning
        for (row_i, col_i), dummy in self.detection_dictionary.items():
            self.dirty_dataset.iat[row_i, col_i] = np.nan

        self.dirty_dataset = self.dirty_dataset.apply(pd.to_numeric, errors='coerce')
        self.dirty_dataset[self.numerical_features] = self.scaler.fit_transform(self.dirty_dataset[self.numerical_features])


        # Execution of respective cleaners on dirty dataset
        for cleaner_idx in range(len(self.cleaner_list)):
            dd = self.dirty_dataset.copy()

            repaired_df = self.clean_errors(cleaner_idx, dd, self.numerical_features,
                                            self.categorical_features, self.detection_dictionary)
            self.cleaned_datasets.append(repaired_df)

        tf.keras.backend.set_floatx('float64')


    def clean_errors(self, cleaner_idx, sampled_dirty_df, numerical_features, categorical_features, sampled_detections):

        """Execute cleaners with respect to the chosen index. Refer to Cleaners.py for explanation of the parameters."""

        if cleaner_idx == 0:
            cleaner = Cleaners()
            repaired_dataset = cleaner.mean_mode_imputer(sampled_dirty_df, numerical_features,
                                                         categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 1:
            cleaner = Cleaners()
            repaired_dataset = cleaner.median_mode_imputer(sampled_dirty_df, numerical_features,
                                                           categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 2:
            cleaner = Cleaners()
            repaired_dataset = cleaner.knn_missForest_imputer_5(sampled_dirty_df, numerical_features,
                                                                categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 3:
            cleaner = Cleaners()
            repaired_dataset = cleaner.knn_missForest_imputer_20(sampled_dirty_df, numerical_features,
                                                                 categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 4:
            cleaner = Cleaners()
            repaired_dataset = cleaner.knn_missForest_imputer_50(sampled_dirty_df, numerical_features,
                                                                 categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 5:
            cleaner = Cleaners()
            repaired_dataset = cleaner.em_missForest_imputer_50(sampled_dirty_df, numerical_features,
                                                                categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 6:
            cleaner = Cleaners()
            repaired_dataset = cleaner.em_missForest_imputer_100(sampled_dirty_df, numerical_features,
                                                                 categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 7:
            cleaner = Cleaners()
            repaired_dataset = cleaner.bayesianRidge_missForest_imputer(sampled_dirty_df, numerical_features,
                                                                        categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 8:
            cleaner = Cleaners()
            repaired_dataset = cleaner.missForest_imputer_50(sampled_dirty_df, numerical_features,
                                                             categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 9:
            cleaner = Cleaners()
            repaired_dataset = cleaner.missForest_imputer_100(sampled_dirty_df, numerical_features,
                                                              categorical_features, sampled_detections)
            return repaired_dataset

        elif cleaner_idx == 10:
            cleaner = Cleaners()
            repaired_dataset = cleaner.missForest_imputer_200(sampled_dirty_df, numerical_features,
                                                              categorical_features, sampled_detections)
            return repaired_dataset


    def rl_cleaner(self, input_shape):

        """Build the cleaner selector (MLP) network."""

        x_input = tf.keras.Input(shape=input_shape)

        x = x_input

        for neurons in self.cleaner_rl_model_neurons:
            x = tf.keras.layers.Dense(units=neurons, activation=self.activation)(x)

        output = tf.keras.layers.Dense(len(self.cleaner_list), activation=tf.nn.softmax)(x)

        cleaner_model = tf.keras.Model(x_input, output)
        return cleaner_model


    def loss_fn(self, cleaner_probs, action, reward):

        """Calculate loss with respect to cleaner probabilities and selected actions."""

        # creates a tensor of probabilities for the actions taken
        dist_probs = tf.convert_to_tensor([s[i] for s, i in zip(cleaner_probs, action)], dtype=tf.float64)

        # calculates the log of the probabilities of the actions taken (adding a small constant self.epsilon to avoid taking log of zero) and sums them all together.
        #  This is a common pattern in RL to convert probabilities into a form that's easier to work with.
        prob = tf.reduce_sum(tf.math.log(dist_probs + self.epsilon))
        
        # explore when: threshold < average prob of all selected cleaners < 1 - threshold
        loss = -prob * reward + \
               1e3 * (tf.maximum(tf.reduce_mean(dist_probs) - self.threshold, 0) +
                      tf.maximum((1 - self.threshold) - tf.reduce_mean(dist_probs), 0))
        return loss

    def train_rl_cleaner(self):
        """Train RDCL framework."""

        # Build cleaner selector network
        self.cleaner_nn = self.rl_cleaner(input_shape=self.feature_vectors.shape[-1])

        optim = tf.keras.optimizers.Adam()  # optimizer for the framework

        # dirty and clean indices
        dirty_row_idx = np.unique([i for i, j in self.detection_dictionary.keys()]).tolist()  # rows containing errors
        clean_row_idx = list(set(dirty_row_idx).symmetric_difference(self.dirty_dataset.index))


        # Training: Cleaner selector + Predictor
        for epoch in range(self.n_epochs):

            # randomly sample a batch
            sample_idx = np.random.choice(len(self.dirty_dataset),
                                          min(self.batch_size, len(self.dirty_dataset)),
                                          replace=False).astype(np.int32).tolist()

            # create sampled clean and dirty dataframes, with new indices
            sampled_dirty_df = self.dirty_dataset.copy().iloc[sample_idx, :].reset_index(drop=True)

            # Recreate the detection dictionary with respect to new batch indices
            sampled_detections = {(sample_idx.index(i), j): value for (i, j), value in
                                  self.detection_dictionary.items()
                                  if i in sample_idx}

            dirty_idx = [sample_idx.index(i) for i in sample_idx if i in dirty_row_idx]

            # Sample feature vectors for the batch
            feature_vectors_batch = self.feature_vectors[sample_idx]

            # Output and sample cleaners from cleaner selector network
            cleaner_probs = self.cleaner_nn(feature_vectors_batch).numpy()
            dist = tfp.distributions.Categorical(probs=cleaner_probs, dtype=tf.float64)  # distribution of cleaner probs
            cleaner_indices = tf.cast(dist.sample(), tf.int32).numpy()  # Sample a cleaner w.r.t. each sample

            print('argmax cleaners: ',set(np.argmax(cleaner_probs, axis=1)))


            # Dummy repaired dataset (to be filled w.r.t. selected cleaners)
            repaired_dataset = pd.DataFrame(np.empty_like(sampled_dirty_df),
                                            columns=sampled_dirty_df.columns)

            # Execute cleaners on dirty batch: replace the samples of pre-cleaned datasets with the dummy dataset's based on selected cleaners
            for cl_idx in set(cleaner_indices):
                # repaired indices for each chosen cleaner
                repaired_indices = np.where(cleaner_indices == cl_idx)[0].tolist()

                cleaned_dataset_batch = self.cleaned_datasets[cl_idx].iloc[sample_idx].reset_index(drop=True)

                # assign cleaned samples by the respective cleaners to dummy dataset
                repaired_dataset.iloc[repaired_indices, :] = cleaned_dataset_batch.iloc[repaired_indices, :]

            # Preprocess sampled dataset to train Predictor network
            self.dataloader_params['train_dataset'], self.dataloader_params['valid_dataset'] = repaired_dataset, self.valid_dataset

            # Train-validation split: train (repaired), validation (clean)
            dl = DataPreprocessing(self.dataloader_params)
            x_train, y_train, x_val, y_val  = dl.preprocess_data()

            # The final version of dirty indices: some of them might have dropped during preprocessing, due to an undetected error remaining as a nan value.
            final_dirty_idx = list(set(dirty_idx).intersection(x_train.index))

            print('\nnumber of dirty samples in the batch: ', len(final_dirty_idx))
            print(f'Epoch: {epoch}, Cleaners chosen for dirty samples: {set(cleaner_indices[final_dirty_idx])}')

            x_train, y_train, x_val, y_val = x_train.values, y_train.values, x_val.values, y_val.values

            # One-hot encoding of labels in case of multiclass classification with MLP network
            if self.problem == 'classification':
                y_train_onehot = np.eye(len(y_train))[y_train.astype(int), :self.n_class].squeeze()
            else:
                y_train_onehot = None

            # Train predictor
            if self.predictor == 'logistic':
                predictor_model = copy.deepcopy(self.predictor_model)
                predictor_model.fit(x_train, y_train.ravel())
                y_pred_valid = predictor_model.predict_proba(x_val)

            elif self.predictor == 'linear':
                predictor_model = copy.deepcopy(self.predictor_model)
                predictor_model.fit(x_train, y_train.ravel())
                y_pred_valid = predictor_model.predict(x_val)  # predictions on sampled clean dataset

            elif self.predictor == 'mlp':
                predictor_model = tf.keras.models.clone_model(self.predictor_model)

                if self.problem == 'classification':
                    predictor_model.compile(optimizer=tf.optimizers.Adam(),
                                            loss=tf.losses.CategoricalCrossentropy(),
                                            metrics=[tf.keras.metrics.AUC(name='ROC', curve='ROC'),
                                                     tf.keras.metrics.AUC(name='PR', curve='PR')])
                elif self.problem == 'regression':
                    predictor_model.compile(optimizer=tf.optimizers.Adam(),
                                            loss=tf.losses.MeanSquaredError(),
                                            metrics=[tf.keras.metrics.MeanSquaredError()])

                # callbacks_ = [tf.keras.callbacks.ModelCheckpoint(filepath='./best_model.h5', save_best_only=True)]

                predictor_model.fit(x=x_train,
                                    y=(y_train_onehot if self.problem == 'classification' else y_train),
                                    batch_size=min(self.batch_size, len(x_train)),
                                    epochs=self.n_iterations,
                                    shuffle=True,
                                    # callbacks=callbacks_,
                                    verbose=False,
                                    validation_dataset=(x_val, y_val))

                predictor_model.load_weights('./best_model.h5')

                y_pred_valid = predictor_model.predict(x_val)


            # Predictor performance
            # f1 score
            if self.eval_metric == 'f1 score':
                predictor_perf = metrics.f1_score(y_true=y_val,
                                                  y_pred=(np.argmax(y_pred_valid, axis=1) if self.n_class > 2
                                                          else np.round(y_pred_valid[:, 1])),
                                                  average=('macro' if self.n_class>2 else 'binary'))
            # auc score
            elif self.eval_metric == 'roc auc':
                predictor_perf = metrics.roc_auc_score(y_true=(y_val_onehot if self.n_class>2 else y_val),
                                                       y_score=(y_pred_valid if self.n_class>2 else y_pred_valid[:, 1]))

            # accuracy
            elif self.eval_metric == 'accuracy':
                predictor_perf = metrics.accuracy_score(y_true=y_val,
                                                        y_pred=(np.argmax(y_pred_valid, axis=1) if self.n_class>2
                                                                else np.round(y_pred_valid[:, 1])))
            # mse
            elif self.eval_metric == 'mse':
                predictor_perf = -metrics.mean_squared_error(y_val, y_pred_valid)

            # rmse
            elif self.eval_metric == 'rmse':
                predictor_perf = -metrics.mean_squared_error(y_val, y_pred_valid, squared=False)

            # mae
            elif self.eval_metric == 'mae':
                predictor_perf = -metrics.mean_absolute_error(y_val, y_pred_valid)

            # r2 score
            elif self.eval_metric == 'r2':
                predictor_perf = metrics.r2_score(y_val, y_pred_valid)



            # RL reward: current predictor score - moving average of previous predictor scores
            baseline = (np.mean(self.baseline_rewards) if len(self.baseline_rewards) else predictor_perf)
            reward = predictor_perf - baseline

            self.baseline_rewards.append(predictor_perf) # store rewards to deque, automatically drops rewards from the head, if max length is exceeded



            # Forward pass
            with tf.GradientTape() as tape:
                probs = self.cleaner_nn(feature_vectors_batch[final_dirty_idx], training=True)
                loss = self.loss_fn(cleaner_probs=probs,
                                    action=cleaner_indices[final_dirty_idx].tolist(),
                                    reward=reward)  # compute loss

            # Backward pass
            grads = tape.gradient(loss, self.cleaner_nn.trainable_weights)

            # Update weights
            optim.apply_gradients(zip(grads, self.cleaner_nn.trainable_weights))

            # track progress
            print(f'Reward: {reward}, Loss: {loss}\n')
