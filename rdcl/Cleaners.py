import numpy as np
import pandas as pd

from sklearn import impute
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# baran/raha
# import raha
# from raha import Detection, Correction
# from raha.dataset import Dataset

from missingpy import MissForest
from impyute.imputation.cs import em as impEM


class Cleaners:
    def __init__(self, ):
        pass

    def mean_mode_imputer(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values with mean (numerical) and mode (categorical) values.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if not len(detections):
            print("No errors detected!")
            return dirty_dataset

        repair_dataset = dirty_dataset.copy()
        repair_dataset_final = dirty_dataset.copy()

        df_num = repair_dataset[numerical_features]

        num_imp = df_num.mean()

        if len(categorical_features):
            df_cat = repair_dataset[categorical_features]
            cat_imp = df_cat.mode().dropna().squeeze(axis=0)

            impute = pd.concat([num_imp, cat_imp], axis=0)
        else:
            impute = num_imp


        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = impute[repair_dataset.columns[col_i]]

        return repair_dataset_final

    def median_mode_imputer(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values with median (numerical) and mode (categorical) values.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if not len(detections):
            print("No errors detected!")
            return dirty_dataset

        repair_dataset = dirty_dataset.copy()
        repair_dataset_final = dirty_dataset.copy()

        df_num = repair_dataset[numerical_features]

        num_imp = df_num.median()

        if len(categorical_features):
            df_cat = repair_dataset[categorical_features]
            cat_imp = df_cat.mode().dropna().squeeze(axis=0)

            impute = pd.concat([num_imp, cat_imp], axis=0)
        else:
            impute = num_imp

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = impute[repair_dataset.columns[col_i]]

        return repair_dataset_final

    def knn_missForest_imputer_5(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using KNN regression with n_neigbors=5.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            imputer = impute.KNNImputer(missing_values=np.nan, n_neighbors=5)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def knn_missForest_imputer_10(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using KNN regression with n_neigbors=10.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            imputer = impute.KNNImputer(missing_values=np.nan, n_neighbors=10)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def knn_missForest_imputer_20(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using KNN regression with n_neigbors=20.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """
        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            imputer = impute.KNNImputer(missing_values=np.nan, n_neighbors=20)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def knn_missForest_imputer_50(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using KNN regression with n_neigbors=50.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            imputer = impute.KNNImputer(missing_values=np.nan, n_neighbors=50)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def em_missForest_imputer_50(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using Expectation-Maximization for loops=50.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            cols_with_nans = repair_dataset.columns[repair_dataset.isnull().any()]
            num_cols_with_nans = [col for col in cols_with_nans if col in numerical_features]

            repair_dataset[num_cols_with_nans] = impEM(repair_dataset[num_cols_with_nans].values.astype(np.float64),
                                                       loops=50)

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(np.float64), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def em_missForest_imputer_100(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using Expectation-Maximization for loops=100.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()
            cols_with_nans = repair_dataset.columns[repair_dataset.isnull().any()]
            num_cols_with_nans = [col for col in cols_with_nans if col in numerical_features]

            repair_dataset[num_cols_with_nans] = impEM(repair_dataset[num_cols_with_nans].values.astype(np.float64),
                                                       loops=100)

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(np.float64), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def decisionTree_missForest_imputer(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using decision trees.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            estimator = DecisionTreeRegressor(max_features='sqrt')

            imputer = IterativeImputer(estimator=estimator, missing_values=np.nan)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def bayesianRidge_missForest_imputer(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using Bayesian Ridge regression.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            estimator = BayesianRidge()

            imputer = IterativeImputer(estimator=estimator, missing_values=np.nan)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in repair_dataset[categorical_features].columns]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(
                repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def extraTrees_missForest_imputer(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using extra trees regression.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        if len(numerical_features):
            repair_dataset = dirty_dataset.copy()
            repair_dataset_final = dirty_dataset.copy()

            estimator = ExtraTreesRegressor(n_estimators=10)

            imputer = IterativeImputer(estimator=estimator, missing_values=np.nan)
            repair_dataset[numerical_features] = imputer.fit_transform(repair_dataset[numerical_features])

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            # repair_dataset = (repair_dataset.to_frame() if len(repair_dataset.shape) == 1 else repair_dataset)
            repair_dataset[categorical_features] = encoder.fit_transform(repair_dataset[categorical_features].values)

            columns = repair_dataset.columns
            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in categorical_features]

            # impute
            imputer = MissForest(missing_values=np.nan)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(repair_dataset[categorical_features].values)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def missForest_imputer_50(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using MissForest algorithm with tree depth=50.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        repair_dataset = dirty_dataset.copy()
        repair_dataset_final = dirty_dataset.copy()

        columns = repair_dataset.columns

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in categorical_features]

            # impute np.nan values
            imputer = MissForest(missing_values=np.nan, n_estimators=50)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)
            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(repair_dataset[categorical_features].values)

        else:
            imputer = MissForest(missing_values=np.nan, n_estimators=50)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float))

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def missForest_imputer_100(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using MissForest algorithm with tree depth=100.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        repair_dataset = dirty_dataset.copy()
        repair_dataset_final = dirty_dataset.copy()

        columns = repair_dataset.columns

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in categorical_features]

            # impute np.nan values
            imputer = MissForest(missing_values=np.nan, n_estimators=100)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)
            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        else:
            imputer = MissForest(missing_values=np.nan, n_estimators=100)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float))

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final

    def missForest_imputer_200(self, dirty_dataset, numerical_features, categorical_features, detections):
        """
        Imputation of missing values using MissForest algorithm with tree depth=200.

        Arguments:
            dirty_dataset (pd.DataFrame): dirty dataframe including errors
            numerical_features (str): numerical features present in the dirty data
            categorical_features (str): categorical features present in the dirty data
            detections (str): detection dictionary including row and colum indices of errors

        Returns:
            repair_dataset_final (pd.DataFrame): cleaned datasets by the respective method
        """

        repair_dataset = dirty_dataset.copy()
        repair_dataset_final = dirty_dataset.copy()

        columns = repair_dataset.columns

        if len(categorical_features):
            encoder = preprocessing.OrdinalEncoder()
            repair_dataset[categorical_features] = encoder.fit_transform(
                repair_dataset[categorical_features].values)

            cat_indices = [i for i, c in enumerate(repair_dataset.columns) if
                           c in categorical_features]

            # impute np.nan values
            imputer = MissForest(missing_values=np.nan, n_estimators=200)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float), cat_vars=cat_indices)

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)
            # decode encoded representation
            repair_dataset[categorical_features] = encoder.inverse_transform(
                repair_dataset[categorical_features].values)

        else:
            imputer = MissForest(missing_values=np.nan, n_estimators=200)
            repair_dataset = imputer.fit_transform(repair_dataset.values.astype(float))

            repair_dataset = pd.DataFrame(repair_dataset, columns=columns)

        for (row_i, col_i), dummy in detections.items():
            repair_dataset_final.iat[row_i, col_i] = repair_dataset.iat[row_i, col_i]

        return repair_dataset_final
