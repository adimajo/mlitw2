"""
handling models.

.. autosummary::
    :toctree:

    model_path
    data_path
    ModelHandler
"""
import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import importlib
from flask import jsonify
from scipy import stats
from .gini import delong_roc_variance
from .train import train
from .predict_assemble import predict_assemble

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')


def model_path(model_name):
    """
    Generating or getting a model path

    :param str model_name: name of the model
    :return: path
    """
    return os.path.join(MODEL_DIR, "{}.pkl".format(model_name))


def data_path(test=True):
    """
    Generating or getting a model path

    :return: path
    """
    if test:
        return os.path.join(DATA_DIR, "mle_fraud_test_sample.csv")
    return os.path.join(DATA_DIR, "mle_fraud_test.csv")


class ModelHandler(object):
    """
    Handling models
    """
    def __init__(self):
        """
        Just a model
        """
        self.model = None
        self.model_name = None,
        self.cost_incurred_for_fraud = None,
        self.proportion_for_test = None,
        self.cross_validation = None,
        self.weights = None,
        self.weak_learner = None

    def get_model(self, model_name):
        """
        Getter for model

        :param model_name: name of the model to get and load
        :return: model
        """
        if self.model is None:
            self = self.load(model_name)
        return self

    def save(self, **kwargs):
        """
        Saves a model
        """
        setattr(self, 'model_name', kwargs['model_name'])
        if self.model is not None:
            with open(model_path(model_name=self.model_name), 'wb') as pickle_path:
                pickle.dump(self, pickle_path)
        else:
            logger.error('Trying to save a model that has not been fitted.')

    def load(self, model_name):
        """
        Load a model

        :param model_name: the name of the model to load
        :return: the model
        """
        with open(model_path(model_name=model_name), 'rb') as pickle_path:
            return pickle.load(pickle_path)

    def fit(self, **kwargs):
        """
        Fits a new model

        :return: JSON response
        """

        for key in ('cost_incurred_for_fraud',
                    'proportion_for_test',
                    'cross_validation',
                    'weights',
                    'weak_learner'):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Get data from data directory
        logger.info("Reading the data")
        data = pd.read_csv(data_path(), sep=",")
        X = data[["Var_1",
                  "Var_2",
                  "Var_3",
                  "Var_4",
                  "Var_5",
                  "Var_6",
                  "Var_7",
                  "Var_8"]].values

        labels = data[["transaction_status"]].values

        # Transform
        if not self.weak_learner == "ASSEMBLE":
            rows = np.logical_or(labels == "LEGIT", labels == "FRAUD")
            y = labels[rows]
            X = X[rows.flatten(), :]
            y = (y == "FRAUD") * 1
            logger.info("Successfully read and transformed the data")
            logger.info("Train / test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.proportion_for_test)
            model = getattr(importlib.import_module("sklearn.linear_model"), self.weak_learner)()
        else:
            y = (labels == "FRAUD") * 1 - (labels == "LEGIT") * 1
            logger.info("Successfully read and transformed the data")
            logger.info("Train / test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.proportion_for_test)
            X_train = np.concatenate((X_train, X_test[(y_test == 0).flatten(), :]), axis=0)
            y_train = np.concatenate((y_train, y_test[(y_test == 0).flatten(), :]), axis=0)
            logger.info("Putting blocked transactions back in train")
            X_test = X_test[np.logical_not((y_test == 0).flatten()), :]
            y_test = y_test[np.logical_not((y_test == 0).flatten())]
            logger.info("Re-shuffling train")
            rand_int = np.random.randint(0, X_train.shape[0], size=X_train.shape[0])
            X_train = X_train[rand_int, :]
            y_train = y_train[rand_int]
            model = getattr(importlib.import_module("sklearn.tree"), "DecisionTreeClassifier")(max_depth=10)

        self.model = train(model=model, X=X_train, y=y_train, weak_learner=self.weak_learner)

        # Prob. predictions
        if not self.weak_learner == "ASSEMBLE":
            y_pred_test = self.model.predict_proba(X_test)[:, 1]
            y_pred_train = self.model.predict_proba(X_train)[:, 1]
        else:
            y_pred_test = predict_assemble(X=X_test, models=self.model[0], weights=self.model[1])
            y_labels_pred_test = (y_pred_test > 0) * 1
            y_pred_train = predict_assemble(X=X_train, models=self.model[0], weights=self.model[1])
            y_labels_pred_train = (y_pred_train > 0) * 1

        if not self.weak_learner == "ASSEMBLE":
            # Calculate optimal threshold
            optimal_threshold_test = X_test[:, 0] / (X_test[:, 0] + self.cost_incurred_for_fraud)
            optimal_threshold_train = X_train[:, 0] / (X_train[:, 0] + self.cost_incurred_for_fraud)
            y_labels_pred_test = y_pred_test > optimal_threshold_test
            y_labels_pred_train = y_pred_train > optimal_threshold_train

        # Calculate classification metrics
        TP_test = np.sum(np.logical_and(y_labels_pred_test == 1, y_test == 1))
        TN_test = np.sum(np.logical_and(y_labels_pred_test == 0, y_test == 0))
        FN_test = np.sum(np.logical_and(y_labels_pred_test == 0, y_test == 1))
        FP_test = np.sum(np.logical_and(y_labels_pred_test == 1, y_test == 0))
        detection_rate_test = TP_test / (TP_test + FN_test)
        precision_rate_test = TP_test / (TP_test + FP_test)
        F1_score_test = 2 * detection_rate_test * precision_rate_test / (detection_rate_test + precision_rate_test)

        TP_train = np.sum(np.logical_and(y_labels_pred_train == 1, y_train == 1))
        TN_train = np.sum(np.logical_and(y_labels_pred_train == 0, y_train == 0))
        FN_train = np.sum(np.logical_and(y_labels_pred_train == 0, y_train == 1))
        FP_train = np.sum(np.logical_and(y_labels_pred_train == 1, y_train == 0))
        detection_rate_train = TP_train / (TP_train + FN_train)
        precision_rate_train = TP_train / (TP_train + FP_train)
        F1_score_train = 2 * detection_rate_train * precision_rate_train / (detection_rate_train + precision_rate_train)

        if not self.weak_learner == "ASSEMBLE":
            # Test Gini and CI
            auc_test, auc_cov_test = delong_roc_variance(
                y_test,
                y_pred_test)
            auc_std_test = np.sqrt(auc_cov_test)
            lower_upper_q_test = np.abs(np.array([0, 1]) - (1 - .95) / 2)
            ci_test = stats.norm.ppf(lower_upper_q_test,
                                     loc=auc_test,
                                     scale=auc_std_test)
            ci_test[ci_test > 1] = 1
            # Train Gini and CI
            auc_train, auc_cov_train = delong_roc_variance(
                y_train,
                y_pred_train)
            auc_std_train = np.sqrt(auc_cov_train)
            lower_upper_q_train = np.abs(np.array([0, 1]) - (1 - .95) / 2)
            ci_train = stats.norm.ppf(
                lower_upper_q_train,
                loc=auc_train,
                scale=auc_std_train)
            ci_train[ci_train > 1] = 1
            financial_gain_train = np.sum(X_train[(y_train == 0).flatten(),
                                                  0]) - self.cost_incurred_for_fraud * np.sum(y_train == 1)
            financial_gain_test = np.sum(X_test[(y_test == 0).flatten(),
                                                0]) - self.cost_incurred_for_fraud * np.sum(y_test == 1)
        else:
            auc_test = np.nan
            ci_test = np.nan
            optimal_threshold_test = 0
            auc_train = np.nan
            ci_train = np.nan
            optimal_threshold_train = 0
            financial_gain_train = np.sum(X_train[(y_train == -1).flatten(),
                                                  0]) - self.cost_incurred_for_fraud * np.sum(y_train == 1)
            financial_gain_test = np.sum(X_test[(y_test == -1).flatten(),
                                                0]) - self.cost_incurred_for_fraud * np.sum(y_test == 1)

        result_dict = {
            "test_metrics": {
                "gini": 2 * auc_test - 1,
                "CI": str(2 * ci_test - 1),
                "#samples": X_test.shape[0],
                "#fraud": int(np.sum(y_test == 1)),
                "mean_optimal_threshold": float(np.mean(optimal_threshold_test)),
                "TP": int(TP_test),
                "TN": int(TN_test),
                "FN": int(FN_test),
                "FP": int(FP_test),
                "detection_rate": detection_rate_test,
                "precision rate": precision_rate_test,
                "F-score": F1_score_test,
                "financial_gain": financial_gain_test
            },
            "train_metrics": {
                "cross_validation": self.cross_validation,
                "gini": 2 * auc_train - 1,
                "CI": str(2 * ci_train - 1),
                "#samples": X_train.shape[0],
                "#fraud": int(np.sum(y_train == 1)),
                "mean_optimal_threshold": float(np.mean(optimal_threshold_train)),
                "TP": int(TP_train),
                "TN": int(TN_train),
                "FN": int(FN_train),
                "FP": int(FP_train),
                "detection_rate": detection_rate_train,
                "precision rate": precision_rate_train,
                "F-score": F1_score_train,
                "financial_gain": financial_gain_train
            }
        }

        response = jsonify(result_dict)
        response.status_code = 200
        return response
