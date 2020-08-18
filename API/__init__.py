"""
API module

Creating classes to learn a model, predict and
serve these predictions.

.. autosummary::
    :toctree:

    models
    train
    predict
    wsgi
    gini
"""
import numpy as np
from datetime import datetime
from flask_restful import reqparse, Resource
from flask import jsonify
from loguru import logger
import importlib
from .predict import predict
from .models import ModelHandler

str_required = {
    'type': str,
    'required': True,
    'default': None
}

datetime_required = {
    'type': lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M'),
    'required': True,
    'default': None
}

int_required = {
    'type': int,
    'required': True,
    'default': None
}

float_required = {
    'type': float,
    'required': True,
    'default': None
}

bool_required = {
    'type': bool,
    'required': True,
    'default': None
}


def check_between_0_1(x):
    """
    Checks if input is between 0 and 1.

    :param x: input
    :type x: float or int
    :return: x
    """
    if not 0 <= x <= 1:
        logger.error("Invalid value for proportion_for_test: must be between 0 and 1.")
    else:
        return x


b0_1_required = {
    'type': lambda x: check_between_0_1(x),
    'required': True,
    'default': None
}

b0_1_not_required = {
    'type': lambda x: check_between_0_1(x),
    'required': False,
    'default': None
}


def check_learner(learner):
    """
    Checks if the weak learner argument can be imported.

    :param learner: weak learner to use in classification
    :type learner: str
    :return: learner
    :rtype: str
    """
    if learner == "ASSEMBLE":
        logger.info("ASSEMBLE method chosen.")
        return learner
    try:
        getattr(importlib.import_module("sklearn.linear_model"), learner)
        logger.info("Specified learner could be imported.")
    except ImportError as e:
        logger.error("Specified learner is not installed / cannot be imported. " + str(e))
    return learner


learner_required = {
    'type': lambda x: check_learner(x),
    'required': True,
    'default': None
}


train_parser = reqparse.RequestParser()
train_parser.add_argument('model_name',
                          help="A model name to store the resulting model",
                          **str_required)
train_parser.add_argument('cost_incurred_for_fraud',
                          help="A cost for each fraud",
                          **float_required)
train_parser.add_argument('proportion_for_test',
                          help="A proportion of samples (between 0 and 1) to keep for test",
                          **b0_1_required)
train_parser.add_argument('cross_validation',
                          help="Perform cross_validation?",
                          **bool_required)
train_parser.add_argument('weights',
                          help="Use Importance Sampling weights?",
                          **bool_required)
train_parser.add_argument('weak_learner',
                          help="Which learner to use",
                          **learner_required)

predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_name',
                            help="A model name to predict",
                            **str_required)
predict_parser.add_argument('optimal_threshold',
                            help="A threshold to override the calculated one",
                            **b0_1_not_required)
predict_parser.add_argument('Var_1',
                            help='Description of Var_1.',
                            **float_required)
predict_parser.add_argument('Var_2',
                            help='Description of Var_2.',
                            **float_required)
predict_parser.add_argument('Var_3',
                            help="Description of Var_3.",
                            **int_required)
predict_parser.add_argument('Var_4',
                            help="Description of Var_4.",
                            **int_required)
predict_parser.add_argument('Var_5',
                            help='Description of Var_5.',
                            **int_required)
predict_parser.add_argument('Var_6',
                            help="Description of Var_6.",
                            **int_required)
predict_parser.add_argument('Var_7',
                            help='Description of Var_7.',
                            **int_required)
predict_parser.add_argument('Var_8',
                            help='Description of Var_8.',
                            **bool_required)


class Predictor(Resource):
    """
    Flask resource to predict
    """
    def post(self):
        """
        post method for Predictor resource: gets the new predictors in the request, predicts and outputs the score.
        ---
        parameters:
          - in: body
            name: body
            schema:
              id: Predict
              required:
                - model_name
                - Var_1
                - Var_2
                - Var_3
                - Var_4
                - Var_5
                - Var_6
                - Var_7
                - Var_8
              properties:
                model_name:
                  type: string
                  description: A model name to store the resulting model
                Var_1:
                  type: float
                  description: Description of Var_1.
                Var_2:
                  type: float
                  description: Description of Var_2.
                Var_3:
                  type: integer
                  description: Description of Var_3.
                Var_4:
                  type: integer
                  description: Description of Var_4.
                Var_5:
                  type: integer
                  description: Description of Var_5.
                Var_6:
                  type: integer
                  description: Description of Var_6.
                Var_7:
                  type: integer
                  description: Description of Var_7.
                Var_8:
                  type: boolean
                  description: Description of Var_8.
        responses:
            200:
                description: output of the model
            400:
                description: model found but failed
            500:
                description: all other server errors
        """
        kwargs = predict_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        result = predict(**kwargs)
        logger.info("Successfully predicted")

        if not result["score"] == np.nan:
            response = jsonify(result)
            response.status_code = 200
        else:
            response = jsonify("Model failed")
            response.status_code = 400

        return response


class Trainer(Resource):
    """
    Flask resource to train
    """
    def post(self):
        """
        post method for Trainer resource: will train a new model and store it.
        ---
        parameters:
          - in: body
            name: body
            schema:
              id: Predict
              required:
                - model_name
                - cost_incurred_for_fraud
                - proportion_for_test
                - cross_validation
                - weights
                - weak_learner
              properties:
                model_name:
                  type: string
                  description: Name to store the model.
                cost_incurred_for_fraud:
                    type: number
                    description: Cost (positive) when transaction turns out to be a fraud.
                proportion_for_test:
                    type: number
                    description: proportion (should be between 0 and 1) for test dataset
                cross_validation:
                    type: boolean
                    description: whether to perform cross validation (NOT IMPLEMENTED)
                weights:
                    type: boolean
                    description: whether to fit a model for 'blocked' and weight samples accordingly (NOT IMPLEMENTED)
                weak_learner:
                    type: string
                    description: class of weak_learner (for now only from sklearn.linear_model)
        responses:
            200:
                description: model has been trained and saved, outputs some metrics
        """
        kwargs = train_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        handler = ModelHandler()
        response = handler.fit(**kwargs)
        logger.info("Successfully trained")
        handler.save(**kwargs)
        logger.info("Successfully saved")

        return response
