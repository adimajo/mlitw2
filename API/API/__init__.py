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
from flask_restx import Resource, Namespace, fields, reqparse
from flask import jsonify
from loguru import logger

from API import __version__
from API.API.predict import predict

api = Namespace('')

my_output = api.model("output", {'version': fields.String})


@api.route('/version')
class Version(Resource):
    """
    Flask resource to spit current version
    """
    @api.marshal_with(my_output)
    def get(self):
        logger.debug("Successful GET")
        return {"version": __version__}


str_required = {
    'type': str,
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

predict_parser = reqparse.RequestParser()
predict_parser.add_argument('UserId',
                            help='Description of UserId.',
                            **str_required)
predict_parser.add_argument('Event',
                            help='Description of Event.',
                            **str_required)
predict_parser.add_argument('Category',
                            help='Description of Category.',
                            **str_required)


@api.route('/predict')
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
                - UserId
                - Event
                - Category
              properties:
                Var_1:
                  type: float
                  description: Description of Var_1.
                Var_2:
                  type: float
                  description: Description of Var_2.
                Var_3:
                  type: integer
                  description: Description of Var_3.
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

        if not result["is_fake_probability"] == np.nan:
            response = jsonify(result)
            response.status_code = 200
        else:
            response = jsonify("Model failed")
            response.status_code = 400

        return response
