"""
prediction functions for API
"""
import numpy as np
import pandas as pd
from loguru import logger
from .models import ModelHandler


binary_to_decision = {
    0: "LEGIT",
    1: "FRAUD"
}


def predict(
        df,
        model_name,
        optimal_threshold):
    """
    Predicts fake for a new account

    :param pandas.DataFrame df: A model name to store the resulting model
    :param str model_name: A model name to store the resulting model
    :param float optimal_threshold: A threshold
    :return: prediction as a score
    :rtype: float
    """
    if model_name is None:
        logger.error('model_name is required')

    try:
        modele = ModelHandler().get_model(model_name=model_name)
    except FileNotFoundError:
        logger.error('model_name did not match any existing trained model')
        return np.nan

    result = None
    return result
