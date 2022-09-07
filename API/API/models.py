"""
handling models.

.. autosummary::
    :toctree:

    model_path
    ModelHandler
"""
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def model_path(model_name):
    """
    Generating or getting a model path

    :param str model_name: name of the model
    :return: path
    """
    return os.path.join(MODEL_DIR, "{}.pkl".format(model_name))


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

    def get_model(self, model_name):
        """
        Getter for model

        :param model_name: name of the model to get and load
        :return: model
        """
        if self.model is None:
            self = self.load(model_name)
        return self

    def load(self, model_name):
        """
        Load a model

        :param model_name: the name of the model to load
        :return: the model
        """
        with open(model_path(model_name=model_name), 'rb') as pickle_path:
            return pickle.load(pickle_path)
