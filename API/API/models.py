"""
handling models.

.. autosummary::
    :toctree:

    model_path
    ModelHandler
"""
import os
import pickle  # nosec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def model_path():
    """
    Generating or getting a model path

    :return: path
    """
    return os.path.join(MODEL_DIR, "model")


class ModelHandler(object):
    """
    Handling models
    """
    def __init__(self):
        """
        Just a model
        """
        self.model = None
        self.encoder = None

    def get_model(self):
        """
        Getter for model

        :return: model
        """
        if self.model is None or self.encoder is None:
            self.encoder, self.model = self.load()
        return self.encoder, self.model

    def load(self):
        """
        Load a model

        :return: the model
        """
        with open(model_path(), 'rb') as pickle_path:
            encoder = pickle.load(pickle_path)  # nosec
        with open(model_path(), 'rb') as pickle_path:
            model = pickle.load(pickle_path)  # nosec
        return encoder, model
