import numpy as np


def predict_assemble(X, models, weights):
    """
    Prediction from ASSEMBLE model

    :param numpy.ndarray X: features of samples to predict
    :param models: list of weak learners
    :param weights: weights for each weak learner
    :return: class prediction for each point
    """
    if len(models) > 1:
        predict = np.sum(np.concatenate([(weight * model.predict(X)).reshape(-1, 1) for weight, model in zip(weights,
                                                                                                             models)],
                                        axis=1),
                         axis=1).reshape(-1, 1)
    else:
        predict = np.array([weight * model.predict(X) for weight, model in zip(weights, models)]).reshape(-1, 1)
    return predict
