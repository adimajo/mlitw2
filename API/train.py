"""
training functions for API
"""
import numpy as np
import sklearn as sk
from copy import deepcopy
from loguru import logger
from .predict_assemble import predict_assemble


def train(model, X, y, weak_learner):
    """
    Trains the model

    :param model: model to train
    :param X: features
    :type X: numpy.ndarray
    :param y: labels
    :type y: numpy.ndarray
    :param weak_learner: labels
    :type weak_learner: str
    :return: trained model
    """

    if not weak_learner == "ASSEMBLE":
        logger.info("Began training")
        model.fit(X=X, y=y)
        logger.info("Training finished")
        return model
    else:
        boosted_model = []
        boosted_model_weights = []
        y_pseudo = np.copy(y)
        T = 50
        alpha = 1.0
        beta = 0.9
        # Step 2: weights
        weights = (y == 0) * beta / np.sum((y == 0).flatten()) + \
            np.logical_not(y == 0) * beta / np.sum(np.logical_not(y == 0).flatten())
        # Step 3: KNN
        knn = sk.neighbors.KNeighborsClassifier()
        knn.fit(X=X[np.logical_not((y == 0).flatten()), :], y=y[np.logical_not((y == 0).flatten())].ravel())
        y_pseudo[(y == 0).flatten()] = knn.predict(X=X[(y == 0).flatten(), :]).reshape(-1, 1)
        # Step 4: weak learner
        model.fit(X=X, y=y_pseudo)
        # Step 5:Loop
        for _ in range(1, T):
            # Step 6: predictions of weak learner
            y_hat = model.predict(X=X).reshape(-1, 1)
            # Step 7: epsilon
            epsilon = np.sum(weights[np.logical_not(y_hat == y_pseudo)])
            # Step 8: stopping rule
            if epsilon > 0.5:
                break
            # Step 9: weight of new weak learner
            w_t = epsilon * np.log((1 - epsilon) / epsilon)  # TODO: enlever epsilon
            if w_t == np.nan:  # Cas rare de sur-apprentissage d'un weak learner
                w_t = 0.1  # Arbitraire
            # Step 10: new model
            boosted_model.append(deepcopy(model))
            boosted_model_weights.append(w_t)
            # Step 11: pseudo-classes (unclear if predictions have to be converted back to -1 / 1)
            predictions = predict_assemble(X=X,
                                           models=boosted_model,
                                           weights=boosted_model_weights)
            y_pseudo[(y == 0).flatten()] = predictions[(y == 0).flatten()]
            nominateur = np.array([alpha * np.exp(- y_pseudo_i * prediction) for y_pseudo_i,
                                                                                 prediction in zip(y_pseudo,
                                                                                                   predictions)])
            denominateur = sum(nominateur)
            # Step 12: cost function
            weights = nominateur / denominateur
            # Step 13: discarded
            # Step 14: new weak learner
            model.fit(X=X, y=y_pseudo)
        return [boosted_model, boosted_model_weights]
