"""
prediction functions for API
"""
import numpy as np
import pandas as pd
from loguru import logger
from .models import ModelHandler
from .predict_assemble import predict_assemble


binary_to_decision = {
    0: "LEGIT",
    1: "FRAUD"
}


def predict(
        model_name,
        optimal_threshold,
        Var_1,
        Var_2,
        Var_3,
        Var_4,
        Var_5,
        Var_6,
        Var_7,
        Var_8
):
    """
    Predicts fraud for a new customer

    .. todo:: parse more elegantly kwargs

    :param model_name: A model name to store the resulting model
    :param optimal_threshold: A threshold to override the calculated one
    :param Var_1: Description of Var_1.
    :param Var_2: Description of Var_2.
    :param Var_3: Description of Var_3.
    :param Var_4: Description of Var_4.
    :param Var_5: Description of Var_5.
    :param Var_6: Description of Var_6.
    :param Var_7: Description of Var_7.
    :param Var_8: Description of Var_8.
        risky.
    :return: prediction as a score
    :rtype: float
    """
    if model_name is None:
        logger.error('model_name is required')

    df_to_predict = pd.DataFrame({
        "Var_1": Var_1,
        "Var_2": Var_2,
        "Var_3": Var_3,
        "Var_4": Var_4,
        "Var_5": Var_5,
        "Var_6": Var_6,
        "Var_7": Var_7,
        "Var_8": Var_8
    }, index=[0])

    try:
        modele = ModelHandler().get_model(model_name=model_name)
    except FileNotFoundError:
        logger.error('model_name did not match any existing trained model')
        return np.nan

    if modele.weak_learner == "ASSEMBLE":
        score = predict_assemble(X=df_to_predict.values, models=modele.model[0], weights=modele.model[1])
        score = score[0][0]
        decision = (score > 0) * 1
        # TODO: calculate the optimal threshold for ASSEMBLE method (not trivial since it's not a probability)
        optimal_threshold_calculated = 0
    else:
        score = modele.model.predict_proba(df_to_predict)[0][1]
        optimal_threshold_calculated = Var_1 / (Var_1 + modele.cost_incurred_for_fraud)
        if optimal_threshold is not None:
            decision = (score > optimal_threshold) * 1
        else:
            decision = (score > optimal_threshold_calculated) * 1

    dict_result = {
        "score": score,
        "optimal_threshold_calculated": optimal_threshold_calculated,
        "optimal_threshold_provided": optimal_threshold,
        "decision": binary_to_decision[decision]
    }

    return dict_result
