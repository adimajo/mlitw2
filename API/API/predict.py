"""
prediction functions for API
"""
import numpy as np
import pandas as pd
from loguru import logger

from API import utils
from API.API.models import ModelHandler


def predict(df):
    """
    Predicts fake for a new account

    :param pandas.DataFrame df: A model name to store the resulting model
    :return: prediction as a score
    :rtype: float
    """

    try:
        encoder, model = ModelHandler().get_model()
    except FileNotFoundError:
        logger.error('modele not found')
        return np.nan

    engineered_set = pd.concat([utils.number_of_clicks(df),
                                utils.ad_email(df),
                                utils.number_of_categories(df),
                                utils.total_time(df),
                                utils.most_frequent(df),
                                utils.n_changes(df),
                                df.groupby('UserId')['Fake'].first().to_frame()],
                               axis=1)

    engineered_set[["number_clicks", "ad_email", "n_categories", "n_changes"]] = engineered_set[
        ["number_clicks", "ad_email", "n_categories", "n_changes"]].div(engineered_set["total_time"], axis=0)
    engineered_set.drop('total_time', axis=1, inplace=True)

    engineered_set_object = engineered_set[['Event', 'Category']]
    codes = encoder.transform(engineered_set_object).toarray()
    feature_names = encoder.get_feature_names_out()

    X = pd.concat([engineered_set[["number_clicks", "ad_email", "n_categories", "n_changes"]],
                   pd.DataFrame(codes, columns=feature_names, index=engineered_set.index).astype(int)], axis=1)

    return model.predict_proba(X)[:, 1]
