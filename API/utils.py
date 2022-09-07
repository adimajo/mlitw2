"""
utils module

.. autosummary::
    :toctree:

    number_of_clicks
    ad_email
    number_of_categories
    total_time
    most_frequent
    n_changes
"""
import pandas as pd
from loguru import logger


def number_of_clicks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the number of clicks per user on the whole set

    :param pandas.DataFrame df: per user per timeframe dataframe with events
    :rtype: pandas.DataFrame
    """
    try:
        return df.loc[df.Event.isin(['click_ad', 'click_carrousel']), ['UserId']].groupby(
            ['UserId']).value_counts().reset_index()[['UserId', 0]].rename(
            columns={0: 'number_clicks'}).set_index('UserId')
    except:
        logger.info("Something wrong with number_of_clicks")
        return pd.DataFrame({'number_clicks': [0] * len(df)}, index=df.index)


def ad_email(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the number of click_ad and send_email per user on the whole set

    :param pandas.DataFrame df: per user per timeframe dataframe with events
    :rtype: pandas.DataFrame
    """
    try:
        return df.loc[df.Event.isin(['click_ad', 'send_email']), ['UserId']].groupby(
            ['UserId']).value_counts().reset_index()[['UserId', 0]].rename(
            columns={0: 'ad_email'}).set_index('UserId')
    except:
        logger.info("Something wrong with ad_email")
        return pd.DataFrame({'ad_email': [0] * len(df)}, index=df.index)


def number_of_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the number of number_of_categories per user on the whole set

    :param pandas.DataFrame df: per user per timeframe dataframe with categories
    :rtype: pandas.DataFrame
    """
    try:
        return df[['Category', 'UserId']].groupby(['UserId']).nunique().rename(
            columns={'Category': 'n_categories'})
    except:
        logger.info("Something wrong with number_of_categories")
        return pd.DataFrame({'n_categories': [0] * len(df)}, index=df.index)


def total_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the total_time per user on the whole set

    :param pandas.DataFrame df: per user per timeframe dataframe
    :rtype: pandas.DataFrame
    """
    try:
        return df.groupby('UserId').size().reset_index().set_index('UserId').rename(columns={0: 'total_time'})
    except:
        logger.info("Something wrong with total_time")
        return pd.DataFrame({'total_time': [1] * len(df)}, index=df.index)


def most_frequent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the most frequent event and category per user

    :param pandas.DataFrame df: per user per timeframe dataframe
    :rtype: pandas.DataFrame
    """
    try:
        return df.groupby('UserId')[['Event', 'Category']].agg(lambda x: pd.Series.mode(x)[0])
    except:
        logger.info("Something wrong with most_frequent")
        return pd.DataFrame({'Event': ['send_email'] * len(df), 'Category': ['Phone'] * len(df)}, index=df.index)


def n_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the number of changes of category per user

    :param pandas.DataFrame df: per user per timeframe dataframe
    :rtype: pandas.DataFrame
    """
    try:
        usergroups = df.groupby('UserId')['Category']
        return pd.DataFrame.from_dict({user: sum((1 for i, x in enumerate(
            chunk.values[:-1]) if x != chunk.values[i + 1])) for user, chunk in usergroups},
            orient='index').rename(columns={0: 'n_changes'})
    except:
        logger.info("Something wrong with n_changes")
        return pd.DataFrame({'n_changes': [0] * len(df)}, index=df.index)
