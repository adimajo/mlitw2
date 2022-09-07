#!/usr/bin/env python3
import sys
import os
try:
    from loguru import logger
    import pickle
    import pandas as pd
    import sklearn as sk
except ModuleNotFoundError:
    print("Some required modules are not installed in your current environment.")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        return pd.DataFrame.from_dict({user: sum((1 for i, x in enumerate(chunk.values[:-1]) if x != chunk.values[i + 1]))
                                       for user, chunk in usergroups}, orient='index').rename(columns={0: 'n_changes'})
    except:
        logger.info("Something wrong with n_changes")
        return pd.DataFrame({'n_changes': [0] * len(df)}, index=df.index)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        logger.error(f"Usage: {sys.argv[0]} input_csv_file output_csv_file")
        sys.exit(1)

    logger.info("Reading test file")
    try:
        testfile = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        logger.error("The path seems incorrect")
        sys.exit(1)

    for variable in ['UserId', 'Category', 'Event']:
        if variable not in testfile.columns.to_list():
            logger.error(f"{variable} is missing")
            sys.exit(1)

    logger.info("Feature engineering")
    engineered_set = pd.concat([number_of_clicks(testfile),
                                ad_email(testfile),
                                number_of_categories(testfile),
                                total_time(testfile),
                                most_frequent(testfile),
                                n_changes(testfile),
                                testfile.groupby('UserId')['Fake'].first().to_frame()],
                               axis=1)

    logger.info("Unpickling model")
    try:
        with open(os.path.join(BASE_DIR, "models/encoder"), "rb") as enc:
            encoder = pickle.load(enc)
        with open(os.path.join(BASE_DIR, "models/model"), "rb") as mod:
            model = pickle.load(mod)
    except Exception as e:
        logger.error(f"Could not load model: {e}.")
        sys.exit(1)

    logger.info("Dividing by number of timeframes")
    engineered_set[["number_clicks", "ad_email", "n_categories", "n_changes"]] = engineered_set[
        ["number_clicks", "ad_email", "n_categories", "n_changes"]].div(engineered_set["total_time"], axis=0)
    engineered_set.drop('total_time', axis=1, inplace=True)

    logger.info("Onehotencoding")
    engineered_set_object = engineered_set[['Event', 'Category']]
    codes = encoder.transform(engineered_set_object).toarray()
    feature_names = encoder.get_feature_names_out()

    X = pd.concat([engineered_set[["number_clicks", "ad_email", "n_categories", "n_changes"]],
                   pd.DataFrame(codes, columns=feature_names, index=engineered_set.index).astype(int)], axis=1)

    logger.info("Predicting")
    predictions = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({"is_fake_probability": predictions}, index=engineered_set.index).reset_index().rename(
        columns={'index': 'UserId'})
    logger.info(f"Writing results to {sys.argv[2]}")
    result.to_csv(sys.argv[2])
    logger.info("Success")
    logger.info("Suggested threshold: 0.057")
    sys.exit(0)
