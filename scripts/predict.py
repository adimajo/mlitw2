#!/usr/bin/env python3
import sys
import os
try:
    from loguru import logger
    import pickle  # nosec
    import pandas as pd
except ModuleNotFoundError:
    print("Some required modules are not installed in your current environment.")
    sys.exit(1)
try:
    from API import utils
except ModuleNotFoundError:
    print("API module not found. Try pointing PYTHONPATH to the root of MLitw2")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    engineered_set = pd.concat([utils.number_of_clicks(testfile),
                                utils.ad_email(testfile),
                                utils.number_of_categories(testfile),
                                utils.total_time(testfile),
                                utils.most_frequent(testfile),
                                utils.n_changes(testfile),
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
