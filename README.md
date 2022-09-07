[![Python Flask docker](https://github.com/adimajo/MLitw2/actions/workflows/python-flask.yml/badge.svg)](https://github.com/adimajo/MLitw2/actions/workflows/python-flask.yml)
[![Build Status](https://app.travis-ci.com/adimajo/MLitw2.svg?token=opB6ydhp1rfhZkQiU4AY&branch=master)](https://app.travis-ci.com/adimajo/MLitw2)
[![Coverage status](https://codecov.io/gh/adimajo/MLitw2/branch/master/graph/badge.svg)](https://codecov.io/github/adimajo/MLitw2?branch=master)

# Machine Learning interview

The following is the code I produced for a technical "homework" as part of an interview
for an ML Engineer role.

Goal: predict whether a user is a bot / fake account (positive class in what follows) or not.

*Put the data in the `data/` folder.*

## Questions

A marketplace is being attacked by bots that produce fake clicks and leads.
The marketplace reputation might be affected if sellers get tons of fake leads and receive spam from bots.
On top of that, these bots introduce noise to our models in production that rely on user behavioural data.
We need to save [COMPANY]'s reputation detecting these fake users. To do so, we have a dataset of logs of a span
of five minutes. Each entry contains the user id (UserId), the action that a user made (Event), the category it
interacted with (Category) and a column (Fake) indicating if that user is fake (1 is fake, 0 is a real user).

## Answer

`PYTHONPATH=. python scripts/predict.py data/fake_users_test.csv data/fake_users_test_prob.csv`

## Installing

`git clone https://github.com/adimajo/MLitw2.git`

If `pipenv` is not installed:

`pip install pipenv`

Install dependencies (optionally use `--dev` flag for tests):

`pipenv install`

## Running tests

(If `--dev` dependencies installed)

`pipenv run coverage run -m pytest`

Coverage report:

`pipenv run coverage report`

## Running the API

### Debug mode

`export PYTHONPATH=.`

`python -m API/wsgi.py`

### gunicorn

(If on Windows, use `waitress`)

`gunicorn --worker-class gevent --workers 2 --bind 0.0.0.0:8000 API.wsgi:app`

### supervisor

(If on Windows, use `waitress` and `supervisor-win`)

`supervisord -c supervisor.d/API.conf`

### Examples

#### Train

##### Logistic regression

POST at http://0.0.0.0:8000/train

```
{
    "model_name": "logistic_1.0",
    "cost_incurred_for_fraud": 15.0,
    "proportion_for_test": 0.2,
    "cross_validation": false,
    "weights": false,
    "weak_learner": "LogisticRegression"
}
```

Response:
```
{
    "test_metrics": {
        "gini": 0.9439039797852178,
        "CI": "[0.87708979 1.        ]",
        "#samples": 1618,
        "#fraud": 35,
        "mean_optimal_threshold": 0.5568081098755129,
        "TP": 13,
        "TN": 1581,
        "FN": 22,
        "FP": 2,
        "detection_rate": 0.37142857142857144,
        "precision rate": 0.8666666666666667,
        "F-score": 0.52,
        "financial_gain": 29371.009370883305
    },
    "train_metrics": {
        "cross_validation": false,
        "gini": 0.9115651341396958,
        "CI": "[0.87906583 0.94406444]",
        "#samples": 6472,
        "#fraud": 159,
        "mean_optimal_threshold": 0.5580248150689585,
        "TP": 63,
        "TN": 6295,
        "FN": 96,
        "FP": 18,
        "detection_rate": 0.39622641509433965,
        "precision rate": 0.7777777777777778,
        "F-score": 0.525,
        "financial_gain": 117454.84921713671
    }
}
```

#### Predict

POST at http://0.0.0.0:8000/predict

```
{
    "model_name": "logistic_1.0",
    "Var_1": 10.0,
    "Var_2": 10.0,
    "Var_3": 4,
    "Var_4": 4,
    "Var_5": 10,
    "Var_6": 200,
    "Var_7": 20,
    "Var_8": 0
}
```

Response:
```
{
    "score": 0.023666641995322858,
    "optimal_threshold_calculated": 0.4,
    "optimal_threshold_provided": null,
    "decision": "LEGIT"
}
```

## Docker

To build the docker image locally:

`docker build -t mlitw2:1.0 .`

To run the docker image:

###### `docker run -d -p 8000:8000 -p 7000:7000 mlitw2:1.0`

## Using this work

The API is available on port 8000.

### train endpoint

The train endpoint trains a model and stores it in the `model` folder.

### predict endpoint

The predict endpoint predicts the probability of a new transaction being a fraud
given all its characteristics, a model name (previously trained), and optionaly a threshold.

## Documentation

The documentation is available in the `docs` folder. It is also provided on port 7000.

To build the documentation, go to the `docs` folder and type `make html`.

## Optimal decision

From [The Foundations of Cost-Sensitive Learning, Charles Elkan](http://web.cs.iastate.edu/~honavar/elkan.pdf), 
it's clear that for fixed costs (i.e. per sample), the optimal decision is (as implemented in `API/models/fit`), M/(M+F).

It's logical since if F is high, you want to block an ad for which the probability of being a fraud is relatively low.

## Possible extensions

### Technical

* train and predict in separate APIs (different needs regarding resources and predict does not need the data)
* online learning pipeline: at least feeding back blocked transactions should be easy
* implement more tests
* ...
