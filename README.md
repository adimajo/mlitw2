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

#### Predict

POST at http://0.0.0.0:8000/predict

*Sketch*

```
{
    "UserId": [0],
    "Event": [0],
    "Category": [0],
}
```

Response:
```
{
    "UserId": [0],
    "is_fake_probability": [0.023666641995322858]
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
