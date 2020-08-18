# Machine Learning interview

The following is the code I produced for a technical "homework" as part of an interview
for an ML Engineer role.

Note: I removed the git history and tried to "anonymize" the features.

May it serve you, be it for an interview or for your work!

Goal: predict whether a transaction is a fraud (positive class in what follows) or not.

Note: I removed some features, renamed them and purposely sampled the data.

## Questions

1. Implement ASSEMBLE.AdaBoost from [Exploiting Unlabeled Data in Ensemble Methods, Kristin P. Bennet et al](http://homepages.rpi.edu/~bennek/kdd-KristinBennett1.pdf).

1. Build a training pipeline with:
    * training of your implemented algorithm;
    * observe the resulting confusion matrix on your test set.

1. Optimal decision (i.e. probability threshold) given cost matrix.

1. Create an API which serves the prediction.

## Installing

`git clone https://github.com/adimajo/MLitw.git`

If `pipenv` is not installed:

`pip install pipenv`

Install dependencies (optionally use `--dev` flag for tests):

`pipenv install`

## Running tests

(If `--dev` dependencies installed)

`pipenv run coverage run -m pytest`

Coverage report:

`pipenv run coverage report`

Alternatively, using tox (see `tox.ini` - WARNING: all python versions must be available):

`pipenv run tox`

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

##### ASSEMBLE

POST at http://0.0.0.0:8000/train

```
{
    "model_name": "assemble_1.0",
    "cost_incurred_for_fraud": 15.0,
    "proportion_for_test": 0.2,
    "cross_validation": false,
    "weights": false,
    "weak_learner": "ASSEMBLE"
}
```

Response:
```
{
    "test_metrics": {
        "gini":NaN,
        "CI": "nan",
        "#samples": 1605,
        "#fraud": 40,
        "mean_optimal_threshold": 0.0,
        "TP": 0,
        "TN": 0,
        "FN": 40,
        "FP": 0,
        "detection_rate": 0.0,
        "precision rate":NaN,
        "F-score":NaN,
        "financial_gain": 29139.50734011332
    },
    "train_metrics": {
        "cross_validation": false,
        "gini":NaN,
        "CI": "nan",
        "#samples": 8395,
        "#fraud": 157,
        "mean_optimal_threshold": 0.0,
        "TP": 0,
        "TN": 1945,
        "FN": 157,
        "FP": 0,
        "detection_rate": 0.0,
        "precision rate":NaN,
        "F-score":NaN,
        "financial_gain": 116782.44418863454
    }
}
```

#### Predict

POST at http://0.0.0.0:8000/predict

```
{
    "model_name": "assemble_1.0",
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
    "score":NaN,
    "optimal_threshold_calculated": 0,
    "optimal_threshold_provided": null,
    "decision": "LEGIT"
}
```

## Docker

To build the docker image:

`docker build -t mlitw:1.0 .`

To run the docker image:

###### `docker run -d -p 8000:8000 -p 7000:7000 mlitw:1.0`

## Using this work

The API is available on port 8000.

### train endpoint

The train endpoint trains a model and stores it in the `model` folder.

### predict endpoint

The predict endpoint predicts the probability of a new transaction being a fraud
given all its characteristics, a model name (previously trained), and optionaly a threshold.

### spec endpoint

The spec endpoint provides the JSON Swagger of the API. You can use [petstore](https://petstore.swagger.io/)
to make it look better!

## Documentation

The documentation is available in the `docs` folder. It is also provided on port 7000.

To build the documentation, go to the `docs` folder and type `make html`.

## Optimal decision

From [The Foundations of Cost-Sensitive Learning, Charles Elkan](http://web.cs.iastate.edu/~honavar/elkan.pdf), 
it's clear that for fixed costs (i.e. per sample), the optimal decision is (as implemented in `API/models/fit`- NOT
in a separate method), M/(M+F).

It's logical since if F is high, you want to block a transaction for which the probability of being a fraud is relatively low.

Nevertheless, here the cost of rightly classifying a transaction is sample-dependant (the amount varies with the transaction).
In this case, it's unclear if this threshold is still ideal, or if more sophisticated approaches (e.g. [exemple-dependent cost-sensitive methods](http://albahnsen.github.io/CostSensitiveClassification/Intro.html)).

## Possible extensions

### Technical

* Docs and API in separate Dockers
* train and predict in separate APIs (different needs regarding resources and predict does not need the data)
* online learning pipeline: at least feeding back blocked transactions should be easy
* implementing importance sampling weights as in [Learning and Evaluating Classifiers under Sample Selection Bias, Zadrozny B.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.170&rep=rep1&type=pdf)
* having the possibility to choose between different types of weak learners for ASSEMBLE method
* having the possibility to search for hyperparameters of ASSEMBLE and weak learners
* better data model for ASSEMBLE model (currently implemented as a list of models and a list of weights)
* implement tests
* properly configure coverage (e.g. through `.coveragerc` to exclude `tests` and `site-pacakges`)
* ...

### Theory

Depending on p("BLOCKED" | x, y), the weak learner used (and if it's well specified), it might even be useless / a bad idea to use unlabelled data.
* implementing tests (everything is there but coverage is 0)
* check if there are clever things to do when a user is several times in the database and / or sample only one line for the model (i.i.d. assumption).
