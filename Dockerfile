FROM python:3.7.2

RUN pip install pipenv

ADD . /fraud_score

WORKDIR /fraud_score

EXPOSE 7000

EXPOSE 8000

RUN pipenv install --skip-lock --dev

CMD pipenv run supervisord -n -c supervisor.d/API.conf
