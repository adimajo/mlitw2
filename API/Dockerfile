FROM python:3.8-slim

RUN pip install pipenv

ADD . /fake_score

WORKDIR /fake_score

EXPOSE 8000

RUN pipenv install --skip-lock

CMD pipenv run supervisord -n -c supervisor.d/API.conf
