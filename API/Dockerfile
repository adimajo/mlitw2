FROM python:3.8-slim

# copy project
COPY . API/

EXPOSE 8000

# install dependencies
RUN pip install --upgrade pip && pip install pipenv
RUN pipenv install --system --deploy
RUN chown nobody:nogroup /API

USER nobody

ENTRYPOINT gunicorn API.wsgi:app --bind 0.0.0.0:8000
