import json

import pytest

from API.wsgi import app
from API import __version__


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_version(client, caplog):
    response = client.get('/version')
    assert json.loads(response.data.decode("utf-8"))["version"] == __version__
    assert "Successful GET" in caplog.records[-1].message
