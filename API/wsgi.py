

from flask import Flask
from flask_restx import Api
from flask_wtf.csrf import CSRFProtect

from API import __version__
from API.API import api as api_namespace


def create_app():
    """
    Creates the model serving Flask app
    :return: Flask app
    """
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    api = Api(
        title='WhiteApp Flask',
        version=__version__,
        description='API blanche pour framework Flask sous Python',
    )
    api.add_namespace(api_namespace)
    csrf = CSRFProtect()
    csrf.init_app(app)
    api.init_app(app)
    return app


app = create_app()


if __name__ == '__main__':  # pragma: no cover
    app = create_app()
    app.run(debug=True, port=7000)  # nosec
