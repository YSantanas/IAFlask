from flask import Flask
from .views import pagina

from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap=Bootstrap()


def create_app(config):
    #decidimos que se reinicie con la configuracion.
    app.config.from_object(config)
    #se puede trabajar con bootstrap através de la herencia.
    bootstrap.init_app(app)

    app.register_blueprint(pagina)
    return app