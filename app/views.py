from flask import Blueprint
from flask import render_template

#blueprint permimte usar url
pagina = Blueprint('pagina',__name__)


#Error 404

@pagina.app_errorhandler(404)
def pagina_no_encontrada(error):#parametro obligatorio, todo error regresa dos parametros.
    return render_template('errores/404.html'), 404

@pagina.route('/')
def index():
    return render_template('index.html', title='Inicio')