from flask import Blueprint
from flask import render_template,request

from .forms import LoginForm

#blueprint permimte usar url
pagina = Blueprint('pagina',__name__)


#Error 404

@pagina.app_errorhandler(404)
def pagina_no_encontrada(error):#parametro obligatorio, todo error regresa dos parametros.
    return render_template('errores/404.html'), 404

@pagina.route('/')
def index():
    return render_template('index.html', title='Inicio')

    
@pagina.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm(request.form)
    if request.method =='POST':
        #se imprime los datos
        print(form.username.data)
        print(form.password.data)
        #se imprime en consola
        print("¡Una Nueva secion creada!")

    return render_template('auth/login.html', title='Login',form=form)