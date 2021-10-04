from io import BytesIO
from json import dumps
from flask import render_template, Blueprint, jsonify, request, make_response, Response
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
from matplotlib.figure import Figure
from apyori import apriori, dump_as_json          # Algoritmo apriori

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

@pagina.route('/read_csv', methods=['POST'])
def read_csv():
    if request.method =='POST':
        flask_file = request.files['file']

        confianza = request.form['confianza']
        soporte = request.form['soporte']
        elevacion = request.form['elevacion']
        print(confianza)
        print(soporte)
        print(elevacion)
        
        # if not a CSV file, return error
        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({"message": "Seleccione un archivo CSV"}), 400)

        movies_data = pd.read_csv(flask_file, header=None)

        #Se incluyen todas las transacciones en una sola lista
        transactions = movies_data.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'

        #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        transaction_list = pd.DataFrame(transactions)
        transaction_list['Frecuencia'] = 1

        #Se agrupa los elementos
        transaction_list = transaction_list.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        transaction_list['Porcentaje'] = (transaction_list['Frecuencia'] / transaction_list['Frecuencia'].sum()) #Porcentaje
        transaction_list = transaction_list.rename(columns={0 : 'Item'})


        # Se genera un gráfico de barras
        fig = Figure()
        fig.set_size_inches(16,20)
        ax = fig.subplots()
        ax.plot([2,2])
        ax.barh(transaction_list['Item'], transaction_list['Frecuencia'], color='blue')
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Item')
        ax.set_title('Frecuencia de los Items')
        ax.set_yticks(transaction_list['Item'])
        ax.set_yticklabels(transaction_list['Item'])
        ax.grid(True)
        fig.savefig(fname="app/static/img/practica_1.png", format='png')

        json_data = movies_data.head(5).to_json(orient='records')
        json_transactions = transaction_list.to_json(orient='records')

        return jsonify({
            "data": json_data,
            "transactions": json_transactions,
            "graph": "static/img/practica_1.png"
        })

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})
