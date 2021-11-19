from io import BytesIO
import io
from json import dumps
from os import name
from flask import render_template, Blueprint, jsonify, request, make_response, Response
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
from matplotlib.figure import Figure
from apyori import apriori, dump_as_json          # Algoritmo apriori
from uuid import uuid4
import base64


from .forms import LoginForm

# practica 3
# _____________________________________
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
# _____________________________________
# practica 4 y 6
# HAY QUE CARGARLO AL PROYECTO
# Para la generación de gráficas a partir de los datos
import matplotlib.pyplot as plt
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# %matplotlib inline
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
# --- practica 5 y 6
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#

# ------------------------------------------------
# _____     Practica 7 y 8     _________
# ------------------------------------------------
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, max_error, r2_score

# practica 8
from sklearn import model_selection

# ------------------------------------------------
# _____      Practica  9 y 10      _________
# ------------------------------------------------

# Se importan las bibliotecas necesarias para generar el modelo de regresión logística
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# _____            _______________       _________
# ------------------------------------------------


# blueprint permimte usar url
pagina = Blueprint('pagina', __name__)


# Error 404

@pagina.app_errorhandler(404)
# parametro obligatorio, todo error regresa dos parametros.
def pagina_no_encontrada(error):
    return render_template('errores/404.html'), 404


@pagina.route('/')
def index():
    return render_template('index.html', title='Inicio')


"""
@pagina.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm(request.form)
    if request.method =='POST':
        # se imprime los datos
        print(form.username.data)
        print(form.password.data)
        # se imprime en consola
        print("¡Una Nueva secion creada!")

    return render_template('auth/login.html', title='Login',form=form)
"""


@pagina.route('/read_csv', methods=['POST'])
def read_csv():
    if request.method == 'POST':

        flask_file = request.files['file']

        confianza = request.form['confianza']
        soporte = request.form['soporte']
        elevacion = request.form['elevacion']
        print(confianza)
        print(soporte)
        print(elevacion)

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        movies_data = pd.read_csv(flask_file, header=None)

        # Se incluyen todas las transacciones en una sola lista
        # -1 significa 'dimensión desconocida'
        transactions = movies_data.values.reshape(-1).tolist()

        # Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        transaction_list = pd.DataFrame(transactions)
        transaction_list['Frecuencia'] = 1

        # Se agrupa los elementos
        transaction_list = transaction_list.groupby(by=[0], as_index=False).count(
        ).sort_values(by=['Frecuencia'], ascending=True)  # Conteo
        transaction_list['Porcentaje'] = (
            transaction_list['Frecuencia'] / transaction_list['Frecuencia'].sum())  # Porcentaje
        transaction_list = transaction_list.rename(columns={0: 'Item'})

        # Se genera un gráfico de barras
        fig = Figure()
        fig.set_size_inches(16, 20)
        ax = fig.subplots()
        ax.plot([2, 2])
        ax.barh(transaction_list['Item'],
                transaction_list['Frecuencia'], color='blue')
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Item')
        ax.set_title('Frecuencia de los Items')
        ax.set_yticks(transaction_list['Item'])
        ax.set_yticklabels(transaction_list['Item'])
        ax.grid(True)

        nombre_temporal_uuid = str(uuid4().hex)
        nombre_temporal_png = 'app/static/img/' + nombre_temporal_uuid + '.png'

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png, format='png')

        json_data = movies_data.head(5).to_json(orient='records')
        json_transactions = transaction_list.to_json(orient='records')

        return jsonify({
            "data": json_data,
            "transactions": json_transactions,
            "graph": "static/img/practica_1.png"
        })

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})


@pagina.route('/read_csv2', methods=['POST'])
def read_csv2():
    if request.method == 'POST':
        input_a = int(request.form['input_a'])
        input_b = int(request.form['input_b'])
        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        archivo = pd.read_csv(flask_file)
        # print(archivo)

        # Se crea una matriz (dataframe) usando la lista.
        nuevoArchivo = pd.DataFrame(archivo)

        json_data = nuevoArchivo.to_json(orient='records')

        # _________________ENTRANDO EN PRACTICA3_________________
        # _______________________________________________________
        # -- se crea la matriz de distancia ECUCLIDIANA

        DstEuclidiana = cdist(
            archivo.iloc[0:10], archivo.iloc[0:10], metric='euclidean')
        MEuclidiana = pd.DataFrame(DstEuclidiana)
        # print(MEuclidiana)  # revisar que s emuestre en consola
        # se desea que se muestre la distancia entre dos objetos
        # esto da un valor numerico.

        tabla_euclidiana = MEuclidiana.to_json(orient='records')

        # se solicita el elemento, a comparar 1 y 2
        Objeto1 = archivo.iloc[input_a]
        Objeto2 = archivo.iloc[input_b]
        dstEuclidiana = distance.euclidean(Objeto1, Objeto2)
        # print(dstEuclidiana)  # este es el valor flotante retornado,

        dist_euclidiana = float(dstEuclidiana)

        # -- se crea la matriz de distancia para Chebyshev

        DstChebyshev = cdist(
            archivo.iloc[0:10], archivo.iloc[0:10], metric='chebyshev')
        MChebyshev = pd.DataFrame(DstChebyshev)
        # print(MChebyshev)

        tabla_chebyshev = MChebyshev.to_json(orient='records')

        # nuevamente se desea conocer la distancia entre dos objetos, seleccionados
        # por el usuario.
        # se solicita el elemento, a comparar 1 y 2
        Objeto1 = archivo.iloc[input_a]
        Objeto2 = archivo.iloc[input_b]
        dstChebyshev = distance.chebyshev(Objeto1, Objeto2)
        # print(dstChebyshev)  # valor numerico retornado

        dist_chebyshev = float(dstChebyshev)

        # -- se crea la matriz de distancia para Manhattan

        DstManhattan = cdist(
            archivo.iloc[0:10], archivo.iloc[0:10], metric='cityblock')
        MManhattan = pd.DataFrame(DstManhattan)
        # print(MManhattan)

        tabla_manhattan = MManhattan.to_json(orient='records')

        # nuevamente se desea conocer la distancia entre dos objetos, seleccionados
        # por el usuario.

        Objeto1 = archivo.iloc[input_a]
        # se solicita el elemento, a comparar 1 y 2
        Objeto2 = archivo.iloc[input_b]
        dstManhattan = distance.cityblock(Objeto1, Objeto2)
        # print(dstManhattan)

        dist_manhattan = float(dstManhattan)

        # -- se crea la matriz de distancia para Minkowski

        DstMinkowski = cdist(
            archivo.iloc[0:10], archivo.iloc[0:10], metric='minkowski', p=1.5)
        MMinkowski = pd.DataFrame(DstMinkowski)
        # print(MMinkowski)

        tabla_minkowski = MMinkowski.to_json(orient='records')

        # nuevamente se desea conocer la distancia entre dos objetos, seleccionados
        # por el usuario.

        Objeto1 = archivo.iloc[input_a]
        # se solicita el elemento, a comparar 1 y 2
        Objeto2 = archivo.iloc[input_b]
        dstMinkowski = distance.minkowski(Objeto1, Objeto2, p=1.5)
        # print(dstMinkowski)

        dist_minkowski = float(dstMinkowski)

        return jsonify({
            "data": json_data,
            "data_euclidiana": tabla_euclidiana,
            "data_chebyshev": tabla_chebyshev,
            "data_manhattan": tabla_manhattan,
            "data_minkowski": tabla_minkowski,
            "dist_euclidiana": dist_euclidiana,
            "dist_chebyshev": dist_chebyshev,
            "dist_manhattan": dist_manhattan,
            "dist_minkowski": dist_minkowski
        })

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})


# _________________________________________
# _____________Practica 4__________________
# _________________________________________
@pagina.route('/read_csv3', methods=['POST'])
def read_csv3():

    if request.method == 'POST':

        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        Hipoteca = pd.read_csv(flask_file)

        print(Hipoteca)

        print(Hipoteca.groupby('comprar').size())

        data_table_1 = pd.DataFrame(Hipoteca)

        json_data_1 = data_table_1.head(10).to_json(orient='records')

#_________________________________________

#NO SIRVE PARA QUITAR LOS ENCABEZADOS Y PONER INDICES

        Hipoteca2 = pd.read_csv(flask_file,header=None)


        MatrizHipoteca = np.array(Hipoteca2[['ingresos', 'gastos comunes', 'pago coche', 'gastos otros', 'ahorros', 'vivienda', 'estado civil', 'hijos', 'trabajo']])
        data_table_2 = pd.DataFrame(MatrizHipoteca)
        json_data_2 = data_table_2.to_json()


#__________________________________________





        # Se genera un gráfico de dispersión
        fig = Figure()
        fig.set_size_inches(4, 4)
        ax = fig.add_subplot(111)
        # sns.scatterplot(x='ahorros', y='ingresos',
        #                 data=Hipoteca, hue='comprar')

        ax.scatter(Hipoteca['ahorros'], Hipoteca['ingresos'],
                   c=Hipoteca['comprar'])
        ax.set_title('Grafico de dispersión')
        ax.set_xlabel('Ahorros')
        ax.set_ylabel('Ingresos')

        nombre_temporal_uuid = str(uuid4().hex)
        nombre_temporal_png = 'app/static/img/' + nombre_temporal_uuid + '.png'

        # Replace "app" to ""
        new_name = nombre_temporal_png.replace('app/', '')

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png,
            format='png',
        )

        # Retorna imagen en base64
        return jsonify({'status': 'success', 'data': json_data_1, 'image': new_name})

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})

#     Hipoteca.info()

#     print(Hipoteca.groupby('comprar').size())

# # **2) Selección de características**


#     sns.pairplot(Hipoteca, hue='comprar')
#     plt.show()


#     sns.scatterplot(x='ahorros', y='ingresos', data=Hipoteca, hue='comprar')
#     plt.title('Gráfico de dispersión')
#     plt.xlabel('Ahorros')
#     plt.ylabel('Ingresos')
#     plt.show()

#     CorrHipoteca = Hipoteca.corr(method='pearson')
#     CorrHipoteca

#     print(CorrHipoteca['ingresos'].sort_values(
#         ascending=False)[:10], '\n')  # Top 10 valores
#
#     plt.figure(figsize=(14, 7))
#     MatrizInf = np.triu(CorrHipoteca)
#     sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
#     plt.show()



#     MatrizHipoteca = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche',
#                               'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
#     pd.DataFrame(MatrizHipoteca)
# # MatrizHipoteca = Hipoteca.iloc[:, 0:9].values     #iloc para seleccionar filas y columnas según su posición

# # **3) Aplicación del algoritmo**

#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     # Se instancia el objeto StandardScaler o MinMaxScaler
#     estandarizar = StandardScaler()
#     # Se calculan la media y desviación y se escalan los datos
#     MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)

#     pd.DataFrame(MEstandarizada)

# # Se importan las bibliotecas de clustering jerárquico para crear el árbol

#     plt.figure(figsize=(10, 7))
#     plt.title("Casos de hipoteca")
#     plt.xlabel('Hipoteca')
#     plt.ylabel('Distancia')
#     Arbol = shc.dendrogram(shc.linkage(
#         MEstandarizada, method='complete', metric='euclidean'))
# #plt.axhline(y=5.4, color='orange', linestyle='--')
# # Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)

# # Se crean las etiquetas de los elementos en los clústeres
#     MJerarquico = AgglomerativeClustering(
#         n_clusters=7, linkage='complete', affinity='euclidean')
#     MJerarquico.fit_predict(MEstandarizada)
#     MJerarquico.labels_

#     Hipoteca = Hipoteca.drop(columns=['comprar'])
#     Hipoteca['clusterH'] = MJerarquico.labels_
#     Hipoteca

# # Cantidad de elementos en los clusters
#     Hipoteca.groupby(['clusterH'])['clusterH'].count()

#     Hipoteca[Hipoteca.clusterH == 6]

#     CentroidesH = Hipoteca.groupby('clusterH').mean()
#     CentroidesH

#     plt.figure(figsize=(10, 7))
#     plt.scatter(MEstandarizada[:, 0],
#                 MEstandarizada[:, 1], c=MJerarquico.labels_)
#     plt.grid()
#     plt.show()
# # _________________________________________
# # _____________Practica 5__________________
# # _________________________________________
# # COMENTARIO PARA CENTRARSE EN UNA PRACTICA


# """

@pagina.route('/read_csv4', methods=['POST'])
def read_csv4():

    if request.method == 'POST':

        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        Hipoteca = pd.read_csv(flask_file)

        print(Hipoteca)

        print(Hipoteca.groupby('comprar').size())

        data_table_3 = pd.DataFrame(Hipoteca)
        json_data3 = data_table_3.head(5).to_json(orient='records')
        
        
        
                # Se genera un gráfico de dispersión
        fig = Figure()
        fig.set_size_inches(4, 4)
        ax = fig.add_subplot(111)
        # sns.scatterplot(x='ahorros', y='ingresos',
        #                 data=Hipoteca, hue='comprar')

        ax.scatter(Hipoteca['ahorros'], Hipoteca['ingresos'],
                   c=Hipoteca['comprar'])
        ax.set_title('Grafico de dispersión')
        ax.set_xlabel('Ahorros')
        ax.set_ylabel('Ingresos')

        nombre_temporal_uuid = str(uuid4().hex)
        nombre_temporal_png = 'app/static/img/' + nombre_temporal_uuid + '.png'

        # Replace "app" to ""
        new_name = nombre_temporal_png.replace('app/', '')

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png,
            format='png',
        )

        # Retorna imagen en base64
        return jsonify({'status': 'success', 'data': json_data3, 'graph': new_name})

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})
        
# #### **2) Selección de características**

#     sns.pairplot(Hipoteca, hue='comprar')
#     plt.show()

#     sns.scatterplot(x='ahorros', y ='ingresos', data=Hipoteca, hue='comprar')
#     plt.title('Gráfico de dispersión')
#     plt.xlabel('Ahorros')
#     plt.ylabel('Ingresos')
#     plt.show()


#     CorrHipoteca = Hipoteca.corr(method='pearson')
#     CorrHipoteca

#     print(CorrHipoteca['ingresos'].sort_values(ascending=False)[:10], '\n')   #Top 10 valores

#     plt.figure(figsize=(14,7))
#     MatrizInf = np.triu(CorrHipoteca)
#     sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
#     plt.show()


#     MatrizHipoteca = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
#     pd.DataFrame(MatrizHipoteca)
# #MatrizHipoteca = Hipoteca.iloc[:, 0:9].values     #iloc para seleccionar filas y columnas según su posición

# #### **3) Aplicación del algoritmo**


#     estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
#     MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos

#     pd.DataFrame(MEstandarizada)


# #Definición de k clusters para K-means
# #Se utiliza random_state para inicializar el generador interno de números aleatorios
#     SSE = []
#     for i in range(2, 12):
#         km = KMeans(n_clusters=i, random_state=0)
#         km.fit(MEstandarizada)
#         SSE.append(km.inertia_)

# #Se grafica SSE en función de k
#     plt.figure(figsize=(10, 7))
#     plt.plot(range(2, 12), SSE, marker='o')
#     plt.xlabel('Cantidad de clusters *k*')
#     plt.ylabel('SSE')
#     plt.title('Elbow Method')
#     plt.show()


# # !pip install kneed


#     kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
#     kl.elbow

#     plt.style.use('ggplot')
#     kl.plot_knee()

# #Se crean las etiquetas de los elementos en los clusters
#     MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
#     MParticional.predict(MEstandarizada)
#     MParticional.labels_

#     Hipoteca = Hipoteca.drop(columns=['comprar'])
#     Hipoteca['clusterP'] = MParticional.labels_
#     Hipoteca

# #Cantidad de elementos en los clusters
#     Hipoteca.groupby(['clusterP'])['clusterP'].count()

#     Hipoteca[Hipoteca.clusterP == 0]

# #Obtención de los centroides
#     CentroidesP = Hipoteca.groupby('clusterP').mean()
#     CentroidesP


# # Gráfica de los elementos y los centros de los clusters

#     plt.rcParams['figure.figsize'] = (10, 7)
#     plt.style.use('ggplot')
#     colores=['red', 'blue', 'green', 'yellow']
#     asignar=[]
#     for row in MParticional.labels_:
#         asignar.append(colores[row])

#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(MEstandarizada[:, 0],
#             MEstandarizada[:, 1],
#             MEstandarizada[:, 2], marker='o', c=asignar, s=60)
#     ax.scatter(MParticional.cluster_centers_[:, 0],
#             MParticional.cluster_centers_[:, 1],
#             MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
#     plt.show()
# #_________________________________________
# #_____________Practica 7__________________
# #_________________________________________


# @pagina.route('/read_csv6', methods=['POST'])
# def read_csv6():

# #### **1) Importar las bibliotecas necesarias y los datos**

#     from google.colab import files
#     files.upload()

#     RGeofisicos = pd.read_csv('RGeofisicos.csv')
#     RGeofisicos

# #### **2) Gráfica de las mediciones de aceite**

#     plt.figure(figsize=(20, 5))
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC1'], color='green', marker='o', label='RC1')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC2'], color='purple', marker='o', label='RC2')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC3'], color='blue', marker='o', label='RC3')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC4'], color='yellow', marker='o', label='RC4')
#     plt.xlabel('Profundidad / Pies')
#     plt.ylabel('Porcentaje / %')
#     plt.title('Registros geofísicos convencionales')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# #### **3) Aplicación del algoritmo**


# #Se seleccionan las variables predictoras (X) y la variable a pronosticar (Y)

# X_train = np.array(RGeofisicos[['Profundidad', 'RC1', 'RC2','RC3']])
# pd.DataFrame(X_train)

# Y_train = np.array(RGeofisicos[['RC4']])
# pd.DataFrame(Y_train)

# #Se entrena el modelo a través de una Regresión Lineal Múltiple

# RLMultiple = linear_model.LinearRegression()
# RLMultiple.fit(X_train, Y_train)                 #Se entrena el modelo


# #Se genera el pronóstico
# Y_pronostico = RLMultiple.predict(X_train)
# pd.DataFrame(Y_pronostico)

# RGeofisicos['Pronostico'] = Y_pronostico
# RGeofisicos

# #### **4) Obtención de los coeficientes, intercepto, error y Score**


#     print('Coeficientes: \n', RLMultiple.coef_)
#     print('Intercepto: \n', RLMultiple.intercept_)
#     print("Residuo: %.4f" % max_error(Y_train, Y_pronostico))
#     print("MSE: %.4f" % mean_squared_error(Y_train, Y_pronostico))
#     print("RMSE: %.4f" % mean_squared_error(Y_train, Y_pronostico, squared=False))  #True devuelve MSE, False devuelve RMSE
#     print('Score (Bondad de ajuste): %.4f' % r2_score(Y_train, Y_pronostico))

# #### **5) Conformación del modelo de pronóstico**


# #### **6) Proyección de los valores reales y pronosticados**


#     plt.figure(figsize=(20, 5))
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC1'], color='green', marker='o', label='RC1')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC2'], color='purple', marker='o', label='RC2')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC3'], color='blue', marker='o', label='RC3')
#     plt.plot(RGeofisicos['Profundidad'], RGeofisicos['RC4'], color='yellow', marker='o', label='RC4')
#     plt.plot(RGeofisicos['Profundidad'], Y_pronostico, color='red', marker='o', label='Pronóstico')
#     plt.xlabel('Profundidad / Pies')
#     plt.ylabel('Porcentaje / %')
#     plt.title('Registros geofísicos convencionales')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(20, 5))
#     plt.plot(RGeofisicos['Profundidad'], Y_pronostico, color='red', marker='o', label='Pronóstico')
#     plt.xlabel('Profundidad / Pies')
#     plt.ylabel('Porcentaje / %')
#     plt.title('Registros geofísicos convencionales')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# #### **7) Nuevos pronósticos**

#     ROS = pd.DataFrame({'Profundidad': [5680.5], 'RC1': [0.45], 'RC2': [0.64], 'RC3': [0.5]})
#     RLMultiple.predict(ROS)

# #_________________________________________
# #_____________Practica 8__________________
# #_________________________________________

# @pagina.route('/read_csv7', methods=['POST'])
# def read_csv7():
#     from google.colab import files
#     files.upload()

#     BCancer = pd.read_csv('WDBCOriginal.csv')
#     BCancer

# #### **2) Gráfica del área del tumor por paciente**
#     plt.figure(figsize=(20, 5))
#     plt.plot(BCancer['IDNumber'], BCancer['Area'], color='green', marker='o', label='Area')
#     plt.xlabel('Paciente')
#     plt.ylabel('Tamaño del tumor')
#     plt.title('Pacientes con tumores cancerígenos')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# #### **3) Selección de características**
#     plt.figure(figsize=(14,7))
#     MatrizInf = np.triu(BCancer.corr())
#     sns.heatmap(BCancer.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
#     plt.show()


# #### **4) Aplicación del algoritmo**
#     X = np.array(BCancer[['Texture',
#                         'Perimeter',
#                         'Smoothness',
#                         'Compactness',
#                         'Symmetry',
#                         'FractalDimension']])
#     pd.DataFrame(X)

# #['Radius', 'Texture', 'Perimeter', 'Smoothness', 'Compactness',	'Concavity', 'ConcavePoints', 'Symmetry',	'FractalDimension']

#     Y = np.array(BCancer[['Area']])
#     pd.DataFrame(Y)


#     X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
#                                                                         test_size = 0.2,
#                                                                         random_state = 1234,
#                                                                         shuffle = True)

#     pd.DataFrame(X_train)
# #pd.DataFrame(X_test)
#     pd.DataFrame(Y_train)
# #pd.DataFrame(Y_test)
#     RLMultiple = linear_model.LinearRegression()
#     RLMultiple.fit(X_train, Y_train)                 #Se entrena el modelo


# #Se genera el pronóstico
#     Y_Pronostico = RLMultiple.predict(X_test)
#     pd.DataFrame(Y_Pronostico)

#     r2_score(Y_test, Y_Pronostico)

# #### **5) Obtención de los coeficientes, intercepto, error y Score**

#     print('Coeficientes: \n', RLMultiple.coef_)
#     print('Intercepto: \n', RLMultiple.intercept_)
#     print("Residuo: %.4f" % max_error(Y_test, Y_Pronostico))
#     print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
#     print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
#     print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))


#     AreaTumor = pd.DataFrame({'Texture': [18.32], 'Perimeter': [66.82], 'Smoothness': [0.08142], 'Compactness': [0.04462], 'Symmetry': [0.2372], 'FractalDimension': [0.05768]})
#     RLMultiple.predict(AreaTumor)


# #_________________________________________
# #_____________Practica 9__________________
# #_________________________________________

# @pagina.route('/read_csv8', methods=['POST'])
# def read_csv8():

#     from google.colab import files
#     files.upload()

#     BCancer = pd.read_csv('WDBCOriginal.csv')
#     BCancer

#     print(BCancer.groupby('Diagnosis').size())

# #### **2) Selección de características**

#     sns.pairplot(BCancer, hue='Diagnosis')
#     plt.show()

# #plt.plot(BCancer['Radius'], BCancer['Perimeter'], 'b+')
#     sns.scatterplot(x='Radius', y ='Perimeter', data=BCancer, hue='Diagnosis')
#     plt.title('Gráfico de dispersión')
#     plt.xlabel('Radius')
#     plt.ylabel('Perimeter')
#     plt.show()

#     CorrBCancer = BCancer.corr(method='pearson')
#     CorrBCancer

#     plt.figure(figsize=(14,7))
#     MatrizInf = np.triu(BCancer.corr())
#     sns.heatmap(BCancer.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
#     plt.show()


#     BCancer = BCancer.replace({'M': 0, 'B': 1})
#     BCancer

#     print(BCancer.groupby('Diagnosis').size())

# #Variables predictoras
#     X = np.array(BCancer[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
# #X = BCancer.iloc[:, [3, 5, 6, 7, 10, 11]].values  #iloc para seleccionar filas y columnas según su posición
#     pd.DataFrame(X)

# #Variable clase
#     Y = np.array(BCancer[['Diagnosis']])
#     pd.DataFrame(Y)

#     plt.figure(figsize=(10, 7))
#     plt.scatter(X[:,0], X[:,1], c = BCancer.Diagnosis)
#     plt.grid()
#     plt.xlabel('Texture')
#     plt.ylabel('Area')
#     plt.show()

# #### **4) Aplicación del algoritmo**

#     X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
#                                                                                     test_size = 0.2,
#                                                                                     random_state = 1234,
#                                                                                     shuffle = True)

#     pd.DataFrame(X_train)

#     pd.DataFrame(Y_train)

# #Se entrena el modelo a partir de los datos de entrada
#     Clasificacion = linear_model.LogisticRegression()
#     Clasificacion.fit(X_train, Y_train)

# #Se generan las probabilidades

# #Predicciones probabilísticas de los datos de prueba
#     Probabilidad = Clasificacion.predict_proba(X_validation)
#     pd.DataFrame(Probabilidad)

# #Predicciones con clasificación final
#     Predicciones = Clasificacion.predict(X_validation)
#     pd.DataFrame(Predicciones)

# #Se calcula el exactitud promedio de la validación
#     Clasificacion.score(X_validation, Y_validation)

# #### **5) Validación del modelo**

# #Matriz de clasificación
#     Y_Clasificacion = Clasificacion.predict(X_validation)
#     Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),
#                                     Y_Clasificacion,
#                                     rownames=['Real'],
#                                     colnames=['Clasificación'])
#     Matriz_Clasificacion

# #Reporte de la clasificación
#     print("Exactitud", Clasificacion.score(X_validation, Y_validation))
#     print(classification_report(Y_validation, Y_Clasificacion))

# #### **6) Ecuación del modelo de clasificación**

# #Ecuación del modelo
#     print("Intercept:", Clasificacion.intercept_)
#     print('Coeficientes: \n', Clasificacion.coef_)


# #Paciente P-842302 (1) -Tumor Maligno-
#     PacienteID1 = pd.DataFrame({'Texture': [10.38],
#                                 'Area': [1001.0],
#                                 'Smoothness': [0.11840],
#                                 'Compactness': [0.27760],
#                                 'Symmetry': [0.2419],
#                                 'FractalDimension': [0.07871]})
#     Clasificacion.predict(PacienteID1)

# #Paciente P-92751 (569) -Tumor Benigno-
#     PacienteID2 = pd.DataFrame({'Texture': [24.54],
#                                 'Area': [181.0],
#                                 'Smoothness': [0.05263],
#                                 'Compactness': [0.04362],
#                                 'Symmetry': [0.1587],
#                                 'FractalDimension': [0.05884]})
#     Clasificacion.predict(PacienteID2)

# #_________________________________________
# #_____________Practica 10__________________
# #_________________________________________

# @pagina.route('/read_csv9', methods=['POST'])
# def read_csv9():
#     from google.colab import files
#     files.upload()

#     Hipoteca = pd.read_csv('Hipoteca.csv')
#     Hipoteca

#     print(Hipoteca.groupby('comprar').size())

# #### **2) Selección de características**

#     sns.pairplot(Hipoteca, hue='comprar')
#     plt.show()

#     sns.scatterplot(x='ahorros', y ='ingresos', data=Hipoteca, hue='comprar')
#     plt.title('Gráfico de dispersión')
#     plt.xlabel('Ahorros')
#     plt.ylabel('Ingresos')
#     plt.show()

#     CorrHipoteca = Hipoteca.corr(method='pearson')
#     CorrHipoteca

#     plt.figure(figsize=(14,7))
#     MatrizInf = np.triu(Hipoteca.corr())
#     sns.heatmap(Hipoteca.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
#     plt.show()

# #Variables predictoras
#     X = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
#     pd.DataFrame(X)

# #Variable clase
#     Y = np.array(Hipoteca[['comprar']])
#     pd.DataFrame(Y)

#     plt.figure(figsize=(10, 7))
#     plt.scatter(X[:,0], X[:,5], c = Hipoteca.comprar)
#     plt.grid()
#     plt.xlabel('ingresos')
#     plt.ylabel('vivienda')
#     plt.show()

# #### **4) Aplicación del algoritmo**

#     X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
#                                                                                     test_size = 0.2,
#                                                                                     random_state = 1234,
#                                                                                     shuffle = True)

#     pd.DataFrame(X_train)

#     pd.DataFrame(Y_train)

# #Se entrena el modelo a partir de los datos de entrada
#     Clasificacion = linear_model.LogisticRegression()
#     Clasificacion.fit(X_train, Y_train)

# #Se generan las probabilidades

# #Predicciones probabilísticas
#     Probabilidad = Clasificacion.predict_proba(X_validation)
#     pd.DataFrame(Probabilidad)

# #Predicciones con clasificación final
#     Predicciones = Clasificacion.predict(X_validation)
#     pd.DataFrame(Predicciones)

# #A manera de referencia se calcula la exactitud promedio
#     Clasificacion.score(X_validation, Y_validation)

# #### **5) Validación del modelo**


# #Matriz de clasificación
#     Y_Clasificacion = Clasificacion.predict(X_validation)
#     Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),
#                                     Y_Clasificacion,
#                                     rownames=['Reales'],
#                                     colnames=['Clasificación'])
#     Matriz_Clasificacion

# #Reporte de la clasificación
#     print("Exactitud", Clasificacion.score(X_validation, Y_validation))
#     print(classification_report(Y_validation, Y_Clasificacion))

# #### **6) Ecuación del modelo de clasificación**

# #Ecuación del modelo
#     print("Intercept:", Clasificacion.intercept_)
#     print('Coeficientes: \n', Clasificacion.coef_)


# # 0=alquilar y 1=crédito
#     HipotecaID250 = pd.DataFrame({'ingresos': [6000],
#                                 'gastos_comunes': [1000],
#                                 'pago_coche': [0],
#                                 'gastos_otros': [600],
#                                 'ahorros': [50000],
#                                 'vivienda': [400000],
#                                 'estado_civil': [0],
#                                 'hijos': [2],
#                                 'trabajo': [2]
#                                 })
#     Clasificacion.predict(HipotecaID250)

# # 0=alquilar y 1=crédito
#     HipotecaID251 = pd.DataFrame({'ingresos': [6745],
#                                 'gastos_comunes': [944],
#                                 'pago_coche': [123],
#                                 'gastos_otros': [429],
#                                 'ahorros': [43240],
#                                 'vivienda': [636897],
#                                 'estado_civil': [1],
#                                 'hijos': [3],
#                                 'trabajo': [6]
#                               })
#     Clasificacion.predict(HipotecaID251)


#     """
# # CIERRE DE COMENTARIO PARA LA PRACTICA 4
