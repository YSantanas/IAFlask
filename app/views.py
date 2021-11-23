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

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.tree import export_graphviz


from sklearn.tree import plot_tree
import graphviz as g
from graphviz import Source

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


@pagina.route('/algoritmoApriori', methods=['POST'])
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


@pagina.route('/medidasDistancias', methods=['POST'])
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
@pagina.route('/clusterJerarquico', methods=['POST'])
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

# _________________________________________


# ___________________________GRAFICA DE CLUSTER________________________________

        MatrizHipoteca = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche',
                        'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
#     pd.DataFrame(MatrizHipoteca)
# # MatrizHipoteca = Hipoteca.iloc[:, 0:9].values     #iloc para seleccionar filas y columnas según su posición

# # **3) Aplicación del algoritmo**

#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     # Se instancia el objeto StandardScaler o MinMaxScaler
        estandarizar = StandardScaler()
#     # Se calculan la media y desviación y se escalan los datos
        MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)

#     pd.DataFrame(MEstandarizada)

# # Se importan las bibliotecas de clustering jerárquico para crear el árbol

        plt.figure(figsize=(10, 7))
        plt.title("Casos de hipoteca")
        plt.xlabel('Hipoteca')
        plt.ylabel('Distancia')
        Arbol = shc.dendrogram(shc.linkage(
                MEstandarizada, method='complete', metric='euclidean'))
# #plt.axhline(y=5.4, color='orange', linestyle='--')
# # Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)
        buffer= BytesIO()
        
        
        nombre_temporal_uuid001 = str(uuid4().hex)
        nombre_temporal_png001 = 'app/static/img/' + nombre_temporal_uuid001 + '.png'

        # Replace "app" to ""
        new_name001 = nombre_temporal_png001.replace('app/', '')
        

        plt.savefig(
            fname=nombre_temporal_png001,
            format='png',
        )
        buffer.close()
        
# __________________________________________
# ___________________________GRAFICA DE DISPERSION 1________________________________

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
        return jsonify({'status': 'success', 'data': json_data_1, 'image': new_name,'image2': new_name001})

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

@pagina.route('/clusterParticional', methods=['POST'])
def read_csv4():

    if request.method == 'POST':

        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        Hipoteca = pd.read_csv(flask_file)

        print(Hipoteca)

        print(Hipoteca.groupby('comprar').size())

        data_table_3 = pd.DataFrame(Hipoteca)
        json_data3 = data_table_3.head(10).to_json(orient='records')

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


@pagina.route('/regresionLineal', methods=['POST'])
def read_csv5():

    if request.method == 'POST':

        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        RGeofisicos = pd.read_csv(flask_file)
        print(RGeofisicos)

        data_table_4 = pd.DataFrame(RGeofisicos)
        json_data4 = data_table_4.head(10).to_json(orient='records')

# ___________________________GRAFICA1________________________________

        # Se genera un gráfico
        fig = Figure()
        fig.set_size_inches(20, 5)
        ax = fig.subplots()
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC2'],
                color='purple', marker='o', label='RC2')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC2'],
                color='purple', marker='o', label='RC2')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC3'],
                color='blue', marker='o', label='RC3')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC4'],
                color='yellow', marker='o', label='RC4')
        ax.set_xlabel('Profundidad')
        ax.set_ylabel('Porcentaje')
        ax.set_title('Registros_geofsicos convencionales')
        ax.grid(True)
        ax.legend()

        nombre_temporal_uuid = str(uuid4().hex)
        nombre_temporal_png = 'app/static/img/' + nombre_temporal_uuid + '.png'

        # Replace "app" to ""
        new_name2 = nombre_temporal_png.replace('app/', '')

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png,
            format='png',
        )


# #### **3) Aplicación del algoritmo**


# #Se seleccionan las variables predictoras (X) y la variable a pronosticar (Y)

        X_train = np.array(RGeofisicos[['Profundidad', 'RC1', 'RC2', 'RC3']])
        Xpronostic = pd.DataFrame(X_train)

        Y_train = np.array(RGeofisicos[['RC4']])
        ypronostic = pd.DataFrame(Y_train)

# #Se entrena el modelo a través de una Regresión Lineal Múltiple

        RLMultiple = linear_model.LinearRegression()
        RLMultiple.fit(X_train, Y_train)  # Se entrena el modelo


# #Se genera el pronóstico
        Y_pronostico = RLMultiple.predict(X_train)
        Ypronostic = pd.DataFrame(Y_pronostico)
        RGeofisicos1 = RGeofisicos

        RGeofisicos1['Pronostico'] = Y_pronostico

# AQUI TABLA DONDE SE ANEXA LA COLUMNA PRONOSTICO

        data_table_5 = pd.DataFrame(RGeofisicos1)
        json_data5 = data_table_5.head(10).to_json(orient='records')


# #### **4) Obtención de los coeficientes, intercepto, error y Score**


#     print('Coeficientes: \n', RLMultiple.coef_)
#     print('Intercepto: \n', RLMultiple.intercept_)
#     print("Residuo: %.4f" % max_error(Y_train, Y_pronostico))
#     print("MSE: %.4f" % mean_squared_error(Y_train, Y_pronostico))
#     print("RMSE: %.4f" % mean_squared_error(Y_train, Y_pronostico, squared=False))  #True devuelve MSE, False devuelve RMSE
#     print('Score (Bondad de ajuste): %.4f' % r2_score(Y_train, Y_pronostico))


# #### **6) Proyección de los valores reales y pronosticados**

# ___________________________GRAFICA2________________________________

        # Se genera un gráfico
        fig = Figure()
        fig.set_size_inches(20, 5)
        ax = fig.subplots()
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC1'],
                color='green', marker='o', label='RC1')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC2'],
                color='purple', marker='o', label='RC2')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC3'],
                color='blue', marker='o', label='RC3')
        ax.plot(RGeofisicos['Profundidad'], RGeofisicos['RC4'],
                color='yellow', marker='o', label='RC4')
        ax.set_xlabel('Profundidad')
        ax.set_ylabel('Porcentaje')
        ax.set_title('Registros_geofsicos convencionales')
        ax.grid(True)
        ax.legend()

        nombre_temporal_uuid2 = str(uuid4().hex)
        nombre_temporal_png2 = 'app/static/img/' + nombre_temporal_uuid2 + '.png'

        # Replace "app" to ""
        new_name3 = nombre_temporal_png2.replace('app/', '')

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png2,
            format='png',
        )


# ___________________________GRAFICA3________________________________

        # Se genera un gráfico
        fig = Figure()
        fig.set_size_inches(20, 5)
        ax = fig.subplots()
        ax.plot(RGeofisicos['Profundidad'], Y_pronostico,
                color='red', marker='o', label='Pronóstico')
        ax.set_xlabel('Profundidad')
        ax.set_ylabel('Porcentaje')
        ax.set_title('Registros_geofsicos convencionales')
        ax.grid(True)
        ax.legend()

        nombre_temporal_uuid3 = str(uuid4().hex)
        nombre_temporal_png3 = 'app/static/img/' + nombre_temporal_uuid3 + '.png'

        # Replace "app" to ""
        new_name4 = nombre_temporal_png3.replace('app/', '')

        # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png3,
            format='png',
        )


# #### **7) Nuevos pronósticos**

#     ROS = pd.DataFrame({'Profundidad': [5680.5], 'RC1': [0.45], 'RC2': [0.64], 'RC3': [0.5]})
#     RLMultiple.predict(ROS)

        # Retorna imagen en base64
        return jsonify({'status': 'success', 'data': json_data4, 'data2': json_data5, 'graph': new_name2, 'graph2': new_name3, 'graph3': new_name4})

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})


# #_________________________________________
# #_____________Practica 11__________________
# #_________________________________________


@pagina.route('/pronosticoArbol', methods=['POST'])
def read_csv6():
    if request.method == 'POST':

        flask_file = request.files['file']

        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({'message': 'Seleccione un archivo CSV'}), 400)

        BCancer = pd.read_csv(flask_file)

        data_table_11 = pd.DataFrame(BCancer)
        json_data11 = data_table_11.head(10).to_json(orient='records')


# ___________________________GRAFICA1________________________________


# **2) Gráfica del área del tumor por paciente**

        # Se genera un gráfico
        fig = Figure()
        fig.set_size_inches(20, 5)
        ax = fig.subplots()
        ax.plot(BCancer['IDNumber'], BCancer['Area'],
                color='green', marker='o', label='Area')
        ax.set_xlabel('Paciente')
        ax.set_ylabel('Tamaño del tumor')
        ax.set_title('Pacientes con tumores cancerígenos')
        ax.grid(True)
        ax.legend()

        nombre_temporal_uuid11 = str(uuid4().hex)
        nombre_temporal_png11 = 'app/static/img/' + nombre_temporal_uuid11 + '.png'

        # Replace "app" to ""
        new_name11 = nombre_temporal_png11.replace('app/', '')

     # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png11,
            format='png',
        )


# #### **3) Selección de características** MAPA DE CALOR

# plt.figure(figsize=(14,7))
# MatrizInf = np.triu(BCancer.corr())
# sns.heatmap(BCancer.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
# plt.show()


# #### **4) Aplicación del algoritmo**


#Se seleccionan las variables predictoras (X) y la variable a pronosticar (Y)

        X = np.array(BCancer[['Texture',
                              'Perimeter',
                              'Smoothness',
                              'Compactness',
                              'Symmetry',
                              'FractalDimension']])
        tabla2= pd.DataFrame(X)
        
        json_data12 = tabla2.head(10).to_json(orient='records')
        
# #X = np.array(BCancer[['Radius', 'Texture', 'Perimeter', 'Smoothness', 'Compactness',	'Concavity', 'ConcavePoints', 'Symmetry',	'FractalDimension']
# #pd.DataFrame(X)

        Y = np.array(BCancer[['Area']])


 #Se hace la división de los datos

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                             test_size = 0.2,
                                                                              random_state = 1234,
                                                                              shuffle = True)


#DATOS QUE SE PIDEN EN INPUT  -------------->>>>>
# pd.DataFrame(X_train)
# #pd.DataFrame(X_test)

# pd.DataFrame(Y_train)
# #pd.DataFrame(Y_test)

# #Se entrena el modelo a través de un Árbol de Decisión (Regresión)

        PronosticoAD = DecisionTreeRegressor()
        PronosticoAD.fit(X_train, Y_train)

# #PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2)
# #PronosticoAD.fit(X_train, Y_train)


 #Se genera el pronóstico
        Y_Pronostico = PronosticoAD.predict(X_test)
# pd.DataFrame(Y_Pronostico)

        Valores = pd.DataFrame(Y_test, Y_Pronostico)
# Valores


# ___________________________GRAFICA2________________________________


# **2) Gráfica del área del tumor por paciente**

        # Se genera un gráfico
        fig = Figure()
        fig.set_size_inches(20, 5)
        ax = fig.subplots()
        ax.plot(Y_test, color='green', marker='o', label='Y_test')
        ax.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
        ax.set_xlabel('Paciente')
        ax.set_ylabel('Tamaño del tumor')
        ax.set_title('Pacientes con tumores cancerígenos')
        ax.grid(True)
        ax.legend()

        nombre_temporal_uuid12 = str(uuid4().hex)
        nombre_temporal_png12 = 'app/static/img/' + nombre_temporal_uuid12 + '.png'

        # Replace "app" to ""
        new_name12 = nombre_temporal_png12.replace('app/', '')

     # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png12,
            format='png',
        )


        r2_score(Y_test, Y_Pronostico)

# #### **5) Obtención de los parámetros del modelo**

# print('Criterio: \n', PronosticoAD.criterion)
# print('Importancia variables: \n', PronosticoAD.feature_importances_)
# print("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
# print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
# print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
# print('Score: %.4f' % r2_score(Y_test, Y_Pronostico))

        Importancia = pd.DataFrame({'Variable': list(BCancer[['Texture', 'Perimeter', 'Smoothness',
                                                             'Compactness', 'Symmetry', 'FractalDimension']]),
                                                             'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
# Importancia
        tabla3= pd.DataFrame(Importancia)
        
        json_data13 = tabla3.head(10).to_json(orient='records')
# #### **6) Conformación del modelo de pronóstico**


# #!pip install graphviz

# #import graphviz
# #

 # Se crea un objeto para visualizar el árbol
  # Se incluyen los nombres de las variables para imprimirlos en el árbol
        Elementos = export_graphviz(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness',
                                                                   'Compactness', 'Symmetry', 'FractalDimension'])
        Arbol = g.Source(Elementos,filename="test.gv", format="png")
        
        
        
        
        nombre_temporal_uuid14 = str(uuid4().hex)
        nombre_temporal_png14 = 'app/static/img/' + nombre_temporal_uuid14 + '.png'

        # Replace "app" to ""
        new_name14 = nombre_temporal_png14.replace('app/', '')

     # Se guarda el gráfico en un archivo PNG
        fig.savefig(
            fname=nombre_temporal_png14,
            format='png',
        )
        
# Arbol

# 
# plt.figure(figsize=(16,16))
# plot_tree(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness',
#                                          'Compactness', 'Symmetry', 'FractalDimension'])
# plt.show()

# from sklearn.tree import export_text
# Reporte = export_text(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness',
#                                                      'Compactness', 'Symmetry', 'FractalDimension'])
# print(Reporte)


# #### **7) Nuevos pronósticos**


# AreaTumorID1 = pd.DataFrame({'Texture': [10.38],
#                              'Perimeter': [122.8],
#                              'Smoothness': [0.11840],
#                              'Compactness': [0.27760],
#                              'Symmetry': [0.2419],
#                              'FractalDimension': [0.07871]})
# PronosticoAD.predict(AreaTumorID1)

        # Retorna imagen en base64
        return jsonify({'status': 'success', 'data': json_data11, 'data2': json_data12,'data3': json_data13,'graph':new_name11,'graph2':new_name12})

    return jsonify({'status': 'error', 'message': 'Error al leer el archivo'})
# #_________________________________________
# #_____________Practica 12__________________
# #_________________________________________

# @pagina.route('/clasificacionArbol', methods=['POST'])
# def read_csv7():

# from google.colab import files
# files.upload()

# BCancer = pd.read_csv('WDBCOriginal.csv')
# BCancer

# print(BCancer.groupby('Diagnosis').size())

# #### **2) Selección de características**


# plt.figure(figsize=(14,7))
# MatrizInf = np.triu(BCancer.corr())
# sns.heatmap(BCancer.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
# plt.show()


# #### **3) Definición de variables predictoras y variable clase**


# BCancer = BCancer.replace({'M': 'Malignant', 'B': 'Benign'})
# BCancer

# print(BCancer.groupby('Diagnosis').size())

# #Variables predictoras
# X = np.array(BCancer[['Texture',
#                       'Area',
#                       'Smoothness',
#                       'Compactness',
#                       'Symmetry',
#                       'FractalDimension']])
# pd.DataFrame(X)

# #X = np.array(BCancer[['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',	'Concavity', 'ConcavePoints', 'Symmetry',	'FractalDimension']])
# #pd.DataFrame(X)

# #Variable clase
# Y = np.array(BCancer[['Diagnosis']])
# pd.DataFrame(Y)

# #### **4) División de datos y aplicación del algoritmo**


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn import model_selection

# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
#                                                                                 test_size = 0.2,
#                                                                                 random_state = 0,
#                                                                                 shuffle = True)

# pd.DataFrame(X_train)

# pd.DataFrame(Y_train)

# #Se entrena el modelo a partir de los datos de entrada
# ClasificacionAD = DecisionTreeClassifier()
# ClasificacionAD.fit(X_train, Y_train)

# #ClasificacionAD = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2)
# #ClasificacionAD.fit(X_train, Y_train)


# #Se etiquetan las clasificaciones
# Y_Clasificacion = ClasificacionAD.predict(X_validation)
# pd.DataFrame(Y_Clasificacion)

# Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
# Valores

# #Se calcula la exactitud promedio de la validación
# ClasificacionAD.score(X_validation, Y_validation)

# #### **5) Validación del modelo**


# #Matriz de clasificación
# Y_Clasificacion = ClasificacionAD.predict(X_validation)
# Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),
#                                    Y_Clasificacion,
#                                    rownames=['Real'],
#                                    colnames=['Clasificación'])
# Matriz_Clasificacion

# #Reporte de la clasificación
# print('Criterio: \n', ClasificacionAD.criterion)
# print('Importancia variables: \n', ClasificacionAD.feature_importances_)
# print("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
# print(classification_report(Y_validation, Y_Clasificacion))

# Importancia = pd.DataFrame({'Variable': list(BCancer[['Texture', 'Area', 'Smoothness',
#                                                      'Compactness', 'Symmetry', 'FractalDimension']]),
#                             'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
# Importancia

# #### **6) Eficiencia y conformación del modelo de clasificación**


# #!pip install graphviz
# import graphviz
# from sklearn.tree import export_graphviz

# # Se crea un objeto para visualizar el árbol
# Elementos = export_graphviz(ClasificacionAD,
#                             feature_names = ['Texture', 'Area', 'Smoothness',
#                                              'Compactness', 'Symmetry', 'FractalDimension'],
#                             class_names = Y_Clasificacion)
# Arbol = graphviz.Source(Elementos)
# Arbol

# from sklearn.tree import plot_tree
# plt.figure(figsize=(16,16))
# plot_tree(ClasificacionAD,
#           feature_names = ['Texture', 'Area', 'Smoothness',
#                            'Compactness', 'Symmetry', 'FractalDimension'],
#           class_names = Y_Clasificacion)
# plt.show()

# from sklearn.tree import export_text
# Reporte = export_text(ClasificacionAD,
#                       feature_names = ['Texture', 'Area', 'Smoothness',
#                                        'Compactness', 'Symmetry', 'FractalDimension'])
# print(Reporte)

# #### **7) Nuevas clasificaciones**

# #Paciente P-842302 (1) -Tumor Maligno-
# PacienteID1 = pd.DataFrame({'Texture': [10.38],
#                             'Area': [1001.0],
#                             'Smoothness': [0.11840],
#                             'Compactness': [0.27760],
#                             'Symmetry': [0.2419],
#                             'FractalDimension': [0.07871]})
# ClasificacionAD.predict(PacienteID1)

# #Paciente P-92751 (569) -Tumor Benigno-
# PacienteID2 = pd.DataFrame({'Texture': [24.54],
#                             'Area': [181.0],
#                             'Smoothness': [0.05263],
#                             'Compactness': [0.04362],
#                             'Symmetry': [0.1587],
#                             'FractalDimension': [0.05884]})
# ClasificacionAD.predict(PacienteID2)
