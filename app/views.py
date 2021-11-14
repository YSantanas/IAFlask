from io import BytesIO
from json import dumps
from flask import render_template, Blueprint, jsonify, request, make_response, Response
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
from matplotlib.figure import Figure
from apyori import apriori, dump_as_json          # Algoritmo apriori
from uuid import uuid4

from .forms import LoginForm

# practica 3
# _____________________________________
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
# _____________________________________
#practica 4 y 5
#HAY QUE CARGARLO AL PROYECTO
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# %matplotlib inline
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
#
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
#


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
        #se imprime los datos
        print(form.username.data)
        print(form.password.data)
        #se imprime en consola
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

        # if not a CSV file, return error
        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({"message": "Seleccione un archivo CSV"}), 400)

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
        flask_file = request.files['file']
        input_a = int(request.form['input_a'])
        input_b = int(request.form['input_b'])

        # if not a CSV file, return error
        if not flask_file.filename.endswith('.csv'):
            return make_response(jsonify({"message": "Seleccione un archivo CSV"}), 400)

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


#_________________________________________
#_____________Practica 4__________________
#_________________________________________
@pagina.route('/read_csv3', methods=['POST'])
def read_csv3():

    from google.colab import files
    files.upload() 

#from google.colab import drive
#drive.mount('/content/drive')

    Hipoteca = pd.read_csv("Hipoteca.csv")
    Hipoteca

    Hipoteca.info()

    print(Hipoteca.groupby('comprar').size())

#### **2) Selección de características**


    sns.pairplot(Hipoteca, hue='comprar')
    plt.show()

    sns.scatterplot(x='ahorros', y ='ingresos', data=Hipoteca, hue='comprar')
    plt.title('Gráfico de dispersión')
    plt.xlabel('Ahorros')
    plt.ylabel('Ingresos')
    plt.show()



    CorrHipoteca = Hipoteca.corr(method='pearson')
    CorrHipoteca

    print(CorrHipoteca['ingresos'].sort_values(ascending=False)[:10], '\n')   #Top 10 valores

    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrHipoteca)
    sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.show()


    MatrizHipoteca = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
    pd.DataFrame(MatrizHipoteca)
#MatrizHipoteca = Hipoteca.iloc[:, 0:9].values     #iloc para seleccionar filas y columnas según su posición

#### **3) Aplicación del algoritmo**



    from sklearn.preprocessing import StandardScaler, MinMaxScaler  
    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos

    pd.DataFrame(MEstandarizada)

#Se importan las bibliotecas de clustering jerárquico para crear el árbol

    plt.figure(figsize=(10, 7))
    plt.title("Casos de hipoteca")
    plt.xlabel('Hipoteca')
    plt.ylabel('Distancia')
    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
#plt.axhline(y=5.4, color='orange', linestyle='--')
#Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)

#Se crean las etiquetas de los elementos en los clústeres
    MJerarquico = AgglomerativeClustering(n_clusters=7, linkage='complete', affinity='euclidean')
    MJerarquico.fit_predict(MEstandarizada)
    MJerarquico.labels_

    Hipoteca = Hipoteca.drop(columns=['comprar'])
    Hipoteca['clusterH'] = MJerarquico.labels_
    Hipoteca

#Cantidad de elementos en los clusters
    Hipoteca.groupby(['clusterH'])['clusterH'].count()

    Hipoteca[Hipoteca.clusterH == 6]

    CentroidesH = Hipoteca.groupby('clusterH').mean()
    CentroidesH

    plt.figure(figsize=(10, 7))
    plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
    plt.grid()
    plt.show()  
#_________________________________________
#_____________Practica 5__________________
#_________________________________________

@pagina.route('/read_csv4', methods=['POST'])
def read_csv4():

    from google.colab import files
    files.upload() 

#from google.colab import drive
#drive.mount('/content/drive')

    Hipoteca = pd.read_csv("Hipoteca.csv")
    Hipoteca

    Hipoteca.info()

    print(Hipoteca.groupby('comprar').size())

#### **2) Selección de características**

    sns.pairplot(Hipoteca, hue='comprar')
    plt.show()

    sns.scatterplot(x='ahorros', y ='ingresos', data=Hipoteca, hue='comprar')
    plt.title('Gráfico de dispersión')
    plt.xlabel('Ahorros')
    plt.ylabel('Ingresos')
    plt.show()


    CorrHipoteca = Hipoteca.corr(method='pearson')
    CorrHipoteca

    print(CorrHipoteca['ingresos'].sort_values(ascending=False)[:10], '\n')   #Top 10 valores

    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrHipoteca)
    sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.show()


    MatrizHipoteca = np.array(Hipoteca[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
    pd.DataFrame(MatrizHipoteca) 
#MatrizHipoteca = Hipoteca.iloc[:, 0:9].values     #iloc para seleccionar filas y columnas según su posición

#### **3) Aplicación del algoritmo**



    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos

    pd.DataFrame(MEstandarizada)



#Definición de k clusters para K-means
#Se utiliza random_state para inicializar el generador interno de números aleatorios
    SSE = []
    for i in range(2, 12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)

#Se grafica SSE en función de k
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 12), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()



# !pip install kneed


    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
    kl.elbow

    plt.style.use('ggplot')
    kl.plot_knee()

#Se crean las etiquetas de los elementos en los clusters
    MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    MParticional.labels_

    Hipoteca = Hipoteca.drop(columns=['comprar'])
    Hipoteca['clusterP'] = MParticional.labels_
    Hipoteca

#Cantidad de elementos en los clusters
    Hipoteca.groupby(['clusterP'])['clusterP'].count()

    Hipoteca[Hipoteca.clusterP == 0]

"""Obtención de los centroides"""

    CentroidesP = Hipoteca.groupby('clusterP').mean()
    CentroidesP



# Gráfica de los elementos y los centros de los clusters

    plt.rcParams['figure.figsize'] = (10, 7)
    plt.style.use('ggplot')
    colores=['red', 'blue', 'green', 'yellow']
    asignar=[]
    for row in MParticional.labels_:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(MEstandarizada[:, 0], 
            MEstandarizada[:, 1], 
            MEstandarizada[:, 2], marker='o', c=asignar, s=60)
    ax.scatter(MParticional.cluster_centers_[:, 0], 
            MParticional.cluster_centers_[:, 1], 
            MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
    plt.show()
#_________________________________________
#_____________Practica 6__________________
#_________________________________________

@pagina.route('/read_csv5', methods=['POST'])
def read_csv5():
#_________________________________________
#_____________Practica 7__________________
#_________________________________________


@pagina.route('/read_csv6', methods=['POST'])
def read_csv6():

#_________________________________________
#_____________Practica 8__________________
#_________________________________________

@pagina.route('/read_csv7', methods=['POST'])
def read_csv7():

#_________________________________________
#_____________Practica 9__________________
#_________________________________________

@pagina.route('/read_csv8', methods=['POST'])
def read_csv8():

#_________________________________________
#_____________Practica 10__________________
#_________________________________________

@pagina.route('/read_csv9', methods=['POST'])
def read_csv9():