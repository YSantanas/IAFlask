from app import create_app
from flask_script import Manager
from flask_cors import CORS

#importamos el diccionario de configuracion
from config import config
#no se importa las clases sino el diccionario

config_Class= config['desarrollo']

app= create_app(config_Class)
CORS(app)

if __name__== '__main__':
    manager= Manager(app)
    manager.run()

#Configuraciones para el DEBUG
