from app import create_app
from flask_script import Manager

#importamos el diccionario de configuracion
from config import config
#no se importa las clases sino el diccionario

config_Class= config['desarrollo']

app= create_app(config_Class)

if __name__== '__main__':
    manager= Manager(app)
    manager.run()

#Configuraciones para el DEBUG
