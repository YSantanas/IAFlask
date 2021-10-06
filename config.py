
# configuraciones para el servidor y el debug

class Config:
    pass


class desarrolladorConfig(Config):
    DEBUG = True


config = {
    'desarrollo': desarrolladorConfig,
    'defecto': desarrolladorConfig
}
