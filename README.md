# Detector Aedes

[![Coverage Status](https://coveralls.io/repos/github/datosgobar/detector-aedes/badge.svg?branch=master)](https://coveralls.io/github/datosgobar/detector-aedes?branch=master)
[![Build Status](https://travis-ci.org/datosgobar/detector-aedes.svg?branch=master)](https://travis-ci.org/datosgobar/detector-aedes)
[![PyPI](https://badge.fury.io/py/detector-aedes.svg)](http://badge.fury.io/py/detector-aedes)
[![Stories in Ready](https://badge.waffle.io/datosgobar/detector-aedes.png?label=ready&title=Ready)](https://waffle.io/datosgobar/detector-aedes)
[![Documentation Status](https://readthedocs.org/projects/detector-aedes/badge/?version=latest)](http://detector-aedes.readthedocs.io/en/latest/?badge=latest)


Algoritmos de Visión Computacional para analizar imágenes de Ovisensores.


* Licencia: MIT license
* Documentación: https://detector-aedes.readthedocs.io


## Instalación

El paquete se encuentra disponible en pip:

```bash
$ pip install detector_aedes
```

Luego buscar la carpeta de instalación y renombrar el archivo config_sample.yml a config.yml

### Instalacion de Desarrollo

Para instalar en modo desarrollo ir a la carpeta en la que fue clonado el reporsitorio:

```bash
$ cd path/to/detector-aedes
```
y luego instalar los requerimientos con:

```bash
$ pip install -r requirements.txt
$ pip install -r requirements_dev.txt
```
finalmente instalar el paquete en modo desarrollo con:
```bash
$ pip install -e .
```
Luego renombrar el archivo config_sample.yml a config.yml

```bash
$ mv config_sample.yml config.yml
```


## Uso

El siguiente es un ejemplo de uso rapido que recorre todas las imagenes de una
carpeta y graba la salida a un archivo CSV:
```Python
from detector_aedes import AedesDetector, FolderInputConnector, FileOutputConnector
ic = FolderInputConnector('/carpeta/que/contenga/imagenes')
fc = FileOutputConnector('/ruta/a/archivo/de/salida.csv')
ad = AedesDetector(input_connector=ic, output_connector=fc)
ad.process(show_results=False)
```
Para una descripcion mas detallada referirse al [Manual de Uso](docs/MANUAL.md)

## Tests

Para correr los tests hace falta instalar las dependencias de desarrollo:

```bash
$ pip install -r requirements_dev.txt
```
y luego correr los tests con:

```bash
$ nosetests
```

## Créditos

Agradecemos al Dr. Nicolas Schweigmann del Grupo de Estudio de Mosquitos de la UBA y al equipo de Epidemiología del GCBA por su ayuda en el desarrollo de este software.

## Contacto

Te invitamos a [crearnos un issue](https://github.com/datosgobar/detector-aedes/issues/new?title=Encontre%20un%20bug%20en%20detector-aedes) en caso de que encuentres algún bug o tengas feedback de alguna parte de `detector-aedes`.

Para todo lo demás, podés mandarnos tu comentario o consulta a [datos@modernizacion.gob.ar](mailto:datos@modernizacion.gob.ar).
