"""One liner

Description....
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

import requests
import xml.etree.ElementTree as ET
import numpy as np
from requests.auth import HTTPDigestAuth
from PIL import Image
from StringIO import StringIO
import os
import warnings
import yaml

base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "config.yml"), 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
cfg_odk = cfg['odk']


class ODKInputConnector():
    """ ODKInputConnector(user, password, formID)
        Conector para obtener imagenes del servidor de Open Data Kit
    """

    def __init__(self, user, password, formID):
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, password)
        self.formID = formID

    def get_image_ids(self):
        """ Devuelve una lista con todas las uuids de imagenes en el servidor
        """
        params = {'formId': self.formID}
        response = self.session.get(cfg_odk['URL_ENTRIES'], params=params)
        xml = ET.fromstring(response.content)
        idList = list(xml)[0]
        uuids = [child.text for child in idList]
        return uuids

    def get_image(self, uuid):
        """ Devuelve un numpy array con la imagen correspondiente a ese uuid
        """
        params = {'blobKey': cfg_odk['BLOBKEY_BASE'] % uuid}
        response = self.session.get(cfg_odk['URL_BINARYDATA'], params=params)
        try:
            image = Image.open(StringIO(response.content))
            return np.array(image)
        except IOError:
            warnings.warn("No se pudo abrir la foto con uuid: %s" % uuid)
            return None


class FolderInputConnector():
    """ FolderInputConnector(base_path)
        Conector para obtener imagenes de la carpeta ubicada en base_path
    """

    def __init__(self, base_path):
        self.base_path = base_path

    def get_image_ids(self):
        """ Devuelve una lista con todas los nombres de archivo de imagenes en
        la carpetas.
        """
        image_paths = []
        for root, dirs, files in os.walk(self.base_path):
            for file_name in filter(self._is_image_file, files):
                image_paths.append(os.path.join(root, file_name))
        return image_paths

    def get_image(self, fileid):
        """ Devuelve un numpy array con la imagen correspondiente a ese archivo
        """
        image = Image.open(fileid)
        return np.array(image)

    @staticmethod
    def _is_image_file(filename):
        return 'jpg' in filename or 'png' in filename


class FileOutputConnector():
    "TODO"
    def __init__(self, filename):
        self.filename = filename
        self.filehandle = open(filename, 'w')

    def write_output():
        pass


class FusionTablesOutputConnector():
    "TODO"
    pass
