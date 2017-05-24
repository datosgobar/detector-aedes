#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""One liner

Description....
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement


import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import pickle
import yaml

# Para el procesamiento de imagenes
from skimage import io
from skimage import transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import regionprops
from skimage import feature
from skimage.transform import resize, rotate
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage.future import graph
from skimage import morphology
from skimage import measure
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "config.yml"), 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
cfg_images = cfg['images']


class AedesDetector():
    """Detector de Huevos en Ovisensores.

    `AedesDetector(input_connector, output_connector)`
    Algoritmos de vision computacional para reconocer la presencia de
    huevos en fotos de ovisensores.

     Args:
            input_connector (:obj:): Conector a la fuente de imagenes a
                analizar.
            output_connector (:obj:): Conector a la salida de los resulados del
                análisis.
    """

    def __init__(self, input_connector, output_connector):
        self.input_connector = input_connector
        self.output_connector = output_connector
        self.stick_analizer = StickAnalizer()
        self.egg_finder = EggFinder()

    def process(self, image_id=None, show_results=False, start_from=0):
        """Función principal para procesar las imágenes disponibles.

        `process(self, image_id=None, show_results=False, start_from=0)`


        Args:
            image_id (str, optional): Este parametro debe corresponder al ID de
                una imagen. Si está definido se procesa solo esa imagen.
            show_results (bool, optional): Mostrar un gráfico del procesamiento
                para cada imagen a medida que se procesan.
            start_from (int, optional): Comienza a analizar a partir de la
                imagen `start_from`-ésima imagen.
        """
        if image_id:
            self._process_one_image(image_id)
            if show_results:
                self.plot_classification()
        else:
            image_ids = self.input_connector.get_image_ids()
            for image_id in image_ids[start_from:]:
                self._process_one_image(image_id)
                if show_results:
                    self.plot_classification()

    def _process_one_image(self, image_id):
        """ Función interna para procesar una sola imagen.
        _process_one_image(self, image_id)
        Args:
            image_id (str): Debe corresponder a un ID de imagen en el
            `input_connector`

        """
        image = self.input_connector.get_image(image_id)
        if image is None:
            warnings.warn("La imagen {:s} no pudo ser cargada"
                          .format(image_id))
        self.stick_analizer.set_current_image(image)
        self.clipped_image = self.stick_analizer.get_clipped_image()
        self.egg_finder.find_in(self.clipped_image)
        self.egg_finder.classify()

    def test_stick_algorithm(self):
        # Viejo! Necesita ser refactoreado
        for root, dirs, files in os.walk(self.im_path):
            for file_name in filter(self._is_image_file, files):
                image = os.path.join(root, file_name)
                self._load_image(image)
                self._find_stick()
                if not hasattr(self, 'fig'):
                    self.fig = plt.figure()
                    self.fig.canvas.mpl_connect('key_press_event',
                                                self._handle_fig_event)
                    self.fig.canvas.mpl_connect('button_press_event',
                                                self._handle_fig_event)
                self.next_figure = False
                self.fig.clf()
                ax = self.fig.add_subplot(121)
                ax.imshow(self.curr_im)
                self.fig.canvas.draw()
                while not self.next_figure:
                    self.fig.waitforbuttonpress()

    def train_model(self, trainsamples=None, show_mask=False):
        self.egg_finder.start_trainning()
        if isinstance(trainsamples, list):
            samples = trainsamples
        else:
            im_ids = self.input_connector.get_image_ids()
            if len(im_ids) < trainsamples:
                raise ValueError('No hay suficientes imagenes para entrenar.\
                                 (Nfiles=%i)' % len(im_ids))
            samples = np.random.choice(im_ids, trainsamples,
                                       replace=False)  # Imagenes al azar
        for im_id in samples:
            image = self.input_connector.get_image(im_id)
            self.stick_analizer.set_current_image(image)
            self.clipped_image = self.stick_analizer.get_clipped_image()
            self.egg_finder.find_in(self.clipped_image)
            self.plot_selector(show_mask=show_mask)
        self.egg_finder.save_model()

    def _handle_fig_event(self, event):
        "Función interna para manejar eventos de ventanas en modo interactivo."
        if event.key != 'n':
            self.next_figure = False
        else:
            self.next_figure = True

    def _onpick(self, event):
        "Función interna para manejar eventos de ventanas en modo interactivo."
        event.artist._edgecolors[event.ind, :] = (0, 0, 1, 1)
        self.temp_targets[event.ind] = 1

    def plot_classification(self):
        "Graficar la clasificacion par una dada imágen"
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('key_press_event',
                                        self._handle_fig_event)
            self.fig.canvas.mpl_connect('button_press_event',
                                        self._handle_fig_event)
        self.next_figure = False
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.imshow(self.clipped_image)
        ax.imshow(self.egg_finder.mask, alpha=0.5)
        ax.scatter(self.egg_finder.measures[self.egg_finder.classes == 0, -2],
                   self.egg_finder.measures[self.egg_finder.classes == 0, -1],
                   facecolors='none', s=80, alpha=0.5,
                   color='g')
        ax.scatter(self.egg_finder.measures[self.egg_finder.classes == 1, -2],
                   self.egg_finder.measures[self.egg_finder.classes == 1, -1],
                   facecolors='none', s=80, alpha=0.5,
                   color='r')
        self.fig.canvas.draw()
        while not self.next_figure:
            self.fig.waitforbuttonpress()

    def plot_selector(self, show_mask=False):
        """Graficar los puntos candidatos y permitir al usuario elegir los
           positivos.

           USO:
                - Hacer click sobre todos los candidatos considerados positivos
                (van a cambiar de color).

                - Apretar N para pasar a la imagen siguiente
        """
        self.temp_targets = np.zeros(len(self.egg_finder.measures))
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('key_press_event',
                                        self._handle_fig_event)
            self.fig.canvas.mpl_connect('button_press_event',
                                        self._handle_fig_event)
            self.fig.canvas.mpl_connect('pick_event', self._onpick)
            self.fig.suptitle(('Haga click en los objetivos. Apriete N para',
                               'pasar a la siguiente imagen.'))
        self.next_figure = False
        self.good_points = []
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.imshow(self.clipped_image)
        if show_mask:
            ax.imshow(self.egg_finder.mask, alpha=0.5)
        colors = np.array(['r', 'y'])
        ax.scatter(self.egg_finder.measures[:, -2],
                   self.egg_finder.measures[:, -1],
                   facecolors='none', s=80, alpha=0.5,
                   color=colors[self.egg_finder.type],
                   picker=5
                   )
        self.fig.canvas.draw()
        while not self.next_figure:
            self.fig.waitforbuttonpress()
            self.fig.canvas.draw()
        self.egg_finder.push_user_input(self.egg_finder.measures,
                                        self.temp_targets)


class StickAnalizer():

    def __init__(self, lowres_width=100, target_size=0.5):
        self.lowres_width = lowres_width
        self.target_size = target_size
        self._load_patches()

    def set_current_image(self, image):
        orig_size = image.shape[0:2]
        self.scale = (orig_size[1] / self.lowres_width)
        new_size = np.array(orig_size) / self.scale
        image_lowres = transform.resize(image, new_size)
        self.curr_im = image
        self.curr_im_lowres = image_lowres
        self.curr_im_lowres_g = rgb2gray(image_lowres)

    def get_clipped_image(self):
        self._find_stick()
        return self.curr_im

    def _find_stick(self):

        mask1, disp1, angle1 = self.polish_mask(
            self.binary_by_colors(self.curr_im_lowres,
                                  self.target_colors,
                                  thresh=0.2))
        mask2, disp2, angle2 = self.polish_mask(
            self.binary_by_edges(self.curr_im_lowres_g))
        mask3, disp3, angle3 = self.polish_mask(
            self.binary_by_thresholding(self.curr_im_lowres_g))

        # Eligo el metodo que da menor disparidad
        disparities = [disp1, disp2, disp3]
        masks = [mask1, mask2, mask3]
        angles = [angle1, angle2, angle3]
        idx_min_disp = np.argmin(disparities)
        binary_image = masks[idx_min_disp]
        orientation = angles[idx_min_disp]

        # Roto la imagen y la mascara
        binary_image = resize(binary_image, self.curr_im.shape[:2])
        binary_image = rotate(binary_image,
                              -orientation * 180 / np.pi, resize=True) > 0
        rotated_curr_im = rotate(self.curr_im,
                                 -orientation * 180 / np.pi, resize=True)
        rotated_curr_im[~binary_image] = np.tile([1, 1, 1],
                                                 (np.sum(~binary_image), 1))
        props = regionprops(binary_image.astype(int))[0]
        roi = np.array(props.bbox)
        minrow, mincol, maxrow, maxcol = roi
        self.curr_im = rotated_curr_im[minrow:maxrow, mincol:maxcol]

    def _load_patches(self):
        """Crear medias para extraccion por color a partir de recortes del
        bajalenguas"""
        self.target_colors = []
        patch_dir = cfg_images['patch_dir']
        patches = os.listdir(patch_dir)
        for patch in patches:
            impatch = io.imread(os.path.join(patch_dir, patch))
            impatch_hsv = rgb2hsv(impatch)
            segments_slic = segmentation.slic(impatch, n_segments=2,
                                              compactness=10, sigma=1)
            for lab in np.unique(segments_slic):
                mediana = np.median(impatch_hsv[segments_slic == lab], axis=0)
                self.target_colors.append(mediana)

    @staticmethod
    def binary_by_edges(img_g):
        "Segmentacion por bordes"
        cedges = feature.canny(img_g, sigma=2, high_threshold=0.9,
                               low_threshold=0.2, use_quantiles=True)
        return cedges

    @staticmethod
    def binary_by_thresholding(img_g):
        "Segmentacion por umbral de intensidad"
        th = threshold_otsu(img_g)
        binary_mask = img_g < th
        return binary_mask

    @staticmethod
    def binary_by_colors(img, target_colors, thresh=0.1):
        "Segmentacion por color"
        segments_slic = segmentation.slic(img, n_segments=300,
                                          compactness=10, sigma=1)
        g = graph.rag_mean_color(img, segments_slic)
        graphcut = graph.cut_threshold(segments_slic, g, 0.1)
        g = graph.rag_mean_color(img, graphcut)
        good_nodes = []
        for nid in g.nodes():
            color = g.node[nid]['mean color']
            color = rgb2hsv(color[None, None, :])
            minimo = np.min(pairwise_distances(color[0, :, :], target_colors))
            if minimo < thresh:
                good_nodes.append(nid)
        binary_mask = np.zeros(graphcut.shape, dtype=bool)
        for gn in good_nodes:
            binary_mask = binary_mask + (graphcut == gn)
        return binary_mask

    def polish_mask(self, binary_mask):
        "Elije la mejor region y completa huecos"
        filled = morphology.convex_hull_object(binary_mask)
        labels = measure.label(filled)
        rprops = measure.regionprops(labels)
        if len(rprops) == 0:
            return binary_mask, np.inf, -1
        disparity = self.calculate_disparity(rprops,
                                             np.prod(binary_mask.shape))
        I = np.argmin(disparity)
        polished_mask = (labels == (I + 1))
        polished_mask = morphology.convex_hull_image(polished_mask)
        return polished_mask, disparity[I], rprops[I].orientation

    @staticmethod
    def _disparity_fun(geometric_props):
        """Calcula la disparidad entre las propiedades de una region y
        las propiedades objetivo elegidas en la configuracion"""
        targets = np.array(cfg['stick']['geometry'])
        weights = np.array(cfg['stick']['geometry-weights'])
        return np.sum((weights * np.log(geometric_props / targets))**2)

    def calculate_disparity(self, rprops, imarea, method='props'):
        """Calcula la disparidad entre todas las regiones candidatas y
        las propiedades objetivo elegidas en la configuracion

        Args:
            - rprops (list): Lista de propiedades de las regiones candidatas
            - imarea (int): Area de la imagen
            - method (str, optional): Metodo a utilizar para comparar las
                regiones con el objetivo. (Puede ser `props` o `hu`)
                    - `props`: usa tres propiedades geometricas.
                    - `hu`: usa los hu-moments
        """
        if method == 'props':
            descriptors = []
            for r in rprops:
                try:
                    descriptors.append(np.array(
                        (r.major_axis_length / r.minor_axis_length,
                         r.area / (r.major_axis_length * r.minor_axis_length),
                         float(r.area) / imarea
                         ), dtype=float))
                except ZeroDivisionError:
                    descriptors.append(np.array((999, 999, 999), dtype=float))
            disparity = map(self._disparity_fun, descriptors)
        elif method == 'hu':
            MAX_MOMENT = cfg['stick']['max-hu']
            TARGET_HU = np.array(cfg['stick']['hu-moments'])
            disparity = []
            for r in rprops:
                disparity.append(np.sum((np.log(r.moments_hu[:MAX_MOMENT]) -
                                         np.log(TARGET_HU[:MAX_MOMENT]))**2))
        else:
            raise ValueError('No existe el metodo ' + str(method))
        return disparity


class EggFinder():

    if cfg['model']['default_dir'] != 'None':
        base_path = cfg['model']['default_dir']
    else:
        base_path = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = os.path.join(base_path, 'models', 'aedes-model.pkl')
    DATA_PATH = os.path.join(base_path, 'models', 'aedes-data-model.pkl')
    STANDARD_MAJOR_AXIS = cfg['eggs']['geometry']['major_axis']
    STANDARD_MINOR_AXIS = cfg['eggs']['geometry']['minor_axis']
    STANDARD_AREA = STANDARD_MAJOR_AXIS * STANDARD_MINOR_AXIS
    TOL = cfg['eggs']['tolerance']  # Tolerancia para variaciones de tamanio
    TOL = 1 + TOL / 100  # Convertir de porcentaje a fraccion

    def __init__(self):
        self._load_model()

    def find_in(self, image, method='threshold'):
        curr_im_g = rgb2gray(image)
        # Segmento por color

        if method == 'threshold':
            threshold = 0.2
            mask = (curr_im_g < threshold)
            if np.mean(mask) > 0.07:
                threshold = np.percentile(curr_im_g, 5)
                mask = (curr_im_g < threshold)
            labels = measure.label(mask)
            self.mask = mask
        elif method == 'quickshift':
            # Los parametros deben ser adaptados a la escala de la imagen
            labels = segmentation.quickshift(image,
                                             kernel_size=3,
                                             max_dist=6,
                                             ratio=0.5)
        # Calculo propiedades de las regiones segmentadas
        regions = regionprops(labels, intensity_image=curr_im_g)
        if len(regions) == 0:
            return
        py, px = np.array([x.centroid for x in regions]).T
        alto_im = curr_im_g.shape[0]
        areas = np.array([x.area for x in regions])
        areas = areas.astype(float) / alto_im**2
        perimetros = np.array([x.perimeter for x in regions])
        perimetros = perimetros.astype(float) / alto_im
        major_axis = np.array([x.major_axis_length for x in regions])
        major_axis = major_axis.astype(float) / alto_im
        minor_axis = np.array([x.minor_axis_length for x in regions])
        minor_axis = minor_axis.astype(float) / alto_im
        convex_areas = np.array([x.convex_area for x in regions])
        convex_areas = convex_areas.astype(float) / alto_im**2
        diff_areas = convex_areas - areas
        intensity = np.array([x.mean_intensity for x in regions])
        labels = np.arange(len(regions)) + 1

        gi = self._filter_candidates(minor_axis, major_axis, areas)
        gi = np.arange(len(minor_axis))
        self.measures = np.vstack((areas[gi], perimetros[gi], major_axis[gi],
                                   minor_axis[gi], diff_areas[gi],
                                   intensity[gi], px[gi], py[gi])).T
        self.classify()

    def _filter_candidates(self, minor_axis, major_axis, areas):
        """Filtra a los posibles candidatos en base a requisitos mínimos
        o máximos de tamaño"""
        singles = ((minor_axis > self.STANDARD_MINOR_AXIS / self.TOL) &
                   (minor_axis < self.STANDARD_MINOR_AXIS * self.TOL) &
                   (major_axis > self.STANDARD_MAJOR_AXIS / self.TOL) &
                   (major_axis < self.STANDARD_MAJOR_AXIS * self.TOL) &
                   (areas > self.STANDARD_AREA / self.TOL) &
                   (areas < self.STANDARD_AREA * self.TOL)
                   )
        # Esto es por si hay 2 o 3 huevos pegados
        multiples = (((areas > self.STANDARD_AREA * (2 - 1. / self.TOL)) &
                     (areas < self.STANDARD_AREA * (2 + self.TOL))) |
                     ((areas > self.STANDARD_AREA * (3 - 1. / self.TOL)) &
                     (areas < self.STANDARD_AREA * (3 + self.TOL)))
                     )
        good_indices = singles | multiples
        print("Total singles: %i" % np.sum(singles))
        print("Total multiples: %i" % np.sum(multiples))
        self.type = singles + multiples * 2
        self.type = self.type[good_indices] - 1
        return good_indices

    def classify(self):
        try:
            self.classes = self.model.predict(self.measures)
        except AttributeError:
            self.classes = None

    def start_trainning(self):
        self.all_targets = np.array([])
        self.all_measures = np.zeros((0, 8))

    def push_user_input(self, measures, targets):
        self.all_targets = np.r_[self.all_targets, targets]
        self.all_measures = np.vstack([self.all_measures, measures])

    def _load_model(self):
        "Carga el modelo de clasificador de de arbol (DecisionTree)"
        if os.path.isfile(self.MODEL_PATH):
            model_file = open(self.MODEL_PATH, 'r')
            self.model = pickle.load(model_file)
            model_file.close()
        else:
            print("No Model")

    def _load_data(self):
        "Carga datos guardados de entrenamiento"
        if os.path.isfile(self.DATA_PATH):
            model_file = open(self.DATA_PATH, 'r')
            self.all_measures, self.all_targets = pickle.load(model_file)
            model_file.close()
        else:
            print("No Data")

    def test_model(self):
        "Medir el desempeño del clasificador"
        if len(self.all_targets) == 0:
            self._load_data()
        model = RandomForestClassifier()
        scores = cross_val_score(model, self.all_measures, self.all_targets,
                                 cv=5, scoring='f1')
        print("Score F1: %f" % np.mean(scores))

    def save_data(self):
        "Guardar datos de entrenamiento"
        data = (self.all_measures, self.all_targets)
        data_file = open(self.DATA_PATH, 'w')
        pickle.dump(data, data_file)
        data_file.close()

    def save_model(self):
        "Guardar modelo entrenado"
        model = RandomForestClassifier()
        model.fit(self.all_measures, self.all_targets)
        model_file = open(self.MODEL_PATH, 'w')
        pickle.dump(model, model_file)
        model_file.close()
        self.model = model
