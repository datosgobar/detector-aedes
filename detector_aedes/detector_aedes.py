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
from analyzers import StickAnalizerHough, EllipseFinder


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

    def __init__(self, input_connector, output_connector=None):
        self.input_connector = input_connector
        self.output_connector = output_connector
        self.stick_analizer = StickAnalizerHough()
        self.egg_finder = EllipseFinder()

    def process(self, image_id=None, show_results=False, start_from=0,
                ):
        """Función principal para procesar las imágenes disponibles.

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
                print('Procesando {:s}'.format(image_id))
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
        self.image = image
        if image is None:
            warnings.warn("La imagen {:s} no pudo ser cargada"
                          .format(image_id))
        self.stick_analizer.set_current_image(image)
        self.stick_status, self.stick_limits = self.stick_analizer.get_limits()
        if self.stick_status != 'Sin bajalenguas':
            finder_status, egg_props = self.egg_finder.find_in(image, limits=self.stick_limits,
                                                         show_settings=True)
            self.finder_status = finder_status
            if finder_status in ['Status OK', 'Early stop']:
                self.egg_props = egg_props
                self.classify(method='Thresholds')
            else:
                self.egg_count = None
                self.doubt_count = None
            self.output_connector.write_output(image_id,
                                               self.stick_status + ' / ' + finder_status,
                                               self.egg_count, self.doubt_count)
            print(self.egg_count)
        else:
            print(self.stick_status)
            self.output_connector.write_output(image_id,
                                               self.stick_status,
                                               None, None)

    def classify(self, method='Threshods'):
        centroids_i, centroids_j, correlations, contrasts, aspects = self.egg_props.T
        self.centroids = zip(centroids_i, centroids_j)
        self.good_points = (correlations > 0.8) & (contrasts > 0.3)
        self.semi_points = (correlations > 0.8) & (contrasts <= 0.3)
        self.egg_count = np.sum(self.good_points)
        self.doubt_count = np.sum(self.semi_points)

    # FUNCION DE ENTRENAMIENTO DEL CLASIFICADOR (DEPRECADO)
    #    def train_model(self, trainsamples=None, show_mask=False):
    #     self.egg_finder.start_trainning()
    #     if isinstance(trainsamples, list):
    #         samples = trainsamples
    #     else:
    #         im_ids = self.input_connector.get_image_ids()
    #         if len(im_ids) < trainsamples:
    #             raise ValueError('No hay suficientes imagenes para entrenar.\
    #                              (Nfiles=%i)' % len(im_ids))
    #         samples = np.random.choice(im_ids, trainsamples,
    #                                    replace=False)  # Imagenes al azar
    #     for im_id in samples:
    #         image = self.input_connector.get_image(im_id)
    #         self.stick_analizer.set_current_image(image)
    #         self.clipped_image = self.stick_analizer.get_clipped_image()
    #         self.egg_finder.find_in(self.clipped_image)
    #         self.plot_selector(show_mask=show_mask)
    #     self.egg_finder.save_model()

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
        if self.stick_status == 'Sin bajalenguas':
            print("No puedo graficar: " + self.stick_status)
            return
        if self.finder_status != 'Status OK':
            print("No puedo graficar: " + self.finder_status)
            return
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('key_press_event',
                                        self._handle_fig_event)
            self.fig.canvas.mpl_connect('button_press_event',
                                        self._handle_fig_event)
        self.next_figure = False
        self.fig.clf()
        ax = self.fig.add_subplot(121)
        ax.set_title('Presione N para continuar analizando la proxima foto.')
        ax.imshow(self.image)
        if self.stick_limits:
            count = -1
            for angle, dist in zip(*self.stick_limits):
                count += 1
                dist = dist * np.amin(self.image.shape[:2])
                y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1 = (dist - self.image.shape[1] * np.cos(angle)) / np.sin(angle)
                ax.plot((0, self.image.shape[1]), (y0, y1), '-',
                         color=['r', 'g'][count], linewidth=0.5)
        if len(self.centroids) > 0:
            centers = np.vstack(self.centroids)
            ax.scatter(centers[self.good_points, 1],
                       centers[self.good_points, 0],
                       s=80, marker='x',
                       color='r')
            ax.scatter(centers[self.semi_points, 1],
                       centers[self.semi_points, 0],
                       s=80, marker='x',
                       color='y')
            # for c, q, cont in zip(centers, self.good_points, self.egg_props[2]):
            #     if q:
            #         ax.text(c[1], c[0], cont)
        ax2 = self.fig.add_subplot(122)
        ax2.plot(self.egg_props[:, 2], self.egg_props[:, 3], '.')
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
