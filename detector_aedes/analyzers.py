#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import yaml
import os
import numpy as np

# Para el procesamiento de imagenes
from skimage import io
from skimage import transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import regionprops, label
from skimage import feature
from skimage.transform import resize, rotate, hough_line, hough_line_peaks
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage import segmentation
from skimage.future import graph
from skimage import morphology
from skimage import measure
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "config.yml"), 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
cfg_images = cfg['images']


class StickAnalizer():
    """Clase base para buscar el bajalenguas."""

    def __init__(self, lowres_width=100):
        self.lowres_width = lowres_width

    def set_current_image(self, image):
        """Definir la imagen de trabajo."""
        orig_size = image.shape[0:2]
        if np.diff(orig_size) < 0:
            image = np.swapaxes(image, 0, 1)
        self.scale = (np.amin(orig_size) / self.lowres_width)
        new_size = np.array(orig_size) / self.scale
        image_lowres = transform.resize(image, new_size)
        self.curr_im = image
        self.curr_im_lowres = image_lowres
        self.curr_im_lowres_g = rgb2gray(image_lowres)


class StickAnalizerHough(StickAnalizer):
    """Busca lineas de Hough paralelas"""

    def get_limits(self, max_width_proportion=0.45):
        """Busca dos lineas paralelas que correspondan al bajalenguas

        Args:
            max_width_proportion (float): Maxima proporcion del alto de la
                imagen (considerada apaisada) que puede abarcar el ancho del
                bajalenguas.
        Returns:
            status (str): Una descripcion del resultado de la busqueda.
            limits ([angles, dists]): Contiene dos numpy arrays. El primero
                contiene dos angulos y el segundo dos distancias. Cada par de
                angulo-distancia define la recta de arriba y de abajo del bajalenguas.
        """
        max_angle_diff = 5. / 180 * np.pi
        im_width = np.amin(self.curr_im_lowres_g.shape)
        min_dist = int(1. / 6 * im_width)
        sigma = 3
        edges = canny(self.curr_im_lowres_g, sigma)
        while np.mean(edges) < 0.01:
            sigma = (sigma - 0.1)
            if sigma < 0:
                break
            edges = canny(self.curr_im_lowres_g, sigma)
        self.edges = edges
        h, theta, d = hough_line(edges)
        params = hough_line_peaks(h, theta, d, num_peaks=6,
                                  min_distance=min_dist)
        self.params = params
        # Normalizo al ancho de la imagen
        dists = params[2] / im_width
        angles = params[1]
        dangles = pairwise_distances(angles[:, None])
        dangles = np.dstack((dangles, np.abs(dangles - np.pi)))
        dangles = np.amin(dangles, 2)
        np.fill_diagonal(dangles, np.inf)
        i, j = np.unravel_index(np.argmin(dangles), dangles.shape)
        angles = np.array([angles[i], angles[j]])
        dists = np.array([dists[i], dists[j]])
        # Ordeno los bordes para que el de arriba quede primero
        norm_dist = np.sign(angles) * dists
        sort_idx = np.argsort(norm_dist)
        if i == j:
            status = 'Sin bajalenguas'
            limits = None
        elif dangles[i, j] > max_angle_diff:
            status = 'Sin bajalenguas - mal paralelismo'
            limits = None
        elif abs(np.diff(norm_dist)) > max_width_proportion:
            status = 'Sin bajalenguas - mal ratio'
            limits = None
        elif abs(angles[0]) < 20. / 180 * np.pi:
            status = 'Sin bajalenguas - mala inclinacion'
            limits = None
        else:
            status = 'Con bajalenguas'
            limits = [angles[sort_idx], dists[sort_idx]]
        return status, limits


class SearchParams():
    """Parametros geometricos para EllipseFinder"""
    TRY_EGGS = [2, 3, 4]
    STANDARD_MAJOR_AXIS = cfg['eggs']['geometry']['major_axis']
    STANDARD_MINOR_AXIS = cfg['eggs']['geometry']['minor_axis']
    STANDARD_AREA = np.pi * STANDARD_MAJOR_AXIS * STANDARD_MINOR_AXIS / 4
    # Tolerancia para variaciones de tamanio
    TOL = float(cfg['eggs']['tolerance'])
    TOL = 1 + TOL / 100  # Convertir de porcentaje a fraccion

    def __init__(self, im_size, limits, dmin):
        angles, dists = limits
        self.im_height = np.amin(im_size)
        norm_dist = np.sign(angles) * dists
        scale = np.abs(np.diff(norm_dist)) * self.im_height
        self.scale = scale  # Ancho del bajalenguas
        normalized_dists = dists * self.im_height
        self.limits = (angles, normalized_dists)  # Limites del bajalenguas
        self.cut_width = int(self.TOL * self.STANDARD_MAJOR_AXIS * scale)  # Tamaño de la region a inspeccionar
        self.cut_width_multi = int(self.cut_width * 2)  # Tamaño de la region para huevos multiples
        self.area_up = self.STANDARD_AREA * self.TOL * scale**2
        self.area_down = self.STANDARD_AREA / self.TOL * scale**2
        self.major_axis_up = self.STANDARD_MAJOR_AXIS * self.TOL * scale
        self.major_axis_down = self.STANDARD_MAJOR_AXIS / self.TOL * scale
        self.major_axis_mean = self.STANDARD_MAJOR_AXIS * scale
        self.minor_axis_up = self.STANDARD_MINOR_AXIS * self.TOL * scale
        self.minor_axis_down = self.STANDARD_MINOR_AXIS / self.TOL * scale
        self.minor_axis_mean = self.STANDARD_MINOR_AXIS * scale
        if dmin:  # Distancia minima entre regiones
            self.dmin = dmin
        else:
            self.dmin = 1.5 * self.minor_axis_mean

    def __str__(self):
        report_vars = ['scale', 'area_up', 'area_down', 'major_axis_mean',
                       'minor_axis_mean', 'cut_width', 'cut_width_multi',
                       'dmin', 'limits']
        report = []
        for name in report_vars:
            report.append("%s: %s" % (name, str(getattr(self, name))))
        return ', '.join(report)

class EllipseFinder():
    u"""Busca elipses del tamaño definido en la configuración."""

    def find_in(self, img_g, limits=None, max_thres=0.6, thresh_step=0.05,
                dmin=None, show_settings=False):
        u"""Busca elipses del tamaño definido en la configuración.

        Para esto aplica una serie sucesiva de umbrales entre 0 y `max_thres`
        de a pasos definidos por `thresh_step`. En cada etapa busca manchas
        conexas de tamanios definidos en el archivo de configuracion e intenta
        ajustarlos por una o mas elipses

        Args:
            img_g (np.array): Imagen a analizar, si no es blanco y negro
                se convierte automaticamente.
            limits (number): Angulos y distancias de los bordes del
                bajalenguas (segun se obtienen de `StickAnalizerHough`).
            max_thres (float, optional): Maximo valor de intenisdad a usar
                como umbral para la imagen (0 es negro 1 es blanco)
            tresh_step (float, optional): Centroide de la elipse
            dmin (int, optional): Tamaño del template (matriz de `res` x `res`)
            show_settings (bool, optional): Tamaño del template (matriz de `res` x `res`

        Returns:
            status (str): Estado en el que termino la busqueda.
            res (numpy.array): Array de Nx5 donde N es el numero de regiones
                halladas. Las cinco columnas corresponden a las siguientes
                propiedades de cada region: centroide_i, centroide_j,
                correlacion, contraste, aspecto.
        """

        if len(img_g.shape) > 2:
            img_g = rgb2gray(img_g)
        if not limits:
            return ('No busco sin bajalenguas', None)
        p = SearchParams(img_g.shape, limits, dmin)
        if p.cut_width <= 5:
            return ('Imagen con poca resolucion', None)

        # Numpy array para los resultados
        # Columnas: centroid_i, centroid_j, correlations, contrasts, aspects
        res = np.array([[], [], [], [], []]).T
        seen_centroids = []
        if show_settings:
            print(p)

        # Loop Principal
        for th in np.arange(0, max_thres, thresh_step):
            binary = img_g < th
            labels = label(binary)
            regions = regionprops(labels)
            for region in regions:
                if region.area < p.area_down or region.area > 4 * p.area_up:
                    continue
                if not self._region_is_in_stick(region, p):
                    continue
                if len(seen_centroids) == 0:
                    dist_mat = np.inf
                else:
                    dist_mat = pairwise_distances(np.array([region.centroid]),
                                                  np.vstack(seen_centroids))
                if np.any(dist_mat < p.dmin):
                    continue  # Solo sigue si es un centroide nuevo
                seen_centroids.append(region.centroid)
                myreg = self.cut_region(region.centroid, img_g, delta=p.cut_width)
                recorte, new_region, labels, i_max_region = myreg
                if recorte is None:
                    continue
                contrast = self.calculate_contrast(recorte, labels == (i_max_region + 1))
                if contrast < 0.1:
                    continue
                # Si el area es compatible con un solo huevo
                if region.area < p.area_up:
                    if region.convex_area > region.area * 1.5:
                        continue  # No seguir si la region es muy no-convexa
                    min_ax = new_region.minor_axis_length
                    maj_ax = new_region.major_axis_length
                    try:
                        aspect = maj_ax / min_ax
                    except ZeroDivisionError:
                        aspect = np.nan
                    template = self.generate_template(p.minor_axis_mean,
                                                      p.major_axis_mean,
                                                      -new_region.orientation,
                                                      (new_region.centroid[1], new_region.centroid[0]),
                                                      2 * p.cut_width)
                    correlation = self._nan_correlation(template, recorte)
                    if correlation < 0.5:
                        continue
                    if cfg['debug_mode'] and correlation > 0.8:
                        self._plot_debug(myreg, p, dist_mat)
                    c_i = region.centroid[0] + new_region.centroid[0] - p.cut_width
                    c_j = region.centroid[1] + new_region.centroid[1] - p.cut_width
                    res = np.vstack([res, [c_i, c_j, correlation, contrast, aspect]])
                # Si creemos que hay varios pegados
                elif region.area < 4 * p.area_up:
                    myreg = self.cut_region(region.centroid, img_g,
                                            delta=p.cut_width_multi, target_max=True)
                    if myreg[0] is None:
                        continue
                    out = self._correlate_many_eggs(myreg, p)
                    temp_xcorrs, temp_aspects, temp_centroids = out
                    i_max_corrs = np.argmax(temp_xcorrs)
                    max_corr = temp_xcorrs[i_max_corrs]
                    if max_corr < 0.5:
                        continue
                    if cfg['debug_mode'] and max_corr > 0.8:
                        self._plot_debug(myreg, p, dist_mat)
                    # Agregar los resultados   TERMINAR!
                    add_corrs = [max_corr] * p.TRY_EGGS[i_max_corrs]
                    add_aspects = temp_aspects[i_max_corrs]
                    add_contrasts = [contrast] * p.TRY_EGGS[i_max_corrs]
                    referenced_centroids = [(region.centroid[0] - p.cut_width_multi + c[0],
                                            region.centroid[1] - p.cut_width_multi + c[1])
                                            for c in temp_centroids[i_max_corrs]
                                            ]
                    referenced_centroids = np.array(referenced_centroids).T
                    add_centroids_i, add_centroids_j = referenced_centroids
                    add_res = np.vstack([add_centroids_i, add_centroids_j,
                                         add_corrs, add_contrasts, add_aspects]).T
                    res = np.vstack([res, add_res])

        return ('Status OK', res)

    def _plot_debug(self, myreg, params, dist_mat):
        recorte, new_region, labels, i_max_region = myreg
        plt.imshow(recorte)
        plt.draw()
        plt.waitforbuttonpress()

    def _correlate_many_eggs(self, myreg, params):
        """Calcula las correlaciones probando con templates de distinta
        cantidad de huevos. """
        recorte, new_region, labels, i_max_region = myreg
        mask = (labels == i_max_region + 1)
        idx = np.flatnonzero(mask)
        i, j = np.unravel_index(idx, mask.shape)
        X = np.vstack((j, i)).T
        temp_xcorrs = []
        temp_aspects = [[] for x in params.TRY_EGGS]
        temp_centroids = [[] for x in params.TRY_EGGS]
        for iegg, eggnum in enumerate(params.TRY_EGGS):
            gm = GaussianMixture(n_components=eggnum)
            gm.fit(X)
            templates = []
            for n, covariances in enumerate(gm.covariances_):
                v, w = np.linalg.eigh(covariances)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                aspect = v[1] / v[0]
                if not isinstance(aspect, float):
                    aspect = aspect[0]
                temp_aspects[iegg].append(aspect)
                temp_centroids[iegg].append(np.flipud(gm.means_[n, :2]))
                t = self.generate_template(params.minor_axis_mean, params.major_axis_mean,
                                           np.pi / 2 + angle,
                                           gm.means_[n, :2], 2 * params.cut_width_multi)
                t[np.isnan(t)] = np.inf
                templates.append(t)
            templates = np.dstack(templates)
            template = np.amin(templates, -1)
            template[np.isinf(template)] = np.nan
            correlation = self._nan_correlation(recorte, template)
            temp_xcorrs.append(correlation)
        return temp_xcorrs, temp_aspects, temp_centroids

    @staticmethod
    def _correct_corr(R, n, k):
        """Corrige la correlacion por cantidad de parametros.

        Args:
            R (float): Correlacion
            n (int): Cantidad de datos
            k (int): Cantidad de parametros
        Returns:
            corrected (float): El valor corregido de R
        """
        try:
            corrected = 1 - ((1 - R**2) * float(n - 1) / (n - k - 1))
        except:
            corrected = None
        return corrected

    @staticmethod
    def _nan_correlation(matrix1, matrix2):
        """Calcula la correlacion entre dos matrices excluyendo los valores nan."""
        x = matrix1.flatten()
        y = matrix2.flatten()
        gi = (~np.isnan(x)) & (~np.isnan(y))
        corr_mat = np.corrcoef(x[gi], y[gi])
        return corr_mat[0, 1]

    @staticmethod
    def cut_region(centroid, img, delta=15, target_max=False):
        """Recorta una region cuadrada alrededor de un punto.

        Luego de hacer el recorte se aplica un umbral con el metodo de Otsu
        para binarizar la imagen. Se conserva solo la region mas grande o la
        mas centra segun el valor de `target_max`

        Args:
            centroid (tuple of floats): Coordenadas (fila, columna) del centro
                alredeor del cual se recorta.
            img (numpy array): Imagen base a recortar.
            delta (int): Se recorta una distancia de +- delta alredeor del centro
            target_max (bool): Si es True, solo se conserva la region conexa
                mas grande. Si es False, se conserva la region conexa que esta
                en el centro.
        """
        i = int(centroid[0])
        i_min = i - delta
        i_max = i + delta
        if i_min < 0 or i_max >= img.shape[0]:
            return None, None, None, None
        j = int(centroid[1])
        j_min = j - delta
        j_max = j + delta
        if j_min < 0 or j_max >= img.shape[1]:
            return None, None, None, None
        recorte = img[i_min:i_max, j_min:j_max]
        thresh = threshold_otsu(recorte)
        new_mask = recorte < thresh
        labels = label(new_mask)
        new_props = regionprops(labels)
        if target_max:
            sizes = [r.area for r in new_props]
            i_target_region = np.argmax(sizes)
        else:
            i_target_region = labels[delta, delta] - 1
        new_region = new_props[i_target_region]
        return recorte, new_region, labels, i_target_region

    @staticmethod
    def calculate_contrast(img, mask):
        """Calcula el contraste en una imagen en base a una mascara."""
        Imin = np.mean(img[mask])
        Imax = np.mean(img[~mask])
        contrast = (Imax - Imin) / (Imax + Imin)
        return contrast

    def _region_is_in_stick(self, region, params):
        """Decide si una dada region esta o no dentro del bajalenguas."""
        ylimits = []
        ys, xs = region.centroid
        for angle, dist in zip(*params.limits):
            yl = -np.cos(angle) / np.sin(angle) * xs + dist / np.sin(angle)
            ylimits.append(yl)
        point_in_stick = (ys > (ylimits[0] - 20)) & (ys < (ylimits[1] + 20))
        return point_in_stick

    @staticmethod
    def generate_template(minor_axis, major_axis, angle, centroid, res):
        """Crea un `template` de huevo.

        Args:
            - minor_axis (number): Eje menor de la elipse.
            - major_axis (number): Eje mayor de la elipse.
            - angle (float): Angulo de la elipse.
            - centroid (tuple of numbers): Centroide de la elipse
            - res (int): Tamaño del template (matriz de `res` x `res`)

        Returns:
            - template (`res` x `res` array): template de un huevo de acuerdo
            a los parametrs dados.
        """
        A = major_axis
        B = minor_axis
        xc = centroid[0]
        yc = centroid[1]
        X, Y = np.meshgrid(np.arange(res, dtype=float),
                           np.arange(res, dtype=float))
        Xcorr = (X - xc)
        Ycorr = (Y - yc)
        Zel = (((Xcorr * np.cos(angle) + Ycorr * np.sin(angle)) / A)**2 +
               ((Xcorr * np.sin(angle) - Ycorr * np.cos(angle)) / B)**2)
        B = B * 1.5
        Zel_edge = (((Xcorr * np.cos(angle) + Ycorr * np.sin(angle)) / A)**2 +
                    ((Xcorr * np.sin(angle) - Ycorr * np.cos(angle)) / B)**2)
        Zel[Zel > 1] = 1
        Zel[(Zel_edge < 1.2) & (Zel_edge > 0.9)] = 1.1
        Zel[(Zel < 0.6)] = np.amax(Zel[(Zel < 0.5)])
        Zconv = convolve2d(Zel, np.ones((3, 3)) / 9, mode='same')
        Zconv[Zel_edge > 1.4] = 1
        Zconv[Zel_edge > 1.9] = np.nan
        return Zconv

###############################################################################
#
#      A Partir de aca siguen los algoritmos viejos.
#
#
#
#
###############################################################################


class StickAnalizerMulti(StickAnalizer):
    """Busca el bajalenguas con tres metodos distintos.

    Los tres metodos son:
     - A partir de los bordes.
     - Por el color.
     - Umbral de brillo.
    """

    def __init__(self, *args, **kwargs):
        super(StickAnalizerMulti, self).__init__(*args, **kwargs)
        self._load_patches()

    def get_clipped_image(self):
        """Devuelve la imagen recortada."""
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
