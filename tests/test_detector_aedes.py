#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests del modulo detector_aedes."""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement
import unittest
import nose
import os

import detector_aedes as da
import numpy as np
from skimage.io import imread

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class DetectorAedesTestCase(unittest.TestCase):

    def setUp(self):
        self.img = imread(os.path.join(TEST_DIR, 'test_data', 'test_image.png'))

    def tearDown(self):
        pass

    def test_stick_finder(self):
        sah = da.StickAnalizerHough()
        sah.set_current_image(self.img)
        status, limits = sah.get_limits()
        assert(status == 'Con bajalenguas')

    def test_egg_counter(self):
        sah = da.StickAnalizerHough()
        sah.set_current_image(self.img)
        status, limits = sah.get_limits()
        el = da.EllipseFinder()
        status, out_data = el.find_in(self.img, limits=limits, max_thres=0.6)
        centroids_i, centroids_j, correlations, contrasts, aspects = out_data.T
        egg_count = np.sum((correlations > 0.8) & (contrasts > 0.4))
        assert(egg_count == 3)

    def test_main_process(self):
        ic = da.FolderInputConnector(os.path.join(TEST_DIR, 'test_data'))
        oc = da.FileOutputConnector(os.path.join(TEST_DIR, 'test_data', 'test_out.csv'))
        aedes = da.AedesDetector(input_connector=ic, output_connector=oc)
        aedes.process()
        fsample = open(os.path.join(TEST_DIR, 'test_data', 'sample_test_out.csv'), 'r')
        fout = open(os.path.join(TEST_DIR, 'test_data', 'test_out.csv'), 'r')
        for l1, l2 in zip(fsample.readlines(), fout.readlines()):
            l1_split = l1.strip().split(',')
            l2_split = l2.strip().split(',')
            print(l1_split)
            print(l2_split)
            assert(l1_split[1:] == l2_split[1:]) # Ignora el id que depende de la carpeta en que este
        fsample.close()
        fout.close()


if __name__ == '__main__':
    nose.run(defaultTest=__name__)
