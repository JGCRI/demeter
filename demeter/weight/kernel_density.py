#!/usr/bin/env python

"""
Kernel density algorithm.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com)
"""

import numpy as np

from scipy import ndimage

import demeter.demeter_io.writer as wdr


class KernelDensity:

    def __init__(self, resolution, spat_coords, final_landclasses, kerneldistance, ngrids, kernel_map_dir, order_rules,
                 map_kernels):

        self.resolution = resolution
        self.spat_coords = spat_coords
        self.l_fcs = len(final_landclasses)
        self.final_landclasses = final_landclasses
        self.kerneldistance = kerneldistance
        self.ngrids = ngrids
        self.order_rules = order_rules
        self.kernel_map_dir = kernel_map_dir
        self.map_kernels = map_kernels

    def global_system(self):
        """
        Create geographic latitude and longitude coordinates for map grid system from user-defined resolution.

        :param resolution:              User-defined resolution setting
        :return:                        latitude and longitude arrays
        """

        # get latitude and longitude for grid system
        lat = np.arange(90 - self.resolution / 2., -90, -self.resolution)
        lon = np.arange(-180 + self.resolution / 2., 180, self.resolution)

        return lat, lon

    def compute_cell_index(self, lat, lon):
        """
        Compute grid-cell indices to convert from native resolution to user-defined resolution.  Do this only for
        start year.

        :return:
        """
        # get the number of grid cells in the base land use layer data
        n = len(self.spat_coords[:, 0])

        # create zeros array to hold indices
        cellindexresin = np.zeros((2, n))

        # populate grid indicies array with index location of base layer coord in global system (lat, lon)
        for i in range(n):
            cellindexresin[0, i] = np.argmin(abs(lat - self.spat_coords[i, 0]))
            cellindexresin[1, i] = np.argmin(abs(lon - self.spat_coords[i, 1]))

        return cellindexresin

    def prep_arrays(self, lat, lon):
        """
        Prepare empty arrays used in kernel density calculation.

        :return:
        """
        l_lat = len(lat)
        l_lon = len(lon)

        # create empty arrays used in kernel density calculation
        pft_maps = np.zeros((l_lat, l_lon, self.l_fcs))
        kernel_maps = np.zeros((l_lat, l_lon, self.l_fcs))
        kernel_vector = np.zeros((self.ngrids, self.l_fcs))
        weights = np.zeros((self.kerneldistance, self.kerneldistance))

        return pft_maps, kernel_maps, kernel_vector, weights

    def dist_iter(self):
        """
        Create distance iterator for weight assignment.

        :param weights:
        :return:
        """
        rkd = range(self.kerneldistance)
        l = []
        for i in rkd:
            for j in rkd:
                l.append([i, j])

        return l

    def convolution_filter(self, weights):
        """
        Convolution filter (distance weighted, function of square of the distance).

        :return:
        """
        # create iterator and get
        rw = self.dist_iter()

        # populate convolution filter
        for i, j in rw:

            # calculate weighted distance
            dist = np.sqrt(np.power(abs(i - (self.kerneldistance - 1) / 2.), 2)
                           + np.power(abs(j - (self.kerneldistance - 1) / 2.), 2))

            # assign to weights
            weights[i, j] = 1 / np.power(dist, 2)

        return weights

    def preprocess_kernel_density(self):
        """
        Calculate and map PFT kernel density.

        :return:
        """
        # get latitude and longitude index for map grid system
        lat, lon = self.global_system()

        # compute grid-cell indices to convert from native resolution to user-defined resolution.
        #   Do this only for start year.
        cellindexresin = self.compute_cell_index(lat, lon)

        # prepare empty arrays used in kernel density calculation.
        kd_arrays = self.prep_arrays(lat, lon)
        pft_maps, kernel_maps, kernel_vector, weights = kd_arrays

        # convolution filter (distance weighted, function of square of the distance)
        weights = self.convolution_filter(weights)

        return [lat, lon, cellindexresin, pft_maps, kernel_maps, kernel_vector, weights]

    def apply_convolution(self, cellindexresin, pft_maps, kernel_maps, lat, lon, yr, kernel_vector, weights,
                          spat_ludataharm):
        """
        Apply convolution filter to compute kernel density.

        :return:
        """
        for pft_order in np.unique(self.order_rules):

            # get target PFT
            pft = np.where(self.order_rules == pft_order)[0][0]

            # get final land class name
            flc = self.final_landclasses[pft]

            # populate pft_maps array with base land use layer data
            pft_maps[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :]), pft] = spat_ludataharm[:, pft]

            # apply image filter
            kernel_maps[:, :, pft] = ndimage.filters.convolve(pft_maps[:, :, pft], weights, output=None, mode='wrap')

            # attributing min value to grid-cells with zeros, otherwise they have no chance of getting selected,
            #   while we might need them.
            kernel_maps[:, :, pft][kernel_maps[:, :, pft] == 0] = np.nanmin(kernel_maps[:, :, pft][kernel_maps[:, :, pft] > 0])

            # add to map array if user selects to plot them
            if self.map_kernels == 1:
                wdr.map_kernel_density(pft_maps[:, :, pft], kernel_maps[:, :, pft], lat, lon, flc, yr, self.kernel_map_dir)

            # reshaping to the spatial grid-cell data (vector)
            kernel_vector[:, pft] = kernel_maps[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :]), pft]

        return kernel_vector
