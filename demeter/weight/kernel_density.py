"""
Kernel density algorithm.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com)
"""

import logging

import numpy as np

from scipy import ndimage
import threading
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import demeter.demeter_io.writer as wdr
import demeter.tolerance as tol

logger = logging.getLogger('demeter_runtime')


def handle_single_pft(pft_order, order_rules, final_landclasses, pft_maps, cellindexresin,
                      spat_ludataharm, kernel_maps, kernel_vector, weights, area_tolerance_scale=1.0):

    """Helper function to handle pft convultion filters. This is passed to apply_convultiion This is used to parallelize the convultion filter operation to speed up processing.

       Parameters:
           pft_order (List[int]): The order of pfts
           order_rules (List[str]): The order rules defined by user
           final_landclasses (List[int]): A list of land classes
           pft_maps (dict): A map of pfts
           cellindexresin (Union[None, int]): Input resolution
           spat_ludataharm (pd.Dataframe): Dataframe of land use


       Returns:
           list: processed convultion filters for each LT.

       """
    pft = np.where(order_rules == pft_order)[0][0]
    # get final land class name
    flc = final_landclasses[pft]

    # populate pft_maps array with base land use layer data
    pft_maps[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :]), pft] = spat_ludataharm[:, pft]

    # apply image filter

    kernel_maps[:, :, pft] = ndimage.convolve(pft_maps[:, :, pft], weights, output=None, mode='wrap')

    # attributing min value to grid-cells with zeros, otherwise they have no chance of getting selected,
    # while we might need them.

    min_seed = tol.KERNEL_MIN_SEED * area_tolerance_scale
    kernel_maps[:, :, pft][
    kernel_maps[:, :, pft] == 0] = min_seed  # np.nanmin(kernel_maps[:, :, pft], [kernel_maps[:, :, pft] > 0])

    # reshaping to the spatial grid-cell data (vector)
    kernel_vector[:, pft] = kernel_maps[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :]), pft]


def nearest_index(grid, values):
    """Index of the grid coordinate nearest to each value.

    Vectorized equivalent of ``np.argmin(abs(grid - value))`` for every value, including returning the first
    (lowest) index when two grid coordinates are equally near.

    :param grid:                    Strictly monotonic (ascending or descending) 1D array of coordinates
    :param values:                  1D array of coordinates to locate

    :return:                        Integer array of grid indices

    """
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(grid)

    descending = n > 1 and grid[0] > grid[-1]
    ascending_grid = grid[::-1] if descending else grid

    # the nearest coordinate of a monotonic grid is one of the two that bracket the value
    upper = np.clip(np.searchsorted(ascending_grid, values), 0, n - 1)
    lower = np.clip(upper - 1, 0, n - 1)

    if descending:
        upper, lower = n - 1 - upper, n - 1 - lower

    first = np.minimum(upper, lower)
    second = np.maximum(upper, lower)

    d_first = np.abs(grid[first] - values)
    d_second = np.abs(grid[second] - values)

    return np.where(d_second < d_first, second, first)


class KernelDensity:

    def __init__(self, resolution, spat_coords, final_landclasses, kernel_distance, ngrids, order_rules,
                 area_tolerance_scale=1.0):

        self.resolution = resolution
        self.spat_coords = spat_coords
        self.l_fcs = len(final_landclasses)
        self.final_landclasses = final_landclasses
        self.kernel_distance = kernel_distance
        self.ngrids = ngrids
        self.order_rules = order_rules
        self.area_tolerance_scale = area_tolerance_scale

    # minimum longitudinal extent in degrees of the observed data for the domain to be treated as global
    GLOBAL_LONGITUDE_EXTENT = 350.0

    def is_global(self):
        """Return True if the observed data spans the globe longitudinally."""
        return (self.spat_coords[:, 1].max() - self.spat_coords[:, 1].min()) >= self.GLOBAL_LONGITUDE_EXTENT

    def global_system(self):
        """
        Create geographic latitude and longitude coordinates for map grid system from user-defined resolution.

        A global domain uses the full global grid; a regional domain uses a grid inferred from the bounds of the
        observed data.

        :return:                        latitude and longitude arrays
        """
        if self.is_global():
            lat = np.arange(90 - self.resolution / 2., -90, -self.resolution)
            lon = np.arange(-180 + self.resolution / 2., 180, self.resolution)

        else:
            # grid coordinates coincide with the cell centers of the observed data, from the northernmost row and the
            #   westernmost column, so every cell maps to its own grid position
            max_lat = self.spat_coords[:, 0].max()
            min_lon = self.spat_coords[:, 1].min()

            n_lat = int(np.rint((max_lat - self.spat_coords[:, 0].min()) / self.resolution)) + 1
            n_lon = int(np.rint((self.spat_coords[:, 1].max() - min_lon) / self.resolution)) + 1

            lat = max_lat - self.resolution * np.arange(n_lat)
            lon = min_lon + self.resolution * np.arange(n_lon)

        return lat, lon

    def compute_cell_index(self, lat, lon):
        """
        Compute grid-cell indices to convert from native resolution to user-defined resolution.  Do this only for
        start year.

        :return:
        """
        # create array to hold indices
        cellindexresin = np.zeros((2, len(self.spat_coords[:, 0])))

        # populate grid indicies array with index location of base layer coord in global system (lat, lon)
        cellindexresin[0, :] = nearest_index(lat, self.spat_coords[:, 0])
        cellindexresin[1, :] = nearest_index(lon, self.spat_coords[:, 1])

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
        #pft_map_single = np.zeros((l_lat, l_lon, 1))
        kernel_maps = np.zeros((l_lat, l_lon, self.l_fcs))
        #kernel_map_single = np.zeros((l_lat, l_lon, 1))
        kernel_vector = np.zeros((self.ngrids, self.l_fcs))
        kernel_vector_temp = np.zeros((self.ngrids, self.l_fcs))
        weights = np.zeros((self.kernel_distance, self.kernel_distance))

        return pft_maps, kernel_maps, kernel_vector, weights

    def dist_iter(self):
        """
        Create distance iterator for weight assignment.

        :param weights:
        :return:
        """
        rkd = range(self.kernel_distance)
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
            dist = np.sqrt(np.power(abs(i - (self.kernel_distance - 1) / 2.), 2)
                           + np.power(abs(j - (self.kernel_distance - 1) / 2.), 2))

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

        # cells sharing a grid position would overwrite each other's land cover in the kernel density maps
        n_positions = len(np.unique(cellindexresin.astype(np.int64), axis=1).T)
        if n_positions != cellindexresin.shape[1]:
            logger.warning(f'{cellindexresin.shape[1] - n_positions} grid cells share a kernel density grid position '
                           f'with another cell; `spatial_resolution` ({self.resolution}) may be coarser than the '
                           f'observed data.')

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
        pool = ThreadPool(len(np.unique(self.order_rules)))
        #pool=ThreadPool(10)
        aux_val=np.unique(self.order_rules)
        order_rules= self.order_rules
        final_landclasses= self.final_landclasses

        pool.starmap(handle_single_pft, zip(aux_val,repeat(order_rules), repeat(final_landclasses),
                                                                  repeat(pft_maps),repeat(cellindexresin),
                                                                  repeat(spat_ludataharm),
                                                                  repeat(kernel_maps), repeat(kernel_vector),repeat(weights),
                                                                  repeat(self.area_tolerance_scale)))
        pool.terminate()

        return kernel_vector



