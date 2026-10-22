"""
Scale-aware area tolerances.

The default convergence tolerance and the rounding applied to target land change were calibrated in km2 for the
v2 default resolution of 0.25 degree.  At finer resolutions those absolute values become a large fraction of a grid
cell and erase real change, so they are scaled by the ratio of cell area to the reference cell area.  At the
reference resolution the scale is exactly 1 and results are bit-for-bit identical to v2.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER
"""

import numpy as np

# resolution in geographic degrees at which the default area tolerances were calibrated
REFERENCE_RESOLUTION = 0.25

# default convergence tolerance in km2 at the reference resolution
DEFAULT_ERRORTOL_KM2 = 0.001

# decimals (in km2 at the reference resolution) used to remove values very close to zero from target change
ROUNDING_DECIMALS = 4
ROUNDING_FACTOR = 10.0 ** ROUNDING_DECIMALS

# value assigned to zero kernel density cells (in km2 at the reference resolution) so they can still be selected
KERNEL_MIN_SEED = 0.0000000001

# floor (in km2 at the reference resolution) for the maximum kernel density used in normalization
KERNEL_NORMALIZATION_FLOOR = 0.00000001


def area_tolerance_scale(resolution):
    """Ratio of the grid cell area at ``resolution`` to the grid cell area at the reference resolution.

    :param resolution:              Spatial resolution in geographic degrees

    :return:                        float scale factor

    """
    return (resolution / REFERENCE_RESOLUTION) ** 2


def default_errortol(resolution):
    """Default convergence tolerance in km2 for a spatial resolution in geographic degrees."""
    return DEFAULT_ERRORTOL_KM2 * area_tolerance_scale(resolution)


def round_area(value, scale):
    """Round a scalar area in km2 to the scaled rounding precision.

    Equivalent to ``round(value, 4)`` when ``scale`` is 1.

    """
    return round(value / scale, ROUNDING_DECIMALS) * scale


def is_negative_area(value, scale):
    """Return True if ``round_area(value, scale) < 0`` without rounding.

    Mirrors NumPy's rounding (``rint(x * 10**decimals) / 10**decimals``) applied to ``np.float64`` values: the rounded
    value is negative exactly when the scaled product is below -0.5.

    """
    return (value / scale) * ROUNDING_FACTOR < -0.5


def round_area_array(arr, scale):
    """Round an array of areas in km2 to the scaled rounding precision.

    Equivalent to ``np.around(arr, decimals=4)`` when ``scale`` is 1.

    """
    return np.around(arr / scale, decimals=ROUNDING_DECIMALS) * scale
