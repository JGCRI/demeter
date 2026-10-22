"""Kernel density helpers."""

import numpy as np
import pytest

from demeter.weight.kernel_density import nearest_index


def _reference(grid, values):
    return np.array([np.argmin(abs(grid - v)) for v in values])


@pytest.mark.parametrize("res", [0.5, 0.25, 0.0833333, 0.00151872768])
@pytest.mark.parametrize("descending", [False, True])
def test_nearest_index_matches_argmin(res, descending):
    rng = np.random.default_rng(42)
    lo, hi = -5.0, 5.0
    grid = np.arange(hi - res / 2, lo, -res) if descending else np.arange(lo + res / 2, hi, res)
    centers = np.arange(lo + res / 2, hi, res)[:2000]

    # include exact ties (half a cell from two grid coordinates) and values outside the grid
    values = np.concatenate([centers, centers + res / 2, centers - res / 2, grid[:1000],
                             rng.uniform(lo - 1, hi + 1, 5000)])

    np.testing.assert_array_equal(nearest_index(grid, values), _reference(grid, values))


def test_nearest_index_single_coordinate():
    np.testing.assert_array_equal(nearest_index(np.array([1.0]), np.array([-3.0, 1.0, 9.0])), [0, 0, 0])
