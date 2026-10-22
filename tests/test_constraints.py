"""Constraint files are aligned to the base layer by cell id."""

import logging

import numpy as np
import pytest

from demeter.constraints import ApplyConstraints, ValidationException


def _constraints(grid_ids):
    c = ApplyConstraints.__new__(ApplyConstraints)
    c.spat_grid_id = np.asarray(grid_ids)
    c.ngrids = len(grid_ids)
    c.logger = logging.getLogger("demeter_runtime")
    return c


def test_shuffled_rows_are_aligned_by_id():
    grid_ids = np.array([3, 7, 11, 12])
    values = np.array([0.3, 0.7, 0.11, 0.12])
    order = np.array([2, 0, 3, 1])

    aligned = _constraints(grid_ids).align_constraint(grid_ids[order].astype(float), values[order], "c.csv")

    np.testing.assert_array_equal(aligned, values)


def test_unmatched_ids_fall_back_to_row_order(caplog):
    values = np.array([0.1, 0.2, 0.3])

    with caplog.at_level(logging.WARNING, logger="demeter_runtime"):
        aligned = _constraints([10, 20, 30]).align_constraint(np.array([1.0, 2.0, 3.0]), values, "c.csv")

    np.testing.assert_array_equal(aligned, values)
    assert "assuming its rows are in base layer order" in caplog.text


def test_wrong_length_without_matching_ids_raises():
    with pytest.raises(ValidationException, match="one row per grid cell"):
        _constraints([10, 20, 30]).align_constraint(np.array([1.0, 2.0]), np.array([0.1, 0.2]), "c.csv")
