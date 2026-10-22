"""
Utilities shared by the intensification and expansion algorithms.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER
"""

import logging
import os
import threading

import numpy as np


# upper bound on land swap iterations for a single land class pair in one region and metric
MAX_CONVERSION_ITERATIONS = 100000

logger = logging.getLogger('demeter_runtime')


def iteration_limit_reached(iterations, reg, met, pft, pft_toconv):
    """Return True and warn if a conversion loop has reached the iteration cap."""
    if iterations >= MAX_CONVERSION_ITERATIONS:
        logger.warning(f'Land conversion stopped after {MAX_CONVERSION_ITERATIONS} iterations without converging for '
                       f'region index {reg}, metric {met}, land class {pft} from {pft_toconv}.')
        return True
    return False


class DiagnosticWriter:
    """Thread-safe writer for a per time step diagnostic CSV file.

    :param base_file:           Full path with file name and extension of the diagnostic file; the time step is
                                appended to the file name
    :param yr:                  Time step being processed

    """

    HEADER = 'region_id,metric_id,from_pft,to_pft,target_value\n'

    def __init__(self, base_file, yr):

        file_name, extension = os.path.splitext(base_file)
        self.path = '{0}_{1}{2}'.format(file_name, yr, extension)

        self._lock = threading.Lock()
        self._file = open(self.path, 'w')
        self._file.write(self.HEADER)

    def write(self, line):
        """Write a line; safe to call from multiple threads."""
        with self._lock:
            self._file.write(line)

    def close(self):
        """Close the file."""
        with self._lock:
            self._file.close()


def open_diagnostic(c, base_file, yr):
    """Return a DiagnosticWriter if the user selected diagnostics and outputs are written; otherwise None."""
    if c.diagnostic == 1 and c.write_outputs:
        return DiagnosticWriter(base_file, yr)

    return None


class RuleOrder:
    """Precomputed processing order of land classes and of their conversion targets.

    Equivalent to looking up ``np.where(order_rules == value)[0][0]`` for each unique treatment order value and
    ``np.where(transition_rules[pft, :] == rank)[0][0]`` for each conversion priority rank, computed once instead of
    in every region and metric.

    :param order_rules:         Treatment order value per land class
    :param transition_rules:    Conversion priority rank (land class x land class)

    """

    def __init__(self, order_rules, transition_rules):

        order_rules = np.asarray(order_rules)
        transition_rules = np.asarray(transition_rules)

        # land class indices in the order they are treated
        self.pfts = [int(np.where(order_rules == value)[0][0]) for value in np.unique(order_rules)]

        # for each land class, the land classes to convert in order of priority
        self.targets = {pft: [int(np.where(transition_rules[pft, :] == rank)[0][0])
                              for rank in np.arange(1, len(transition_rules[pft]), 1)]
                        for pft in self.pfts}


class TransitionRecorder:
    """Sparse, thread-safe record of land converted between land classes during a time step.

    Replaces a dense (cells x land classes x land classes) array, which does not fit in memory at high resolution.

    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cells = []
        self._from = []
        self._to = []
        self._area = []

    def record(self, cells, from_pft, to_pft, area):
        """Record area (km2) converted from one land class to another in each of the given grid cells."""
        if len(cells) == 0:
            return

        with self._lock:
            self._cells.append(np.asarray(cells, dtype=np.int64))
            self._from.append(np.full(len(cells), from_pft, dtype=np.int64))
            self._to.append(np.full(len(cells), to_pft, dtype=np.int64))
            self._area.append(np.asarray(area, dtype=float))

    def arrays(self):
        """Return (cells, from_pft, to_pft, area) arrays of every recorded conversion."""
        if not self._cells:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, np.empty(0, dtype=float)

        return (np.concatenate(self._cells), np.concatenate(self._from), np.concatenate(self._to),
                np.concatenate(self._area))

    def summarize(self, spat_region, spat_metric, final_landclasses):
        """Total converted area per region, metric, and land class pair.

        :param spat_region:             Region id of each grid cell
        :param spat_metric:             Original metric (e.g., basin) id of each grid cell
        :param final_landclasses:       Land class names indexed by land class position

        :return:                        Data frame [region_id, metric_id, from_landclass, to_landclass, sqkm_change]

        """
        import pandas as pd

        cells, from_pft, to_pft, area = self.arrays()
        landclasses = np.asarray(final_landclasses)

        df = pd.DataFrame(dict(region_id=np.asarray(spat_region)[cells],
                               metric_id=np.asarray(spat_metric)[cells].astype(np.int64),
                               from_landclass=landclasses[from_pft],
                               to_landclass=landclasses[to_pft],
                               sqkm_change=area))

        keys = ['region_id', 'metric_id', 'from_landclass', 'to_landclass']
        return df.groupby(keys, as_index=False)['sqkm_change'].sum()
