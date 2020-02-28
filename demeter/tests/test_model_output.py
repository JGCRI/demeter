"""test__outputs.py

Tests to ensure high-level functionality and outputs remain consistent.

@author Chris R. Vernon (chris.vernon@pnnl.gov)
@license BSD 2-Clause

"""

import os
import pkg_resources
import unittest

import pandas as pd

from demeter import Demeter


class TestOutputs(unittest.TestCase):
    """Test configuration integrity."""

    RUN_DIR = pkg_resources.resource_filename('demeter', 'tests/data')
    COMP_2015 = pkg_resources.resource_filename('demeter', 'tests/data/comp_data/landcover_2015_timestep.csv')
    CONFIG_FILE = pkg_resources.resource_filename('demeter', 'tests/data/config.ini')

    def test_proj_outputs(self):
        """Test for projection outputs"""

        # read in comp data from rasters to arrays
        comp_df = pd.read_csv(TestOutputs.COMP_2015)

        run = Demeter(config_file=TestOutputs.CONFIG_FILE, run_single_land_region={'region_id': 5, 'metric_id': 142})

        run.initialize()

        run.c.root_dir = TestOutputs.RUN_DIR

        for _ in range(2005, 2020, 5):

            run.timestep.next()

        run.close()

        # read in run data from rasters to arrays
        output_dir = os.path.split(run.c.out_dir)[-1]
        run_2015 = os.path.join(TestOutputs.RUN_DIR, 'outputs', output_dir, 'spatial_landcover_tabular', 'landcover_2015_timestep.csv')
        run_df = pd.read_csv(run_2015)

        # test equality
        pd.testing.assert_frame_equal(comp_df, run_df)


if __name__ == '__main__':
    unittest.main()
