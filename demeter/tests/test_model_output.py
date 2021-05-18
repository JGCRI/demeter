"""test_model_outputs.py

Tests to ensure high-level functionality and outputs remain consistent.

@author Chris R. Vernon (chris.vernon@pnnl.gov)
@license BSD 2-Clause

"""

import pkg_resources
import unittest

import pandas as pd

from demeter import Model


class TestOutputs(unittest.TestCase):
    """Test configuration integrity."""

    RUN_DIR = pkg_resources.resource_filename('demeter', 'tests/data')
    GCAMWRAPPER_DF = pd.read_pickle(pkg_resources.resource_filename('demeter', 'tests/data/inputs/projected/land_df.pkl'))
    COMP_2010 = pd.read_pickle(pkg_resources.resource_filename('demeter', 'tests/data/comp_data/demeter_2010.pkl'))
    COMP_2015 = pd.read_pickle(pkg_resources.resource_filename('demeter', 'tests/data/comp_data/demeter_2015.pkl'))

    def test_proj_outputs_using_args(self):
        """Test for projection outputs by passing arguments"""

        # instantiate demeter model
        model = Model(run_dir=self.RUN_DIR,
                      gcamwrapper_df=self.GCAMWRAPPER_DF,
                      write_outputs=False)

        # initialize demeter
        model.initialize()

        # process first year
        demeter_2010 = model.process_step()
        demeter_2015 = model.process_step()

        # cleanup logger
        model.cleanup()

        # test equality
        pd.testing.assert_frame_equal(self.COMP_2010, demeter_2010)
        pd.testing.assert_frame_equal(self.COMP_2015, demeter_2015)


if __name__ == '__main__':
    unittest.main()
