"""
Demeter example run.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import os

from demeter.model import Demeter


if __name__ == "__main__":

    # config file in example directory
    ini = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')

    # instantiate Demeter
    dm = Demeter(config=ini)

    # run all time steps as set in config file
    # dm.execute()

    # run a random ensemble of parameters
    dm.ensemble()

    # clean up
    del dm
