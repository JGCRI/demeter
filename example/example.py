"""
Demeter example run.
"""

import os

from demeter.model import Demeter


if __name__ == "__main__":

    # default
    ini = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')

    # instantiate Demeter
    dm = Demeter(config=ini)

    # run all time steps as set in config file
    # dm.execute()

    # run a random ensemble of configurations
    dm.ensemble()

    # clean up
    del dm
