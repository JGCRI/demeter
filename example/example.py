"""
Demeter example run.
"""

import os

from demeter.model import Demeter


def main(cfg=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')):

    # instantiate Demeter
    dm = Demeter(config=cfg)

    # run all time steps as set in config file
    dm.execute()


if __name__ == "__main__":

    # path to config file
    ini = '/Users/ladmin/desktop/luh_demeter/config_luh.ini'

    main()