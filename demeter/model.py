"""
Class to run Demeter model for all defined steps.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import logging
import os
import sys
import time

from demeter.config_reader import ReadConfig
from demeter.process import ProcessStep
from demeter.staging import Stage


class Demeter:

    def __init__(self, root_dir=None, config_file=None, run_single_land_region=None):

        self.dir = root_dir
        self.config_file = config_file
        self.run_single_land_region = run_single_land_region

        # instantiate config
        self.c = ReadConfig(self.config_file, self.run_single_land_region)

        # build logfile path
        self.logfile = os.path.join(self.c.log_dir, 'logfile_{}_{}.log'.format(self.c.scenario, self.c.dt))

        self.s = None
        self.timestep = None

    @staticmethod
    def make_dir(pth):
        """Create dir if not exists."""

        if not os.path.exists(pth):
            os.makedirs(pth)

    def init_log(self):
        """Initialize project-wide logger. The logger outputs to both stdout and a file."""

        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.DEBUG

        logger = logging.getLogger()
        logger.setLevel(log_level)

        # logger console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(log_level)
        c_handler.setFormatter(log_format)
        logger.addHandler(c_handler)

        # logger file handler
        f_handler = logging.FileHandler(self.logfile)
        c_handler.setLevel(log_level)
        c_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    def initialize(self):
        """Setup model."""
        # build output directory first to store logfile and other outputs
        self.make_dir(self.c.out_dir)

        # initialize logger
        self.init_log()

        logging.info("Start time:  {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

        # log run parameters
        logging.info("Model configuration:")
        logging.info("\tconfig_file = {}".format(self.config_file))
        logging.info("\troot_dir = {}".format(self.c.root_dir))
        logging.info("\tin_dir = {}".format(self.c.in_dir))
        logging.info("\tout_dir = {}".format(self.c.out_dir))

        # prepare data for processing
        self.s = Stage(self.c)

        # set up time step generator
        self.timestep = self.run_timestep()

    def run_timestep(self):
        """Process time step"""

        for idx, step in enumerate(self.s.user_years):

            yield ProcessStep(self.c, self.s, idx, step)

    def close(self):
        """End model run and close log files"""

        logging.info("End time:  {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    def execute(self):
        """
        Execute main downscaling routine.
        """
        # set start time
        t0 = time.time()

        # set up pre time step
        self.initialize()

        # run for each time step
        for idx, step in enumerate(self.s.user_years):

            ProcessStep(self.c, self.s, idx, step)

        self.close()


if __name__ == '__main__':

    ini = '/Users/d3y010/projects/demeter/data/test_config.ini'

    # instantiate demeter
    dm = Demeter(config_file=ini, run_single_land_region={'metric_id':  142, 'region_id': 5})

    dm.execute()

    # clean up
    del dm
