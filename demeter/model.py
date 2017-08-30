"""
Class to run Demeter model for all defined steps.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import os.path as op
import sys
import time
import traceback

from demeter.config_reader import ReadConfig
from demeter.logger import Logger
from demeter.process import ProcessStep
from demeter.staging import Stage


class Demeter(Logger):

    def __init__(self,
                 root_dir=op.dirname(op.realpath(__file__)),
                 config=op.join(op.dirname(op.dirname(op.realpath(__file__))), 'config.ini')):

        self.dir = root_dir
        self.ini = config
        self.c = None
        self.s = None
        self.process_step = None
        self.rg = None

    @staticmethod
    def log_config(c, log):
        """
        Log validated configuration options.
        """
        for i in dir(c):

            # create configuration object from string
            x = eval('c.{0}'.format(i))

            # ignore magic objects
            if type(x) == str and i[:2] != '__':

                # log result
                log.debug('CONFIG: [PARAMETER] {0} -- [VALUE] {1}'.format(i, x))

    def make_logfile(self):
        """
        Make log file.

        :return                               log file object
        """
        # create logfile path and name
        f = op.join(self.dir, '{0}/logfile_{1}_{2}.log'.format(self.c.log_dir, self.c.scenario, self.c.dt))

        # parameterize logger
        self.log = Logger(f, self.c.scenario).make_log()

    def setup(self):
        """
        Setup model.
        """
        # instantiate config
        self.c = ReadConfig(self.ini)

        # instantiate log file
        self.make_logfile()

        # create log header
        self.log.info('START')

        # log validated configuration
        self.log_config(self.c, self.log)

        # prepare data for processing
        self.s = Stage(self.c, self.log)

    def execute(self):
        """
        Execute main downscaling routine.
        """
        # set start time
        t0 = time.time()

        try:

            # set up pre time step
            self.setup()

            # run for each time step
            for idx, step in enumerate(self.s.user_years):

                ProcessStep(self.c, self.log, self.s, idx, step)

        except:

            # catch all exceptions and their traceback
            e = sys.exc_info()[0]
            t = traceback.format_exc()

            # log exception and traceback as error
            self.log.error(e)
            self.log.error(t)

        finally:

            self.log.info('PERFORMANCE:  Model completed in {0} minutes'.format((time.time() - t0) / 60))
            self.log.info('END')
            self.log = None