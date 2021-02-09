"""
Class to run Demeter model for all defined steps.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import argparse
import os
import sys
import time
import traceback

from demeter.config_reader import ReadConfig
from demeter.logger import Logger
from demeter.process import ProcessStep
from demeter.staging import Stage


class ValidationException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class Demeter:
    """Run the Demeter model.

    :param root_dir:                        Full path with filename and extension to the directory containing the
                                            directory structure.
    """

    def __init__(self, **kwargs):

        self.params = kwargs
        self.c = None
        self.s = None
        self.process_step = None
        self.rg = None
        self.f = None

    @staticmethod
    def log_config(c, log):
        """Log validated configuration options."""

        for i in dir(c):

            # create configuration object from string
            x = eval('c.{0}'.format(i))

            # ignore magic objects
            if type(x) == str and i[:2] != '__':

                # log result
                log.debug('CONFIG: [PARAMETER] {0} -- [VALUE] {1}'.format(i, x))

    def make_logfile(self):
        """Make log file.

        :return                               log file object

        """
        # create logfile path and name
        self.f = os.path.join(self.c.run_dir, '{0}/logfile_{1}_{2}.log'.format(self.c.log_output_dir, self.c.scenario, self.c.dt))

        # parameterize logger
        self.log = Logger(self.f, self.c.scenario).make_log()

    def setup(self):
        """Setup model."""
        # instantiate config
        self.c = ReadConfig(self.params)

        # instantiate log file
        self.make_logfile()

        # create log header
        self.log.info('START')

        # log validated configuration
        self.log_config(self.c, self.log)

        # prepare data for processing
        self.s = Stage(self.c, self.log)

    def execute(self):
        """Execute main downscaling routine."""

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

            # close all open log handlers
            Logger(self.f, self.c.scenario).close_logger(self.log)

        finally:

            self.log.info('PERFORMANCE:  Model completed in {0} minutes'.format((time.time() - t0) / 60))
            self.log.info('END')
            self.log = None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Demeter.')

    parser.add_argument('--config_file', dest='config_file', action='store', type=str, default=None, help='Full path with file name and extension to the input configuration INI file')
    parser.add_argument('--run_dir', dest='run_dir', action='store', type=str, default='/data/demeter_data', help='Full path to the directory containing the input and output directories')
    parser.add_argument('--start_year', dest='start_year', action='store', type=str, default=2005, help='Year to start the downscaling')
    parser.add_argument('--end_year', dest='end_year', action='store', type=str, default=2010, help='Year to process through for the downscaling')
    parser.add_argument('--target_years_output', dest='target_years_output', action='store', type=str, default='all', help="years to save data for, default is 'all'; otherwise a semicolon delimited string e.g, 2005;2050")
    parser.add_argument('--gcam_database', dest='gcam_database', action='store', type=str, default='/data/gcam_output_data/database_basexdb', help="full path to a GCAM output database")

    args = parser.parse_args()

    if os.path.isfile is False:
        print('ERROR:  Config file not found.')
        print('You entered:  {0}'.format(args.config_file))
        print('Please enter a full path file name with extension to config file and retry.')
        raise ValidationException

    # instantiate and run demeter
    dm = Demeter(config_file=args.config_file,
                 run_dir=args.run_dir,
                 start_year=args.start_year,
                 end_year=args.end_year,
                 target_years_output=args.target_years_output,
                 gcam_database=args.gcam_database)

    dm.execute()
