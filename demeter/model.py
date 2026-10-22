"""
Class to run Demeter model for all defined steps.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import argparse
import os
import time

from demeter.config_reader import ReadConfig
from demeter.process import ProcessStep
from demeter.staging import Stage


class ValidationException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class Model:
    """Run the Demeter model.

    :param root_dir:                        Full path with filename and extension to the directory containing the
                                            directory structure.
    """

    def __init__(self, **kwargs):

        self.config = ReadConfig(kwargs)

    def initialize(self):
        """Setup model."""
        # create log header
        self.config.logger.info('START')

        # prepare data for processing
        self.s = Stage(self.config)

        # build step generator
        self.step_generator = self.build_generator()

    def build_generator(self):
        """Construct step generator."""

        # run for each time step
        for idx, step in enumerate(self.s.user_years):
            yield ProcessStep(self.config, self.s, idx, step, write_outputs=self.config.write_outputs)

    def process_step(self):
        """Process a single time step."""

        try:
            step = next(self.step_generator)

        except StopIteration:
            # all time steps have been processed
            self.cleanup()
            return None

        except Exception:
            self.cleanup()
            raise

        return step.output_df

    def cleanup(self):
        """Clean up logger and model instance."""

        self.config.logger.info('END')

        # close all open log handlers
        self.config.logger_ini.close_logger()

    def execute(self):
        """Execute main downscaling routine."""

        # set start time
        t0 = time.time()

        try:

            # set up pre time step
            self.initialize()

            # run for each time step
            for idx, step in enumerate(self.s.user_years):

                ProcessStep(self.config, self.s, idx, step, write_outputs=self.config.write_outputs)

        finally:

            self.config.logger.info('PERFORMANCE:  Model completed in {0} minutes'.format((time.time() - t0) / 60))
            self.cleanup()


def run_model(**kwargs):
    """Convenience wrapper for the 'demeter.Model' class to run all time steps sequentially."""

    # instantiate the model class
    model = Model(**kwargs)

    # run all steps then clean up
    model.execute()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Demeter.')

    parser.add_argument('--config_file', dest='config_file', action='store', type=str, default=None, help='Full path with file name and extension to the input configuration INI file')
    parser.add_argument('--run_dir', dest='run_dir', action='store', type=str, default=None, help='Full path to the directory containing the input and output directories')
    parser.add_argument('--start_year', dest='start_year', action='store', type=str, default=None, help='Year to start the downscaling')
    parser.add_argument('--end_year', dest='end_year', action='store', type=str, default=None, help='Year to process through for the downscaling')
    parser.add_argument('--target_years_output', dest='target_years_output', action='store', type=str, default=None, help="years to save data for, default is 'all'; otherwise a semicolon delimited string e.g, 2005;2050")
    parser.add_argument('--gcam_database', dest='gcam_database', action='store', type=str, default=None, help="full path to a GCAM output database")

    args = parser.parse_args()

    if args.config_file is not None and not os.path.isfile(args.config_file):
        raise ValidationException(f"Config file not found:  {args.config_file}.  Please enter a full path file name "
                                  f"with extension to config file and retry.")

    # only pass arguments the user provided so they take precedence over, but do not mask, the config file
    model_args = {k: v for k, v in vars(args).items() if v is not None}

    # instantiate and run demeter
    dm = Model(**model_args)

    dm.execute()
