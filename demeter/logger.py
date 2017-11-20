"""
Class to build logger.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL)
"""

import logging
import sys


class Logger:

    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def __init__(self, log_file, scenario, console_off=False):

        self.f = log_file
        self.scenario = scenario
        self.console_off = console_off

    @staticmethod
    def get_logger(scenario):
        """
        Instantiate logger.  Use scenario name in log message.

        :param scenario:                    user defined scenario name from config
        :return                             logger instance
        """
        return logging.getLogger(scenario)

    @classmethod
    def console_handler(cls):
        """
        Handler to log to console.

        :return:                            console handler object
        """
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # add formatter to handler
        ch.setFormatter(logging.Formatter(cls.LOG_FORMAT))

        return ch

    @classmethod
    def set_file(cls, f):
        """
        Instantiate file logger.

        :param f:                           full path and file name to log file
        """
        # instantiate log file
        logging.basicConfig(filename=f, level=logging.DEBUG, format=cls.LOG_FORMAT)

    def make_log(self):
        """
        Create console and file log instance.

        :return:                            logger object
        """
        # instantiate logger
        log = self.get_logger(self.scenario)

        if self.console_off is False:

            # create console handler
            ch = self.console_handler()

            # add console handler to logger
            log.addHandler(ch)

        # create file logging functionality
        self.set_file(self.f)

        return log

    def close_logger(self, log):
        """
        Close filehandler.
        :return:
        """
        handlers = log.handlers[:]
        for h in handlers:
            h.close()
            log.removeHandler(h)



