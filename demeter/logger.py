"""Logger

@author   Chris R. Vernon
@email:   chris.vernon@pnnl.gov

License:  BSD 2-Clause, see LICENSE and DISCLAIMER files

"""

import logging
import sys


class Logger:
    """Initialize project-wide logger. The logger outputs to both stdout and a file."""

    # output format for log string
    LOG_FORMAT_STRING = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def __init__(self):

        # initialize logger instance
        self.logger = logging.getLogger('demeter_runtime')
        self.logger.setLevel(logging.INFO)

        # generate log formatter
        self.log_format = logging.Formatter(self.LOG_FORMAT_STRING)

        # logger console handler
        self.console_handler()

    def console_handler(self):
        """Construct console handler."""

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

    def file_handler(self, logfile, write_logfile):
        """Construct file handler."""

        # logger file handler
        if write_logfile:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(self.log_format)
            self.logger.addHandler(file_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()
