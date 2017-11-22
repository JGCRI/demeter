"""
Class to run Demeter model for all defined steps.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""
import datetime
from joblib import Parallel, delayed
import os.path as op
import sys
import time
import traceback

from demeter.config_reader import ReadConfig, ReadConfigShuffle, ReadConfigInitial
from demeter.logger import Logger
from demeter.process import ProcessStep
from demeter.staging import Stage
from demeter.ensemble.ensemble import RandomConfig
from demeter.weight.kernel_density import KernelDensity


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
        self.f = None

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
        self.f = op.join(self.dir, '{0}/logfile_{1}_{2}.log'.format(self.c.log_dir, self.c.scenario, self.c.dt))

        # parameterize logger
        self.log = Logger(self.f, self.c.scenario).make_log()

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
            try:
                self.log.error(e)
                self.log.error(t)

                # close all open log handlers
                Logger(self.f, self.c.scenario).close_logger(self.log)

            except AttributeError:
                print(e)
                print(t)

        finally:

            try:
                self.log.info('PERFORMANCE:  Model completed in {0} minutes'.format((time.time() - t0) / 60))
                self.log.info('END')
                self.log = None

            except AttributeError:
                pass

    def ensemble(self):
        """
        Execute random runs in parallel.

        @:param jobs    number of cores used (-1 is all cores, -2 all but one)
        """

        # instantiate config
        c = ReadConfigInitial(self.ini)

        # alter initial setup with mutated parameters
        rc = RandomConfig(c)

        # run it on all available cores but one, verbose at 10 is a job report after each iteration
        Parallel(n_jobs=c.n_jobs, verbose=10)(delayed(_shuffle)(self.dir, i, c) for i in rc.mix)


def _make_logfile(f, scenario, console_off=True):
    """
    Make log file.

    :return                               log file object
    """
    # parameterize logger
    return Logger(f, scenario, console_off=console_off).make_log()


def _log_config(c, log):
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


def _get_outdir(pth, scenario, suffix):
    """
    Create output directory unique name.
    """
    # get date time string
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

    # create unique output dir name
    v = '{0}_{1}_{2}'.format(scenario, dt, suffix)

    # create run specific directory matching the format used to create the log file
    return op.join(pth, v)


def _shuffle(dir, i, oc):
    """
    Build function to pass to joblib.
    """
    log = None

    # set start time
    t0 = time.time()

    onumstr = ''

    try:

        # unpack
        priority_alloc = i[0]
        treatment_order = i[1]
        intensification_ratio = i[2]
        selection_threshold = i[3]
        kernel_distance = i[4]
        scenario_suffix = i[5]

        # read in config file
        out_dir = _get_outdir(oc.out_dir, oc.scenario, scenario_suffix)
        c = ReadConfigShuffle(oc.ini_file, new_out_dir=out_dir)

        # build log file name with scenario suffix
        log_file = op.join(dir, '{0}/logfile_{1}_{2}_{3}.log'.format(c.log_dir, c.scenario, c.dt, scenario_suffix))

        # instantiate log file
        log = _make_logfile(log_file, c.scenario)

        # create log header
        log.info('START')

        # substitute parameters into loaded config
        c.intensification_ratio = intensification_ratio
        c.selection_threshold = selection_threshold
        c.kerneldistance = kernel_distance

        # log altered config file
        _log_config(c, log)

        # # prepare data for processing
        s = Stage(c, log)

        # substitute parameters into staged data
        s.transition_rules = priority_alloc
        s.order_rules = treatment_order

        # log changes in config
        log.info('New priority allocation: {0}'.format(s.transition_rules))
        log.info('New treatment order: {0}'.format(s.order_rules))
        log.info('New intensification ratio: {0}'.format(c.intensification_ratio))
        log.info('New selection_threshold: {0}'.format(c.selection_threshold))
        log.info('New kerneldistance: {0}'.format(c.kerneldistance))

        log.info("Creating and processing kernel density...")
        # set start time for kernel density
        tz0 = time.time()

        # create kd filter
        s.kd = KernelDensity(c.resin, s.spat_coords, s.final_landclasses,
                                  c.kerneldistance, s.ngrids, c.kernel_map_dir,
                                  s.order_rules,
                                  c.map_kernels)

        # preprocess kernel density data
        s.lat, s.lon, s.cellindexresin, s.pft_maps, s.kernel_maps, s.kernel_vector, \
        s.weights = s.kd.preprocess_kernel_density()

        # log processing time for kernel density
        log.info('PERFORMANCE:  Kernel density filter prepared in {0} seconds'.format(time.time() - tz0))

        # run for each time step
        for idx, step in enumerate(s.user_years):
            ProcessStep(c, log, s, idx, step)

    except:

        # catch all exceptions and their traceback
        e = sys.exc_info()[0]
        t = traceback.format_exc()

        # log exception and traceback as error
        try:
            log.error(e)
            log.error(t)

            # close all open log handlers
            Logger(log_file, c.scenario).close_logger(log)

        except AttributeError:
            pass

    finally:

        try:
            log.info('PERFORMANCE:  Model completed in {0} minutes'.format((time.time() - t0) / 60))
            log.info('END')
            log = None

        except AttributeError:
            pass


if __name__ == '__main__':

    # terminal option for running without installing demeter
    args = sys.argv[1:]

    if len(args) != 2:
        print("""USAGE:  Two arguments should be passed.
                        1) Full path file name with extension for config file and
                        2) the type of run you wish to conduct "standard" or "ensemble"
                    """)
        print('Exiting...')
        sys.exit(1)

    # explode args
    ini = args[0]

    # run mode
    mode = args[1]

    if op.isfile is False:
        print('ERROR:  Config file not found.')
        print('You entered:  {0}'.format(ini))
        print('Please enter a full path file name with extension to config file and retry.')
        sys.exit(1)

    if mode.lower() not in ('standard', 'ensemble'):
        print("""ERROR:  Run mode passed '{0}' not a valid option.  Either select 'standard' or 'ensemble' """.format(mode))
        print('Exiting...')
        sys.exit(1)

    # instantiate demeter
    dm = Demeter(config=ini)

    # run user specified call
    if mode == 'ensemble':
        dm.ensemble()
    else:
        dm.execute()

    # clean up
    del dm
