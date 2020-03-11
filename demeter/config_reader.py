"""
Reads config.ini and type cast parameters.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import datetime
import logging
import os
import sys

from configobj import ConfigObj


class ValidationException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)


class ReadConfig:

    def __init__(self, config_file, run_single_land_region=None):

        # check ini file
        ini_file = config_file

        # to run a single land region, None to run global all at once
        self.run_single_land_region = run_single_land_region

        # initialize console logger for model initialization
        self.log = self.console_logger()

        # get current time
        self.dt = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

        # check and validate ini file exists
        self.check_exist(ini_file, 'file', self.log)

        # instantiate config object
        self.config = ConfigObj(ini_file)

        # create and validate structure full paths
        s = self.config['STRUCTURE']
        self.root_dir = self.check_exist(s['root_dir'], 'dir', self.log)
        self.in_dir = self.check_exist(os.path.join(self.root_dir, s['in_dir']), 'dir', self.log)
        self.out_dir = self.get_outdir(s['out_dir'])

        # create and validate input dir full paths
        i = self.config['INPUTS']
        self.alloc_dir = self.check_exist(os.path.join(self.in_dir, i['allocation_dir']), 'dir', self.log)
        self.base_dir = self.check_exist(os.path.join(self.in_dir, i['observed_dir']), 'dir', self.log)
        self.projected_dir = self.check_exist(os.path.join(self.in_dir, i['projected_dir']), 'dir', self.log)
        self.ref_dir = self.check_exist(os.path.join(self.in_dir, i['ref_dir']), 'dir', self.log)
        self.constraints_dir = self.create_dir(os.path.join(self.in_dir, i['constraints_dir']), self.log)

        # create and validate allocation input file full paths
        a = i['ALLOCATION']
        self.spatial_allocation = self.check_exist(os.path.join(self.alloc_dir, a['spatial_allocation']), 'file', self.log)
        self.gcam_allocation = self.check_exist(os.path.join(self.alloc_dir, a['gcam_allocation']), 'file', self.log)
        self.treatment_order = self.check_exist(os.path.join(self.alloc_dir, a['treatment_order']), 'file', self.log)
        self.constraints = self.check_exist(os.path.join(self.alloc_dir, a['constraints']), 'file', self.log)
        self.kernel_allocation = self.check_exist(os.path.join(self.alloc_dir, a['kernel_allocation']), 'file', self.log)
        self.priority_allocation = self.check_exist(os.path.join(self.alloc_dir, a['transition_order']), 'file', self.log)

        # create and validate constraints input file full paths
        self.constraint_files = self.get_constraints()

        # create and validate base lulc input file full path
        c = i['OBSERVED']
        self.first_mod_file = self.check_exist(os.path.join(self.base_dir, c['observed_lu_data']), 'file', self.log)

        # create and validate projected lulc input file full path
        g = i['PROJECTED']
        self.lu_file = self.check_exist(os.path.join(self.projected_dir, g['projected_lu_data']), 'file', self.log)

        # create and validate reference input file full paths
        r = i['REFERENCE']
        self.gcam_regnamefile = self.check_exist(os.path.join(self.ref_dir, r['gcam_regnamefile']), 'file', self.log)
        # self.region_coords = self.check_exist(os.path.join(self.ref_dir, r['region_coords']), 'file', self.log)
        # self.country_coords = self.check_exist(os.path.join(self.ref_dir, r['country_coords']), 'file', self.log)

        # create and validate output dir full paths
        o = self.config['OUTPUTS']
        self.diag_dir = self.create_dir(os.path.join(self.out_dir, o['diag_dir']), self.log)
        self.log_dir = self.create_dir(os.path.join(self.out_dir, o['log_dir']), self.log)
        self.kernel_map_dir = self.create_dir(os.path.join(self.out_dir, o['kernel_map_dir']), self.log)
        self.transition_tabular_dir = self.create_dir(os.path.join(self.out_dir, o['transition_tabular']), self.log)
        self.transiton_map_dir = self.create_dir(os.path.join(self.out_dir  , o['transition_maps']), self.log)
        self.luc_intense_p1_dir = self.create_dir(os.path.join(self.out_dir, o['luc_intense_p1_dir']), self.log)
        self.luc_intense_p2_dir = self.create_dir(os.path.join(self.out_dir, o['luc_intense_p2_dir']), self.log)
        self.luc_expand_dir = self.create_dir(os.path.join(self.out_dir, o['luc_expand_dir']), self.log)
        self.luc_ts_luc = self.create_dir(os.path.join(self.out_dir, o['luc_timestep']), self.log)
        self.lc_per_step_csv = self.create_dir(os.path.join(self.out_dir, o['lc_per_step_csv']), self.log)
        self.lc_per_step_nc = self.create_dir(os.path.join(self.out_dir, o['lc_per_step_nc']), self.log)
        self.lc_per_step_shp = self.create_dir(os.path.join(self.out_dir, o['lc_per_step_shp']), self.log)

        # create and validate diagnostics file full paths
        d = o['DIAGNOSTICS']
        self.harm_coeff_file = os.path.join(self.diag_dir, d['harm_coeff'])
        self.intense_pass1_diag = os.path.join(self.diag_dir, d['intense_pass1_diag'])
        self.intense_pass2_diag = os.path.join(self.diag_dir, d['intense_pass2_diag'])
        self.expansion_diag = os.path.join(self.diag_dir, d['expansion_diag'])

        # assign and type run specific parameters
        p = self.config['PARAMS']
        self.model = self.ck_len(p['model'], 'model')
        self.metric = self.ck_vals(p['metric'].upper(), 'metric', ['BASIN', 'AEZ'])
        self.run_desc = self.ck_len(p['run_desc'], 'run_desc')
        self.use_constraints = self.ck_vals(int(p['use_constraints']), 'use_constraints', [0, 1])
        self.agg_level = self.ck_agg(p['agg_level'], self.log)
        self.resin = self.ck_limit(float(p['spatial_resolution']), 'spatial_resolution', [0, 1])
        self.pkey = p['observed_id_field']
        self.errortol = self.ck_limit(float(p['errortol']), 'errortol', [0, 1])
        self.year_b = self.ck_yr(p['start_year'], 'start_year')
        self.year_e = self.ck_yr(p['end_year'], 'end_year')
        self.timestep = self.ck_ts(p['timestep'], self.year_b, self.year_e)
        self.proj_factor = self.ck_type(p['proj_factor'], 'proj_factor', 'int')
        self.scenario = self.ck_len(p['scenario'], 'scenario')
        self.diagnostic = self.ck_vals(int(p['diagnostic']), 'diagnostic', [0, 1])
        self.intensification_ratio = self.ck_limit(float(p['intensification_ratio']), 'intensification_ratio', [0, 1])
        self.selection_threshold = self.ck_limit(float(p['selection_threshold']), 'selection_threshold', [0, 1])
        self.map_kernels = self.ck_limit(int(p['map_kernels']), 'map_kernels', [0, 1])
        self.map_luc = self.ck_limit(int(p['map_luc_pft']), 'map_luc_pft', [0, 1])
        self.map_luc_steps = self.ck_limit(int(p['map_luc_steps']), 'map_luc_steps', [0, 1])

        # 180 is the max longitude value
        self.kerneldistance = self.ck_limit(int(p['kernel_distance']), 'kernel_distance', [0, (180 / self.resin)])

        self.target_years_output = self.set_target(p['target_years_output'])
        self.save_tabular = self.ck_limit(int(p['save_tabular']), 'save_tabular', [0, 1])
        self.tabular_units = self.ck_vals(p['tabular_units'].lower(), 'tabular_units', ['fraction', 'sqkm'])
        self.stochastic_expansion = self.ck_vals(int(p['stochastic_expansion']), 'stochastic_expansion', [0, 1])
        self.save_transitions = self.ck_vals(int(p['save_transitions']), 'save_transitions', [0, 1])
        self.save_transition_maps = self.ck_vals(int(p['map_transitions']), 'map_transitions', [0, 1])
        self.save_shapefile = self.ck_vals(int(p['save_shapefile']), 'save_shapefile', [0, 1])
        self.save_netcdf_yr = self.ck_vals(int(p['save_netcdf_yr']), 'save_netcdf_yr', [0, 1])
        self.save_netcdf_lc = self.ck_vals(int(p['save_netcdf_lc']), 'save_netcdf_lc', [0, 1])

        try:
            self.save_ascii_max = self.ck_vals(int(p['save_ascii_max']), 'save_ascii_max', [0, 1])

        except KeyError:
            self.save_ascii_max = 0

        # turn on tabular land cover data output if writing a shapefile
        if self.save_shapefile == 1:
            self.save_tabular = 1

    @staticmethod
    def ck_type(v, p, tp):
        """
        Ensure desired type conversion can be achieved.

        :param v:           value
        :param p:           name of parameter
        :return:            value
        """
        if tp == 'int':
            try:
                return int(v)
            except ValueError:
                raise ValueError('Value "{0}" for parameter "{1}" shoud be an integer.  Exiting...'.format(v, p))
        elif tp == 'float':
            try:
                return float(v)
            except ValueError:
                raise ValueError('Value "{0}" for parameter "{1}" shoud be a decimal.  Exiting...'.format(v, p))

    @staticmethod
    def ck_ts(t, st_y, ed_y):
        """
        Make sure time step fits in year bounds.

        :param t:           time step
        :param st_y:        start year
        :param ed_y:        end year
        :return:            time step
        """
        rng = (ed_y - st_y)
        ts = int(t)

        if (rng == 0) and (ts != 1):
            raise ValidationException('Parameter "timestep" value must be 1 if only running one year.  Your start year and end year are the same in your config file.  Exiting...')
        elif (rng == 0) and (ts == 1):
            return ts

        ck = rng / ts

        if ck == 0:
            raise ValidationException('Parameter "timestep" value "{0}" is too large for start year of "{1}" and end year of "{2}".  Max time step available based on year range is "{3}".  Exiting...'.format(t, st_y, ed_y, ed_y - st_y))
        else:
            return ts

    @staticmethod
    def ck_yr(y, p):
        """
        Make sure year is four digits.

        :param y:           year
        :param p:           name of parameter
        :return:            int
        """
        if len(y) != 4:
            raise ValidationException('Year must be in four digit format (e.g., 2005) for parameter "{}". Value entered was "{}". Exiting...'.format(p, y))
        else:
            return int(y)

    @staticmethod
    def ck_len(s, p, l=30):
        """
        Ensure len of string is less than or equal to value.

        :param s:           string
        :param p:           name of parameter
        :param l:           int of max length
        :return:            string
        """
        if len(s) > l:
            raise ValidationException('Length of "{}" exceeds the max length of 20.  Please revise.  Exiting...'.format(p))
        else:
            return s

    @staticmethod
    def ck_vals(v, p, l):
        """
        Ensure target value is an available option.

        :param v:           value
        :param p:           name of parameter
        :param l:           list or tuple of available options for parameter
        :return:            value
        """
        if v in l:
            return v
        else:
            raise ValidationException('Value "{0}" not in acceptable values for parameter "{1}".  Acceptable values are:  {2}.  Exiting...'.format(v, p, l))

    @staticmethod
    def ck_limit(v, p, l):
        """
        Ensure target value falls within limits.

        :param v:           value
        :param p:           name of parameter
        :param l:           list of start and end range of acceptable values
        :return:            value
        """
        if (v >= l[0]) and (v <= l[1]):
            return v
        else:
            raise ValidationException('Value "{0}" does not fall within acceptable range of values for parameter {1} where min >= {2} and max <= {3}. Exiting...'.format(v, p, l[0], l[1]))

    @staticmethod
    def check_exist(f, kind, log):
        """
        Check file or directory existence.

        :param f        file or directory full path
        :param kind     either 'file' or 'dir'
        :return         either path or error
        """
        if kind == 'file' and os.path.isfile(f) is False:
            log.error("File not found:  {0}".format(f))
            log.error("Confirm path and retry.")
            raise IOError('File not found: {0}. Confirm path and retry.'.format(f))
        elif kind == 'dir' and os.path.isdir(f) is False:
            log.error("Directory not found:  {0}".format(f))
            log.error("Confirm path and retry.")
            raise IOError('Directory not found: {0}. Confirm path and retry.'.format(f))
        else:
            return f

    @staticmethod
    def create_dir(d, log):
        """
        Create directory.

        :param dir:     Target directory to create
        :return:        Either path or error
        """
        try:
            if os.path.isdir(d) is False:
                os.makedirs(d)
            return d
        except Exception as e:
            log.error(e)
            log.error("ERROR:  Failed to create directory.")
            raise

    @staticmethod
    def ck_agg(a, log):
        """
        Check aggregation level.  1 if by only region, 2 if by region and Basin or AEZ.
        """
        try:
            agg = int(a)
        except TypeError:
            log.error('"agg_level" parameter must be either  1 or 2.  Exiting...')
            raise

        if agg < 1 or agg > 2:
            log.error('"agg_level" parameter must be either 1 or 2.  Exiting...')
            raise ValidationException

        else:
            return agg

    def set_target(self, t):
        """
        Set target years to look for when output products.  Only the years in this list
        will be output.  If none specified, all will be used.
        """
        if t.lower().strip() == 'all':
            return range(self.year_b, self.year_e + self.timestep, self.timestep)
        else:
            return [int(i) for i in t.strip().split(';')]

    def console_logger(self):
        """
        Instantiate console logger to log any errors in config.ini file that the user
        must repair before model initialization.

        :return:  logger object
        """
        # set up logger
        log = logging.getLogger('demeter_initialization_logger')
        log.setLevel(logging.INFO)

        # set up console handler
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        cns = logging.StreamHandler(sys.stdout)
        cns.setLevel(logging.INFO)
        cns.setFormatter(fmt)
        log.addHandler(cns)

        return log

    def get_outdir(self, out):
        """
        Create output directory unique name.
        """
        # create output directory root path
        pth = os.path.join(self.root_dir, out)

        # create unique output dir name
        v = '{0}_{1}'.format(self.config['PARAMS']['scenario'], self.dt)

        # create run specific directory matching the format used to create the log file
        return self.create_dir(os.path.join(pth, v), self.log)

    def get_constraints(self):
        """
        Get a list of constraint files in dir and validate if the user has chosen to use them.

        :return:        List of full path constraint files
        """
        l = []

        # if user wants non-kernel density constraints applied...
        if int(self.config['PARAMS']['use_constraints']) == 1:

            # get list of files in constraints dir
            for i in os.listdir(self.constraints_dir):
                if os.path.splitext(i)[1] == '.csv':
                    l.append(self.check_exist(os.path.join(self.constraints_dir, i), 'file', self.log))

            return l

        else:
            return list()
