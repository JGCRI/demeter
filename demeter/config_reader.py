"""
Reads config.ini and type cast parameters.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""

import datetime
import logging
import os
import pkg_resources

from configobj import ConfigObj
from demeter.logger import Logger


class ValidationException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class ReadConfig:
    DATETIME_FORMAT = '%Y-%m-%d_%Hh%Mm%Ss'

    def __init__(self, params):

        # initilize logger
        self.logger_ini = Logger()
        self.logger = self.logger_ini.logger

        # get current time
        self.dt = datetime.datetime.now().strftime(ReadConfig.DATETIME_FORMAT)

        # configuration file
        self.config_file = params.get('config_file', None)

        if self.config_file is None:

            # these are broken out to maintain consistency with the configuration file sections
            structure_params = params
            input_params = params
            allocation_params = params
            observed_params = params
            projected_params = params
            output_params = params
            diagnostic_params = params
            run_params = params
            mapping_params = params
            ncdf_params = params

        else:

            # instantiate config object
            self.config = ConfigObj(self.check_exist(self.config_file, 'file'))

            # instantiate config file sections
            structure_params = self.config.get('STRUCTURE', None)
            input_params = self.config.get('INPUTS', None)
            allocation_params = input_params.get('ALLOCATION', None)
            observed_params = input_params.get('OBSERVED', None)
            projected_params = input_params.get('PROJECTED', None)
            output_params = self.config.get('OUTPUTS', {})
            diagnostic_params = self.config.get('OUTPUTS', {}).get('DIAGNOSTICS')
            mapping_params = input_params.get('MAPPING', None)
            ncdf_params = input_params.get('NCDF_PARAM', None)
            run_params = self.config.get('PARAMS', None)

        # choice to write log to file
        self.write_logfile = params.get('write_logfile', None)

        if self.write_logfile is None:
            self.write_logfile = run_params.get('write_logfile', True)

        # choice to write outputs
        self.write_outputs = params.get('write_outputs', None)

        if self.write_outputs is None:
            self.write_outputs = run_params.get('write_outputs', True)

        # scenario is used to build the output directory name
        self.scenario = run_params.get('scenario', 'example')

        # use the run directory provided by the user if present
        if structure_params.get('run_dir') is None:
            self.run_dir = pkg_resources.resource_filename('demeter', 'tests/data')
        else:
            self.run_dir = params.get('run_dir', structure_params.get('run_dir', '/code/example/data'))

        self.input_dir = os.path.join(self.run_dir, structure_params.get('input_dir', 'inputs'))
        self.output_dir = self.get_outdir(os.path.join(self.run_dir, structure_params.get('output_dir', 'outputs')))

        # input data directories
        self.allocation_dir = os.path.join(self.input_dir, input_params.get('allocation_dir', 'allocation'))
        self.observed_dir = os.path.join(self.input_dir, input_params.get('observed_dir', 'observed'))
        self.constraints_dir = os.path.join(self.input_dir, input_params.get('constraints_dir', 'constraints'))
        self.projected_dir = os.path.join(self.input_dir, input_params.get('projected_dir', 'projected'))
        self.mapping_dir = os.path.join(self.input_dir, input_params.get('mapping_dir', 'mapping'))

        # allocation files
        self.spatial_allocation_file = os.path.join(self.allocation_dir,
                                                    allocation_params.get('spatial_allocation_file',
                                                                          'csdms_observed_allocation.csv'))
        self.gcam_allocation_file = os.path.join(self.allocation_dir, allocation_params.get('gcam_allocation_file',
                                                                                            'csdms_projected_allocation.csv'))
        self.kernel_allocation_file = os.path.join(self.allocation_dir, allocation_params.get('kernel_allocation_file',
                                                                                              'csdms_kernel_weighting_allocation.csv'))
        self.transition_order_file = os.path.join(self.allocation_dir, allocation_params.get('transition_order_file',
                                                                                             'csdms_transition_allocation.csv'))
        self.treatment_order_file = os.path.join(self.allocation_dir, allocation_params.get('treatment_order_file',
                                                                                            'csdms_order_allocation.csv'))
        self.constraints_file = os.path.join(self.allocation_dir, allocation_params.get('constraints_file',
                                                                                        'csdms_constraint_allocation.csv'))

        # Mapping files
        self.gcam_region_names_file = os.path.join(self.mapping_dir, mapping_params.get('region_mapping_file',
                                                                                        'gcam_regions_32.csv'))
        self.gcam_basin_names_file = os.path.join(self.mapping_dir, mapping_params.get('basin_mapping_file',
                                                                                       'gcam_basin_lookup.csv'))

        # observed data
        self.observed_lu_file = os.path.join(self.observed_dir, observed_params.get('observed_lu_file',
                                                                                    'gcam_reg32_basin235_modis_v6_2010_mirca_2000_0p5deg_sqdeg_wgs84_07may2021.zip'))
        self.logger.info(f'Using `observed_lu_file`:  {self.observed_lu_file}')

        # projected data
        # look for this first in code; this only comes in from the main function - not the config file
        self.gcamwrapper_df = params.get('gcamwrapper_df', None)

        if (projected_params is not None) and (self.gcamwrapper_df is None):
            self.projected_lu_file = os.path.join(self.projected_dir, projected_params.get('projected_lu_file', None))
            self.gcam_database = projected_params.get('gcam_database', None)
            self.crop_type = self.valid_string(projected_params.get('crop_type', 'BOTH').upper(), 'crop_type',
                                               ['IRR', 'RFD', 'BOTH'])
            self.gcam_query = pkg_resources.resource_filename('demeter', 'data/query_land_reg32_basin235_gcam5p0.xml')

            if self.gcam_database is not None:
                self.gcam_database_dir = os.path.dirname(self.gcam_database)
                self.gcam_database_name = os.path.basename(self.gcam_database)

        # reference data

        # output directories
        self.diagnostics_output_dir = os.path.join(self.output_dir,
                                                   output_params.get('diagnostics_output_dir', 'diagnostics'))
        self.log_output_dir = os.path.join(self.output_dir, output_params.get('log_output_dir', 'log_files'))
        self.transitions_tabular_output_dir = os.path.join(self.output_dir,
                                                           output_params.get('transitions_tabular_output_dir',
                                                                             'transition_tabular'))
        self.transitions_maps_output_dir = os.path.join(self.output_dir,
                                                        output_params.get('transitions_maps_output_dir',
                                                                          'transition_maps'))
        self.intensification_pass1_output_dir = os.path.join(self.output_dir,
                                                             output_params.get('intensification_pass1_output_dir',
                                                                               'luc_intensification_pass1'))
        self.intensification_pass2_output_dir = os.path.join(self.output_dir,
                                                             output_params.get('intensification_pass2_output_dir',
                                                                               'luc_intensification_pass2'))
        self.extensification_output_dir = os.path.join(self.output_dir, output_params.get('extensification_output_dir',
                                                                                          'luc_extensification'))
        self.luc_timestep = os.path.join(self.output_dir, output_params.get('luc_timestep', 'luc_timestep'))
        self.lu_csv_output_dir = os.path.join(self.output_dir,
                                              output_params.get('lu_csv_output_dir', 'spatial_landcover_tabular'))
        self.lu_netcdf_output_dir = os.path.join(self.output_dir,
                                                 output_params.get('lu_netcdf_output_dir', 'spatial_landcover_netcdf'))
        self.lu_shapefile_output_dir = os.path.join(self.output_dir, output_params.get('lu_shapefile_output_dir',
                                                                                       'spatial_landcover_shapefile'))

        # diagnostics
        if diagnostic_params is not None:
            self.harmonization_coefficent_array = os.path.join(self.diagnostics_output_dir,
                                                               diagnostic_params.get('harmonization_coefficent_array',
                                                                                     'harmonization_coeff.npy'))
            self.intensification_pass1_file = os.path.join(self.diagnostics_output_dir,
                                                           diagnostic_params.get('intensification_pass1_file',
                                                                                 'intensification_pass_one_diag.csv'))
            self.intensification_pass2_file = os.path.join(self.diagnostics_output_dir,
                                                           diagnostic_params.get('intensification_pass2_file',
                                                                                 'intensification_pass_two_diag.csv'))
            self.extensification_file = os.path.join(self.diagnostics_output_dir,
                                                     diagnostic_params.get('extensification_file',
                                                                           'expansion_diag.csv'))

        # initialize file logger
        self.run_desc = run_params.get('run_desc', 'demeter_example')
        log_basename = f"logfile_{self.scenario}_{self.dt}.log"
        self.create_dir(self.log_output_dir)
        self.logfile = os.path.join(self.log_output_dir, log_basename)
        self.logger_ini.file_handler(self.logfile, self.write_logfile)

        # run parameters
        self.model = run_params.get('model', 'GCAM')
        self.metric = run_params.get('metric', 'basin').lower()
        self.agg_level = self.valid_integer(run_params.get('agg_level', 2), 'agg_level', [1, 2])
        self.observed_id_field = run_params.get('observed_id_field', 'target_fid')
        self.start_year = self.ck_yr(run_params.get('start_year', 2010), 'start_year')
        self.end_year = self.ck_yr(run_params.get('end_year', 2015), 'end_year')
        self.use_constraints = self.valid_integer(run_params.get('use_constraints', 1), 'use_constraints', [0, 1])
        self.spatial_resolution = self.valid_limit(run_params.get('spatial_resolution', 0.25), 'spatial_resolution',
                                                   [0.0, 1000000.0], 'float')
        self.regrid_resolution = self.valid_limit(run_params.get('regrid_resolution', self.spatial_resolution),
                                                  'regrid_resolution', [0.0, 1000000.0], 'float')

        self.external_scenario_PFT_name = (run_params.get('PFT_name_to_replace', 'Urban'))
        self.external_scenario = (run_params.get('ext_scenario', 'SSP1'))
        self.stitch_external = self.valid_integer(run_params.get('stitch_external', 0), 'stitch_external', [1, 0])
        self.path_to_external = (run_params.get('path_to_external', self.input_dir))

        self.errortol = self.valid_limit(run_params.get('errortol', 0.001), 'errortol', [0.0, 1000000.0], 'float')
        self.timestep = self.valid_limit(run_params.get('timestep', 5), 'timestep', [1, 1000000], 'int')
        self.proj_factor = self.valid_limit(run_params.get('proj_factor', 1000), 'proj_factor', [1, 10000000000], 'int')
        self.diagnostic = self.valid_integer(run_params.get('diagnostic', 0), 'diagnostic', [0, 1])
        self.intensification_ratio = self.valid_limit(run_params.get('intensification_ratio', 0.8),
                                                      'intensification_ratio', [0.0, 1.0], 'float')
        self.stochastic_expansion = self.valid_integer(run_params.get('stochastic_expansion', 0),
                                                       'stochastic_expansion', [0, 1])
        self.selection_threshold = self.valid_limit(run_params.get('selection_threshold', 0.75),
                                                    'intensification_ratio', [0.0, 1.0], 'float')
        self.kernel_distance = self.valid_limit(run_params.get('kernel_distance', 10), 'kernel_distance',
                                                [0, 10000000000], 'int')
        self.target_years_output = self.set_target(run_params.get('target_years_output', 'all'))
        self.save_tabular = self.valid_integer(run_params.get('save_tabular', 0), 'save_tabular', [0, 1])
        # self.map_transitions=self.valid_integer(run_params.get('map_transitions', 0), 'map_transitions', [0, 1])
        self.tabular_units = self.valid_string(run_params.get('tabular_units', 'sqkm'), 'tabular_units',
                                               ['sqkm', 'fraction'])
        self.save_transitions = self.valid_integer(run_params.get('save_transitions', 0), 'save_transitions', [0, 1])
        self.save_shapefile = self.valid_integer(run_params.get('save_shapefile', 0), 'save_shapefile', [0, 1])
        self.save_netcdf_yr = self.valid_integer(run_params.get('save_netcdf_yr', 1), 'save_netcdf_yr', [0, 1])

        # create and validate constraints input file full paths
        self.constraint_files = self.get_constraints()

        self.logger.info(f'Using `run_dir`:  {self.run_dir}')

        # turn on tabular land cover data output if writing a shapefile
        if self.diagnostic:
            self.create_dir(self.diagnostics_output_dir)

        if self.save_tabular or self.save_shapefile:
            self.create_dir(self.lu_csv_output_dir)

        if self.save_netcdf_yr:
            self.create_dir(self.lu_netcdf_output_dir)

        if self.save_transitions:
            self.create_dir(self.transitions_tabular_output_dir)

        if self.save_shapefile:
            self.create_dir(self.lu_shapefile_output_dir)

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
                raise ValueError('Value "{0}" for parameter "{1}" should be an integer.  Exiting...'.format(v, p))
        elif tp == 'float':
            try:
                return float(v)
            except ValueError:
                raise ValueError('Value "{0}" for parameter "{1}" should be a decimal.  Exiting...'.format(v, p))

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
            raise ValidationException(
                'Parameter "timestep" value must be 1 if only running one year.  Your start year and end year are the same in your config file.  Exiting...')
        elif (rng == 0) and (ts == 1):
            return ts

        ck = rng / ts

        if ck == 0:
            raise ValidationException(
                'Parameter "timestep" value "{0}" is too large for start year of "{1}" and end year of "{2}".  Max time step available based on year range is "{3}".  Exiting...'.format(
                    t, st_y, ed_y, ed_y - st_y))
        else:
            return ts

    @staticmethod
    def ck_yr(y, p, lower_year=500, upper_year=3000):
        """Make sure year is four digits.

        :param y:           year
        :param p:           name of parameter

        :return:            int

        """

        if type(y) == int:

            if y < lower_year:
                raise ValidationException("'{}' must be >= {}".format(p, lower_year))

            if y > upper_year:
                raise ValidationException("'{}' must be <= {}".format(p, lower_year))
        else:
            try:
                y = int(y)

                if y < lower_year:
                    raise ValidationException("'{}' must be >= {}".format(p, lower_year))

                if y > upper_year:
                    raise ValidationException("'{}' must be <= {}".format(p, lower_year))

            except ValueError:
                raise ValidationException(
                    'Year must be in four digit format (e.g., 2005) for parameter "{}". Value entered was "{}". Exiting...'.format(
                        p, y))

        return y

    @staticmethod
    def vaild_length(value, parameter, max_characters=30):
        """Ensure len of string is less than or equal to value.

        :param value:                           string
        :param parameter:                       parameter name
        :param max_characters:                  int of max length

        :return:                                string

        """

        if len(value) > max_characters:
            raise ValidationException(
                'Length of "{}" exceeds the max length of {}.  Please revise.  Exiting...'.format(parameter,
                                                                                                  max_characters))

    @staticmethod
    def valid_string(v, p, l):
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
            raise ValidationException(
                'Value "{0}" not in acceptable values for parameter "{1}".  Acceptable values are:  {2}.  Exiting...'.format(
                    v, p, l))

    def valid_integer(self, v, p, l):
        """
        Ensure target value is an available option.

        :param v:           value
        :param p:           name of parameter
        :param l:           list or tuple of available options for parameter

        :return:            value

        """
        value = self.ck_type(v, p, 'int')

        if value in l:
            return value
        else:
            raise ValidationException(
                'Value "{0}" not in acceptable values for parameter "{1}".  Acceptable values are:  {2}.  Exiting...'.format(
                    v, p, l))

    def valid_limit(self, v, p, l, typ):
        """Ensure target value falls within limits.

        :param v:           value
        :param p:           name of parameter
        :param l:           list of start and end range of acceptable values
        :param typ:         intended type

        :return:            value

        """
        value = self.ck_type(v, p, typ)

        if (value >= l[0]) and (value <= l[1]):
            return value
        else:
            raise ValidationException(
                'Value "{0}" does not fall within acceptable range of values for parameter {1} where min >= {2} and max <= {3}. Exiting...'.format(
                    value, p, l[0], l[1]))

    @staticmethod
    def check_exist(f, kind):
        """Check file or directory existence.

        :param f        file or directory full path
        :param kind     either 'file' or 'dir'

        :return         either path or error

        """
        if kind == 'file' and os.path.isfile(f) is False:
            logging.error("File not found:  {0}".format(f))
            logging.error("Confirm path and retry.")
            raise IOError('File not found: {0}. Confirm path and retry.'.format(f))

        elif kind == 'dir' and os.path.isdir(f) is False:
            logging.error("Directory not found:  {0}".format(f))
            logging.error("Confirm path and retry.")
            raise IOError('Directory not found: {0}. Confirm path and retry.'.format(f))

        else:
            return f

    def create_dir(self, d):
        """Create directory.

        :param d:     Target directory to create

        :return:        Either path or error

        """

        if self.write_outputs:
            try:
                if os.path.isdir(d) is False:
                    os.makedirs(d)

            except:
                logging.error("ERROR:  Failed to create directory.")
                raise

    @staticmethod
    def ck_agg(a):
        """Check aggregation level.  1 if by only region, 2 if by region and Basin or AEZ."""
        try:
            agg = int(a)
        except TypeError:
            logging.error('"agg_level" parameter must be either  1 or 2.  Exiting...')
            raise

        if agg < 1 or agg > 2:
            logging.error('"agg_level" parameter must be either 1 or 2.  Exiting...')
            raise ValidationException

        else:
            return agg

    def set_target(self, t):
        """Set target years to look for when output products.  Only the years in this list
        will be output.  If none specified, all will be used.

        """
        yr = str(t)

        if yr.lower().strip() == 'all':
            return range(self.start_year, self.end_year + self.timestep, self.timestep)
        else:
            return [int(i) for i in yr.strip().split(';')]

    def get_outdir(self, out):
        """Create output directory unique name."""

        # create output directory root path
        pth = os.path.join(self.run_dir, out)

        # create unique output dir name
        v = '{0}_{1}'.format(self.scenario, self.dt)

        # create run specific directory matching the format used to create the log file
        out_dir = os.path.join(pth, v)
        self.create_dir(out_dir)

        return out_dir

    def get_constraints(self):
        """Get a list of constraint files in dir and validate if the user has chosen to use them.

        :return:                        List of full path constraint files

        """

        l = []

        # if user wants non-kernel density constraints applied...
        if int(self.use_constraints) == 1:

            # get list of files in constraints dir
            for i in os.listdir(self.constraints_dir):
                if os.path.splitext(i)[1].lower() == '.csv':
                    l.append(self.check_exist(os.path.join(self.constraints_dir, i), 'file'))

            return l

        else:
            return list()
