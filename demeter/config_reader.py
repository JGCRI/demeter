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
import pkg_resources

from configobj import ConfigObj


class ValidationException(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class ReadConfig:

    DATETIME_FORMAT = '%Y-%m-%d_%Hh%Mm%Ss'

    def __init__(self, params):

        # initialize console logger for model initialization
        self.log = self.console_logger()

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
            reference_params = params
            output_params = params
            diagnostic_params = params
            run_params = params

        else:

            # instantiate config object
            self.config = ConfigObj(self.check_exist(self.config_file, 'file', self.log))

            # instantiate config file sections
            structure_params = self.config.get('STRUCTURE', None)
            input_params = self.config.get('INPUTS', None)
            allocation_params = input_params.get('ALLOCATION', None)
            observed_params = input_params.get('OBSERVED', None)
            projected_params = input_params.get('PROJECTED', None)
            reference_params = input_params.get('REFERENCE', None)
            output_params = self.config.get('OUTPUTS', None)
            diagnostic_params = output_params.get('DIAGNOSTICS', None)
            run_params = self.config.get('PARAMS', None)

        # scenario is used to build the output directory name
        self.scenario = run_params.get('scenario', 'example')

        # use the run directory provided by the user if present
        if structure_params.get('run_dir') is None:
            self.run_dir = pkg_resources.resource_filename('demeter', 'tests/data')
        else:
            self.run_dir = params.get('run_dir', structure_params.get('run_dir', '/code/example/data'))

        self.log.info(f'Using `run_dir`:  {self.run_dir}')

        self.input_dir = os.path.join(self.run_dir, structure_params.get('input_dir', 'inputs'))
        self.output_dir = self.get_outdir(os.path.join(self.run_dir, structure_params.get('output_dir', 'outputs')))

        # input data directories
        self.allocation_dir = os.path.join(self.input_dir, input_params.get('allocation_dir', 'allocation'))
        self.observed_dir = os.path.join(self.input_dir, input_params.get('observed_dir', 'observed'))
        self.constraints_dir = os.path.join(self.input_dir, input_params.get('constraints_dir', 'constraints'))
        self.projected_dir = os.path.join(self.input_dir, input_params.get('projected_dir', 'projected'))
        self.reference_dir = os.path.join(self.input_dir, input_params.get('reference_dir', 'reference'))

        # allocation files
        self.spatial_allocation_file = os.path.join(self.allocation_dir, allocation_params.get('spatial_allocation_file', 'gcam_regbasin_modis_v6_type5_5arcmin_observed_alloc.csv'))
        self.gcam_allocation_file = os.path.join(self.allocation_dir, allocation_params.get('gcam_allocation_file', 'gcam_regbasin_modis_v6_type5_5arcmin_projected_alloc.csv'))
        self.kernel_allocation_file = os.path.join(self.allocation_dir, allocation_params.get('kernel_allocation_file', 'kernel_density_weighting.csv'))
        self.transition_order_file = os.path.join(self.allocation_dir, allocation_params.get('transition_order_file', 'transition_priority.csv'))
        self.treatment_order_file = os.path.join(self.allocation_dir, allocation_params.get('treatment_order_file', 'treatment_order.csv'))
        self.constraints_file = os.path.join(self.allocation_dir, allocation_params.get('constraints_file', 'constraint_weighting.csv'))

        # observed data
        self.observed_lu_file = os.path.join(self.observed_dir, observed_params.get('observed_lu_file', 'gcam_reg32_basin235_modis_v6_2010_5arcmin_sqdeg_wgs84_11Jul2019.zip'))

        # projected data
        self.projected_lu_file = os.path.join(self.projected_dir, projected_params.get('projected_lu_file', 'gcam_ref_scenario_reg32_basin235_v5p1p3.csv'))
        self.gcam_database = projected_params.get('gcam_database', None)
        self.crop_type = self.valid_string(projected_params.get('crop_type', 'BOTH').upper(), 'crop_type', ['IRR', 'RFD', 'BOTH'])

        if self.gcam_database is not None:
            self.gcam_database_dir = os.path.dirname(self.gcam_database)
            self.gcam_database_name = os.path.basename(self.gcam_database)

        # reference data
        self.gcam_region_names_file = os.path.join(self.reference_dir, reference_params.get('gcam_region_names_file', 'gcam_regions_32.csv'))
        self.gcam_region_coords_file = os.path.join(self.reference_dir, reference_params.get('gcam_region_coords_file', 'regioncoord.csv'))
        self.gcam_country_coords_file = os.path.join(self.reference_dir, reference_params.get('gcam_country_coords_file', 'countrycoord.csv'))
        self.gcam_basin_names_file = os.path.join(self.reference_dir, reference_params.get('gcam_basin_names_file', 'gcam_basin_lookup.csv'))
        self.gcam_query = os.path.join(self.reference_dir, projected_params.get('gcam_query', 'query_land_reg32_basin235_gcam5p0.xml'))

        # outputs directories
        self.diagnostics_output_dir = os.path.join(self.output_dir, output_params.get('diagnostics_output_dir', 'diagnostics'))
        self.log_output_dir = os.path.join(self.output_dir, output_params.get('log_output_dir', 'log_files'))
        self.kernel_maps_output_dir = os.path.join(self.output_dir, output_params.get('kernel_maps_output_dir', 'kernel_density'))
        self.transitions_tabular_output_dir = os.path.join(self.output_dir, output_params.get('transitions_tabular_output_dir', 'transition_tabular'))
        self.transitions_maps_output_dir = os.path.join(self.output_dir, output_params.get('transitions_maps_output_dir', 'transition_maps'))
        self.intensification_pass1_output_dir = os.path.join(self.output_dir, output_params.get('intensification_pass1_output_dir', 'luc_intensification_pass1'))
        self.intensification_pass2_output_dir = os.path.join(self.output_dir, output_params.get('intensification_pass2_output_dir', 'luc_intensification_pass2'))
        self.extensification_output_dir = os.path.join(self.output_dir, output_params.get('extensification_output_dir', 'luc_extensification'))
        self.luc_timestep = os.path.join(self.output_dir, output_params.get('luc_timestep', 'luc_timestep'))
        self.lu_csv_output_dir = os.path.join(self.output_dir, output_params.get('lu_csv_output_dir', 'spatial_landcover_tabular'))
        self.lu_netcdf_output_dir = os.path.join(self.output_dir, output_params.get('lu_netcdf_output_dir', 'spatial_landcover_netcdf'))
        self.lu_shapefile_output_dir = os.path.join(self.output_dir, output_params.get('lu_shapefile_output_dir', 'spatial_landcover_shapefile'))

        # diagnostics
        self.harmonization_coefficent_array = os.path.join(self.diagnostics_output_dir, diagnostic_params.get('harmonization_coefficent_array', 'harmonization_coeff.npy'))
        self.intensification_pass1_file = os.path.join(self.diagnostics_output_dir, diagnostic_params.get('intensification_pass1_file', 'intensification_pass_one_diag.csv'))
        self.intensification_pass2_file = os.path.join(self.diagnostics_output_dir, diagnostic_params.get('intensification_pass2_file', 'intensification_pass_two_diag.csv'))
        self.extensification_file = os.path.join(self.diagnostics_output_dir, diagnostic_params.get('extensification_file', 'expansion_diag.csv'))

        # run parameters
        self.model = run_params.get('model', 'GCAM')
        self.metric = run_params.get('metric', 'BASIN')
        self.run_desc = run_params.get('run_desc', 'demeter_example')
        self.agg_level = self.valid_integer(run_params.get('agg_level', 2), 'agg_level', [1, 2])
        self.observed_id_field = run_params.get('observed_id_field', 'target_fid')
        self.start_year = self.ck_yr(run_params.get('start_year', 2010), 'start_year')
        self.end_year = self.ck_yr(run_params.get('end_year', 2020), 'end_year')
        self.use_constraints = self.valid_integer(run_params.get('use_constraints', 1), 'use_constraints', [0, 1])
        self.spatial_resolution = self.valid_limit(run_params.get('spatial_resolution', 0.25), 'spatial_resolution', [0.0, 1000000.0], 'float')
        self.errortol = self.valid_limit(run_params.get('errortol', 0.001), 'errortol', [0.0, 1000000.0], 'float')
        self.timestep = self.valid_limit(run_params.get('timestep', 1), 'timestep', [1, 1000000], 'int')
        self.proj_factor = self.valid_limit(run_params.get('proj_factor', 1000), 'proj_factor', [1, 10000000000], 'int')
        self.diagnostic = self.valid_integer(run_params.get('diagnostic', 0), 'diagnostic', [0, 1])
        self.intensification_ratio = self.valid_limit(run_params.get('intensification_ratio', 0.8), 'intensification_ratio', [0.0, 1.0], 'float')
        self.stochastic_expansion = self.valid_integer(run_params.get('stochastic_expansion', 0), 'stochastic_expansion', [0, 1])
        self.selection_threshold = self.valid_limit(run_params.get('selection_threshold', 0.75), 'intensification_ratio', [0.0, 1.0], 'float')
        self.kernel_distance = self.valid_limit(run_params.get('kernel_distance', 10), 'kernel_distance', [0, 10000000000], 'int')
        self.map_kernels = self.valid_integer(run_params.get('map_kernels', 0), 'map_kernels', [0, 1])
        self.map_luc_pft = self.valid_integer(run_params.get('map_luc_pft', 0), 'map_luc_pft', [0, 1])
        self.map_luc_steps = self.valid_integer(run_params.get('map_luc_steps', 0), 'map_luc_steps', [0, 1])
        self.map_transitions = self.valid_integer(run_params.get('map_transitions', 0), 'map_transitions', [0, 1])
        self.target_years_output = self.set_target(run_params.get('target_years_output', 'all'))
        self.save_tabular = self.valid_integer(run_params.get('save_tabular', 1), 'save_tabular', [0, 1])
        self.tabular_units = self.valid_string(run_params.get('tabular_units', 'sqkm'), 'tabular_units', ['sqkm', 'fraction'])
        self.save_transitions = self.valid_integer(run_params.get('save_transitions', 0), 'save_transitions', [0, 1])
        self.save_shapefile = self.valid_integer(run_params.get('save_shapefile', 0), 'save_shapefile', [0, 1])
        self.save_netcdf_yr = self.valid_integer(run_params.get('save_netcdf_yr', 0), 'save_netcdf_yr', [0, 1])
        self.save_netcdf_lc = self.valid_integer(run_params.get('save_netcdf_lc', 0), 'save_netcdf_lc', [0, 1])
        self.save_ascii_max = self.valid_integer(run_params.get('save_ascii_max', 0), 'save_ascii_max', [0, 1])

        # create and validate constraints input file full paths
        self.constraint_files = self.get_constraints()

        # turn on tabular land cover data output if writing a shapefile
        if self.save_shapefile == 1:
            self.save_tabular = 1

        # create needed output directories
        self.create_dir(self.log_output_dir)

        if self.diagnostic:
            self.create_dir(self.diagnostics_output_dir)

        if self.map_kernels:
            self.create_dir(self.kernel_maps_output_dir)

        if self.map_luc_pft:
            self.create_dir(self.map_luc_pft)

        if self.map_transitions:
            self.create_dir(self.transitions_maps_output_dir)

        if self.save_tabular or self.save_shapefile:
            self.create_dir(self.lu_csv_output_dir)

        if self.save_transitions:
            self.create_dir(self.transitions_tabular_output_dir)

        if self.save_shapefile:
            self.create_dir(self.lu_shapefile_output_dir)

        if self.save_netcdf_yr or self.save_netcdf_lc:
            self.create_dir(self.lu_netcdf_output_dir)

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
            raise ValidationException('Parameter "timestep" value must be 1 if only running one year.  Your start year and end year are the same in your config file.  Exiting...')
        elif (rng == 0) and (ts == 1):
            return ts

        ck = rng / ts

        if ck == 0:
            raise ValidationException('Parameter "timestep" value "{0}" is too large for start year of "{1}" and end year of "{2}".  Max time step available based on year range is "{3}".  Exiting...'.format(t, st_y, ed_y, ed_y - st_y))
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
                raise ValidationException('Year must be in four digit format (e.g., 2005) for parameter "{}". Value entered was "{}". Exiting...'.format(p, y))

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
            raise ValidationException('Length of "{}" exceeds the max length of {}.  Please revise.  Exiting...'.format(parameter, max_characters))

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
            raise ValidationException('Value "{0}" not in acceptable values for parameter "{1}".  Acceptable values are:  {2}.  Exiting...'.format(v, p, l))

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
            raise ValidationException('Value "{0}" not in acceptable values for parameter "{1}".  Acceptable values are:  {2}.  Exiting...'.format(v, p, l))

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
            raise ValidationException('Value "{0}" does not fall within acceptable range of values for parameter {1} where min >= {2} and max <= {3}. Exiting...'.format(value, p, l[0], l[1]))

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

    def create_dir(self, d):
        """Create directory.

        :param d:     Target directory to create

        :return:        Either path or error

        """

        try:
            if os.path.isdir(d) is False:
                os.makedirs(d)

        except:
            self.log.error("ERROR:  Failed to create directory.")
            raise

    @staticmethod
    def ck_agg(a, log):
        """Check aggregation level.  1 if by only region, 2 if by region and Basin or AEZ."""
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
        """Set target years to look for when output products.  Only the years in this list
        will be output.  If none specified, all will be used.

        """
        yr = str(t)

        if yr.lower().strip() == 'all':
            return range(self.start_year, self.end_year + self.timestep, self.timestep)
        else:
            return [int(i) for i in yr.strip().split(';')]

    @staticmethod
    def console_logger():
        """Instantiate console logger to log any errors in config.ini file that the user
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
                    l.append(self.check_exist(os.path.join(self.constraints_dir, i), 'file', self.log))

            return l

        else:
            return list()
