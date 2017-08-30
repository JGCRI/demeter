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


class ReadConfig:

    def __init__(self, ini_file):

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
        self.base_dir = self.check_exist(os.path.join(self.in_dir, i['base_dir']), 'dir', self.log)
        self.constraints_dir = self.check_exist(os.path.join(self.in_dir, i['constraints_dir']), 'dir', self.log)
        self.projected_dir = self.check_exist(os.path.join(self.in_dir, i['projected_dir']), 'dir', self.log)
        self.ref_dir = self.check_exist(os.path.join(self.in_dir, i['ref_dir']), 'dir', self.log)

        # create and validate allocation input file full paths
        a = i['ALLOCATION']
        self.spatial_allocation = self.check_exist(os.path.join(self.alloc_dir, a['spatial_allocation']), 'file', self.log)
        self.gcam_allocation = self.check_exist(os.path.join(self.alloc_dir, a['gcam_allocation']), 'file', self.log)
        self.kernel_allocation = self.check_exist(os.path.join(self.alloc_dir, a['kernel_allocation']), 'file', self.log)
        self.priority_allocation = self.check_exist(os.path.join(self.alloc_dir, a['priority_allocation']), 'file', self.log)
        self.treatment_order = self.check_exist(os.path.join(self.alloc_dir, a['treatment_order']), 'file', self.log)
        self.constraints = self.check_exist(os.path.join(self.alloc_dir, a['constraints']), 'file', self.log)

        # create and validate constraints input file full paths
        self.constraint_files = self.get_constraints()

        # create and validate base lulc input file full path
        c = i['BASE']
        self.first_mod_file = self.check_exist(os.path.join(self.base_dir, c['base_lu_data']), 'file', self.log)

        # create and validate projected lulc input file full path
        g = i['PROJECTED']
        self.lu_file = self.check_exist(os.path.join(self.projected_dir, g['projected_lu_data']), 'file', self.log)

        # create and validate reference input file full paths
        r = i['REFERENCE']
        self.aez_coords = self.check_exist(os.path.join(self.ref_dir, r['aez_coords']), 'file', self.log)
        # self.basin_coords = self.check_exist(os.path.join(self.ref_dir, r['basin_coords']), 'file')
        self.country_coords = self.check_exist(os.path.join(self.ref_dir, r['country_coords']), 'file', self.log)
        self.gcam_regnamefile = self.check_exist(os.path.join(self.ref_dir, r['gcam_regnamefile']), 'file', self.log)
        self.region_coords = self.check_exist(os.path.join(self.ref_dir, r['region_coords']), 'file', self.log)

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
        self.luc_ts_luc = self.create_dir(os.path.join(self.out_dir, o['luc_ts_luc']), self.log)
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
        self.model = p['model']
        self.metric = p['metric'].upper()
        self.run_desc = p['run_desc']
        self.use_constraints = int(p['use_constraints'])
        self.agg_level = int(p['agg_level'])
        self.resin = float(p['resin'])
        self.pkey = p['base_id_field']
        self.errortol = float(p['errortol'])
        self.year_b = int(p['year_b'])
        self.year_e = int(p['year_e'])
        self.timestep = int(p['timestep'])
        self.scenario = p['scenario']
        self.diagnostic = int(p['diagnostic'])
        self.intensification_ratio = float(p['intensification_ratio'])
        self.selection_threshold = float(p['selection_threshold'])
        self.map_kernels = int(p['map_kernels'])
        self.map_luc = int(p['map_luc'])
        self.map_luc_steps = int(p['map_luc_steps'])
        self.kerneldistance = int(p['kerneldistance'])
        self.map_tot_luc = int(p['map_tot_luc'])
        self.save_netcdf = int(p['save_netcdf'])
        self.map_constraints = int(p['map_constraints'])
        self.stochastic_expansion = int(p['stochastic_expansion'])
        self.save_transitions = int(p['save_transitions'])
        self.save_transition_maps = int(p['map_transitions'])
        self.save_shapefile = int(p['save_shapefile'])

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
            sys.exit()
        elif kind == 'dir' and os.path.isdir(f) is False:
            log.error("Directory not found:  {0}".format(f))
            log.error("Confirm path and retry.")
            sys.exit()
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
            sys.exit()

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
                    l.append(self.check_exist(os.path.join(self.constraints_dir, i), 'file'), self.log)

            return l

        else:
            return list()


if __name__ == "__main__":

    ini = '/users/d3y010/repos/github/demeter/config.ini'

    ReadConfig(ini)