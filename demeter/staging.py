"""
Prepare data for processing.


Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)

"""

import numpy as np
import pandas as pd
import time
import os

import demeter.demeter_io.reader as rdr
import demeter.demeter_io.writer as wdr
import demeter.reconcile as rec
from demeter.constraints import ApplyConstraints
from demeter.weight.kernel_density import KernelDensity


class Stage:
    """Prepare data for processing."""

    def __init__(self, config, log):

        self.config = config
        self.log = log

        # GCAM region_name: region_id as dictionary
        self.d_regnm_id = rdr.to_dict(self.config.gcam_region_names_file, header=True, swap=True)

        # GCAM region_id: region_name as dictionary
        self.d_regid_nm = rdr.to_dict(self.config.gcam_region_names_file, header=True, swap=False)

        # GCAM basin_id: basin_glu_name as dictionary
        self.d_bsnnm_id = rdr.to_dict(self.config.gcam_basin_names_file, header=True, swap=True, value_col=2)

        self.d_reg_nm = None
        self.final_landclasses = None
        self.spat_landclasses = None
        self.spat_agg = None
        self.gcam_landclasses = None
        self.gcam_agg = None
        self.transition_rules = None
        self.order_rules = None
        self.constraint_names = None
        self.constraint_weights = None
        self.kernel_constraints = None
        self.constraint_rules = None
        self.user_years = None
        self.gcam_ludata = None
        self.gcam_aez = None
        self.gcam_landname = None
        self.gcam_regionnumber = None
        self.allreg = None
        self.allregnumber = None
        self.allregaez = None
        self.allaez = None
        self.spat_ludata = None
        self.spat_water = None
        self.spat_coords = None
        self.spat_aez_region = None
        self.spat_grid_id = None
        self.spat_aez = None
        self.spat_region = None
        self.ngrids = None
        self.cellarea = None
        self.celltrunk = None
        self.spat_regaezarea = None
        self.gcam_regaezarea = None
        self.areacoef = None
        self.gcam_regaezareaharm = None
        self.ixr_idm = None
        self.ixy_ixr_ixm = None
        self.gcam_ludata = None
        self.spat_ludataharm = None
        self.spat_ludataharm_orig_steps = None
        self.spat_ludataharm_orig = None
        self.lat = None
        self.lon = None
        self.cellindexresin = None
        self.pft_maps = None
        self.kernel_maps = None
        self.kernel_vector = None
        self.weights = None
        self.cst = None
        self.kd = None
        self.gcam_landmatrix = None
        self.ixr_ixm_ixg = None
        self.metric_id_array = None
        self.sequence_metric_dict = None
        self.metric_not_in_prj = None
        self.metric_sequence_list, self.region_sequence_list = self.prep_reference()

        # populate
        self.stage()

    def read_allocation(self):
        """Read in allocation files."""

        self.log.info("Reading allocation input files...")

        # set start time
        t0 = time.time()

        # create spatial allocation array
        spat_alloc = rdr.read_allocation_data(self.config.spatial_allocation_file, lc_col='category')
        self.final_landclasses, self.spat_landclasses, self.spat_agg = spat_alloc

        # create GCAM allocation array
        proj_alloc = rdr.read_allocation_data(self.config.gcam_allocation_file, lc_col='category', output_level=2)
        self.gcam_landclasses, self.gcam_agg = proj_alloc

        # create transition priority rules array
        self.transition_rules = rdr.read_allocation_data(self.config.transition_order_file, lc_col='category', output_level=1)

        # create treatment order rules array
        self.order_rules = rdr.to_list(self.config.treatment_order_file)

        # create constraints weighting array
        if self.config.use_constraints == 1:
            constraint_alloc = rdr.read_allocation_data(self.config.constraints_file, lc_col='category', output_level=2)
            self.constraint_names, self.constraint_weights = constraint_alloc
        else:
            self.constraint_names = []
            self.constraint_weights = []

        # create kernel density weighting array; 1 equals no constraints
        self.kernel_constraints = rdr.read_allocation_data(self.config.kernel_allocation_file, lc_col='category', output_level=1)

        # add kernel density constraints to constrain rules array; if there are no other constraints, apply KD array
        try:
            self.constraint_rules = np.insert(self.constraint_weights, [0], self.kernel_constraints, axis=0)
        except ValueError:
            self.constraint_rules = self.kernel_constraints

        self.log.info('PERFORMANCE:  Allocation files processed in {0} seconds'.format(time.time() - t0))

    def prep_reference(self):
        """Read the corresponding reference file to the associated basin or AEZ metric.
        Also read in region ids.

        :param f:                       Full path with filename and extension to the input file
        :param metric:                  basin or aez
        :return:                        Sorted list of metric ids, Sorted list of region ids
        """
        # if basin
        met = self.config.metric.lower()
        if met == 'basin':
            df = pd.read_csv(os.path.join(self.config.reference_dir, 'gcam_basin_lookup.csv'), usecols=['basin_id'])
            m = sorted(df['basin_id'].tolist())

        # if AEZ, use 1 through 18 - this will not change
        elif met == 'aez':
            m = list(range(1, 19, 1))

        # read in region ids
        rdf = pd.read_csv(os.path.join(self.config.reference_dir, 'gcam_regions_32.csv'), usecols=['gcam_region_id'])
        r = sorted(rdf['gcam_region_id'].tolist())

        return m, r

    def prep_projected(self):
        """
        Prepare projected land allocation data.
        """

        self.log.info("Preparing projected land use data...")

        # set start time
        t0 = time.time()

        if self.config.gcam_database is None:
            self.log.info(f"Using projected GCAM data from:  {self.config.projected_lu_file}")
            lu = self.config.projected_lu_file
        else:
            self.log.info(f"Using projected GCAM data from:  {self.config.gcam_database}")
            lu = rdr.read_gcam_land(self.config.gcam_database_dir, self.config.gcam_database_name, self.config.gcam_query, self.d_bsnnm_id,
                                    self.config.metric, self.config.crop_type)

        # extract and process data contained from the land allocation GCAM output file
        gcam_data = rdr.read_gcam_file(self.log, lu, self.gcam_landclasses, start_yr=self.config.start_year,
                                       end_yr=self.config.end_year, scenario=self.config.scenario, region_dict=self.d_regnm_id,
                                       agg_level=self.config.agg_level, area_factor=self.config.proj_factor,
                                       metric_seq=self.metric_sequence_list)

        # unpack variables
        self.user_years, self.gcam_ludata, self.gcam_aez, self.gcam_landname, self.gcam_regionnumber, self.allreg, \
        self.allregnumber, self.allregaez, self.allaez, self.metric_id_array, self.sequence_metric_dict = gcam_data

        self.log.info('PERFORMANCE:  Projected landuse data prepared in {0} seconds'.format(time.time() - t0))

    def prep_base(self):
        """
        Prepare base layer land use data.
        """

        self.log.info("Preparing base layer land use data...")

        # set start time
        t0 = time.time()

        # extract and process base layer land cover data
        base_data = rdr.read_base(self.log, self.c, self.spat_landclasses, self.sequence_metric_dict,
                                  metric_seq=self.metric_sequence_list, region_seq=self.region_sequence_list)

        # unpack variables
        self.spat_ludata, self.spat_water, self.spat_coords, self.spat_aez_region, self.spat_grid_id, self.spat_aez, \
        self.spat_region, self.ngrids, self.cellarea, self.celltrunk, self.sequence_metric_dict = base_data

        self.log.info('PERFORMANCE:  Base spatial landuse data prepared in {0} seconds'.format(time.time() - t0))

    def harmony(self):
        """
        Harmonize grid area between projected and base layer land allocation.
        """

        self.log.info("Harmonizing grid area...")

        # reset start time
        t0 = time.time()

        # reconcile GCAM land use area with base layer land use data
        recon_data = rec.reconcile(self.allreg, self.allaez, self.allregnumber, self.allregaez, self.spat_aez,
                                   self.spat_region, self.spat_ludata, self.user_years, self.gcam_ludata, self.gcam_aez,
                                   self.gcam_regionnumber)

        # unpack variables
        self.spat_regaezarea, self.gcam_regaezarea, self.areacoef, self.gcam_regaezareaharm, self.ixr_idm, \
        self.ixy_ixr_ixm, self.gcam_ludata = recon_data

        # write harmonization coefficient array as a diagnostics file
        if self.config.diagnostic == 1:
            wdr.save_array(self.areacoef, self.config.harm_coeff_file)

        self.log.info('PERFORMANCE:  Harmonization completed in {0} seconds'.format(time.time() - t0))

    def constrain(self):
        """
        Apply constraints to projected and base layer land allocation and use data.
        """

        self.log.info("Applying base layer land use constraints and prepping future projection constraints...")

        # set start time
        t0 = time.time()

        # apply user-defined constraints to base land use layer data and GCAM land use data
        self.cst = ApplyConstraints(self.allreg, self.allaez, self.final_landclasses, self.user_years, self.ixr_idm,
                                   self.allregaez, self.spat_region, self.allregnumber, self.spat_aez,
                                   self.gcam_landclasses, self.gcam_regionnumber, self.gcam_aez, self.gcam_landname,
                                   self.gcam_agg, self.gcam_ludata, self.ngrids, self.constraint_names,
                                   self.spat_landclasses, self.spat_agg, self.spat_ludata, self.config.map_luc_steps,
                                   self.config.map_luc_pft, self.config.constraint_files)

        # apply spatial constraints
        self.spat_ludataharm, self.spat_ludataharm_orig_steps, self.spat_ludataharm_orig = self.cst.apply_spat_constraints()

        self.log.info('PERFORMANCE:  Constraints applied to projected and spatial data in {0} seconds'.format(time.time() - t0))

    def kernel_filter(self):
        """
        Create kernel density filter.
        """
        self.log.info("Creating and processing kernel density...")

        # reset start time
        t0 = time.time()

        # instantiate kernel density class
        self.kd = KernelDensity(self.config.spatial_resolution, self.spat_coords, self.final_landclasses,
                                self.config.kernel_distance, self.ngrids, self.config.kernel_maps_output_dir,
                                self.order_rules, self.config.map_kernels)

        # preprocess year-independent kernel density data
        self.lat, self.lon, self.cellindexresin, self.pft_maps, self.kernel_maps, self.kernel_vector, self.weights = self.kd.preprocess_kernel_density()

        # log processing time
        self.log.info('PERFORMANCE:  Kernel density filter prepared in {0} seconds'.format(time.time() - t0))

    def set_step_zero(self):
        """
        Set data for initial time step
        """
        self.gcam_landmatrix, self.ixr_ixm_ixg = self.cst.apply_constraints_zero()

    def stage(self):
        """
        Run processing that prepares data used in the processing of each time step.
        """

        # read in allocation files
        self.read_allocation()

        # prepare projected land allocation data
        self.prep_projected()

        # prepare base land use data
        self.prep_base()

        # harmonize grid area between projected and base layer land allocation
        self.harmony()

        # apply constraints
        self.constrain()

        # create kernel density filter if not running multiple jobs
        self.kernel_filter()

        # set data for step zero
        self.set_step_zero()
