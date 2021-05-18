"""
Prepare data for processing.


Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)

"""

import os
import time
import logging

import numpy as np
import pandas as pd

import demeter.demeter_io.reader as rdr
import demeter.demeter_io.writer as wdr
import demeter.reconcile as rec
import demeter.preprocess_data as proc

from demeter.constraints import ApplyConstraints
from demeter.weight.kernel_density import KernelDensity


class Stage:
    """Prepare data for processing."""

    def __init__(self, config):

        self.config = config

        # GCAM region_name: region_id as dictionary
        self.d_regnm_id = rdr.to_dict(self.config.gcam_region_names_file, header=True, swap=True)

        # GCAM region_id: region_name as dictionary
        self.d_regid_nm = rdr.to_dict(self.config.gcam_region_names_file, header=True, swap=False)

        # GCAM basin_id: basin_glu_name as dictionary
        self.d_bsnnm_id = rdr.to_dict(self.config.gcam_basin_names_file, header=True, swap=True, value_col=2)

        self.config.logger.info("Reading allocation input files...")

        # set start time
        t0 = time.time()

        # create spatial allocation array
        spat_alloc = rdr.read_allocation_data(self.config.spatial_allocation_file, lc_col='category')
        self.final_landclasses, self.observed_landclasses, self.observed_array = spat_alloc

        # create GCAM allocation array
        proj_alloc = rdr.read_allocation_data(self.config.gcam_allocation_file, lc_col='category', output_level=2)
        self.gcam_landclasses, self.gcam_array = proj_alloc

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
            self.constraint_rules = self.kernel_constraints.copy()

        self.config.logger.info('PERFORMANCE:  Allocation files processed in {0} seconds'.format(time.time() - t0))

        self.metric_sequence_list, self.region_sequence_list = self.prep_reference()

        # populate
        self.stage()

    def prep_reference(self):
        """Read the corresponding reference file to the associated basin or AEZ metric.
        Also read in region ids.

        :param f:                       Full path with filename and extension to the input file
        :param metric:                  basin or aez

        :return:                        Sorted list of metric ids, Sorted list of region ids

        """

        # if basin
        if self.config.metric == 'basin':
            df = pd.read_csv(self.config.gcam_basin_names_file, usecols=['basin_id'])
            m = sorted(df['basin_id'].tolist())

        # if AEZ, use 1 through 18 - this will not change
        elif self.config.metric == 'aez':
            m = list(range(1, 19, 1))

        # read in region ids
        rdf = pd.read_csv(self.config.gcam_region_names_file, usecols=['gcam_region_id'])
        r = sorted(rdf['gcam_region_id'].tolist())

        return m, r

    def prep_projected(self):
        """
        Prepare projected land allocation data.
        """

        self.config.logger.info("Preparing projected land use data...")

        # set start time
        t0 = time.time()

        if self.config.gcamwrapper_df is not None:

            self.config.logger.info(f"Using projected GCAM data from `gcamwrapper` data frame")
            projected_land_cover_file = proc.format_gcam_data(self.config.gcamwrapper_df,
                                                              f_out='',
                                                              start_year=self.config.start_year,
                                                              through_year=self.config.end_year,
                                                              region_name_field='gcam_region_name',
                                                              region_id_field='gcam_region_id',
                                                              basin_name_field='glu_name',
                                                              basin_id_field='basin_id',
                                                              output_to_csv=False)

        elif self.config.gcam_database is not None:

            self.config.logger.info(f"Using projected GCAM data from:  {self.config.gcam_database}")
            projected_land_cover_file = rdr.read_gcam_land(self.config.gcam_database_dir,
                                                           self.config.gcam_database_name,
                                                           self.config.gcam_query, self.d_bsnnm_id,
                                                           self.config.metric, self.config.crop_type)


        else:
            self.config.logger.info(f"Using projected GCAM data from:  {self.config.projected_lu_file}")
            projected_land_cover_file = self.config.projected_lu_file

        # extract and process data contained from the land allocation GCAM output file
        gcam_data = rdr.read_gcam_file(projected_land_cover_file,
                                       self.gcam_landclasses,
                                       start_yr=self.config.start_year,
                                       end_yr=self.config.end_year,
                                       scenario=self.config.scenario,
                                       region_dict=self.d_regnm_id,
                                       agg_level=self.config.agg_level,
                                       area_factor=self.config.proj_factor,
                                       metric_seq=self.metric_sequence_list,
                                       logger=self.config.logger)

        # unpack variables
        self.user_years, self.gcam_ludata, self.gcam_aez, self.gcam_landname, self.gcam_regionnumber, self.allreg, \
        self.allregnumber, self.allregaez, self.allaez, self.metric_id_array, self.sequence_metric_dict = gcam_data

        self.config.logger.info('PERFORMANCE:  Projected landuse data prepared in {0} seconds'.format(time.time() - t0))

    def prep_base(self):
        """
        Prepare base layer land use data.
        """

        self.config.logger.info("Preparing base layer land use data...")

        # set start time
        t0 = time.time()

        # extract and process base layer land cover data
        base_data = rdr.read_base(self.config, self.observed_landclasses, self.sequence_metric_dict,
                                  metric_seq=self.metric_sequence_list, region_seq=self.region_sequence_list)

        # unpack variables
        self.spat_ludata, self.spat_water, self.spat_coords, self.spat_aez_region, self.spat_grid_id, self.spat_aez, \
        self.spat_region, self.ngrids, self.cellarea, self.celltrunk, self.sequence_metric_dict = base_data

        self.config.logger.info('PERFORMANCE:  Base spatial landuse data prepared in {0} seconds'.format(time.time() - t0))

    def harmony(self):
        """
        Harmonize grid area between projected and base layer land allocation.
        """

        self.config.logger.info("Harmonizing grid area...")

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

        self.config.logger.info('PERFORMANCE:  Harmonization completed in {0} seconds'.format(time.time() - t0))

    def set_constraints(self):
        """
        Apply constraints to projected and base layer land allocation and use data.
        """

        self.config.logger.info("Applying base layer land use constraints and prepping future projection constraints...")

        # set start time
        t0 = time.time()

        # apply user-defined constraints to base land use layer data and GCAM land use data
        self.cst = ApplyConstraints(self.allreg, self.allaez, self.final_landclasses, self.user_years, self.ixr_idm,
                                   self.allregaez, self.spat_region, self.allregnumber, self.spat_aez,
                                   self.gcam_landclasses, self.gcam_regionnumber, self.gcam_aez, self.gcam_landname,
                                   self.gcam_array, self.gcam_ludata, self.ngrids, self.constraint_names,
                                   self.observed_landclasses, self.observed_array, self.spat_ludata, self.config.map_luc_steps,
                                   self.config.map_luc_pft, self.config.constraint_files, self.config.logger)

        # apply spatial constraints
        self.spat_ludataharm, self.spat_ludataharm_orig_steps, self.spat_ludataharm_orig = self.cst.apply_spat_constraints()

        self.config.logger.info('PERFORMANCE:  Constraints applied to projected and spatial data in {0} seconds'.format(time.time() - t0))

    def kernel_filter(self):
        """
        Create kernel density filter.
        """
        self.config.logger.info("Creating and processing kernel density...")

        # reset start time
        t0 = time.time()

        # instantiate kernel density class
        self.kd = KernelDensity(self.config.spatial_resolution, self.spat_coords, self.final_landclasses,
                                self.config.kernel_distance, self.ngrids, self.config.kernel_maps_output_dir,
                                self.order_rules, self.config.map_kernels)

        # preprocess year-independent kernel density data
        self.lat, self.lon, self.cellindexresin, self.pft_maps, self.kernel_maps, self.kernel_vector, self.weights = self.kd.preprocess_kernel_density()

        # log processing time
        self.config.logger.info('PERFORMANCE:  Kernel density filter prepared in {0} seconds'.format(time.time() - t0))

    def set_step_zero(self):
        """
        Set data for initial time step
        """
        self.gcam_landmatrix, self.ixr_ixm_ixg = self.cst.apply_constraints_zero()

    def stage(self):
        """
        Run processing that prepares data used in the processing of each time step.
        """

        # prepare projected land allocation data
        self.prep_projected()

        # prepare base land use data
        self.prep_base()

        # harmonize grid area between projected and base layer land allocation
        self.harmony()

        # apply constraints
        self.set_constraints()

        # create kernel density filter if not running multiple jobs
        self.kernel_filter()

        # set data for step zero
        self.set_step_zero()
