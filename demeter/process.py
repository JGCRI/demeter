"""
Processes the calculation, map creation, and logging of statistical methods used in determining land use change.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)

"""

import numpy as np
import os

import demeter.change.intensification as itz
import demeter.change.expansion as exp
import demeter.demeter_io.writer as wdr


class ProcessStep:
    """Process downscaling of a time step."""

    def __init__(self, config, s, step_idx, step, write_outputs=False):

        self.config = config
        self.s = s
        self.step_idx = step_idx
        self.step = step
        self.spat_landmatrix = None
        self.land_mismatch = None
        self.target_change = None
        self.transitions = None
        self.l_spat_region = len(self.s.spat_region)
        self.l_order_rules = len(self.s.order_rules)
        self.l_fcs = len(s.final_landclasses)
        self.write_outputs = write_outputs

        # populate
        self.output_df = self.process()

    def prep_step(self):
        """Prepare step-specific data."""

        # create data summary by region, metric, final land class to conduct the GCAM aggregation
        self.spat_landmatrix = self.s.cst.build_spat_landmatrix(self.s.spat_ludataharm)

        # assign GCAM land use arrays to class
        gcam_cns = self.s.cst.apply_gcam_constraints(self.step_idx, self.s.gcam_landmatrix, self.spat_landmatrix,
                                                    self.s.ixr_ixm_ixg)

        # unpack constraint attributes
        self.s.gcam_landmatrix, self.spat_landmatrix, self.land_mismatch, self.target_change = gcam_cns

        # calculate kernel density
        self.s.kernel_vector = self.s.kd.apply_convolution(self.s.cellindexresin, self.s.pft_maps, self.s.kernel_maps,
                                                           self.s.lat, self.s.lon, self.step, self.s.kernel_vector,
                                                           self.s.weights, self.s.spat_ludataharm)

        # create transition array to store data
        self.transitions = np.zeros(shape=(self.l_spat_region, self.l_order_rules, self.l_order_rules))

    def intense_pass(self, pass_num):
        """Conduct the first pass of intensification."""

        self.config.logger.info("Applying intensification: pass {0} for time step {1}...".format(pass_num, self.step))

        # set pass dir
        od = 'self.config.intensification_pass{0}_output_dir'.format(pass_num)
        out_dir = eval(od)

        # apply intensification
        itz_pass = itz.apply_intensification(self.config.logger, pass_num, self.config, self.s.spat_region, self.s.order_rules,
                                                 self.s.allregnumber, self.s.allregaez, self.s.spat_ludata,
                                                 self.spat_landmatrix, self.s.gcam_landmatrix, self.step_idx,
                                                 self.s.d_regid_nm, self.target_change, self.s.spat_ludataharm,
                                                 self.s.spat_aez, self.s.kernel_vector, self.s.cst.cons_data,
                                                 self.s.final_landclasses, self.s.spat_ludataharm_orig_steps, self.step,
                                                 self.land_mismatch, self.s.constraint_rules, self.s.transition_rules,
                                                 self.transitions)

        # unpack
        self.s.spat_ludataharm, self.s.spat_ludataharm_orig_steps, self.land_mismatch, self.s.cons_data, \
        self.transitions, self.target_change = itz_pass

        # create intensification first pass map if user-selected
        if self.config.map_luc_steps == 1:

            self.config.logger.info("Creating LUC intensification pass {0} maps for time step {1}...".format(pass_num, self.step))

            wdr.map_luc(self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.spat_ludataharm_orig_steps / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.cellindexresin, self.s.lat, self.s.lon, self.s.final_landclasses, self.step,
                        self.config.region_coords, self.config.country_coords, out_dir,
                        'intensification_pass{0}'.format(pass_num))

            # set prev year array to current year
            self.s.spat_ludataharm_orig_steps = self.s.spat_ludataharm * 1.

    def expansion_pass(self):
        """
        Conduct expansion pass.
        """

        self.config.logger.info("Applying expansion for time step {0}...".format(self.step))

        # apply expansion
        exp_pass = exp.apply_expansion(self.config.logger, self.config, self.s.allregnumber, self.s.allregaez, self.s.spat_ludataharm,
                                       self.s.spat_region, self.s.spat_aez, self.s.kernel_vector, self.s.cons_data,
                                       self.s.order_rules, self.s.final_landclasses, self.s.constraint_rules,
                                       self.s.transition_rules, self.land_mismatch, self.transitions,
                                       self.s.spat_ludataharm_orig_steps, self.target_change, self.step)

        # unpack
        self.s.spat_ludataharm, self.s.spat_ludataharm_orig_steps, self.land_mismatch, self.s.cons_data, \
        self.transitions, self.target_change = exp_pass

        # create maps if user-selected
        if self.config.map_luc_steps == 1:

            self.config.logger.info("Creating LUC expansion maps for time step {0}...".format(self.step))

            wdr.map_luc(self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.spat_ludataharm_orig_steps / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.cellindexresin, self.s.lat, self.s.lon, self.s.final_landclasses, self.step,
                        self.config.region_coords, self.config.country_coords, self.config.luc_expand_dir, 'expansion')

            # set prev year array to current year
            self.s.spat_ludataharm_orig_steps = self.s.spat_ludataharm * 1.

    def outputs(self):
        """
        Create time step specific outputs.

        :param spat_ludataharm:         harmonized land cover in sqkm per grid cell per land class (n_cells, n_landclasses)
        :param spat_ludataharm_orig:    same as spat_ludataharm but for the previous time step (n_cells, n_landclasses)
        :param cellarea:                cell area in sqkm for each grid cell (n_cells)
        :param celltrunk:               actual percentage of the grid cell included in the data (n_cells)
        :param l_fcs:                   the number of land classes
        :param cellindexresin:          index of x, y for grid cell location (position, n_cells)
        :param lat:                     geographic coordinate for each latitude in grid
        :param lon:                     geographic coordinate for each longitude in grid
        :param step:                    time step being processed
        :param region_coords:           full path with extension to region coords file
        :param country_coords:          full path with extension to country coords file
        :param luc_ts_luc:              path to luc_timestep output dir
        :param transitions:             area in sqkm of each transition from one land class to another (n_cells, n_landclasses, n_landclasses)
        :param user_years:              a list of user selected years to process

        """

        # convert metric_id back to the original
        revert_metric_dict = {self.s.sequence_metric_dict[k]: k for k in self.s.sequence_metric_dict.keys()}
        orig_spat_aez = np.vectorize(revert_metric_dict.get)(self.s.spat_aez)

        # convert land cover from sqkm per grid cell per land class to fraction for mapping (n_grids, n_landclasses)
        map_fraction_lu = self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T

        # do the same for the previous or starting step for mapping
        map_fraction_lu_prev = self.s.spat_ludataharm_orig / np.tile(self.s.cellarea, (self.l_fcs, 1)).T

        # convert land cover from sqkm per grid cell per land class to fraction (n_grids, n_landclasses)
        fraction_lu = self.s.spat_ludataharm / np.tile(self.s.cellarea * self.s.celltrunk, (self.l_fcs, 1)).T

        # create map grids of spatial data in grid cell fraction; -9999 is NODATA; (lat_val, lon_val, n_landclasses)
        map_grid_prev = np.zeros((len(self.s.lat), len(self.s.lon), len(self.s.final_landclasses))) + -9999
        map_grid_now = np.zeros((len(self.s.lat), len(self.s.lon), len(self.s.final_landclasses))) + -9999
        map_grid_prev[np.int_(self.s.cellindexresin[0, :]), np.int_(self.s.cellindexresin[1, :]), :] = map_fraction_lu_prev
        map_grid_now[np.int_(self.s.cellindexresin[0, :]), np.int_(self.s.cellindexresin[1, :]), :] = map_fraction_lu
        map_grid_chg = map_grid_now - map_grid_prev

        # optionally map time step
        if (self.config.map_luc_pft == 1) and (self.step in self.config.target_years_output):

            self.config.logger.info("Mapping land cover change for time step {0}...".format(self.step))

            wdr.map_luc(map_fraction_lu, map_fraction_lu_prev, self.s.cellindexresin, self.s.lat, self.s.lon,
                        self.s.final_landclasses, self.step, self.config.region_coords, self.config.country_coords,
                        self.config.luc_ts_luc, 'timestep_luc')

            # set prev year array to current year for next time step iteration
            self.s.spat_ludataharm_orig = self.s.spat_ludataharm * 1.

        # optionally save land cover transitions as a CSV
        if (self.config.save_transitions == 1) and (self.step in self.config.target_years_output):

            self.config.logger.info("Saving land cover transition files for time step {0}...".format(self.step))

            wdr.write_transitions(self.s, self.config, self.step, self.transitions)

        # optionally create land cover transition maps
        if (self.config.map_transitions == 1) and (self.step in self.config.target_years_output):

            self.config.logger.info("Saving land cover transition maps for time step {0}...".format(self.step))

            wdr.map_transitions(self.s, self.config, self.step, self.transitions)

        # create a NetCDF file of land cover fraction for each year by grid cell containing each land class
        if (self.config.save_netcdf_yr == 1) and (self.step in self.config.target_years_output):

            self.config.logger.info("Saving output in NetCDF format for time step {0} per land class...".format(self.step))

            # create out path and file name for NetCDF file
            netcdf_yr_out = os.path.join(self.config.lc_per_step_nc, 'lc_yearly_{0}.nc'.format(self.step))

            wdr.to_netcdf_yr(fraction_lu, self.s.cellindexresin, self.s.lat, self.s.lon, self.config.resin,
                             self.s.final_landclasses, self.step, self.config.model, netcdf_yr_out)

        # create a NetCDF file of land cover fraction for each land class by grid cell containing each year
        if (self.config.save_netcdf_lc == 1) and (self.step in self.config.target_years_output):

            self.config.logger.info("Saving stacked land class for time step {0}...".format(self.step))

            wdr.to_netcdf_lc(map_grid_now, self.s.lat, self.s.lon, self.config.resin,
                             self.s.final_landclasses, self.s.user_years, self.step,
                             self.config.model, self.config.lc_per_step_nc)

        # save land cover data for the time step
        if (self.config.save_tabular == 1) and (self.step in self.config.target_years_output):
            # self.config.logger.info("Saving tabular land cover data for time step {0}...".format(self.step))
            return wdr.lc_timestep_csv(self.config, self.step, self.s.final_landclasses, self.s.spat_coords, orig_spat_aez,
                                self.s.spat_region, self.s.spat_water, self.s.cellarea, self.s.spat_ludataharm,
                                self.config.metric, self.config.tabular_units, self.write_outputs)

        # optionally save land cover data for the time step as a shapefile
        if (self.config.save_shapefile == 1) and (self.step in self.config.target_years_output):
            self.config.logger.info("Saving land cover data for time step as a shapefile {0}".format(self.step))
            wdr.to_shp(self.config, self.step, self.s.final_landclasses)

        # create an ASCII raster with the land class number having the maximum area for each grid cell
        if (self.config.save_ascii_max == 1) and (self.step in self.config.target_years_output):
            self.config.logger.info("Saving output in ASCII raster format for time step {0}...".format(self.step))

            # call function for output object using available data detailed in this methods docstring
            wdr.max_ascii_rast(map_grid_now, self.config.out_dir, self.step, cellsize=self.config.resin)

        # --------- NEW OUTPUT PARAM HERE --------- #
        # Create a conditional statement after the following for your extended format where
        #   your parameter created in config_reader.py is in the place of 'self.config.save_ascii_max' with the same
        #   'self.config.' prefix.  Then call your function from writer.py using wdr as the prefix
        #   alias (e.g., wdr.your_function).

        # --------- END OUTPUT EXTENSION --------- #

    def process(self):
        """
        Process downscaling of a time step.
        """

        # prepare step-specific data
        self.prep_step()

        # apply first pass of intensification
        self.intense_pass(1)

        # apply expansion
        self.expansion_pass()

        # apply second pass of intensification
        self.intense_pass(2)

        # outputs
        return self.outputs()
