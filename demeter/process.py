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
    """
    Process downscaling of a time step.
    """

    def __init__(self, c, log, s, step_idx, step):

        self.c = c
        self.log = log
        self.s = s
        self.step_idx = step_idx
        self.step = step
        self.spat_landmatrix = None
        self.land_mismatch = None
        self.target_change = None
        self.transitions = None
        self.l_spat_region = len(s.spat_region)
        self.l_order_rules = len(s.order_rules)
        self.l_fcs = len(s.final_landclasses)

        # populate
        self.process()

    def prep_step(self):
        """
        Prepare step-specific data.
        """

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
        """
        Conduct the first pass of intensification.
        """
        self.log.info("Applying intensification: pass {0} for time step {1}...".format(pass_num, self.step))

        # set pass dir
        od = 'self.c.luc_intense_p{0}_dir'.format(pass_num)
        out_dir = eval(od)

        # apply intensification
        itz_pass = itz.apply_intensification(self.log, pass_num, self.c, self.s.spat_region, self.s.order_rules,
                                                 self.s.allregnumber, self.s.allregaez, self.s.spat_ludata,
                                                 self.spat_landmatrix, self.s.gcam_landmatrix, self.step_idx,
                                                 self.s.d_regid_nm, self.target_change, self.s.spat_ludataharm,
                                                 self.s.spat_aez, self.s.kernel_vector, self.s.cst.cons_data,
                                                 self.s.final_landclasses, self.s.spat_ludataharm_orig_steps, self.step,
                                                 self.land_mismatch, self.s.constrain_rules, self.s.transition_rules,
                                                 self.transitions)

        # unpack
        self.s.spat_ludataharm, self.s.spat_ludataharm_orig_steps, self.land_mismatch, self.s.cons_data, \
        self.transitions, self.target_change = itz_pass

        # create intensification first pass map if user-selected
        if self.c.map_luc_steps == 1:

            self.log.info("Creating LUC intensification pass {0} maps for time step {1}...".format(pass_num, self.step))

            wdr.map_luc(self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.spat_ludataharm_orig_steps / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.cellindexresin, self.s.lat, self.s.lon, self.s.final_landclasses, self.step,
                        self.c.region_coords, self.c.country_coords, out_dir,
                        'intensification_pass{0}'.format(pass_num))

            # set prev year array to current year
            self.s.spat_ludataharm_orig_steps = self.s.spat_ludataharm * 1.

    def expansion_pass(self):
        """
        Conduct expansion pass.
        """

        self.log.info("Applying expansion for time step {0}...".format(self.step))

        # apply expansion
        exp_pass = exp.apply_expansion(self.log, self.c, self.s.allregnumber, self.s.allregaez, self.s.spat_ludataharm,
                                       self.s.spat_region, self.s.spat_aez, self.s.kernel_vector, self.s.cons_data,
                                       self.s.order_rules, self.s.final_landclasses, self.s.constrain_rules,
                                       self.s.transition_rules, self.land_mismatch, self.transitions,
                                       self.s.spat_ludataharm_orig_steps, self.target_change, self.step)

        # unpack
        self.s.spat_ludataharm, self.s.spat_ludataharm_orig_steps, self.land_mismatch, self.s.cons_data, \
        self.transitions, self.target_change = exp_pass

        # create maps if user-selected
        if self.c.map_luc_steps == 1:

            self.log.info("Creating LUC expansion maps for time step {0}...".format(self.step))

            wdr.map_luc(self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.spat_ludataharm_orig_steps / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.cellindexresin, self.s.lat, self.s.lon, self.s.final_landclasses, self.step,
                        self.c.region_coords, self.c.country_coords, self.c.luc_expand_dir, 'expansion')

            # set prev year array to current year
            self.s.spat_ludataharm_orig_steps = self.s.spat_ludataharm * 1.

    def outputs(self):
        """
        Create time step specific outputs.
        """

        # optionally map time step
        if (self.c.map_luc == 1) and (self.step in self.c.target_years_output):

            self.log.info("Mapping land cover change for time step {0}...".format(self.step))

            wdr.map_luc(self.s.spat_ludataharm / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.spat_ludataharm_orig / np.tile(self.s.cellarea, (self.l_fcs, 1)).T,
                        self.s.cellindexresin, self.s.lat, self.s.lon, self.s.final_landclasses, self.step,
                        self.c.region_coords, self.c.country_coords, self.c.luc_ts_luc, 'timestep_luc')

            # set prev year array to current year for next time step iteration
            self.s.spat_ludataharm_orig = self.s.spat_ludataharm * 1.

        # optionally save land cover transitions as a CSV
        if (self.c.save_transitions == 1) and (self.step in self.c.target_years_output):

            self.log.info("Saving land cover transition files for time step {0}...".format(self.step))

            wdr.write_transitions(self.s, self.c, self.step, self.transitions)

        # optionally create land cover transition maps
        if (self.c.save_transition_maps == 1) and (self.step  in self.c.target_years_output):

            self.log.info("Saving land cover transition maps for time step {0}...".format(self.step))

            wdr.map_transitions(self.s, self.c, self.step, self.transitions)

        # create a NetCDF file of land cover percent by land class by grid cell containing each year interpolated to one-year intervals
        if (self.c.save_netcdf_lc == 1) and (self.step in self.c.target_years_output):

            pass

            # self.log.info("Saving output in NetCDF format for time step {0}...".format(self.step))

            # # create out path and file name for NetCDF file
            # netcdf_outfile = os.path.join(self.c.lc_per_step_nc, 'landcover_{0}.nc')

            # # create NetCDF file for each PFT that interpolates 5-year to yearly
            # wdr.to_netcdf_lc(self.s.spat_ludataharm / np.tile(self.s.cellarea * self.s.celltrunk, (self.l_fcs, 1)).T,
            #                   self.s.cellindexresin, self.s.lat, self.s.lon, self.c.resin, self.s.final_landclasses,
            #                   self.step, self.s.user_years, netcdf_outfile, self.c.timestep, self.c.model)

        # create a NetCDF file of land cover percent for each year by grid cell containing each land class
        if (self.c.save_netcdf_yr == 1) and (self.step in self.c.target_years_output):

            pass

            # self.log.info("Saving output in NetCDF format for time step {0}...".format(self.step))

        # save land cover data for the time step
        if (self.c.save_tabular == 1) and (self.step in self.c.target_years_output):
            self.log.info("Saving tabular land cover data for time step {0}...".format(self.step))
            wdr.lc_timestep_csv(self.c, self.step, self.s.final_landclasses, self.s.spat_coords, self.s.spat_aez,
                                self.s.spat_region, self.s.spat_water, self.s.cellarea, self.s.spat_ludataharm,
                                self.c.metric, self.c.tabular_units)

        # optionally save land cover data for the time step as a shapefile
        if (self.c.save_shapefile == 1) and (self.step in self.c.target_years_output):
            self.log.info("Saving land cover data for time step as a shapefile {0}".format(self.step))
            wdr.to_shp(self.c, self.step, self.s.final_landclasses)

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
        self.outputs()
