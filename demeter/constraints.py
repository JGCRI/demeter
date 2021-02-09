"""
Apply user-defined constraints to base land use layer and predicted allocation data.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)
"""
import os
import numpy as np

import demeter.demeter_io.reader as rdr


class ValidationException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class ApplyConstraints:

    def __init__(self, allreg, allaez, final_landclasses, user_years, ixr_ixm, allregaez, spat_region, allregnumber,
                 spat_aez, gcam_landclasses, gcam_regionnumber, gcam_aez, gcam_landname, gcam_agg, gcam_ludata, ngrids,
                 constraint_names, spat_landclasses, spat_agg, spat_ludata, map_luc_steps, map_luc,
                 constraint_files):

        self.allreg = allreg
        self.allaez = allaez
        self.final_landclasses = final_landclasses
        self.l_allreg = len(allreg)
        self.l_allaez = len(allaez)
        self.l_flcs = len(final_landclasses)
        self.user_years = user_years
        self.ixr_ixm = ixr_ixm
        self.allregaez = allregaez
        self.spat_region = spat_region
        self.allregnumber = allregnumber
        self.spat_aez = spat_aez
        self.gcam_landclasses = gcam_landclasses
        self.gcam_regionnumber = gcam_regionnumber
        self.gcam_aez = gcam_aez
        self.gcam_landname = gcam_landname
        self.gcam_agg = gcam_agg
        self.gcam_ludata = gcam_ludata
        self.ngrids = ngrids
        self.constraint_names = constraint_names
        self.spat_landclasses = spat_landclasses
        self.spat_agg = spat_agg
        self.spat_ludata = spat_ludata
        self.map_luc_steps = map_luc_steps
        self.map_luc = map_luc
        self.constraint_files = constraint_files

        # assign array holding constraints
        self.cons_data = self.compile_constraints()

    def compile_constraints(self):
        """
        Create array to house constraints. Populate constraints.  Position 0 is reserved for kernel density constraints
        with are created on-the-fly later in the model.

        :param constraint_files                 Array of full path file names to constraint data
        :param ngrids:                          Number of grid cells in base land use layer
        :return:                                Numpy array housing [ empty (for kernel density), cons_n...]
        """

        # create an empty array to house constraints data, add one for kernel density space holder
        cons_data = np.zeros((self.ngrids, len(self.constraint_files) + 1))

        # add constraint data to output array upon import, leave position 0 for kernel density rules
        for f in self.constraint_files:

            # get index number from file name
            idx = int(os.path.basename(f).split('_')[0])

            # read in file and add to out array
            cons_data[:, idx] = rdr.to_array(f, target_index=1)

        return cons_data

    def apply_spat_constraints(self):
        """
        Apply user-defined constraints to base land use layer data.  This bins the spatial data into the final
        functional types.

        :return:
        """

        # create array to house the constrained base layer land cover data
        spat_ludataharm = np.zeros((self.ngrids, self.l_flcs))

        # assign the amount of PFT after splitting based upon user specification
        for idx, i in enumerate(self.spat_landclasses):

            # assign target row
            t = self.spat_agg[idx, :]

            # if non-permitted constrain assignment occurred in spatial allocation rules file, exit
            if np.sum(t) > 1:
                print("\nERROR: Aggregation numbers for PFT {0} in spatial allocation file sum up to more than 1.".format(i))
                print("Please correct and try again.")
                print("Exiting...\n")
                raise ValidationException

            # if individual values sum to greater than 1
            if np.sum(t > 0) > 1:
                a = np.tile(self.spat_ludata[:, idx], (np.sum(t > 0), 1)).T * np.tile(t[t > 0], (self.ngrids, 1))
                spat_ludataharm[:, t > 0] += a

            elif np.sum(t > 0) == 1:
                b = self.spat_ludata[:, idx:idx+1] * np.tile(t[t > 0], (self.ngrids, 1))
                spat_ludataharm[:, t > 0] += b

        # keep original data for mapping
        spat_ludataharm_orig_steps = spat_ludataharm * 1.
        spat_ludataharm_orig = spat_ludataharm * 1.

        return spat_ludataharm, spat_ludataharm_orig_steps, spat_ludataharm_orig

    def reg_metric_lcs_zip(self, lcs_list):
        """
        Create list of lists containing [region_ix, metric_ix, land cover class ix].

        """
        l = []
        for reg, met in self.ixr_ixm:
            for idx, lc in enumerate(lcs_list):
                l.append([reg, met, idx])

        return l

    def build_gcam_landmatrix(self):
        """
        Create zero array to house GCAM land use summary.

        :return:        Zero 3D array of region, metric, final land classes
        """
        return np.zeros((len(self.user_years), self.l_allreg, self.l_allaez, self.l_flcs))

    def build_spat_landmatrix(self, spat_ludataharm):
        """
        Create data summary by region, metric, final land class to conduct the GCAM aggregation.

        :return
        """
        # create iterator for [region_ix, metric_ix, final_lcs_ix]
        l = self.reg_metric_lcs_zip(self.final_landclasses)

        # create zero array to house base land use layer summary
        spat_landmatrix = np.zeros((self.l_allreg, self.l_allaez, self.l_flcs))

        for reg, met, lix in l:

            # create array of index from base layer data that meets criteria
            regaezind = np.where((self.spat_region == self.allregnumber[reg]) & (self.spat_aez == self.allregaez[reg][met]))[0]

            # populate base land use layer summary array with summed area in km2
            spat_landmatrix[reg, self.allregaez[reg][met] - 1, lix] = np.sum(spat_ludataharm[regaezind, lix])

        return spat_landmatrix

    def apply_constraints_zero(self):
        """
        Apply constraints that occur independent of target year.

        :param self:
        :return:
        """
        # create zero array to house GCAM land use summary
        gcam_landmatrix = self.build_gcam_landmatrix()

        # create iterator for [region_ix, metric_ix, gcam_lcs_ix]
        ixr_ixm_ixg = self.reg_metric_lcs_zip(self.gcam_landclasses)

        return gcam_landmatrix, ixr_ixm_ixg

    def apply_gcam_constraints(self, yr_idx, gcam_landmatrix, spat_landmatrix, ixr_ixm_ixg):
        """
        Apply user-defined constraints to GCAM land use data.

        :return:                target_change; change per [region, metric, pft]
        """
        # populate data summary matrix and convert GCAM data to common PFT scheme
        for reg, met, gix in ixr_ixm_ixg:

            # set target value
            t = self.gcam_agg[gix, :]

            # create array of index from GCAM land use data that meets criteria
            regaezlandind = np.where((self.gcam_regionnumber == self.allregnumber[reg])
                                     & (self.gcam_aez == self.allregaez[reg][met])
                                     & (self.gcam_landname == self.gcam_landclasses[gix]))[0]

            # check for missing aggregation setting in GCAM allocation file
            if np.sum(t) == 0:
                print("\nERROR: No aggregation class defined for PFT {0} in the GCAM allocation file".format(self.gcam_landclasses(gix)))
                print("Please correct and try again.")
                print("Exiting...\n")
                raise ValidationException

            # Examine the case of one-to-many recombination (e.g., rockicedesert to snow and sparse). Data is split into
            #   the new categories following their share in the base land use layer within the considered region,
            #   metric.  In the case that the actually do not exist in the considered region, metric; do a 50-50 split.
            if np.sum(t > 0) > 1:

                # where there is no value
                if np.sum(spat_landmatrix[reg, self.allregaez[reg][met] - 1, t == 1]) == 0:

                    # calculate split
                    xs = np.sum(self.gcam_ludata[regaezlandind, yr_idx]) / float(np.sum(t == 1))

                    # add to summary array
                    gcam_landmatrix[yr_idx, reg, self.allregaez[reg][met] - 1, t == 1] += xs

                # when value exist
                else:

                    # calculate proportion from area
                    xf = spat_landmatrix[reg, self.allregaez[reg][met] - 1, t == 1] \
                        / np.sum(spat_landmatrix[reg, self.allregaez[reg][met] - 1, t == 1]) \
                        * np.sum(self.gcam_ludata[regaezlandind, yr_idx])

                    # add to summary array
                    gcam_landmatrix[yr_idx, reg, self.allregaez[reg][met] - 1, t == 1] += xf

            # case of one-to-one recombination (e.g., unmanagedforest to forests
            if np.sum(t > 0) == 1:

                # sum GCAM land use data fitting criteria
                xo = np.sum(self.gcam_ludata[regaezlandind, yr_idx])

                # add to summary array
                gcam_landmatrix[yr_idx, reg, self.allregaez[reg][met] - 1, t == 1] += xo

        # compute land mistmatch (in base year) and land use change (in other years) for all regions, metrics, PFT
        land_mismatch = gcam_landmatrix[yr_idx, :, :, :] - spat_landmatrix
        target_change = gcam_landmatrix[yr_idx, :, :, :] - spat_landmatrix

        # round target change array to remove handle values very close to 0
        target_change = np.around(target_change, decimals=4)

        return [gcam_landmatrix, spat_landmatrix, land_mismatch, target_change]
