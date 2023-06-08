"""
Land use intensification algorithm.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com)
"""

import os
import numpy as np
import threading
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import demeter.demeter_io.writer as wdr

def intense_parallel_helper(regix_metix,spat_region, order_rules, allregnumber, allregmet, spat_ludata,
                          spat_landmatrix, gcam_landmatrix, yr_idx, d_regid_nm, target_change, spat_ludataharm,
                          spat_met, kernel_vector, cons_data, final_landclasses,spat_ludataharm_orig_steps, yr,
                          land_mismatch, constraint_rules, transition_rules,log, pass_number, c,diag_file):

    reg_idx, met_idx = regix_metix
   # print("processing region " + str(reg_idx))

        # set previous region index to current
        #prev_reg = reg_idx

    # update user per region change

        # update user per region change
    regnumber, reg_idx, target_intensification = _create_summary(reg_idx, allregnumber, spat_ludata,
                                                                      spat_landmatrix, gcam_landmatrix, d_regid_nm,
                                                                      log, spat_region, yr_idx, target_change,
                                                                      pass_number, c)

    # calculate and write area diagnostic
    # diff_diagnostic(c.diag_dir, d_regid_nm, gcam_landmatrix, spat_landmatrix, reg_idx, yr, yr_idx)

    # retrieve region and metric number
    metnumber = allregmet[reg_idx][met_idx]

    # create data subset
    reg_met_mask = (spat_region == regnumber) & (spat_met == metnumber)
    spat_ludataharm_sub = spat_ludataharm[reg_met_mask]
    kernel_vector_sub = kernel_vector[reg_met_mask]
    cons_data_sub = cons_data[reg_met_mask]

    # calculate intensification
    citz = _intensification(c.diagnostic, diag_file, spat_ludataharm_sub, target_intensification, kernel_vector_sub,
                            cons_data_sub, reg_idx, metnumber, order_rules, final_landclasses, c.errortol,
                            constraint_rules, target_change, transition_rules, land_mismatch)

    # apply intensification
    spat_ludataharm[reg_met_mask], trans_mat, target_change, target_intensification = citz

    # log transition
    # transitions[reg_met_mask, :, :] += trans_mat

    # calculate non-achieved change


    non_chg = np.sum(abs(target_change[:, :, :])) / 2.0

    if non_chg > 0:
       non_chg_per = np.sum(abs(target_change[:, :, :].flatten())) / np.sum(abs(land_mismatch[:, :, :].flatten())) * 100
    else:
       non_chg_per = 0

    #log.info("Total non-achieved intensification change for pass {0} time step {1}:  {2} km2 ({3} %)".format(pass_number, yr, non_chg, non_chg_per))






def diff_diagnostic(diag_outdir, d_regid_nm, gcam_landmatrix, spat_landmatrix, reg, yr, yr_idx):
    """
    Computes the difference between base land use layer and GCAM land use data. The start year represents the
    difference within observations and subsequent years represent land use change.

    :return:
    """
    # set outfile names
    gcam_out = os.path.join(diag_outdir, "{0}_{1}_gcam_landmatrix.csv".format(d_regid_nm[str(reg+1)], yr))
    base_out = os.path.join(diag_outdir, "{0}_{1}_spat_landmatrix.csv".format(d_regid_nm[str(reg+1)], yr))

    # write files
    wdr.array_to_csv(gcam_landmatrix[yr_idx, reg, :, :], gcam_out)
    wdr.array_to_csv(spat_landmatrix[reg, :, :], base_out)





def reg_metric_iter(allregnumber, allregmet):
    """
    Create region, metric iterator.

    :return:            List [[reg_idx, metric_idx], n]
    """
    l = []
    for reg in range(len(allregnumber)):
        for met in range(len(allregmet[reg])):
            l.append([reg, met])

    return l


def _convert_pft(notdone, int_target, metnumber, pft_toconv, spat_ludataharm_sub, pft, cons_data_subpft, reg,
                target_intensification, trans_mat, target_change, errortol, diag_file, diagnostic):
    """
    Apply conversion to every qualifying PFT.

    :return:            Array of PFTs
    """
    if diagnostic == 1:
        diag_file.write('{},{},{},{},{}\n'.format(reg+1, metnumber, pft, pft_toconv, int_target))

    while notdone:
        # grid cells with both the expanding and to-convert PFT
        exist_cells = np.where((spat_ludataharm_sub[:, pft] > 0)
                               & (spat_ludataharm_sub[:, pft_toconv] > 0))[0]

        # intensification constraints on grid cells, weighted by the user-input constrain weights
        cons_cells = cons_data_subpft[exist_cells, :]

        # combine and normalize constraints
        mean_cons_cells = np.nansum(cons_cells, axis=1) / np.nanmean(np.nansum(cons_cells, axis=1))
        mean_cons_cells /= np.nanmax([0.00000001, np.nanmax(mean_cons_cells)])

        # when there are no constraints because none is applied to the PFT (all NaN), then constrain is 1 (no constrain)
        mean_cons_cells[np.isnan(mean_cons_cells)] = 1.

        # intensification_likelihood = mean_cons_cells
        intensification_likelihood = np.power(mean_cons_cells, 2)

        # checking for non-zero values (if all is zero, no grid-cell will ever be selected)
        if np.nanmax(intensification_likelihood) == 0:
            intensification_likelihood[:] = 1.

        # total area that the PFT to convert could give
        swaparea = min([int_target, target_change[reg, metnumber - 1, pft_toconv] * -1,
                        np.sum(spat_ludataharm_sub[exist_cells, pft_toconv])])
        swaparea = min([swaparea, int_target])

        # applying the constraints, the less the constrain, the higher the fraction of potential expansion is allowed
        potexpansion = swaparea * intensification_likelihood / np.sum(intensification_likelihood)

        # actual expansion: for each grid cell get the minimum of: potential expansion, and actual expansion
        actexpansion = np.amin([potexpansion, spat_ludataharm_sub[exist_cells, pft_toconv]], axis=0)

        # applying land swap between both PFTs
        spat_ludataharm_sub[exist_cells, pft] += actexpansion
        spat_ludataharm_sub[exist_cells, pft_toconv] -= actexpansion

        # update the target change values
        actual_expansion_sum = np.sum(actexpansion)
        target_change[reg, metnumber - 1, pft] -= actual_expansion_sum
        int_target -= actual_expansion_sum
        target_intensification[metnumber - 1, pft] -= actual_expansion_sum
        target_change[reg, metnumber - 1, pft_toconv] += actual_expansion_sum
        target_intensification[metnumber - 1, pft_toconv] += actual_expansion_sum
        trans_mat[exist_cells, pft, pft_toconv] += actexpansion

        # account for target change minuscule values when evaluating notdone
        tc = round(target_change[reg, metnumber - 1, pft_toconv], 4)

        # updating notdone: if the intensification target has been reached, or if there are no more
        #   of the PFTs to convert in the considered grid cells, the break the loop
        notdone = (int_target > errortol) \
                  & (tc < -errortol) \
                  & (np.sum(spat_ludataharm_sub[exist_cells, pft_toconv]) > errortol) \
                  & (len(exist_cells) > 0) \
                  & (np.sum(mean_cons_cells) != len(mean_cons_cells))

        if diagnostic == 1:
            diag_file.write('{},{},{},{},{}\n'.format(reg + 1, metnumber, pft, pft_toconv, int_target))

    return int_target, target_change, trans_mat, target_intensification


def _intensification(diagnostic, diag_file, spat_ludataharm_sub, target_intensification, kernel_vector_sub,
                     cons_data_sub_o, reg, metnumber, order_rules, final_landclasses, errortol, constraint_rules,
                     target_change, transition_rules, land_mismatch):
    """
    Calculate intensification.  Follow user-defined order of treatment.
    :return:
    """
    # get lengths for array creation
    l_shs = len(spat_ludataharm_sub[:, 0])
    l_ord = len(order_rules)

    # initialize transition arrays
    trans_mat = np.zeros((l_shs, l_ord, l_ord))

    # process PFTs in order
    for pft_ord in np.unique(order_rules):

        # create a copy of the cons_data_sub array
        cons_data_sub = cons_data_sub_o.copy()

        # lookup PFT and final land class
        pft = np.where(order_rules == pft_ord)[0][0]
        fcs = final_landclasses[pft]

        # define intensification targets
        int_target = target_intensification[metnumber - 1, pft]
        int_target_const = target_intensification[metnumber - 1, pft]

        # determine if intensification is wanted by the user
        if int_target <= errortol:
            # print("\nNo intensification desired for:  {0}, {1}".format(fcs, int_target))
            pass
        else:
            # print("\nIntensification desired for:  {0}, {1}".format(fcs, int_target))

            # retrieve constraints for the PFT (e.g., soil quality, protection status, etc.)
            cons_rules_pft = constraint_rules[:, pft]

            # add kernel density to the constraints and normalize their value
            kdc = kernel_vector_sub[:, pft] / np.nanmax([0.00000001, np.nanmax(kernel_vector_sub[:, pft])])
            cons_data_sub[:, -1] = kdc

            # create index order for constraints array where kernel density will be position 0
            cons_idx_order = [0 if i == cons_data_sub.shape[1]-1 else i+1 for i in range(cons_data_sub.shape[1])]

            # reorder constraint weights array
            c_arg = np.argsort(cons_idx_order)
            cons_data_sub = cons_data_sub[:, c_arg]

            # apply the weight of each constrain for the target pft
            cons_data_subpft = cons_data_sub

            # invert negative constraints
            arr = np.ones(shape=np.shape(cons_data_sub[:, cons_rules_pft < 0])) + cons_data_subpft[:, cons_rules_pft < 0]
            cons_data_subpft[:, cons_rules_pft < 0] = arr

            # multiply negative constraints weight by -1 to turn it positive
            cons_rules_pft[cons_rules_pft < 0] *= -1

            # apply constraint to spatial data
            cons_data_subpft *= np.tile(cons_rules_pft, (len(spat_ludataharm_sub[:, pft]), 1))

            # zero means that constraint does not apply to the PFT, we turn these values into NaN
            cons_data_subpft[:, cons_rules_pft == 0] = np.nan

            # iterate through the conversion priority rules to find other PFTs that are contracting (where expansion
            #   can occur)
            for pft_tcvo in np.arange(1, len(transition_rules[pft]), 1):

                # get the PFT to convert and its final land class name
                pft_toconv = np.where(transition_rules[pft, :] == pft_tcvo)[0][0]

                # round target change value to ensure minuscule values near zero are not counted as negative
                if round(target_change[reg, metnumber - 1, pft_toconv], 4) < 0:
                    tce = 1
                else:
                    tce = 0

                # see if PFT to convert has to be converted in the target region, metric
                notdone = (int_target > errortol) & (tce == 1)

                # identify grid cells with both expanding and to-convert PFT
                exist_cells = np.where((spat_ludataharm_sub[:, pft] > 0) & (spat_ludataharm_sub[:, pft_toconv] > 0))[0]

                # apply conversion to every qualifying PFT
                if len(exist_cells) > 0:
                    cpft = _convert_pft(notdone, int_target, metnumber, pft_toconv, spat_ludataharm_sub, pft,
                                        cons_data_subpft, reg, target_intensification, trans_mat, target_change,
                                        errortol, diag_file, diagnostic)

                    int_target, target_change, trans_mat, target_intensification = cpft

        # report how much intensification we've achieved
        achieved = int_target_const - target_change[reg, metnumber - 1, pft]
        per_numer = land_mismatch[reg, metnumber - 1, pft] - target_change[reg, metnumber - 1, pft]
        per_denom = land_mismatch[reg, metnumber - 1, pft]

        # check for 0
        if per_numer == 0 or per_denom == 0:
            percent = 0
        else:
            percent = (per_numer / per_denom) * 100

    return spat_ludataharm_sub, trans_mat, target_change, target_intensification


def _create_summary(reg_idx, allregnumber, spat_ludata, spat_landmatrix, gcam_landmatrix, d_regid_nm,
                    log, spat_region, yr_idx, target_change, pass_number, c):
    """
    Create summary data for log output per region.
    """

    # get region numbers from index
    regnumber = allregnumber[reg_idx]

    # calculate summary
    tot_spat_area = np.sum(spat_ludata[spat_region == allregnumber[reg_idx]])
    tot_harm_spat = np.sum(spat_landmatrix[reg_idx, :, :])
    tot_harm_gcam = np.sum(gcam_landmatrix[yr_idx, reg_idx, :, :])

    # log summary
    # log.info("Processing region:  {0}".format(d_regid_nm[str(reg_idx + 1)]))
    # log.info("Total spatial land area:  {0} km2".format(tot_spat_area))
    # log.info("Total harmonized spatial land area:  {0} km2".format(tot_harm_spat))
    # log.info("Total harmonized GCAM land area:  {0} km2".format(tot_harm_gcam))

    # update prev idx
    prev_reg = reg_idx

    # set target intensification limit and store change
    target_intensification = target_change[reg_idx, :, :] * 1.

    # apply intensification ratio if this is the first intensification pass
    if pass_number == 1:
        target_intensification[target_intensification > 0] *= c.intensification_ratio



    return regnumber, prev_reg, target_intensification


def apply_intensification(log, pass_number, c, spat_region, order_rules, allregnumber, allregmet, spat_ludata,
                          spat_landmatrix, gcam_landmatrix, yr_idx, d_regid_nm, target_change, spat_ludataharm,
                          spat_met, kernel_vector, cons_data, final_landclasses,spat_ludataharm_orig_steps, yr,
                          land_mismatch, constraint_rules, transition_rules):
    """
    There are two ways to expand land covers:
    1) on grid-cells where they do exist (intensification, at the expense of contracting land covers)
    2) on grid-cells where they do not exist, although preferentially close to grid-cells where the landcover
    is found (expansion, or proximity expansion)
    There is a parameter (intensification_ratio) to control the desired ratio of intensification versus expansion.
    The downscaling first intensifies, until it reaches either that ratio, or the maximum intensification
    it can achieve. The rest is done by proximity expansion.
    Target_intensification represents how much we'd like to do by intensification given that ratio.

    :return:
    """

    # open diagnostic file if user-selected
    if c.diagnostic == 1:
        diag_str = 'c.intense_pass{0}_diag'.format(pass_number)
        diag_fn, diag_ext = os.path.splitext(eval(diag_str))
        diag_fp = '{0}_{1}{2}'.format(diag_fn, yr, diag_ext)
        diag_file = open(diag_fp, 'w')
        diag_file.write('region_id,metric_id,from_pft,to_pft,target_value\n')

    else:
        diag_file = None

    # build region_idx, metric_idx iterator
    regix_metix = reg_metric_iter(allregnumber, allregmet)

    pool = ThreadPool(len(np.unique(regix_metix)))

    pool.starmap(intense_parallel_helper,zip(regix_metix,repeat(spat_region), repeat(order_rules), repeat(allregnumber), repeat(allregmet), repeat(spat_ludata),
                          repeat(spat_landmatrix), repeat(gcam_landmatrix), repeat(yr_idx), repeat(d_regid_nm), repeat(target_change), repeat(spat_ludataharm),
                          repeat(spat_met), repeat(kernel_vector), repeat(cons_data), repeat(final_landclasses),repeat(spat_ludataharm_orig_steps), repeat(yr),
                          repeat(land_mismatch), repeat(constraint_rules), repeat(transition_rules),repeat(log), repeat(pass_number), repeat(c),repeat(diag_file)))
    # for each region
    #for index, pkg in enumerate(regix_metix):

        # unpack index vars

    pool.terminate()
    return [spat_ludataharm, spat_ludataharm_orig_steps, land_mismatch, cons_data, target_change]
