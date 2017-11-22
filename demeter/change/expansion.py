"""
Land use expansion algorithm.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com)
"""

import numpy as np
import os
from scipy import stats


def _convert_pft(notdone, exp_target, met_idx, pft_toconv, spat_ludataharm_sub, pft, cons_data_subpft, reg,
                trans_mat, non_exist_cells, stochastic_expansion, selection_threshold, target_change, errortol,
                diag_file, diagnostic):
    """
    Apply conversion to every qualifying PFT.

    :return:            Array of PFTs
    """
    if diagnostic == 1:
        diag_file.write('{},{},{},{},{}\n'.format(reg + 1, met_idx+1, pft, pft_toconv, exp_target))

    while notdone:
        # grid cells with both the expanding and to-convert PFT
        exist_cells = np.where(non_exist_cells & (spat_ludataharm_sub[:, pft_toconv] > 0))[0]

        # intensification constraints on grid cells, weighted by the user-input constrain weights
        cons_cells = cons_data_subpft[exist_cells, :]

        # combine and normalize constraints
        mean_cons_cells = np.nansum(cons_cells, axis=1) / np.nanmean(np.nansum(cons_cells, axis=1))
        mean_cons_cells /= np.nanmax([0.00000001, np.nanmax(mean_cons_cells)])

        # when there are no constraints because none is applied to the PFT (all NaN),
        #   then constrain is 1 (no constrain).
        mean_cons_cells[np.isnan(mean_cons_cells)] = 0.

        # combined grid-cell value of kernel density and constraints for stochastic draw of which grid cells will
        #   receive expansion
        expansion_likelihood = np.power(mean_cons_cells, 2)

        # checking for non-zero values (if all is zero, no grid-cell will ever be selected)
        if np.nanmax(expansion_likelihood) == 0:
            expansion_likelihood[:] = 1.

        # select the grid cells to expand; user defines whether to use stochastic draw or select the grid cells
        #   with the highest likelihood
        if stochastic_expansion == 1:
            drawcells = stats.binom.rvs(1, expansion_likelihood / np.nanmax(expansion_likelihood))
        else:
            drawcells = expansion_likelihood >= selection_threshold * np.nanmax(expansion_likelihood)

        # capture the result of the draw
        candidatecells = np.where(drawcells == 1)[0]

        # total area that the PFT to convert could give
        swaparea = min([exp_target, target_change[reg, met_idx, pft_toconv] * -1,
                        np.sum(spat_ludataharm_sub[exist_cells[candidatecells], pft_toconv])])
        swaparea = min([swaparea, exp_target])

        # applying the constraints, the less the constrain, the higher the fraction of potential
        #   expansion is allowed
        potexpansion = swaparea * expansion_likelihood[candidatecells] / np.sum(expansion_likelihood[candidatecells])

        # actual expansion: for each grid cell get the minimum of: potential expansion,
        #   and actual expansion
        actexpansion = np.amin([potexpansion, spat_ludataharm_sub[exist_cells[candidatecells], pft_toconv]], axis=0)

        # applying land swap between both PFTs
        spat_ludataharm_sub[exist_cells[candidatecells], pft] += actexpansion
        spat_ludataharm_sub[exist_cells[candidatecells], pft_toconv] -= actexpansion

        # update the target change values
        target_change[reg, met_idx, pft] -= np.sum(actexpansion)
        exp_target -= np.sum(actexpansion)
        target_change[reg, met_idx, pft_toconv] += np.sum(actexpansion)
        trans_mat[exist_cells[candidatecells], pft, pft_toconv] += actexpansion

        # account for target change minuscule values when evaluating notdone
        tc = round(target_change[reg, met_idx, pft_toconv], 4)

        # updating notdone: if the intensification target has been reached, or if there are no more
        #   of the PFTs to convert in the considered grid cells, the break the loop
        notdone = (exp_target > errortol) \
                  & (tc < -errortol) \
                  & (np.sum(spat_ludataharm_sub[exist_cells, pft_toconv]) > errortol) \
                  & (len(exist_cells) > 0) \
                  & (np.sum(mean_cons_cells) != len(mean_cons_cells))

        if diagnostic == 1:
            diag_file.write('{},{},{},{},{}\n'.format(reg + 1, met_idx+1, pft, pft_toconv, exp_target))

    return exp_target, target_change, trans_mat


def _expansion(diagnostic, diag_file, spat_ludataharm_sub, kernel_vector_sub, cons_data_sub_o, reg_idx, met_idx, order_rules, final_landclasses,
               errortol, constrain_rules, transition_rules, stochastic_expansion, selection_threshold, land_mismatch,
               target_change):

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
        exp_target = target_change[reg_idx, met_idx, pft]
        exp_target_const = target_change[reg_idx, met_idx, pft]

        # determine if expansion is wanted
        if exp_target <= errortol:
            # print("\nNo expansion desired for:  {0}, {1}".format(fcs, round(exp_target, 4)))
            pass

        else:
            # print("\nExpansion desired for:  {0}, {1}".format(fcs, round(exp_target, 4)))

            # retrieve expansion constraints for the PFT (e.g., soil quality, protection status, etc.)
            cons_rules_pft = constrain_rules[:, pft]

            # add kernel density to the constraints and normalize their value (all constraints are [0 1])
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
            cons_data_subpft[:, cons_rules_pft < 0] = np.ones(shape=np.shape(cons_data_sub[:, cons_rules_pft < 0])) \
                                                      + cons_data_subpft[:, cons_rules_pft < 0]

            # multiply negative constraints weight by -1 to turn it positive
            cons_rules_pft[cons_rules_pft < 0] *= -1

            # zero means that constrain does not apply to the PFT, we turn these values into NaN
            cons_data_subpft *= np.tile(cons_rules_pft, (len(spat_ludataharm_sub[:, pft]), 1))

            cons_data_subpft[:, cons_rules_pft == 0] = np.nan

            # iterate through the conversion priority rules to find other PFTs that are contracting (where expansion
            #   can occur)
            for pft_tcvo in np.arange(1, len(transition_rules[pft]), 1):

                # get the PFT to convert and its final land class name
                pft_toconv = np.where(transition_rules[pft, :] == pft_tcvo)[0][0]

                # round target change value to ensure minuscule values near zero are not counted as negative
                if round(target_change[reg_idx, met_idx, pft_toconv], 4) < 0:
                    tce = 1
                else:
                    tce = 0

                # see if PFT to convert has to be converted in the target region, metric
                notdone = (exp_target > errortol) & (tce == 1)

                # identify grid cells without expanding but to-convert PFT
                non_exist_cells = spat_ludataharm_sub[:, pft] == 0
                exist_cells = np.where(non_exist_cells & (spat_ludataharm_sub[:, pft_toconv] > 0))[0]

                # apply conversion to every qualifying PFT
                if len(exist_cells) > 0:

                    exp_target, target_change, trans_mat = _convert_pft(notdone, exp_target, met_idx, pft_toconv,
                                                                        spat_ludataharm_sub, pft, cons_data_subpft, reg_idx,
                                                                        trans_mat, non_exist_cells, stochastic_expansion,
                                                                        selection_threshold, target_change, errortol,
                                                                        diag_file, diagnostic)

        # report how much expansion that has been achieved
        achieved = exp_target_const - target_change[reg_idx, met_idx, pft]
        per_numer = land_mismatch[reg_idx, met_idx, pft] - target_change[reg_idx, met_idx, pft]
        per_denom = land_mismatch[reg_idx, met_idx, pft]

        # check for 0
        if per_numer == 0 or per_denom == 0:
            percent = 0
        else:
            percent = (per_numer / per_denom) * 100

    return spat_ludataharm_sub, target_change, trans_mat


def _reg_metric_iter(allregnumber, allregmet):
    """
    Create region, metric iterator.

    :return:            List [[reg_idx, metric_idx], n]
    """
    l = []
    for reg in range(len(allregnumber)):
        for met in range(len(allregmet[reg])):
            l.append([reg, met])

    return l


def apply_expansion(log, c, allregnumber, allregmet, spat_ludataharm, spat_region, spat_met, kernel_vector, cons_data,
                    order_rules, final_landclasses, constrain_rules, transition_rules, land_mismatch, transitions,
                    spat_ludataharm_orig_steps, target_change, yr):

    # open diagnostic file if user-selected
    if c.diagnostic == 1:
        diag_fn, diag_ext = os.path.splitext(c.expansion_diag)
        diag_fp = '{0}_{1}{2}'.format(diag_fn, yr, diag_ext)
        diag_file = open(diag_fp, 'w')
        diag_file.write('region_id,metric_id,from_pft,to_pft,target_value\n')

    else:
        diag_file = None

    # build region_idx, metric_idx iterator
    regix_metix = _reg_metric_iter(allregnumber, allregmet)

    for reg_idx, met_idx in regix_metix:

        # get region and metric from index
        regnumber = allregnumber[reg_idx]
        metnumber = allregmet[reg_idx][met_idx]
        metnum_idx = metnumber - 1

        # create data subset
        spat_ludataharm_sub = spat_ludataharm[(spat_region == regnumber) & (spat_met == metnumber)]
        kernel_vector_sub = kernel_vector[(spat_region == regnumber) & (spat_met == metnumber)]
        cons_data_sub = cons_data[(spat_region == regnumber) & (spat_met == metnumber)]

        # calculate expansion for each PFT
        exp = _expansion(c.diagnostic, diag_file, spat_ludataharm_sub, kernel_vector_sub, cons_data_sub, reg_idx,
                         metnum_idx, order_rules, final_landclasses, c.errortol, constrain_rules, transition_rules,
                         c.stochastic_expansion, c.selection_threshold, land_mismatch, target_change)

        # apply expansion and update transitions
        spat_ludataharm[(spat_region == regnumber) & (spat_met == metnumber)], target_change, trans_mat = exp
        transitions[(spat_region == regnumber) & (spat_met == metnumber), :, :] += trans_mat

    # calculate non-achieved change
    non_chg = np.sum(abs(target_change[:, :, :])) / 2.

    if non_chg > 0:
        non_chg_per = np.sum(abs(target_change[:, :, :].flatten())) / np.sum(abs(land_mismatch[:, :, :].flatten())) * 100

    else:
        non_chg_per = 0

    log.info("Total non-achieved expansion change for time step {0}:  {1} km2 ({2} %)".format(yr, non_chg, non_chg_per))

    # close file if diagnostic
    if c.diagnostic == 1:
        diag_file.close()

    return [spat_ludataharm, spat_ludataharm_orig_steps, land_mismatch, cons_data,
            transitions, target_change]