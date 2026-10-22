"""
The total area in the gridded base layer data may not be equal to the projected data due to water inclusion,
or boundary resolution, etc.  Therefore, adjustments must be make to the projected data since there is no way of
allocating additional land.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)
"""

import numpy as np


def reg_metric_zip(allregnumber, allregmetric):
    """
    Combine every region index to every contained metric index into a list of lists [region_ix, metric_ix] .

    :param allregnumber:
    :param allregmetric:
    :return:
    """
    l = []
    for index, item in enumerate(allregnumber):
        for idx, i in enumerate(allregmetric[index]):
            l.append([index, idx])

    return l


def yr_reg_metric_zip(yr_list, r_m_list):
    """
    Create list of lists containing [yr, region_ix, metric_ix].

    :param yr_list:
    :param r_m_list:
    :return:
    """

    l = []
    for index, yr in enumerate(yr_list):
        for reg_ix, metric_ix in r_m_list:
            l.append([index, reg_ix, metric_ix])

    return l


def _base_land_area(spat_regaezarea, reg, aez, spat_aez, allregaez, spat_region, allregnumber, spat_ludata):
    """
    Populate array representing the sum of grid cell area minus water bodies from base land use data.

    :return:
    """

    # get a list of every index position where the where statement is true
    indaezreg = np.where((spat_aez == allregaez[reg][aez]) & (spat_region == allregnumber[reg]))[0]

    # add base data sum of selected data to array
    spat_regaezarea[reg, allregaez[reg][aez] - 1] = np.sum(spat_ludata[indaezreg])


def _harmonize(gcam_ludata, gcam_aez, allregaez, gcam_regionnumber, allregnumber, reg, aez, yr, spat_regaezarea,
              gcam_regaezarea, areacoef, gcam_regaezareaharm):
    """
    Adust GCAM land area for each GCAM PFT and region, metric so the total GCAM land area is equal to the base layer
    land use data.  To do this, a harmonization coefficient is applied to all land types (e.g., if area needs to
    be reduced by 10%, all GCAM PFT are as well.

    :return:
    """

    # sum all GCAM land use data matching the target region and year
    gcam_area = np.sum(gcam_ludata[(gcam_aez == allregaez[reg][aez]) & (gcam_regionnumber == allregnumber[reg]), yr])

    # added summed data to array
    gcam_regaezarea[reg, allregaez[reg][aez] - 1, yr] = gcam_area

    # if there is area associated with the region, metric, year then...
    if gcam_area > 0:

        # calculate the harmonization coefficient for land types (ratio of base land use data over the GCAM area
        #   for the target region, metric
        harm_coef = spat_regaezarea[reg, allregaez[reg][aez] - 1] / gcam_regaezarea[reg, allregaez[reg][aez] - 1, yr]
        areacoef[reg, allregaez[reg][aez] - 1, yr] = harm_coef

        # apply harmonization coefficient to the GCAM land use area array to correct the existing data
        corrected_area = gcam_ludata[(gcam_aez == allregaez[reg][aez]) & (gcam_regionnumber == allregnumber[reg]), yr] \
                         * areacoef[reg, allregaez[reg][aez] - 1, yr]
        gcam_ludata[(gcam_aez == allregaez[reg][aez]) & (gcam_regionnumber == allregnumber[reg]), yr] = corrected_area

        # apply the corrected summed GCAM land use area per region, metric, and yr to the output array
        s = np.sum(gcam_ludata[(gcam_aez == allregaez[reg][aez]) & (gcam_regionnumber == allregnumber[reg]), yr])
        gcam_regaezareaharm[reg, allregaez[reg][aez] - 1, yr] = s


def reconcile(allreg, allaez, allregnumber, allregaez, spat_aez, spat_region, spat_ludata, user_years, gcam_ludata,
              gcam_aez, gcam_regionnumber):
    """
    The total area in the gridded base layer data may not be equal to the original GCAM data due to water inclusion,
    or boundary resolution, etc.  Therefore, adjustments must be make to the GCAM data since there is no way of
    allocating additional land.

    :param allreg:
    :param allaez:
    :param allregnumber:
    :param allregaez:
    :param spat_aez:
    :param spat_region:
    :param spat_ludata:
    :return:
    """

    # get array lengths
    l_allreg = len(allreg)
    l_allaez = len(allaez)
    l_years = len(user_years)

    # create zero array to house data
    spat_regaezarea = np.zeros((l_allreg, l_allaez))

    # create zero array to house GCAM land use area per region, metric, and year
    gcam_regaezarea = np.zeros((l_allreg, l_allaez, l_years))

    # create zero array to house the harmonization coefficient for each region, metric, and year
    areacoef = np.zeros((l_allreg, l_allaez, l_years))

    # create zero array to house corrected GCAM land use area sum per region, metric, and year
    gcam_regaezareaharm = np.zeros((l_allreg, l_allaez, l_years))

    # create list of [ region_idx, metric_idx ]
    ixr_idm = reg_metric_zip(allregnumber, allregaez)

    # create list of [ year_idx, region_idx, metric_idx ]
    ixy_ixr_ixm = yr_reg_metric_zip(user_years, ixr_idm)

    # populate array containing the sum of cell area minus water bodies from base layer land use data
    for reg, aez in ixr_idm:

        # populate array
        _base_land_area(spat_regaezarea, reg, aez, spat_aez, allregaez, spat_region, allregnumber, spat_ludata)

    # adjust GCAM land use area to match the base layer area assumptions
    for yr, reg, aez in ixy_ixr_ixm:

        # apply harmonization
        _harmonize(gcam_ludata, gcam_aez, allregaez, gcam_regionnumber, allregnumber, reg, aez, yr, spat_regaezarea,
                  gcam_regaezarea, areacoef, gcam_regaezareaharm)

    return [spat_regaezarea, gcam_regaezarea, areacoef, gcam_regaezareaharm, ixr_idm, ixy_ixr_ixm, gcam_ludata]