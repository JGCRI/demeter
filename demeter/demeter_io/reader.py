"""
Read and format input data.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""


import os
import logging

import numpy as np
import pandas as pd
import gcamreader


class ValidationException(Exception):
    """Validation exception for error in runtime test."""
    def __init__(self, *args):
        Exception.__init__(self, *args)


def to_dict(f, header=False, delim=',', swap=False, value_col=1):
    """Return a dictionary of key: value pairs.  Supports only key to one value.

    :param f:           Full path to input file
    :param header:      If header exists True, else False (default)
    :param delim:       Set delimiter as string; default is comma
    :param swap:        Change the order of the key, value pair
    :param value_col:   Column index of dict values (or keys if swap is True)

    :return:            Key: value pair dictionary

    """

    d = {}
    with open(f) as get:
        for idx, line in enumerate(get):

            # strip returns and split line by delimiter
            item = line.strip().split(delim)

            # skip header if exists
            if header is True and idx == 0:
                continue

            # add key: value pair to dict
            if swap:
                d[item[value_col]] = item[0]
            else:
                d[item[0]] = item[value_col]

    return d


def to_list(f, header=True, delim=','):
    """
    Retuns values as a list of integers where first column is row names and there
    is only one value column.

    :param f:           Full path to input file
    :param header       If header exists True, else False (default)
    :param delim:       Set delimiter as string; default is comma
    :return:            List of integers
    """

    l = []
    with open(f) as get:
        for idx, line in enumerate(get):

            # skip header if exists
            if header is True and idx == 0:
                continue

            # strip returns and split line by delimiter
            item = line.strip().split(delim)

            # append index 1 items
            l.append(int(item[1]))

    return l


def read_allocation_data(f, lc_col, output_level=3, delim=','):
    """Converts an allocation file to a numpy array.  Returns final land cover class and target
    land cover class names as lists.

    :param f:               Input allocation file with header
    :param lc_col:          Target land cover field name in header (located a zero index)
    :param output_level     If 3 all variables will be returned (default); 2, target lcs list and array; 3, array
    :param delim:           Delimiter type; default is comma

    :return:                [0] List of final land cover classes
                            [1]list of target land cover classes
                            [2]  numpy array of allocation values
    """

    # make target land cover field name lower case
    col = lc_col.lower()

    # check for empty file; if blank return empty array
    if os.stat(f).st_size > 0:

        # import file as a pandas dataframe
        df = pd.read_csv(f, delimiter=delim)

        # rename all columns as lower case
        df.columns = [c.lower() for c in df.columns]

        # extract target land cover classes as a list; make lower case
        target_land_classes = df[col].str.lower().tolist()

        # extract final land cover classes as a list; remove target land cover field name
        final_land_classes = [i for i in df.columns if i != col]

        # extract target land cover values only from the dataframe and create Numpy array
        land_cover_array = df[final_land_classes].values

        if output_level == 3:
            return final_land_classes, target_land_classes, land_cover_array

        elif output_level == 2:
            return target_land_classes, land_cover_array

        elif output_level == 1:
            return land_cover_array

    else:
        return list(), np.empty(shape=0, dtype=np.float)


def _check_constraints(allocate, actual):
    """Checks to see if all land classes that are in the projection file are accounted for in the allocation file.

    :param allocate:            land classes from the allocation file
    :param actual:              land classes from the projection file

    """

    # make lower case
    act = [i.lower() for i in actual]
    alloc = [x.lower() for x in allocate]

    # create a list of elements not accounted for
    act_extra = np.setdiff1d(alloc, act)
    alloc_extra = np.setdiff1d(act, alloc)

    # get lengths
    l_act = len(act_extra)
    l_alloc = len(alloc_extra)

    # if there are extra allocation land classes not in the projected file, and vice versa
    if (l_alloc > 0) and (l_act > 0):
        m1 = "Land classes in allocation file but not in projected model data:  {0}".format(alloc_extra)
        m2 = "Land classes in projected model but not in allocation file:  {0}".format(act_extra)
        logging.warning(m1)
        logging.warning(m2)

    elif (l_alloc > 0) and (l_act == 0):
        m1 = "Land classes in allocation file but not in projected model data:  {0}".format(alloc_extra)
        logging.warning(m1)

    elif (l_alloc == 0) and (l_act > 0):
        m2 = "Land classes in projected model but not in allocation file:  {0}".format(act_extra)
        logging.warning(m2)


def _get_steps(df, start_step, end_step):
    """Create a list of projected time steps from the header that are within the user specified range

    :param df:                  Projected data, data frame
    :param start_step:          First time step value
    :param end_step:            End time step value

    :return:                    List of target steps

    """

    l = []
    for i in df.columns:
        try:
            y = int(i)
            print(y)
            if start_step <= y <= end_step:
                l.append(y)
        except ValueError:
            pass

    return l


def read_gcam_land(db_path, db_file, f_queries, d_basin_name, subreg, crop_water_src):
    """Query GCAM database for irrigated land area per region, subregion, and crop type.

    :param db_path:         Full path to the input GCAM database
    :param f_queries:       Full path to the XML query file
    :param d_basin_name:    A dictionary of 'basin_glu_name' : basin_id
    :param subreg:          Agg level of GCAM database: either AEZ or BASIN
    :param crop_water_src:  Filter for crop type: one of IRR, RFD, or BOTH

    :return:                A pandas DataFrame containing region, subregion,
                            crop type, and irrigated area per year in thousands km2

    """

    # instantiate GCAM db
    conn = gcamreader.LocalDBConn(db_path, db_file, suppress_gabble=False)

    # get queries
    q = gcamreader.parse_batch_query(f_queries)

    # assume target query is first in query list
    land_alloc = conn.runQuery(q[0])

    # split 'land-allocation' column into components
    if subreg == 'AEZ':
        raise ValueError("Using AEZs are no longer supported with `gcamreader`")

    elif subreg == 'BASIN':
        # expected format: landclass_basin-glu-name_USE_management
        cnames = ['landclass', 'metric_id', 'use', 'mgmt']

        # clean data: simplify biomass_type to just 'biomass'; temporarily
        # change Root_Tuber to RootTuber so we can split on underscores
        land_alloc['land-allocation'].replace(r'^biomass_[^_]*_', r'biomass_', regex=True, inplace=True)
        land_alloc['land-allocation'] = land_alloc['land-allocation'].str.replace('Root_Tuber', 'RootTuber')
        land_alloc[cnames] = land_alloc['land-allocation'].str.split('_', expand=True)
        land_alloc['landclass'] = land_alloc['landclass'].str.replace('RootTuber', 'Root_Tuber')

        land_alloc['metric_id'] = land_alloc['metric_id'].map(d_basin_name)
        land_alloc.drop('mgmt', axis=1, inplace=True)

    # filter out irrigated or rainfed crops, as specified in the config file
    if crop_water_src != 'BOTH':
        land_alloc = land_alloc[land_alloc['use'] == crop_water_src]

    # drop unused columns
    land_alloc.drop(['Units', 'scenario', 'land-allocation', 'use'], axis=1, inplace=True)

    # sum hi and lo management allocation (and biomass_type)
    land_alloc = land_alloc.groupby(['region', 'landclass', 'metric_id', 'Year']).sum()
    land_alloc.reset_index(inplace=True)

    # convert shape
    piv = pd.pivot_table(land_alloc, values='value',
                         index=['region', 'landclass', 'metric_id'],
                         columns='Year', fill_value=0)
    piv.reset_index(inplace=True)
    piv['metric_id'] = piv['metric_id'].astype(np.int64)
    piv.columns = piv.columns.astype(str)

    return piv


def read_gcam_file(gcam_data, gcam_landclasses, start_yr, end_yr, timestep, scenario, region_dict, agg_level, metric_seq,
                   area_factor=1000, logger=None):
    """
    Read and process the GCAM land allocation output file.

    :param gcam_data:           GCAM land allocation file or data frame from gcamreader
    :param name_col:            Field name of the column containing the region and either AEZ or basin number
    :param metric:              AEZ or Basin
    :param start_yr:            User-defined GCAM start year to process from configuration file
    :param end_yr:              User-defined GCAM end year to process from configuration file
    :param scenario:            GCAM scenario name contained in file that the user wishes to process; set in config.ini
    :param region_dict:         The reference dictionary for GCAM region_name: region_id
    :param metric_seq:          An ordered list of expected metric ids
    :param area_factor:         The factor that will be a multiplier to the land use area that is in thousands km
    :return:                    A list of the following (represents the target user-defined scenario):
                                    user_years:             a list of target GCAM years as int
                                    gcam_ludata:            Numpy array of land use area per row per year
                                    gcam_metric:            Numpy array of AEZ or Basin numbers per row
                                    gcam_landname:          Numpy array of the GCAM land use name per row
                                    gcam_regionnumber:      Numpy array of GCAM region numbers per row
                                    allreg:                 Numpy array of unique region names
                                    allregnumber:           Numpy array of unique region numbers
                                    allregaez:              List of lists, metric ids per region

    """

    # if land allocation data is not already a DataFrame, read GCAM output file and skip title row
    gdf = gcam_data if isinstance(gcam_data, pd.DataFrame) else pd.read_csv(gcam_data)

    # make sure all land classes in the projected file are in the allocation file and vice versa

    _check_constraints(gcam_landclasses, gdf['landclass'].tolist())

    # assign user-defined scenario to data frame
    gdf['scenario'] = scenario

    # check for extractor output which makes region ... gcam_region_name
    try:
        gdf['region'] = gdf['gcam_region_name']
    except KeyError:
        pass

    # create a list of GCAM years from header that are within the user specified year range
    model_year_list_int = list(range(start_yr, end_yr + timestep, timestep))

    # create land use area per year array converted from thousands km using area_factor
    model_year_list_str = [str(yr) for yr in model_year_list_int]

    gcam_data_array_km = gdf[model_year_list_str].values * area_factor

    # create field for land class all lower case
    gdf['gcam_landname'] = gdf['landclass'].apply(lambda x: x.lower())

    # get a list of basin ids that are in the master list but not the projected data
    unique_basin_list = np.sort(gdf['metric_id'].unique())
    basins_not_in_prj = sorted(list(set(metric_seq) - set(unique_basin_list)))

    # create dictionary to look up metric id to its index to act as a proxy for non-sequential values
    sequence_metric_dict = {i: ix+1 for ix, i in enumerate(gdf['metric_id'].sort_values().unique())}

    max_prj_metric = max([sequence_metric_dict[k] for k in sequence_metric_dict.keys()]) + 1

    for i in basins_not_in_prj:
        sequence_metric_dict[i] = max_prj_metric
        max_prj_metric += 1

    # create field for metric id that has sequential metric ids
    gdf['gcam_metric'] = gdf['metric_id'].map(lambda x: sequence_metric_dict[x])

    # check field for GCAM region number based off of region name; if region name is None, assign 1
    ck_reg = gdf['region'].unique()
    if (len(ck_reg)) == 1 and (ck_reg[0] == 1):
        gdf['gcam_regionnumber'] = 1
    else:
        gdf['gcam_regionnumber'] = gdf['region'].map(lambda x: int(region_dict[x]))

    # create an array of AEZ or Basin positions
    gcam_metric  = gdf['gcam_metric'].values

    # create an array of AEZ or Basin ids; formerly gcam_aez; this has the original metric values - not sequential
    metric_id_array = gdf['metric_id'].values

    # create an array of projected land use names
    gcam_landname = gdf['gcam_landname'].values

    # create an array of GCAM region numbers
    gcam_regionnumber = gdf['gcam_regionnumber'].values

    # create a list of GCAM regions represented
    l_allreg = gdf['region'].unique().tolist()

    # create a list of all GCAM region numbers represented
    l_allregnumber = gdf['gcam_regionnumber'].unique().tolist()

    # Add Taiwan region id (30) and region name 'Taiwan' as a part of China which is how GCAM constructs its land use
    #   data; the value will be added to China later in the code and is only added for computation purposes and will
    #   have no associated calculation.
    if agg_level == 2:
        l_allreg.append('Taiwan')
        l_allregnumber.append(30)

    # convert lists to array and sort
    allreg = np.array(l_allreg)
    allregnumber = np.array(l_allregnumber)
    allreg.sort()
    allregnumber.sort()
    allmetric = np.unique(gcam_metric)

    # create a list of lists of basin ids per region; add blank list for Taiwan if running GCAM REGION-AEZ
    xdf = gdf.groupby('gcam_regionnumber')['gcam_metric'].apply(list)
    allregaez = xdf.apply(lambda x: list(np.unique(x))).tolist()

    # log the number of regions and metric_ids
    logger.info('Number of regions from projected file:  {0}'.format(len(allregnumber)))
    logger.info('Number of basins from projected file:  {0}'.format(len(allmetric)))

    # add Taiwan region space holder if aggregated by GCAM region
    if agg_level == 2:
        taiwan_idx = np.where(allreg == 'Taiwan')[0][0]-1
        allregaez.insert(taiwan_idx, [])

    return [model_year_list_int, gcam_data_array_km, gcam_metric, gcam_landname, gcam_regionnumber, allreg, allregnumber, allregaez,
            allmetric, metric_id_array, sequence_metric_dict]


def read_base(config, observed_landclasses, sequence_metric_dict, metric_seq, region_seq, logger=None):
    """Read and process base layer land cover file.

    :param config:                           Configuration object
    :param observed_landclasses:            A list of land classes represented in the observed data
    :param sequence_metric_dict:        A dictionary of projected metric ids to their original id
    :param metric_seq:                  An ordered list of expected metric ids
    :param region_seq:                  An ordered list of expected region ids

    """

    df = pd.read_csv(config.observed_lu_file, compression='infer')

    # rename columns as lower case
    df.columns = [i.lower() for i in df.columns]

    # create array with only spatial land cover values
    try:
        spat_ludata = df[observed_landclasses].values
    except KeyError as e:
        logger.error('Fields are listed in the spatial allocation file that do not exist in the base layer.')
        logger.error(e)

    # create array of latitude, longitude coordinates
    try:
        spat_coords = df[['latcoord', 'loncoord']].values
    except KeyError:
        spat_coords = df[['latitude', 'longitude']].values

    # create array of metric (AEZ or basin id) per region; naming convention (regionmetric); formerly spat_aezreg
    try:
        spat_metric_region = df['regaez'].values
    except:
        spat_metric_region = None

    # create array of grid ids
    spat_grid_id = df[config.observed_id_field].values

    # create array of water areas
    try:
        spat_water = df['water'].values
    except KeyError:
        logger.warning('Water not represented in base layer.  Representing water as 0 percent of grid.')
        spat_water = np.zeros_like(spat_grid_id)

    spat_region = df['region_id'].values
    spat_metric = df['{0}_id'.format(config.metric)].values

    # ensure that the observed data represents all expected region ids
    unique_spat_region = np.unique(spat_region)
    valid_region_test = set(region_seq) - set(unique_spat_region)

    if len(valid_region_test) > 0:
        logger.error('Observed spatial data must have all regions represented.')
        raise ValidationException

    # ensure that the observed data represents all expected metric ids
    unique_spat_metric = np.unique(spat_metric)
    valid_metric_test = set(metric_seq) - set(unique_spat_metric)

    if len(valid_metric_test) > 0:
        logger.error('Observed spatial data must have all {}_id represented.'.format(config.metric))
        raise ValidationException

    # account for 0 designation in observed data for unclassified
    if 0 in np.unique(spat_metric):
        sequence_metric_dict[0] = 0

    # adjust the numbering of metrics in the observed data

    spat_metric = np.vectorize(sequence_metric_dict.get)(spat_metric)

    # get the total number of grid cells
    ngrids = len(df)

    # change spatial region value for Taiwan from 30 to 11 for China to account for GCAM allocation procedure
    if config.model.lower() == 'gcam' and config.agg_level == 2:
        spat_region[spat_region == 30] = 11

    # cell area from lat: lat_correction_factor * (lat_km at equator * lon_km at equator) * (resolution squared) = sqkm
    cellarea = np.cos(np.radians(spat_coords[:, 0])) * (111.32 * 110.57) * (config.spatial_resolution**2)

    # create an array with the actual percentage of the grid cell included in the data; some are cut by AEZ or Basin
    #   polygons others have no-data in land cover
    celltrunk = (np.sum(spat_ludata, axis=1) + spat_water) / (config.spatial_resolution ** 2)

    # adjust land cover area based on the percentage of the grid cell represented
    spat_ludata = spat_ludata / (config.spatial_resolution ** 2) * np.transpose([cellarea, ] * len(observed_landclasses))

    return [spat_ludata, spat_water, spat_coords, spat_metric_region, spat_grid_id, spat_metric, spat_region, ngrids,
            cellarea, celltrunk, sequence_metric_dict]


def to_array(f, target_index, delim=','):
    """
    Read file to Numpy array and slice out a single field by the target index.

    :param f:
    :param target_index:
    :param delim:
    :return:
    """

    # read as array
    arr = np.loadtxt(f, delimiter=delim)

    # select column by index
    c = arr[:, target_index]

    return c


def csv_to_array(f):
    """
    Read CSV file to NumPy array

    :param f:             Full path to input file
    :return:              array
    """
    return np.genfromtxt(f, delimiter=',')