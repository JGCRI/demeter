"""
Read and format input data.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""
import numpy as np
import os
import pandas as pd


class ValidationException(Exception):
    """Validation exception for error in runtime test."""
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)


def to_dict(f, header=False, delim=',', swap=False):
    """
    Return a dictionary of key: value pairs.  Supports only key to one value.

    :param f:           Full path to input file
    :param header:      If header exists True, else False (default)
    :param delim:       Set delimiter as string; default is comma
    :param swap:        Change the order of the key, value pair
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
                d[item[1]] = item[0]
            else:
                d[item[0]] = item[1]

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
    with open(f, 'rU') as get:
        for idx, line in enumerate(get):

            # skip header if exists
            if header is True and idx == 0:
                continue

            # strip returns and split line by delimiter
            item = line.strip().split(delim)

            # append index 1 items
            l.append(int(item[1]))

    return l


def read_alloc(f, lc_col, output_level=3, delim=','):
    """
    Converts an allocation file to a numpy array.  Returns final land cover class and target
    land cover class names as lists.

    :param f:               Input allocation file with header
    :param lc_col:          Target land cover field name in header (located a zero index)
    :param output_level     If 3 all variables will be returned (default); 2, target lcs list and array; 3, array
    :param delim:           Delimiter type; default is comma
    :return:                List of final land cover classes, list of target land cover classes,
                            numpy array of allocation values
    """

    # make target land cover field name lower case
    col = lc_col.lower()

    # check for empty file; if blank return empty array
    if os.stat(f).st_size != 0:

        # import file as a pandas dataframe
        df = pd.read_csv(f, delimiter=delim)

        # rename all columns as lower case
        df.columns = [c.lower() for c in df.columns]

        # extract target land cover classes as a list; make lower case
        tcs = df[col].str.lower().tolist()

        # extract final land cover classes as a list; remove target land cover field name
        fcs = [i for i in df.columns if i != col]

        # extract target land cover values only from the dataframe and create Numpy array
        arr = df[fcs].values

        if output_level == 3:
            return fcs, tcs, arr

        elif output_level == 2:
            return tcs, arr

        elif output_level == 1:
            return arr

    else:
        return list(), np.empty(shape=0, dtype=np.float)


def _check_constraints(log_obj, allocate, actual):
    """
    Checks to see if all land classes that are in the projection file are accounted for in the allocation file.

    :param log_obj:             logger object
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
        log_obj.warning(m1)
        log_obj.warning(m2)

    elif (l_alloc > 0) and (l_act == 0):
        m1 = "Land classes in allocation file but not in projected model data:  {0}".format(alloc_extra)
        log_obj.warning(m1)

    elif (l_alloc == 0) and (l_act > 0):
        m2 = "Land classes in projected model but not in allocation file:  {0}".format(act_extra)
        log_obj.warning(m2)


def _get_steps(df, start_step, end_step):
    """
    Create a list of projected time steps from the header that are within the user specified range

    :param df:                  Projected data, data frame
    :param start_step:          First time step value
    :param end_step:            End time step value
    :return:                    List of target steps
    """

    l = []
    for i in df.columns:
        try:
            y = int(i)
            if start_step <= y <= end_step:
                l.append(y)
        except ValueError:
            pass

    return l


def read_gcam_file(log, f, gcam_landclasses, start_yr, end_yr, scenario, region_dict, agg_level, metric_seq,
                   area_factor=1000):
    """
    Read and process the GCAM land allocation output file.

    :param f:                   GCAM land allocation file
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
    # read GCAM output file as a dataframe; skip title row
    gdf = pd.read_csv(f, header=0)

    # make sure all land classes in the projected file are in the allocation file and vice versa
    _check_constraints(log, gcam_landclasses, gdf['landclass'].tolist())

    # assign user-defined scenario to data frame
    gdf['scenario'] = scenario

    # create a list of GCAM years from header that are within the user specified year range
    user_years = _get_steps(gdf, start_yr, end_yr)

    # create land use area per year array converted from thousands km using area_factor
    target_years = [str(yr) for yr in user_years]
    gcam_ludata = gdf[target_years].values * area_factor

    # create field for land class
    gdf['gcam_landname'] = gdf['landclass'].apply(lambda x: x.lower())

    unique_metric_list = np.sort(gdf['metric_id'].unique())

    # get a list of metrics that are in the master list but not the projected data
    metric_not_in_prj = sorted(list(set(metric_seq) - set(unique_metric_list)))

    # create dictionary to look up metric id to its index to act as a proxy for non-sequential values
    sequence_metric_dict = {i: ix+1 for ix, i in enumerate(gdf['metric_id'].sort_values().unique())}

    max_prj_metric = max([sequence_metric_dict[k] for k in sequence_metric_dict.keys()]) + 1

    for i in metric_not_in_prj:
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

    # create a list of lists of AEZ or Basin ids per region; add blank list for Taiwan if running GCAM REGION-AEZ
    xdf = gdf.groupby('gcam_regionnumber')['gcam_metric'].apply(list)
    allregaez = xdf.apply(lambda x: list(np.unique(x))).tolist()

    # log the number of regions and metric_ids
    log.info('Number of regions from projected file:  {0}'.format(len(allregnumber)))
    log.info('Number of basins or AEZs from projected file:  {0}'.format(len(allmetric)))

    # add Taiwan region space holder if aggregated by GCAM region
    if agg_level == 2:
        taiwan_idx = np.where(allreg == 'Taiwan')[0][0]-1
        allregaez.insert(taiwan_idx, [])

    return [user_years, gcam_ludata, gcam_metric, gcam_landname, gcam_regionnumber, allreg, allregnumber, allregaez,
            allmetric, metric_id_array, sequence_metric_dict]


def read_base(log, c, spat_landclasses, sequence_metric_dict, metric_seq, region_seq):
    """
    Read and process base layer land cover file.

    :param c:                           Configuration object
    :param spat_landclasses:            A list of land classes represented in the observed data
    :param sequence_metric_dict:        A dictionary of projected metric ids to their original id
    :param metric_seq:                  An ordered list of expected metric ids
    :param region_seq:                  An ordered list of expected region ids
    :return:
    """
    df = pd.read_csv(c.first_mod_file)

    # rename columns as lower case
    df.columns = [i.lower() for i in df.columns]

    # create array with only spatial land cover values
    try:
        spat_ludata = df[spat_landclasses].values
    except KeyError as e:
        log.error('Fields are listed in the spatial allocation file that do not exist in the base layer.')
        log.error(e)

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
    spat_grid_id = df[c.pkey].values

    # create array of water areas
    try:
        spat_water = df['water'].values
    except KeyError:
        log.warning('Water not represented in base layer.  Representing water as 0 percent of grid.')
        spat_water = np.zeros_like(spat_grid_id)

    spat_region = df['region_id'].values
    spat_metric = df['{0}_id'.format(c.metric.lower())].values

    # ensure that the observed data represents all expected region ids
    unique_spat_region = np.unique(spat_region)
    valid_region_test = set(region_seq) - set(unique_spat_region)

    if len(valid_region_test) > 0:
        log.error('Observed spatial data must have all regions represented.')
        raise ValidationException

    # ensure that the observed data represents all expected metric ids
    unique_spat_metric = np.unique(spat_metric)
    valid_metric_test = set(metric_seq) - set(unique_spat_metric)

    if len(valid_metric_test) > 0:
        log.error('Observed spatial data must have all {}_id represented.'.format(c.metric.lower()))
        raise ValidationException

    # account for 0 designation in observed data for unclassified
    if 0 in np.unique(spat_metric):
        sequence_metric_dict[0] = 0

    # adjust the numbering of metrics in the observed data
    spat_metric = np.vectorize(sequence_metric_dict.get)(spat_metric)

    # get the total number of grid cells
    ngrids = len(df)

    # change spatial region value for Taiwan from 30 to 11 for China to account for GCAM allocation procedure
    if c.model.lower() == 'gcam' and c.agg_level == 2:
        spat_region[spat_region == 30] = 11

    # cell area from lat: lat_correction_factor * (lat_km at equator * lon_km at equator) * (resolution squared) = sqkm
    cellarea = np.cos(np.radians(spat_coords[:, 0])) * (111.32 * 110.57) * (c.resin**2)

    # create an array with the actual percentage of the grid cell included in the data; some are cut by AEZ or Basin
    #   polygons others have no-data in land cover
    celltrunk = (np.sum(spat_ludata, axis=1) + spat_water) / (c.resin ** 2)

    # adjust land cover area based on the percentage of the grid cell represented
    spat_ludata = spat_ludata / (c.resin ** 2) * np.transpose([cellarea, ] * len(spat_landclasses))

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