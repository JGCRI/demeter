"""
Write data to multiple file outputs.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com); Caleb J. Braun (caleb.braun@pnnl.gov)
"""
import os
import numpy as np
from scipy import io as sio
import pandas as pd
from demeter import ncdf_conversion as nc
import xarray as xr


def array_to_csv(arr, out_file):
    """
    Save numpy array as a CSV file.

    :param arr:             Input Numpy array
    :param out_file:        Output CSV file
    """
    np.savetxt(out_file, arr, delimiter=',')


def save_array(arr, out_file):
    """
    Save numpy array to NPY file.

    :param arr:             Input Numpy array
    :param out_file:        Output NPY file
    """
    np.save(out_file, arr)


def lc_timestep_csv(c, yr, final_landclasses, spat_coords, metric_id_array, gcam_regionnumber, spat_water, cellarea,
                    spat_ludataharm, metric, units='fraction', write_outputs=False, write_ncdf=False,sce="default",resolution=0.05,write_csv=False,regrid_res=0.05,
                    stitch_external=0,path_to_external="",external_scenario_PFT_name="",external_scenario=""):
    """Save land cover data for each time step as a CSV file."""

    # create out path and file name
    lc_ts_file = os.path.join(c.lu_csv_output_dir, 'landcover_{0}_timestep.csv'.format(yr))

    # create header
    hdr = "latitude,longitude,{0}_id,region_id,water,{1}".format(metric.lower(), ','.join(final_landclasses))

    # format data
    arr = np.hstack((
                        # latitude, longitude
                        spat_coords,

                        # metric id
                        np.reshape(metric_id_array, (-1, 1)),

                        # region id
                        np.reshape(gcam_regionnumber, (-1, 1)),

                        # water
                        np.reshape(spat_water / (c.spatial_resolution**2) * cellarea, (-1, 1)),

                        # final land cover classes
                        spat_ludataharm
                       ))

    # set output units
    if units == 'sqkm':
        pass

    elif units == 'fraction':
        arr[:, 4:] = np.true_divide(arr[:, 4:], arr[:, 4:].sum(axis=1, keepdims=True))

    # save to file
    if write_outputs and write_csv:
        np.savetxt(lc_ts_file, arr, fmt='%g', delimiter=',', header=hdr, comments='')

    columns = hdr.split(',')

    if write_outputs == False:
        write_ncdf = False

    if write_ncdf:
        t_joined= pd.DataFrame(data=arr, columns=columns)
        if stitch_external==1:

           path_to_sce= path_to_external+external_scenario+str(yr)+".nc"

           if os.path.isfile(path_to_sce):
               print("User has elected to stitch new scenario with current demeter scenarios. Please make sure all scenarios are stored in correct location")
               src_raster = xr.open_dataset(path_to_sce)
               target_resolution = (resolution, resolution)
               resampled = src_raster
               longitude_list = generate_scaled_coordinates_helper(-180, 180,resolution, ascending=True)

               # positive to negative scaling required for latitude output
               latitude_list = generate_scaled_coordinates_helper(-90, 90, resolution, ascending=False)


               resampled = resampled.interp(lat=latitude_list,
                               lon=longitude_list)


               sce_df = resampled.to_dataframe()

               # Reset the index to convert the multi-index into columns
               sce_df.reset_index(inplace=True)
               # Resample the raster to the target resolution
               sce_df.dropna(inplace=True)
               sce_df = sce_df[['lat', 'lon', 'value']]



               sce_df = sce_df.rename(columns={'value': 'frac','lat':'latitude',
                                               'lon':'longitude'})

               sce_df.latitude = sce_df.latitude.round(3)
               sce_df.longitude = sce_df.longitude.round(3)

               #sce_df.to_csv('C:/Projects/sce.csv', index=False)
               or_df = pd.DataFrame(data=arr, columns=columns)

               or_df.latitude = or_df.latitude.round(3)
               or_df.longitude = or_df.longitude.round(3)

               #or_df.to_csv('C:/Projects/original.csv', index=False)
               t_temp = pd.merge(or_df, sce_df, on=['latitude', 'longitude'], how='left')

               t_temp['frac'] = np.where(t_temp['frac'].isna(), 0, t_temp['frac'])
               t_temp['frac'] = np.where(t_temp['water'] >=0.25, 0, t_temp['frac'])
               t_temp['adj'] = 1 - t_temp['frac']

               t_temp.dropna(inplace=True)

               lu_file_for_col = or_df.drop(['latitude', 'longitude',external_scenario_PFT_name,'region_id','basin_id'], axis=1)
               pft_columns = lu_file_for_col.columns
               t_temp['sum_non_ext']= t_temp[pft_columns].sum(axis=1)
               t_temp[pft_columns] = t_temp[pft_columns].multiply(t_temp['adj'], axis="index")
               t_temp[pft_columns] = t_temp[pft_columns].divide(t_temp['sum_non_ext'], axis="index")

               t_temp[external_scenario_PFT_name] = t_temp['frac']
               t_joined = t_temp.drop(['frac', 'adj','sum_non_ext'], axis=1)
               t_joined = t_joined.dropna()
               t_joined = t_joined.reset_index(drop=True)
               #t_joined.to_csv("C:/Projects/sce.csv")


        x= nc.DemeterToNetcdf(scenario_name= str(sce),
                       project_name="",
                       start_year=2005,
                       end_year=2005,
                       resolution= resolution,
                       csv_input=write_csv,
                       regrid_resolution=regrid_res,
                       df=t_joined)

        x.process_output(input_file_directory=c.lu_csv_output_dir,
                         output_file_directory=c.lu_netcdf_output_dir,
                         target_year=yr)

    return pd.DataFrame(data=arr, columns=columns)


def write_transitions(s, c, step, transitions):
    """
    Save land cover transitions per time step to a CSV file.

    :param order_rules:
    :param final_landclasses:
    """

    for index in np.unique(s.order_rules):

        from_pft = np.where(s.order_rules == index)[0][0]
        from_fcs = s.final_landclasses[from_pft]

        for idx in np.arange(1, len(s.transition_rules[from_pft]), 1):

            to_pft = np.where(s.transition_rules[from_pft, :] == idx)[0][0]
            to_fcs = s.final_landclasses[to_pft]

            hdr = 'latitude,longitude,region_id,metric_id,sqkm_change'

            # create out file name
            f = os.path.join(c.transition_tabular_dir, 'lc_transitons_{0}_to_{1}_{2}.csv'.format(from_fcs, to_fcs, step))

            # create data array
            arr = np.hstack((
                s.spat_coords, # latitude, longitude
                np.reshape(s.spat_region, (-1, 1)),
                np.reshape(s.spat_aez, (-1, 1)),
                np.reshape(transitions[:, to_pft, from_pft], (-1, 1))))

            # save array as text file
            np.savetxt(f, arr, fmt='%g', delimiter=',', header=hdr, comments='')


def to_netcdf_yr(spat_lc, map_idx, lat, lon, resin, final_landclasses, yr, model, out_file):
    """
    Build a NetCDF file for each time step that contains the gridded fraction
    of land cover for each land class.

    :param spat_lc:                 An array of gridded data as fraction land cover (n_grids, n_landclasses)
    :param map_idx:                 An array of cell index positions for spatially mapping the gridded data (n_grids, n_landclasses)
    :param lat:                     An array of latitude values for mapping (n)
    :param lon:                     An array of longitude values for mapping (n)
    :param resin:                   The input spatial resolution in geographic degrees (float)
    :param final_landclasses:       An array of land classes (n_classes)
    :param yr:                      The target time step (int)
    :param model:                   The name of the model running (str)
    :param out_file:                A full path string of the output file with extension (str)
    :return:                        A NetCDF classic file.
    """

    # create NetCDF file
    with sio.netcdf_file(out_file, 'w') as f:

        # add scenario
        f.history = 'test file'

        # create dimensions
        f.createDimension('lat', len(lat))
        f.createDimension('lon', len(lon))
        f.createDimension('pft', len(final_landclasses))
        f.createDimension('nv', 2)

        # create variables
        lts = f.createVariable('lat', 'f4', ('lat',))
        lns = f.createVariable('lon', 'f4', ('lon',))
        lcs = f.createVariable('pft', 'i', ('pft',))

        lc_frac = f.createVariable('landcoverfraction', 'f8', ('pft', 'lat', 'lon',))

        # create metadata
        lts.units = 'degrees_north'
        lts.standard_name = 'latitude'
        lns.units = 'degrees_east'
        lns.standard_name = 'longitude'
        lcs.description = 'Land cover class'

        lc_frac.units = 'fraction'
        lc_frac.scale_factor = 1.
        lc_frac.add_offset = 0.
        lc_frac.projection = 'WGS84'
        lc_frac.description = 'Fraction land cover for {0} at {1} degree.'.format(yr, resin)
        lc_frac.comment = 'See scale_factor (divide by 100 to get percentage, offset is zero)'
        lc_frac.title = 'Downscaled land use projections at {0} degree, downscaled from {1}'.format(resin, model)

        # assign data
        lts[:] = lat
        lns[:] = lon
        lcs[:] = range(1, len(final_landclasses) + 1)

        # set missing value to -1
        lc_frac.missing_value = -1.

        for pft in range(0, len(final_landclasses), 1):

            # create land use matrix and populate with -1
            pft_mat = np.zeros(shape=(len(lat), len(lon))) - 1

            # extract base land use data for the target PFT
            slh = spat_lc[:, pft]

            # assign values to matrix
            pft_mat[np.int_(map_idx[0, :]), np.int_(map_idx[1, :])] = slh

            # set negative values to -1
            pft_mat[pft_mat < 0] = -1

            # assign to variable
            lc_frac[pft, :, :] = pft_mat


def to_netcdf_lc(spat_lc, lat, lon, resin, final_landclasses, years, step, model, out_dir):
    """
    Build a NetCDF file for each land class that contains the gridded fraction
    of land cover of that land class over all simulation years.

    :param spat_lc:            A 3D array representing fraction of land cover (lat, lon, fraction landclass)
    :param lat:                An array of latitude values for mapping (n)
    :param lon:                An array of longitude values for mapping (n)
    :param resin:              The input spatial resolution in geographic degrees (float)
    :param final_landclasses:  An array of land classes (n_classes)
    :param years:              A list of output years (int)
    :param step:               The current time step (int)
    :param model:              The name of the model running (str)
    :param out_dir:            A full path string of the output directory (str)
    :return:                   A NetCDF classic file.
    """

    temp_file_prefix = 'tmp_lc_'
    out_file_prefix = 'lc_yearly_'

    # just save yearly data until the final year
    if step != years[-1]:
        np.save('{0}/{1}{2}'.format(out_dir, temp_file_prefix, step), spat_lc)
        return

    # at the final year, gather data from all temporary files into one 4D array
    # with dimensions (lat, lon, year, landclass)
    tmp_files = ['{0}/{1}'.format(out_dir, f) for f in os.listdir(out_dir) if 'tmp_lc_' in f]
    lc_yearly = [np.load(f) for f in tmp_files]
    lc_yearly = np.stack(lc_yearly + [spat_lc], 2)

    # set negative values to -1
    lc_yearly[lc_yearly < 0] = -1

    # remove temporary files
    for tf in tmp_files:
        os.remove(tf)

    # output NetCDF file for each land class over all years
    for lc_index, lc in enumerate(final_landclasses):

        out_fname = '{0}/{1}{2}.nc'.format(out_dir, out_file_prefix, lc)

        # create NetCDF file
        with sio.netcdf_file(out_fname, 'w') as f:

            # create dimensions
            f.createDimension('lat', len(lat))
            f.createDimension('lon', len(lon))
            f.createDimension('time', len(years))

            # create variables
            lts = f.createVariable('lat', 'f4', ('lat',))
            lns = f.createVariable('lon', 'f4', ('lon',))
            times = f.createVariable('time', 'i4', ('time',))

            lc_frac = f.createVariable('landcoverfraction', 'f8', ('lat', 'lon', 'time'))

            # create metadata
            lts.units = 'degrees_north'
            lts.standard_name = 'latitude'
            lns.units = 'degrees_east'
            lns.standard_name = 'longitude'
            times.description = 'years'

            lc_frac.units = 'fraction'
            lc_frac.scale_factor = 1.
            lc_frac.add_offset = 0.
            lc_frac.projection = 'WGS84'
            lc_frac.description = 'Fraction land cover for {0} at {1} degree.'.format(lc, resin)
            lc_frac.comment = 'See scale_factor (divide by 100 to get percentage, offset is zero)'
            lc_frac.title = 'Downscaled land use projections at {0} degree, downscaled from {1}'.format(resin, model)

            lc_frac.missing_value = -1.

            # Add data to netcdf object
            lts[:] = lat
            lns[:] = lon
            times[:] = years
            lc_frac[:] = lc_yearly[:, :, :, lc_index]


def arr_to_ascii(arr, r_ascii, xll=-180, yll=-90, cellsize=0.25, nodata=-9999):
    """
    Convert a numpy array to an ASCII raster.

    :@param arr:            2D array
    :@param r_ascii:        Full path to outfile with extension
    :@param xll:            Longitude coordinate for lower left corner
    :@param yll:            Latitude coordinate for lower left corner
    :@param cellsize:       Cell size in geographic degrees
    :@param nodata:         Value representing NODATA
    """

    # get number of rows and columns of array
    nrows = arr.shape[0]
    ncols = arr.shape[1]

    # create ASCII raster file
    with open(r_ascii, 'w') as rast:

        # write header
        rast.write('ncols {}\n'.format(ncols))
        rast.write('nrows {}\n'.format(nrows))
        rast.write('xllcorner {}\n'.format(xll))
        rast.write('yllcorner {}\n'.format(yll))
        rast.write('cellsize {}\n'.format(cellsize))
        rast.write('nodata_value {}\n'.format(nodata))

        # write array
        np.savetxt(rast, arr, fmt='%.15g')


def max_ascii_rast(arr, out_dir, step, alg='max', nodata=-9999, xll=-180, yll=-90, cellsize=0.25):
    """
    Return the land class index containing the maximum value in the array axis.

    NOTE:
    Replace NaN with your nodata value.
    If all classes 0, then -9999
    If multiple classes have the same max value, get class with largest index

    :@param arr:            3D array (landclass, col, row)
    :@param alg:            Algorithm to extract the land class index from values
    :@param out_rast:       Full path to outfile with extension
    :@param xll:            Longitude coordinate for lower left corner
    :@param yll:            Latitude coordinate for lower left corner
    :@param cellsize:       Cell size in geographic degrees
    :@param nodata:         Value representing NODATA
    """
    # create out path and file name for the output file
    ascii_max_dir = os.path.join(out_dir, 'ascii_max_raster')

    # create output dir if it does not exist
    if os.path.isdir(ascii_max_dir):
        pass
    else:
        os.mkdir(ascii_max_dir)

    # create empty ascii grid array
    ascii_grd = np.zeros(shape=(arr.shape[2], arr.shape[0], arr.shape[1]))

    for x in range(arr.shape[2]):
        ascii_grd[x, :, :] = arr[:, :, x]

    out_rast = os.path.join(ascii_max_dir, 'lc_maxarea_{0}.asc'.format(step))

    # create a mask of where values are NaN for all land class indices
    lc_all_nan = np.all(np.isnan(ascii_grd), axis=0)

    # convert array by selection type
    if alg == 'max':
        # Reverse the array before finding the max. This is done because we
        # want the class with the largest index, however np.nanargmax() returns
        # the first (smallest) index it comes across.
        arr_rev = ascii_grd[::-1]

        # replace NaN with zero
        arr_rev = np.nan_to_num(arr_rev)

        # get land class index containing the max value (ignoring NaNs)
        arr_max = np.nanargmax(arr_rev, axis=0)

        # flip the indices back to represent their position in the original array
        final_arr = (ascii_grd.shape[0] - 1) - arr_max

    elif alg == 'min':
        arr_rev = ascii_grd[::-1]
        arr_rev[np.where(np.isnan(arr_rev))] = np.inf # replace NaN with inf
        arr_min = np.nanargmin(arr_rev, axis=0)
        final_arr = (ascii_grd.shape[0] - 1) - arr_min

    else:
        raise ValueError('Value "{}" for parameter "alg" not a valid option'.format(alg))

    # replace indices where all values were NaN with nodata value
    final_arr = np.where(lc_all_nan, nodata, final_arr)

    # create output raster
    arr_to_ascii(final_arr, out_rast, xll=-xll, yll=-yll, cellsize=cellsize, nodata=nodata)

def generate_scaled_coordinates_helper(coord_min: float,
                                    coord_max: float,
                                    res: float,
                                    ascending: bool = True,
                                    decimals: int = 3) -> np.array:
        """Generate a list of evenly-spaced coordinate pairs for the output grid based on lat, lon values.

        :param coord_min:                   Minimum coordinate in range.
        :type coord_min:                    float

        :param coord_max:                   Maximum coordinate in range.
        :type coord_max:                    float

        :param ascending:                   Ascend coordinate values if True; descend if False
        :type ascending:                    bool

        :param decimals:                    Number of desired decimals to round to.
        :type decimals:                     int

        :returns:                           Array of coordinate values.

        """

        # distance between centroid and edge of grid cell
        center_spacing = res / 2

        if ascending:
            return np.arange(coord_min + center_spacing, coord_max, res).round(decimals)

        else:
            return np.arange(coord_max - center_spacing, coord_min, -res).round(decimals)