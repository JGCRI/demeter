"""
Write data to multiple file outputs.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com); Caleb J. Braun (caleb.braun@pnnl.gov)
"""
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import io as sio
import shapefile

import demeter.demeter_io.reader as rdr


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


def to_shp(c, yr, final_landclasses):
    """
    Build shapefile containing landcover per grid location.

    :param c:                       config object
    :param yr:                      target year
    :param final_landclasses:       list of final land classes
    """

    # instantiate the writer object
    w = shapefile.Writer(shapeType=shapefile.POINT)

    # field list
    fields = ['latitude', 'longitude', '{0}_id'.format(c.metric.lower()), 'region_id', 'water']

    # set schema
    w.field('latitude', 'F', decimal=10)
    w.field('longitude', 'F', decimal=10)
    w.field('{0}_id'.format(c.metric.lower()), 'C')
    w.field('region_id', 'C')
    w.field('water', 'F', decimal=10)

    # prep string for record
    s = 'w.record(lat, lon, met, reg, wat,'

    # add functional type fields and prepare record string
    for idx, fci in enumerate(final_landclasses):

        fc = fci.lower()

        w.field(fc, 'F', decimal=10)

        fields.append(fc)

        # append fields to record string
        if idx+1 < len(final_landclasses):
            s += "float(r[d['{0}']]),".format(fc)

        else:
            s += "float(r[d['{0}']]))".format(fc)

    # read landcover CSV
    with open(os.path.join(c.lc_per_step_csv, 'landcover_{0}_timestep.csv'.format(yr))) as get:

        # get the header as a list
        hdr = get.next().strip().split(',')

        # get the index locations of field to write in input file
        d = {k: hdr.index(k) for k in fields}

        for row in get:

            # row to list
            r = [i.lower() for i in row.strip().split(',')]

            lat = float(r[d['latitude']])
            lon = float(r[d['longitude']])
            met = r[d['{0}_id'.format(c.metric.lower())]]
            reg = r[d['region_id']]
            wat = float(r[d['water']])

            # add geometry to shapefile
            w.point(lon, lat)

            # add attribute to shapefile
            eval(s)

    # save output
    out_shp = os.path.join(c.lc_per_step_shp, 'landcover_{0}_timestep.shp'.format(yr))
    w.save(out_shp)


def lc_timestep_csv(c, yr, final_landclasses, spat_coords, metric_id_array, gcam_regionnumber, spat_water, cellarea,
                    spat_ludataharm, metric, units='percent'):
    """
    Save land cover data for each time step as a CSV file.
    """

    # create out path and file name
    lc_ts_file = os.path.join(c.lc_per_step_csv, 'landcover_{0}_timestep.csv'.format(yr))

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
                        np.reshape(spat_water / (c.resin**2) * cellarea, (-1, 1)),

                        # final land cover classes
                        spat_ludataharm
                       ))

    # set output units
    if units == 'sqkm':
        pass

    elif units == 'percent':
        arr[:, 4:] = np.true_divide(arr[:, 4:], arr[:, 4:].sum(axis=1, keepdims=True))

    # save to file
    np.savetxt(lc_ts_file, arr, fmt='%g', delimiter=',', header=hdr, comments='')


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
    Build a NetCDF file for each time step that contains the gridded percent
    land cover for each land class.

    :param spat_lc:                 An array of gridded data as percent land cover (n_grids, n_landclasses)
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

        lc_perc = f.createVariable('landcoverpercentage', 'f8', ('pft', 'lat', 'lon',))

        # create metadata
        lts.units = 'degrees_north'
        lts.standard_name = 'latitude'
        lns.units = 'degrees_east'
        lns.standard_name = 'longitude'
        lcs.description = 'Land cover class'

        lc_perc.units = 'percentage'
        lc_perc.scale_factor = 1.
        lc_perc.add_offset = 0.
        lc_perc.projection = 'WGS84'
        lc_perc.description = 'Percent land cover for {0} at {1} degree.'.format(yr, resin)
        lc_perc.comment = 'See scale_factor (divide by 100 to get percentage, offset is zero)'
        lc_perc.title = 'Downscaled land use projections at {0} degree, downscaled from {1}'.format(resin, model)

        # assign data
        lts[:] = lat
        lns[:] = lon
        lcs[:] = range(1, len(final_landclasses))

        # set missing value to -1
        lc_perc.missing_value = -1.

        for pft in range(0, len(final_landclasses), 1):

            # create land use matrix and populate with -1
            pft_mat = np.zeros(shape=(len(lat), len(lon))) - 1

            # extract base land use data for the target PFT
            slh = spat_lc[:, pft]

            # assign values to matrix
            pft_mat[np.int_(map_idx[0, :]), np.int_(map_idx[1, :])] = slh

            # multiply by scale factor for percentage
            pft_mat *= lc_perc.scale_factor

            # set negative values to -1
            pft_mat[pft_mat < 0] = -1

            # assign to variable
            lc_perc[pft, :, :] = pft_mat


def map_kernel_density(spatdata, kerneldata, lat, lon, pft_name, yr, out_path):
    """
    Maps kernel density computed through convolution filter

    :param spatdata:
    :param kerneldata:
    :param lat:
    :param lon:
    :param pftname:
    :param year:
    :param outpathfig:
    :param filename:
    :return:
    """
    # set map extent
    extent = [lon[0], lon[-1], lat[-1], lat[0]]

    # set up figure
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title(pft_name + ' cover:', fontsize=6)
    ax2 = fig.add_subplot(212)
    ax2.set_title(pft_name + ' kernel density:', fontsize=6)

    # set color map
    cmap = cm.get_cmap('jet')

    # stage image
    ax1.imshow(spatdata, cmap=cmap, extent=extent, interpolation='nearest', origin='upper', aspect='auto')
    ax2.imshow(kerneldata, cmap=cmap, extent=extent, interpolation='nearest', origin='upper', aspect='auto')

    # save file
    plt.savefig(os.path.join(out_path, "kernel_density_{0}_{1}.png".format(pft_name, yr)), dpi=300)

    # clear the figure so it may be used elsewhere
    fig.clf()

    # close the figure
    plt.close(fig)


def map_luc(spat_ludataharm, spat_ludataharm_orig, cellindexresin, lat, lon, final_landclasses, yr, region_file,
            country_file, out_dir, process_type):
    """
    Map land use change for each PFT.

    :return:
    """
    # create lengths for arrays
    l_lat = len(lat)
    l_lon = len(lon)
    l_fcs = len(final_landclasses)

    # set map extent
    mapextent = [lon[0], lon[-1], lat[-1], lat[0]]

    # set up arrays
    pft_orig = np.zeros((l_lat, l_lon)) * np.nan
    pft_now = np.zeros((l_lat, l_lon)) * np.nan

    # load region and country boundary files
    reg_coords = rdr.csv_to_array(region_file)
    country_coords = rdr.csv_to_array(country_file)

    for pft in range(l_fcs):

        # get name of target PFT
        pft_name = final_landclasses[pft]

        # populate change arrays
        pft_orig[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :])] = spat_ludataharm_orig[:, pft]
        pft_now[np.int_(cellindexresin[0, :]), np.int_(cellindexresin[1, :])] = spat_ludataharm[:, pft]
        pft_change = pft_now - pft_orig

        # set up figure and axes
        fig = plt.figure(figsize=(10, 14))
        ax1 = plt.subplot2grid((3, 8), (0, 0), colspan=7)
        ax1b = plt.subplot2grid((3, 8), (0, 7))
        ax2 = plt.subplot2grid((3, 8), (1, 0), colspan=7)
        ax2b = plt.subplot2grid((3, 8), (1, 7))
        ax3 = plt.subplot2grid((3, 8), (2, 0), colspan=7)
        ax3b = plt.subplot2grid((3, 8), (2, 7))

        # set titles
        ax1.set_title("{0} {1} BEFORE:".format(yr, pft_name), fontsize=10)
        ax2.set_title("{0} {1} AFTER:".format(yr, pft_name), fontsize=10)
        ax3.set_title("{0} {1} CHANGE:".format(yr, pft_name), fontsize=10)

        # color bar for before and after plots
        cmapbfaf = cm.get_cmap('YlOrBr')

        # plot before
        a1 = ax1.imshow(pft_orig, cmap=cmapbfaf, extent=mapextent, interpolation='nearest', origin='upper',
                        aspect='auto')
        ax1.plot(reg_coords[0, :], reg_coords[1, :], color='black', linestyle='-', linewidth=0.5)
        ax1.plot(country_coords[0, :], country_coords[1, :], color='0.8', linestyle='-', linewidth=0.2)
        ax1.axis(mapextent)
        fig.colorbar(a1, cax=ax1b, orientation='vertical')

        # plot after
        a2 = ax2.imshow(pft_now, cmap=cmapbfaf, extent=mapextent, interpolation='nearest', origin='upper',
                        aspect='auto')
        ax2.plot(reg_coords[0, :], reg_coords[1, :], color='black', linestyle='-', linewidth=0.5)
        ax2.plot(country_coords[0, :], country_coords[1, :], color='0.8', linestyle='-', linewidth=0.2)
        ax2.axis(mapextent)
        fig.colorbar(a2, cax=ax2b, orientation='vertical')

        # color bar for change plot
        cmapchg = cm.get_cmap('seismic')

        # plot change
        barmin = np.nanmin(pft_change) / 2.
        barmax = np.nanmax(pft_change) / 2.
        barmax = np.nanmax(abs(np.array([barmin, barmax]))) / 2.
        barmin = barmax * -1
        a3 = ax3.imshow(pft_change, vmin=barmin, vmax=barmax, cmap=cmapchg, extent=mapextent,
                        interpolation='nearest', origin='upper', aspect='auto')
        ax3.plot(reg_coords[0, :], reg_coords[1, :], color='black', linestyle='-', linewidth=0.5)
        ax3.plot(country_coords[0, :], country_coords[1, :], color='0.8', linestyle='-', linewidth=0.2)
        ax3.axis(mapextent)
        fig.colorbar(a3, cax=ax3b, orientation='vertical')

        # save file
        plt.savefig(os.path.join(out_dir, "luc_{0}_{1}_{2}.png".format(pft_name, yr, process_type)), dpi=300)

        # clear figure
        fig.clf()

        # close plot object
        plt.close(fig)


def map_transitions(s, c, step, transitions, dpi=150):
    """
    Map land cover transitions for each time step.

    :param s:
    :param c:
    """

    for index in np.unique(s.order_rules):

        from_pft = np.where(s.order_rules == index)[0][0]
        from_fcs = s.final_landclasses[from_pft]

        for idx in np.arange(1, len(s.transition_rules[from_pft]), 1):

            to_pft = np.where(s.transition_rules[from_pft, :] == idx)[0][0]
            to_fcs = s.final_landclasses[to_pft]

            # create data array
            arr = np.hstack((
                s.spat_coords, # latitude, longitude
                np.reshape(s.spat_region, (-1, 1)),
                np.reshape(s.spat_aez, (-1, 1)),
                np.reshape(transitions[:, to_pft, from_pft], (-1, 1))))

            # create map extent
            ext = [s.lon[0], s.lon[-1], s.lat[-1], s.lat[0]]

            # build nan array
            arr_t = np.zeros(shape=(len(s.lat), len(s.lon))) * np.nan

            # convert sqkm to fraction of area
            # km_to_fract = np.tile(np.cos(np.radians(s.lat)) * 111.32**2 * (180. / len(s.lat))**2, len(s.lon), 1).T

            # reshape transitions array for map
            lu = np.reshape(transitions[:, to_pft, from_pft], (-1, 1)) / np.tile(s.cellarea, (1, 1)).T

            # populate array
            arr_t[np.int_(s.cellindexresin[0, :]), np.int_(s.cellindexresin[1, :])] = lu[:, 0]

            # create figure
            fig = plt.figure(figsize=(12, 5))

            # set up axis
            ax1 = plt.subplot2grid((3, 8), (0, 0), colspan=7, rowspan=3)
            ax1.set_title('Transition from {0} to {1}:'.format(from_fcs, to_fcs), fontsize=10)
            clr = cm.get_cmap('YlOrBr')
            a1 = ax1.imshow(arr_t, cmap=clr, extent=ext, interpolation='nearest', origin='upper', aspect='auto')
            ax1.axis(ext)
            fig.colorbar(a1, orientation='vertical')

            # create out file name
            f = os.path.join(c.transiton_map_dir, 'lc_transitons_{0}_to_{1}_{2}.png'.format(from_fcs, to_fcs, step))

            # save figure
            plt.savefig(f, dpi=dpi)

            # clean up
            fig.clf()
            plt.close(fig)


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


def max_ascii_rast(arr, out_rast, alg='max', nodata=-9999, xll=-180, yll=-90, cellsize=0.25):
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

    # create a mask of where values are NaN for all land class indices
    lc_all_nan = np.all(np.isnan(arr), axis=0)

    # convert array by selection type
    if alg == 'max':
        # Reverse the array before finding the max. This is done because we
        # want the class with the largest index, however np.nanargmax() returns
        # the first (smallest) index it comes across.
        arr_rev = arr[::-1]

        # replace NaN with zero
        arr_rev = np.nan_to_num(arr_rev)

        # get land class index containing the max value (ignoring NaNs)
        arr_max = np.nanargmax(arr_rev, axis=0)

        # flip the indices back to represent their position in the original array
        final_arr = (arr.shape[0] - 1) - arr_max

    elif alg == 'min':
        arr_rev = arr[::-1]
        arr_rev[np.where(np.isnan(arr_rev))] = np.inf # replace NaN with inf
        arr_min = np.nanargmin(arr_rev, axis=0)
        final_arr = (arr.shape[0] - 1) - arr_min

    else:
        raise ValueError('Value "{}" for parameter "alg" not a valid option'.format(alg))

    # replace indices where all values were NaN with nodata value
    final_arr = np.where(lc_all_nan, nodata, final_arr)

    # create output raster
    arr_to_ascii(final_arr, out_rast, xll=-xll, yll=-yll, cellsize=cellsize, nodata=nodata)
