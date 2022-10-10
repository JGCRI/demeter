
import xarray as x
import pandas as pd
import numpy as np
import os.path
import netCDF4
import rioxarray
import sys
from datetime import date

#Define compression parameters here
comp = dict(zlib=True, complevel=5, dtype="float32")

#Set grid resolution
GRID_RESOLUTION = 0.05

#Define scenario
sce= "rcp85cooler_ssp3"

#Define start and end years and interval
start_year=2005
end_year=2100
interval= 5

#Define the bounds we want to generate the ncdf4
xmin=-179.975
xmax=180
ymin=-90.025
ymax=90

#Define path to results and base layer
path_to_results= "C:/Projects/IM3/im3_demeter_clm/outputs/rcp85cooler_ssp3_2022-10-06_10h51m16s/spatial_landcover_tabular/"
path_to_layer = "C:/Projects/im3_demeter_clm/inputs/observed/baselayerdata_region_basin_0.05deg.csv"

#Set year of analysis
years =[]
for k in range(start_year,end_year+interval,interval):
     years.append(k)

# create evenly-spaced coordinate pairs grid on the basis of lat lon values
lon1 = np.arange(-179.975, 180, GRID_RESOLUTION)
lon1 = np.round(lon1,3)
lat1 = np.arange(-90.025, 90, GRID_RESOLUTION)
lat1 = np.round(lat1,3)


#Start year processing
for j in years:
    #Read in US outputs
    lu_file = pd.read_csv(path_to_results+"landcover"+"_"+str(j)+"_timestep.csv", index_col=False)
    lu_file_for_col = lu_file.drop(['latitude','longitude'],axis=1)

    columns = lu_file_for_col.columns


    for i in columns:

        print("Start processing: "+str(i)+" in year "+str(j))
        temp_lu_file = lu_file[['latitude','longitude',i]]

        temp_lu_file.latitude=temp_lu_file.latitude.round(3)
        temp_lu_file.longitude = temp_lu_file.longitude.round(3)
        #print(temp_lu_file['latitude'])
        df_cut = temp_lu_file[temp_lu_file['longitude'].isin(lon1)]
        if(len(df_cut.index) != len(temp_lu_file.index)):
            sys.exit("Longitudes dont match up. Please check input params for xmin, xmax, ymin, ymax!")
        df_cut = temp_lu_file[temp_lu_file['latitude'].isin(lat1)]
        if (len(df_cut.index) != len(temp_lu_file.index)):
            sys.exit("Latitudes dont match up. Please check input params for xmin, xmax, ymin, ymax!")
        temp_lu_file =temp_lu_file.rename(columns={'latitude':'lat','longitude':'lon'})
        temp_lu_file = temp_lu_file.set_index(['lat', 'lon'])


        xr = temp_lu_file.to_xarray()
        xr = xr.sortby(['lat','lon'])
        xr = xr.reindex(lat=lat1,lon=lon1)
        xr.rio.write_crs("epsg:4326", inplace=True)

        #Define encoding here
        encoding = {str(i): comp}
        #Check if file exists. If it does, append to it else create a new file
        file_exists = os.path.exists("C:/Projects/ncdf_outputs/im3_demeter_us_"+str(sce)+"_"+str(j)+".nc")
        if file_exists:
            xr.to_netcdf("C:/Projects/ncdf_outputs/im3_demeter_us_"+str(sce)+"_"+str(j)+".nc", mode="a",encoding=encoding,engine='netcdf4',format='NETCDF4')
        else:
            #Add metadata
            xr.attrs['scenario_name'] = str(sce)
            xr.attrs['datum'] = "WGS84"
            xr.attrs['coordinate_reference_system'] = "EPSG : 4326"
            xr.attrs['demeter_version'] = "1.31"
            xr.attrs['creation_date'] = str(date.today())
            xr.attrs['other_info'] = "Includes 32 land types from CLM, water fraction and GCAM region name and basin ID"
            #ADD Citation of GCAM version
            #Add cittaion of demeter

            xr.to_netcdf("C:/Projects/ncdf_outputs/im3_demeter_us_"+str(sce)+"_"+str(j)+".nc",encoding=encoding,engine='netcdf4',format='NETCDF4')

