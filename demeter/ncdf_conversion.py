import os
from datetime import date
import xarray as xr
import pandas as pd
import numpy as np



class DemeterToNetcdf:
    """Convert Demeter output files to a NetCDF file.

    :param base_year_file:                  Full path with file name and extension to the input base year file.
    :type base_year_file:                   str

    :param start_year:                      Start year of the desired extraction.
    :type start_year:                       int

    """

    # parameters for NetCDF output
    COMPRESSION_PARAMETERS = dict(zlib=True,
                                  complevel=5,
                                  dtype="float32")

    def __init__(self,
                 base_year_file: str = None,
                 start_year: int = 2005,
                 end_year: int = 2005,
                 year_interval: int = 5,
                 xmin: float = -180,
                 xmax: float = 180,
                 ymin: float = -90,
                 ymax: float = 90,
                 resolution: float = 0.05,
                 project_name: str = "",
                 scenario_name: str = "",
                 demeter_version: str = "1.31"):

        self.base_year_file = base_year_file
        self.resolution = resolution
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.demeter_version = demeter_version

        # get a list of years to process
        self.year_list = [i for i in range(start_year, end_year + year_interval, year_interval)]

        # generate evenly-spaced coordinate pairs for the output grid based on lat, lon values
        self.longitude_list = self.generate_scaled_coordinates(xmin, xmax, ascending=True)

        # positive to negative scaling required for latitude output
        self.latitude_list = self.generate_scaled_coordinates(ymin, ymax, ascending=False)

    def generate_scaled_coordinates(self,
                                    coord_min: float,
                                    coord_max: float,
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
        center_spacing = self.resolution / 2

        if ascending:
            return np.arange(coord_min + center_spacing, coord_max, self.resolution).round(decimals)

        else:
            return np.arange(coord_max - center_spacing, coord_min, -self.resolution).round(decimals)

    def process_output(self,
                       input_file_directory: str,
                       output_file_directory: str,
                       target_year: int) -> xr.Dataset:
        """Process output for a target year for all land classes.

        :param input_file_directory:                Full path to the directory where demeter stores its outputs.
        :type input_file_directory:                 str

        :param output_file_directory:               Full path to the directory where the NetCDF files should be written.
        :type output_file_directory:                str

        :param target_year:                         Target year to process.
        :type target_year:                          int

        :returns:                                   Xarray dataset

        """

        # construct target file name
        target_file_name = os.path.join(input_file_directory, f"landcover_{target_year}_timestep.csv")

        # read in outputs to a data frame
        lu_file = pd.read_csv(target_file_name, index_col=False)

        # drop coordinate fields
        lu_file_for_col = lu_file.drop(['latitude', 'longitude'], axis=1)

        columns = lu_file_for_col.columns

        for index, i in enumerate(columns):

            print(f"Processing: {i} in year {target_year}")

            temp_lu_file = lu_file[['latitude', 'longitude', i]].copy()

            temp_lu_file.latitude = temp_lu_file.latitude.round(3)
            temp_lu_file.longitude = temp_lu_file.longitude.round(3)

            df_cut = temp_lu_file[temp_lu_file['longitude'].isin(self.longitude_list)]

            if len(df_cut.index) != len(temp_lu_file.index):
                msg = "Longitudes dont match up. Please check input params for xmin, xmax, ymin, ymax!"
                raise AssertionError(msg)

            df_cut = temp_lu_file[temp_lu_file['latitude'].isin(self.latitude_list)]

            if len(df_cut.index) != len(temp_lu_file.index):
                msg = "Latitudes dont match up. Please check input params for xmin, xmax, ymin, ymax!"
                raise AssertionError(msg)

            temp_lu_file = temp_lu_file.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            temp_lu_file = temp_lu_file.set_index(['lat', 'lon'])

            ds = temp_lu_file.to_xarray()
            ds = ds.sortby(['lon','lat'])

            ds = ds.reindex(lat=self.latitude_list,
                            lon=self.longitude_list)



            # define encoding here
            encoding = {str(i): DemeterToNetcdf.COMPRESSION_PARAMETERS}

            # check if file exists. If it does, append to it else create a new file
            output_file_name = f"{self.project_name + '_'}demeter_{self.scenario_name + '_'}{target_year}.nc"
            output_file_path = os.path.join(output_file_directory, output_file_name)

            if os.path.exists(output_file_path):

                ds = ds.rename({"lat": "latitude",
                                "lon": "longitude"})

                ds[i].attrs["long_name"] = i

                if i not in ["region_id", "basin_id"]:
                   ds = ds.rename({i: f"PFT{index}"})
                   encoding = {f"PFT{index}": DemeterToNetcdf.COMPRESSION_PARAMETERS}

                ds.to_netcdf(output_file_path,
                             mode="a",
                             encoding=encoding,
                             engine='netcdf4',
                             format='NETCDF4')

            else:

                # add metadata
                ds.attrs['scenario_name'] = self.scenario_name
                ds.attrs['datum'] = "WGS84"
                ds.attrs['coordinate_reference_system'] = "EPSG : 4326"
                ds.attrs['demeter_version'] = self.demeter_version
                ds.attrs['creation_date'] = str(date.today())
                ds.attrs['other_info'] = "Includes 32 land types from CLM, water fraction and GCAM region name and basin ID"
                # TODO Citation of GCAM version
                # TODO citaion of demeter

                ds[i].attrs["long_name"] = i

                ds = ds.rename({"lat": "latitude",
                                "lon": "longitude"})

                if i not in ["region_id", "basin_id"]:
                    ds = ds.rename({i: f"PFT{index}"})
                    encoding = {f"PFT{index}": DemeterToNetcdf.COMPRESSION_PARAMETERS}

                ds.to_netcdf(output_file_path,
                             encoding=encoding,
                             engine='netcdf4',
                             format='NETCDF4')

        return ds


if __name__ == "__main__":

    input_file_directory = "/Users/d3y010/Downloads"
    output_file_directory = "/Users/d3y010/Desktop"
    target_year = 2005

    # instantiate class
    x = DemeterToNetcdf(scenario_name="rcp85cooler",
                        project_name="im3")

    # convert demeter output to a NetCDF file
    x.process_output(input_file_directory=input_file_directory,
                     output_file_directory=output_file_directory,
                     target_year=target_year)
