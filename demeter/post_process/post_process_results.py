import glob
import pandas as pd
import numpy as np
import xarray as xr


class DataProcessor:
    def __init__(self, folder_path, resolution=0.5):
        self.folder_path = folder_path
        self.resolution = resolution
        self.dfs = []

    def calculate_pixel_area(self, latitude):
        earth_radius = 6371000.0  # in meters
        lat_rad = np.radians(latitude)
        pixel_width_meters = 2 * earth_radius * np.pi * np.cos(lat_rad) * self.resolution / 360.0
        pixel_height_meters = 2 * earth_radius * np.pi * self.resolution / 360.0
        pixel_area_sq_meters = pixel_width_meters * pixel_height_meters
        return pixel_area_sq_meters * 1e-6  # Convert to square kilometers

    def process_files(self):
        folders = glob.glob(self.folder_path)
        for folder in folders:
            sel_folder = folder.replace("\\", "/")
            file_path = sel_folder + "/spatial_landcover_netcdf/"
            nc_files = glob.glob(file_path + "*.nc")

            for nc_file in nc_files:
                main_nc = xr.open_dataset(nc_file)
                sce_attribute = main_nc.attrs['scenario_name']
                main_nc = main_nc.to_dataframe()
                main_nc.reset_index(inplace=True)
                main_nc.dropna(inplace=True)
                year = nc_file[-7:-3]
                main_nc['pixel_area'] = self.calculate_pixel_area(main_nc['latitude'])

                # min_lat, max_lat, min_lon, max_lon = 27.98036759814649, 50.37966027007533, -75.78187893993685, -27.98036759814649
                # main_nc = main_nc[
                #    (main_nc['latitude'] >= min_lat) & (main_nc['latitude'] <= max_lat) & (
                #            main_nc['longitude'] >= min_lon) & (main_nc['longitude'] <= max_lon)]
                main_nc = main_nc.rename(columns={'basin_id': 'metric_id'})

                lu_file_for_col = main_nc.drop(['latitude', 'longitude', 'region_id', 'metric_id'], axis=1)

                columns = lu_file_for_col.columns
                main_nc[columns] = main_nc[columns].multiply(main_nc['pixel_area'], axis="index")
                grouped_df = main_nc.groupby(["region_id", "metric_id"])[columns].sum().reset_index()
                grouped_df["year"] = year
                grouped_df["scenario_name"] = sce_attribute
                grouped_df.reset_index(inplace=True)
                grouped_df = grouped_df.drop(['index'], axis=1)
                melted_df = grouped_df.melt(id_vars=["region_id", "scenario_name", "year", "metric_id"],
                                            var_name="land_type", value_name="land_in_km2")
                print("Completed " + sce_attribute + " in year " + year)
                self.dfs.append(melted_df)

    def concatenate_and_save(self, output_file):
        concatenated_df = pd.concat(self.dfs, ignore_index=True)
        concatenated_df.to_csv(output_file, index=False)

# Create an instance of the DataProcessor class
# folder_path = "C:/Projects/demeter_EPPA/outputs/*"
# output_file = 'C:/Projects/EPPA_demeter_region_wise_test.csv'
# processor = DataProcessor(folder_path, resolution=0.5)

# Process files and save the concatenated DataFrame
# processor.process_files()
# processor.concatenate_and_save(output_file)
