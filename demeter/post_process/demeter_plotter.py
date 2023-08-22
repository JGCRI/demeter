import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


class LandCoverPlotter:
    def __init__(self, folder_path, PFT_name, region_id, out_path):
        self.folder_path = folder_path
        self.PFT_name = PFT_name
        self.region_id = region_id
        self.out_path = out_path
        self.reg_name = "global"
        self.process_folders()

    def process_folders(self):
        folders = glob.glob(self.folder_path)
        for folder in folders:
            sel_folder = folder.replace("\\", "/")
            self.process_files(sel_folder)

    def process_files(self, sel_folder):
        file_path = sel_folder + "/spatial_landcover_netcdf/"
        nc_files = glob.glob(file_path + "*.nc")
        for nc_file_sel in nc_files:
            year = nc_file_sel[-7:].replace(".nc", "")
            main_nc = xr.open_dataset(nc_file_sel)
            sce_attribute = main_nc.attrs['scenario_name']
            data_variable = main_nc[self.PFT_name]
            if self.region_id > 0:
                self.reg_name = "reg_id" + str(self.region_id)
                region_var = main_nc['region_id']
                data_variable = data_variable.where(region_var == self.region_id)
                mask_data = region_var.to_dataframe()
                mask_data.reset_index(inplace=True)
                mask_data = mask_data[mask_data['region_id'] == self.region_id]
                min_lat = mask_data["latitude"].min()
                max_lat = mask_data["latitude"].max()
                max_lon = mask_data["longitude"].max()
                min_lon = mask_data["longitude"].min()
            long_name = data_variable.attrs['long_name']
            masked_data = np.ma.masked_where(np.logical_or(np.isnan(data_variable), data_variable < 0.01),
                                             data_variable)

            fig, ax = plt.subplots()
            cax = ax.imshow(masked_data, cmap='viridis', aspect='auto', origin='upper', vmin=0.05, vmax=1)
            cbar = fig.colorbar(cax)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Values for- ' + long_name + ' in year ' + year + ' in ' + sce_attribute)
            output_filename = self.out_path + long_name + "_" + sce_attribute + year + self.reg_name + ".jpg"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print("Done processing LT- " + long_name + " in year " + year + " in sce " + sce_attribute)
            plt.close()

# if __name__ == "__main__":
#    folder_path = "C:/Projects/demeter_EPPA/outputs/*"
#    PFT_name = "PFT0"
#    region_id = 0
#    out_path = "C:/Projects/EPPA_plots/"

#    plotter = LandCoverPlotter(folder_path, PFT_name, region_id, out_path)
