from demeter.post_process import demeter_plotter as plotter
from demeter.post_process import Post_process_results as proc

# Example on how to run post process results
folder_path = "C:/Projects/demeter_EPPA/outputs/*"
output_file = 'C:/Projects/EPPA_demeter_region_wise_test.csv'
processor = proc.DataProcessor(folder_path, resolution=0.5)

# Process files and save the concatenated DataFrame
processor.process_files()
processor.concatenate_and_save(output_file)

# Example on how to run plotter
folder_path = "C:/Projects/demeter_EPPA/outputs/*"
PFT_name = "PFT9"
region_id = 0
out_path = "C:/Projects/EPPA_plots/"

plotter = plotter.LandCoverPlotter(folder_path, PFT_name, region_id, out_path)
