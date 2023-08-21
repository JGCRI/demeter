import demeter
import glob
from configobj import ConfigObj
import time


# First generate all config_files
folder_path = "C:/Projects/demeter_EPPA/inputs/projected"

csv_files = glob.glob(folder_path + "/*.csv")

config_file = 'C:/Projects/demeter_EPPA/unharmonized_eppa_bau.ini'
config = ConfigObj(config_file)


for name in csv_files:
    new_csv = name.replace(".csv", "")
    new_csv = new_csv.replace(str(folder_path), "")
    new_csv = new_csv.replace("\\", "")

    config['INPUTS']['PROJECTED']['projected_lu_file']= new_csv+".csv"
    config['PARAMS']['scenario'] = new_csv


    config.filename="C:/Projects/demeter_EPPA/config_files/" + new_csv + '.ini'
    config.write()

print("Done generating files for EPPA run!")

config_folder_path = "C:/Projects/demeter_EPPA/config_files/"
configfiles = glob.glob(config_folder_path + "/*.ini")
#configfiles= configfiles[1]

for i in configfiles:
    config_file = i
    config_file = config_file.replace("\\","/")
    #print(config_file)
# demeter.get_package_data("C:/Projects/")

# run all time steps
    from demeter import run_model
    run_model(config_file=config_file,
          write_outputs=True,
          write_logfile=False)
    print("Ran scenario- "+ config_file)
    time.sleep(1)

print("Ran all EPPA scenarios stored in projected folder!")


