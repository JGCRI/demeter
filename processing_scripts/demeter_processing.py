import demeter

config_file = 'C:/Projects/IM3/im3_demeter_clm/clm_rcp85cooler_ssp.ini'

# run all time steps
demeter.run_model(config_file=config_file,
                  write_outputs=True)


