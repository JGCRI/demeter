

run_demeter_R <- function(path_to_python= "../venv/Scripts/python.exe",
                          path_to_inputs="C:/Projects/demeter_sample/config_gcam_reference.ini",
                          write_outputs=TRUE){

  Sys.setenv(RETICULATE_PYTHON = path_to_python)
  print("Setup complete for env")
  demeter_R <- reticulate::import("demeter")
  print("Package imported")
  demeter_R$run_model(config_file=path_to_inputs, write_outputs=TRUE)
  print("Demeter run successfully completed in R!")

  }


port_model_to_R <- function(path_to_python= "../venv/Scripts/python.exe",
                            model_name="demeter"){

  Sys.setenv(RETICULATE_PYTHON = path_to_python)
  print("Setup complete for env")
  model <- reticulate::import(model_name)

  return(model)
}
