# Rdemeter

This is an R interface for demeter.

To use the R functionality, please use the code below by activating the R proj file.

User will first have to ensure that the python version of demeter is installed.

After that, user can run the below to intialize Rdemeter. Note that the user will need to install the `reticulate` R
package.

```R

devtools::load_all(.)
Rdemeter <- port_model_to_R(model_name="demeter",
                             path_to_python="where python for demeter is installed")
```

After this, the user can access any function from demeter in the `Rdemeter` object, by using `Rdemeter$function_name`.
User can also make use of the convenience function below to pass a demeter config file from R-

```R

run_demeter_R(path_to_inputs=" Add path to config here",
              write_outputs=TRUE)

```





