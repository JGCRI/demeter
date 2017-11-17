# Demeter

## A land use land cover disaggregation and change detection model

## NOTICE:
This repository uses Git Large File Storage (LFS). Please download and install from here: [https://github.com/git-lfs/git-lfs/wiki/Installation](https://github.com/git-lfs/git-lfs/wiki/Installation)

Once installed, run the following command before cloning this repository: `git lfs install`

## Overview

The Global Change Assessment Model (GCAM) is a global integrated assessment model used to project future societal and environmental scenarios, based on economic modeling and on a detailed representation of food and energy production systems. The terrestrial module in GCAM represents agricultural activities and ecosystems dynamics at the subregional scale, and must be downscaled to be used for impact assessments in gridded models (e.g., climate models). This downscaling algorithm of the GCAM model, which generates gridded time series of global land use and land cover (LULC) from any GCAM scenario. The downscaling is based on a number of user-defined rules and drivers, including transition priorities (e.g., crop expansion preferentially into grasslands rather than forests) and spatial constraints (e.g., nutrient availability). The default parameterization is evaluated using historical LULC change data, and a sensitivity experiment provides insights on the most critical parameters and how their influence changes regionally and in time. Finally, a reference scenario and a climate mitigation scenario are downscaled to illustrate the gridded land use outcomes of different policies on agricultural expansion and forest management. Features of the downscaling can be modified by providing new input data or customizing the configuration file. Those features include spatial resolution as well as the number and type of land classes being downscaled, thereby providing flexibility to adapt GCAM LULC scenarios to the requirements of a wide range of models and applications.

## Installation
The following step will get Demeter ready to use:
1.  This repository uses the Git Large File Storage (LFS) extension (see https://git-lfs.github.com/ for details).  Please run the following command before cloning if you do not already have Git LFS installed:
`git lfs install`
2.  Clone Demeter into your desired location git clone https://github.com/IMMM-SFA/demeter.git`
3.  From the directory you cloned Demeter into run `python setup.py install` .  This will install Demeter as a Python package on your machine and install of the needed dependencies.
4.  Setup your configuration file (.ini).  Examples are located in the "example" directory.  Be sure to change the root directory to the directory that holds your data (use the 'demeter/example' directory as an example).
5. If running Demeter from an IDE:  Be sure to include the path to your config file.  See the "demeter/example/example.py" script as a reference.
6. If running Demeter from terminal:  Run model.py found in demeter/demeter/model.py passing the full path to the config file as the only argument. (e.g., `python model.py /users/ladmin/repos/github/demeter/example/config.ini`)

If a permissions error is encountered either run the command sudo or on Windows open cmd as an administrator.

## Setting up demeter

Demeter requires the user to prepare several files to conduct a run. The following describes what the input files are and how to prepare them.

### Base layer data

The base layer data file is stored in its own directory in the inputs directory.  This file represents the percentage of each land class existing within a grid cell. The latitude and longitude for the centroid of each grid cell is also required in this file to be labeled as 'latcoord' and 'loncoord' respectively.  A feature id, or 'fid', which is a unique identifier for each grid cell is also required. The 'fid' is used as a primary key to join any constraint data provided by the user during runtime. Two levels of ids relating the base layer data to the projected land allocation data zones must be provided; these were originally developed for GCAM's region/AEZ structure.  However, this will be changed in future versions so the user will have the option to spatially join provided boundaries of their projection zones to the base layer during runtime.  Currently the user must include a 'region_id' and an 'aez_id' or 'basin_id' that relates each base layer grid cell to the projected zone.  There must be no commas in the field names and these field names are to be included in the input base layer file.  The base layer file is to be saved with a '.txt' extension.  Future versions will allow the user to provide these as a shapfile or geojson file.  Each land class field in the base layer file must have it's values represented as the percentage of grid cell as a decimal (from 0.0 to 1.0).  See the input base layer file in the example setup for a reference.

### Projected land allocation

Projected land allocation data has currently been tested with data from GCAM.  This data has been formatted to contain the following fields 'region', 'landclass', 'metric_id'. 'region' is the GCAM region name that is joined to its region id during runtime.  'landclass' is the land class name of each functional type. 'metric_id' is either the AEZ or Basin id depending upon which version of GCAM the user is running.  Each projected year must have its own header such as '1990', '2000', '2090', etc.  The values for land allocation are processed in Demeter as square kilometers though GCAM outputs as thousands of square kilometers; to handle this a factor can be defined in the configuration file.  The header must be in the first line of the file and field names should contain no commas.  See the input projected file in the example setup for a reference.

### Allocation files

Allocation files are stored in their own directory nested in the inputs directory. These files are used to define how Demeter will interpret and relate the base layer data and the projected land allocation data.  They are also used to describe the order by which land classes will be processed and how they are to intensify or expand over other classes.
