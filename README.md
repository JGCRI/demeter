# Demeter

## A land-use and land-cover disaggregation and change detection model

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

## Setup
Demeter requires the setup of several input files to begin a run.  Examples of all input files can be found in the ‘examples’ directory and the expected file structure is outlined in the following:

-	Example directory
        -   Inputs directory
            -	Allocation directory
                    -	Constraint weighting file
                    -	GCAM landclass allocation file
                    -	Kernel density weighting file
                    -	Spatial landclass allocation file
                    -	Transition priority file
                    -	Treatment order file
            -	Observed spatial data directory
                    -	Observed spatial data file
            -	Constraint data directory
                    -	Constraint files
            -	Projected GCAM land allocation directory
                    -	GCAM land allocation file
            -	Reference data directory
                    -	Reference files

The following describes the requirements and format of each input.

### Observed spatial data:

This file represents the area in square degrees of each land class existing within a grid cell.  The grid cell size is defined by the user.  This file must be presented as a comma-separated values (CSV) file having a header in the first row and must contain the field names and fields described in Table 1.


|   Field	|   Description   |         
| --------- | --------------- |
| fid | Unique integer ID for each grid cell latitude, longitude |
| landclass | Each land class field name (e.g., shrub, grass, etc.).  Field names must not include commas. |
| region_id	| The integer ID of the GCAM region that the grid cell is contained in.  Exact field name spelling required. |
| metric_id	| The integer ID of the GCAM AEZ or basin that the grid cell is contained in.  Exact field name spelling required. |
| latitude | The geographic latitude value of the grid cell centroid as a signed float.  Exact field name spelling required. |
| longitude	| The geographic longitude value of the grid cell centroid as a signed float.  Exact field name spelling required. |
**Table 1.**  Observed spatial data required fields and their descriptions.
