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

### Projected land allocation data:

This file is the formatted GCAM run output for land allocation projections.  Since the format of this file can vary based on GCAM user preference, the file must be formatted to meet Demeter input requirements as described in Table 2.  The file must be a CSV file having the header in the first row.

| Field	| Description |
| --- | --- |
| region	| The text name of the GCAM region. Exact field name spelling required. |
| landclass | Each land class field name (e.g., shrub, grass, etc.).  Field names must not include commas. |
| year | Each year of the GCAM run as an integer (e.g., 2005, 2010, etc.) |
| metric_id |	The integer ID of the GCAM AEZ or basin.  Exact field name spelling required. |

**Table 2.** Projected land allocation required fields from GCAM.

### Allocation files:

#### *Constraint weighting:*

Weight each constraint as it is to be applied per functional type.  If no constraints are desired, a user should simply provide a header-only file.  Value from -1.0 to 1.0.  A use case would be weighting functional types for a soil quality constraint:  a constraint weighted at -1 is fully constrained but inversely applied (e.g., grasslands that are opportunistic and grow readily in areas with a low soil quality); a constraint weighted at 0 would indicate that there is no application of the soil quality constraint to the functional type (e.g., forest, etc.); a functional type weighted at 1 for soil quality would indicate that high soil quality will highly influence where the type will be spatially allocated (e.g. cropland.).  These constraints are developed in separate files as described in the following Constraints section.  See the constraint weighting file in the example inputs for reference. 

#### *Kernel density weighting:*

Weight the degree to which land types subjected to a kernel density filter will be utilized during expansion to each functional type.  Value from 0.0 to 1.0. See the kernel density weighting file in the example inputs for reference.

#### *Transition Priority:*

This ordering defines the preferential order of final functional type expansion (e.g., crops expanding into grasslands rather than forests).  See the priority allocation file in the example inputs for reference.  See the priority allocation file in the example inputs for reference.  

#### *Treatment order:*

Defines the order in which final functional types are downscaled.  This will influence the results (e.g., if crops are downscaled first and overtake grassland, grassland will not be available for shrubs to overtake when processing shrub land).  See the treatment order file in the example inputs for reference. 

#### *Observational spatial data class allocation:*

This file defines how the land-use and land-cover classes in the OSD will be binned into final functional types.  Final functional types are defined by the user and serve to place projected land allocation data from GCAM on a common scale with the on-the-ground representation of land-use and land-cover represented in the OSD.  See the Observed spatial data class allocation file in the example inputs for reference. 

#### *Projected land class allocation:*

This file defines how the land-use and land-cover classes in the GCAM projected land allocation data will be binned into final functional types.  See the projected land class allocation file in the example inputs for reference.  

#### *Constraints (not required):*

Constraints such as soil quality may be desirable to the user and can be prepared by assigning a weighted value from 0.0 to 1.0 for each grid cell in the OSD. A value of 0.0 would represent a fully constrained cell where an allocated functional type could not utilize and 1.0 would represent a grid cell where there is no constraining data.  Users should note that constraining a grid cell to 0.0 may impede the ability to be able to achieve a projected land allocation from GCAM since land area is being excluded that GCAM expects.  Each constraint file must have two fields:  fid and weight.  The fid field should correspond to the fid field in the OSD input and the weight field should be the weight of the constraint per the cell corresponding to the OSD input.  Each file should be a CSV with no header.
 
### Configuration file:

Demeter’s configuration file allows the user to customize each run and define where file inputs are and outputs will be.  The configuration options and hierarchical level are described in Table 3.

| Level | Parameter | Description |
| --- | --- | --- |
| [STRUCTURE |	root_dir | The full path of the root directory where the inputs and outputs directory are stored |
| [STRUCTURE |	in_dir	| The name of the input directory |
| [STRUCTURE |	out_dir	 | The name of the output directory |
[INPUTS]	allocation_dir	The name of the directory that holds the allocation files
[INPUTS]	observed_dir	The name of the directory that holds the observed spatial data file
[INPUTS]	constraints_dir	The name of the directory that holds the constraints files
[INPUTS]	projected_dir	The name of the directory that holds the GCAM projected land allocation file
[INPUTS]	ref_dir	The name of the directory that holds the reference files
[INPUTS][ALLOCATION]	spatial_allocation	The file name with extension of the observed spatial data class allocation 
[INPUTS][ALLOCATION]	gcam_allocation	The file name with extension of the projected land class allocation
[INPUTS][ALLOCATION]	kernel_allocation	The file name with extension of the kernel density weighting
[INPUTS][ALLOCATION]	priority_allocation	The file name with extension of the priority allocation
[INPUTS][ALLOCATION]	treatment_order	The file name with extension of the treatment order
[INPUTS][ALLOCATION]	constraints	The file name with extension of the constraint weighting
[INPUTS][OBSERVED]	observed_lu_data	The file name with extension of the observational spatial data
[INPUTS][PROJECTED]	projected_lu_data	The file name with extension of the projected land allocation data from GCAM
[INPUTS][REFERENCE]	gcam_regnamefile	The file name with extension of the GCAM region name to region id lookup 
[INPUTS][REFERENCE]	region_coords	A CSV file of GCAM region coordinates for each grid cell
[INPUTS][REFERENCE]	country_coords	A CSV file of GCAM country coordinates for each grid cell
[OUTPUTS]	diag_dir	The name of the directory that diagnostics outputs will be kept
[OUTPUTS]	log_dir	The name of the directory that the log file outputs will be kept
[OUTPUTS]	kernel_map_dir	The name of the directory that kernel density map outputs will be kept
[OUTPUTS]	transition_tabular	The name of the directory that tabular land transition outputs will be kept
[OUTPUTS]	transition_maps	The name of the directory that land transition map outputs will be kept
[OUTPUTS]	luc_intense_p1_dir	The name of the directory that the land intensification first pass map outputs will be kept
[OUTPUTS]	luc_intense_p2_dir	The name of the directory that the land intensification second pass map outputs will be kept
[OUTPUTS]	luc_expand_dir	The name of the directory that the land expansion map outputs will be kept
[OUTPUTS]	luc_ts_luc	The name of the directory that the land use change per time step map outputs will be kept
[OUTPUTS]	lc_per_step_csv	The name of the directory that the tabular land change per time step outputs will be kept
[OUTPUTS]	lc_per_step_nc	The name of the directory that the NetCDF land change per time step outputs will be kept
[OUTPUTS]	lc_per_step_shp	The name of the directory that the Shapefile land change per time step outputs will be kept
[OUTPUTS][DIAGNOSTICS]	harm_coeff	The file name with extension of the NumPy array that will hold the harmonization coefficient data
[OUTPUTS][DIAGNOSTICS]	intense_pass1_diag	The file name with extension of the CSV that will hold the land allocation per time step per functional type for the first pass of intensification
[OUTPUTS][DIAGNOSTICS]	intense_pass2_diag	The file name with extension of the CSV that will hold the land allocation per time step per functional type for the second pass of intensification
[OUTPUTS][DIAGNOSTICS]	expansion_diag	The file name with extension of the CSV that will hold the land allocation per time step per functional type for the expansion pass
[PARAMS]	model	The model name providing the projected land allocation data (e.g., GCAM)
[PARAMS]	metric	Either AEZ or BASIN
[PARAMS]	scenario	Scenario name
[PARAMS]	run_desc	The description of the current run
[PARAMS]	agg_level	1 if only by metric, 2 if by region and metric
[PARAMS]	observed_id_field	Observed spatial data unique field name (e.g. fid)
[PARAMS]	start_year	First time step to process (e.g., 2005)
[PARAMS]	end_year	Last time step to process (e.g., 2100)
[PARAMS]	use_constraints	1 to use constraints, 0 to ignore constraints
[PARAMS]	spatial_resolution	Spatial resolution of the observed spatial data in decimal degrees (e.g. 0.25)
[PARAMS]	errortol	Allowable error tolerance in square kilometres for non-accomplished change
[PARAMS]	timestep	Time step interval (e.g., 5)
[PARAMS]	proj_factor	Factor to multiply the projected land allocation by
[PARAMS]	diagnostic	0 to not output diagnostics, 1 to output
[PARAMS]	intensification_ratio	Ideal fraction of land change that will occur during intensification.  The remainder will be through expansion.  Value from 0.0 to 1.0.
[PARAMS]	stochastic_expansion	0 to not conduct stochastic expansion of grid cells, 1 to conduct
[PARAMS]	selection_threshold	Threshold above which grid cells are selected to receive expansion for a target functional type from the kernel density filter.  Value from 0.0 to 1.0; where 0 lets all land cells receive expansion and 1 only lets only the grid cells with the maximum likelihood expand.
[PARAMS]	kernel_distance	Radius in grid cells used to build the kernel density convolution filter used during expansion
[PARAMS]	map_kernels	0 to not map kernel density, 1 to map
[PARAMS]	map_luc_pft	0 to not map land change per land class per time step, 1 to map
[PARAMS]	map_luc_steps	0 to not map land change per time step per land class for intensification and expansion, 1 to map
[PARAMS]	map_transitions	0 to not map land transitions, 1 to map
[PARAMS]	target_years_output	Years to save data for; default is ‘all’; otherwise a semicolon delimited string (e.g., 2005; 2020)
[PARAMS]	save_tabular	Save tabular spatial land cover as a CSV; define tabular units in tabular_units param
[PARAMS]	tabular_units	Units to output the spatial land cover data in; either ‘sqkm’ or ‘percent’
[PARAMS]	save_transitions	0 to not write CSV files for each land transitions per land type, 1 to write
[PARAMS]	save_shapefile	0 to not write a Shapefile for each time step containing for all functional types, 1 to write; output units will be same as tabular data
[PARAMS]	save_netcdf_yr	0 to not write a NetCDF file of land cover percent for each year by grid cell containing each class; 1 to write
[PARAMS]	save_netcdf_lc	0 to not write a NetCDF file of land cover percent by land class by grid cell containing each year interpolated to one-year intervals; 1 to write
[ENSEMBLE]	permutations	If running an ensemble of configurations, this is the number of permutations to process
[ENSEMBLE]	limits_file	If running an ensemble of configurations, this is the full path to a CSV file containing limits to generate ensembles of certain parameters.
[ENSEMBLE]	n_jobs	If running an ensemble of configurations, this is the number of CPU’s to spread the parallel processing over.  -1 is all, -2 is all but one, 4 is four, etc.
Table 3.  Configuration file hierarchy, parameters, and descriptions.
