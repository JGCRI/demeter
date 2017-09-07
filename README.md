# Demeter

## A land use land cover disaggregation and change detection model

## NOTICE:
This repository uses Git Large File Storage (LFS). Please download and install from here: [https://github.com/git-lfs/git-lfs/wiki/Installation](https://github.com/git-lfs/git-lfs/wiki/Installation)

Once installed, run the following command before cloning this repository: `git lfs install`

## Overview

The Global Change Assessment Model (GCAM) is a global integrated assessment model used to project future societal and environmental scenarios, based on economic modeling and on a detailed representation of food and energy production systems. The terrestrial module in GCAM represents agricultural activities and ecosystems dynamics at the subregional scale, and must be downscaled to be used for impact assessments in gridded models (e.g., climate models). This downscaling algorithm of the GCAM model, which generates gridded time series of global land use and land cover (LULC) from any GCAM scenario. The downscaling is based on a number of user-defined rules and drivers, including transition priorities (e.g., crop expansion preferentially into grasslands rather than forests) and spatial constraints (e.g., nutrient availability). The default parameterization is evaluated using historical LULC change data, and a sensitivity experiment provides insights on the most critical parameters and how their influence changes regionally and in time. Finally, a reference scenario and a climate mitigation scenario are downscaled to illustrate the gridded land use outcomes of different policies on agricultural expansion and forest management. Features of the downscaling can be modified by providing new input data or customizing the configuration file. Those features include spatial resolution as well as the number and type of land classes being downscaled, thereby providing flexibility to adapt GCAM LULC scenarios to the requirements of a wide range of models and applications.

## Installation
Once cloned, run the setup.py file like so from terminal or command line:
`python setup.py install`

If a permissions error is encountered either run the command sudo or on Windows open cmd as an administrator.
