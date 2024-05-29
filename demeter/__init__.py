"""
Demeter:  a land use land cover disaggregation and change detection model.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)
"""
from .model import Model, run_model
from .install_supplement import get_package_data

from .preprocess_data import format_gcam_data, FormatGcamDataFrame
from .post_process.demeter_plotter import LandCoverPlotter as LandCoverPlotter
from .post_process.post_process_results import DataProcessor as DataProcessor
