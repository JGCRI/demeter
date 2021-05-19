"""
Demeter:  a land use land cover disaggregation and change detection model.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (PNNL); Yannick le Page (niquya@gmail.com)
"""
from demeter.model import Model, run_model
from demeter.install_supplement import get_package_data

from demeter.preprocess_data import format_gcam_data

__author__ = "Chris R. Vernon (chris.vernon@pnnl.gov); Yannick le Page (niquya@gmail.com)"
__version__ = '1.2.0'

__all__ = ['Model', 'format_gcam_data', 'run_model', 'get_package_data']
