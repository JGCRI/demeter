"""
Command line interface.

    demeter run --config-file config.ini [--set PARAM=VALUE ...]
    demeter validate-config --config-file config.ini
    demeter get-data DIRECTORY

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER
"""

import argparse
import os
import sys

from demeter._version import __version__


def _parse_overrides(pairs):
    """Convert ``PARAM=VALUE`` strings into a dictionary of keyword arguments."""
    overrides = {}
    for pair in pairs or []:
        if '=' not in pair:
            raise argparse.ArgumentTypeError(f"Expected PARAM=VALUE, got '{pair}'")
        key, value = pair.split('=', 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _model_kwargs(args):
    """Keyword arguments for the model: config file, explicit options, then --set overrides."""
    kwargs = dict(config_file=args.config_file)

    for name in ('run_dir', 'start_year', 'end_year', 'target_years_output', 'gcam_database', 'scenario'):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value

    kwargs.update(_parse_overrides(args.set))
    return kwargs


def _check_config_file(path):
    if not os.path.isfile(path):
        sys.exit(f"Config file not found:  {path}")


def cmd_run(args):
    from demeter.model import run_model

    _check_config_file(args.config_file)
    kwargs = _model_kwargs(args)
    kwargs['write_outputs'] = not args.no_write_outputs
    model = run_model(**kwargs)

    if kwargs['write_outputs']:
        print(f"Outputs written to:  {model.config.output_dir}")


def cmd_validate_config(args):
    from demeter.config_reader import ReadConfig

    _check_config_file(args.config_file)
    kwargs = _model_kwargs(args)
    kwargs.update(write_outputs=False, write_logfile=False)

    try:
        c = ReadConfig(kwargs)
    except Exception as e:
        sys.exit(f"Invalid configuration:  {e}")

    missing = [f for f in (c.observed_lu_file, c.spatial_allocation_file, c.gcam_allocation_file,
                           c.kernel_allocation_file, c.transition_order_file, c.treatment_order_file,
                           c.gcam_region_names_file, c.gcam_basin_names_file) if not os.path.isfile(f)]

    if c.projected_lu_file is not None and not os.path.isfile(c.projected_lu_file):
        missing.append(c.projected_lu_file)

    print(f"run_dir:             {c.run_dir}")
    print(f"years:               {c.start_year}-{c.end_year} every {c.timestep}")
    print(f"spatial_resolution:  {c.spatial_resolution}")
    print(f"errortol:            {c.errortol} km2 (area tolerance scale {c.area_tolerance_scale})")
    print(f"observed_lu_file:    {c.observed_lu_file}")
    print(f"projected input:     {c.projected_lu_file or c.gcam_database or 'none'}")

    c.logger_ini.close_logger()

    if missing:
        sys.exit("Missing input files:\n  " + "\n  ".join(missing))

    print("Configuration is valid.")


def cmd_get_data(args):
    from demeter.install_supplement import get_package_data

    get_package_data(args.directory)


def build_parser():
    parser = argparse.ArgumentParser(prog='demeter', description='Demeter land use and land cover downscaling.')
    parser.add_argument('--version', action='version', version=f'demeter {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    def add_model_options(p):
        p.add_argument('--config-file', '-c', required=True, help='Full path to the configuration INI file')
        p.add_argument('--run-dir', dest='run_dir', help='Directory containing the input and output directories')
        p.add_argument('--start-year', dest='start_year', help='Year to start the downscaling')
        p.add_argument('--end-year', dest='end_year', help='Year to process through')
        p.add_argument('--target-years-output', dest='target_years_output',
                       help="Years to save, 'all' or semicolon delimited (e.g., 2005;2050)")
        p.add_argument('--gcam-database', dest='gcam_database', help='Full path to a GCAM output database')
        p.add_argument('--scenario', help='Scenario name used in output names')
        p.add_argument('--set', action='append', metavar='PARAM=VALUE',
                       help='Override any configuration parameter; may be repeated')

    p_run = sub.add_parser('run', help='Run the model')
    add_model_options(p_run)
    p_run.add_argument('--no-write-outputs', action='store_true', help='Run without writing output files')
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser('validate-config', help='Check a configuration and its input files without running')
    add_model_options(p_val)
    p_val.set_defaults(func=cmd_validate_config)

    p_data = sub.add_parser('get-data', help='Download the example data supplement')
    p_data.add_argument('directory', help='Directory to install the example data into')
    p_data.set_defaults(func=cmd_get_data)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
