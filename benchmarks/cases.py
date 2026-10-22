"""Benchmark case definitions.

Each case returns the keyword arguments passed to ``demeter.Model`` plus the list of input files whose
checksums are recorded in the benchmark manifest, so a comparison can confirm both runs used identical inputs.

Cases only use the public ``demeter.Model`` API so they run against any code revision (e.g., a ``git worktree``
of ``v2.0.2``) selected with ``--code-path``.

"""

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def read_projected(data_dir):
    """Read the bundled gcamwrapper-format projected land allocation.

    Uses the compressed CSV when present (parsed with exact float round trip) and falls back to the pickle shipped by
    older revisions so cases also run against them.

    """
    projected_dir = Path(data_dir) / "inputs" / "projected"
    csv = projected_dir / "land_df.csv.gz"
    if csv.exists():
        # pandas >= 3 parses floats with exact round trip by default and deprecates the option
        kwargs = {} if int(pd.__version__.split(".")[0]) >= 3 else dict(float_precision="round_trip")
        return pd.read_csv(csv, **kwargs)
    return pd.read_pickle(projected_dir / "land_df.pickle")


def _files_under(directory):
    """Return a sorted list of all files under a directory."""
    return sorted(str(p) for p in Path(directory).rglob("*") if p.is_file())


def global_0p5deg(code_path, write_outputs=False):
    """Bundled global 0.5 degree GCAM 32-region / 235-basin test case (2010, 2015).

    Uses the data in ``<code_path>/demeter/tests/data`` unless ``DEMETER_BENCH_GLOBAL_DIR`` is set.  Matches the
    configuration of the historical regression test (keyword arguments only, all other parameters default).

    """
    data_dir = Path(os.environ.get("DEMETER_BENCH_GLOBAL_DIR", Path(code_path) / "demeter" / "tests" / "data"))
    kwargs = dict(run_dir=str(data_dir),
                  gcamwrapper_df=read_projected(data_dir),
                  write_outputs=write_outputs,
                  write_logfile=False)

    inputs = [f for f in _files_under(data_dir / "inputs") if not f.endswith((".pkl", ".pickle"))]

    return dict(kwargs=kwargs, inputs=inputs, data_dir=str(data_dir), cleanup=None)


def global_0p5deg_negcons(code_path, write_outputs=False):
    """Global 0.5 degree case with the soil quality constraint made negative for all land classes.

    The bundled constraints are all positive, so they cannot detect defects in negative-constraint handling.  This
    variant copies the bundled inputs to a temporary directory and flips the sign of the ``soil_quality`` weights.

    """
    src_dir = Path(os.environ.get("DEMETER_BENCH_GLOBAL_DIR", Path(code_path) / "demeter" / "tests" / "data"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="demeter_bench_negcons_"))
    shutil.copytree(src_dir / "inputs", tmp_dir / "inputs", ignore=shutil.ignore_patterns("*.pkl"))

    cons_file = tmp_dir / "inputs" / "allocation" / "csdms_constraint_allocation.csv"
    df = pd.read_csv(cons_file)
    row = df["category"] == "soil_quality"
    value_cols = [c for c in df.columns if c != "category"]
    df.loc[row, value_cols] = -df.loc[row, value_cols]
    df.to_csv(cons_file, index=False)

    kwargs = dict(run_dir=str(tmp_dir),
                  gcamwrapper_df=read_projected(src_dir),
                  write_outputs=write_outputs,
                  write_logfile=False)

    return dict(kwargs=kwargs, inputs=_files_under(tmp_dir / "inputs"), data_dir=str(tmp_dir),
                cleanup=lambda: shutil.rmtree(tmp_dir, ignore_errors=True))


def texas_150m(code_path, write_outputs=False):
    """High resolution (150 m) regional Texas case driven by a user-supplied configuration file.

    Set ``DEMETER_BENCH_150M_CONFIG`` to the full path of the case INI file.  The data stays outside the repository.

    """
    config_file = os.environ.get("DEMETER_BENCH_150M_CONFIG")

    if config_file is None or not os.path.isfile(config_file):
        raise FileNotFoundError("Set DEMETER_BENCH_150M_CONFIG to the full path of the 150 m case INI file.")

    kwargs = dict(config_file=config_file, write_outputs=write_outputs, write_logfile=False)

    # record the config and every file in the inputs directory next to it
    inputs = [config_file]
    input_dir = Path(config_file).parent / "inputs"
    if input_dir.is_dir():
        inputs += _files_under(input_dir)

    return dict(kwargs=kwargs, inputs=inputs, data_dir=str(Path(config_file).parent), cleanup=None)


# Texas bounding box used to derive a regional case from the bundled global data
TEXAS_BOUNDS = dict(lat_min=25.5, lat_max=36.5, lon_min=-106.75, lon_max=-93.5)


def texas_0p5deg(code_path, write_outputs=False):
    """Regional case: the USA cells of the bundled global data inside a Texas bounding box (0.5 degree, 2010-2015).

    Exercises the regional code path (kernel density grid from data bounds) that high resolution regional runs use.
    Mapping, projected, and constraint inputs are subset to the same region and basins.

    """
    src_dir = Path(os.environ.get("DEMETER_BENCH_GLOBAL_DIR", Path(code_path) / "demeter" / "tests" / "data"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="demeter_bench_texas_"))
    shutil.copytree(src_dir / "inputs", tmp_dir / "inputs", ignore=shutil.ignore_patterns("*.pkl", "*.zip", "*.pickle"))

    b = TEXAS_BOUNDS
    observed_name = "gcam_reg32_basin235_modis_v6_2010_mirca_2000_0p5deg_sqdeg_wgs84_07may2021.zip"
    obs = pd.read_csv(src_dir / "inputs" / "observed" / observed_name)
    obs = obs[obs.latitude.between(b["lat_min"], b["lat_max"]) & obs.longitude.between(b["lon_min"], b["lon_max"])
              & (obs.region_id == 1)]
    obs.to_csv(tmp_dir / "inputs" / "observed" / "texas_0p5deg.csv", index=False)

    # constraint files are aligned to the observed rows by order; keep the rows of the retained cells
    for f in sorted((tmp_dir / "inputs" / "constraints").glob("*.csv")):
        cons = pd.read_csv(f, header=None)
        cons = cons.set_index(0).loc[obs.target_fid.values].reset_index()
        cons.to_csv(f, header=False, index=False)

    regions = pd.read_csv(src_dir / "inputs" / "mapping" / "gcam_regions_32.csv")
    regions[regions.gcam_region_id == 1].to_csv(tmp_dir / "inputs" / "mapping" / "gcam_regions_32.csv", index=False)

    basins = pd.read_csv(src_dir / "inputs" / "mapping" / "gcam_basin_lookup.csv")
    basins = basins[basins.basin_id.isin(obs.basin_id.unique())]
    basins.to_csv(tmp_dir / "inputs" / "mapping" / "gcam_basin_lookup.csv", index=False)

    gcam_df = read_projected(src_dir)
    glu = gcam_df["child-nodes"].str.split("_").str[1]
    gcam_df = gcam_df[(gcam_df.region == "USA") & glu.isin(basins.glu_name)].reset_index(drop=True)

    kwargs = dict(run_dir=str(tmp_dir),
                  gcamwrapper_df=gcam_df,
                  observed_lu_file="texas_0p5deg.csv",
                  spatial_resolution=0.5,
                  write_outputs=write_outputs,
                  write_logfile=False)

    return dict(kwargs=kwargs, inputs=_files_under(tmp_dir / "inputs"), data_dir=str(tmp_dir),
                cleanup=lambda: shutil.rmtree(tmp_dir, ignore_errors=True))


CASES = {
    "global_0p5deg": global_0p5deg,
    "global_0p5deg_negcons": global_0p5deg_negcons,
    "texas_0p5deg": texas_0p5deg,
    "texas_150m": texas_150m,
}


def get_case(name, code_path, write_outputs=False):
    """Return the case specification for a case name."""
    try:
        return CASES[name](code_path, write_outputs=write_outputs)
    except KeyError:
        raise KeyError(f"Unknown case '{name}'.  Available cases:  {sorted(CASES)}")
