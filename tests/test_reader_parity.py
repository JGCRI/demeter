"""The NetCDF and CSV observed basemap readers must produce the same model results.

Constraints are disabled because constraint files are aligned to the basemap by row order, which differs between
the two formats (plan item 1.16).

"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from cases import read_projected  # noqa: E402
from run_benchmark import _region_totals  # noqa: E402

from demeter.demeter_io.reader import read_observed_table  # noqa: E402

DATA_DIR = REPO_ROOT / "demeter" / "tests" / "data"
OBSERVED_ZIP = "gcam_reg32_basin235_modis_v6_2010_mirca_2000_0p5deg_sqdeg_wgs84_07may2021.zip"
ID_COLUMNS = ("latitude", "longitude", "basin_id", "region_id")


def _write_netcdf(df, resolution, path):
    """Write a basemap frame (square degrees) as a NetCDF of cell fractions with `long_name` attributes."""
    values = [c for c in df.columns if c not in ID_COLUMNS]
    df = df.copy()
    df[values] = df[values] / (resolution * resolution)

    ds = df.set_index(["latitude", "longitude"]).to_xarray()
    for i, name in enumerate(values):
        ds = ds.rename({name: f"var{i}"})
        ds[f"var{i}"].attrs["long_name"] = name
    ds.to_netcdf(path)


def test_read_observed_table_netcdf_matches_csv(tmp_path):
    csv = pd.DataFrame(dict(latitude=[0.25, 0.25, -0.25], longitude=[-0.25, 0.25, 0.25], basin_id=[1, 1, 2],
                            region_id=[3, 3, 3], water=[0.0, 0.01, 0.0], forest=[0.2, 0.1, 0.25]))
    _write_netcdf(csv, 0.5, tmp_path / "basemap.nc")

    nc = read_observed_table(str(tmp_path / "basemap.nc"), 0.5)
    nc = nc.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    expected = csv.sort_values(["latitude", "longitude"]).reset_index(drop=True)

    assert list(nc["fid"]) == [1, 2, 3]
    for col in ("water", "forest", "basin_id", "region_id"):
        np.testing.assert_allclose(nc[col].to_numpy(), expected[col].to_numpy(), rtol=1e-12)


def _run(run_dir, observed_file, **kwargs):
    import demeter

    gcam_df = read_projected(DATA_DIR)
    model = demeter.Model(run_dir=str(run_dir), gcamwrapper_df=gcam_df, write_outputs=False, write_logfile=False,
                          use_constraints=0, observed_lu_file=observed_file, **kwargs)
    model.initialize()

    results = {}
    try:
        for year in list(model.s.user_years):
            next(model.step_generator)
            groups, totals = _region_totals(np, model.s, model.s.spat_ludataharm)
            results[year] = (groups, totals, model.s.spat_ludataharm.copy(), model.s.spat_coords.copy())
    finally:
        model.cleanup()
    return results


@pytest.mark.slow
def test_netcdf_and_csv_basemaps_give_same_results(tmp_path):
    shutil.copytree(DATA_DIR / "inputs", tmp_path / "inputs", ignore=shutil.ignore_patterns("*.pkl", "*.pickle"))

    # the bundled case runs with the default spatial_resolution of 0.25
    df = pd.read_csv(tmp_path / "inputs" / "observed" / OBSERVED_ZIP).drop(columns=["target_fid"])
    _write_netcdf(df, 0.25, tmp_path / "inputs" / "observed" / "basemap.nc")

    from_csv = _run(tmp_path, OBSERVED_ZIP)
    from_nc = _run(tmp_path, "basemap.nc", observed_id_field="fid")

    for year in from_csv:
        groups_a, totals_a, state_a, coords_a = from_csv[year]
        groups_b, totals_b, state_b, coords_b = from_nc[year]

        np.testing.assert_array_equal(groups_a, groups_b)
        np.testing.assert_allclose(totals_b, totals_a, rtol=1e-9, atol=1e-6)

        # NetCDF cells are ordered by coordinate; align before comparing cell by cell
        index_b = pd.MultiIndex.from_arrays([coords_b[:, 0], coords_b[:, 1]])
        order = index_b.get_indexer(pd.MultiIndex.from_arrays([coords_a[:, 0], coords_a[:, 1]]))
        np.testing.assert_allclose(state_b[order], state_a, rtol=1e-9, atol=1e-6)
