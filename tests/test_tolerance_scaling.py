"""Scale invariance of the downscaling algorithm.

High resolution runs have grid cells with tiny areas.  Shrinking every area in the bundled global case by the
factor separating a 0.25 degree cell from a 150 m cell, with the area tolerances scaled to match, must reproduce the
baseline divided by that factor.  With absolute (unscaled) tolerances the same run reallocates ~300,000 km2.

"""

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from cases import read_projected  # noqa: E402
from run_benchmark import _region_totals  # noqa: E402

BASELINE_DIR = BENCH_DIR / "baselines" / "v3.0.0-b0" / "global_0p5deg"
DATA_DIR = REPO_ROOT / "demeter" / "tests" / "data"
OBSERVED_ZIP = "gcam_reg32_basin235_modis_v6_2010_mirca_2000_0p5deg_sqdeg_wgs84_07may2021.zip"

# ratio of a 0.25 degree cell area to a 150 m cell area
SCALE = (0.25 / 0.00151872768) ** 2

# observed basemap columns that are identifiers or coordinates rather than areas
NON_AREA_COLUMNS = {"latcoord", "loncoord", "latitude", "longitude", "region_id", "basin_id", "target_fid", "fid",
                    "regaez", "aez_id"}


def _scaled_inputs(tmp_path):
    """Copy the bundled inputs with every area divided by SCALE; return run_dir and the projected data frame."""
    shutil.copytree(DATA_DIR / "inputs", tmp_path / "inputs", ignore=shutil.ignore_patterns("*.pkl", "*.pickle"))

    obs_zip = tmp_path / "inputs" / "observed" / OBSERVED_ZIP
    df = pd.read_csv(obs_zip, compression="infer")
    area_columns = [c for c in df.columns if c.lower() not in NON_AREA_COLUMNS]
    df[area_columns] = df[area_columns] / SCALE

    obs_zip.unlink()
    with zipfile.ZipFile(obs_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(obs_zip.stem + ".csv", df.to_csv(index=False))

    gcam_df = read_projected(DATA_DIR)
    gcam_df["land-allocation"] = gcam_df["land-allocation"] / SCALE

    return tmp_path, gcam_df


@pytest.mark.slow
def test_scaled_areas_reproduce_baseline(tmp_path):
    import demeter

    run_dir, gcam_df = _scaled_inputs(tmp_path)

    model = demeter.Model(run_dir=str(run_dir), gcamwrapper_df=gcam_df, write_outputs=False, write_logfile=False)

    # emulate the tolerances of a run whose cell areas are SCALE times smaller than the reference resolution
    model.config.area_tolerance_scale = 1.0 / SCALE
    model.config.errortol = 0.001 / SCALE

    model.initialize()

    try:
        for year in list(model.s.user_years):
            next(model.step_generator)
            expected = np.load(BASELINE_DIR / f"summary_{year}.npz")

            groups, totals = _region_totals(np, model.s, model.s.spat_ludataharm)

            np.testing.assert_array_equal(groups, expected["region_metric"])
            np.testing.assert_allclose(totals * SCALE, expected["region_metric_totals"], rtol=1e-9, atol=1e-6,
                                       err_msg=f"{year}: scaled run does not reproduce the baseline")
    finally:
        model.cleanup()
