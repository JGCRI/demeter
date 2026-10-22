"""Command line interface, driven through a configuration file for the bundled global case."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "demeter" / "tests" / "data"
BENCH_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from cases import read_projected  # noqa: E402

from demeter import cli, format_gcam_data  # noqa: E402

BASELINE_DIR = BENCH_DIR / "baselines" / (BENCH_DIR / "baselines" / "CURRENT").read_text().strip() / "global_0p5deg"


@pytest.fixture
def config_file(tmp_path):
    """Run directory with the bundled inputs and the projected data as a GCAM-format CSV."""
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True)
    for sub in ("allocation", "constraints", "mapping", "observed"):
        (run_dir / "inputs" / sub).symlink_to(DATA_DIR / "inputs" / sub)

    (run_dir / "inputs" / "projected").mkdir()
    projected = format_gcam_data(read_projected(DATA_DIR), start_year=2010, through_year=2015)
    projected.to_csv(run_dir / "inputs" / "projected" / "gcam_land.csv", index=False)

    ini = tmp_path / "config.ini"
    ini.write_text(f"""[STRUCTURE]
run_dir = {run_dir}

[INPUTS]
    [[ALLOCATION]]
    [[OBSERVED]]
    [[PROJECTED]]
    projected_lu_file = gcam_land.csv
    [[MAPPING]]

[PARAMS]
scenario = cli
start_year = 2010
end_year = 2015
timestep = 5
""")
    return ini


def test_validate_config(config_file, capsys):
    cli.main(["validate-config", "--config-file", str(config_file)])
    assert "Configuration is valid." in capsys.readouterr().out


def test_validate_config_reports_missing_inputs(config_file):
    with pytest.raises(SystemExit, match="Missing input files"):
        cli.main(["validate-config", "--config-file", str(config_file), "--set", "projected_lu_file=missing.csv"])


def test_unknown_config_file():
    with pytest.raises(SystemExit, match="Config file not found"):
        cli.main(["run", "--config-file", "/nonexistent/config.ini"])


@pytest.mark.slow
def test_run_matches_baseline(config_file, capsys):
    cli.main(["run", "--config-file", str(config_file), "--set", "save_tabular=1", "--set", "save_netcdf_yr=0"])
    out_dir = Path(capsys.readouterr().out.strip().split("Outputs written to:")[-1].strip())

    for year in (2010, 2015):
        df = pd.read_csv(out_dir / "spatial_landcover_tabular" / f"landcover_{year}_timestep.csv")
        expected = np.load(BASELINE_DIR / f"summary_{year}.npz")
        landclasses = [str(c) for c in expected["final_landclasses"]]

        # the tabular output reports cells to 6 significant digits
        totals = df.groupby(["region_id", "basin_id"])[landclasses].sum()
        keys = [tuple(k) for k in expected["region_metric"]]
        np.testing.assert_allclose(totals.loc[keys].to_numpy(), expected["region_metric_totals"], rtol=1e-5, atol=1e-3)
