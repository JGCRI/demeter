"""End-to-end run writing every output type, including diagnostics."""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "demeter" / "tests" / "data"


@pytest.mark.slow
def test_run_writes_outputs_and_diagnostics(tmp_path):
    import demeter

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "inputs").symlink_to(DATA_DIR / "inputs")

    gcam_df = pd.read_pickle(DATA_DIR / "inputs" / "projected" / "land_df.pickle")

    model = demeter.run_model(run_dir=str(run_dir), gcamwrapper_df=gcam_df, write_outputs=True, write_logfile=True,
                              diagnostic=1, save_tabular=1, save_netcdf_yr=1)

    out = Path(model.config.output_dir)

    for year in (2010, 2015):
        assert (out / "spatial_landcover_tabular" / f"landcover_{year}_timestep.csv").is_file()
        assert len(list((out / "spatial_landcover_netcdf").glob(f"*_{year}.nc"))) == 1

        for name in ("intensification_pass_one_diag", "intensification_pass_two_diag", "expansion_diag"):
            diag = pd.read_csv(out / "diagnostics" / f"{name}_{year}.csv")
            assert list(diag.columns) == ["region_id", "metric_id", "from_pft", "to_pft", "target_value"]
            assert len(diag) > 0

    assert (out / "diagnostics" / "harmonization_coeff.npy").is_file()
    assert len(list((out / "log_files").glob("*.log"))) == 1
