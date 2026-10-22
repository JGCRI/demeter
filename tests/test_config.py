"""Configuration parsing and validation."""

import os
from pathlib import Path

import pytest

from demeter.config_reader import ReadConfig, ValidationException

DATA_DIR = Path(__file__).resolve().parents[1] / "demeter" / "tests" / "data"


def _config(**kwargs):
    params = dict(run_dir=str(DATA_DIR), write_outputs=False, write_logfile=False, projected_lu_file="land.csv")
    params.update(kwargs)
    return ReadConfig(params)


def test_defaults_from_keywords():
    c = _config()
    assert c.start_year == 2010
    assert c.end_year == 2015
    assert c.timestep == 5
    assert c.area_tolerance_scale == 1.0
    assert c.errortol == 0.001


def test_errortol_scales_with_resolution():
    c = _config(spatial_resolution=0.125)
    assert c.area_tolerance_scale == pytest.approx(0.25)
    assert c.errortol == pytest.approx(0.00025)


def test_explicit_errortol_is_absolute():
    c = _config(spatial_resolution=0.125, errortol=0.01)
    assert c.errortol == 0.01


def test_year_upper_bound_message():
    with pytest.raises(ValidationException, match="must be <= 3000"):
        ReadConfig.ck_yr(3001, "end_year")


def test_selection_threshold_validated_by_name():
    with pytest.raises(ValidationException, match="selection_threshold"):
        _config(selection_threshold=2.0)


def test_timestep_larger_than_range_rejected():
    with pytest.raises(ValidationException, match="too large"):
        _config(start_year=2010, end_year=2015, timestep=10)


def test_keywords_override_config_file(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(f"""[STRUCTURE]
run_dir = {DATA_DIR}

[INPUTS]
    [[ALLOCATION]]
    [[OBSERVED]]
    [[PROJECTED]]
    projected_lu_file = land.csv
    [[MAPPING]]

[PARAMS]
start_year = 2010
end_year = 2015
timestep = 5
scenario = from_file
""")
    c = ReadConfig(dict(config_file=str(ini), write_outputs=False, write_logfile=False, scenario="from_kwargs"))
    assert c.scenario == "from_kwargs"
    assert c.start_year == 2010

    c = ReadConfig(dict(config_file=str(ini), write_outputs=False, write_logfile=False))
    assert c.scenario == "from_file"


def test_relative_run_dir_not_duplicated(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "inputs").symlink_to(DATA_DIR / "inputs")
    monkeypatch.chdir(tmp_path)

    c = ReadConfig(dict(run_dir="run", write_outputs=True, write_logfile=False, save_netcdf_yr=0,
                         projected_lu_file="land.csv"))
    assert Path(c.output_dir).parent == Path("run") / "outputs"
    assert Path(c.output_dir).is_dir()


def test_output_dirs_unique_within_same_second(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "inputs").symlink_to(DATA_DIR / "inputs")

    params = dict(run_dir=str(run_dir), write_outputs=True, write_logfile=False, save_netcdf_yr=0,
                  projected_lu_file="land.csv")
    a = ReadConfig(dict(params))
    b = ReadConfig(dict(params))
    b.dt = a.dt
    b_dir = b.get_outdir(os.path.join(str(run_dir), "outputs"))
    assert b_dir != a.output_dir
