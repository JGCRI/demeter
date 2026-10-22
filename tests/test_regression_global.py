"""Regression test of the bundled global 0.5 degree case against the committed benchmark baseline.

Compares region x basin x land class area totals and the residual (non-achieved) target change for every time step
against ``benchmarks/baselines/<BASELINE_ID>/global_0p5deg/summary_<year>.npz``.  Cell-level comparison is done by
``benchmarks/compare.py`` against the full baseline arrays.

"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import cases  # noqa: E402
from run_benchmark import _region_totals  # noqa: E402

BASELINE_ID = "v3.0.0-b4"
BASELINE_DIR = BENCH_DIR / "baselines" / BASELINE_ID / "global_0p5deg"

# absolute tolerance in km2 for aggregated areas; cross-platform floating point noise is ~1e-11 km2 per cell
ATOL_KM2 = 1e-6
RTOL = 1e-9


@pytest.mark.slow
def test_global_0p5deg_matches_baseline():
    import demeter

    spec = cases.get_case("global_0p5deg", REPO_ROOT)
    model = demeter.Model(**spec["kwargs"])
    model.initialize()

    try:
        for year in list(model.s.user_years):
            step = next(model.step_generator)
            expected = np.load(BASELINE_DIR / f"summary_{year}.npz")

            groups, totals = _region_totals(np, model.s, model.s.spat_ludataharm)

            np.testing.assert_array_equal(groups, expected["region_metric"],
                                          err_msg=f"{year}: region/basin set differs from baseline")
            np.testing.assert_allclose(totals, expected["region_metric_totals"], rtol=RTOL, atol=ATOL_KM2,
                                       err_msg=f"{year}: region x basin x land class areas differ from baseline")
            np.testing.assert_allclose(step.target_change, expected["target_change"], rtol=RTOL, atol=ATOL_KM2,
                                       err_msg=f"{year}: residual target change differs from baseline")
    finally:
        model.cleanup()
