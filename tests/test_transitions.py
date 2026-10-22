"""Recorded land cover transitions must account for all land change."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import cases  # noqa: E402
from run_benchmark import _region_totals  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("case", ["global_0p5deg", "texas_0p5deg"])
def test_transitions_balance_net_change(case):
    import demeter

    spec = cases.get_case(case, REPO_ROOT)
    model = demeter.Model(**spec["kwargs"], save_transitions=1)
    model.initialize()

    try:
        landclasses = list(model.s.final_landclasses)
        for _ in list(model.s.user_years):
            groups, before = _region_totals(np, model.s, model.s.spat_ludataharm)
            step = next(model.step_generator)
            _, after = _region_totals(np, model.s, model.s.spat_ludataharm)

            net = pd.DataFrame(after - before, columns=landclasses)
            net["region_id"], net["metric_id"] = groups[:, 0], groups[:, 1]
            net = net.melt(id_vars=["region_id", "metric_id"], var_name="landclass", value_name="net")

            t = step.transitions_df
            gains = t.groupby(["region_id", "metric_id", "to_landclass"])["sqkm_change"].sum()
            losses = t.groupby(["region_id", "metric_id", "from_landclass"])["sqkm_change"].sum()
            gains.index.names = losses.index.names = ["region_id", "metric_id", "landclass"]
            recorded = gains.sub(losses, fill_value=0).rename("recorded").reset_index()

            merged = net.merge(recorded, on=["region_id", "metric_id", "landclass"], how="left").fillna(0)

            assert t["sqkm_change"].sum() > 0
            assert (t["from_landclass"] != t["to_landclass"]).all()
            np.testing.assert_allclose(merged["recorded"], merged["net"], rtol=1e-9, atol=1e-6)
    finally:
        model.cleanup()
        if spec.get("cleanup"):
            spec["cleanup"]()
