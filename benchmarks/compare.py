"""Compare two Demeter benchmark result directories for accuracy and performance.

Usage:

    python benchmarks/compare.py BASE_DIR NEW_DIR [--atol 1e-6] [--rtol 1e-9] [--perf-threshold 10] [--json OUT]

Verdicts:

    IDENTICAL           every land cover state array is bit-for-bit identical
    WITHIN_TOLERANCE    differences are within ``atol + rtol * |base|`` (atol widened to the larger noise floor)
    DRIFT               outputs changed beyond tolerance -- requires a re-baseline and a BASELINE_CHANGELOG entry
    INCOMPATIBLE        array shapes, years, or land classes differ so values cannot be compared cell by cell

Exit code is 0 for IDENTICAL / WITHIN_TOLERANCE, 1 for DRIFT / INCOMPATIBLE, and 3 for a performance regression
when ``--fail-on-perf`` is set.

"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VERDICT_ORDER = ["IDENTICAL", "WITHIN_TOLERANCE", "DRIFT", "INCOMPATIBLE"]


def _load(result_dir):
    """Load the manifest, run-0 metrics, inputs, and array paths of a result directory."""
    d = Path(result_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    metrics = json.loads((d / "run_0" / "metrics.json").read_text())
    inputs_file = d / "run_0" / "inputs.json"
    inputs = json.loads(inputs_file.read_text()) if inputs_file.exists() else {}
    return dict(dir=d, manifest=manifest, metrics=metrics, inputs=inputs)


def _relative_input_key(path):
    """Key an input file by its path below the 'inputs' directory so different checkouts compare."""
    parts = Path(path).parts
    if "inputs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("inputs")
        return "/".join(parts[idx:])
    return Path(path).name


def compare_inputs(base, new):
    """Report input files whose checksums differ or that exist in only one run."""
    b = {_relative_input_key(k): v for k, v in base["inputs"].items()}
    n = {_relative_input_key(k): v for k, v in new["inputs"].items()}
    return dict(changed=sorted(k for k in b.keys() & n.keys() if b[k] != n[k]),
                only_base=sorted(b.keys() - n.keys()),
                only_new=sorted(n.keys() - b.keys()))


def _array_diff(a, b, atol, rtol):
    """Summarize differences between two equally shaped arrays."""
    diff = b - a
    absdiff = np.abs(diff)
    exceed = absdiff > (atol + rtol * np.abs(a))
    rows_exceed = exceed.any(axis=1) if exceed.ndim > 1 else exceed
    return dict(identical=bool(np.array_equal(a, b)),
                max_abs=float(absdiff.max()) if absdiff.size else 0.0,
                rmse=float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
                sum_abs=float(absdiff.sum()),
                n_cells_exceeding=int(rows_exceed.sum()),
                n_values_exceeding=int(exceed.sum()),
                within_tolerance=bool(not exceed.any()))


def _top_region_diffs(b, n, landclasses, top):
    """Largest changes in area per (region, metric, land class)."""
    bk = {tuple(k): i for i, k in enumerate(b["region_metric"].tolist())}
    nk = {tuple(k): i for i, k in enumerate(n["region_metric"].tolist())}
    rows = []
    for key in bk.keys() | nk.keys():
        bv = b["region_metric_totals"][bk[key]] if key in bk else np.zeros(len(landclasses))
        nv = n["region_metric_totals"][nk[key]] if key in nk else np.zeros(len(landclasses))
        for j, (x, y) in enumerate(zip(bv, nv)):
            if x != y:
                rows.append(dict(region_id=int(key[0]), metric_id=int(key[1]), landclass=str(landclasses[j]),
                                 base_km2=float(x), new_km2=float(y), diff_km2=float(y - x)))
    rows.sort(key=lambda r: abs(r["diff_km2"]), reverse=True)
    return rows[:top], len(rows)


def compare_accuracy(base, new, atol, rtol, top):
    """Compare per-step land cover state, model output, and aggregate metrics."""
    floor = max(base["manifest"]["noise_floor"]["max_abs_diff_km2"], new["manifest"]["noise_floor"]["max_abs_diff_km2"])
    atol_eff = max(atol, floor)

    base_years = [s["year"] for s in base["metrics"]["steps"]]
    new_years = [s["year"] for s in new["metrics"]["steps"]]
    years = [y for y in base_years if y in new_years]

    landclasses = None
    base_npz = base["dir"] / "run_0" / "base.npz"
    if base_npz.exists():
        landclasses = np.load(base_npz)["final_landclasses"]

    steps = []
    for year in years:
        bm = next(s for s in base["metrics"]["steps"] if s["year"] == year)
        nm = next(s for s in new["metrics"]["steps"] if s["year"] == year)
        b = np.load(base["dir"] / "run_0" / f"step_{year}.npz")
        n = np.load(new["dir"] / "run_0" / f"step_{year}.npz")

        entry = dict(year=year)

        if b["spat_ludataharm"].shape != n["spat_ludataharm"].shape:
            entry["verdict"] = "INCOMPATIBLE"
            entry["reason"] = f"state shape {b['spat_ludataharm'].shape} vs {n['spat_ludataharm'].shape}"
            steps.append(entry)
            continue

        state = _array_diff(b["spat_ludataharm"], n["spat_ludataharm"], atol_eff, rtol)
        entry["state"] = state

        if b["output"].shape == n["output"].shape and b["output"].size:
            entry["output"] = _array_diff(b["output"], n["output"], atol_eff, rtol)
            entry["output_columns_equal"] = b["output_columns"].tolist() == n["output_columns"].tolist()
        elif b["output"].shape != n["output"].shape:
            entry["output"] = dict(incompatible=True, base_shape=list(b["output"].shape),
                                   new_shape=list(n["output"].shape))

        entry["residual_change"] = _array_diff(b["target_change"], n["target_change"], atol_eff, rtol) \
            if b["target_change"].shape == n["target_change"].shape else dict(incompatible=True)

        entry["metrics"] = {k: dict(base=bm[k], new=nm[k])
                            for k in ("total_area_km2", "non_achieved_pct", "residual_change_abs_km2",
                                      "conservation_max_abs_cell_change_km2", "n_negative", "n_nonfinite")}

        if landclasses is not None and not state["identical"]:
            entry["top_region_diffs"], entry["n_region_diffs"] = _top_region_diffs(b, n, landclasses, top)

        if state["identical"]:
            entry["verdict"] = "IDENTICAL"
        elif state["within_tolerance"]:
            entry["verdict"] = "WITHIN_TOLERANCE"
        else:
            entry["verdict"] = "DRIFT"
        steps.append(entry)

    if set(base_years) != set(new_years):
        steps.append(dict(year=None, verdict="INCOMPATIBLE",
                          reason=f"years differ: base {base_years} vs new {new_years}"))

    verdict = max((s["verdict"] for s in steps), key=VERDICT_ORDER.index) if steps else "INCOMPATIBLE"
    return dict(verdict=verdict, atol=atol, atol_effective=atol_eff, rtol=rtol, noise_floor_km2=floor, steps=steps)


def compare_performance(base, new, threshold_pct, min_seconds):
    """Compare median timings and peak memory.  Timings shorter than ``min_seconds`` are never flagged."""
    b = base["manifest"]["timing_summary"]
    n = new["manifest"]["timing_summary"]
    rows = []
    for key in [k for k in b if k in n]:
        bv, nv = b[key], n[key]
        pct = ((nv - bv) / bv * 100.0) if bv else None
        too_short = key != "peak_rss_bytes" and max(bv, nv) < min_seconds
        rows.append(dict(metric=key, base=bv, new=nv, pct_change=pct,
                         regression=bool(pct is not None and pct > threshold_pct and not too_short)))
    return dict(threshold_pct=threshold_pct, min_seconds=min_seconds, rows=rows,
                any_regression=any(r["regression"] for r in rows))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_markdown(base, new, inputs, acc, perf):
    """Render the comparison as a markdown report."""
    bm, nm = base["manifest"], new["manifest"]
    lines = [f"# Benchmark comparison: `{bm['label']}` → `{nm['label']}`", "",
             f"**Accuracy verdict: {acc['verdict']}**  (atol={acc['atol_effective']:g} km², rtol={acc['rtol']:g}, "
             f"noise floor={acc['noise_floor_km2']:g} km²)", "",
             "| | base | new |", "|---|---|---|",
             f"| case | {bm['case']} | {nm['case']} |",
             f"| code | {bm['code'].get('describe')} | {nm['code'].get('describe')} |",
             f"| python | {bm['environment']['python']} | {nm['environment']['python']} |",
             f"| numpy / pandas / scipy | {bm['environment']['packages'].get('numpy')} / "
             f"{bm['environment']['packages'].get('pandas')} / {bm['environment']['packages'].get('scipy')} | "
             f"{nm['environment']['packages'].get('numpy')} / {nm['environment']['packages'].get('pandas')} / "
             f"{nm['environment']['packages'].get('scipy')} |",
             f"| repeats | {bm['repeats']} | {nm['repeats']} |",
             f"| deterministic across repeats | {bm['noise_floor']['identical_across_runs']} | "
             f"{nm['noise_floor']['identical_across_runs']} |", ""]

    if acc["atol_effective"] > acc["atol"]:
        lines += [f"> ⚠️ **Nondeterministic results:** repeats differ by up to {acc['noise_floor_km2']:g} km², so the "
                  f"tolerance was widened from {acc['atol']:g} to {acc['atol_effective']:g} km². Real drift smaller "
                  f"than this cannot be detected.", ""]

    if any(inputs.values()):
        lines += ["**Input differences:**", ""]
        for k, v in inputs.items():
            if v:
                lines.append(f"- {k}: {', '.join(v)}")
        lines.append("")

    lines += ["## Accuracy by time step", "",
              "| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |",
              "|---|---|---|---|---|---|"]
    for s in acc["steps"]:
        if "state" not in s:
            lines.append(f"| {s['year']} | {s['verdict']} | — | — | — | {s.get('reason', '')} |")
            continue
        m = s["metrics"]["non_achieved_pct"]
        lines.append(f"| {s['year']} | {s['verdict']} | {_fmt(s['state']['max_abs'])} | {_fmt(s['state']['rmse'])} | "
                     f"{s['state']['n_cells_exceeding']} | {_fmt(m['base'])} → {_fmt(m['new'])} |")
    lines.append("")

    for s in acc["steps"]:
        if s.get("top_region_diffs"):
            lines += [f"### {s['year']}: largest region × basin × land class changes "
                      f"({s['n_region_diffs']} entries changed)", "",
                      "| region | basin | land class | base km² | new km² | Δ km² |", "|---|---|---|---|---|---|"]
            for r in s["top_region_diffs"]:
                lines.append(f"| {r['region_id']} | {r['metric_id']} | {r['landclass']} | {_fmt(r['base_km2'])} | "
                             f"{_fmt(r['new_km2'])} | {_fmt(r['diff_km2'])} |")
            lines.append("")

    lines += ["## Performance (median across repeats)", "",
              f"Regression threshold: +{perf['threshold_pct']:g}% (timings under {perf['min_seconds']:g} s are not "
              f"flagged)", "",
              "| metric | base | new | change |", "|---|---|---|---|"]
    for r in perf["rows"]:
        unit = " MB" if r["metric"] == "peak_rss_bytes" else " s"
        scale = 1 / 1e6 if r["metric"] == "peak_rss_bytes" else 1
        pct = "—" if r["pct_change"] is None else f"{r['pct_change']:+.1f}%"
        flag = " ⚠️" if r["regression"] else ""
        lines.append(f"| {r['metric']} | {r['base'] * scale:.3f}{unit} | {r['new'] * scale:.3f}{unit} | {pct}{flag} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="Baseline benchmark result directory")
    parser.add_argument("new", help="New benchmark result directory")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance in km² (default 1e-6)")
    parser.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance (default 1e-9)")
    parser.add_argument("--perf-threshold", type=float, default=10.0, help="Percent slowdown flagged (default 10)")
    parser.add_argument("--perf-min-seconds", type=float, default=1.0,
                        help="Do not flag timings shorter than this many seconds (default 1.0)")
    parser.add_argument("--top", type=int, default=15, help="Number of region-level changes to list")
    parser.add_argument("--json", help="Write the full comparison as JSON to this file")
    parser.add_argument("--markdown", help="Write the markdown report to this file")
    parser.add_argument("--fail-on-perf", action="store_true", help="Exit 3 on a performance regression")
    args = parser.parse_args()

    base, new = _load(args.base), _load(args.new)
    inputs = compare_inputs(base, new)
    acc = compare_accuracy(base, new, args.atol, args.rtol, args.top)
    perf = compare_performance(base, new, args.perf_threshold, args.perf_min_seconds)

    report = render_markdown(base, new, inputs, acc, perf)
    print(report)

    if args.markdown:
        Path(args.markdown).write_text(report)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(inputs=inputs, accuracy=acc, performance=perf), indent=2))

    if acc["verdict"] in ("DRIFT", "INCOMPATIBLE"):
        sys.exit(1)
    if args.fail_on_perf and perf["any_regression"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
