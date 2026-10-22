"""Run a Demeter benchmark case and record accuracy artifacts and performance metrics.

Each repeat runs in a fresh subprocess (so peak memory is per run) inside a temporary working directory (so stray
files written to the CWD do not pollute the repository).  Thread counts of numerical libraries are pinned to one.

Usage:

    python benchmarks/run_benchmark.py --case global_0p5deg --out benchmarks/results/<label> --repeat 3
    python benchmarks/run_benchmark.py --case global_0p5deg --out /tmp/v202 --code-path /tmp/demeter_v2.0.2

Outputs written to ``--out``:

    manifest.json               environment, code revision, input checksums, noise floor, timing summary
    run_<i>/metrics.json        per-run accuracy and performance metrics
    run_<i>/stdout.log          model console output
    run_0/base.npz              initial state (coordinates, region, metric, harmonized base land cover)
    run_0/step_<year>.npz       per-step land cover state, model output, target/residual change, region totals

"""

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent

DETERMINISM_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "xarray", "netCDF4", "configobj", "gcamreader"]

PERF_PATTERN = re.compile(r"PERFORMANCE:\s+(?P<label>.*?)\s+(?:completed\s+)?in\s+(?P<value>[-+\d.eE]+)\s+(?P<unit>seconds|minutes)")


def sha256_file(path, chunk=1 << 20):
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr):
    """Return the SHA-256 hex digest of an array's dtype, shape, and contiguous bytes."""
    import numpy as np

    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def peak_rss_bytes():
    """Return the peak resident set size of this process in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def git_info(path):
    """Return the git revision and dirty state for the repository containing path."""
    def _git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()

    sha = _git("rev-parse", "HEAD")
    return dict(sha=sha or None,
                describe=_git("describe", "--tags", "--always", "--dirty") or None,
                branch=_git("rev-parse", "--abbrev-ref", "HEAD") or None,
                dirty=bool(_git("status", "--porcelain", "--untracked-files=no")))


def package_versions():
    """Return installed versions of the numerical stack."""
    from importlib import metadata

    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class _CaptureHandler(logging.Handler):
    """Collect log records emitted by the Demeter runtime logger."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(dict(t=record.created, level=record.levelname, msg=record.getMessage()))


def _parse_performance(messages):
    """Extract the model's own 'PERFORMANCE:' timing log lines as seconds."""
    out = []
    for m in messages:
        match = PERF_PATTERN.search(m["msg"])
        if match:
            value = float(match.group("value"))
            if match.group("unit") == "minutes":
                value *= 60.0
            out.append(dict(label=match.group("label"), seconds=value))
    return out


def _region_totals(np, s, spat_ludataharm):
    """Sum land cover area per (region, original metric id) for every land class."""
    revert = {v: k for k, v in s.sequence_metric_dict.items()}
    metric_orig = np.array([revert.get(i, -1) for i in s.spat_aez])
    keys = np.stack([s.spat_region.astype(np.int64), metric_orig.astype(np.int64)], axis=1)
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    totals = np.zeros((len(groups), spat_ludataharm.shape[1]))
    np.add.at(totals, inverse.ravel(), spat_ludataharm)
    return groups, totals


def single_run(args):
    """Execute one model run in the current process and write its artifacts."""
    sys.path.insert(0, str(Path(args.code_path).resolve()))
    sys.path.insert(1, str(BENCH_DIR))

    import numpy as np
    import demeter
    import cases

    run_dir = Path(args.run_out)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_arrays = args.save_arrays

    capture = _CaptureHandler()
    logging.getLogger("demeter_runtime").addHandler(capture)

    spec = cases.get_case(args.case, args.code_path, write_outputs=args.write_outputs)
    kwargs = spec["kwargs"]

    # checksum inputs before running since some cases remove their temporary inputs afterward
    input_hashes = {p: sha256_file(p) for p in spec["inputs"]} if save_arrays else {}

    timings = {}
    steps = []

    try:
        t_start = time.perf_counter()
        model = demeter.Model(**kwargs)
        timings["config_s"] = time.perf_counter() - t_start

        t0 = time.perf_counter()
        model.initialize()
        timings["initialize_s"] = time.perf_counter() - t0

        s = model.s
        base = s.spat_ludataharm.copy()
        base_cell_total = base.sum(axis=1)

        if save_arrays:
            revert = {v: k for k, v in s.sequence_metric_dict.items()}
            np.savez_compressed(run_dir / "base.npz",
                                spat_coords=s.spat_coords,
                                spat_region=s.spat_region,
                                spat_metric_seq=s.spat_aez,
                                spat_metric=np.array([revert.get(i, -1) for i in s.spat_aez]),
                                cellarea=s.cellarea,
                                celltrunk=s.celltrunk,
                                spat_ludataharm=base,
                                final_landclasses=np.array(s.final_landclasses))

        for year in list(s.user_years):
            t0 = time.perf_counter()
            step = next(model.step_generator)
            elapsed = time.perf_counter() - t0

            harm = s.spat_ludataharm.copy()
            output_df = step.output_df
            output = output_df.to_numpy() if output_df is not None else np.empty((0, 0))
            output_columns = list(output_df.columns) if output_df is not None else []
            groups, totals = _region_totals(np, s, harm)

            cell_total = harm.sum(axis=1)
            mismatch = np.asarray(step.land_mismatch)
            residual = np.asarray(step.target_change)
            target_abs = float(np.sum(np.abs(mismatch)) / 2.0)
            residual_abs = float(np.sum(np.abs(residual)) / 2.0)

            metrics = dict(
                year=int(year),
                step_s=elapsed,
                n_cells=int(harm.shape[0]),
                n_landclasses=int(harm.shape[1]),
                total_area_km2=float(harm.sum()),
                conservation_max_abs_cell_change_km2=float(np.max(np.abs(cell_total - base_cell_total))),
                conservation_total_change_km2=float(cell_total.sum() - base_cell_total.sum()),
                n_negative=int(np.sum(harm < -1e-9)),
                min_value=float(harm.min()),
                n_nonfinite=int(np.sum(~np.isfinite(harm))),
                target_change_abs_km2=target_abs,
                residual_change_abs_km2=residual_abs,
                non_achieved_pct=(residual_abs / target_abs * 100.0) if target_abs > 0 else 0.0,
                sha256_spat_ludataharm=sha256_array(harm),
                sha256_output=sha256_array(output),
                output_shape=list(output.shape),
            )
            steps.append(metrics)

            if save_arrays:
                np.savez_compressed(run_dir / f"step_{year}.npz",
                                    spat_ludataharm=harm,
                                    output=output,
                                    output_columns=np.array(output_columns),
                                    land_mismatch=mismatch,
                                    target_change=residual,
                                    region_metric=groups,
                                    region_metric_totals=totals)

        model.cleanup()
        timings["total_s"] = time.perf_counter() - t_start

    finally:
        if spec.get("cleanup"):
            spec["cleanup"]()

    result = dict(
        case=args.case,
        demeter_file=demeter.__file__,
        timings=timings,
        model_performance=_parse_performance(capture.messages),
        peak_rss_bytes=peak_rss_bytes(),
        steps=steps,
        n_warnings=sum(1 for m in capture.messages if m["level"] == "WARNING"),
        n_errors=sum(1 for m in capture.messages if m["level"] == "ERROR"),
    )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # record input checksums once, from the run that saved arrays
    if save_arrays:
        with open(run_dir / "inputs.json", "w") as f:
            json.dump(input_hashes, f, indent=2)


def _noise_floor(out_dir, n_runs):
    """Compare every repeat against run 0 and report the largest difference observed."""
    import numpy as np

    metrics = [json.loads((out_dir / f"run_{i}" / "metrics.json").read_text()) for i in range(n_runs)]
    hashes = [[s["sha256_spat_ludataharm"] for s in m["steps"]] for m in metrics]
    identical = all(h == hashes[0] for h in hashes)

    floor = dict(identical_across_runs=identical, max_abs_diff_km2=0.0)
    if not identical:
        max_diff = 0.0
        for i in range(1, n_runs):
            for step in metrics[0]["steps"]:
                a = np.load(out_dir / "run_0" / f"step_{step['year']}.npz")["spat_ludataharm"]
                b = np.load(out_dir / f"run_{i}" / f"step_{step['year']}.npz")["spat_ludataharm"]
                max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        floor["max_abs_diff_km2"] = max_diff
    return floor, metrics


def _summarize_timings(metrics):
    """Median of each timing across repeats."""
    import statistics

    keys = sorted({k for m in metrics for k in m["timings"]})
    summary = {k: statistics.median(m["timings"][k] for m in metrics if k in m["timings"]) for k in keys}
    summary["peak_rss_bytes"] = statistics.median(m["peak_rss_bytes"] for m in metrics)

    years = [s["year"] for s in metrics[0]["steps"]]
    for idx, year in enumerate(years):
        summary[f"step_{year}_s"] = statistics.median(m["steps"][idx]["step_s"] for m in metrics)

    labels = [p["label"] for p in metrics[0]["model_performance"]]
    for idx, label in enumerate(labels):
        try:
            summary[f"model: {label}"] = statistics.median(m["model_performance"][idx]["seconds"] for m in metrics)
        except IndexError:
            pass
    return summary


def orchestrate(args):
    """Run all repeats in subprocesses and write the manifest."""
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            sys.exit(f"Output directory '{out_dir}' is not empty; use --force to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    code_path = Path(args.code_path).resolve()
    env = dict(os.environ, **DETERMINISM_ENV)
    env.pop("PYTHONPATH", None)

    for i in range(args.repeat):
        run_out = out_dir / f"run_{i}"
        run_out.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--case", args.case,
               "--code-path", str(code_path), "--run-out", str(run_out)]
        if args.write_outputs:
            cmd.append("--write-outputs")
        cmd.append("--save-arrays")

        print(f"[{args.case}] run {i + 1}/{args.repeat} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="demeter_bench_cwd_") as cwd, \
                open(run_out / "stdout.log", "w") as log:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            stray = sorted(os.listdir(cwd))

        if proc.returncode != 0:
            tail = (run_out / "stdout.log").read_text().splitlines()[-25:]
            sys.exit(f"Run {i} failed with exit code {proc.returncode}:\n" + "\n".join(tail))

        if stray:
            m = json.loads((run_out / "metrics.json").read_text())
            m["stray_cwd_files"] = stray
            (run_out / "metrics.json").write_text(json.dumps(m, indent=2))

    floor, metrics = _noise_floor(out_dir, args.repeat)

    # keep full arrays only for run 0 unless asked otherwise
    if not args.keep_all_arrays:
        for i in range(1, args.repeat):
            for f in (out_dir / f"run_{i}").glob("*.npz"):
                f.unlink()
            (out_dir / f"run_{i}" / "inputs.json").unlink(missing_ok=True)

    manifest = dict(
        case=args.case,
        label=args.label or out_dir.name,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code=dict(path=str(code_path), demeter_file=metrics[0]["demeter_file"], **git_info(code_path)),
        harness=git_info(REPO_ROOT),
        environment=dict(python=platform.python_version(),
                         implementation=platform.python_implementation(),
                         platform=platform.platform(),
                         machine=platform.machine(),
                         cpu_count=os.cpu_count(),
                         packages=package_versions(),
                         determinism_env=DETERMINISM_ENV),
        repeats=args.repeat,
        write_outputs=args.write_outputs,
        noise_floor=floor,
        timing_summary=_summarize_timings(metrics),
        years=[s["year"] for s in metrics[0]["steps"]],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(dict(out=str(out_dir), noise_floor=floor, timing_summary=manifest["timing_summary"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="Benchmark case name (see benchmarks/cases.py)")
    parser.add_argument("--out", help="Output directory for the benchmark results")
    parser.add_argument("--repeat", type=int, default=3, help="Number of repeats (default 3)")
    parser.add_argument("--code-path", default=str(REPO_ROOT),
                        help="Root of the demeter source tree to benchmark (default: this repository)")
    parser.add_argument("--label", help="Label recorded in the manifest (default: output directory name)")
    parser.add_argument("--write-outputs", action="store_true", help="Let the model write its output files")
    parser.add_argument("--keep-all-arrays", action="store_true", help="Keep arrays for every repeat")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory")

    # internal: used by the orchestrator to execute a single run in a subprocess
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-out", help=argparse.SUPPRESS)
    parser.add_argument("--save-arrays", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.single:
        single_run(args)
    else:
        if not args.out:
            parser.error("--out is required")
        orchestrate(args)


if __name__ == "__main__":
    main()
