"""Promote a benchmark result directory to a named baseline.

Copies ``run_benchmark.py`` output into ``benchmarks/baselines/<baseline_id>/<case>/`` and writes:

    summary_<year>.npz      small, committed: region x metric x land class totals and residual target change
    SHA256SUMS              checksums of every file, including the large (uncommitted) arrays

The large ``base.npz`` / ``step_<year>.npz`` arrays are gitignored; distribute them separately and verify them
against ``SHA256SUMS``.

Usage:

    python benchmarks/promote_baseline.py benchmarks/results/<dir> --id v3.0.0-b0

"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", help="Benchmark result directory created by run_benchmark.py")
    parser.add_argument("--id", required=True, help="Baseline identifier, e.g. v3.0.0-b0")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing baseline for this case")
    args = parser.parse_args()

    src = Path(args.result_dir).resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    dest = BENCH_DIR / "baselines" / args.id / manifest["case"]

    if dest.exists():
        if not args.force:
            raise SystemExit(f"Baseline '{dest}' exists; use --force to overwrite.")
        shutil.rmtree(dest)

    shutil.copytree(src, dest)

    manifest["baseline_id"] = args.id
    manifest["promoted_from"] = str(src)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))

    landclasses = np.load(dest / "run_0" / "base.npz")["final_landclasses"]
    for year in manifest["years"]:
        step = np.load(dest / "run_0" / f"step_{year}.npz")
        np.savez_compressed(dest / f"summary_{year}.npz",
                            region_metric=step["region_metric"],
                            region_metric_totals=step["region_metric_totals"],
                            target_change=step["target_change"],
                            final_landclasses=landclasses)

    sums = [f"{_sha256(p)}  {p.relative_to(dest)}" for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"]
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    print(f"Promoted {src} -> {dest}")


if __name__ == "__main__":
    main()
