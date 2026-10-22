# Demeter benchmarks

Accuracy and performance benchmarks used to detect **any** change in model outcomes across the v3.0.0 upgrade.
Every change is compared against a frozen baseline; changes that alter outputs must be re-baselined explicitly.

## Cases

| Case | Data | Notes |
|---|---|---|
| `global_0p5deg` | bundled `demeter/tests/data` (32 regions, 235 basins, 2010–2015) | matches the historical regression test configuration |
| `global_0p5deg_negcons` | as above with the `soil_quality` constraint made negative | exercises negative-constraint handling |
| `texas_150m` | user-supplied; set `DEMETER_BENCH_150M_CONFIG=/path/to/config.ini` | data stays outside git |

## Workflow

```bash
# 1. run the benchmark for the current working tree (3 repeats; each in its own subprocess and temporary CWD)
python benchmarks/run_benchmark.py --case global_0p5deg --out benchmarks/results/my_change --repeat 3

# 2. compare against the current baseline
python benchmarks/compare.py benchmarks/baselines/v3.0.0-b0/global_0p5deg benchmarks/results/my_change

# 3. only for an intentional, reviewed output change: promote a new baseline and log it
python benchmarks/promote_baseline.py benchmarks/results/my_change --id v3.0.0-b1
#    then add an entry to benchmarks/BASELINE_CHANGELOG.md and bump BASELINE_ID in tests/test_regression_global.py
```

Benchmark any revision with `--code-path`, e.g. a worktree of a tag:

```bash
git worktree add --detach /tmp/demeter_v2.0.2 v2.0.2
python benchmarks/run_benchmark.py --case global_0p5deg --code-path /tmp/demeter_v2.0.2 --out benchmarks/results/v2.0.2
```

## Verdicts (`compare.py`)

| Verdict | Meaning | Exit |
|---|---|---|
| `IDENTICAL` | every land cover state array is bit-for-bit identical | 0 |
| `WITHIN_TOLERANCE` | all values within `atol + rtol·|base|` (defaults 1e-6 km², 1e-9); atol is widened to the measured noise floor | 0 |
| `DRIFT` | outputs changed beyond tolerance; requires re-baseline + changelog entry | 1 |
| `INCOMPATIBLE` | shapes, years, or land classes differ | 1 |

Performance is compared on the median across repeats; slowdowns over `--perf-threshold` (default 10%) are flagged,
and `--fail-on-perf` turns them into exit code 3.

## What is recorded

- **Accuracy (per step):** harmonized land cover state (km², cells × land classes), model output frame,
  region × basin × land class totals, initial target change and residual (non-achieved) change, per-cell area
  conservation, negative / non-finite counts, SHA-256 of state and output.
- **Performance:** wall time for configuration, staging, and each step; the model's own `PERFORMANCE:` log timings;
  peak RSS per run.
- **Provenance:** code revision (sha, describe, dirty), harness revision, Python and package versions, platform,
  CPU count, pinned thread environment, input file checksums, and run-to-run noise floor.

## Baseline storage

`benchmarks/baselines/<id>/<case>/` commits only small files (`manifest.json`, `run_*/metrics.json`,
`run_0/inputs.json`, `summary_<year>.npz`, `SHA256SUMS`). The full arrays (`run_0/base.npz`, `run_0/step_*.npz`)
are gitignored, distributed separately, and verified with `SHA256SUMS`. The pytest regression test
(`tests/test_regression_global.py`) only needs the committed summaries.

## Determinism

Thread counts for BLAS/OpenMP are pinned to 1 during benchmarks. Without pinning, the model's thread pools
oversubscribe the CPU and the global case runs ~2.6× slower (90 s vs 34 s on a 16-core M-series Mac).
