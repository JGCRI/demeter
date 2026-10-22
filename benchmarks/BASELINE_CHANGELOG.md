# Baseline changelog

Every intentional change to model outputs is recorded here with the `compare.py` verdict that justified it.

## v3.0.0-b0 — 2026-10-22

Initial frozen baseline for the v3.0.0 upgrade.

| Case | Code | Deterministic | Noise floor | Median total | Peak RSS |
|---|---|---|---|---|---|
| `global_0p5deg` | `v2.0.2` (`8bcd4b3`) | yes (3/3 bit-identical) | 0 km² | 34.0 s | 5.1 GB |
| `global_0p5deg_negcons` | `v2.0.2` (`8bcd4b3`) | **no** | 9.41 km² | ~34 s | ~5.1 GB |
| `texas_150m` | `version/v3.0.0` (`da6ce72`) | pending — needs `DEMETER_BENCH_150M_CONFIG` | | | |

Environment: Python 3.11.7, numpy 1.26.4, pandas 2.2.1, scipy 1.12.0, xarray 2024.2.0, netCDF4 1.6.5, macOS arm64,
16 CPUs, BLAS/OpenMP threads pinned to 1.

Notes:
- The global case is taken from `v2.0.2` because `version/v3.0.0` (`da6ce72`) cannot read the bundled zipped CSV
  basemap (plan item 1.7). Once 1.7 lands, any difference from this baseline is attributable to the branch's
  150 m commits (e.g., the kernel density grid now inferred from data bounds) and will be reviewed as an [OUT] change.
- Reproduces the historical `demeter/tests/data/comp_data` pickles to within 1.1e-11 km² (cross-platform floating
  point noise; the legacy test compared with pandas' default rtol of 1e-5).
- `global_0p5deg_negcons` is nondeterministic run to run, confirming plan item 1.1 (constraint weights mutated in
  place by concurrent region/basin threads). It is kept as a baseline to verify the fix restores determinism.

## v3.0.0-b1 — 2026-10-22 — plan 1.1: constraint weights no longer mutated in place

**Status: pending sign-off.**

`cons_rules_pft = constraint_rules[:, pft]` was a view, so flipping the sign of negative weights changed the shared
array. Only the first region × basin × land class processed (in nondeterministic thread order) applied a negative
constraint as negative; all later ones treated it as positive. The weights are now copied.

| Case | vs b0 | Deterministic | Notes |
|---|---|---|---|
| `global_0p5deg` | IDENTICAL | yes | no negative constraints |
| `global_0p5deg_negcons` | DRIFT: max 1,612 km²/cell; 11,808 (2010) / 12,418 (2015) of 67,420 cells; non-achieved 31.9859 → 31.9802 % (2010), 88.6525 → 88.6327 % (2015) | **yes** (was 9.41 km² run-to-run noise) | reports: `benchmarks/reports/out-1.1_*.md` |

Code: branch `v3/out-1.1-constraint-copy`. Timing: 16.9 s (b0: 34.0 s; speedups from earlier output-neutral commits).

Open question raised by this fix (plan 1.19): negative constraints are "inverted" as `1 + x` (since the first
commit, 513dd79) rather than `1 - x`, which shifts rather than inverts their effect.

## v3.0.0-b2 — 2026-10-22 — plan 1.14: Taiwan placeholder inserted at the wrong region

**Status: pending sign-off.**

With `agg_level=2` the GCAM reader adds a placeholder region (Taiwan, id 30) with no basins. The placeholder was
inserted at `alphabetical index of "Taiwan" - 1` into a list ordered by region **number**. That only matches the
numeric position for the full 32-region set; for any regional subset the basins shift to the wrong region. In a
USA-only domain, USA received the empty list, so **no land change was downscaled at all** (target change 0, output =
base map, no warning). The placeholder is now inserted at the numeric position of region 30.

New case `texas_0p5deg` (bundled USA cells inside a Texas bounding box; regions, basins, projected data, and
constraints subset to match) exercises the regional code path. Its "before" is the pre-fix zero-change run.

| Case | vs previous | Notes |
|---|---|---|
| `global_0p5deg` | IDENTICAL to b1 | alphabetical and numeric positions coincide (29) for 32 regions |
| `global_0p5deg_negcons` | IDENTICAL to b1 | |
| `texas_0p5deg` (new) | DRIFT vs pre-fix: all 412 cells change (max 1,597 km²); target 516,878 km² (2010) / 209,212 km² (2015) now applied; non-achieved 0 → 37.98 % / 81.98 % | report: `benchmarks/reports/out-1.14_texas_0p5deg.md` |

⚠️ High resolution regional runs made with `version/v3.0.0` at `da6ce72` and a reduced region mapping file are
likely affected (output equal to the base map). Not verified: the 150 m inputs are unavailable.

*b2 note:* `texas_0p5deg` was re-recorded with `spatial_resolution=0.5` (matching its data) before b3; the first
recording used the legacy default of 0.25.

## v3.0.0-b3 — 2026-10-22 — plan 1.4: regional kernel density grid aligned to cell centers

**Status: pending sign-off.**

For regional domains the kernel density grid started half a cell inside the data bounds
(`max_lat - res/2`, `min_lon + res/2`). Each cell center was then equidistant from two grid coordinates and the tie
went to the lower index, so the first row and column of cells landed on the same grid positions as the second and
overwrote each other: 35 of 412 cells (8.5%) in `texas_0p5deg`. The regional grid now coincides with the cell centers;
a warning is logged if cells still share positions (e.g., `spatial_resolution` coarser than the data).

| Case | vs b2 | Notes |
|---|---|---|
| `global_0p5deg` | IDENTICAL | global domains use the global grid |
| `global_0p5deg_negcons` | IDENTICAL | |
| `texas_0p5deg` | DRIFT: max 377 km²/cell; 396 (2010) / 406 (2015) of 412 cells; non-achieved 38.19 → 38.13 % / 81.57 → 81.68 % | report: `benchmarks/reports/out-1.4_texas_0p5deg.md` |
