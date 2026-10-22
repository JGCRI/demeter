# Benchmark comparison: `v2.0.2-global-negcons` → `out-1.1-global_0p5deg_negcons`

**Accuracy verdict: DRIFT**  (atol=9.40846 km², rtol=1e-09, noise floor=9.40846 km²)

| | base | new |
|---|---|---|
| case | global_0p5deg_negcons | global_0p5deg_negcons |
| code | v2.0.2 | v2.0.1-29-g972c75d-dirty |
| python | 3.11.7 | 3.11.7 |
| numpy / pandas / scipy | 1.26.4 / 2.2.1 / 1.12.0 | 1.26.4 / 2.2.1 / 1.12.0 |
| repeats | 3 | 3 |
| deterministic across repeats | False | True |

> ⚠️ **Nondeterministic results:** repeats differ by up to 9.40846 km², so the tolerance was widened from 1e-06 to 9.40846 km². Real drift smaller than this cannot be detected.

## Accuracy by time step

| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |
|---|---|---|---|---|---|
| 2010 | DRIFT | 1612.11 | 7.79212 | 11808 | 31.9859 → 31.9802 |
| 2015 | DRIFT | 1547.22 | 8.13596 | 12418 | 88.6525 → 88.6327 |

### 2010: largest region × basin × land class changes (7222 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 4 | 171 | grassland | 305951 | 314407 | 8455.54 |
| 4 | 171 | unmanagedpasture | 64770.5 | 56314.9 | -8455.54 |
| 10 | 56 | unmanagedpasture | 206915 | 211699 | 4784.03 |
| 10 | 56 | rockicedesert | 25940.9 | 21209.3 | -4731.59 |
| 10 | 42 | rockicedesert | 95224.9 | 98017.4 | 2792.45 |
| 10 | 42 | unmanagedpasture | 569166 | 566404 | -2762.08 |
| 14 | 39 | unmanagedpasture | 21449.6 | 18908.3 | -2541.3 |
| 14 | 39 | otherarableland | 31070 | 33545.2 | 2475.16 |
| 17 | 123 | otherarableland | 9808.79 | 12137.6 | 2328.85 |
| 27 | 58 | tundra | 4807 | 6831.59 | 2024.59 |
| 27 | 58 | unmanagedpasture | 21291 | 19266.4 | -2024.59 |
| 17 | 123 | grassland | 68247.9 | 66641 | -1606.9 |
| 1 | 229 | shrubland | 72191 | 73605.3 | 1414.31 |
| 17 | 140 | otherarableland | 20426.7 | 19071.6 | -1355.03 |
| 11 | 106 | unmanagedpasture | 9133.51 | 7888.2 | -1245.31 |

### 2015: largest region × basin × land class changes (8572 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 14 | 39 | unmanagedpasture | 21449.6 | 18908.3 | -2541.3 |
| 14 | 39 | otherarableland | 31278.2 | 33711.3 | 2433.16 |
| 17 | 123 | otherarableland | 9808.79 | 12137.6 | 2328.85 |
| 17 | 123 | grassland | 69716.8 | 68109.9 | -1606.9 |
| 2 | 151 | unmanagedpasture | 37866.3 | 36302 | -1564.33 |
| 2 | 151 | otherarableland | 7878.89 | 9443.22 | 1564.33 |
| 1 | 229 | shrubland | 72546.8 | 73961.1 | 1414.31 |
| 17 | 140 | otherarableland | 21878.2 | 20518.9 | -1359.34 |
| 11 | 106 | unmanagedpasture | 9133.51 | 7888.2 | -1245.31 |
| 1 | 229 | unmanagedpasture | 4710.1 | 3502.21 | -1207.89 |
| 17 | 140 | shrubland | 18690.5 | 19888.3 | 1197.81 |
| 11 | 106 | grassland | 308911 | 309851 | 940.178 |
| 17 | 123 | unmanagedpasture | 2025.24 | 1144.5 | -880.742 |
| 2 | 183 | grassland | 12065.4 | 11338.3 | -727.116 |
| 5 | 142 | unmanagedpasture | 13101.8 | 13828.8 | 727.015 |

## Performance (median across repeats)

Regression threshold: +10% (timings under 1 s are not flagged)

| metric | base | new | change |
|---|---|---|---|
| config_s | 0.000 s | 0.000 s | +24.0% |
| initialize_s | 0.577 s | 0.335 s | -42.0% |
| total_s | 34.215 s | 17.020 s | -50.3% |
| peak_rss_bytes | 5111.202 MB | 1894.842 MB | -62.9% |
| step_2010_s | 17.494 s | 9.533 s | -45.5% |
| step_2015_s | 14.988 s | 5.945 s | -60.3% |
| model: Allocation files processed | 0.004 s | 0.005 s | +2.8% |
| model: Projected landuse data prepared | 0.060 s | 0.081 s | +34.6% |
| model: Base spatial landuse data prepared | 0.114 s | 0.119 s | +4.4% |
| model: Harmonization | 0.091 s | 0.097 s | +6.3% |
| model: Constraints applied to projected and spatial data | 0.031 s | 0.031 s | -0.7% |
| model: Kernel density filter prepared | 0.268 s | 0.007 s | -97.6% |
