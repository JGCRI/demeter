# Benchmark comparison: `texas-probe` → `out-1.14-texas_0p5deg`

**Accuracy verdict: DRIFT**  (atol=1e-06 km², rtol=1e-09, noise floor=0 km²)

| | base | new |
|---|---|---|
| case | texas_0p5deg | texas_0p5deg |
| code | v2.0.1-30-ga3b7d79-dirty | v2.0.1-30-ga3b7d79-dirty |
| python | 3.11.7 | 3.11.7 |
| numpy / pandas / scipy | 1.26.4 / 2.2.1 / 1.12.0 | 1.26.4 / 2.2.1 / 1.12.0 |
| repeats | 1 | 3 |
| deterministic across repeats | True | True |

## Accuracy by time step

| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |
|---|---|---|---|---|---|
| 2010 | DRIFT | 970.5 | 123.371 | 412 | 0 → 37.9805 |
| 2015 | DRIFT | 1597.09 | 136.298 | 412 | 0 → 81.9844 |

### 2010: largest region × basin × land class changes (156 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 225 | tundra | 97545.3 | 20589.4 | -76955.9 |
| 1 | 225 | unmanagedpasture | 0 | 74554.8 | 74554.8 |
| 1 | 224 | tundra | 65808.3 | 1676.38 | -64131.9 |
| 1 | 224 | unmanagedpasture | 0 | 59548 | 59548 |
| 1 | 225 | otherarableland | 0 | 50749.2 | 50749.2 |
| 1 | 224 | grassland | 65808.3 | 27125.7 | -38682.6 |
| 1 | 224 | otherarableland | 0 | 37239.2 | 37239.2 |
| 1 | 231 | tundra | 37705.1 | 2549.32 | -35155.8 |
| 1 | 225 | forest | 39543.2 | 8015.38 | -31527.8 |
| 1 | 231 | unmanagedpasture | 0 | 22045.5 | 22045.5 |
| 1 | 224 | forest | 28141.6 | 6759.6 | -21382 |
| 1 | 225 | grassland | 97545.3 | 83563.4 | -13981.9 |
| 1 | 231 | grassland | 37705.1 | 24529.8 | -13175.3 |
| 1 | 231 | unmanagedforest | 2620.84 | 13685.1 | 11064.2 |
| 1 | 231 | otherarableland | 0 | 9920.34 | 9920.34 |

### 2015: largest region × basin × land class changes (156 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 225 | unmanagedpasture | 0 | 97904.5 | 97904.5 |
| 1 | 225 | tundra | 97545.3 | 2.80844e-05 | -97545.3 |
| 1 | 224 | tundra | 65808.3 | 0 | -65808.3 |
| 1 | 224 | unmanagedpasture | 0 | 63129.1 | 63129.1 |
| 1 | 225 | otherarableland | 0 | 46989.2 | 46989.2 |
| 1 | 231 | tundra | 37705.1 | 4.62513e-05 | -37705.1 |
| 1 | 224 | grassland | 65808.3 | 28169.6 | -37638.7 |
| 1 | 224 | otherarableland | 0 | 34190.1 | 34190.1 |
| 1 | 225 | forest | 39543.2 | 8358.53 | -31184.7 |
| 1 | 231 | unmanagedpasture | 0 | 25734.1 | 25734.1 |
| 1 | 224 | forest | 28141.6 | 7159.45 | -20982.1 |
| 1 | 231 | grassland | 37705.1 | 23360.1 | -14345 |
| 1 | 225 | grassland | 97545.3 | 84298.7 | -13246.6 |
| 1 | 231 | unmanagedforest | 2620.84 | 13901.9 | 11281.1 |
| 1 | 231 | otherarableland | 0 | 9300.53 | 9300.53 |

## Performance (median across repeats)

Regression threshold: +10% (timings under 1 s are not flagged)

| metric | base | new | change |
|---|---|---|---|
| config_s | 0.000 s | 0.000 s | -8.3% |
| initialize_s | 0.022 s | 0.021 s | -2.3% |
| total_s | 0.106 s | 0.236 s | +123.4% |
| peak_rss_bytes | 294.420 MB | 294.273 MB | -0.1% |
| step_2010_s | 0.043 s | 0.129 s | +198.9% |
| step_2015_s | 0.024 s | 0.069 s | +181.3% |
| model: Allocation files processed | 0.005 s | 0.005 s | -2.6% |
| model: Projected landuse data prepared | 0.012 s | 0.012 s | -3.0% |
| model: Base spatial landuse data prepared | 0.002 s | 0.002 s | -6.0% |
| model: Harmonization | 0.000 s | 0.000 s | +123.0% |
| model: Constraints applied to projected and spatial data | 0.001 s | 0.001 s | -0.5% |
| model: Kernel density filter prepared | 0.000 s | 0.000 s | -11.0% |
