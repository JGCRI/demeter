# Benchmark comparison: `out-1.4-texas_0p5deg` → `out-1.5-texas`

**Accuracy verdict: DRIFT**  (atol=1e-06 km², rtol=1e-09, noise floor=0 km²)

| | base | new |
|---|---|---|
| case | texas_0p5deg | texas_0p5deg |
| code | v2.0.1-31-g0369b84-dirty | v2.0.1-32-g97238ae-dirty |
| python | 3.11.7 | 3.11.7 |
| numpy / pandas / scipy | 1.26.4 / 2.2.1 / 1.12.0 | 1.26.4 / 2.2.1 / 1.12.0 |
| repeats | 3 | 3 |
| deterministic across repeats | True | True |

## Accuracy by time step

| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |
|---|---|---|---|---|---|
| 2010 | DRIFT | 515.59 | 10.6513 | 397 | 38.1316 → 38.2917 |
| 2015 | DRIFT | 534.055 | 10.6328 | 408 | 81.6795 → 81.359 |

### 2010: largest region × basin × land class changes (94 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 224 | unmanagedpasture | 59623.7 | 59145.2 | -478.437 |
| 1 | 224 | tundra | 1603.08 | 2079.17 | 476.097 |
| 1 | 231 | tundra | 2579.54 | 2930.65 | 351.106 |
| 1 | 231 | unmanagedpasture | 22015.2 | 21664.1 | -351.106 |
| 1 | 224 | pasture | 1907.36 | 1909.7 | 2.33988 |
| 1 | 225 | unmanagedpasture | 73730.8 | 73730.8 | 0.0111415 |
| 1 | 225 | tundra | 21413.5 | 21413.4 | -0.0111415 |
| 1 | 231 | shrubland | 73909.9 | 73909.9 | -4.36557e-11 |
| 1 | 225 | otherarableland | 50749.2 | 50749.2 | -4.36557e-11 |
| 1 | 225 | unmanagedforest | 44081.4 | 44081.4 | 1.45519e-11 |
| 1 | 224 | unmanagedforest | 37175.1 | 37175.1 | -1.45519e-11 |
| 1 | 231 | unmanagedforest | 13685.1 | 13685.1 | 1.09139e-11 |
| 1 | 231 | otherarableland | 9920.34 | 9920.34 | -7.27596e-12 |
| 1 | 224 | crop1_rfd | 20190.6 | 20190.6 | -7.27596e-12 |
| 1 | 224 | otherarableland | 37239.2 | 37239.2 | -7.27596e-12 |

### 2015: largest region × basin × land class changes (102 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 224 | unmanagedpasture | 63129.6 | 63129.1 | -0.468054 |
| 1 | 224 | pasture | 2555.34 | 2555.81 | 0.467883 |
| 1 | 231 | crop25_irr | 451.598 | 451.595 | -0.00359354 |
| 1 | 231 | unmanagedpasture | 25734.1 | 25734.1 | 0.00359224 |
| 1 | 224 | tundra | 1.33058e-05 | 0.000184667 | 0.000171361 |
| 1 | 225 | unmanagedpasture | 97904.5 | 97904.5 | -5.94087e-05 |
| 1 | 225 | tundra | 0.000115065 | 0.000173602 | 5.85366e-05 |
| 1 | 231 | tundra | 2.86393e-05 | 2.99354e-05 | 1.29604e-06 |
| 1 | 225 | crop11_rfd | 11.0094 | 11.0094 | 4.8724e-07 |
| 1 | 225 | crop10_rfd | 11.0094 | 11.0094 | 3.84959e-07 |
| 1 | 225 | grassland | 84298.7 | 84298.7 | 4.36557e-11 |
| 1 | 225 | unmanagedforest | 43623.1 | 43623.1 | -2.91038e-11 |
| 1 | 231 | grassland | 23360.1 | 23360.1 | 2.18279e-11 |
| 1 | 231 | shrubland | 73741 | 73741 | 1.45519e-11 |
| 1 | 224 | otherarableland | 34190.1 | 34190.1 | -1.45519e-11 |

## Performance (median across repeats)

Regression threshold: +10% (timings under 1 s are not flagged)

| metric | base | new | change |
|---|---|---|---|
| config_s | 0.000 s | 0.000 s | +1.6% |
| initialize_s | 0.019 s | 0.019 s | -1.4% |
| total_s | 0.217 s | 0.217 s | -0.2% |
| peak_rss_bytes | 294.224 MB | 294.306 MB | +0.0% |
| step_2010_s | 0.122 s | 0.118 s | -3.2% |
| step_2015_s | 0.064 s | 0.064 s | +1.4% |
| model: Allocation files processed | 0.005 s | 0.004 s | -4.0% |
| model: Projected landuse data prepared | 0.010 s | 0.010 s | -1.7% |
| model: Base spatial landuse data prepared | 0.002 s | 0.002 s | -0.7% |
| model: Harmonization | 0.000 s | 0.000 s | +0.0% |
| model: Constraints applied to projected and spatial data | 0.001 s | 0.001 s | +4.0% |
| model: Kernel density filter prepared | 0.000 s | 0.001 s | +2.4% |
