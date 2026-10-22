# Benchmark comparison: `b2-texas-res0.5` → `out-1.4-texas_0p5deg`

**Accuracy verdict: DRIFT**  (atol=1e-06 km², rtol=1e-09, noise floor=0 km²)

| | base | new |
|---|---|---|
| case | texas_0p5deg | texas_0p5deg |
| code | v2.0.1-31-g0369b84-dirty | v2.0.1-31-g0369b84-dirty |
| python | 3.11.7 | 3.11.7 |
| numpy / pandas / scipy | 1.26.4 / 2.2.1 / 1.12.0 | 1.26.4 / 2.2.1 / 1.12.0 |
| repeats | 3 | 3 |
| deterministic across repeats | True | True |

## Accuracy by time step

| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |
|---|---|---|---|---|---|
| 2010 | DRIFT | 376.865 | 8.48529 | 396 | 38.1862 → 38.1316 |
| 2015 | DRIFT | 379.315 | 8.42486 | 406 | 81.57 → 81.6795 |

### 2010: largest region × basin × land class changes (87 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 224 | unmanagedpasture | 59348.4 | 59623.7 | 275.306 |
| 1 | 224 | tundra | 1878.26 | 1603.08 | -275.186 |
| 1 | 231 | tundra | 2586.42 | 2579.54 | -6.88174 |
| 1 | 231 | unmanagedpasture | 22008.4 | 22015.2 | 6.88174 |
| 1 | 225 | unmanagedpasture | 73730.9 | 73730.8 | -0.158163 |
| 1 | 225 | tundra | 21413.3 | 21413.5 | 0.158163 |
| 1 | 224 | pasture | 1907.48 | 1907.36 | -0.119129 |
| 1 | 225 | otherarableland | 50749.2 | 50749.2 | 3.63798e-11 |
| 1 | 224 | otherarableland | 37239.2 | 37239.2 | 2.91038e-11 |
| 1 | 231 | shrubland | 73909.9 | 73909.9 | 1.45519e-11 |
| 1 | 225 | unmanagedforest | 44081.4 | 44081.4 | -1.45519e-11 |
| 1 | 224 | unmanagedforest | 37175.1 | 37175.1 | 1.45519e-11 |
| 1 | 224 | crop1_rfd | 20190.6 | 20190.6 | -7.27596e-12 |
| 1 | 224 | forest | 6759.6 | 6759.6 | -4.54747e-12 |
| 1 | 231 | grassland | 24529.8 | 24529.8 | -3.63798e-12 |

### 2015: largest region × basin × land class changes (94 entries changed)

| region | basin | land class | base km² | new km² | Δ km² |
|---|---|---|---|---|---|
| 1 | 224 | pasture | 2555.37 | 2555.34 | -0.0237686 |
| 1 | 224 | unmanagedpasture | 63129.6 | 63129.6 | 0.0236774 |
| 1 | 231 | unmanagedpasture | 25734.1 | 25734.1 | 0.00014395 |
| 1 | 231 | tundra | 0.000172589 | 2.86393e-05 | -0.00014395 |
| 1 | 225 | tundra | 0 | 0.000115065 | 0.000115065 |
| 1 | 225 | unmanagedpasture | 97904.5 | 97904.5 | -0.000115056 |
| 1 | 224 | crop9_irr | 309.296 | 309.296 | 7.78387e-05 |
| 1 | 224 | tundra | 0 | 1.33058e-05 | 1.33058e-05 |
| 1 | 225 | crop10_rfd | 11.0094 | 11.0094 | -4.51652e-09 |
| 1 | 225 | crop11_rfd | 11.0094 | 11.0094 | -4.51652e-09 |
| 1 | 231 | shrubland | 73741 | 73741 | -4.36557e-11 |
| 1 | 225 | grassland | 84298.7 | 84298.7 | -2.91038e-11 |
| 1 | 225 | otherarableland | 46989.2 | 46989.2 | -2.18279e-11 |
| 1 | 225 | unmanagedforest | 43623.1 | 43623.1 | 2.18279e-11 |
| 1 | 231 | unmanagedforest | 13901.9 | 13901.9 | 1.09139e-11 |

## Performance (median across repeats)

Regression threshold: +10% (timings under 1 s are not flagged)

| metric | base | new | change |
|---|---|---|---|
| config_s | 0.000 s | 0.000 s | -5.7% |
| initialize_s | 0.018 s | 0.019 s | +3.3% |
| total_s | 0.213 s | 0.217 s | +1.8% |
| peak_rss_bytes | 294.470 MB | 294.224 MB | -0.1% |
| step_2010_s | 0.116 s | 0.122 s | +5.3% |
| step_2015_s | 0.062 s | 0.064 s | +2.8% |
| model: Allocation files processed | 0.004 s | 0.005 s | +5.2% |
| model: Projected landuse data prepared | 0.010 s | 0.010 s | +0.4% |
| model: Base spatial landuse data prepared | 0.002 s | 0.002 s | +4.0% |
| model: Harmonization | 0.000 s | 0.000 s | +1.1% |
| model: Constraints applied to projected and spatial data | 0.001 s | 0.001 s | -0.9% |
| model: Kernel density filter prepared | 0.000 s | 0.000 s | +53.9% |
