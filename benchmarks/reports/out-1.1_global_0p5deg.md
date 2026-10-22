# Benchmark comparison: `v2.0.2-global` → `out-1.1-global_0p5deg`

**Accuracy verdict: IDENTICAL**  (atol=1e-06 km², rtol=1e-09, noise floor=0 km²)

| | base | new |
|---|---|---|
| case | global_0p5deg | global_0p5deg |
| code | v2.0.2 | v2.0.1-29-g972c75d-dirty |
| python | 3.11.7 | 3.11.7 |
| numpy / pandas / scipy | 1.26.4 / 2.2.1 / 1.12.0 | 1.26.4 / 2.2.1 / 1.12.0 |
| repeats | 3 | 3 |
| deterministic across repeats | True | True |

## Accuracy by time step

| year | verdict | max abs Δ (km²) | RMSE (km²) | cells beyond tol | non-achieved % (base → new) |
|---|---|---|---|---|---|
| 2010 | IDENTICAL | 0 | 0 | 0 | 31.9859 → 31.9859 |
| 2015 | IDENTICAL | 0 | 0 | 0 | 88.6525 → 88.6525 |

## Performance (median across repeats)

Regression threshold: +10% (timings under 1 s are not flagged)

| metric | base | new | change |
|---|---|---|---|
| config_s | 0.000 s | 0.000 s | +18.0% |
| initialize_s | 0.594 s | 0.307 s | -48.4% |
| total_s | 34.026 s | 16.898 s | -50.3% |
| peak_rss_bytes | 5147.836 MB | 1898.185 MB | -63.1% |
| step_2010_s | 17.557 s | 9.568 s | -45.5% |
| step_2015_s | 14.772 s | 5.800 s | -60.7% |
| model: Allocation files processed | 0.007 s | 0.005 s | -23.1% |
| model: Projected landuse data prepared | 0.063 s | 0.079 s | +24.2% |
| model: Base spatial landuse data prepared | 0.119 s | 0.121 s | +1.5% |
| model: Harmonization | 0.093 s | 0.061 s | -34.3% |
| model: Constraints applied to projected and spatial data | 0.032 s | 0.031 s | -1.0% |
| model: Kernel density filter prepared | 0.273 s | 0.006 s | -97.6% |
