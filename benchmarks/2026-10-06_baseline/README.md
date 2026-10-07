# Benchmark, 2026-10-06 (commit 045e0e2)

Adaptive Gaussians against pixel FWI (with and without TV) on all five models,
three data conditions (SNR 10 and SNR 5 with noise seeds 0-2, noiseless), from a
starting model estimated from the data. Settings were tuned on Marmousi only
and frozen (`settings.json`). Reproduce with `python scripts/benchmark.py run`
and `report` at that commit.

- `report.md`: table and summary; `baseline.png`: SNR 10, seed 0 velocities.
- `runs/`: per-fit scores (`.json`) and final velocity (`.npy`, float32, (z, x), m/s).
- `starts/`: the starting trend estimated for each dataset.

Known problems at this commit, found in these results:

1. The deep zone (below 300 m) ends worse than its start in 29 of 35 Gaussian fits.
2. The held-out growth stop never fired: with 60 steps per band there is only
   one population edit per band, and the stop needs two to compare.
