# Benchmark with data-driven growth, 2026-10-07 (commit 0a57264)

As `2026-10-07_baseline_v3` (same starts, in `../2026-10-07_baseline_v3/starts`),
with no caps on the Gaussian population and split children protected from
merging for one band.

- Lowest RMSE out of 15 model x condition cells: Gaussians 8, regularized
  pixels 6, plain pixels 1. Above 300 m: Gaussians 11, 3, 1. Below 300 m:
  Gaussians 10, 3, 2.
- Mean population: 187 Gaussians noiseless, 96 at SNR 10, 69 at SNR 5, with
  0.8, 7.9 and 11.5 held-out growth stops per fit.
- Deep zone worse than its start: Gaussians 24 of 35 fits, regularized pixels
  24, plain pixels 33.
