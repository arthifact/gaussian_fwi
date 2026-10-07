# Benchmark with stable starts, 2026-10-07 (commit 7ac5a67)

Adaptive Gaussians against regularized and plain pixel FWI on all five models,
three data conditions (SNR 10 and SNR 5 with noise seeds 0-2, noiseless). Starts
are estimated from the data on the fine first grid; the Gaussian and
regularized pixel fits use the illumination-weighted pull (weight 0.01).

- Lowest error above 300 m (where the survey constrains velocity): Gaussians
  10 of 15 model x condition cells, regularized pixels 3, plain pixels 2.
- Lowest overall RMSE: Gaussians 6, regularized pixels 8, plain pixels 1.
- Gaussians keep 51 (noiseless), 38 (SNR 10) and 35 (SNR 5) components on
  average, stopping growth 0.4, 8.3 and 11.6 times per fit.
- Deep zone worse than its start: Gaussians 22 of 35 fits, regularized pixels
  24, plain pixels 33, mostly by tens of m/s.
- Overthrust at SNR 5 has an unstable start (2250-3875 m/s at the bottom),
  which drives its large spread for every method.
