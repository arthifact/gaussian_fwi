# Benchmark with the illumination-weighted pull, 2026-10-07 (commit 2688e79)

Same design as `2026-10-06_baseline`, with the pull toward the start (weight
0.01, illumination-weighted) on the Gaussian and regularized pixel fits. The
pull weight was chosen on SEAM, Sigsbee2A and BP 2004 against their true
models, so Marmousi and Overthrust are the clean check of it.

Findings:

- Growth stops by itself and more often with noise: 50 Gaussians (noiseless),
  36.5 (SNR 10), 35 (SNR 5); 0, 7.7 and 11.6 growth stops per fit.
- Lowest error above 300 m in 9 of 15 model x condition cells; lowest overall
  RMSE in 5 of 15 (regularized pixels 9).
- Deep zone worse than its start in 21 of 35 Gaussian fits (29 without the pull).
- Known problem: the starting trend's bottom velocity flips between modes with
  the noise seed (see `starts/`), e.g. Marmousi 2700 or 4500 m/s, which drives
  the large spreads for every method.
