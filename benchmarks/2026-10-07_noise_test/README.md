# Noise-scaled pull, 2026-10-07

Pull toward the start with weight 0.5 x the noise fraction measured from each
dataset (`gaussian_fwi.noise_fraction`), against the fixed weight 0.01 used in
`2026-10-07_baseline_v4`. SNR 5 (seeds 0 and 1) and noiseless, all five models,
both representations; starts from `2026-10-07_baseline_v3/starts`.

Gaussians at SNR 5: overall RMSE 693 -> 680, deep zone vs start +74 -> +39 m/s,
fits more than 50 m/s worse at depth 7 -> 3, above 300 m 490 -> 516.
Gaussians noiseless (now no pull): overall 506 -> 493, above 300 m 384 -> 356.
