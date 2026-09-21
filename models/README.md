# Prepared velocity models

`marmousi.npy` — float32 `(z, x)`, 70 × 70 samples, 1500–4500 m/s, intended for
10 m spacing (690 m × 690 m).

Prepared from the public Marmousi model by bilinear resampling to 70 × 70
(`scipy.ndimage.zoom`, `order=1`) followed by an independent min–max map to
1500–4500 m/s. That normalization rescales velocity contrasts, so this array is
a development model for exercising the method, not the published Marmousi
velocities. It is a previously inspected synthetic model, not an unseen survey.

## Use

    python models/make_observations.py models/marmousi.npy data/marmousi_real.pt
    python models/run_marmousi.py --steps-per-stage 60
    python models/plot_result.py results/marmousi models/marmousi.npy figure.png

## What the demonstration does and does not establish

Fitting data you generated with the same solver, on the same grid, without
noise, from the exact source wavelet is an inverse crime: the forward operator
is invertible almost by construction and any method will look strong. By
default `make_observations.py` removes those advantages.

| Condition | Default | Why |
|---|---|---|
| Modelling grid | 2× finer than the inversion's, resampled onto it | The inversion's operator is not the one that made the data |
| Noise | Band-limited, survey S/N 10 | Real traces are not clean |
| Source wavelet | 8% peak-frequency error vs the true one | The source is estimated in practice, not known |
| Velocity bounds | 1400–5000 m/s, not the reference's 1500–4500 | The true range is not known in advance |

Together these leave roughly a 16% relative waveform difference from the
self-consistent case. `--ideal` restores the inverse crime for comparison, which
is the honest way to show how much of a result rests on it.

Measured on this model at a deliberately short 240-update horizon, with matched
bounds and profile, only the data conditions differing:

| | Ideal (inverse crime) | Realistic |
|---|---|---|
| Velocity RMSE | 512.8 -> 395.8 m/s | 512.8 -> 512.2 m/s |
| Final training waveform loss | ~1e-4 per band | 0.42 / 0.23 / 0.16 / 0.17 |

Under the crime the misfit collapses to near machine precision because the
operator is exactly invertible; under realistic conditions it floors at a noise
and physics limit. At that horizon the crime accounts for essentially all of the
apparent velocity recovery. Note the confound before drawing conclusions: 240
updates is 6% of the accepted 4,000-update budget and the loss was still
descending, so this compares data conditions, not converged results.

One subtlety worth knowing if you change the modelling grid: Deepwave injects a
source as `amplitude * dt²`, so refining the time step alone scales recorded
amplitudes by `1/refinement²`. The generator compensates, and
`tests/unit/test_models.py` pins that it keeps doing so — without it the
inversion chases a discretization artifact rather than the physics.

What remains favourable, and should be stated alongside any result: the physics
is acoustic, isotropic and noise-stationary; the geometry is a dense fixed
surface spread (8 shots, 68 receivers at 10 m); usable energy extends down to
the 4 Hz first stage; and the model is 2D and small. This exercises the software
on a realistic-but-synthetic problem. It is not a field result and not a
benchmark against other methods — the manuscript carries the evaluated study.

`run_marmousi.py` fits the bundle and scores the result against this array. The
reference is used only to create the data and to evaluate afterwards; it never
enters the inversion, which sees waveforms, geometry and partitions alone.

Omitting `--steps-per-stage` runs the accepted profile: 4 stages × 1,000
updates. Expect roughly 75 minutes on ten CPU threads for this model and survey.
