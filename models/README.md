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

## Capacity is the variable worth showing

`capacity_study.py` fits the same survey at several capacities. A surface spread
cannot constrain velocity below roughly half its maximum offset, so errors are
reported inside and outside that illuminated depth rather than as one number
that hides the distinction.

Measured on this model at a 240-update horizon, illuminated depth 300 m:

| Capacity | Params | Params/cell | RMSE all | RMSE <300 m | RMSE >300 m | Roughness |
|---|---|---|---|---|---|---|
| initial | — | — | 512.8 | 231.4 | 648.0 | 0.0 |
| 8x8 | 380 | 0.08 | 500.2 | 158.8 | 647.2 | 37.8 |
| 16x16 | 1532 | 0.31 | 522.3 | 167.0 | 675.6 | 53.7 |
| 32x32 | 6068 | 1.24 | 512.2 | 184.8 | 658.4 | 92.5 |

All three reach the same waveform fit (final training bands within a few percent
of 0.43 / 0.24 / 0.16 / 0.18). The extra 5,688 parameters buy no data fit: they
are spent on noise and on structure in the unilluminated zone, where the
63-Gaussian field instead stays at its prior. Note that 32x32 seeding gives
1.24 parameters per grid cell — more freedom than a grid of the same model — so
at that capacity the representation supplies no dimensionality reduction at all.

This is regularization by parameterization. A conventional smoothness prior
does not substitute for it. Raising the TV weight on the 32x32 field, at the
same horizon:

| Setup | Params | RMSE all | RMSE <300 m | RMSE >300 m | Roughness |
|---|---|---|---|---|---|
| 8x8, tv=1e-4 | 380 | 500.2 | 158.8 | 647.2 | 37.8 |
| 32x32, tv=1e-4 | 6068 | 512.2 | 184.8 | 658.4 | 92.5 |
| 32x32, tv=1e-3 | 6044 | 511.8 | 182.5 | 658.3 | 83.2 |
| 32x32, tv=1e-2 | 6038 | 538.2 | 168.6 | 696.9 | 48.2 |

A hundredfold TV weight smooths the field and helps the illuminated zone, but
degrades the unilluminated zone and the model as a whole, and still does not
reach the low-capacity result. TV penalizes gradients uniformly, so it blurs
real structure where the data constrain it while still permitting noise-driven
structure where they do not. Lowering capacity removes the freedom instead.

Two limits on that conclusion. This is not a grid-FWI baseline: the control is
a 32x32 Gaussian field with a stronger penalty, not 4,900 free pixels, so it
does not establish an advantage over grid inversion. And every run here learns
geometry, so the study varies component count, not whether geometry is movable;
the fixed-versus-learned comparison belongs to the manuscript's supervised
study, and this release requires a fully trainable field.

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
