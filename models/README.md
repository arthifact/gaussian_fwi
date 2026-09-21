# Prepared velocity models

`marmousi.npy` — float32 `(z, x)`, 70 × 70 samples, 1500–4500 m/s, intended for
10 m spacing (690 m × 690 m).

Prepared from the public Marmousi model by bilinear resampling to 70 × 70
(`scipy.ndimage.zoom`, `order=1`) followed by an independent min–max map to
1500–4500 m/s. That normalization rescales velocity contrasts, so this array is
a development model for exercising the method, not the published Marmousi
velocities. It is a previously inspected synthetic model, not an unseen survey.

## Use

    python models/make_observations.py models/marmousi.npy data/marmousi.pt
    python models/run_marmousi.py --steps-per-stage 60

`make_observations.py` forward models a fixed-spread surface survey with the
same solver and grid the inversion uses, and writes an observation-only bundle
containing traces, acquisition and disjoint train/validation/test receiver
partitions. A fit of that bundle is therefore a software demonstration under
favourable, self-consistent conditions — it does not establish accuracy on
field data, a different solver, or an unseen model.

`run_marmousi.py` fits the bundle and scores the result against this array. The
reference is used only to create the data and to evaluate afterwards; it never
enters the inversion, which sees waveforms alone.

Omitting `--steps-per-stage` runs the accepted profile: 4 stages × 1,000
updates. Expect roughly 75 minutes on ten CPU threads for this model and survey.
