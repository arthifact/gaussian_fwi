# Gaussian FWI

Full-waveform inversion on a continuous, adaptive Gaussian velocity field.

The model is a depth background plus signed, anisotropic Gaussian components.
Training waveforms optimize each component's amplitude, physical center and full
positive-definite covariance. One density controller clones, splits or prunes
components inside a refinement window, then optimization continues with a fixed
population. Each frequency stage repeats that cycle.

## Install

Python 3.12.

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install --no-build-isolation -e .

## Run

An [observation bundle](#observation-bundle) holds measured traces, acquisition
geometry and disjoint train/validation/test receiver partitions. Target
velocities are never part of the input.

```python
import gaussian_fwi as gfwi

fit = gfwi.run("data/observations.pt", output="results/fit01")

print(fit.summary())
velocity = fit.velocity          # (z, x) in m/s, on the acquisition grid
```

`run` executes the accepted profile, records the configuration, source hashes,
dependency versions and observation identity, then independently repropagates
the saved field before returning. `fit.verified` is true when that replay
reproduced the saved prediction exactly.

Declare any deviation from the accepted profile explicitly; it is validated
immediately and stored with the run:

```python
profile = gfwi.baseline(steps_per_stage=400, cutoffs=[4.0, 7.0])
fit = gfwi.run("data/observations.pt", output="results/fit02", profile=profile)
```

Reopen a finished run, or continue a paused one:

```python
fit = gfwi.Run.open("results/fit01")
fit.report, fit.history, fit.refinement, fit.provenance   # recorded JSON
fit.plot("fit01.png")                                     # needs the plot extra
```

The equivalent command line is:

    gaussian-fwi --observations data/observations.pt \
                 --config configs/baseline.json \
                 --output results/fit01

Add `--device cuda --dtype float64` for the accepted GPU path, which enables
deterministic operations and disables AMP/TF32. CUDA requests never fall back to
CPU. Use `--checkpoint-interval 100 --max-seconds 780` to pause a long fit at an
update boundary, then `--resume path/to/update_00001200.pt` into a new directory.

Output directories must be new; existing results are never replaced.

### Accepted profile

| Setting | Value |
|---|---|
| Initial 2D population | 32 × 32 zero-amplitude seeds |
| Adaptive ceiling | 8,192 Gaussians; growth is not forced |
| Cumulative frequency cutoffs | 4 / 7 / 12 / 20 Hz |
| Updates | 1,000 per stage; 4,000 total |
| Refinement opportunities | Updates 100, 150, …, 450 of each stage |
| Fixed-population settling | Updates 500–1,000 of each stage |
| Validation selection | Every 25 updates, within settling only |

These are explicit development settings, not universally tuned constants. Check
acquisition bandwidth, physical sampling and initialization for each case.

### Observation bundle

A `.pt` file holding exactly `acquisition`, `traces` and `partitions`. Arrays and
acquisition indices are `(z, x)` or `(z, y, x)`; continuous query points are
`(x, z)` or `(x, y, z)`. Distances are meters, velocities m/s, times seconds.
Source amplitudes have shape `(shots, sources, time)` and traces
`(shots, receivers, time)`. Training and validation partitions must be nonempty
and disjoint; a test partition is optional and is scored only after selection.
Build one with `gaussian_fwi.save_observations`.

## Test

    python -m unittest discover -s tests/unit

The checks cover analytic derivatives, acoustic gradients, density-control
transactions, optimizer-state transfer, rollback, restart identity, data
separation and saved-field replay. They are software correctness checks, not
evidence of accuracy on new data.

## Scope

This inverts point sources and receivers. A component's geometry is an
inspectable optimization variable, not automatically a geological body. Waveform
loss is not velocity error. The density controller's gradient score is a
proposal heuristic and its guard is sampled, not an all-coordinate guarantee.

[Citation metadata](CITATION.cff) identifies the software. No open-source license
has been granted for this repository; dependencies retain their own licenses.
