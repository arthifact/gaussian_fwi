# Engineering specification: one Gaussian FWI method

## Bounded stage and compatibility

Version 0.3 consolidates the previously separate direct/scheduled interfaces into
one `gaussian_fwi` package, one inversion engine, one density controller and one
`configs/baseline.json`. Acoustic support is internal to `gaussian_fwi.core`.
Removed controllers, wrapper packages, profiles and historical experiment
outputs are absent from this source tree. Their old commits and external
research archive preserve provenance.

This is a declared numerical-policy and API change. The new training format is
`gaussian-fwi-adc-training-v1`. Reject older training formats before writing or
propagating. Historical saved fields remain readable; exact field/waveform
replay is distinct from reproducing an old optimization trajectory. The field,
decoder and acoustic arithmetic are retained, with namespace imports adjusted.

The [algorithm specification](ALGORITHM.md) defines the literature mapping,
full-training physical gradient statistic, clone/split/prune operations, sampled
field guard, per-band refinement window and settling-only validation selection.
No alternative controller or silent policy switch is permitted.

## Independent acceptance

- Analytical kernel, dense/sparse/fused gradients, physical finite differences
  and acoustic directional derivatives for every parameter family.
- Mean-of-norms selection, deterministic tie order, clone/split/prune decisions,
  independent child-distribution checks and caller RNG isolation.
- Full-SPD constraints, physical sampling floors, cumulative event bounds,
  survivor Adam state, fresh child state and atomic rollback after failure.
- 2D/3D completion, float32/float64 consistency, zero-gradient behavior,
  training/validation/test separation and fixed-population settling.
- Exact completed-stage CPU restart, observation-content identity, obsolete
  checkpoint rejection, saved-field reload and fresh waveform replay.
- Actual forward/adjoint counters, clean isolated-wheel tests, blocked obsolete
  imports, single-profile structure, documentation links and Linux CI.

Run focused tests while developing and the complete gate before release:

    python tools/validate.py --release

Each release writes a fresh ignored directory containing logs, exact source and
verification-input hashes, environment, wheel identity and discovered test IDs.
The installed suite must have no failures, skips or expected failures.
Historical field references are optional and kept outside the source repository.

## Current acceptance and limitations

Local release acceptance passed 61 source tests and the identical 61 tests from
a clean isolated wheel, with no skips. Historical 2D/3D field and waveform replay
was exact. An independent AST audit preserved the arithmetic in 14 field/core
files, excluding namespace imports. A separate 4,000-update CPU diagnostic
completed and passed saved-field, work-count, population and NumPy field-oracle
checks. It does not establish convergence. Generated logs, numerical results and
figures stay outside version control. GitHub Actions runs the same source/wheel
gate on Linux for every main-branch update.
Evidence establishes the tested software properties, not a universal optimum
for the heuristic density settings or a scientific performance result.

The observation-only runner executes CPU fits and supports completed-stage
restart. CUDA shot accumulation, GPU gradient/state checks and mid-stage restart
remain the next bounded engineering stage. The Windows 11 / RTX 4060 campaign
must pass those gates before long runs. See [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md).

The [research protocol](BASELINE_PROTOCOL.md) allocates substantial optimization
budgets and independent evaluation. No convergence, SOTA, Nature-readiness or
large-scale 3D claim follows from the release tests or a small synthetic fit.
