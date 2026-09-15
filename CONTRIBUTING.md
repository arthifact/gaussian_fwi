# Contributing

Use an issue to describe the concrete problem, expected behavior and a minimal
reproduction. Keep numerical changes bounded and state compatibility before
implementation. Open a pull request with the change, independent acceptance
check, commands, evidence and limitations.

## Development

    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -e .
    python tools/validate.py --release

Run the relevant focused checks while developing. The final source/wheel gate
must pass in a fresh output directory. Use independent formulas, finite
differences, independent sampling/state invariants and completing integration checks.
Short numerical fixtures do not replace substantial research fits.

## Numerical contract

- Tensor axes are (z,x) or (z,y,x); physical points are (x,z) or (x,y,z), in meters.
- Velocity and signed amplitudes are m/s; covariance is full SPD in square meters.
- Field and propagation grid shape/spacing must agree.
- Training waveforms drive optimization and topology; validation selects.
  Keep test waveforms and target velocities out of FWI tuning and ranking.
- Preserve surviving Adam state, initialize newborn state, seed at zero
  amplitude and roll back rejected topology edits atomically.
- Count actual forward/adjoint solves, including recomputation and independent checks.
- Never overwrite an existing fit. Retain unsuccessful and negative outcomes.

## Repository scope

Commit source, small independent test fixtures, configurations and documentation.
Keep generated results, additional datasets, environments, credentials and
private research notes out of the repository. The five preserved development
arrays and their manifest in models/ are explicitly tracked; preserve their
identities and document their prepared-data limitations. Document data access and identities in the
appropriate external campaign record.

Follow the [baseline protocol](docs/BASELINE_PROTOCOL.md) for scientific budgets
and [engineering specification](docs/ENGINEERING_SPEC.md) for release acceptance.

Maintain one public inversion method. Specify mechanism ablations in the external
research record rather than adding competing production controllers. Sampled
splits and clones can change the field; do not assert moment conservation.
