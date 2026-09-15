# Numerical verification

Tests use independent formulas, finite differences, statistical distributions
and state invariants. Small synthetic acquisitions support completing fit and
restart checks; they are not research benchmarks. No private datasets, previous
results or removed inversion package is required.

    python -m unittest discover -s tests/unit -v
    python tools/validate.py --release

The release gate checks repository structure and links, lints source, runs the
source suite, builds a clean wheel and repeats every test outside the checkout
in isolated Python. Removed packages and external FWI projects are blocked from
import. Empty discovery, failures, skips and expected failures reject the release.

| Tests | Independent evidence |
|---|---|
| numerics, decoder, fused | Scalar kernel formula, dense NumPy/autograd values and derivatives, acoustic finite differences |
| refinement, topology | Physical center finite differences; 4,096 sampled children per dimension; edit selection, cumulative bounds, atomic state rollback |
| inversion, restart_identity | 2D/3D completion, settling selection, float32/64, exact CPU restart, format/content rejection and counted work |
| isolation | Test-waveform perturbations cannot change the selected field or edits |
| physical_sampling, spatial_continuation | Analytic sampling, independent covariance eigenvalues and counterexamples to excessive resolution claims |
| footprints, observation_identity | Independent physical source/receiver operators and content hashes |
| runner | Portable observation schema, fresh output, full CLI completion and exact saved-field waveform replay |

Optional `--reference` fixtures must contain `2/` and `3/`, each with a compatible
`field.pt` and `predictions.pt` generated using the documented `_problems.py`
acquisition. This gate loads historical fields and reproduces their saved
velocity and waveforms exactly. It does **not** resume an old controller or
assert equivalence between different training algorithms.

Verification outputs use a fresh ignored `results/validation/` directory. The
manifest records wheel/source identities, environment and every test ID. Longer
behavioral diagnostics and [research fits](../docs/BASELINE_PROTOCOL.md) have
separate physical, convergence and comparison requirements.
