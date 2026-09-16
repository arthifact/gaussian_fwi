# Numerical verification

Tests use independent formulas, finite differences, statistical distributions
and state invariants. Small synthetic acquisitions support completing fit and
restart checks; they are not research benchmarks. No private datasets, previous
results or removed inversion package is required.

    python -m unittest discover -s tests/unit -v
    python tools/validate.py --release

CUDA checks are explicit, hardware-dependent additions to the release gate:

    python tools/check_cuda.py --output results/cuda/derivatives
    python tools/check_cuda_runtime.py --observations data/observations.pt --output results/cuda/runtime

Use fresh output directories. The runtime check reuses independent refinement
invariants on CUDA float64 and checks completed-stage/update restart, waveform replay,
actual shot counts and test-data isolation. It requires the source fixtures and
an observation bundle. Use `--shot-batch-size 3` to exercise explicit grouping.
Use `--accumulate-shots --shot-batch-size 2` to check an unequal final batch on
a three-shot bundle, and `--wavefield-storage cpu` or `disk` for explicit
uncompressed storage. These modes retain the same physical solve accounting.
Its short runs do not establish research convergence or intended-scale capacity.

The release gate checks repository structure and links, lints source, runs the
source suite, builds a clean wheel and repeats every test outside the checkout
in isolated Python. Removed packages and external FWI projects are blocked from
import. Empty discovery, failures, skips and expected failures reject the release.

| Tests | Independent evidence |
|---|---|
| numerics, decoder, fused | Scalar kernel formula, dense NumPy/autograd values and derivatives, acoustic finite differences |
| refinement, topology | Physical center finite differences; 4,096 sampled children per dimension; edit selection, cumulative bounds, atomic state rollback |
| inversion, restart_identity | 2D/3D completion, settling selection, float32/64, exact CPU restart, format/content rejection and counted work |
| update_restart | Interruptions before/after edits and within settling; exact field, Adam and selected-state recovery; no duplicate solves; corrupt-state rejection |
| shot_accumulation | Unequal global loss weights, every parameter gradient and density score, finite differences, exact restart/isolation, storage-policy rejection and released batch tensor lifetimes |
| isolation | Test-waveform perturbations cannot change the selected field or edits |
| physical_sampling, spatial_continuation | Analytic sampling, independent covariance eigenvalues and counterexamples to excessive resolution claims |
| footprints, observation_identity | Independent physical source/receiver operators and content hashes |
| runner | Portable observation schema, fresh output, full CLI completion and exact saved-field waveform replay |
| wavefield_lifecycle | Retained real acoustic graphs with injected adjoint failure; owned scratch cleanup, surviving checkpoints, directory replacement and unrelated-file protection |

Optional `--reference` fixtures must contain `2/` and `3/`, each with a compatible
`field.pt` and `predictions.pt` generated using the documented `_problems.py`
acquisition. This gate loads historical fields and reproduces their saved
velocity and waveforms exactly. It does **not** resume an old controller or
assert equivalence between different training algorithms.

Verification outputs use a fresh ignored `results/validation/` directory. The
manifest records wheel/source identities, environment and every test ID. Longer
behavioral diagnostics and [research fits](../docs/BASELINE_PROTOCOL.md) have
separate physical, convergence and comparison requirements.

## Optional CUDA diagnostic

    python tools/check_cuda.py --output results/cuda/first_check

This separate tool requires CUDA and fails rather than skipping unavailable
hardware. Tiny 2D/3D fixtures compare CPU/CUDA float32/float64 field values,
waveforms, all parameter-family training gradients and one Adam update. Float64
CUDA acoustic directional derivatives are checked against two central-difference
step sizes. Tolerances are fixed in the tool and the engineering specification.
The evidence includes source identities, versions, actual acoustic work and
synchronized update time/memory. First-use timing includes initialization and is
not a throughput benchmark. The tool does not certify a GPU inversion trajectory,
shot accumulation, topology state/restart or campaign-scale memory.
