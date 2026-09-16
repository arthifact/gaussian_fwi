# Gaussian FWI

**Full-waveform inversion with a continuous, adaptive Gaussian velocity field.**

The research focus is an explicit, differentiable representation for macro
models, localized detail and future geological constraints. The first application
combines broad structure and local refinement in one physical field. See the
[research direction](docs/RESEARCH_DIRECTION.md) for capability questions,
demonstration stages and planned extensions.

The [baseline closeout](docs/BASELINE_CLOSEOUT.md) fixes the foundation's completion
criteria. The [working manuscript](docs/MANUSCRIPT.md) develops the representation
and macro/detail application, with separate supervised and waveform evidence.
The foundation stage is complete: 18 supervised fits, five archived acoustic
fits, and 76 passing checks in both the source and isolated-wheel release.
The [acceptance matrix](docs/BASELINE_ACCEPTANCE.md) records the reusable
contracts and measured limitations. Future applications build on this baseline.

One method, one Python package, one baseline configuration. Training waveforms
optimize signed amplitudes, physical centers, full positive-definite covariances
and a depth background. A single density controller clones, splits or prunes
Gaussians during an initial refinement window, followed by optimization with a
fixed population. Each frequency stage follows this same cycle.

The density-control mechanism follows
[Kerbl et al., ACM TOG / SIGGRAPH 2023](https://doi.org/10.1145/3592433).
The [algorithm specification](docs/ALGORITHM.md) makes every FWI adaptation
explicit. The [evaluation protocol](docs/BASELINE_PROTOCOL.md) defines evidence
for the representation and its applications. Software correctness, learned
recovery and comparative performance are assessed separately.

## Install

Python 3.12 is the reference interpreter. From a source checkout:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -e .
    python tools/check_environment.py

See [installation](docs/INSTALLATION.md) for Windows 11 / WSL 2 and the RTX 4060.
The CLI supports CPU and explicit CUDA float64 execution. Short CUDA runs have
gradient, state, rollback and completed-stage/update restart checks. Full-gradient
shot accumulation and explicit uncompressed storage have separate numerical and
resource acceptance. The [GPU campaign](docs/GPU_CAMPAIGN.md) records tested
workloads and the sustained research checks still required.

## Run

Prepare an [observation bundle](docs/DATA_FORMAT.md) containing measured traces,
acquisition geometry and disjoint training/validation/test receiver partitions.
Target velocities are excluded from the inversion input.

    python run.py \
      --observations data/observations.pt \
      --config configs/baseline.json \
      --output results/my_fit

The installed command is `gaussian-fwi` with the same arguments. Python users
can use the [single inversion API](docs/API.md).

For the accepted CUDA path, add `--device cuda --dtype float64`. CUDA requests
fail if the GPU is unavailable; they never fall back to CPU. The runner enables
deterministic operations, disables AMP/TF32 and caps its CUDA allocator at 70%
of device memory. CPU remains the default for backward compatibility. Sparse
neighbor search and file handling still use the CPU.
Deterministic settings do not guarantee bitwise repetition of long GPU fits;
the [engineering evidence](docs/ENGINEERING_SPEC.md) records failed strict
long-prefix comparisons alongside passing local gradient and restart checks.

Use `--checkpoint-interval 100 --max-seconds 780 --diagnostics` to save completed
updates and pause a long fit at a consistent boundary. Continue with `--resume
path/to/update_00001200.pt` instead of `--config`, using a fresh output directory.
The numerical horizon stays fixed across segments. A paused result is not a
completed or validation-selected fit. `--shot-batch-size 3` groups compatible
finite-footprint shots; measure memory and accuracy for the intended acquisition.
Add `--accumulate-shots` with a smaller `--shot-batch-size` to free each batch's
wavefields before the next, while preserving one update over all training shots.
Completed-update restart inherits this setting. See the [API](docs/API.md) for
normalization, work accounting and the finite-footprint acquisition requirement.
If one batch still exceeds GPU memory, explicit `--wavefield-storage cpu` keeps
uncompressed intermediate wavefields in host RAM. Propagation stays on the GPU;
measure host-memory headroom and transfer cost before a larger fit.
`--wavefield-storage disk` uses uncompressed temporary files inside the new
output instead. Invocation-owned temporary wavefields are cleaned on completion,
pause and checked failure; checkpoints and unrelated files are preserved.
It requires disk space and adds I/O time; memory and recovery
acceptance for the tested workloads is recorded in the engineering specification.

| Baseline setting | Value |
|---|---|
| Initial 2D population | 32 × 32 zero-amplitude seeds |
| Adaptive ceiling | 8,192 Gaussians; growth is not forced |
| Cumulative frequency cutoffs | 4 / 7 / 12 / 20 Hz |
| Development allowance | 1,000 updates per stage; 4,000 total |
| Refinement opportunities | Updates 100, 150, …, 450 of each stage |
| Fixed-population settling | Updates 500–1,000 of each stage |
| Validation selection | Every 25 updates, within settling only |

Check acquisition bandwidth, physical sampling and initialization for each
case. These are explicit development settings, not universally tuned constants.
The protocol allocates longer 8,000- and 16,000-update runs when convergence
requires them. Changing the stage length also changes the default refinement
window; see the [controlled extension procedure](docs/BASELINE_PROTOCOL.md).

Each run records configuration, source/data identities, loss history, density
edits, actual forward/adjoint calls, selected field and completed-stage
checkpoints. A fresh propagation verifies the reloaded field and predictions.
Output directories must be new. Results, datasets and old experiments are
excluded from the repository.

## Verify

    python tools/validate.py --release

This checks repository structure and links, lints source, runs independent
numerical/integration tests, builds a clean wheel and repeats the tests outside
the checkout. It requires no geological datasets or historical outputs.
[Verification details](tests/README.md) explain what each group establishes.

## Repository

| Path | Purpose |
|---|---|
| gaussian_fwi/ | Field, decoder, one inversion engine and one density controller |
| gaussian_fwi/core/ | Internal acoustic, measurement, geometry and I/O support |
| configs/baseline.json | The single production profile |
| docs/ | Algorithm, mathematical properties, API and evaluation protocol |
| tests/ | Independent numerical and integration checks |
| tools/ | Environment and release verification |

Version 0.3 removes the previous direct/scheduled APIs and profiles. Old field
exports remain readable; old training checkpoints require their original
source. See [compatibility](docs/ENGINEERING_SPEC.md).

[Citation metadata](CITATION.cff) identifies the software; a manuscript citation
will be added when available. No open-source license has been granted for this
repository. Dependencies retain their respective licenses.
