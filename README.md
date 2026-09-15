# Gaussian FWI

**Full-waveform inversion with a continuous, adaptive Gaussian velocity field.**

One method, one Python package, one baseline configuration. Training waveforms
optimize signed amplitudes, physical centers, full positive-definite covariances
and a depth background. A single density controller clones, splits or prunes
Gaussians during an initial refinement window, followed by optimization with a
fixed population. Each frequency stage follows this same cycle.

The density-control mechanism follows
[Kerbl et al., ACM TOG / SIGGRAPH 2023](https://doi.org/10.1145/3592433).
The [algorithm specification](docs/ALGORITHM.md) makes every FWI adaptation
explicit. This is a research implementation: correctness checks and literature
precedent do not establish seismic superiority, convergence or publication
readiness. The [evaluation protocol](docs/BASELINE_PROTOCOL.md) defines that work.

## Install

Python 3.12 is the reference interpreter. From a source checkout:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -e .
    python tools/check_environment.py

See [installation](docs/INSTALLATION.md) for Windows 11 / WSL 2 and the RTX 4060.
The current CLI runs on CPU. CUDA shot batching and reliable mid-stage restart
require acceptance before the [GPU campaign](docs/GPU_CAMPAIGN.md).

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
Output directories must be new. Results, generated observations and old
experiments are excluded from the repository. Five preserved 70 × 70
[development models](models/README.md) are included with their original
provenance manifest; they are resampled and velocity-normalized inputs.

## Verify

    python tools/validate.py --release

This checks repository structure and links, lints source, runs independent
numerical/integration tests, builds a clean wheel and repeats the tests outside
the checkout. The numerical tests require no external geological datasets or historical outputs.
[Verification details](tests/README.md) explain what each group establishes.

## Repository

| Path | Purpose |
|---|---|
| gaussian_fwi/ | Field, decoder, one inversion engine and one density controller |
| gaussian_fwi/core/ | Internal acoustic, measurement, geometry and I/O support |
| configs/baseline.json | The single production profile |
| models/ | Five preserved development arrays, provenance and loading notes |
| docs/ | Algorithm, mathematical properties, API and evaluation protocol |
| tests/ | Independent numerical and integration checks |
| tools/ | Environment and release verification |

Version 0.3 removes the previous direct/scheduled APIs and profiles. Old field
exports remain readable; old training checkpoints require their original
source. See [compatibility](docs/ENGINEERING_SPEC.md).

[Citation metadata](CITATION.cff) identifies the software; a manuscript citation
will be added when available. No open-source license has been granted for this
repository. Dependencies retain their respective licenses.
