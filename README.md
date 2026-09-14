# Gaussian FWI

**Adaptive anisotropic Gaussian velocity fields for full-waveform inversion.**

Gaussian FWI represents a continuous velocity field with signed Gaussian
amplitudes, movable centers, full positive-definite covariance and a trainable
depth background. Acoustic waveforms drive optimization. A direct refinement
controller adjusts the population while preserving surviving optimizer state.

This repository contains the research baseline, numerical tests and protocol for
the next evaluation campaign. It includes no previous experiment outputs,
geological datasets, fitted checkpoints or learned-initializer studies.

## Method

- **Explicit representation:** physical coordinates and full covariance in 2D/3D.
- **Adaptive capacity:** insertion, splitting, cloning, merging and pruning.
- **Controlled edits:** zero-amplitude insertion, cumulative sampled field-change
  limits, fresh newborn Adam state and atomic rollback.
- **Sparse evaluation:** streamed Gaussian-query interactions with a fused decoder.
- **Waveform-only fitting:** training drives optimization; validation selects;
  test waveforms and reference velocities are evaluated afterward.
- **Reproducible state:** field export/reload, observation-content identities and
  completed-stage restart.

See the [method derivation](docs/METHODS.md) and [API guide](dynamic_refinement/README.md).

## Installation

Python 3.12 is the reference interpreter. From a source checkout:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -e .
    python tools/check_environment.py

Windows users can follow the [installation guide](docs/INSTALLATION.md), including
the proposed WSL 2 setup for the RTX 4060. Install a CUDA-enabled PyTorch build
before GPU environment checks. The current command-line baseline runs on CPU;
shot batching, GPU numerical acceptance and mid-stage restart are planned in the
[GPU campaign](docs/GPU_CAMPAIGN.md).

## Run the baseline

Prepare an observation bundle with acquisition geometry, measured traces and
disjoint receiver partitions. The [data format](docs/DATA_FORMAT.md) includes
the exact schema and a save/load example. No target velocity enters this runner.

    python run.py \
      --observations data/observations.pt \
      --config configs/direct.json \
      --output results/my_fit

The installed equivalent is gaussian-fwi with the same arguments. Use
configs/scheduled.json for prescribed population growth.

| Profile | Population | Frequency bands | Development allowance |
|---|---|---|---|
| Direct | Initial 1,024; adaptive ceiling 8,192 | Cumulative 4 / 7 / 12 / 20 Hz | 1,000 updates per stage; 4,000 total |
| Scheduled | Cumulative 1,024 / 2,624 / 4,928 / 8,064 | Cumulative 4 / 7 / 12 / 20 Hz | 1,000 updates per stage; 4,000 total |

Both profiles select validation checkpoints every 25 updates. Their physical
scales and acquisition requirements must be checked for each dataset. The
allowance is a development starting point, not a convergence guarantee.
Longer planned runs use 2,000 or 4,000 updates per stage:

    python run.py \
      --observations data/observations.pt \
      --config configs/direct.json \
      --steps-per-stage 2000 \
      --output results/my_8000_update_fit

Each run writes its configuration, source/input identities, solver counts,
training history, selected field, predictions and stage checkpoints. A separate
propagation verifies the saved CPU field. Output directories must be fresh.
Generated files are excluded from version control.

## Research protocol

The [baseline protocol](docs/BASELINE_PROTOCOL.md) specifies the central claim,
accuracy/storage comparisons, adaptation controls and independent waveform
evaluation. Initial representation allowances are 5,000 updates, with development
extensions to 10,000 and 20,000. Four-stage FWI allowances are 4,000, 8,000 and
16,000 total updates. Count actual forward/adjoint work and report incomplete
convergence explicitly.

Numerical tests establish software properties. Competitive reconstruction,
compactness, GPU performance and application-scale 3D require their own evidence.

## Verify

    python tools/validate.py --release

The release gate checks repository structure and documentation, runs lint and
source tests, builds a clean wheel, and reruns the tests outside the checkout.
It requires no datasets or previous results. Optional external reference
checkpoints can be supplied for historical regression checks. See
[tests/README.md](tests/README.md).

## Repository layout

| Path | Purpose |
|---|---|
| dynamic_refinement/ | Main adaptive inversion API |
| gaussian_fwi/ | Field, sparse decoder, topology and optimizer machinery |
| fwi_core/ | Physical grids, acoustic/measurement operators, objectives and I/O |
| configs/ | Explicit direct and scheduled development profiles |
| docs/ | Method, data format, installation and research protocol |
| tests/ | Independent numerical and integration checks |
| tools/ | Environment, repository and release verification |

Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md). Software citation
metadata is in [CITATION.cff](CITATION.cff); a paper citation will be added when
available. Dependencies retain their respective licenses.
