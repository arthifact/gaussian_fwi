# Engineering specification

## Current baseline

The shareable baseline contains adaptive Gaussian inversion, the scheduled
control, acoustic/footprint operators, portable observation I/O and independent
numerical checks. Generated outputs, native datasets, previous campaigns and
learned-initializer experiments are excluded from source control and packaging.

Core field, decoder, propagation, optimization and topology arithmetic are
preserved. Checkpoint format identifiers and their compatibility branches retain
their meanings. Experimental initializer APIs are outside this package.
The observation-only CLI replaces the former dataset-specific launcher; it
accepts arbitrary valid acquisition grids and has no reference-velocity input.

The direct and scheduled profiles explicitly allocate 1,000 updates per band
and validation every 25 updates. This is a prospective development allowance,
not evidence of convergence. GPU shot accumulation and completed-update restart
are separate planned work in [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md).

## Change acceptance

Before implementation, state the bounded change, compatibility and independent
check. Reproduce a numerical defect or check an independent invariant. Preserve
survivor/newborn optimizer semantics, zero-amplitude insertion, sampled cumulative
field-change limits, atomic rollback and actual solve counts.

The clean-repository acceptance requires:

- Core source identity reconciliation and unchanged mathematical behavior.
- Independent field/gradient, physical sampling, optimizer/topology and
  data-isolation checks.
- Completing observation-only fits without native data, exact saved CPU
  propagation, rejected target-bearing bundles and no output overwrite.
- Source tests and isolated-wheel tests with no external project imports.
- Package allowlist, local documentation links and clean source publication.
- Optional external historical replay when reference checkpoints are supplied.

## Commands

    python tools/check_environment.py
    python tools/check_repository.py
    python -m unittest discover -s tests/unit -v
    python tools/validate.py --release

The release tool creates a fresh directory under results/validation/. To use a
particular fresh destination:

    python tools/validate.py --release --output results/validation/my_release

Historical checks are optional and use a separately retained reference with
2/ and 3/ subdirectories:

    python tools/validate.py --release \
      --output results/validation/my_replay \
      --reference /path/to/reference

Do not commit generated verification directories. Preserve full reports and
source/input identities in an external evidence archive or CI artifacts.
These checks establish software properties; substantial research fits follow
[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md).

## Next bounded work

Initial clean-export acceptance completed on Python 3.12.13 / macOS arm64:
72 source tests, 72 isolated-wheel tests and exact external historical 2D/3D
replays passed. The core numerical implementation comprises 26 unchanged source
files. The cleaned suite omits tests for excluded experimental modules and adds
portable bundle/CLI acceptance; no retained numerical tolerance was relaxed.
Verification outputs remain outside the published source tree.

Implement an explicit CUDA campaign path with correctly normalized shot
accumulation and completed-update restart. Predeclare numerical tolerances,
test independent gradients and state invariants, then profile on the actual
RTX 4060. CPU release success does not certify GPU performance or accuracy.
