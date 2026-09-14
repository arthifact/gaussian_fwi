# Numerical verification

Tests use independent formulas, finite differences, ownership/state invariants
and completing inversion/restart checks. All inputs are generated in memory or
come from the small configuration fixture. No geological data, old results or
experiment package is required.

    python -m unittest discover -s tests/unit -v
    python tools/validate.py --release

The release command runs repository checks, lint and source tests, then builds
a clean wheel and runs every retained test outside the checkout. It rejects
skipped, failed or incomplete installed test discovery. External FWI projects
and the removed experiment package are blocked from import.

Coverage includes full covariance, signed kernels, physical queries, sparse/
fused gradients, wave-equation parameter gradients, topology rollback and Adam
state, source footprints, observation identities, held-out isolation, portable
I/O, command-line completion and exact CPU stage restart.

Optional external historical references can be passed through --reference.
Outputs go to a fresh ignored results/validation/ directory. These are software
checks; [research fits](../docs/BASELINE_PROTOCOL.md) have separate substantial
budgets, physical acceptance and independent evaluation.
