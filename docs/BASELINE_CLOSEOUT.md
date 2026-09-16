# Baseline closeout and manuscript boundary

## What is being finished

Deliver one stable, documented adaptive Gaussian velocity representation and
one observation-only acoustic inversion method, with a reproducible paper record.
The [research direction](RESEARCH_DIRECTION.md) defines the contribution:
learning velocity together with its spatial parameterization. Macro models with
local detail are the first representation application. Future applications can
use the same field, gradients, geometry and saved states.

This stage does not add stratigraphic learning, alternative-model search,
uncertainty machinery, a macro-preservation constraint or another engine. Those
ideas are later consumers or extensions of the foundation. General convergence
and every possible application are not release criteria.

## Finite completion checklist

| Deliverable | Completion criterion |
|---|---|
| One method and interface | One package, field, inversion loop, density controller, baseline JSON and observation-only runner; documented physical coordinates and units |
| Numerical correctness | Existing independent value/derivative, covariance, bound, sampling, topology/Adam, isolation, restart and replay checks pass for the final source |
| Failure handling | Newly owned disk scratch is cleaned on checked failures; original errors and recoverable checkpoints survive; unrelated files remain untouched |
| Release acceptance | Fresh Ruff, repository/link, source-test and isolated-wheel gates pass; exact source, wheel and verification inputs are archived |
| Representation usefulness | Complete the declared supervised macro/detail study and controls; independently evaluate saved fields, report every run and all allowed budget extensions |
| FWI application evidence | Incorporate existing audited observation-only fits and their actual work, populations and observed limitations, preserving their original source identity |
| Working paper | Abstract, introduction, related work, method, results, figures, discussion and reproducibility statement trace to saved evidence |
| Handoff | Publishable source remains free of datasets/results; one immutable local release record identifies the implementation on which later applications build |

The completion decision concerns these deliverables. An unfavorable experiment
changes the claim supported by the paper. A concrete software defect reopens
the relevant check. A new application or a speculative improvement does not
reopen the entire baseline.

## Stability contract

The software contract covers finite physically bounded fields, full SPD
covariances with declared numerical width limits, checked first derivatives,
zero-amplitude seeding, signed-amplitude edit semantics, survivor/fresh-child
Adam state, cumulative sampled event guards and atomic rejection. It covers
explicit data partitioning, counted work, immutable outputs, saved-field replay
and completion-boundary continuation.

CPU fixtures have exact continuation checks. CUDA float64 has accepted local
gradient/state/restart checks on the documented workloads. Existing failed
strict long-prefix comparisons remain failed and are reported separately;
deterministic runtime settings are not a promise of bitwise equality across
long GPU trajectories. Finite-budget optimizer fluctuations and imperfect
inverse recovery are scientific observations, not automatically code defects.

GPU resource support is limited to measured workloads. The earlier source-change
large recovery test is described as such. The current closeout does not claim
that every larger 3D survey fits an 8 GB GPU. New workloads receive their own
resource checks without redesigning the field representation.

## Minimum paper evidence

The accepted software checks establish the method's implementation. The
constructed seven-component demonstration explains explicit local control and
scale coupling. The new supervised study tests learned representation on two
analytic structures independent of the Gaussian decoder, with three starts and
three mechanism controls. The archived acoustic fits test integration with FWI.
These evidence classes remain clearly separated throughout the manuscript.

Compare fixed geometry with learned geometry at the same initial component
count, and fixed with adaptive populations from paired initializations. Record
stored and optimized parameter counts, attained populations and actual time;
the controls do not automatically match final capacity or effective
regularization. Report macro error, detail error, off-grid reconstruction,
covariance geometry and finite-budget learning trends. Claims about stronger
performance than other representations require additional relevant comparative
evidence; that claim is not required to finish the reusable baseline.

The [baseline protocol](BASELINE_PROTOCOL.md) governs fitting allowances and data
separation. The [engineering specification](ENGINEERING_SPEC.md) records exact
changes, checks, evidence paths and any remaining operating limitations.

The [acceptance matrix](BASELINE_ACCEPTANCE.md) defines which implementation
contracts are accepted and which evidence later applications can reuse. The
[working manuscript](MANUSCRIPT.md) is the paper source for this stage.

Current status, 16 September 2026: the foundation stage is complete. All 18
declared supervised fits finished 320,000 updates; their independent post-fit
audit passed. The final working manuscript reports every control and seed,
the constructed local-control diagnostic and five archived acoustic fits.
The fresh release passed 76 source and 76 isolated-wheel tests, with separate
CUDA runtime and failure/recovery acceptance. Exact source, accepted wheel,
paper PDF/HTML and scientific evidence are sealed in
`results/baseline_closeout_20260916T155000Z/`.

The strongest replicated representation result is learned geometry; population
growth adds an improvement in one paired case. The manuscript retains that
distinction and the recorded operating limits. New applications can now use the
accepted implementation. Author/affiliation details, public archival identifiers,
owner-selected licensing and venue formatting remain editorial release work;
they do not require reopening the foundation or rerunning the completed study.
