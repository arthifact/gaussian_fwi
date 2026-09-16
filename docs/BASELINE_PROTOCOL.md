# Research evaluation protocol

## Question and scope

Evaluate one method: waveform inversion using the continuous Gaussian field and
[adaptive density-control algorithm](ALGORITHM.md). The primary question is how
learnable geometry and spatial scale support useful velocity-model workflows.
The first application is a smooth macro model with localized fine detail. The
[research direction](RESEARCH_DIRECTION.md) distinguishes current capabilities,
constructed diagnostics, learned demonstrations and future extensions. Accuracy,
structure, storage and work are measured for the particular application.

The [baseline closeout](BASELINE_CLOSEOUT.md) fixes the present foundation and
paper stage: accepted software, the declared macro/detail representation study
and the archived acoustic application evidence. The broader held-out geology,
coverage/noise and external-comparison campaigns below support the corresponding
future generalization or relative-performance claims. They do not reopen an
accepted implementation or prevent a new application from using it.

There is one shipped optimizer and one baseline profile. Use controlled mechanism
ablations to isolate the representation claims. An independently maintained
external reference is needed when a specific relative-performance claim calls
for it; beating grid FWI is not the overall objective. Controls are experiment
definitions, not additional public inversion engines in this repository. Record
each external implementation, version and numerical acceptance before using it.

## Freeze the experiment before final evaluation

Record physical units and domain, data provenance, wavelet, acquisition,
propagation/PML settings, initial background, noise model, partitions, loss
normalization, regularization, Gaussian seed, density settings, budget and
selection rules. Use development cases to select these settings; reserve final
cases and test receivers for evaluation. Preserve failures and every attempted
configuration in an external immutable campaign record.

Train with training waveforms. Select within each stage's fixed-population
settling window using validation waveforms. Test data and reference velocity
must not affect fitting, density decisions, tuning or checkpoint selection.
Generate independent evaluation data on a justified finer discretization or
independent solver, with consistent physical source/receiver footprints. A
same-grid noiseless reconstruction is an engineering diagnostic, not adequate
independent seismic evidence.

## Substantial convergence allowances

| Development tier | Updates per frequency stage | Four-stage total |
|---|---:|---:|
| Initial | 1,000 | 4,000 |
| Extension | 2,000 | 8,000 |
| Further extension | 4,000 | 16,000 |

One update includes all training shots, irrespective of microbatch size. Short
unit fixtures and throughput probes are not fits. Extend development runs when
late training/validation trends, selected-step locations or budget extensions
show meaningful unfinished optimization. Preserve each run; do not overwrite it.
An exhausted budget is not proof of convergence. Report late-window loss
changes and parameter/field changes alongside complete curves.

Changing `steps_per_stage` with an unchanged `stop_fraction` also changes the
number of density opportunities. For a controlled study of longer settling,
keep the absolute refinement stop at update 500: use stop fractions 0.5, 0.25
and 0.125 for the three tiers above, keeping warm-up, interval and all other
settings fixed. Store each complete profile in the campaign record. Comparisons
that instead scale the refinement window evaluate a different resource
allocation and must say so. Current stage restart does not extend an already
completed stage; run each declared budget from the same initialization until a
separately accepted continuation facility exists.

A supervised representation diagnostic has a separate question:
accuracy of fitting a known field. Budget 5,000 updates, then 10,000 or 20,000
when needed. Label truth supervision explicitly and never present these fits
as FWI results. No supervised study engine is bundled here.

## Evaluation matrix

For the macro/detail application, predeclare both a spatial macro operator and
the local structures of interest. Component width alone does not define a
low-pass model. Report macro error, detail error, their coupling, physical scale,
parameter count and full cost. Frozen broad parameters do not imply an unchanged
smoothed total field. Constructed-field checks establish controllability only;
supervised fits and waveform-only recovery provide distinct evidence. Geometry
or component-density plots must not be labeled seismic resolution or uncertainty
without a separately calibrated diagnostic.

Begin with two development FWI cases exposing distinct structure: inclined thin
layers and a faulted or curved model. Include both positive and negative
contrasts, coverage limitations and a nontrivial initial model. Then freeze a
held-out matrix spanning at least three distinct structures and three recorded
random seeds per configuration. Determine final run counts from measured GPU
cost, not from short CPU fixture timing.

Ablate the density mechanism only in explicitly archived research branches or
external harnesses: fixed population as a control, clone/split contribution,
pruning and field-change safeguards. Keep initialization and data identical,
measure convergence and account for all work. Do not equate equal iteration
counts with equal compute, capacity or effective regularization. A factorial
study is useful if interactions matter; it is not evidence until run.

Compare at equal measured work and across accuracy/storage curves where
possible. Report population history, full-covariance parameter counts, serialized
field size, optimizer and wavefield memory, elapsed synchronized time, all
forward/adjoint shot solves, and any recomputation. Report uncertainty across
seeds and cases rather than selecting the best reconstruction.

## Independent acceptance and reporting

After checkpoint selection, reload the field in a clean process, verify SPD and
physical constraints, and propagate fresh waveforms. Independently recompute
metrics from saved arrays. Report training, validation and test waveform errors
separately from velocity RMSE/relative error. Include geological structure or
interface metrics where the claim requires them, acquisition coverage, noise
sensitivity and failure cases. A visually attractive image alone is insufficient.

Sampling checks should refine the same physical domain, assess interpolation
and amplitude errors, and separately check waveform convergence. The covariance
width floor does not certify alias-free summed fields or seismic resolution.

A publication claim requires completed evidence for the stated capabilities,
appropriate controls, justified physical assumptions and reproducible methods.
Define the contribution relative to prior Gaussian/RBF and implicit-field
inversion literature. The software release gate establishes software properties;
application and algorithmic claims require their own experiments.
