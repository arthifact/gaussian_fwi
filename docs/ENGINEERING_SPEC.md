# Engineering specification: one Gaussian FWI method

## Current research direction

The owner's 16 September 2026 clarification makes the primary contribution an
explicit, differentiable, adaptive velocity representation for several workflows.
The first application is a smooth macro model with localized fine detail.
[RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md) defines the active questions,
capability evidence and extension boundaries. A general claim of superiority
over grid FWI is not a project requirement. The dated campaign records below
retain their original questions, measurements and limitations; this clarification
does not relabel their evidence or alter their sealed artifacts.

## Bounded stage and compatibility

Version 0.3 consolidates the previously separate direct/scheduled interfaces into
one `gaussian_fwi` package, one inversion engine, one density controller and one
`configs/baseline.json`. Acoustic support is internal to `gaussian_fwi.core`.
Removed controllers, wrapper packages, profiles and historical experiment
outputs are absent from this source tree. Their old commits and external
research archive preserve provenance.

This is a declared numerical-policy and API change. The new training format is
`gaussian-fwi-adc-training-v1`. Reject older training formats before writing or
propagating. Historical saved fields remain readable; exact field/waveform
replay is distinct from reproducing an old optimization trajectory. The field,
decoder and acoustic arithmetic are retained, with namespace imports adjusted.

The [algorithm specification](ALGORITHM.md) defines the literature mapping,
full-training physical gradient statistic, clone/split/prune operations, sampled
field guard, per-band refinement window and settling-only validation selection.
No alternative controller or silent policy switch is permitted.

## Independent acceptance

- Analytical kernel, dense/sparse/fused gradients, physical finite differences
  and acoustic directional derivatives for every parameter family.
- Mean-of-norms selection, deterministic tie order, clone/split/prune decisions,
  independent child-distribution checks and caller RNG isolation.
- Full-SPD constraints, physical sampling floors, cumulative event bounds,
  survivor Adam state, fresh child state and atomic rollback after failure.
- 2D/3D completion, float32/float64 consistency, zero-gradient behavior,
  training/validation/test separation and fixed-population settling.
- Exact completed-stage CPU restart, observation-content identity, obsolete
  checkpoint rejection, saved-field reload and fresh waveform replay.
- Actual forward/adjoint counters, clean isolated-wheel tests, blocked obsolete
  imports, single-profile structure, documentation links and Linux CI.

Run focused tests while developing and the complete gate before release:

    python tools/validate.py --release

Each release writes a fresh ignored directory containing logs, exact source and
verification-input hashes, environment, wheel identity and discovered test IDs.
The installed suite must have no failures, skips or expected failures.
Historical field references are optional and kept outside the source repository.

## Current acceptance and limitations

The initial local release acceptance passed 61 source tests and the identical 61 tests from
a clean isolated wheel, with no skips. Historical 2D/3D field and waveform replay
was exact. An independent AST audit preserved the arithmetic in 14 field/core
files, excluding namespace imports. A separate 4,000-update CPU diagnostic
completed and passed saved-field, work-count, population and NumPy field-oracle
checks. It does not establish convergence. Generated logs, numerical results and
figures stay outside version control. GitHub Actions runs the same source/wheel
gate on Linux for every main-branch update.
Evidence establishes the tested software properties, not a universal optimum
for the heuristic density settings or a scientific performance result.

The observation-only runner now supports explicit deterministic CUDA float64
as well as CPU. Short CUDA gradient/state, rollback, test-isolation and
completed-stage restart checks passed on the native Windows RTX 4060; detailed
evidence appears in the dated sections below. The latest runtime release passed
72 source and 72 isolated-wheel tests. Completed-update recovery, compatible
shot batching, full-gradient shot accumulation and explicit uncompressed
storage are accepted for the documented fixtures. A declared 20-shot planning
acquisition completed one resource/state check; sustained larger research fits
remain separate work. See [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md).

The dated overnight evidence below records five additional audited development
fits, mixed waveform/velocity accuracy trends, unfinished convergence and failed
strict long-prefix comparisons. Its original research source is archived apart
from the final memory-lifetime corrections. These limitations are part of the
baseline's acceptance record.

The [research protocol](BASELINE_PROTOCOL.md) allocates substantial optimization
budgets and independent evaluation. No convergence, SOTA, Nature-readiness or
large-scale 3D claim follows from the release tests or a small synthetic fit.

## Windows / RTX 4060 first pass (2026-09-15)

Bounded work: install the pinned scientific stack in a fresh local Python 3.12
environment, run source and isolated-wheel acceptance on native Windows, add a
standalone CUDA diagnostic, and complete a timed observation-only software
fixture within the user's one-hour first-pass allowance. The diagnostic calls
the existing field, acoustic operator and optimizer; it introduces no inversion
engine, production profile, training policy or checkpoint-format change.

Independent acceptance is declared before implementation: compare CPU and CUDA
fields, waveforms, training gradients for every parameter family and one Adam
update; check float64 acoustic directional derivatives against central finite
differences at steps 1e-3 and 1e-4. Predeclared relative L2 tolerances are 1e-8
for float64 CPU/CUDA comparisons and 5e-3 for float32 (with an absolute norm
floor of 1e-10). Adam update differences use 2e-2 relative L2 and a 1e-7
absolute norm floor. Directional derivatives use 2e-5 relative plus 2e-11
absolute tolerance. AMP and TF32 are disabled. Record synchronized time, peak
allocated/reserved CUDA memory and actual acoustic work. A failed check remains
a failure; numerical tolerances will not be relaxed silently.

The existing release suite independently checks topology state, rollback,
held-out isolation and CPU restart. The additional CUDA diagnostic does not
certify shot accumulation, GPU topology/restart or campaign-scale memory. A
same-grid synthetic run is a software diagnostic and cannot establish paper
readiness. Evidence and remaining blockers will be recorded after execution.

Additional bounded resource check, declared before execution: one shot on the
campaign planning grid (256x512, 10 m, 4,001 samples at 1 ms, 20 PML cells,
256 receivers, 1,024 zero-amplitude seeds), float32 CUDA with TF32/AMP disabled.
Generate diagnostic observations separately, then measure one existing-field
forward/backward and Adam update. Cap the PyTorch allocator at 70% of device
memory. Require finite values/gradients and exact accounting of two forward
shot solves (including observation generation) and one adjoint. Record the
allocation/reservation peaks and driver-free memory. This checks only the
declared initial population and acquisition; it is not a fit, a fragmented-field
stress test or acceptance for a 20-shot research run.

### Measured first-pass evidence

Native Windows 11 acceptance used Python 3.12.13, PyTorch 2.13.0+cu130,
Deepwave 0.0.27, NumPy 2.5.2 and SciPy 1.18.1 on a Ryzen 9 7940HS and
RTX 4060 Laptop GPU. The project-local environment was installed successfully;
WSL was absent and no WSL installation was needed for these checks.

- The release gate passed 61 source tests and the identical 61 isolated-wheel
  tests with no failures, skips or expected failures. The fresh evidence is
  `results/validation/windows_20260915T172100Z/verification.json`; the wheel
  SHA256 is `54d01135776849f0acf9b1f244280312167ad75002e52b6a8695711836fcfc06`.
  No historical replay archive was supplied in this checkout. This directory
  has no `.git` metadata, so provenance uses recorded source hashes.
- All four CUDA cases (2D/3D, float64/float32) passed, including each parameter
  family, Adam updates and float64 finite differences. The largest CPU/CUDA
  relative L2 discrepancy across compared arrays was 1.96e-13 in float64 and
  4.10e-5 in float32, inside the original tolerances. Evidence:
  `results/cuda/rtx4060_20260915T172100Z/result.json`.
- The separate 256x512 one-shot GPU probe completed one update in 6.34 seconds,
  with 2.67 GiB peak allocated and 2.72 GiB peak reserved CUDA memory and 4.13 GiB
  driver-free memory afterwards. It counted two forward shot solves including
  observation generation and one adjoint. Evidence:
  `results/first_pass_20260915T171800Z/gpu_memory_probe/result.json`.
- A 64x96, two-shot, 1,200-sample CPU observation-only probe completed 16 updates
  across four bands in 32.33 seconds. Its explicitly shortened schedule is saved
  beside the result; `configs/baseline.json` is unchanged. The population changed
  from 1,024 to 917. It counted 44 forward / 32 adjoint shot solves during fitting,
  two more forwards for runner verification and two for the clean-process audit
  (plus two separate observation-generation forwards). The audit reproduced the
  saved field and waveforms exactly, independently recomputed all waveform scores
  with NumPy, checked SPD and settling/topology constraints, and found a maximum
  field-oracle discrepancy of 0.000293 m/s. Evaluation-only velocity RMSE changed
  from 47.79 to 25.18 m/s. Evidence and PNG/PDF figures are under
  `results/first_pass_20260915T171800Z/probe/`.

The CPU probe's median ordinary update took 1.51 seconds. A linear 4,000-update
estimate is about 1.67 hours before population growth and additional overhead,
outside the user's one-hour first-pass allowance. Consequently no 4,000-update
fit was launched. The 16-update probe provides no late-window convergence
evidence. Its same-grid observations, short record, two shots and initial model
close to the reference make it unsuitable for scientific performance claims.
The GPU measurement covers one initial population and one update only; it does
not justify extrapolation to 20-shot batches, 8,192 fragmented kernels or 3D fits.

### Next steps for the supplied models

Inspect the user's model files, physical grid spacing, tensor axes and velocity
units, and distinguish reference models from initial models. Preserve the inputs
outside Git. Define development/evaluation cases and generate observation-only
bundles in a separate process under the baseline protocol. Implement and
independently accept full-gradient shot accumulation and completed-update
snapshots, including GPU topology, rollback, held-out isolation and interrupted
continuation, before long GPU fits. Measure the largest planned population and
actual acquisition, then choose a declared multi-hour development budget.
Convergence extensions, independent/finer observations, external grid-FWI
comparisons, mechanism ablations and held-out cases/seeds remain required for
publication claims. The numerical method and baseline profile were not changed
by this first pass.

## Supplied development-model intake (2026-09-15)

Bounded work, declared before numerical preparation: inspect the five supplied
70x70 arrays and verify their manifest hashes, dtype, tensor axes, units and
finite values; prepare observation-only diagnostic bundles and run the existing
CPU method for a short completion check within the remaining first-pass budget.
The physical spacing must be explicitly declared because the supplied files do
not record it. A 10 m, 690x690 m diagnostic domain is proposed; this does not
reconstruct the original benchmark domains. All inputs remain unchanged and
all experiment scripts, data, profiles and results stay in ignored directories.
No public API, numerical method, baseline or checkpoint format changes.

For the acquisition diagnostic, predeclare three sources, 31 receivers, a
15 Hz Ricker wavelet, a 1.2 s record at 1 ms, fourth-order propagation and 200 m
physical PML widths. Source and receiver Gaussian footprints have 10 m sigma.
Generate observations separately on nested 10/5/2.5 m grids covering the same
domain, using a piecewise bilinear extension of the supplied prepared model.
Preserve physical source/receiver centers, footprint sigmas and integrated source
strength (logical source amplitudes scale inversely with cell area). Save the
5 m predictions as diagnostic targets for the 10 m acquisition; use the 2.5 m
predictions to report discretization discrepancy, not to select optimization
settings. This finer propagation does not recover lost geological resolution.

Independent acceptance: require unchanged input hashes, aligned coarse nodes,
matching physical extents and PML widths, unit footprint masses and consistent
integrated source strength; require finite nonzero traces and exact actual
shot-solve accounting. Compare per-band waveforms across the nested grids and
retain all results, including poor agreement. The observation loader must accept
the saved bundles and reproduce content identities. A clean-process audit must
replay selected fields and predictions and independently compute field and
waveform scores. Use the same background and 16-update diagnostic profile for
every model, with zero-amplitude seeds, fixed receiver partitions and no tuning
from reference-velocity errors. These checks cannot establish convergence,
benchmark superiority or held-out performance.

### Intake and execution evidence

The user confirmed 10 m spacing. All five input hashes match the supplied
manifest, and all arrays are finite float32 (z,x), 70x70, spanning 1,500--4,500
m/s. They are previously resampled and independently rescaled development
inputs. The original benchmark domains and resolutions are not recoverable
from these arrays; the original source files were not supplied.

Five observation-only bundles were prepared under
`data/model_check_20260915T173400Z/`, with 5 m propagation targets for the declared
10 m inversion grid. Geometry, PML extent, coarse-node alignment, independent
bilinear interpolation, footprint masses, integrated source strength and saved
content identities passed. The additional 2.5 m propagation changed the raw
training waveforms relative to 5 m by 0.39--1.39% across the models; 10 m versus
5 m differences were 2.05--5.44%. These are measured discretization discrepancies,
not a universal convergence certificate. CPU/CUDA finite-footprint forward
agreement was also checked on Marmousi at 10 and 5 m; relative L2 discrepancies
were below 9e-6. Generation and these cross-device checks used 51 actual forward
shot solves and no adjoints.

All five 16-update CPU completion checks passed using the same archived profile,
shared 1,500--3,000 m/s background and zero-amplitude initial seeds. Each used
66 forward / 48 adjoint shot solves, three runner-verification forwards and
three clean-process audit forwards. The complete intake/check batch therefore
used 411 forward and 240 adjoint shot solves, including observation generation.
Clean-process field/waveform replay was exact; independent NumPy field and
waveform metrics, SPD, settling and sampled event bounds passed. The largest
field-oracle discrepancy was below 0.000338 m/s.

| Development input | Initial velocity RMSE (m/s) | Selected velocity RMSE (m/s) | Selected population |
|---|---:|---:|---:|
| Marmousi | 512.73 | 488.11 | 1,006 |
| BP2004 | 1,339.93 | 1,297.00 | 1,003 |
| SEAM | 1,344.18 | 1,328.88 | 974 |
| Overthrust | 1,160.47 | 1,106.17 | 1,007 |
| Sigsbee2A | 938.62 | 938.47 | 1,017 |

These velocity metrics were evaluated only after fitting and checkpoint
selection. Training waveform loss fell in every case, while substantial field
errors remain; Sigsbee2A's velocity RMSE barely changed. All stages selected
their final update. Four updates per stage provide no useful late-window
convergence assessment. No settings were adjusted from these results and no
case was selected as a best-performing result.

Evidence, per-split waveform scores, histories, source/input identities and
PNG/PDF figures are in `results/model_check_20260915T173400Z/`, including
`audit.json`, `preparation.json`, `provenance.json` and each immutable run.
The fresh release gate also passed all 61 source and 61 isolated-wheel tests,
with no skips or expected failures, under
`results/validation/model_intake_20260915T173800Z/`.

The next engineering work remains GPU shot accumulation and completed-update
recovery with independent state/topology checks. Longer declared development
budgets and a justified acquisition/initialization study are needed to assess
FWI convergence. Paper evaluation additionally needs original physical model
definitions, an independent evaluation matrix, comparative baselines and
uncertainty across cases/seeds. The prepared 70x70 arrays remain development
diagnostics. No numerical engine, baseline profile or input array was modified.

## Full development allowance requested (2026-09-15)

After the completion probes, the user requested substantially more optimization.
Begin Marmousi with the unchanged baseline: four stages of 1,000 complete
training updates (4,000 total), three shots per update, refinement ending at
update 500 in each stage, and settling-only validation every 25 updates. Reuse
the frozen observation-only 10 m acquisition and 5 m synthetic targets from the
model intake. Use the accepted CPU runner and a fresh output directory; no
numerical/API or checkpoint-format changes. Preserve the short probes.

Independent acceptance after selection: clean-process field and waveform replay,
NumPy covariance/field and per-partition waveform metrics, unchanged input/source
identities, sampled topology limits and actual shot-solve accounting. Expected
fit work is 12,018 forward and 12,000 adjoint shot solves, plus three runner
verification forwards and three separate audit forwards. Evaluate reference
velocity only after selection. Record all stages' selected updates, full curves,
late-window loss changes and field/parameter changes when available. Do not infer
convergence from completing the allowance. If an 8,000-update extension is
warranted, use a fresh run with the same initialization and absolute refinement
stop at 500 (stop_fraction=0.25), as required by the baseline protocol.

This is a multi-hour CPU development run. Existing persistence covers completed
frequency stages; mid-stage recovery is still unavailable.

The user selected Marmousi first, followed by convergence assessment. A
configuration-level limitation was identified while this fixed run executes:
on the supplied 70x70 grid, the sampling floor is 11.8313 m but the baseline
clone cutoff is 6.9 m. Every admissible Gaussian therefore takes the split
branch when eligible for growth; cloning cannot occur. This follows from grid
resolution and the two declared policies, independently of reference values or
optimizer outcomes. Splits, pruning and field-change guards remain active.
Changing spacing alone does not remove this limitation because both lengths
scale with spacing. Record actual edits and account for this restriction in
future mechanism studies; the running baseline is not being retuned.
The calculation is saved in
`results/development_20260915T175000Z/density_scale_analysis.json`.

### Completed Marmousi allowance and independent assessment

The unchanged baseline completed all 4,000 updates on CPU in 7,113.86 seconds
(1 h 58 min 34 s of measured fit time). The runner exited successfully at
19:48 UTC. The temporary system-sleep request was released automatically.
Evidence, PNG/PDF figures and a readable report are preserved in
`results/development_20260915T175000Z/`; the selected fit is `marmousi_4000/`.

Fresh-process field and waveform replay was exact. The independent NumPy
full-covariance field oracle had maximum error 0.00065744 m/s; independent
FFT/preprocessing/partition scores passed the declared tolerances. Covariances
were SPD, with minimum eigenvalue 139.9803 m2. Source/input identities,
settling-only minimum-validation selection, fixed settling populations, event
population accounting and reported sampled event bounds passed. Actual work
was 12,018 forward / 12,000 adjoint shot solves, plus three runner-verification
forwards and three independent-audit forwards. Observation generation and
earlier probes remain separately counted.

After selection, velocity RMSE decreased from 512.73 to 283.94 m/s (44.62%).
Raw, unfiltered relative L2 waveform errors decreased from 32.546% to 0.895%
on training receivers, 32.489% to 1.164% on validation receivers and 27.510%
to 1.124% on test receivers. Test receivers belong to this same development
model/acquisition; this is not held-out geological generalization. The figure
shows recovered broad structure and remaining thin-layer/deep-field errors.

| Highest active band (Hz) | Selected stage update | Training change, updates 900--1000 | Validation change, updates 900--1000 |
|---|---:|---:|---:|
| 4 | 1,000 | -2.81% | -3.56% |
| 7 | 975 | -2.16% | -2.19% |
| 12 | 1,000 | -2.71% | -3.54% |
| 20 | 950 | +182.77% | +97.21% |

Convergence is not established. Earlier stages continued improving near their
allowances. In the final stage, training loss rose to 0.00157266 at update 978,
44.45 times the selected-update value, while the population was fixed. The
regularizer barely changed at that peak. The last update had not recovered;
settling-only validation correctly restored update 950. Histories identify
the loss excursion but cannot determine its parameter/optimizer cause.
Late-window field and parameter changes are unavailable because only selected
completed-stage states are persisted. Stage-to-stage field changes are recorded
separately and do not establish late-window convergence.

All 32 density events accepted only pruning: 74 prunes, no clones and no
splits, leaving 950 Gaussians. Geometry rejected 30,372 proposals; field-change
batch reductions rejected 869 proposals. The largest reported accepted sampled
event change was 0.4834 m/s, below 25 m/s. Event fields are not separately saved
for independent guard recomputation; software tests establish guard/state and
rollback behavior. Together with the clone-cutoff restriction, this fit provides
no evidence of an adaptive-growth benefit. The final field uses 5,702 learned
values (22,808 parameter bytes; 25,817-byte field checkpoint), versus 4,900
float32 grid values (19,600 bytes), so it also provides no compression result.

The release gate previously run into
`results/validation/model_intake_20260915T173800Z/` passed all 61 source and
61 isolated-wheel tests. At completion, all 19 recorded package-source hashes
and 27 verification-input hashes still matched that gate. No numerical source,
production profile or supplied model was changed during the long fit.

Next bounded engineering work: diagnose late settling instability with
completed-update field/optimizer and gradient/state evidence, and independently
accept recovery before larger campaigns. A controlled 8,000-update profile is
archived but unexecuted; it retains refinement stop 500 and the same initial
state. Longer optimization alone is not a stability certificate. All tuning
must remain based on training/validation evidence, not test or true velocity.
CUDA shot accumulation, topology/state and restart gates still precede long GPU
fits. Original physical model definitions, independent comparison methods,
held-out cases and multiple seeds remain necessary for publication claims.
Other supplied models were not queued for long fits, as requested.

## Five-to-fifteen-minute development cycle (2026-09-15)

The user requested a maximum 5--15 minute run rather than repeated two-hour
CPU allowances. Bounded first step: benchmark the existing full three-shot
Marmousi update on CPU and CUDA at the zero seed and saved developed fields.
Reuse the frozen acquisition, observations, preprocessing and objective; load
no reference velocity. Compare field, prediction, objective and every parameter
gradient at identical saved states, then compare one Adam update. Time warmed,
synchronized existing updates and record actual solves and peak device memory.
This is a diagnostic script in ignored results, not a new inversion engine or
an accepted GPU fit. Production method/profile/checkpoint formats are unchanged.

Acceptance uses the existing CUDA diagnostic tolerances: float32 relative L2
5e-3 plus absolute L2 1e-10 for values/gradients; Adam update relative L2 2e-2
plus absolute L2 1e-7. Disable AMP/TF32; cap the CUDA allocator at 70% of VRAM.
Measure the real acquisition before extrapolating. Full-GPU topology, shot
accumulation and restart requirements remain separate. If the requested wall
time requires a shorter allowance, label it a development diagnostic and
archive its complete changed schedule; preserve the 4,000-update baseline fit.

The actual full-shot benchmark measured 1.72--1.89 s per CPU update and
0.477--0.555 s per CUDA update, implying roughly 32--37 minutes for 4,000 GPU
updates before refinement/export overhead. The zero-seed comparison passed;
the developed-state comparison failed the original gradient/update tolerances
(largest gradient relative L2 discrepancy 8.12%). Tolerances are unchanged and
the failure is preserved in `results/fast_cycle_20260915T201200Z/benchmark.json`.
This is not accepted GPU inversion performance. Peak reserved VRAM was 468 MiB.

For an immediate bounded development run, archive an 80-update-per-stage CPU
profile (320 total). Scale schedule times from the baseline by 0.08: warm-up,
interval and minimum age 4; validation interval 2; refinement stop 40. Keep
physical inputs, all shots, initialization, learning rates and all other
policies fixed. Expected fit work is 978 forward / 960 adjoint shot solves,
plus three runner-verification forwards and three independent-audit forwards.
The measured update times predict roughly 9--11 minutes including overhead;
allow at most 15 minutes for the run. This is an explicitly shortened schedule,
not a converged fit or a controlled settling-only extension. Verify clean-process
replay, NumPy field/waveform metrics, selection eligibility, unchanged identities
and work counts after completion; load reference velocity only after selection.

The user then explicitly required GPU-only fitting. The newly started CPU
development run was interrupted and its partial output preserved; do not resume
it. Continue with CUDA diagnostic acceptance and a GPU development allowance.
Investigate the failed near-solution float32 comparison using float64 CPU/CUDA
reference computations, with relative L2 1e-8 plus absolute L2 1e-10 for values
and gradients; keep the original Adam update tolerances. These are small
reference checks, not CPU fitting. Preserve the failure and all original fits.

Bounded implementation if these checks pass: add explicit device/precision
options to the single observation-only runner and portable observation loader,
preserving CPU defaults and existing file formats. Use the existing inversion
engine unchanged, and declare a short CUDA development run capped at 15 minutes.
Require GPU topology/state/rollback checks, completed-stage restart, finite
gradients and measured memory, fresh GPU replay and independent NumPy metrics.
CPU replay stays exact; CUDA replay uses declared numerical tolerances and must
not be mislabeled bitwise-exact. Full campaign shot accumulation and completed-
update recovery remain outside this short-run acceptance and required for the
larger research campaign. Run source/isolated-wheel release checks for any
public device/precision support changes.

The float64 full-acquisition comparison passed, including developed-state
gradients (largest relative L2 discrepancy about 1.35e-10). A CUDA restart
fixture then exposed a different clone choice among coincident parent/child
kernels despite initial losses agreeing to roundoff. CUDA scatter accumulation
is a candidate source; enforce deterministic PyTorch algorithms and the required
cuBLAS workspace configuration before repeating the same fixture unchanged.
Do not alter comparison tolerances or remove the coincident-clone case. This is
a CUDA execution-policy change only; CPU defaults and numerical policies remain.

The unchanged deterministic CUDA fixture passed all seven topology/state tests,
completed-stage restart (including coincident clones), test-waveform isolation
and actual shot accounting. The fresh waveform replay relative L2 discrepancy
was 2.30e-17 against the declared 1e-10 relative plus 1e-10 absolute norm bound.
Preserve both earlier failures and this accepted run in
`results/fast_cycle_20260915T201200Z/cuda_acceptance_deterministic/`.

Declare the timed GPU development allowance before launch: four stages of 200
updates (800 total), with warm-up, density interval and minimum age 10,
validation interval 5 and refinement stopping at update 100. This scales the
baseline schedule by 0.2; all shots, physical inputs, initialization, learning
rates and numerical policies remain fixed. Use deterministic CUDA float64 and
a fresh output. Expected fit work is 2,418 forward / 2,400 adjoint shot solves,
plus three runner-verification forwards and three independent-audit forwards.
Allow at most 15 minutes for the fit and runner verification. This is a shortened
development schedule, not a convergence claim or a controlled extension of the
4,000-update fit. Before evaluation-only reference loading, verify source/input
identities, GPU replay, independent NumPy field and waveform metrics, settling
selection and counted work. Preserve the CPU interruption and all GPU attempts.

The public runtime changes passed `python tools/validate.py --release` into
`results/validation/gpu_runtime_20260915T203400Z/`: repository/link checks,
Ruff and all 62 source plus 62 isolated-wheel tests, with no skips. The GPU
runtime fixture is a separate hardware-dependent gate; CPU release fixtures
remain software tests, not CPU research fits. GPU data movement is explicit
and unavailable CUDA fails without CPU fallback. Sparse neighbor search and
I/O retain their CPU components.

### Completed fast CUDA development run

The frozen 800-update Marmousi profile completed on the RTX 4060 in 636.92 s
of engine-reported fit time and 642.16 s (10 min 42 s) of complete runner wall
time. The automatic 900-second subprocess timeout did not fire. A separate
fresh-process audit and PNG/PDF export completed in approximately 8.5 s.
The output is `results/fast_cycle_20260915T201200Z/marmousi_cuda_800/`; the root
of that experiment contains the archived profile, launch command, timing,
diagnostics, preserved failures and readable report. One-time environment and
runtime implementation/acceptance work is not included in this per-fit timing.

The CUDA float64 field replay was exact; fresh waveform replay relative L2
error was 3.25e-17, within the predeclared 1e-10 relative plus 1e-10 absolute
norm bound. The independent NumPy field maximum error was 4.55e-12 m/s, and
minimum covariance eigenvalue was 139.980388 m2. Independent waveform metrics,
observation/source identities, selection eligibility and actual work counts
passed. Work was 2,418 forward / 2,400 adjoint shot solves plus three runner
and three audit forwards. Peak allocated/reserved CUDA memory was 858.70 /
1,038 MiB, with 5.85 GiB driver-free at completion. These memory figures are
for the tested 70x70, three-shot development acquisition only.

Reference velocity was loaded only after the independent selection/replay
checks. Evaluation RMSE fell from 512.73 to 364.64 m/s (28.88% reduction),
versus 283.94 m/s after the earlier 4,000-update CPU run. Raw waveform relative
L2 errors became 1.506% training, 1.907% validation and 1.927% test. The runtime
reduction combines GPU execution, fewer updates, a scaled density schedule and
float64 arithmetic; it is not an equal-budget speed comparison or proof of
equivalent solution quality.

All four stages selected their final update 200. Over updates 180--200,
training losses decreased by 12.45%, 6.49%, 8.27% and 4.86%; validation losses
decreased by 12.20%, 5.95%, 4.70% and 5.77%. This is unfinished optimization,
not convergence. The 32 density events accepted 92 prunes, no clones and no
splits, ending with 932 Gaussians and 5,594 parameters. The largest reported
sampled event field change was 0.461806 m/s. The sampling/clone incompatibility
and lack of accepted growth remain, as do the absence of saved per-event
fields and late-window parameter/field changes.

This measured profile meets the user's 5--15-minute development-cycle budget
on this machine. It does not replace the research protocol's convergence and
comparative evidence. Next numerical work remains diagnosing settling
instability, accepting completed-update recovery and studying sampling-aware
initialization/density settings from training/validation evidence. Larger GPU
acquisitions still require shot accumulation and intended-scale stress checks.
The interrupted CPU attempt has no final solve counter; its last observed
console update is recorded without pretending it is complete work accounting.

## Three-hour baseline development campaign (2026-09-15)

The user authorized up to three hours from approximately 20:59 UTC for focused
GPU baseline development. Retain the 15-minute limit per process and avoid
repeated broad test runs. This is authorization for implementation and measured
FWI experiments, not a precommitment to a favorable result or paper-ready claim.
Preserve the existing method, production profile and all evidence before edits.

First bounded change: optional completed-update persistence, bounded execution
segments and gradient/update diagnostics in the existing inversion/runner.
Numerical updates, loss, refinement, selection and default execution remain
unchanged. Preserve completed-stage checkpoint compatibility; use a distinct
versioned format for update boundaries. Save field, Adam, topology/controller
statistics, eligible best checkpoint, histories, RNG, cursor and counted work.
Resume into a fresh directory without extra or repeated acoustic solves.
Invalid/incomplete state must fail before output creation or propagation.

Independent acceptance before research use: compare uninterrupted and segmented
CPU trajectories exactly, including a pause before/after density events and
within settling; compare CUDA float64 trajectories at existing 1e-8 relative /
1e-10 absolute tolerances with identical discrete edits. Perturb test traces to
confirm isolation; reject corrupted cursor/controller/optimizer state. Verify
actual solves, no overwritten fits and selected checkpoint equivalence. Run
focused checks once per change and a fresh full source/wheel release gate for
the final public implementation.

Then conduct a bounded training/validation-only configuration study on the
frozen Marmousi acquisition: physically broader zero-amplitude seed populations
and clone thresholds compatible with the existing sampling floor. Keep the
25 m/s cumulative field guard, signed child amplitudes, full SPD covariance and
fresh child Adam state. Declare each complete candidate profile and comparison
criterion before launching it. Do not use test losses or reference velocities
to choose or revise candidates. A gradient score or accepted growth count is
not a measured quality gain. Preserve failures and actual runtime/work.

Use short 800-update GPU schedules for screening, then reserve the remaining
budget for substantial convergence and an independent second-case check. A
candidate must improve validation at a comparable measured work allowance to
justify promotion. Full campaigns/comparative novelty and original physical
benchmark data remain outside what three hours can guarantee. Record actual
evidence, unresolved limitations and next steps; never label an exhausted
allowance or software test as a high-fidelity reconstruction result.

Completed-update CPU 2D/3D checks and the real-acquisition CUDA float64 restart
fixture passed, including retained best selection, density statistics and exact
work counts. Two implementation bugs found by focused checks (portable best
background parameter identity and JSON tuple/list comparison) were corrected;
no numerical tolerance was changed.

Bounded throughput decision before configuration fits: compare existing serial
shot propagation with a single existing acoustic call containing the three
shots, only when their quadrature node counts match. Use unchanged source and
receiver operators, no padding, no changed physical acquisition and no reference
velocity. Compare predictions, objective, every gradient and one Adam update at
zero and developed fields; use the existing float64 1e-8 relative / 1e-10
absolute value/gradient tolerances and 2e-2 / 1e-7 Adam tolerances. Count physical
shot solves separately from batched calls. Only implement explicit batching if
this measured diagnostic passes and offers meaningful runtime savings.

The three-shot diagnostic passed the unchanged float64 tolerances at both
states (largest developed gradient discrepancy 2.53e-10 relative L2). Warm
update medians changed from 0.602/0.695 s to 0.384/0.481 s at zero/developed
fields. Bounded implementation: explicit finite-footprint shot batching groups
consecutive shots with equal quadrature-node counts, without padding or changing
operators. Default batch size one preserves existing behavior and file formats.
Runtime grouping is recorded separately from physical observation identity and
must match a restarted trajectory. Continue to count individual physical shot
solves, and additionally report acoustic batch calls. This is throughput
batching, not memory-saving gradient accumulation. Require independent serial/
batched gradients, unequal final batch accounting, and GPU update restart before
fitting with it; retain the 70% allocator cap.

Screening profiles are now frozen before launch (all four bands, 200 updates
per stage, float64 CUDA, three-shot batching, checkpoint interval 100): C0 is
the previous 32x32 seed with the 0.2-scaled schedule; C1 uses a 16x16 zero seed
and clone extent fraction 0.03 (20.7 m, above the 18.93 m split-width scale),
with all other C0 settings retained; C2 differs from C1 only in warm-up 0,
event interval 5 and minimum age 5, to assess early admissible growth. C0 is
rerun under the same new execution settings for a matched comparison.

Screening promotion requires finite accepted states, at least 10% lower final
settling-selected validation objective than C0, no more learned parameters and
no larger measured runner wall time. Accepted growth alone is insufficient.
If a candidate qualifies, repeat its refinement seed before selecting it for a
4,000-update allowance. Otherwise retain C0 and report a negative mechanism
result. Any later stability candidate must be separately declared from training
diagnostics before its run. Reference velocities and test scores remain sealed
until all configuration decisions for this campaign are frozen. Use Overthrust
as a second structural development check if the remaining budget permits.

### Declared continuous-update stability comparison

The matched C0 800-update run completed in 487.64 seconds, versus 642.16 seconds
for the preceding serial-shot profile, with the same selected validation loss
to numerical precision. Its training-only diagnostics show an ordinary update
near the start of the final band changing the sampled field by 64.46 m/s
maximum (10.60 m/s RMS); a later training loss spike also occurs during settling.
These are continuous optimizer changes, not violations of the 25 m/s density
event guard. Group gradient norms have different units and do not identify a
unique cause. C1 used fewer parameters but had worse selected validation.

Declare a bounded configuration experiment before running either long fit:
P0 is the unmodified production profile (four stages of 1,000 updates, density
stop 500, warm-up/interval/minimum age 50, validation interval 25). S1 differs
only in geometry_lr 0.008 to 0.004 and center_lr_ratio 0.02 to 0.01. The
amplitude/background rate remains 4. Both begin at the same zero-amplitude seed,
use identical observations, deterministic float64 CUDA and three-shot batching.
The shorter C0/C1/C2 schedules are not prefixes of these full schedules.

This tests whether smaller shape updates reduce instability at a substantial
fixed allowance; it does not preselect S1 as a new baseline. Compare settling
loss curves, final settling-selected validation, late-window field/parameter
changes, actual shot solves, elapsed time, memory and population. A promotion
would require at least 10% lower selected validation, no increase in parameter
count or measured time, and replication before changing the production profile.
If no repeat fits the remaining reservation, retain the shipped profile and
report S1 as a development candidate even if it improves validation.

The independent acceptance uses the already accepted restart/batching path,
fresh-process GPU replay, separate NumPy field/waveform evaluation, explicit
selection and solve accounting, and matching the archived complete profiles.
No numerical source changes are required. Each process has a 780-second soft
pause and a 900-second hard timeout; completed-update continuation preserves
the declared 4,000-update trajectory. Reference velocities and test scores stay
sealed until the whole campaign's configuration decisions are frozen.

The completed C1 screen took 397.69 seconds, selected validation 6.64057e-4,
1,418 parameters, one accepted split and 21 prunes. C2 took 488.62 seconds,
selected validation 8.26741e-4, 1,442 parameters, six splits and 22 prunes.
C0 selected 2.09458e-4 with 5,594 parameters and 92 prunes. Neither candidate
met the predeclared validation criterion; C2 also exceeded C0 wall time. More
admissible growth did not improve the measured short-budget fit. Do not promote
either candidate or relax the density guard after observing this result.

If the two substantial fits leave at least 22 minutes before the reporting
deadline, run the already specified C0 short profile on Overthrust and its S1
counterpart (only geometry_lr 0.004 and center_lr_ratio 0.01). Declare these
complete profiles before launch and use the same observation-only selection
and execution. They are a second-case development sensitivity comparison,
not evidence of convergence or a held-out benchmark matrix. Do not revise S1
from Marmousi reference/test results, and retain the production profile unless
all stated promotion and replication conditions have actually been met.

One additional bounded mechanism control is declared now, before final
evaluation: if at least 13 minutes remain after the second-case comparison,
repeat Marmousi C0's 800-update schedule with both minimum_gaussians and
max_gaussians set to its initial population 1,024. These existing limits forbid
all population edits without changing the public method. Archive the profile
only in the campaign. The controller, objective, continuous optimization,
selection and budgets remain identical. Require every history population to
equal 1,024 and every accepted edit list to be empty. This compares the observed
pruning benefit against a fixed population; C0's absence of accepted growth
prevents attributing any difference to cloning or splitting. It is not another
production inversion engine or a search for a favorable result.

The final public implementation passed the fresh release gate in
`results/validation/baseline_campaign_20260915T214317Z/`: repository/link checks,
Ruff, 66 source tests and 66 isolated-wheel tests, with no skips. The gate took
160.15 seconds. Its verification manifest records package, runner, profile and
test/tool source hashes. The separate deterministic CUDA acceptance covers
serial and three-shot completed-update recovery, topology, rollback and test
isolation at the existing tolerances. Keep the numerical source fixed through
the timed comparisons. No repeated full test suite is needed absent a source
change or newly discovered failure.

### Conditional substantial capacity comparison

C1 selected its terminal update in every short stage, so its poor 800-update
validation alone does not distinguish slower convergence from limited capacity.
Before opening any final evaluation, reserve one optional substantial C1 fit
after the two Overthrust short fits, only if at least 46 minutes remain before
the reporting deadline. Use the full P0 schedule and change exactly the C1
seed shape (16x16) and split extent fraction (0.03); all other settings remain
P0, including learning rates. Start from zero amplitude in a fresh directory.
This is a declared accuracy/parameter/work diagnostic at 4,000 updates, not a
promotion of a candidate that failed the short screening rule. The production
profile and original promotion criteria remain unchanged. It takes priority
over the optional 800-update fixed-population control, which retains its own
13-minute budget condition. Use the same independent replay and accounting
acceptance; preserve any budget stop. No additional public numerical code or
test sweep is needed for these existing configuration choices.

The full production-profile P0 CUDA fit completed 4,000 updates in 2,013.05
seconds (33 min 33 s) across four fresh process segments. The runner's selected
field replay was exact and waveform replay relative L2 discrepancy was
2.96e-17, below the unchanged tolerance. Counted fit work was 12,018 forward /
12,000 adjoint physical shot solves, plus three runner verification forwards.
The selected validation objective was 5.95083e-5; each of the four stages
selected its terminal update 1,000. This is a substantial allowance but does
not establish convergence. The population ended at 949 (5,696 parameters),
with 75 prunes and no accepted clones or splits. Independent post-freeze
evaluation remains pending at this entry; no reference or test metric was
used to choose subsequent profiles.

S1 completed the same 4,000-update allowance in 1,910.83 seconds (31 min 51 s),
with selected validation 6.07266e-5, 937 Gaussians and 5,624 parameters. Its
validation was 2.05% worse than P0 despite 5.08% less measured wall time and
12 fewer components. It accepted 87 prunes and no clones/splits; all four
stages again selected update 1,000. Fit work was the same 12,018 forward /
12,000 adjoint physical shot solves. It fails the predeclared validation
promotion criterion, so the production configuration remains unchanged.
The two already declared Overthrust development fits proceed next. Reference
and test metrics remain excluded from configuration decisions.

The Overthrust short comparison completed C0 in 507.69 seconds with selected
validation 0.00161684, 908 components and 5,450 parameters; S1 took 481.67
seconds with validation 0.00196358, 904 components and 5,426 parameters.
S1's validation was 21.45% worse. Both selected stage-terminal update 200
throughout and accepted pruning only (116 and 120 removals respectively).
This second-case development evidence also fails the proposed shape-rate
promotion criterion. Each fit used 2,418 forward / 2,400 adjoint physical
shot solves before runner verification. The conditional full C1 diagnostic
started with more than 46 minutes remaining before the reporting deadline;
the unchanged production profile and reference/test exclusion remain in force.

Recorded final-stage settling diagnostics further delimit the stability claim:
P0's largest training objective was 70.89 times its terminal objective (step
784), versus 31.88 for S1 (step 754). The largest sampled ordinary field-update
maximum during settling was 15.04 m/s for P0 and 11.22 m/s for S1; these are
maxima over the diagnostic cadence, not every update. Over the final 100
updates, validation still decreased 3.50% and 3.03%, respectively. Reduced
spike magnitude did not translate into better selected validation. An explicit
continuous-update stability and convergence study remains necessary; density
transaction safety alone does not establish stable continuous optimization.

The conditional full C1 fit completed 4,000 updates in 1,791.03 seconds
(29 min 51 s), with 245 components and 1,472 parameters. It accepted three
splits and 14 prunes, with no clones; its largest reported cumulative sampled
event change was 23.7073 m/s, below the unchanged 25 m/s guard. Selected
validation was 0.000137584, 2.312 times full P0, despite about 74.16% fewer
parameters and 11.03% less measured runner time. All stages selected update
1,000. C1 did not catch P0 at this fixed allowance, but terminal selections
still preclude attributing the gap solely to a representation capacity limit.
This exploratory comparison does not alter the failed short-screen decision
or the production profile. Physical fit work was again 12,018 forward /
12,000 adjoint solves. The declared 800-update fixed-population control began
with more than 13 minutes remaining before the reporting deadline.

### Completed campaign and frozen evaluation

All nine declared GPU fits and their independent audits completed by 23:45 UTC.
The fixed-population control took 386.86 seconds, retained exactly 1,024
components without edits, and selected validation 0.000209047, versus
0.000209458 for C0 in 487.64 seconds. C0 reduced parameters by 8.98%, but its
pruning provided no validation or runtime benefit in this single short pair.
No candidate met the predeclared promotion criteria; the production profile
SHA256 remains `dd789705e256339d807941d6d417303bd0f9f36aa241bf628c423297b7e82cbc`.

Configuration decisions froze before the new test/reference evaluation. This
does not make these previously inspected development inputs a blind benchmark.
Fresh GPU field replay was exact for all nine fits; waveform replay, independent
NumPy full-covariance fields and partition metrics, SPD, immutable identities,
settling-only selection, fixed settling populations and actual work passed.
The campaign used 16,800 updates, 50,562 forward / 50,400 adjoint shot solves;
runner/audit verification added 54 forwards. The recorded GPU engineering
checks used another 312 forward / 216 adjoint solves. Total research runner
time was 8,465.05 seconds; the longest individual process was 613.47 seconds.

Post-selection Marmousi P0 velocity RMSE was 283.65 m/s from 512.73 m/s, with
raw test waveform relative L2 1.114%. S1 achieved 291.82 m/s and 0.999%; the
smaller C1 achieved 276.68 m/s and 1.584%. These evaluation differences did
not override validation-based decisions. Overthrust C0/S1 RMSE was 982.99 /
1,028.07 m/s from 1,160.47 m/s. Considerable field error remains. Every stage
of all nine runs selected its terminal update. Final-100-update field RMS
changes for P0/S1/C1 were 3.033 / 2.998 / 4.257 m/s, with validation still
decreasing. Neither convergence nor adaptive-growth superiority is established.

Complete profiles, commands, immutable segments, independent audits, metrics,
PNG/PDF figures, a method diagram, reproduction instructions and a qualified
working manuscript are under `results/baseline_campaign_20260915T210000Z/`.
Its `completion.json`, `decisions_frozen.json` and `work_ledger.json` record
scope and timing. The 66-source/66-wheel release gate was already run into a
fresh directory; verify its source hashes before sealing, without repeating
unchanged numerical tests. Next work is full-gradient shot accumulation,
controlled convergence extensions and substantial second-case/mechanism
comparisons. Original physical benchmarks, an independent grid-FWI comparator,
noise/coverage studies and meaningful repetitions remain necessary for a paper.

## Overnight baseline allowance (2026-09-16)

The user authorized at least eight hours of autonomous work while asleep.
Bound this reservation to eight hours from 05:48 UTC, with a reporting reserve
and GPU-only research fits. Continue the existing 780-second soft execution
boundary and 900-second hard timeout per subprocess. Preserve prior fits and
the sealed three-hour campaign. No external publication or paid service is
part of this local research allowance.

First bounded implementation: optional memory-saving gradient accumulation
over the existing finite-footprint shot batches. Backpropagate each acoustic
batch into a detached velocity leaf, free its wavefields, then propagate the
complete velocity gradient through the Gaussian decoder exactly once and add
regularization once. Every update still includes every training shot. Use the
survey-wide training-derived weights and denominators, correctly weight an
unequal final batch, and observe topology only after the complete gradient.
Retain detached predictions for validation at the existing inspection cadence.
Record actual shot solves and acoustic calls, including adjoints already
performed before a history row is written. No extra waveform solve is required.

Compatibility: default execution, objective, numerical configuration and old
checkpoints remain unchanged. Accumulation is an explicit runtime option for
finite-footprint acquisitions; its setting is part of restart specification
and must match. Point acquisitions remain supported by the existing path.
This adds no inversion engine, physical operator or production profile.

Independent acceptance before use: compare full-graph and accumulated losses,
every parameter-family gradient, density scores and one Adam update for unequal
shot batches, nonzero regularization, nonuniform training weights and 2D/3D
fields. Use existing float64 tolerances (1e-8 relative / 1e-10 absolute for
values and gradients; 2e-2 / 1e-7 for Adam), without silent relaxation. Check
finite differences, held-out isolation, exactly counted work, CPU interrupted
continuation, and deterministic CUDA continuation across density and settling.
Measure peak allocated/reserved CUDA memory on the real acquisition and a
bounded stress fixture before claiming memory savings. Run focused checks and
one fresh source/isolated-wheel release gate after the implementation settles.

The planned research queue uses the accepted, frozen numerical source and
training/validation decisions only: Marmousi P0 at 8,000 updates, Overthrust P0
at 4,000, and a matched 4,000-update Marmousi fixed-population control. Controlled
extensions retain the absolute refinement stop at 500, warm-up/interval/age 50,
validation interval 25 and the same zero-amplitude initialization. Declare full
profiles and expected work before launch. If the 8,000-update trajectory still
selects near-terminal states or improves validation by more than 1% over the
last 10% of any stage, use a fresh 16,000-update Marmousi extension when measured
time leaves the reporting reserve. An Overthrust 8,000-update extension has
priority over additional configuration search if its substantial run remains
unfinished by the same criterion. Reserve at least 30 minutes for independent
replay, complete work accounting, figures and the engineering record.

These are previously examined development models. Do not treat them as unseen
held-out geology. Test scores and reference errors remain excluded from every
new budget/configuration/selection decision, and are reopened only after this
campaign's decisions freeze. A fixed-population advantage can motivate future
mechanism research but does not by itself replace the adaptive production
method. Longer settling alone does not certify stability or publication quality.

Predeclare the additional memory stress diagnostic: 256x512 at 10 m, three
shots and 256 receivers, 4,001 samples at 1 ms, fourth-order propagation,
20 PML cells and 10 m Gaussian source/receiver footprints. Prepare synthetic
software-check observations in a separate process from an analytic ramp plus
a smooth perturbation. They are not a geological fit or tuning evidence.
Use deterministic CUDA float64, accumulation with batch size one and the 70%
allocator cap. Measure one complete training update for 1,024 regular zero seeds,
4,096 and 8,192 signed fragmented components in 16/64 blocks, and 8,192 locally
overlapping signed components. Signed developed fixtures explicitly test software
states; actual research fits still start at zero amplitude. Preserve every
attempt, including OOM or timeout, without retrying at a smaller size under the
same name. Bound each case to 180 seconds. Require finite fields/gradients/SPD,
three forward and three adjoint solves per completed update, peak reserved at
most 6 GiB and at least 1 GiB driver-free afterwards before treating a case as
within the planning memory margin. This checks the specified layouts only,
not arbitrary worst-case covariance overlap or large-scale inversion accuracy.

The real-acquisition accumulation benchmark passed all zero/developed-state
loss, gradient and Adam comparisons, including an unequal two-plus-one batch.
At the developed field, full-graph single-shot execution used 904,626,176 bytes
peak allocated; accumulating one shot at a time used 393,845,248 bytes (56.46%
less). Warm update medians were 0.687 and 0.693 seconds, respectively. Three-shot
full-graph execution remained faster at 0.486 seconds, using 993,182,208 peak
allocated bytes. Accordingly the controlled small research fits retain the
previous three-shot full-graph execution; accumulation is accepted and measured
separately for memory-limited acquisitions. This avoids confounding their longer
allowances with a changed reduction order.

All four planning-grid device-storage stress cases failed the first forward
under the unchanged 70% cap; none completed an adjoint or update. Peak allocated
memory reached about 5.55 GiB. These failures are preserved with attempted-call
counters, distinct from completed shot solves. They show that shot accumulation
alone cannot fit this single-shot float64 wavefield storage requirement.

Bounded response, declared before implementation: expose Deepwave's existing
uncompressed CPU wavefield storage as an explicit runtime option. Keep CUDA
propagation, full temporal sampling, acquisition, objective and precision fixed;
no lossy compression, disk mode, automatic fallback or changed memory cap.
Storage placement is excluded from physical observation identity but included
in restart specification. Default device storage and old checkpoints retain
their execution. Compare device/CPU-storage predictions, every gradient family
and Adam updates at zero/developed states at the unchanged tolerances; repeat
CUDA accumulation/state/restart acceptance with CPU storage. Require invalid
options and changed restart storage to fail before solves/output. Repeat the
declared stress layouts into new directories only after these comparisons pass,
including host RAM usage and available RAM. A new source/wheel gate is required
because this is another public runtime change. Deepwave's primary documentation
describes [storage options](https://www.ausargeo.com/deepwave/example_storage);
its optional compression is explicitly not used here.

Uncompressed CPU storage passed the real-acquisition value/gradient/Adam and
CUDA state/restart checks. The larger CPU-storage fixtures were stopped before
propagation because measured available host RAM was below the declared 6 GiB
headroom. No user applications were closed and the host threshold was not
relaxed. Next bounded diagnostic: test Deepwave's existing uncompressed disk
storage in isolated ignored scripts, with temporary wavefields confined to
each diagnostic output. Keep full temporal sampling and every operator fixed.
Compare small real-acquisition values/gradients/Adam at the same tolerances
before repeating the planning-grid stress cases. Only expose this optional
runtime mode if those checks and measured disk/host/GPU usage pass; record I/O
cost and require an explicit output-local scratch path. No automatic fallback.

The disk prototype passed all real-acquisition loss/gradient/Adam comparisons.
All four planning-grid cases then completed three forwards and three adjoints,
finite updates and SPD checks. Measured update times were about 21--34 seconds;
peak allocated memory was about 3.48 GiB, reserved about 5.38 GiB, and driver-free
memory exceeded 1 GiB. No temporary wavefield files remained after backward.
Expose explicit `wavefield_storage="disk"` next, with scratch confined to each
new inversion output and no compression. Require low-level callers to provide
an existing scratch directory; restart preserves mode while creating fresh
scratch in its new output. Recheck public-API CUDA continuation and the release
gate after integration. Preserve the prototype evidence as a distinct diagnostic.

One final bounded resource extension is declared: repeat the analytic planning
fixture with 20 shots and 8,192 signed fragmented Gaussians in 64 blocks, using
the accepted public disk/accumulation path, a one-shot batch and the same
physical/numerical settings. Generate its observations separately. Measure one
full gradient/update plus one existing density transaction, with cumulative
sampled guard 25 m/s, physical solve counts (20 forward / 20 adjoint for the
update), finite/SPD state and host/GPU/disk usage. Require the prior memory
margins and preserve any failure; allow 300 seconds for this update process.
No extra acoustic work may be attributed to density proposals. This is the
declared acquisition-scale software check, not a geological reconstruction.

The first 20-shot disk run reported two forward/adjoint entries before an
allocation failure; the last reported call need not have completed. Investigation
found that the suspended batch generator
retained the preceding large nodal receiver tensor even after its backward.
A weak-reference lifetime test independently reproduces that retention. Bounded
fix: release the generator's nodal-trace reference before yielding the logical
receiver prediction, then release the yielded prediction on continuation.
Do not change arithmetic, precision, batch size, dataset, memory cap or thresholds.
Require the lifetime test to pass, repeat gradient/state checks and the same
20-shot resource/transaction fixture in a fresh attempt, then run a new complete
release gate. Retain the earlier accepted small cases and the larger failure.

The lifetime test passed after that fix, but the unchanged 20-shot fixture
exhausted memory after six forward/adjoint entries, again without certifying
completion of the last reported call. Direct storage inspection found
that each filtered target retained 327,761,920 FFT-buffer bytes for only
163,880,960 logical bytes. Bounded follow-up: in accumulation only, compact
these constant targets without changing values or recomputing weights or
denominators, and fill one preallocated detached survey prediction buffer
instead of retaining separate growing batch outputs and concatenating them.
Keep the existing band-contribution reduction and gradient arithmetic. The
default full-graph path is unchanged. Require existing loss/gradient/restart
checks and the same 20-shot acceptance; preserve both prior failures.

After compaction, the 20-shot/8,192-component public disk-storage fixture passed
one complete update (117.32 seconds) and a density transaction (129.95 seconds
total). It counted exactly 20 forward / 20 adjoint solves and zero additional
event solves. All 8,192 full-gradient scores were present. The initial proposed
batch was rejected; the controller accepted 16 prunes, with independently
recomputed maximum grid-sampled velocity change 0.0721266 m/s. An independent
NumPy additive-edit oracle agreed within 4.55e-13 m/s and survivor Adam state
was exact. No children were accepted in this large case; child-state evidence
comes from the separate CUDA transaction fixtures, not that vacuous check.

Peak allocated/reserved CUDA memory was 5,282,952,704 / 5,765,070,848 bytes
(4.92 / 5.37 GiB), with 1,281,822,720 driver-free bytes afterwards. Peak process
working set was 2,021,466,112 bytes, and no temporary wavefields remained. The
failed allocation attempts, host-headroom stops and pre-fix lifetime regression
remain in the ignored overnight record. These are single-update software
resource checks; sustained large-population fitting, noise/generalization and
convergence still need research evidence. Disk I/O cost and per-update time
preclude treating this as a real-time or short-budget large-acquisition fit.

The final implementation passed all 71 source and 71 isolated-wheel tests,
Ruff and repository/link checks in the fresh directory
`results/validation/overnight_accepted_20260916T063000Z/`; the measured gate
time was 202.35 seconds, with native child exit code zero. The final compact
accumulation comparisons also passed every predeclared zero/developed-state
value, gradient and Adam tolerance against the saved full-graph reference.
No tolerances were relaxed. Keep numerical source fixed for the overnight
research queue. Documentation may record new evidence without repeating the
unchanged numerical suite; recheck source identities and local links.

The supplementary historical-prefix comparison did not pass its stricter
1e-8 relative plus 1e-10 absolute tolerance for every parameter. At the common
1,000-update first-stage boundary, one shear tensor had relative L2 difference
1.77e-8; all other compared field and Adam tensors passed, and topology IDs,
edit ages and counters matched. The maximum decoded field difference was
3.72e-5 m/s (RMS 5.75e-6 m/s). Earlier saved prefixes show initially much smaller
differences that grow during optimization. Preserve the failed check; do not
claim bitwise historical trajectory identity or change the threshold. The
accepted short gradient/restart and release checks remain separate evidence.
No causal attribution to runtime arithmetic or nondeterministic reductions has
been established. As a diagnostic requiring no extra fits, compare the common
prefixes of the already declared 8,000- and conditional 16,000-update runs if
both complete. Their results must not alter fitting, budgets or thresholds.

Predeclare one optional sustained public-runner software check after the research
queue, only if at least 3,000 compute seconds remain before 13:18 UTC. Reuse the
separately generated analytic 20-shot observations (256x512, 4,001 samples),
with zero-amplitude 64x128 seeds and a fixed 8,192-component population. Use
four cumulative bands with three updates per stage, validation every update,
and the fixed-population selection window starting at step two. This is 12
software updates, not a research convergence allowance. Keep the production
objective, learning rates, physical acquisition, float64 and 70% cap; use
disk storage and one-shot accumulation. Declare only a factor-two final
field-sampling diagnostic to bound this software check's separate report cost.

Require complete public CLI output, finite parameters/SPD, exact physical work
(360 fitting forwards and 240 adjoints, plus 20 runner verification forwards),
selected-step eligibility, fixed population, unchanged restart execution, empty
temporary wavefields, peak reserved at most 6 GiB and at least 1 GiB driver-free
after each segment. Use fresh output segments, snapshots every update,
at most four updates per process, the existing 780/900-second soft/hard limits,
and a 2,700-second total attempt limit. Retain any failure or budget omission.
This checks repeated full runner execution and segmented state handling at the
declared scale; it does not replace an uninterrupted-versus-resumed trajectory
comparison, a fragmented large-field stress check or geological evidence.
Independently decode 64 predetermined grid points in NumPy, check all population
covariance eigenvalues and all saved parameter/grid values for finiteness.
Record Windows process read/write byte counters for the optional disk-storage
runner check as well as peak memory and elapsed time. These counters describe
process I/O, not physical SSD flash writes; sustained disk campaigns need their
own measured I/O budget.

Predeclare a short post-queue repeatability diagnostic for the unresolved
historical-prefix mismatch. Use the already saved Marmousi 8,000-update run's
first-stage update-1,000 field and the original observations, with the 4 Hz
training waveform objective. At one identical decoded velocity, repeat six
gradient evaluations with three-shot full-graph execution and six with one-shot
accumulation, using device storage, deterministic CUDA float64 and the 70% cap.
This costs exactly 36 forward and 36 adjoint physical shot solves. Take no Adam
updates and inspect no reference velocities or held-out scores. Compare repeated
predictions, velocity gradients and field pullbacks; also repeat field pullbacks
with one fixed velocity gradient to separate that calculation from acoustic
variation. Preserve raw outputs and bitwise equality as well as the original
1e-8 relative / 1e-10 absolute numerical comparisons. Do not reinterpret the
failed historical check or change its threshold. This is a local operator
diagnostic, not proof of the cause of an entire optimization trajectory.
Run only after the research queue and audits, with a 90-second process timeout
and at least 120 seconds remaining before the compute cutoff; record omission
or failure. Its results cannot alter research fitting or configuration decisions.

Predeclare one optional acquisition-scale restart-prefix comparison after the
12-update public-CLI software case and its initial report. Run only if that case
passes, its first process completed four uninterrupted updates, and at least
1,020 compute seconds remain before 13:18 UTC. Reuse its exact analytic
observations, zero-seeded 8,192-component profile, source, device/storage and
one-shot accumulation. In fresh outputs, execute two updates, resume for two
more, and compare the update-four state with the already recorded continuous
four-update prefix. The extra work is 120 forward and 80 adjoint shot solves;
the reference prefix's work is already counted in the original software case.
The unchanged 12-update optimization horizon crosses a frequency boundary at
update three. Both new segments remain explicitly paused software outputs.

Require field/Adam/controller/selection/history/specification state and initial
cached predictions to match at the existing CUDA restart tolerances (elementwise
1e-8 relative / 1e-10 absolute); integer identity and work state must match
exactly. Compare decoded full-grid velocities, validate each checkpoint's own
field fingerprints and verify candidate RNG continuity across its restart.
Exclude elapsed wall time and cross-run derived field hashes from numerical
comparison. Independent fresh CLI starts have unrelated global RNG states;
record their equality separately, while the controller seed and fixed-population
policy remain identical. No tolerance may be relaxed. Retain both segments and
any failed comparison. Use a 900-second total attempt limit and per-process
780/900-second soft/hard limits, the existing GPU margins and empty scratch
checks. This accepts only that four-update restart prefix, not an uninterrupted
12-update trajectory or geological convergence. Its result cannot change the
research configuration or selection decisions.

The optional 12-update CLI fixture rejected its profile before propagation:
`minimum_age=0` violates the existing positive-integer constraint. Preserve that
attempt and its zero-solve configuration failure. Bounded fixture correction:
use `minimum_age=1` in a newly named profile and attempt, retaining fixed
minimum/maximum population 8,192, all other settings, source, solve expectations,
the 3,000-second launch condition and 900/2,700-second limits. This corrects an
invalid software-fixture input; it does not change the production method or
research fits. Require the same independent field/state/resource acceptance.
The original large restart-prefix probe is omitted because that rejected CLI
attempt has no continuous prefix. If the corrected CLI case passes, declare a
fresh prefix probe using its exact corrected profile and four-update reference.
Retain the original 1,020-second launch condition, 900-second total limit,
120/80 additional solve expectation and numerical/state criteria. Run it only
after a new initial report includes the corrected CLI result. Preserve the
original omission and every corrected-attempt record separately.

The running CLI fixture's queried launch-process counters reported zero write
bytes while checkpoint files exceeded 600 MB. Declare a separate CPU-only
instrumentation probe: write and flush exactly 1 MiB in a fresh worker, record
its own before/after counters and PID, and compare with the parent's queried
`Popen` handle/PID. Preserve the payload and both measurements. This diagnoses
counter scope, uses no acoustic solves and changes no fitting or acceptance
criterion. Do not treat incomplete launch-process counters as worker disk cost.

Bounded post-campaign memory correction (2026-09-16 12:45 UTC): the valid
large CLI fixture completed four updates (120 forwards, 80 adjoints), then
its next process failed before propagation while constructing filtered targets.
Compact each detached constant target at construction rather than waiting for
the first accumulated gradient. Preserve its values, objective, preprocessing,
precision, execution policy and all research decisions. Independently require
bitwise target equality to the original filter and storage equal to logical
tensor bytes immediately after construction; rerun the existing gradient,
unequal-batch, Adam, isolation and restart checks, CUDA disk accumulation check,
and a fresh complete release gate. Preserve the original research source and
71-test release; scientific fits and audits refer to that original source.

If focused checks pass, continue the saved four-update large checkpoint into
fresh outputs using the corrected memory allocation. This is an explicit
source-version transition in a software fixture, not a same-source research
trajectory. Retain the failed process and original 12-update horizon, profile,
memory and oracle criteria, exact cumulative 360/240 fitting and 20/0 verification
work, 780/900-second process limits, and the original attempt deadline
13:12:57 UTC (2,700 seconds from its launch). Count the original four updates
once. Require an exact source diff limited to the declared constant compaction
and its regression check. The optional large same-source prefix comparison is
omitted if its unchanged source/prerequisite/time conditions are not met.

The constructor correction passed six focused checks and the unchanged CUDA
disk-accumulation state/restart/isolation gate. A fresh large continuation passed
initialization but failed during its first adjoint allocation. The error reports
3.05 GiB live allocation, a 1.72 GiB request and 1.07 GiB reserved but unused
under the unchanged 5.60 GiB cap. This suggests allocator fragmentation; it does
not prove the cause. Predeclare one further bounded runtime change: release
unused CUDA allocator blocks once after all initialization, only for accumulated
execution. This cannot release live tensors or alter their values. Preserve CPU
and full-graph execution, solve counting and all numerical/resource limits.
Require the existing numerical/restart checks, a fresh CUDA disk gate and full
release, plus a separately named continuation attempt before the original
13:12:57 deadline. Preserve the failed adjoint as incomplete work, not a completed
update. Stop the optional large workload if this final memory attempt fails.

### Completed overnight development evidence, 16 September 2026

All five declared additional research jobs completed before the 12:24:19 UTC
decision freeze. Their 40,000 optimizer updates used 120,090 fitting forwards
and 120,000 adjoints, with 30 additional runner/audit verification forwards.
The largest actual research process segment was 728.97 seconds. All independent
post-fit audits passed, including field/covariance decoding, waveform scores,
selected-step eligibility, source/input identity and physical work accounting.
Maximum independent sampled field error was 6.37e-12 m/s; maximum relative
waveform-score error was 2.32e-13. These are software agreement measurements,
not geological accuracy. Each original fit, failed attempt and source archive
is retained under `results/overnight_20260916T054800Z/`.

| New development fit | Updates | Runner min | Final K | Selected validation | Evaluation-only velocity RMSE, m/s |
|---|---:|---:|---:|---:|---:|
| Marmousi P0 | 8,000 | 66.27 | 944 | 4.8782153e-5 | 313.89 |
| Overthrust P0 | 4,000 | 36.24 | 935 | 9.6912867e-5 | 208.57 |
| Marmousi fixed population | 4,000 | 32.08 | 1,024 | 5.8059396e-5 | 284.36 |
| Overthrust P0 | 8,000 | 73.66 | 943 | 6.1926625e-5 | 201.83 |
| Marmousi P0 | 16,000 | 133.76 | 954 | 4.4706630e-5 | 334.80 |

Reference velocities and test scores were read for evaluation only after the
research decisions froze. These 70x70 rescaled arrays and their partitions were
already seen in development and are not blind or original-scale benchmarks.
The archived earlier Marmousi P0 4,000-update fit had velocity RMSE 283.65 m/s.
Extending to 8,000 and 16,000 improved validation and test waveform agreement
while worsening that velocity metric. Overthrust 8,000 versus 4,000 improved
validation by 36.10% and velocity RMSE by 3.23%, at 2.03 times runner cost.
The fixed-population Marmousi comparison uses the earlier P0 source and retains
that historical-source confound. No reference/test metric promoted a new profile.

Every new run retains at least one declared unfinished-optimization indicator.
During settling, objective increases exceeding 10% occurred at 29/6,000 adjacent
pairs in Marmousi 8,000; 0/2,000 in Overthrust 4,000; 44/2,000 in fixed Marmousi
4,000; 32/6,000 in Overthrust 8,000; and 192/14,000 in Marmousi 16,000. The largest
adjacent ratio was 5.74 in Marmousi 16,000. These events occurred after population
changes stopped; their cause is not established. No learning-rate remedy was
tuned during this campaign. All accepted new population edits were prunes
(80, 89, 0, 81 and 70 respectively); there is no scientific clone/split efficacy
evidence here. On this grid the clone cutoff is below the sampling width floor,
and Gaussian parameter counts exceed the 4,900 sampled grid values.

The same-source Marmousi 8,000/16,000 common-prefix diagnostic failed strict
parameter/state tolerances at 11 of 20 boundaries through update 2,000. The
maximum decoded field difference was 6.302 m/s. This supplements, rather than
replaces, the failed historical shear comparison. Six repeated gradients per
execution mode at an identical field passed the original within-/cross-mode
numerical criteria but were not bitwise identical; field pullbacks using one
fixed velocity gradient were bitwise equal. The diagnostic cost was 36 forwards
and 36 adjoints, with no optimizer updates. It does not establish a causal
explanation for long-trajectory drift. Deterministic flags are not a proof of
bitwise repeatability; repeated identical runs remain a necessary next control.

The final setup-memory source passed 72 source and 72 isolated-wheel tests,
Ruff and repository checks in a fresh release directory:
`results/validation/overnight_setup_20260916T125200Z/`.
The new regression requires bitwise filtered-target equality and exact logical
storage size immediately after construction. Both new CUDA disk-accumulation
gates passed the unchanged state/restart/isolation/rollback criteria, adding
264 forwards and 168 adjoints. The research itself retains its earlier 71-test
source and completed audits. `setup_source.diff` identifies the two memory
lifetime changes and one regression test; no optimizer, physics, objective,
profile, precision or acceptance threshold was changed. Final large-fixture
acceptance and its exact work are recorded separately below. Uncompleted
adjoint work remains separate from completed solves.

The known-write instrumentation probe confirmed that virtual-environment
launcher counters omitted the actual worker's 1 MiB write. Saved launcher
counters therefore do not establish wavefield disk traffic or SSD write cost.
Future resource acceptance must instrument and verify the actual worker scope.
The failed constructor-only large continuation left one 5,338,934,400-byte
temporary wavefield file. It is retained as failed-attempt scratch outside the
artifact seal; successful empty-scratch checks do not establish exceptional-path
cleanup. Include failed-adjoint disk lifecycle in the next resource study.
Next scientific work requires physical benchmark provenance, an independently
accepted external grid-FWI comparator, complete geometry/population/capacity
controls, convergence and stability studies, and a frozen independent
noise/coverage/structure matrix. Passing the release gate is not a SOTA result.

The final bounded large recovery fixture passed by 13:09 UTC, before its
original 13:12:57 deadline. Twelve updates and final selection/replay completed
across four successful CLI processes: the original four-update prefix, two
four-update continuations, and a final selection/export/replay process. Actual
fitting work was exactly 360 forwards and 240 adjoints; verification added
20 forwards. The original prefix is counted once. The attempt clock including
debugging/idle time was 2,425.01 seconds; successful process time totaled
1,508.88 seconds. All 8,192 components remained; parameters and grid values
were finite, full covariance passed, and 64 predetermined NumPy sample points
differed by at most 2.27e-12 m/s. Maximum reserved memory across successful
segments was 5.39 GiB; the smallest recorded driver-free memory after a segment
was 1.39 GiB. These are segment observations, not continuous extrema.

The first four updates use the archived research source; later processes use
the declared setup-memory correction. This is accepted source-transition
recovery at one analytic workload, not an unchanged-source trajectory comparison,
a fragmented-field sustained run, or geological convergence. The optional
unchanged-source large restart-prefix probe was omitted because its source
prerequisite no longer held. Its tolerance and prerequisite were not relaxed.
The separate earlier one-update fragmented-population stress result remains
distinct. Including both later CUDA gates, the repeatability probe and this
complete recovery case, measured completed GPU engineering work totals 1,575
forwards and 1,124 adjoints. Failed/incomplete attempts and CPU software checks
remain separate, with raw records preserved.

## Representation-focused foundation, 16 September 2026

Bounded change declared before the new demonstration: revise the active research
direction and evaluation protocol around explicit geometry, spatial scale and
application capabilities. Keep the single numerical method, public API, shipped
profile, physical solver, optimizer and data-separation rules unchanged. Preserve
the sealed overnight study. An external grid-FWI comparison becomes relevant
when a particular claim needs it, rather than a universal prerequisite.

The owner selected macro models with local detail as the first demonstration.
Create a separate constructed-field software diagnostic in a fresh ignored
output directory. Use the existing float64 CPU GaussianField, a 10 m 81x161
grid, four prescribed broad signed kernels and three prescribed localized fine
kernels with full SPD covariance. All inserted amplitudes start at zero; then
assign explicitly documented demonstration parameters. Hold the background and
broad parameters fixed and vary the fine amplitudes by lambda in [0,1]. This is
neither a learned reconstruction nor a supervised or waveform inversion fit.
It takes zero optimizer updates and zero acoustic forward/adjoint solves.

Define v_macro=B(u_macro), v_lambda=B(u_macro+lambda*u_fine), and the detail
contribution as v_lambda-v_macro. Because B is nonlinear, the separate raw
contributions must be combined before bounding. Also show a separately labeled
80 m spatially smoothed view with a declared boundary/truncation rule; broad
kernel grouping is not guaranteed to equal a low-pass decomposition. No view
is labeled seismic resolution, missing information or posterior uncertainty.

Independent acceptance: compare point values with a NumPy covariance-solve
oracle at 128 predetermined off-grid points (rtol=1e-10, atol=1e-8 m/s); verify
full-population covariance eigenvalues and finite physical bounds; require zero
detail at lambda=0 and sampled locality outside all fine supports (atol=1e-8
m/s); verify fixed broad/background state and unchanged field parameters when
querying 20/10/5 m grids at shared physical nodes. Compare a fixed weighted
point functional's lambda derivative from autograd, independent analytic
differentiation and central differences (rtol=1e-6, atol=1e-6). Require monotone
pointwise variation between endpoints for this one fixed amplitude direction;
this makes no claim about arbitrary optimization paths. Record actual display
quantization error separately from float64 numerical acceptance.

Deliver a standalone interactive HTML view, PNG/PDF figure, exact parameters,
arrays, source/input hashes and a diagnostic report. Run repository/link checks
and the required fresh source/wheel release gate. Independently verify numerical
source and baseline hashes against the prior accepted release to establish that
this framing and demonstration change did not modify the inversion algorithm.

The constructed diagnostic completed in
`results/representation_foundation_20260916T141100Z/constructed_001/`. Its seven
prescribed kernels contain 44 field scalars. All declared numerical checks
passed: the largest error at the 128 independent off-grid points across lambda
0, 0.5 and 1 was 9.095e-13 m/s. The weighted lambda derivative was
2.1977667152953564 by autograd, 2.197766715295358 analytically and
2.1977667142891733 by central difference. All 10,186 sampled points outside the
three local supports stayed unchanged across 41 activation states. Shared nodes
on 20/10/5 m grids and saved-field reload were exact; frozen state, SPD, physical
width limits, bounds and the declared monotone path passed. Numerical source,
baseline and verification-input hashes matched the prior accepted release.

The 80 m spatially smoothed total changed by at most 37.739 m/s (RMS 6.675 m/s)
while the broad parameters stayed fixed. This is measured scale coupling in a
constructed example, not a reconstruction error. The standalone interactive
view contains 41 float32 display states; the maximum display quantization error
is below 0.000123 m/s and is separate from float64 numerical acceptance. Exact
parameters, arrays, a portable field, PNG/PDF figure and scripts are retained.
The scientific PNG was inspected. Page data and controls are checked by executing
the actual JavaScript in Node with a minimal DOM; this is not a browser layout or
accessibility audit. The first view-check attempt incorrectly required bitwise
equality between NumPy linspace and JavaScript division; that check was corrected
for binary rounding without changing field data or numerical acceptance limits.

A fresh full release passed Ruff, repository/link checks, 72 source tests and
the same 72 isolated-wheel tests in
`results/validation/representation_foundation_20260916T141100Z/`. Only active
documentation changed; the method, shipped profile and observation-only runner
are unchanged. The demonstration has zero optimizer updates and zero acoustic
forward/adjoint solves. It provides neither learned recovery nor adaptive
allocation, data resolution or uncertainty evidence. Next is the explicitly
separated supervised macro/detail study, followed by the waveform-only study
under the existing budgets and frozen data rules. Geological priors and measured
alternative-model proposals remain future algorithmic work.

## Baseline closeout, 16 September 2026

The owner requested a finite end to foundation development and an actual paper.
The active scope is now frozen by [BASELINE_CLOSEOUT.md](BASELINE_CLOSEOUT.md).
Geological priors, new proposal algorithms, new scale constraints and additional
public engines are outside this closeout. Existing accepted evidence is reused.

Bounded engineering change declared before implementation: give the inversion
invocation explicit ownership of its newly created disk-wavefield directory,
and remove only its recognized Deepwave temporary subdirectories when the
invocation exits, including a caught forward/backward failure. Validate resolved
absolute containment and directory identity before recursive cleanup; preserve
unrecognized entries and unrelated files. A cleanup failure must remain visible
without replacing an original inversion error. No arithmetic, objective,
optimizer, density rule, profile, public API or checkpoint format changes.

Independent acceptance: retain a real disk-backed acoustic autograd graph,
inject a failure before its adjoint completes, and check that scratch is cleared
while the original exception, completed-update checkpoint and unrelated sentinel
files survive. Check successful pause/restart identity and acoustic work using
the existing suite. Verify unchanged arithmetic by comparing the normalized AST
of the original inversion body after removing only the new resource context.
Run focused lifecycle checks and a fresh complete source/isolated-wheel gate.

A separate, explicitly velocity-supervised representation study will use the
existing GaussianField, canonical Adam groups and existing density controller.
Its advance declaration is stored in a fresh ignored closeout directory. It
uses analytic curved-layer and faulted-layer fields not constructed from the
Gaussian decoder, fixed geometry/fixed population/fully adaptive controls, three
initialization seeds, and the protocol's 5,000/10,000/20,000-update allowances.
Training samples drive fitting; held-out physical queries are evaluated after
selection. Macro and detail are defined by an explicit 80 m spatial operator.
This study adds no acoustic solves and is not labeled FWI. Existing audited
observation-only fits supply the separate acoustic application evidence.

The closeout delivers one frozen source/wheel record, the numerical and resource
acceptance matrix, the learned representation results and a working manuscript.
Negative scientific outcomes are recorded and restrict claims; they do not
automatically expand this stage into another algorithm-development programme.

The resource-lifetime change passed four new lifecycle tests and the complete
fresh release in `results/validation/baseline_closeout_20260916T155000Z/`:
76 source tests and 76 isolated-wheel tests, Ruff and repository/link checks.
`compatibility.json` in the closeout record confirms exact normalized-AST
identity of `invert` after removing only scratch creation/context. The baseline
JSON hash remains unchanged. The runner adds only error-note metadata to its
failure record; no checkpoint or observation schema changes.

CUDA float64 topology, rollback, completed-stage/update restart, test isolation
and replay passed with one-shot accumulation and uncompressed disk storage in
`cuda_runtime_001/`. A separate retained-graph CUDA failure probe passed with
both full-graph and accumulated execution in `cuda_failure_003/`: respectively
3,136,000 and 1,254,400 bytes of live scratch were removed after injected failure,
the original exception and prefix checkpoint survived, and recovery completed
all 16 fixture updates. The first two probe attempts reached recovery but used
an incorrect reporting assertion (first the expected horizon, then treating
the update-count dictionary as a scalar). Their outputs and scripts remain
preserved; correcting this external assertion did not change package source.

The accepted source supports ordinary exception cleanup, not forced termination
or deletion of operating-system-locked files. Unrecognized scratch entries and
replaced directories are preserved with visible cleanup errors. The historical
strict long-prefix CUDA comparison failures remain failed; their tolerances and
research artifacts are unchanged.

The working paper is now [MANUSCRIPT.md](MANUSCRIPT.md). A declared post-hoc
audit of all five frozen acoustic outputs independently recomputed their full
velocity RMSE and verified consumed artifact/reference hashes. With the same
80 m reflected Gaussian smoothing operator, selected macro RMSE was 95.03--115.05
m/s for Marmousi versus 327.60 initially, and 32.56--35.15 m/s for Overthrust
versus 1099.18 initially. Detail errors also fell relative to initialization.
These are descriptive output metrics with zero new optimization or acoustic
solves, not topology benefit or seismic-resolution estimates. The declaration,
metrics and figures reside in `acoustic_output_audit_001/` in the closeout record.

The first redirected Windows shell commands for release and CUDA acceptance
returned wrapper status 1 despite successful checker artifacts and complete
test logs; native stderr warnings appeared as PowerShell error records. A
fresh full release was therefore run through direct Python subprocess capture
into `results/validation/baseline_closeout_20260916T155000Z_002/`. It again
passed 76 source and 76 isolated-wheel tests, Ruff and repository/link checks,
and its explicitly captured native return code was zero. No implementation or
test tolerance changed. `release_execution.json` preserves this final status.

The supervised execution schedule was amended prospectively after measuring
the first fit's cost. The original second fit completed before its identified
launcher/worker were stopped. Its next-run prefix, containing initial/config
files and no 100-update log row, is preserved and unused. The remaining 16 fits
call the exact original frozen fitting function in three CPU workers with two
threads each and fresh output directories. No targets, seeds, objectives,
topology rules, budgets or selection criteria changed. Timings are shared-host
observations, not isolated throughput comparisons. The execution amendment,
process-identity checks and per-run paths are preserved in the closeout record.

The owner subsequently requested a pause to take the laptop to class. At the
pause, 13 of 18 supervised fits were complete (220,000 updates); three active
fits had independently loadable recovery checkpoints at 16,000, 11,000 and
17,000 updates. All five identified study processes were suspended with exact
in-memory state retained, and their thread states were checked. The separate
keep-awake watcher was stopped. No numerical source, profile, fitting rule or
completed result changed. Held-out evaluation has not started.

`PAUSED_HANDOFF.md`, `travel_pause_001.json` and
`travel_checkpoint_audit_001.json` in the closeout record identify the states
and remaining work. A resume helper validates creation times, executable paths
and command lines before releasing suspension; its check-only path is separate
from resumption. Sleep/hibernation retains these processes. If they no longer
exist, disk recovery must use fresh output directories. Raw elapsed times of
interrupted fits include the pause and must be labeled accordingly. Engineering
acceptance remains complete; the supervised result section, final manuscript
render and sealed handoff await owner-requested continuation.

The owner authorized continuation. All five paused process identities matched,
and the existing processes were resumed without restarting or rolling back a
fit. `travel_resume_001.json` records the 350.964-second suspension and the new
hidden evaluation/keep-awake watcher. The raw elapsed times of the three active
fits include this interruption. Completed snapshots and the pause audit remain
unchanged; the numerical source, profile and declared study rules are unchanged.

All 18 declared supervised fits subsequently completed, totaling 320,000
optimizer updates and zero acoustic forward/adjoint solves. Held-out evaluation
started only after the final fitting index existed. The independent audit in
`supervised_evaluation_001/audit.json` passed paired initialization equality,
selection/continuation reconstruction, physical invariants, fixed controls,
event guards and saved-field replay. The maximum independent NumPy decoding
error was 2.274e-12 m/s. The completed fitting index has SHA-256
`84c69954c1a9b5d915a3134d3d6dd311c415a7eac5914a185c7c5cd12c7f893c`.

At K=24, geometry learning reduced three-seed mean off-grid RMSE from 20.10 to
2.19 m/s on the curved target and from 60.06 to 2.24 m/s on the faulted target.
Mean spatial macro RMSE fell from 2.88 to 0.10 and 13.99 to 0.24 m/s;
local-detail RMSE fell from 37.17 to 5.23 and 48.28 to 2.82 m/s. The comparison
enables 146 optimized variables instead of 26, and uses the declared conditional
budgets rather than matching execution time. Learned fields contain both broad
and narrow signed contributions. These results establish useful representation
of two independently defined analytic targets, not comparative acoustic speed,
blind geological generalization or seismic-resolution calibration.

The population controller accepted one split and no clones or prunes. At update
2,100 of curved seed 0, K changed from 24 to 25 with a cumulative sampled
velocity change of 22.913 m/s. Its selected off-grid RMSE was 1.28 m/s versus
2.42 for the paired fixed-population run at the same 20,000-update allowance.
The other five adaptive/fixed-population pairs have bitwise identical selected
grid fields and query values. Thus replicated geometry benefit and one observed
growth benefit are reported separately. All selected analytic fields store
146--152 scalars and 4,121--5,129 serialized bytes. Fourteen runs reached 20,000
updates and twelve still met the continuation indicator at their final budget.
Full curves, selected steps, per-seed errors and raw shared-host durations are
preserved; global convergence and isolated timing advantages are not claimed.

A reporting-only correction was declared before held-out evaluation completed:
the original image extent treated physical node centers as pixel edges. The
version-two figure renderer reads the same audited arrays and selected geometry,
uses half-cell image boundaries and the original physical domain limits, and
checks exact pixel-center/axis alignment independently. Its audit passed in
`supervised_figures_002/`; the original renderer/figures remain preserved.
The correction adds no fitting, queries, acoustic solves or changed metrics.

The completed [manuscript](MANUSCRIPT.md) reports all representation controls,
every run's work, the constructed local-control result and the five previously
sealed acoustic fits. Its review PDF, embedded-figure HTML, exact editable source
and standalone scientific figures are in `paper_002/` in the closeout record.
The earlier complete render remains in `paper_001/`; the final rendering uses
vector PDF equations at text scale, replacing oversized raster placement.
This presentation correction changes no formulas, parameters or results.
The 20-page final render passed source-copy, result/cost-table transcription,
figure-copy and embedded-asset identity checks in `manuscript_review.json`.
All pages have previews and vector-text page-bound checks; representative
equation, table and scientific-figure pages were visually reviewed. HTML checks
cover embedded asset identities, not browser layout or accessibility.
The final handoff contains the accepted wheel, a clean source/documentation ZIP,
source identities, reproduction instructions, a scoped work ledger and a hash
manifest for the complete record. Numerical source and verification-input
hashes match the final native-zero 76-source/76-wheel release; only documentation
and report assembly followed that acceptance. Repository/link checks are rerun
at sealing. Old fits, failed assertions and earlier sealed records are retained.

The finite foundation stage is complete. Subsequent applications use this
accepted implementation and independently declare any new objective, data
access or acceptance criterion. Geological priors, measured model proposals,
macro-preservation constraints and information diagnostics remain future work.
Author/affiliation details, public archival identifiers, owner-selected licensing
and venue formatting remain paper/public-release metadata. They do not reopen
the completed baseline study or justify another unbounded test campaign.
