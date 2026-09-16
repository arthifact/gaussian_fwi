# Learning the Spatial Complexity of Velocity Models: Adaptive Gaussian Representations for Full-Waveform Inversion

Working manuscript, 16 September 2026. Numerical results distinguish constructed
fields, known-velocity supervision and observation-only acoustic inversion.
The frozen implementation and evidence records are identified below.

## Abstract

A velocity model is both a physical field and a choice of spatial degrees of
freedom. We develop an explicit Gaussian representation that learns velocity
amplitudes together with the positions, orientations and extents of their
spatial support. A bounded population controller complements continuous
parameter learning with discrete clone, split and prune operations. The result
is one differentiable physical field that can be queried at arbitrary
coordinates, inspected component by component, and used by an acoustic
full-waveform inversion operator. We formulate signed velocity contributions,
full positive-definite covariances, physical sampling limits and consistent
optimizer-state transitions. Macro models and local detail provide the first
application: a constructed example demonstrates continuous local control and
quantifies its coupling to a spatially smoothed field; a separately declared
supervised study isolates geometry learning and population adaptation on curved
and faulted velocity structures. Across three paired initializations, learning
geometry at 24 components reduces mean off-grid velocity RMSE from 20.10 to
2.19 m/s on the curved target and from 60.06 to 2.24 m/s on the faulted target,
with simultaneous reductions in macro and local-detail error under the declared
budgets. One accepted split further improves a paired curved-model fit; the
remaining five adaptive-population pairs retain the fixed-population result.
The learned fields store 146--152 scalar parameters. Five archived acoustic development fits provide
40,000 waveform-driven updates with independent field and waveform replay.
These acoustic fits recover useful velocity structure, while also showing that
lower waveform error can coexist with higher velocity error. The contribution
is a reusable, physically specified representation and inversion foundation
whose spatial organization is an explicit optimization variable.

## 1. Introduction

Subsurface velocity models serve several purposes. A smooth model may initialize
waveform inversion or support migration; localized detail may describe a thin
layer, a fault zone or a reservoir-scale contrast. These uses place different
demands on spatial parameterization. Representing every part of a model at the
same nominal scale is a convenient numerical choice, but the physical field
need not inherit that choice permanently.

We ask whether velocity and the spatial organization of its representation can
be learned together in a form that remains directly inspectable. Our model
contains signed, anisotropic Gaussian velocity contributions. Each contribution
has a position, a physical extent, an orientation encoded by full covariance,
and an amplitude. Broad and narrow contributions coexist in the same field.
The wave equation consumes samples of that field; the stored representation
retains its continuous physical coordinates.

Adaptation has two distinct meanings here. Continuous adaptation moves and
reshapes existing support using the training-objective gradient. Discrete
adaptation changes the number of components under an explicit field-change
budget. The purpose is to make the distribution of representational degrees of
freedom responsive to the fitting problem. A high center-gradient score is a
proposal heuristic; it is not a calibrated estimate of geological complexity,
information content or uncertainty.

The first application is a macro model with local detail. This is useful both
as a concrete output and as a demanding test of the representation: broad
structure and localized contrasts must coexist, and their interaction must be
measured. We therefore define a spatial smoothing operator independently of
component width. This makes the macro/detail claim operational and exposes
scale coupling rather than hiding it in a visual decomposition.

Our contributions are a signed full-covariance physical field with checked
first derivatives; one waveform-driven algorithm for geometry and bounded
population adaptation; and an evidence framework connecting local control,
supervised approximation and observation-only inversion. The software exposes
the same field and saved-state semantics to subsequent applications. Geological
priors, alternative-model proposals and calibrated resolution diagnostics can
then be investigated as identifiable extensions of this foundation.

## 2. Relationship to existing representations

Radial-basis velocity models and adaptive parameterization predate this work.
Peters and colleagues describe adaptive meshless acoustic waveform inversion
using compact radial basis functions. Kadu, van Leeuwen and Mulder use radial
basis functions to parameterize a level-set boundary for salt reconstruction.
These are direct precedents for moving beyond a fixed spatial parameterization
in seismic inverse problems.
[Peters et al. (2017)](https://doi.org/10.1016/j.enganabound.2017.07.006);
[Kadu et al. (2017)](https://arxiv.org/abs/1610.00251).

Gaussian bases also occur in seismic inversion. Izadian's time-lapse impedance
inversion uses a Gaussian representation whose centers, widths and number are
set before optimizing its coefficients. That convolutional inverse problem
differs from propagating acoustic waves through a spatial velocity field, but
it is a relevant precedent for Gaussian parameterization. Our proposed
distinction is joint learning of physical geometry and population in the
acoustic velocity representation.
[Izadian (2026)](https://link.springer.com/article/10.1007/s11600-026-01856-x).

Continuous neural fields provide another established route. Implicit seismic
FWI learns a coordinate-to-velocity map, and adaptive Fourier-basis learning
extends its control of spatial frequencies. Our field makes the location,
shape and signed contribution of each spatial element explicit. This is a
different interface for interpretation and model operations, whose usefulness
must be assessed for the intended application.
[Sun et al. (2023)](https://doi.org/10.1029/2022JB025964);
[Kang et al. (2026)](https://doi.org/10.1093/gji/ggaf404).

The interleaving of Gaussian parameter optimization and density control follows
3D Gaussian splatting. Its image-space gradients and optical opacity cannot be
transferred literally to velocity inversion: our gradients use physical
coordinates, amplitudes are signed velocities, and every density event obeys
a cumulative sampled velocity-change limit. Acoustic propagation remains the
forward operator. The rendering speed of Gaussian splatting does not transfer
to the wave equation.
[Kerbl et al. (2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

The contribution is consequently the specified combination of explicit spatial
velocity geometry, signed full-covariance components, bounded population edits
and acoustic optimization, together with demonstrated model operations. We do
not claim that Gaussian bases, continuous velocity values or adaptive inversion
were introduced here.

## 3. Representation and inversion algorithm

### 3.1. A bounded physical field

For physical coordinates p=(x,z), or (x,y,z), define

\[
u_\theta(p)=b_0+(b_1-b_0)z/H+\sum_{k=1}^{K}a_k\kappa(q_k(p)),
q_k=(p-\mu_k)^\top\Sigma_k^{-1}(p-\mu_k),
v_\theta=B(u_\theta).
\]

The background endpoints and signed amplitudes have units m/s, centers have
units m and covariances have units m squared. Tensor axes are (z,x) or (z,y,x).
Let L=diag(exp(ell))(I+S), where S is strictly lower triangular, and set
Sigma=LL^T. The positive-diagonal Cholesky factor covers every full SPD
covariance without redundant rotation parameters. Kernel anisotropy concerns
spatial shape; the propagation model remains scalar acoustic.

The implemented kernel is exp(-q/2) through q=25, smoothly tapered to zero
at q=36 by w(t)=1-10t^3+15t^4-6t^5, t=clip((q-25)/11,0,1).
This compact kernel is C2, allowing supported query/component pairs to be
evaluated sparsely. The velocity bound is

\[
B(u)=v_{\min}+s\log(1+e^{(u-v_{\min})/s})
-s\log(1+e^{(u-v_{\max})/s}).
\]

For finite valid parameters, B is monotone and 1-Lipschitz with range between
the prescribed velocity bounds. Principal standard deviations are projected
to a declared physical interval. The numerical sampling floor uses a specified
Gaussian Nyquist-response criterion; it does not certify the spectral content
of an entire overlapping field. The [mathematical specification](METHODS.md)
provides derivatives, proofs and the exact projection convention.

The field contains K[d+d(d+1)/2+1]+2 stored scalar parameters: 6K+2 in 2D
and 10K+2 in 3D. Query grids, optimizer moments, file metadata and acoustic
wavefields require additional storage. Zero-amplitude initialization preserves
the background. Amplitude gradients can activate these seeds before their
geometry gradients become nonzero.

### 3.2. Training and population adaptation

For training receiver data d_T and the discrete acoustic operator A_h, optimize

\[
J(\theta)=\ell_T(A_h(v_\theta),d_T)+R(v_\theta)+R_{raw}(u_\theta).
\]

Let r_i denote the derivative of this discrete objective with respect to the
raw field value u_i at physical sample p_i. It includes the waveform gradient
pulled through B and the declared regularizers. For a fixed population,

\[
\frac{\partial J}{\partial a_k}=\sum_i r_i\kappa(q_{ki}),
\]

\[
\nabla_{\mu_k}J=-2a_k\sum_i r_i\kappa'(q_{ki})\Sigma_k^{-1}(p_i-\mu_k).
\]

Covariance parameters receive the corresponding chain-rule derivatives through
L. Thus the same objective sensitivity changes both velocity contrast and the
spatial support available to explain the data. Geometry is adaptive to the
training objective; it is not assigned from the reference velocity. Computing
these parameter derivatives uses the same acoustic adjoint results. At zero
amplitude the geometry derivatives vanish, while amplitude derivatives can
activate the seeded components.

The shipped profile uses cumulative frequency bands through 4, 7, 12 and 20 Hz,
fixed training-derived preprocessing, normalized waveform residuals, total
variation and a raw-velocity bound penalty. Adam updates the background,
amplitudes, centers and covariance parameters, followed by geometric projection.
One update includes every training shot. Shot accumulation sums the complete
training gradient with the original survey normalization and adds regularization
once; it does not change the objective in exact arithmetic.

During the refinement window, the controller records each component's mean
physical center-gradient norm. Eligible weak amplitudes are proposed for
pruning. Remaining candidates are ranked by this gradient statistic, then
cloned or split according to their largest principal width relative to the
domain extent. A split samples two child centers from the parent covariance
and reduces their widths by a factor of 1.6. Children inherit the parent's
signed amplitude and receive fresh Adam state. Surviving parameters retain
their optimizer history.

These edits generally change the field. An event is accepted only when its
combined maximum absolute velocity change on the declared sampling grid is at
most 25 m/s. A rejected batch is reduced in its deterministic priority order;
each attempted transaction either commits fully or restores the previous
field, parameter identities, gradients, topology and optimizer state. This
sampled guard is a continuity budget, not an objective-descent test or an
all-coordinate bound. The proposals incur zero additional acoustic solves.

Population changes stop before the settling window. Validation waveform loss
selects only among eligible fixed-population states. Test waveforms and true
velocity play no role in FWI optimization, topology ranking or selection.
Exact scheduling and event semantics are in the [algorithm](ALGORITHM.md).

### 3.3. Macro models and local detail

We distinguish two operations. A component operation holds a selected group
fixed while scaling another group's signed amplitudes by lambda in [0,1]:

\[
v_\lambda=B(u_{broad}+\lambda u_{local}).
\]

This defines a continuous, differentiable local modification for fixed geometry.
Its physical detail contribution is v_lambda-v_0, not B(lambda*u_local),
because the bounding map acts on the combined raw field.

A spatial operation defines M=S_80[v], where S_80 is Gaussian smoothing with
80 m standard deviation, reflected boundaries and four-standard-deviation
truncation. Detail is D=v-M. Broad component membership does not imply exact
spectral separation: changing a localized component can change M. We report
this coupling explicitly. Both operations consume the same physical field;
neither requires another inversion engine.

### 3.4. Implementation and reuse

One Python package provides the field, first-order sparse decoder, controller
and inversion loop. Acoustic propagation and adjoints use Deepwave through
PyTorch; our implementation supplies the physical field, acquisition operators,
objective and stateful adaptation around that propagator.
[Deepwave documentation](https://www.ausargeo.com/deepwave/).
A single observation-only runner loads acquisition data;
reference velocity is excluded from its input schema. Saved fields preserve
physical units, geometry, topology and numerical policy. Completed-update
checkpoints additionally preserve optimizer, controller, selection and work
state. New output directories protect previous fits.
Software checks include two- and three-dimensional fixtures; the scientific
experiments reported in this paper are two-dimensional.

Uncompressed device, host and disk wavefield storage expose a resource choice
without changing propagation precision or temporal sampling. The inversion
owns its newly created disk scratch through completion, pause or ordinary
exception handling. Resource cleanup preserves checkpoints and unrelated
files. Memory support is established on measured workloads, independently of
the number of Gaussian parameters.

## 4. Experiments

### 4.1. Constructed local-control diagnostic

On an 81 by 161 grid at 10 m spacing, four prescribed broad components and
three prescribed local components define a 44-scalar field. All insertion
amplitudes are initially zero; demonstration parameters are then assigned
explicitly. We evaluate 41 local-amplitude states from lambda=0 to 1, hold
background/broad parameters fixed, and inspect the 80 m spatial macro operator.
This experiment performs no learning and no acoustic solves.

### 4.2. Supervised representation study

The separate velocity-supervised experiment fits two analytic 1280 by 640 m
models: curved layers with a compact oscillatory contrast, and a faulted layer
with localized positive and negative contrasts. Tanh interfaces and compact
cosine windows define the targets independently of the Gaussian decoder.
Training uses all 65 by 129 grid samples at 10 m spacing.
Appendix A gives the analytic fields and complete fitting settings.

For seeds 0, 1 and 2, paired initializations contain 24 zero-amplitude components
on a jittered 4 by 6 lattice. The controls optimize (i) background/amplitudes
with fixed geometry, (ii) all parameters with fixed population, and (iii) all
parameters with adaptive population. The adaptive ceiling is 144 components;
the lower bound is 12. These are recorded research settings. The production
baseline profile remains unchanged.

The objective is mean squared velocity error divided by 100 squared, without
a waveform term or additional regularizer. Every fit receives 5,000 updates,
with declared continuation to 10,000 and 20,000 according to training trends
and selected-step location. The absolute density stop is 2,500 throughout.
Selection uses post-update training error every 100 updates in settling.
Held-out evaluation begins after all fitting decisions finish.

We measure full-grid and off-grid reconstruction error, spatial macro error,
detail error, predeclared local-region error, population, principal widths,
optimized/stored parameter counts, serialized size and elapsed time. There
are 1,024 independent off-grid queries. A separate NumPy covariance-solve
decoder checks 128 of them. Fixed geometry has fewer optimized variables than
the other controls, and adaptive final populations can differ; those capacity
differences are part of the result.

The first two fits use the initial sequential CPU schedule. The remaining
independent fits use three workers with two PyTorch threads each, calling the
same frozen fitting function. The execution-only handoff preserves all outputs
and any interrupted next-run prefix. Reported times reflect shared-machine
execution and are not isolated throughput measurements.
The owner requested a travel pause during three faulted-model fits. Their
processes were suspended and resumed with exact in-memory state retained;
the pause lasted 350.964 seconds. Raw elapsed times include this interruption.
Completed-update disk checkpoints were independently loaded and hashed before
resumption. No fitting choices or numerical source changed.

### 4.3. Acoustic application study

We reuse five completed observation-only development fits rather than
retuning them for this paper. The supplied Marmousi and Overthrust arrays have
70 by 70 samples at 10 m spacing and velocities rescaled to 1500--4500 m/s.
They are small development cases, not the original physical benchmarks.
The acquisition has three sources, 31 receivers, a 15 Hz Ricker wavelet and
1,201 samples at 1 ms. Source and receiver footprints are 10 m physical
Gaussians; propagation uses fourth-order spatial accuracy and a 200 m PML.
Receiver partitions contain 16 training, eight validation and seven test
receivers. Observations were generated separately on a 5 m grid over the same
domain, with a further 2.5 m discretization comparison retained in the record.

The runs use 4,000, 8,000 or 16,000 updates across four frequency stages, with
the absolute density stop fixed at update 500 per stage. A 4,000-update
Marmousi control fixes the population at 1,024. Conditional extensions use
training/validation evidence and measured cost; configurations were frozen
before the final reference-velocity and held-out waveform audit. These cases
had already been used in development, so this is not a blind geology test.

## 5. Results

### 5.1. Explicit local control and measured scale coupling

In the constructed diagnostic, the maximum discrepancy from the independent
NumPy decoder at 128 off-grid points was 9.095e-13 m/s. The weighted lambda
derivative was 2.1977667152953564 by autograd, 2.197766715295358 by independent
analytic differentiation and 2.1977667142891733 by central differences. All
10,186 sampled points outside the local supports remained unchanged through
the 41 states. Shared physical nodes on 20, 10 and 5 m grids and saved-field
reload agreed exactly.

The spatially smoothed total changed by at most 37.739 m/s, with RMS change
6.675 m/s, while the broad component parameters remained fixed. This is the
measured distinction between explicit component control and an invariant
spatial macro model. The representation supplies the former; an application
requiring the latter must specify that constraint separately.

**Figure 1.** Constructed continuous local control. Broad field, total field,
local contribution, 80 m spatial macro views and their difference, with a
continuous lambda control in the companion interactive artifact. The figure
reports prescribed parameters and does not imply learned recovery.

### 5.2. Learned geometry and population

All 18 declared fits completed, using 320,000 full-grid optimizer updates and
zero acoustic solves. The table reports mean and sample standard deviation
across the three predeclared seeds; every error is in m/s. Learned geometry
holds K=24 fixed, while adaptive population additionally enables the existing
controller. Macro and detail use the same spatial operator for all controls.

| Target / control | Grid RMSE | Macro RMSE | Detail RMSE | Local detail RMSE | Off-grid RMSE |
|---|---:|---:|---:|---:|---:|
| Curved / Fixed geometry | 18.81 +/- 0.78 | 2.88 +/- 0.68 | 17.30 +/- 0.40 | 37.17 +/- 0.25 | 20.10 +/- 0.44 |
| Curved / Learned geometry | 2.11 +/- 0.20 | 0.10 +/- 0.04 | 2.10 +/- 0.20 | 5.23 +/- 0.44 | 2.19 +/- 0.20 |
| Curved / Adaptive population | 1.72 +/- 0.47 | 0.08 +/- 0.06 | 1.72 +/- 0.47 | 4.27 +/- 1.22 | 1.81 +/- 0.46 |
| Faulted / Fixed geometry | 60.55 +/- 3.04 | 13.99 +/- 1.43 | 50.19 +/- 1.65 | 48.28 +/- 1.17 | 60.06 +/- 2.74 |
| Faulted / Learned geometry | 2.22 +/- 0.60 | 0.24 +/- 0.11 | 2.16 +/- 0.56 | 2.82 +/- 0.91 | 2.24 +/- 0.59 |
| Faulted / Adaptive population | 2.22 +/- 0.60 | 0.24 +/- 0.11 | 2.16 +/- 0.56 | 2.82 +/- 0.91 | 2.24 +/- 0.59 |

The replicated effect is geometry learning. At the same 24 components, mean
off-grid RMSE falls from 20.10 to 2.19 m/s for the curved model and from 60.06
to 2.24 m/s for the faulted model. Both spatial macro error and local-detail
error decrease. The learned field therefore represents broad structure and
localized variation simultaneously, rather than exchanging one for the other
in these examples. This comparison enables additional optimized variables:
fixed geometry optimizes 26 scalars, while learned geometry optimizes 146.
The conditional budgets also differ for some controls. The result concerns
these declared fitting procedures; it is not a comparison at equal optimized
dimension or equal execution time.

Only one population edit was accepted: the curved model at seed 0 splits one
component into two at update 2,100, increasing K from 24 to 25. Its sampled
maximum velocity change is 22.913 m/s, within the 25 m/s event limit. At the
same 20,000-update allowance, selected off-grid RMSE is 1.28 m/s with this
split versus 2.42 m/s for the paired fixed-population fit; local-detail RMSE
is 2.86 versus 5.74 m/s. No clones or prunes are accepted in this study. The
other five adaptive/fixed-population pairs have exactly identical selected
grid fields and off-grid values. This is evidence of useful growth in one
case; the broader three-seed finding is the benefit of learning geometry.

Geometry makes the coexistence of scales explicit. Fixed principal standard
deviations span 138.67--166.40 m. Learned curved fields span approximately
14.62--1,074.46 m across the runs; learned faulted fields span
12.88--1,280.00 m. Broad and narrow components coexist within each learned
field, with nonzero anisotropy and both positive and negative amplitudes.
Some faulted-model widths reach the declared 1,280 m upper projection limit.
These are representation scales, not inferred seismic resolution or uncertainty.

The selected fields store 146 scalars at K=24 or 152 at K=25, compared with
8,385 values in the training grid. Their serialized field files contain
4,121 or 5,129 bytes, respectively. These counts exclude training samples,
optimizer moments and evaluation arrays. They establish compact storage for
these two analytic fields; they do not establish inexpensive acoustic solves.
Fourteen fits reach the 20,000-update ceiling, and twelve still meet the
declared continuation indicator at their final allowance. Complete learning
curves retain late improvements and optimizer fluctuations. The recorded
endpoints are finite-budget results, not assertions of global convergence.
Appendix B reports the selected steps and work for every run.

The independent post-fit audit verified paired initializations, selection and
continuation decisions, physical invariants, fixed controls, event guards and
saved-field replay. Its independent NumPy decoder differs by at most
2.274e-12 m/s at the checked off-grid points. Evaluation consumed the selected
fields only after all fitting decisions finished.

**Figure 2.** Target, reconstruction, spatial macro and local detail for the
predeclared seed 0 across all three controls, with physical covariance ellipses.
Error colors saturate at plus/minus 60 m/s; the reported error metrics use every
unclipped value. Ellipses are two-standard-deviation covariance contours;
compact support extends to six standard deviations. They show geometry,
not uncertainty. Image sample centers coincide with the physical grid nodes.

**Figure 3.** All-seed macro/detail/off-grid errors, learning curves and
population histories. Parameter counts and elapsed work accompany the errors.

### 5.3. Waveform-driven velocity reconstruction

| Development fit | Updates | Time (min) | Final K | Validation loss | Raw test relative L2 | Velocity RMSE (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| Marmousi, adaptive | 8,000 | 66.27 | 944 | 4.87822e-5 | 1.019% | 313.89 |
| Marmousi, fixed population | 4,000 | 32.08 | 1,024 | 5.80594e-5 | 1.054% | 284.36 |
| Marmousi, adaptive | 16,000 | 133.76 | 954 | 4.47066e-5 | 0.945% | 334.80 |
| Overthrust, adaptive | 4,000 | 36.24 | 935 | 9.69129e-5 | 1.166% | 208.57 |
| Overthrust, adaptive | 8,000 | 73.66 | 943 | 6.19266e-5 | 0.982% | 201.83 |

Validation is the selected normalized cumulative-band waveform loss. Raw test
relative L2 is evaluated on the held-out receiver traces without the training
weighting. Velocity RMSE uses the reference field only after selection.
Initial velocity RMSE was 512.73 m/s for Marmousi and 1160.47 m/s for Overthrust.

The fits demonstrate acoustic optimization of the explicit physical field.
Overthrust improves both waveform and velocity errors with additional settling.
Marmousi's longer runs improve waveform fit while increasing velocity RMSE.
The latter is a useful inverse-problem result: waveform improvement alone does
not establish better model recovery.

A post-hoc descriptive audit applies the same 80 m spatial macro operator to
these frozen acoustic outputs. It performs no further fitting or selection.
All five selected fields improve both macro and detail errors relative to
their initial models. The largest remaining errors occur in the detail view.
This identifies a concrete output of the existing workflow: useful broad
structure can be recovered while finer structure remains imperfect.

| Development fit | Initial macro RMSE | Selected macro RMSE | Initial detail RMSE | Selected detail RMSE |
|---|---:|---:|---:|---:|
| Marmousi, adaptive 8k | 327.60 | 96.93 | 375.15 | 292.60 |
| Marmousi, fixed 4k | 327.60 | 95.03 | 375.15 | 261.37 |
| Marmousi, adaptive 16k | 327.60 | 115.05 | 375.15 | 306.98 |
| Overthrust, adaptive 4k | 1099.18 | 35.15 | 319.00 | 204.87 |
| Overthrust, adaptive 8k | 1099.18 | 32.56 | 319.00 | 198.70 |

All entries are m/s. Smoothing is not an orthogonal projection, so macro and
detail squared errors need not sum to total squared error. These numbers
describe the selected fields; they do not assign the improvement to population
adaptation or identify which fine structures the acquisition resolves.

Across these five runs, accepted density edits comprise 320 prunes and no
clones or splits. The 6.9 m clone threshold is below the 11.83 m sampling floor
on this domain. Therefore these runs exercise geometry learning and pruning;
they do not demonstrate useful adaptive growth. Their stored parameter counts
also exceed the 4,900 values of the sampled velocity array. We make no storage
compression claim from these acoustic cases.

The five fits used 120,090 forward and 120,000 adjoint physical shot solves;
runner and independent verification added 30 forward solves. Independent
decoding and waveform-score checks passed. Finite-budget learning curves
retain late changes and occasional settling-phase loss increases; all runs
and selection locations are reported without declaring global convergence.

**Figure 4.** Observation-only Marmousi and Overthrust reconstructions and
waveform/velocity metrics. Field plots retain physical axes and common color
scales. The extended Marmousi result exposes the difference between waveform
fit and evaluation-only velocity accuracy.

### 5.4. Numerical and resource acceptance

The implementation is checked against independent field formulas, parameter
and acoustic finite differences, covariance/sampling invariants, optimizer
state transactions, data-partition perturbations, saved-field replay and
completed-update restart. The final source and isolated-wheel release gates
each passed all 76 tests; Ruff and repository/link checks also passed. The
closeout evidence matrix records the exact source, wheel, test inputs and
environment.

An actual disk-backed acoustic graph was retained while an injected adjoint
failure interrupted execution. The new invocation boundary removed owned
scratch, retained the original exception and preserved a checkpoint that then
resumed successfully. Separate tests preserved unrecognized files and a
replacement directory. These checks establish ordinary exception cleanup;
abrupt process termination and persistent operating-system locks lie outside
that guarantee.

Short CUDA gradient and restart checks pass at their recorded tolerances.
Historical and same-source long-prefix comparisons failed their original
strict state tolerances; the latter reached a maximum field difference of
6.30177 m/s at common first-stage boundaries. These outcomes remain visible.
Deterministic execution settings are not a bitwise long-trajectory guarantee.

A separate 256 by 512, 20-shot analytic fixture completed 12 updates across
four processes using disk storage on an 8 GB RTX 4060 Laptop GPU. It counted
360 fitting forwards, 240 adjoints and 20 verification forwards, with maximum
reserved GPU memory 5.39 GiB. Its first four updates precede an archived memory
correction, so it is explicitly a source-transition recovery check. It supplies
a resource observation rather than a geological convergence result.

## 6. Discussion

The reusable object developed here is a physical velocity field with explicit
local degrees of freedom. Its geometry can be inspected, queried independently
of one propagation grid, differentiated through a training objective and saved
without the original training process. These capabilities are the foundation
for new applications. Their value does not depend on one global ranking of
all inversion methods.

The macro/detail application also clarifies the limits of component semantics.
Wide and narrow kernels can coexist, but they overlap and interact through the
velocity bound. A user who needs an unchanged spatial macro field must specify
an operator and an acceptable change. Similarly, kernel density is a record of
an optimization policy, not a map of missing seismic information. Resolution
requires an independent connection to the acquisition and inverse problem.
[Fichtner and Trampert (2011)](https://academic.oup.com/gji/article/187/3/1604/616815).

Adaptation should be evaluated through its actual decisions and outputs.
Available clone/split operations do not imply accepted growth in every case.
Here geometry learning improves both prescribed structures at every paired
seed, while population growth changes one of six paired representation fits.
The acoustic cases exercise geometry and pruning. These observations locate
the present evidence in specific mechanisms and leave the broader usefulness
of population growth as a question for subsequent applications.
A component-count ceiling is a resource constraint, not an optimal complexity
criterion. Signed amplitudes permit contrasting contributions and cancellation;
an individual kernel is consequently not guaranteed to be a unique geological
object. These properties make controlled geometry and population experiments
essential when interpreting the resulting spatial organization.

The present baseline deliberately retains one objective pipeline and one
population policy. Future work can attach geological penalties to the same
physical parameters, test proposals with measured waveform reductions, or
develop calibrated diagnostics for unresolved structure. Those extensions
must state their own data access, objective and acceptance criteria. They do
not require reopening the field's units, derivative or saved-state contracts.

## 7. Conclusion

Adaptive Gaussian velocity fields make spatial support an explicit part of
model learning. Signed amplitudes, mobile centers and full covariance provide
continuous physical control, while bounded topology transactions support a
variable population. The implementation connects these operations to acoustic
FWI with a defined separation between training, validation and final evaluation.
The completed supervised study shows simultaneous macro and local-detail
reconstruction with 146--152 stored scalars. Learning geometry reduces mean
off-grid error from 20.10 to 2.19 m/s and from 60.06 to 2.24 m/s on the two
analytic targets; bounded population growth provides an additional improvement
in one paired case. Five independently audited acoustic fits establish use of
the representation in observation-only inversion and expose the distinction
between waveform fit and velocity recovery. Together with checked local control
and software contracts, these results provide a concrete foundation for more
specialized model-building algorithms.

## Reproducibility and evidence records

The exact formulas and interfaces are in [METHODS.md](METHODS.md),
[ALGORITHM.md](ALGORITHM.md) and [API.md](API.md). The finite release criteria
are in [BASELINE_CLOSEOUT.md](BASELINE_CLOSEOUT.md); accepted checks, limitations
and source transitions are in [ENGINEERING_SPEC.md](ENGINEERING_SPEC.md).

Local evidence roots are `results/representation_foundation_20260916T141100Z/`
(constructed diagnostic), `results/overnight_20260916T054800Z/` (archived
waveform study and resource checks), and
`results/baseline_closeout_20260916T155000Z/` (final source acceptance and
supervised study). Numerical artifacts and experiment harnesses remain outside
the publishable package. Each sealed record contains source/input identities,
profiles, raw results and independent checks. Public archival identifiers,
author affiliations, redistribution permissions and the submission format are
editorial release metadata to be completed before public submission.

## References

Peters, F. C., Fontes Junior, E. F., Mansur, W. J., Soares Filho, D. M.,
Monteiro, C. S. G., and Carvalho, P. (2017). An adaptive meshless parameterization
for full waveform inversion. Engineering Analysis with Boundary Elements, 83,
113--122. [10.1016/j.enganabound.2017.07.006](https://doi.org/10.1016/j.enganabound.2017.07.006).

Kadu, A., van Leeuwen, T., and Mulder, W. A. (2017). Salt Reconstruction in Full
Waveform Inversion with a Parametric Level-Set Method. IEEE Transactions on
Computational Imaging, 3(2). [10.1109/TCI.2016.2640761](https://doi.org/10.1109/TCI.2016.2640761).

Izadian, S. (2026). Genetic algorithm for time-lapse seismic inversion of CO2
plumes. Acta Geophysica, 74, article 120.
[10.1007/s11600-026-01856-x](https://doi.org/10.1007/s11600-026-01856-x).

Sun, J., Innanen, K., Zhang, T., and Trad, D. (2023). Implicit Seismic Full
Waveform Inversion With Deep Neural Representation. Journal of Geophysical
Research: Solid Earth, 128(3), e2022JB025964.
[10.1029/2022JB025964](https://doi.org/10.1029/2022JB025964).

Kang, B., Chen, R., Yang, K., Li, M., and Wu, B. (2026). Implicit full waveform
inversion with adaptive Fourier frequency bases learning. Geophysical Journal
International, 244(1), ggaf404. Published online 14 October 2025.
[10.1093/gji/ggaf404](https://doi.org/10.1093/gji/ggaf404).

Kerbl, B., Kopanas, G., Leimkuehler, T., and Drettakis, G. (2023). 3D Gaussian
Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics,
42(4). [Author project and paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

Fichtner, A., and Trampert, J. (2011). Resolution analysis in full waveform
inversion. Geophysical Journal International, 187(3), 1604--1624.
[10.1111/j.1365-246X.2011.05218.x](https://doi.org/10.1111/j.1365-246X.2011.05218.x).

Richardson, A. Deepwave: PyTorch-integrated wave propagators.
[Official software documentation](https://www.ausargeo.com/deepwave/).
Accessed 16 September 2026; installed dependency versions are retained in the
accepted environment record.

## Appendix A. Analytic targets and supervised settings

All distances below are meters and velocities are m/s. Define w(t)=cos(pi*t/2)
squared for absolute t at most one, and zero elsewhere. The compact window is
W(x,z;xc,zc,rx,rz)=w((x-xc)/rx)w((z-zc)/rz). The physical domain is
[0,1280] by [0,640]. The curved target is

\[
h_c(x)=220+0.10x+40\sin(2\pi x/1280),
\]

\[
v_c(x,z)=1850+1.6z+180\tanh((z-h_c(x))/80)
+110\tanh((z-(430-0.07x))/60)
+140W(x,z;900,430,240,120)\sin(2\pi(z-h_c(x))/70).
\]

The faulted target is

\[
h_f(x)=260+50\tanh((x-700)/20)+30\sin(2\pi x/1280),
\]

\[
v_f(x,z)=1900+1.5z+260\tanh((z-h_f(x))/35)
+100W(x,z;360,440,220,130)\cos(2\pi(z-0.12x)/90)
-90W(x,z;940,200,180,100).
\]

The local-detail metric uses the curved window's support rectangle, or the
union of the two faulted-window rectangles. It compares the residuals after
the declared 80 m smoothing operation, restricted to those grid points.
Off-grid queries are uniform on the same domain, using NumPy default_rng with
seed 20260916. They test spatial interpolation within these two prescribed
fields; they are not additional geological examples.

All controls start from a 4 by 6 lattice, width ratio 0.65, jitter ratio 0.15
and seeds 0, 1 and 2. Initial amplitudes are zero; background endpoints are
1800 and 3100 m/s. Bounds are 1500 and 4500 m/s, the smooth-bound scale is
20 m/s and the sampling-response parameter is 0.001. Execution uses CPU
float64, the sparse_fused decoder, two PyTorch threads per process and enabled
deterministic algorithms. The field and controller source is frozen for every
fit. Paired initial states are checked for exact equality.

Canonical Adam groups use amplitude/background rate 4, covariance rate 0.008
and center-rate ratio 0.02. The fixed-geometry control sets both geometry rates
to zero and retains their gradients for the controller interface. Its optimized
variables are the amplitudes and two background endpoints. No added regularizer
is used. Every optimizer update evaluates the complete training grid.

The controller has horizon 20,000, warmup 50, interval 50, minimum age 50,
gradient threshold zero, split-extent fraction 0.01, prune-amplitude threshold
0.5 m/s, and a cumulative sampled event limit of 25 m/s. Each event permits at
most eight growth and eight pruning proposals. The adaptive population range
is 12--144; the controls set both bounds to 24. The paired initialization seed
also seeds the controller. A stop fraction of 0.125 fixes the absolute density
stop at 2,500, including when a fit continues to a larger update allowance.

At 5,000 and 10,000 updates, continuation requires either at least 0.5 percent
improvement in the latest 500-update median training loss over the preceding
500 updates, or the best eligible sampled state falling within the last 500
updates. Every trajectory stops by 20,000. Selection compares post-update
training error every 100 updates, starting at 2,500. Complete recovery states
are saved every 1,000 updates, and each allowance boundary preserves its
decision and selected field. The independent post-fit audit recomputes these
decisions before generating evaluation metrics.

## Appendix B. Every supervised fit and its measured work

FG denotes fixed geometry; LG denotes learned geometry with fixed population;
AP denotes adaptive population. Updates are completed full-grid Adam updates;
the selected step uses the predeclared settling-phase rule. Stored/optimized
counts describe the selected field. All 18 runs use zero acoustic solves.

| Run | Updates | Selected step | K | Stored / optimized scalars | Seconds | Field bytes |
|---|---:|---:|---:|---:|---:|---:|
| Curved FG s0 | 10000 | 7200 | 24 | 146 / 26 | 584.5 | 4121 |
| Curved LG s0 | 20000 | 19900 | 24 | 146 / 146 | 1164.4 | 4121 |
| Curved AP s0 | 20000 | 20000 | 25 | 152 / 152 | 1522.1 | 5129 |
| Curved FG s1 | 10000 | 7600 | 24 | 146 / 26 | 824.8 | 4121 |
| Curved LG s1 | 20000 | 19800 | 24 | 146 / 146 | 1528.1 | 4121 |
| Curved AP s1 | 20000 | 19800 | 24 | 146 / 146 | 1394.0 | 4121 |
| Curved FG s2 | 10000 | 7400 | 24 | 146 / 26 | 634.8 | 4121 |
| Curved LG s2 | 20000 | 19700 | 24 | 146 / 146 | 1363.6 | 4121 |
| Curved AP s2 | 20000 | 19700 | 24 | 146 / 146 | 1274.7 | 4121 |
| Faulted FG s0 | 20000 | 15200 | 24 | 146 / 26 | 1349.2 | 4121 |
| Faulted LG s0 | 20000 | 19800 | 24 | 146 / 146 | 1328.5 | 4121 |
| Faulted AP s0 | 20000 | 19800 | 24 | 146 / 146 | 1233.6 | 4121 |
| Faulted FG s1 | 10000 | 7900 | 24 | 146 / 26 | 686.1 | 4121 |
| Faulted LG s1 | 20000 | 19900 | 24 | 146 / 146 | 5035.7 | 4121 |
| Faulted AP s1 | 20000 | 19900 | 24 | 146 / 146 | 5054.8 | 4121 |
| Faulted FG s2 | 20000 | 12900 | 24 | 146 / 26 | 5023.3 | 4121 |
| Faulted LG s2 | 20000 | 19900 | 24 | 146 / 146 | 1640.4 | 4121 |
| Faulted AP s2 | 20000 | 19900 | 24 | 146 / 146 | 1644.6 | 4121 |

Seconds are raw elapsed durations measured inside each fit. They include
shared-host scheduling and, for faulted LG s1, AP s1 and FG s2, the documented
350.964-second travel suspension. The sum of durations is 33,286.96 seconds;
because fits overlap, it is neither campaign elapsed time nor CPU time.
No throughput or speed advantage is inferred from this table. Per-seed errors,
all continuation decisions, full learning histories and component parameters
are retained in the sealed study record.
