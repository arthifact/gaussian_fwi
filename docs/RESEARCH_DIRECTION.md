# An explicit representation for several seismic workflows

## Research question

Can a velocity model with learnable locations, anisotropic spatial scales and
signed contributions provide a useful common representation for macro modelling,
local refinement and geological constraints within full-waveform inversion?

The first application is **a smooth macro model with localized fine detail**,
selected by the owner on 16 September 2026. The intended contribution is a
well-specified representation and evidence of what its controls enable. Success
is defined for each capability; universal superiority over a conventional FWI
method is not a prerequisite. The [evaluation protocol](BASELINE_PROTOCOL.md)
retains substantial budgets, independent checks and data separation.

The [working manuscript](MANUSCRIPT.md) is **Learning the Spatial Complexity of
Velocity Models: Adaptive Gaussian Representations for Full-Waveform Inversion**.
Its central object is a physical
field that can be queried, differentiated, inspected and changed locally, with
one accepted acoustic inversion method. The existing campaign supplies useful
development evidence and identified engineering limitations. Its archived
questions and results remain unchanged.

The [closeout boundary](BASELINE_CLOSEOUT.md) makes this stage finite. The
[acceptance matrix](BASELINE_ACCEPTANCE.md) identifies implementation contracts
that subsequent projects can reuse. The capability studies below are distinct
evidence levels; future geological priors, proposal algorithms and uncertainty
diagnostics are extensions, not prerequisites for completing this foundation.

## What the representation makes explicit

The current field has a depth background and signed, compactly tapered Gaussian
contributions. Each component has a center, full SPD covariance and amplitude.
The [mathematical specification](METHODS.md) gives units, support, bounds and
derivatives. Parameters describe where a contribution acts, its orientation and
extent, and whether it increases or decreases velocity. They are inspectable
degrees of freedom; an individual component need not correspond to a geological
body. Overlapping components and cancellation make decomposition nonunique.

The kernel response lies in [0,1] and changes smoothly with position. This is a
basis response, not a class probability, optical opacity or normalized geological
membership. Amplitudes remain signed velocities in m/s. A fixed population has
derivatives with respect to amplitudes and geometry. Adding and removing
components are discrete decisions; topology and checkpoint selection are not
differentiated through. The sparse decoder currently supports first derivatives.

Conventional grid velocities can also be continuous-valued and differentiable;
the official [Deepwave FWI example](https://www.ausargeo.com/deepwave/example_fwi)
even illustrates a sigmoid-bounded velocity parameterization. Our research
question concerns **movable geometry, variable spatial support and explicit
capacity allocation**. Differentiability alone is not the distinguishing claim.
Continuous field queries still feed a discretized acoustic solver.

| Capability | Present foundation | Evidence still needed for the application |
|---|---|---|
| Broad structure and local detail in one field | Constructed scale coupling, 18 supervised fits with learned recovery, and descriptive macro/detail recovery in five acoustic fits | Stability under different coverage/noise and broader geology |
| Smooth activation of a contribution | Checked continuous local-control path, zero-amplitude seeding and learned signed amplitudes | Application-specific benefits from controlling the activation path |
| Queries on several physical grids | One unchanged field can be sampled at new points | Sampling and propagation convergence at each intended workload |
| Adaptive model capacity | Guarded controller; one accepted split improves one supervised paired fit, while the five acoustic fits accept pruning only | Whether growth provides repeatable benefits across broader applications |
| Geological information | Explicit positions, orientations and spatial scales offer potential handles | A defined prior/objective, independent provenance and an accepted implementation |
| Alternative-model proposals | Existing topology transactions preserve survivor state and support rollback | Measured training-loss trials, a work ledger and a comparison against the same method without trials |
| Diagnostics of information | Geometry and field derivatives can be inspected | Sensitivity, noise and coverage calibration before interpreting them as data support or uncertainty |
| Economical workflows | Learned analytic fields store 146--152 scalars and 4,121--5,129 bytes; physical solve costs are counted separately | End-to-end costs including propagation, decoding, topology, I/O and rejected work |

## First demonstration: macro model with local detail

Separate three levels of evidence.

1. **Constructed-field diagnostic.** Prescribe broad and local components in the
   existing field. Freeze the broad model and vary local amplitudes continuously.
   Check values, derivatives, physical bounds, locality and queries on shared
   physical nodes independently. This demonstrates controllability, with zero
   optimization and zero acoustic solves; it is not a learned reconstruction.
2. **Supervised representation study.** Fit known fields containing inclined or
   curved layers and local structure, including targets not generated by this
   Gaussian representation. Use the separate 5,000/10,000/20,000-update allowances.
   Declare training and held-out spatial samples, selection and budgets before
   fitting. Evaluate macro error, local-detail error, coupling outside the region
   of interest, geometry and cost. Label velocity supervision throughout.
3. **Waveform-driven study.** Use observations alone to learn the field. Declare
   physical acquisition, finer-grid observations, initial model, partitions,
   representation controls and 4,000/8,000/16,000-update budgets. Assess whether
   broad structure is recovered and useful local detail emerges where data
   constrain it. True velocity is used only after selection for evaluation.

For the constructed diagnostic, write the raw field as
`u_macro + lambda * u_detail`, with `lambda` in [0,1]. Apply the existing smooth
velocity bound to the combined raw field:

    v_macro = B(u_macro)
    v_lambda = B(u_macro + lambda * u_detail)
    local_change = v_lambda - v_macro

These are controlled component contributions. Broad kernels do not by themselves
define a low-pass model: signed cancellation, overlap and nonlinear bounding can
change the spectrum. Also report an explicitly defined spatial macro operator,
such as Gaussian smoothing with an 80 m standard deviation, reflected boundaries
and four-standard-deviation truncation. Keep its residual distinct from the
component contribution. Local detail can change this smoothed model even when
all broad parameters are frozen. Measure that coupling.

The completed first study records each of these evidence levels in the
[manuscript](MANUSCRIPT.md). Across three starts, learning geometry at 24
components reduces mean off-grid RMSE from 20.10 to 2.19 m/s on the curved
analytic target and from 60.06 to 2.24 m/s on the faulted target. Macro and
local-detail errors decrease together. One bounded split provides additional
improvement; the remaining five population-control pairs agree exactly.
The baseline and all 18 supervised fits are complete. Additional coverage,
noise, geological priors and proposal studies belong to subsequent work.

## Controls that answer the question

Use controlled variants of the same representation where they isolate a claim:
fixed versus learned geometry; a shared scale versus mixed scales; fixed versus
adaptive population; and smooth macro-only versus macro-plus-detail capacity.
Keep data and initialization comparable, report the resulting parameter counts
and work, and explain changes in effective regularization. These belong in
archived research harnesses, not competing public engines or production profiles.

An external method is useful when the paper makes a relative accuracy, speed,
storage or application claim that needs it. It is not a compulsory gate for
every representation property. Compare equal measured work where appropriate;
Gaussian parameter storage alone cannot establish inexpensive FWI because the
acoustic wavefields still dominate some workloads. Report negative cases and
variability across the frozen cases and seeds.

## Extensions enabled by the foundation

**Stratigraphic information.** A future experiment can penalize departures from
independently supplied bedding directions or interfaces, or constrain an
explicit geological coordinate field. Declare what information is available,
its uncertainty and how it enters the objective. Compare the same Gaussian
method with and without that information. Do not derive a fitting prior from
held-out true velocity. No stratigraphic prior is implemented by this document.

**Proposal and rejection.** A future algorithm can propose local geometry or
population alternatives and evaluate their actual training objective. A
gradient score is a proposal heuristic, not a measured loss decrease. Count all
trial forwards, adjoints and rejected work. Restore rejected model/optimizer
state atomically while retaining the cumulative work ledger. Validation remains
reserved for fixed-population settling selection. This can test access to
different local solutions; it cannot guarantee a global minimum.

**Information-aware detail.** Plotting narrow components or high component density
shows representation choices. A claim about missing information requires a
defined sensitivity or uncertainty diagnostic and independent perturbation,
coverage and noise experiments. Differentiate representation capacity, numerical
sampling adequacy and seismic resolving power in every figure and metric.

## Evidence and paper structure

Organize the paper around the field and its controls, then the macro/detail
application, then measured FWI behaviour and failure cases. Give each capability
a testable claim and an independent check. Discuss related Gaussian/RBF and
implicit-field work before deciding the exact novelty statement. Candidate
extensions are a research programme, not completed paper results.

Keep one numerical method and one shipped profile. Preserve physical units,
full covariance, zero-amplitude initialization, survivor/child optimizer-state
semantics, bounded atomic density events and observation-only inversion. The
[engineering specification](ENGINEERING_SPEC.md) records acceptance and known
limitations; [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md) defines resource gates. This makes
the foundation reusable while allowing each later algorithmic idea to earn its
own evidence.
