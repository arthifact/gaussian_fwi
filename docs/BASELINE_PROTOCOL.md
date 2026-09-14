# Baseline research protocol

## Primary question

Does adapting Gaussian geometry and population improve the accuracy-storage
tradeoff of multiscale velocity fields, and does that benefit survive optimization
from seismic waveforms at a disclosed computational cost?

The contribution under evaluation is an explicit representation with movable
centers, full SPD covariance, signed amplitudes and controlled physical scales.
Compactness, reconstruction quality and speed are separate empirical outcomes.
The [method derivation](METHODS.md) supplies the implemented equations and
bounded mathematical properties.

## Development before evaluation

Use meaningful physical domains, initially planning for a 256x512 grid. Obtain
original geological references or use declared analytic continuous structures.
Upsampling small images does not supply geological detail.

| Work | Initial allowance | Development extensions |
|---|---|---|
| Direct supervised representation | 5,000 full-data updates | 10,000, then 20,000 |
| Waveform inversion | 1,000 updates per band; four bands = 4,000 | 8,000, then 16,000 total |

Tune suitable optimizer schedules on development data with disclosed effort.
Inspect training and validation trajectories, scaled gradients/updates,
oscillation, active width/velocity bounds and topology events. Diagnose thin-bed
errors by separating sampling, width constraints, optimizer stagnation and
capacity. More updates are warranted while useful improvement continues.

For descriptive plateau assessment, inspect three consecutive windows of best
training loss: 500 updates per window for representation; 200 within a fixed
FWI frequency objective. A candidate plateau requires less than 0.5% relative
improvement in each window after the initial allowance. Handle zero loss
explicitly and inspect typical losses so minima do not hide instability.
FWI validation must also be stable. A plateau is not a global-optimum guarantee.

A fit still improving at its budget remains budget-limited. Freeze the evaluation
allowance and selection rules after development; never extend only a favorable
evaluation case. Different multistage horizons can change earlier validation
selection and refinement, so preserve separate runs and count repeated work.

## Representation and mechanism

Use inclined thin layers, a faulted contact and a curved boundary. Compare
Gaussians, a competitive classical compact basis and a scalar continuous neural
representation at capacities matched by total degrees of freedom or stored bytes.
Gaussian capacities are initially 1,024 / 4,096 / 8,192.

Three structures x three capacities x three methods gives 27 primary direct
fits. Predeclare two additional starts at the middle capacity per case/method:
18 additional fits. Use independent denser/interstitial evaluation queries.
Report physical-unit RMSE, relative error, interface/thickness errors where
defined, complete model bytes and time to a declared quality.

Compare fixed/adaptive populations and targeted anisotropy/moving-center
controls. A common ceiling does not guarantee equal selected capacity; report
actual populations and computational work. Choose matching controls during
development. Negative outcomes belong in the report.

## Independent waveform evaluation

The initial design is three new cases x two defensible starts x four arms:
strong multiscale grid FWI, compact classical basis, scalar neural representation
and adaptive Gaussians. These 24 primary fits exclude development and audits.
Comparator adapters are future focused work; they are not included in this
baseline repository.

Add fixed-population Gaussian controls on preselected mechanism cases and grid
FWI followed by compression of the recovered field. The compressor can use that
recovered field, never true velocity or test waveforms. Count compression work.
Choose one difficult acquisition/noise regime during development instead of a
large unmotivated parameter sweep.

Training waveforms drive optimization, regularization choices and topology.
Validation selects checkpoints. Inspect test waveforms and reference velocities
only after the frozen comparisons finish. Geological cases, not individual
traces or seeds, determine the scope of generalization claims.

## Physical and computational verification

Verify spatial, temporal and boundary convergence for the actual physical source/
receiver policy and field scales. Preserve coordinates, integrated source
strength, footprints, physical extent and output times. Require decreasing
successive errors and less than 1% on final declared trace comparisons; disclose
zero-signal handling and any excluded traces. Previously verified surveys do not
automatically certify new ones.

Measure decode/backward, search/transfers, topology, propagation, total wall time
and peak GPU/host memory. Count actual forward/adjoint shot solves, including
generation, validation when separate, forecasts, recovery, recomputation,
independent checks and any pretraining. Equal epochs are not equal work.

The [RTX 4060 plan](GPU_CAMPAIGN.md) specifies the hardware pilot and next
implementation milestones. Do not infer inversion speed from decoder-only timing
or use small correctness fixtures as scientific fits.

## Deliverables

Provide accuracy/storage/time curves, causal controls, waveform-only comparisons
and one substantial application with independent validation. Each fitted case
needs configurations, identities, complete work, loss trajectories and an
independent saved-field/propagation audit. Store these artifacts externally to
Git with a manifest; the source repository remains small.

A genuine 3D representation/scaling example supports dimensional generality.
Practical 3D inversion claims require application-scale 3D FWI. Learned priors,
additional properties and extra applications remain optional.
