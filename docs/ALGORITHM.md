# Gaussian FWI with adaptive density control

## Scope and provenance

The method fits a continuous signed Gaussian velocity field to acoustic
waveforms. It uses one optimization loop with periodic adaptive density control
and a final fixed-population interval in each frequency stage.

The active [research direction](RESEARCH_DIRECTION.md) studies this explicit
representation as a foundation for macro models and local detail. That direction
does not change the implemented algorithm below; geological priors and measured
alternative-model trials remain separately specified future extensions.

Kerbl et al. introduced interleaved Gaussian optimization and density control
for radiance fields. Their method clones small high-gradient Gaussians, splits
large ones and removes low-opacity components. Its image reconstruction results
establish precedent for that mechanism, not evidence of FWI performance.
[Original paper, sections 5 and 7.3](https://arxiv.org/html/2308.04079v1)

This implementation was written independently. The official implementation
clarifies that clones initially copy their parent position, while a binary
split samples child positions from the parent Gaussian and divides its standard
deviations by 1.6. We follow those choices, using full covariance.
[Official density-control implementation](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)

| Component | This FWI implementation |
|---|---|
| Observation operator | Discretized acoustic propagation and receivers |
| Field value | Signed velocity contributions in m/s, with smooth velocity bounds |
| Gradient statistic | Mean norm of the full training objective's physical center gradient |
| Small / large decision | Largest principal standard deviation relative to domain extent |
| Weak-component pruning | Absolute signed amplitude, plus a sampled field-change guard |
| Schedule | Warm-up, periodic opportunities, then settling in every frequency band |
| Selection | Validation checkpoints from settling only |
| Additional safeguards | Physical width limits, resource ceiling, stable IDs and atomic transactions |

Optical opacity reset, visibility statistics and camera-size culling have no
direct equivalent in this signed acoustic model and are not implemented.
Full covariance uses a Cholesky parameterization instead of optical
rotation/scale variables. The decoder evaluates a compactly tapered Gaussian
sum instead of alpha compositing. These are explicit adaptations, not a claim
that the seismic and optical objectives are identical.

## Continuous optimization

The [mathematical specification](METHODS.md) defines the field, physical units,
kernel, bounds and derivatives. At stage \(s\), let \(\mathcal B_s\) contain all
frequency cutoffs up to that stage. The training objective is

\[
J_s(\theta)=\frac{1}{|\mathcal B_s|}\sum_{f\in\mathcal B_s}
 L_{T,f}(A_h(v_\theta),d_T)+R(v_\theta)+R_{\rm raw}(u_\theta).
\]

Only training receivers contribute to \(J_s\). `WaveformObjective` defines the
fixed data normalization, filtering and optional trace preprocessing. Each
update uses all shots. The background, amplitudes, centers, log widths and
shears are optimized with Adam, followed by physical geometry projection.

One regular lattice of zero-amplitude Gaussians is seeded before stage zero.
No new lattice is appended at later frequency stages. Zero amplitude initially
preserves the background; amplitude gradients activate the geometry later.

## Refinement statistic and event ordering

After each training backward pass, before Adam, accumulate

\[
G_k=\frac{1}{n_k}\sum_{t=1}^{n_k}
\left\|\frac{\partial J_s(\theta_t)}{\partial\mu_k}\right\|_2.
\]

This is a mean of gradient norms across updates, not the norm of their mean.
The gradient is with respect to a physical position in meters, so \(G_k\) has
units \(1/\mathrm m\) for this normalized objective. It includes regularization.
The full-shot gradient is formed before taking the norm; this is not an average
of separate per-shot norms or projected camera gradients. At an event, the
statistic reflects pre-update gradients; edits use post-Adam projected geometry.

A completed update (t) is a refinement opportunity when

\[
t>t_{\rm warm},\qquad t\bmod I=0,\qquad
t<\lceil\rho T\rceil.
\]

An opportunity does not force growth. Statistics reset after each event and at
frequency changes. Stable IDs resolve exact priority ties. Components edited
less than `minimum_age` updates ago are ineligible.

1. Propose pruning eligible components with \(|a_k|\le a_{\rm prune}\), in
   increasing amplitude magnitude, subject to pruning and minimum-count limits.
2. Rank remaining eligible components with \(G_k>\tau\), in decreasing score.
3. If the largest principal width is no greater than the configured extent
   fraction, clone; otherwise propose a sampled binary split.
4. Reject inadmissible child centers/covariances. Limit net growth and total
   population. Proposals are ordered with weak pruning first, then ranked growth.
5. Apply the batch transaction. If the cumulative sampled field change exceeds
   its bound, restore everything, halve the ordered batch, and retry. An event
   can accept no edits. There is no waveform-based trial or predicted-gain claim.

The baseline uses \(T=1000\), warm-up 50, interval 50 and \(\rho=0.5\): events
are possible at 100, 150, …, 450. No topology change occurs at or after update
500. All continuous parameters remain trainable through update 1,000.
Validation selection is restricted to this settling phase. Selection and
frequency continuation are discrete operations, not differentiated through.

## Exact edit semantics

**Clone:** retain the parent and append a child with identical center,
covariance and signed amplitude. This adds the parent's contribution again;
it does not preserve the field. The parent retains its Adam state and the
child starts with fresh state. Coincident clones can subsequently diverge
because their optimizer histories differ; useful divergence is not guaranteed.

**Split:** draw two independent \(\xi_j\sim\mathcal N(0,I)\), set

\[
\mu_j=\mu_k+L_k\xi_j,\qquad
\Sigma_j=\Sigma_k/1.6^2,\qquad a_j=a_k,
\]

and replace the parent. A private CPU random generator uses the recorded seed,
global update and parent ID, leaving caller RNG state unchanged. Samples are
not clipped into the domain: inadmissible geometry is rejected. Conditioning
on acceptance can therefore alter the distribution of accepted children.
The proposal distribution is checked independently in 2D/3D. Splitting makes
no mass, moment, field-preservation or loss-decrease guarantee.

**Prune:** remove the selected weak component, retaining at least the configured
minimum. Absolute amplitude handles positive and negative velocity anomalies
symmetrically. It is a heuristic, not a complete measure of a component's
importance to waveform fit. The same event-level field guard applies.

All survivors retain their Adam step counters and moments; children start with
fresh optimizer state. Every accepted event is checked against its **single
pre-event** velocity snapshot:

\[
\max_{x_i\in\text{propagation grid}}
|v_{\rm after}(x_i)-v_{\rm before}(x_i)|\le\Delta v_{\max}.
\]

Rejected transactions restore parameter identities, gradients, metadata, Adam,
IDs and ages atomically. This is a discrete sampled velocity bound, not a bound
between grid points, across events, or on waveform error.

## Development parameters and measured work

The [single profile](../configs/baseline.json) records all settings. The 25 m/s
field-change limit, 0.5 m/s pruning threshold, 0.01 extent fraction, zero
positive-gradient threshold and 8,192-component ceiling are provisional FWI
choices. A zero threshold ranks all eligible components with nonzero gradient;
it does not establish that every such component needs refinement. These values
must be assessed on development data, including scale and noise sensitivity.
They are not transferred optical thresholds or literature-validated FWI values.

For \(S\) stages of \(T\) full-shot updates, an uninterrupted fit performs
\(S(T+1)+2\) forward batch calls and \(ST\) adjoint batch calls: initial
prediction, one evaluation per update plus each terminal state, and the final
selected prediction. Density events add no acoustic solves. The CLI performs
one additional forward batch for export verification. Point-acquisition batch
counts must also be multiplied by the shot count; footprint acquisition records
individual shot calls. Data generation, failed fits and later audits are
separate work categories.

## Evidence boundary

Independent tests cover derivatives, split sampling, edit selection, physical
constraints, transaction state, data separation, completed-stage restart and
export replay. They do not prove global convergence, generalization, superiority
over other representations or a publication-level contribution. Density control
is itself an active research topic; later work revises its allocation criteria.
[Revising Densification in Gaussian Splatting, ECCV 2024](https://arxiv.org/abs/2404.06109)

The [evaluation protocol](BASELINE_PROTOCOL.md) requires substantial runs,
convergence assessment, independent observations, controls appropriate to the
claim, repeated seeds and negative cases. External comparators are used for
claims that require them. Software acceptance and application evidence are
separate gates.
