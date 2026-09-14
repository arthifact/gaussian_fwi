# Adaptive Gaussian velocity fields: methods and short proofs

This note describes the implementation and three mathematical properties of
the Gaussian representation. It contains no
claim of superior reconstruction, convergence or novelty by itself. The
[paper plan](BASELINE_PROTOCOL.md) defines the empirical questions.

## Field and units

For a physical point \(x=(x,z)\) or \((x,y,z)\), measured in metres, define

\[
u_\theta(x)=b_0+(b_1-b_0)\frac{z}{H}
       +\sum_{k=1}^{K} a_k\,\kappa(q_k(x)),\qquad
q_k(x)=(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k).
\]

Here \(H\) is the physical depth extent. The background endpoints \(b_0,b_1\)
and signed, unnormalized peak amplitudes \(a_k\) are velocities in m/s. Centers
\(\mu_k\) are in metres; covariance \(\Sigma_k\) is in square metres; \(q_k\)
is dimensionless. Array axes are (z,x) or (z,y,x), the reverse of the physical
coordinate order. Grid shape and spacing determine the physical domain; changing
the sampling grid does not rescale the field.

The implemented kernel is a compactly tapered Gaussian, not an infinite-support
Gaussian:

\[
\kappa(q)=e^{-q/2}w(t),\quad
t=\operatorname{clip}\!\left(\frac{q-25}{11},0,1\right),\quad
w(t)=1-10t^3+15t^4-6t^5.
\]

It equals \(e^{-q/2}\) through five Mahalanobis standard deviations and vanishes
at six. The code evaluates the equivalent factored polynomial
\((1-t)^3(1+3t+6t^2)\). The taper and its first two derivatives join at the
endpoints, so the compact kernel is \(C^2\). In the transition interval,

\[
\kappa'(q)=e^{-q/2}
\left[-\frac12w(t)-\frac{30}{11}t^2(1-t)^2\right].
\]

Inside \(q\le25\), this reduces to \(-e^{-q/2}/2\); outside \(q\ge36\),
both the kernel and its first derivative are zero. The boundary values agree.

The delivered velocity is \(v_\theta(x)=B(u_\theta(x))\), with

\[
B(u)=v_{\min}+s\log(1+e^{(u-v_{\min})/s})
                 -s\log(1+e^{(u-v_{\max})/s}),\qquad s>0.
\]

The implementation uses a numerically stable equivalent. Current profiles use
\(v_{\min}=1500\), \(v_{\max}=4500\) and \(s=20\), all in m/s. The two
background endpoints remain trainable. Parameter storage is
\(K[d+d(d+1)/2+1]+2\) scalars: **6K+2 in 2D**, **10K+2 in 3D**. Report
serialized metadata, fixed buffers and optimizer memory separately.

## Property 1: full positive covariance and stable velocity bounds

Write \(L_k=D_k(I+S_k)\), where \(D_k\) has positive diagonal widths
\(e^{\ell_{ki}}\), measured numerically in metres, and \(S_k\) is strictly
lower triangular with dimensionless entries. Set \(\Sigma_k=L_kL_k^\top\).

**Proof.** The triangular matrix \(L_k\) has positive diagonal and is invertible.
For every nonzero vector \(h\),

\[
h^\top\Sigma_kh=\lVert L_k^\top h\rVert_2^2>0.
\]

Conversely, every SPD covariance has a unique positive-diagonal Cholesky factor.
Its row diagonal gives \(D_k\), and its off-diagonal entries divided by that
row diagonal give \(S_k\). Thus the parameterization covers full SPD covariance
without a redundant rotation parameterization. Mixtures can still be
nonidentifiable, including by permutation or coincident kernels.

For the bounds, let \(\sigma\) denote the logistic function. Differentiation gives

\[
B'(u)=\sigma((u-v_{\min})/s)-\sigma((u-v_{\max})/s)\in(0,1).
\]

Together with the limits \(B(-\infty)=v_{\min}\) and
\(B(+\infty)=v_{\max}\), this proves boundedness and monotonicity. The mean
value theorem also gives
\(|B(u)-B(\tilde u)|\le |u-\tilde u|\): bounding cannot amplify a raw-field
perturbation in exact arithmetic. It can attenuate gradients near saturation.

These are real-arithmetic properties for finite, valid parameters, not immunity
to overflow or ill conditioning. The code checks finiteness and projects the
principal widths, which are singular values of \(L_k\), into the declared
physical/numerical interval. Projection changes an inadmissible field; it is not
a field-preserving operation. Admissible covariance factors are left unchanged.

## Property 2: the fixed-population gradient is a local sum

Let \(A_h\) be the declared discretized acoustic and measurement operator on its
physical propagation grid. With training data \(d_T\), write the objective as

\[
J(\theta)=\ell_T(A_h(v_\theta),d_T)+R(v_\theta)+R_{\rm raw}(u_\theta).
\]

The loss includes the recorded frequency preprocessing and receiver partition;
regularization weights are part of the run specification. Define the raw-field
adjoint at grid point \(x_i\) by

\[
g_i=B'(u_i)\frac{\partial(\ell_T+R)}{\partial v_i}
                         +\frac{\partial R_{\rm raw}}{\partial u_i}.
\]

For fixed population and fixed numerical policy, put
\(\delta_{ik}=x_i-\mu_k\) and \(P_k=\Sigma_k^{-1}\). The chain rule gives

\[
\frac{\partial J}{\partial a_k}=\sum_i g_i\kappa(q_{ik}),\qquad
\frac{\partial J}{\partial\mu_k}
  =-2a_k\sum_i g_i\kappa'(q_{ik})P_k\delta_{ik},
\]

\[
\frac{\partial J}{\partial P_k}
  =a_k\sum_i g_i\kappa'(q_{ik})\delta_{ik}\delta_{ik}^{\top}.
\]

**Proof.** Differentiate the finite kernel sum using
\(\partial q/\partial\mu=-2P\delta\) and
\(\partial q/\partial P=\delta\delta^\top\), then apply the scalar chain
rule. Ordinary differentiation through \(P=(LL^\top)^{-1}\) gives derivatives
for log widths and shears. Background derivatives follow from the depth ramp.
Only supported Gaussian–query pairs contribute, since \(\kappa=\kappa'=0\)
outside support. A conservative neighbor search therefore preserves this first
derivative even when its candidate list changes.

The sparse decoder streams bounded pair chunks and implements this first-order
adjoint. It does not promise higher derivatives. Its search runs on the CPU;
pair streaming limits pair-buffer memory, not acoustic wavefield memory or total
work. Geometry projection, discrete topology choices and checkpoint selection
are not differentiated through. The acoustic gradient is that of the declared
discretization; correctness of its derivative does not prove physical adequacy.

The current runner profiles use `sparse_fused`: centers, log widths, shears and
amplitudes are concatenated for one sparse evaluation. These are differentiable
tensor operations, so each original Parameter and its Adam history remain in
place. The mathematical field is the same, but floating-point summation order
changes. Fused exports use field format v3; older block-wise sparse and dense
fields retain their existing paths for exact historical replay.

## Property 3: inactive insertion preserves the field

Insert any valid kernel with amplitude \(a_{K+1}=0\), keeping all previous
parameters and numerical settings fixed.

**Proof.** For every physical point,

\[
u_{\rm new}(x)-u_{\rm old}(x)=0\cdot\kappa(q_{K+1}(x))=0.
\]

Applying the same bounding function preserves \(v\) as well. Hence the current
waveform prediction and objective are unchanged under a deterministic fixed
forward operator. Existing Adam states are retained; the new parameter rows
start with fresh state.

An inactive kernel can subsequently learn: its amplitude derivative
\(\sum_i g_i\kappa(q_{i,K+1})\) need not vanish. Its center/covariance
derivatives do vanish at exactly zero amplitude, and become active after a
nonzero amplitude is learned. Insertion therefore offers new descent directions
without an immediate field jump. It guarantees neither a nonzero useful gradient
nor improvement after any particular optimizer step.

The current controller inserts diagonal covariance and permits subsequent full-
covariance learning. The field-preservation argument itself only needs a valid
kernel and zero amplitude.

Splitting, cloning, merging and pruning generally change the field. Do not extend
the insertion proof to those operations or to changing the width/bounding policy.

## Discrete adaptation and evidence boundary

The direct controller uses the regularized training gradient. It observes that
gradient before an ordinary Adam update and scores proposals after the update,
so the gradient is one update old. It ranks local proposals, enforces geometry,
age and capacity limits, and applies at most the configured number of compatible
edits. Prediction includes possible learning by newborn kernels; it is not a
measured waveform decrease. Optional recovery trials compare edited and unedited
anchors using training data and counted work. They are disabled in the current
direct profile.

During one event, every accepted edit is checked against the same pre-event
velocity on the actual propagation-grid points. Thus the cumulative sampled
change remains within the configured limit (25 m/s in the current profile),
rather than receiving a new allowance for each edit. This does not bound changes
between those points or across successive events. An individual failed edit
restores the field, gradients, optimizer state and identity map atomically.
A rejected recovery trial restores its unedited anchor, while deliberately
retiring any newly allocated IDs so the audit history remains unambiguous.

Training waveforms drive gradients and topology decisions. Validation chooses
the stage checkpoint; test waveforms and synthetic true velocity are scored
after fitting. Four stages use the cumulative band sets ending at 4, 7, 12 and
20 Hz. The published development profiles allocate 1,000 ordinary updates each. Equal total updates across another
frequency schedule do not imply equal trajectories, work or convergence.

For an ideal individual Gaussian, the optional width floor
\(h\sqrt{-2\log\epsilon}/\pi\) limits its normalized Fourier response at
angular frequency \(\pi/h\). It does not certify alias-free sampling of the
tapered, summed and bounded field, nor resolve seismic structure at that scale.
Field sampling and waveform propagation require separate diagnostics.

The three properties justify a valid differentiable representation and a
specific safe insertion mechanism. Accuracy versus storage, the causal benefit
of adaptation, practical inversion quality and computational cost are empirical
questions in the paper plan. They cannot be inferred from these proofs.

## Implementation and independent checks

| Claim | Implementation | Independent acceptance already available |
|---|---|---|
| Kernel and first derivatives | [raster.py](../gaussian_fwi/raster.py), [decoder.py](../gaussian_fwi/decoder.py) | Scalar kernel formula, finite differences, dense/sparse adjoints and acoustic parameter-family directional derivatives. |
| Covariance, bounds and storage | [field.py](../gaussian_fwi/field.py) | NumPy eigenvalues, independent covariance solves, physical-query values and parameter counts. |
| Zero-amplitude birth and state | [topology.py](../gaussian_fwi/topology.py), [refinement.py](../gaussian_fwi/refinement.py) | Field preservation, survivor/newborn Adam states and exact rollback invariants. |
| Selection and data separation | [inversion.py](../gaussian_fwi/inversion.py) | Held-out perturbation tests, stage restart and saved-field replay. |
| Width policy | [sampling.py](../gaussian_fwi/sampling.py) | Analytic Gaussian sampling, principal-width checks and explicit interpolation counterexamples. |

Run the checks described in the [engineering specification](ENGINEERING_SPEC.md)
to verify these software properties. The [baseline protocol](BASELINE_PROTOCOL.md)
defines the separate scientific evaluation.
