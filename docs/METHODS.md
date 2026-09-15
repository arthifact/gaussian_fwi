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

The implementation uses a numerically stable equivalent. The baseline profile uses
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

The current runner profile uses `sparse_fused`: centers, log widths, shears and
amplitudes are concatenated for one sparse evaluation. These are differentiable
tensor operations, so each original Parameter and its Adam history remain in
place. The mathematical field is the same, but floating-point summation order
changes. Fused exports use field format v3; older block-wise sparse and dense
fields retain their existing paths for exact historical replay.

## Property 3: zero-amplitude seeding preserves the field

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

The single inversion method seeds one diagonal-covariance lattice at zero
amplitude before optimization, then permits full-covariance learning. Its density
controller subsequently uses clone/split/prune; it does not repeatedly insert
zero-amplitude lattices. The zero-amplitude proof applies to initialization,
not to these later density edits.

## Discrete density control and evidence boundary

The [algorithm specification](ALGORITHM.md) gives the exact periodic gradient
statistic, sampled binary split, clone and signed-amplitude pruning definitions.
All accepted edits in an event share one cumulative propagation-grid velocity
bound and one atomic transaction. Split children inherit signed amplitudes and
have covariance divided by 1.6 squared. These operations do not preserve moments
or pointwise fields and do not guarantee reduced waveform loss.

Training data drive gradients and density decisions. Validation selects only
within the final fixed-population phase of each frequency stage. Test waveforms
and reference velocity cannot influence this process. The default four stages
use cumulative cutoffs 4, 7, 12 and 20 Hz, with 1,000 updates per stage. These
allowances do not establish convergence.

For an ideal individual Gaussian, the optional width floor
\(h\sqrt{-2\log\epsilon}/\pi\) limits its normalized Fourier response at
angular frequency \(\pi/h\). It does not certify alias-free sampling of the
tapered, summed and bounded field, nor seismic resolution at that scale. Field
sampling and waveform propagation require separate diagnostics.

The three properties establish a differentiable representation, full covariance,
bounds and inactive initialization. Accuracy/storage tradeoffs, density-control
benefit, inversion quality and cost are empirical questions.

## Implementation and independent checks

| Claim | Implementation | Independent check |
|---|---|---|
| Kernel and derivatives | [raster.py](../gaussian_fwi/raster.py), [decoder.py](../gaussian_fwi/decoder.py) | Scalar formulas, finite differences, dense/sparse adjoints and acoustic directional derivatives |
| Covariance, bounds and storage | [field.py](../gaussian_fwi/field.py) | NumPy eigenvalues, independent covariance solves, physical queries and parameter counts |
| Density edits and state | [_topology.py](../gaussian_fwi/_topology.py), [refinement.py](../gaussian_fwi/refinement.py) | Sampling distribution, survivor/newborn Adam and exact transaction rollback |
| Selection and separation | [inversion.py](../gaussian_fwi/inversion.py) | Held-out perturbations, stage restart and saved-field replay |
| Width policy | [sampling.py](../gaussian_fwi/sampling.py) | Analytic Gaussian sampling and explicit interpolation counterexamples |

See [verification](../tests/README.md) and the separate
[scientific protocol](BASELINE_PROTOCOL.md).
