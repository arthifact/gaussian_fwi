# Adaptive inversion API

The main entry point starts from a seed population and adapts it from the
regularized training-waveform gradient. Frequency transitions add no prescribed
lattices. Amplitudes, centers, full covariance and the background remain trainable.

    import dynamic_refinement as fwi
    from fwi_core import Preprocessing, Regularization
    from fwi_core.io import load_observations

    observations, partitions = load_observations("data/observations.pt")
    field = fwi.GaussianField(
        observations.acquisition.grid,
        background=(1500.0, 3000.0),
        bounds=(1500.0, 4500.0),
        backend="sparse_fused",
        sampling=fwi.SamplingConfig(max_nyquist_response=1e-3),
    ).to(observations.traces)
    config = fwi.InversionConfig(
        cutoffs=(4.0, 7.0, 12.0, 20.0),
        seed_shape=(32, 32),
        steps_per_stage=1000,
        validation_interval=25,
        refinement=fwi.RefinementConfig(max_gaussians=8192),
    )
    report = fwi.invert(
        field, observations, config, "results/my_fit",
        partitions=partitions,
        preprocessing=Preprocessing(time_gain_power=1.5, trace_balance_cap=5.0),
        regularization=Regularization(tv_weight=1e-4),
    )

Check the physical scales and bands for the supplied acquisition. The
[development protocol](../docs/BASELINE_PROTOCOL.md) specifies longer allowances
and convergence diagnostics. An update budget does not certify reconstruction.

## Adaptation and state

Coverage-aware insertion is the default. Old missing-key checkpoint semantics
remain available through the legacy policy. Applicable operations include
insertion, splitting, cloning, merging, pruning and amplitude reset. With
reset_amplitude=None, an automatic amplitude cap is used; this value does not
disable reset.

Compatible edits share the cumulative sampled field-change limit. Survivors
retain Adam state; newborns start fresh with a minimum age. Insertions start at
zero amplitude. Rejected edits roll back atomically. Proposal scores are
predictions, not measured waveform improvements.

Use operations=("insert", "split") to restrict proposals, operations=() for
fixed-population optimization, and None for the default operation set.
Optional comparison_steps enables explicitly counted equal-work recovery trials.

## Saved fields and restart

    field = fwi.GaussianField.load("results/my_fit/field.pt")
    values = field(points_in_physical_axis_order)
    field, report = fwi.resume(
        "results/my_fit/stage_01.pt",
        observations,
        "results/my_continuation",
    )

Restart follows a completed frequency stage and verifies actual observation
content. It cannot change the saved training horizon or overwrite the source
run. Mid-stage restart is not yet implemented. The scheduled counterpart is
gaussian_fwi.invert / gaussian_fwi.resume.

Field and acquisition grids must agree. To query a different same-domain grid,
use sample_on_grid; do not change field metadata to simulate resampling.
Sampling diagnostics do not certify wave-propagation convergence.

The [method note](../docs/METHODS.md) describes the kernel, covariance,
gradient path, sparse decoder and physical limits.
