# One inversion API

The public entry point is `gaussian_fwi.invert`; `RefinementConfig` configures its
single density controller. There is no method selector or second optimizer.

    import torch
    from gaussian_fwi import (
        GaussianField, InversionConfig, RefinementConfig,
        Preprocessing, Regularization, load_observations, invert, resume,
    )

    observations, partitions = load_observations("data/observations.pt")
    field = GaussianField(
        observations.acquisition.grid,
        background=(1500, 3000), bounds=(1500, 4500),
        sampling={"max_nyquist_response": 0.001}, backend="sparse_fused",
    ).to(observations.traces)
    config = InversionConfig(
        cutoffs=(4, 7, 12, 20), seed_shape=(32, 32),
        steps_per_stage=1000, validation_interval=25,
        refinement=RefinementConfig(),
    )
    report = invert(
        field, observations, config, "results/my_fit",
        partitions=partitions,
        regularization=Regularization(tv_weight=1e-4),
        preprocessing=Preprocessing(time_gain_power=1.5, trace_balance_cap=5),
    )

Set a seed shape with one entry per grid dimension. Seeds are added once before
optimization. For an already seeded field, use `seed_shape=None`. The population
must satisfy the configured bounds. No target velocity is accepted. The field
and observations must use identical physical grids, dtype and device. The CLI
provides the accepted CPU path; general tensor device support is not evidence
of validated CUDA execution.

## State and restart

`invert` updates the supplied field in place. It writes `field.pt`,
`predictions.pt`, `report.json`, `history.json`, `refinement.json`, `topology.json`
and one `stage_XX.pt` per completed frequency stage. Sampling diagnostics are
written when enabled. The CLI additionally records run identities and performs
a separate counted saved-field propagation.

    restored = GaussianField.load("results/my_fit/field.pt")
    velocity = restored()  # (z,x) or (z,y,x), in m/s
    points = torch.tensor([[100., 200.]], dtype=restored.background.dtype)
    values = restored(points)  # physical (x,z) in 2D

    field, report = resume(
        "results/my_fit/stage_00.pt", observations,
        "results/resumed_fit", device="cpu",
    )

Restart continues after a completed stage with the recorded algorithm,
configuration, observations, selected field, Adam state, topology and work.
It does not extend a finished stage or accept a changed optimization policy.
Mid-stage restart remains pending. Identical CPU restart is tested in 2D/3D;
bitwise cross-device equality is not promised.

Output directories must be fresh. Old training formats are rejected before
propagation or output creation. Historical field formats remain readable for
sampling and waveform replay. Their old trajectories require their frozen
software version.

## Diagnostics and configuration

`report["waveforms"]` contains normalized per-band initial/final losses for each
partition; these are not velocity errors. `topology_history` records actual
operations, sampled velocity changes and rejected proposals. Center-gradient
scores use the regularized training objective, with units of inverse meters.
No candidate acoustic trials are used by this algorithm.

All options are defined in [inversion.py](../gaussian_fwi/inversion.py) and
[refinement.py](../gaussian_fwi/refinement.py). The [algorithm](ALGORITHM.md)
explains their units, ordering and limitations. Low-level transaction helpers
are private implementation details, not alternative inversion methods.
