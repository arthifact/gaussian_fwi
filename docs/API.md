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
and observations must use identical physical grids, dtype and device. The loader
accepts explicit `device="cuda", dtype=torch.float64`; it validates the original
bundle before converting tensors. Its stored content hash stays attached to the
source bundle; the runtime identity records converted precision. The CLI option
`--device cuda --dtype float64` also enables the deterministic execution policy
used by the short CUDA acceptance checks. See [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md)
for the tested scope and outstanding requirements for larger fits.

The loader also accepts `shot_batch_size=3` for finite-footprint acquisitions.
Consecutive shots with matching source/receiver quadrature-node counts share an
acoustic call. No nodes are padded or discarded. Physical observation identity
is unchanged; the execution grouping is recorded in the inversion specification
and cannot change during restart. `acquisition.counts` counts physical shot
solves, while `acquisition.batch_counts` counts the underlying acoustic calls.
The default batch size one preserves the previous serial behavior.

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
Completed-update restart is also available. Pass `checkpoint_interval=100` to
`invert` or `resume` to write immutable `update_XXXXXXXX.pt` snapshots. Optional
`max_updates` and `max_seconds` bound this invocation without changing the
declared optimization horizon. Time limits are checked after complete updates;
one slow update can extend a limit. A caller needing a strict process limit
should retain a separate timeout with enough room to save a boundary snapshot.

A paused report has `status="paused"` and a checkpoint name. It exports no final
selected field or test scores. Resume that checkpoint into a new directory with
the same observations/device and execution grouping. It restores the live field,
Adam, accumulated density statistics, selected best state, histories, RNG and
work. A pause adds no acoustic solves. Update and stage formats are versioned
separately; existing stage checkpoints remain supported. CPU 2D/3D interrupted
trajectories are checked exactly; CUDA float64 continuation uses the predeclared
numerical tolerances with deterministic operations enabled.

The CLI uses `--resume <runner checkpoint>` instead of `--config` and reads the
original complete profile from the checkpoint's run directory. Library-created
checkpoints can be continued with the Python `resume` API. Changing a horizon or
optimization policy requires a new experiment.

For finite-footprint acquisitions, `accumulate_shots=True` frees each acoustic
batch's saved wavefields after backward into a velocity leaf. The complete
velocity gradient is then propagated through the field once, regularization
is added once, and the optimizer and density controller use the full-survey
gradient. Set the loader's `shot_batch_size` to bound the live wavefield batch;
an unequal final batch retains its correct survey weight. The CLI equivalents
are `--accumulate-shots --shot-batch-size 1`. This option does not change the
objective or use stochastic shot updates. Point acquisitions retain the default
path and reject this option explicitly.

Accumulation defaults to false. Resume inherits the recorded setting and rejects
an explicit change before propagation. Old checkpoints retain their existing
execution. History counters record actual completed work: with accumulation,
the current update's acoustic adjoints have already occurred when its loss row
is written; without it, they occur after that row. Final work counts agree.

`wavefield_storage="cpu"` (CLI `--wavefield-storage cpu`) explicitly stores
Deepwave's intermediate wavefields in host RAM, without compression or temporal
subsampling. Propagation and gradients still run on the requested device.
Combined with shot accumulation it limits retained storage to one shot batch.
The default is `"device"`; CPU storage requires measured host-memory headroom
and can cost transfer time. Restart inherits storage placement and rejects an
explicit change. Placement does not change observation identity. There is no
automatic CPU fallback or compression option in this interface.

`wavefield_storage="disk"` (CLI `--wavefield-storage disk`) stores uncompressed
wavefields in a `wavefields/` scratch directory inside each new inversion output.
Resume creates fresh scratch in its new output; scratch location is not physical
observation identity. The inversion owns this new directory and removes recognized
Deepwave temporary subdirectories when execution completes, pauses or raises an
exception. Unrecognized entries and replaced directories are preserved. A cleanup
failure raises an error on normal exit, or adds a note to an existing error;
the runner records these notes in `failure.json`. Checkpoints remain untouched.
A killed process or an operating-system file lock can leave scratch behind.
Budget local disk space and I/O
time before large runs. Direct low-level `acquisition.simulate` calls require an
explicit existing `storage_path` when requesting disk mode. Precision, temporal
sampling and physical shot counts remain unchanged in all storage modes.

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

Optional `diagnostics=True` records gradient norms and parameter update norms
by optimizer role, plus sampled completed-update field changes at inspection or
checkpoint intervals. Parameter changes are measured before density events;
field changes include any event and are labeled accordingly. Diagnostics add
field evaluations, but no acoustic solves or optimization decisions.

All options are defined in [inversion.py](../gaussian_fwi/inversion.py) and
[refinement.py](../gaussian_fwi/refinement.py). The [algorithm](ALGORITHM.md)
explains their units, ordering and limitations. Low-level transaction helpers
are private implementation details, not alternative inversion methods.
