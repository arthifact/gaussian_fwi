# Observation bundles

The runner accepts a PyTorch .pt bundle with exactly three top-level keys:

| Key | Contents |
|---|---|
| acquisition | Physical grid, source amplitudes/locations, receiver locations and solver settings |
| traces | Detached measured waveforms with shape (shots, receivers, time) |
| partitions | Disjoint nonempty int64 receiver-index vectors, including train and validation |

An optional test partition is evaluated after selection. Every shot uses the
same receiver-index partition map. Reference velocities, labels for supervised
fitting and any other top-level keys are rejected.

## Physical conventions

- Grid tensors use (z,x) in 2D and (z,y,x) in 3D.
- Integer acquisition indices follow grid-tensor axis order.
- Physical Gaussian query points use (x,z) or (x,y,z), in meters.
- Velocity and signed Gaussian amplitudes are m/s; covariance is in square meters.
- Grid spacing is uniform in meters; source/receiver locations must be in bounds.
- Source amplitudes have shape (shots, sources, time); dt is in seconds.
- Point-acquisition settings include finite-difference accuracy, PML width in
  cells, PML frequency and a fixed maximum velocity.

Grid shape/spacing, dt, source wavelet, receiver geometry and waveform dtype
must be mutually consistent. There is no implicit resizing or velocity
normalization. Source/receiver footprint models must retain the same physical
meaning when a grid is refined.

## Save and load

Once acquisition and measured traces are prepared:

    import torch
    from gaussian_fwi import Observations
    from gaussian_fwi import load_observations, save_observations

    observations = Observations(acquisition, measured_traces.detach())
    partitions = {
        "train": torch.tensor(train_receiver_indices, dtype=torch.int64),
        "validation": torch.tensor(validation_receiver_indices, dtype=torch.int64),
        "test": torch.tensor(test_receiver_indices, dtype=torch.int64),
    }
    save_observations(observations, partitions, "data/observations.pt")
    loaded, partitions = load_observations("data/observations.pt")

Saving detaches and copies tensors to CPU, uses an atomic write and refuses to
replace an existing file. Loading uses weights-only deserialization and maps
tensors to CPU. The loader validates the schema and receiver partitions and
recomputes an identity from actual acquisition and trace content.

For an externally recorded identity:

    loaded, partitions = load_observations(
        "data/observations.pt",
        expected_content_sha256=expected_digest,
    )

Finite physical source/receiver footprints use
gaussian_fwi.core.footprints.FootprintAcquisition and its versioned checkpoint schema.
The same I/O functions support point and footprint acquisitions. Runtime solve
counters do not become observation identity.

## Dataset preparation and evaluation

The five preserved [prepared development models](../models/README.md) are
included in this repository. Keep additional geological datasets and generated
observations outside Git, under data/ or another local storage location. Record their provenance, rights, physical units,
source identities and generation cost in the external campaign record.
These supplied arrays have been resized and velocity-normalized; their manifest
does not specify grid spacing. Declare a physically consistent grid when
generating observations. Their presence does not change the observation-only
inversion input schema.

Generate synthetic observations separately from the inversion worker. Use the
declared independent/finer propagation setup. The worker receives waveforms and
common initialization settings only. Evaluate reference velocity after the
comparison protocol and checkpoint selection are frozen.
