"""Versioned identity of actual acoustic inputs, independent of provenance labels."""

import hashlib
import json

OBSERVATION_IDENTITY_FORMAT = "gaussian-fwi-observation-content-v1"


def observation_sha256(acquisition, traces) -> str:
    """Hash actual solver configuration, acquisition tensors and observed traces.

    Hashing uses canonical JSON and little-endian contiguous bytes. Counters and
    device placement are excluded; dtype, tensor shape and values are included.
    This preserves the original experiment-bundle v1 digest byte for byte.
    """
    digest = hashlib.sha256()
    configuration = {
        "format": OBSERVATION_IDENTITY_FORMAT,
        "grid_shape": list(acquisition.grid.shape),
        "grid_spacing_m": acquisition.grid.spacing,
        "dt_s": acquisition.dt,
        "accuracy": acquisition.accuracy,
        "pml_width": acquisition.pml_width,
        "pml_frequency_hz": acquisition.pml_frequency,
        "max_velocity_m_s": acquisition.max_velocity,
    }
    if hasattr(acquisition, "measurement_metadata"):
        configuration["measurement"] = acquisition.measurement_metadata
    digest.update(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )
    for name, tensor in (
        ("source_amplitudes", acquisition.source_amplitudes),
        ("source_locations", acquisition.source_locations),
        ("receiver_locations", acquisition.receiver_locations),
        ("traces", traces),
    ):
        values = tensor.detach().cpu().contiguous().numpy()
        values = values.astype(values.dtype.newbyteorder("<"), copy=False)
        header = {
            "name": name,
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "bytes": values.nbytes,
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
