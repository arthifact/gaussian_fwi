# Prepared development models

These are the five preserved input arrays from the research workspace. Each is
a **70 × 70 float32 array**, with tensor axes `(z, x)` and velocity in m/s.

| File | Model family |
|---|---|
| [marmousi.npy](marmousi.npy) | Marmousi |
| [bp2004.npy](bp2004.npy) | 2004 BP |
| [seam.npy](seam.npy) | SEAM Phase I |
| [overthrust.npy](overthrust.npy) | Overthrust 2D section |
| [sigsbee2a.npy](sigsbee2a.npy) | Sigsbee2A stratigraphy |

The unmodified [provenance manifest](manifest.json) records original filenames,
encodings, dimensions, units and SHA-256 identities, along with each prepared
file's identity. Preparation used linear interpolation to 70 × 70 and an
independent min-max mapping to 1,500–4,500 m/s. These are previously inspected
development inputs; the original benchmark resolutions and velocity ranges
have been changed. They do not constitute unseen or full-resolution evaluation
data for a paper.

    import numpy as np

    velocity_zx = np.load("models/marmousi.npy", allow_pickle=False)

Grid spacing is not encoded in these arrays or specified by their manifest.
Declare the physical domain and spacing explicitly when preparing an
acquisition; do not infer them from the original model dimensions.

The files are available in a Git checkout and are excluded from the installed
wheel. The [runner](../docs/DATA_FORMAT.md) still takes observation bundles:
generate synthetic observations separately, then keep target velocities out
of inversion inputs and checkpoint selection. Preserve these files unchanged;
place derived observations and results in the ignored data/results directories.
