"""Compare a finished fit with the reference model it was generated from.

    python models/plot_result.py results/marmousi_short models/marmousi.npy figure.png

Panels: reference (evaluation only), initial model, recovered model on one
velocity scale, and the signed recovered-minus-reference difference.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gaussian_fwi as gfwi  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A finished run directory")
    parser.add_argument("reference", type=Path, help="The (z, x) reference array in m/s")
    parser.add_argument("destination", type=Path, help="Figure path")
    parser.add_argument("--spacing", type=float, default=10.0)
    args = parser.parse_args()

    fit = gfwi.Run.open(args.run)
    reference = np.load(args.reference).astype(np.float64)
    recovered = fit.velocity.cpu().numpy().astype(np.float64)
    initial = torch.load(args.run / "fit/predictions.pt", map_location="cpu",
                         weights_only=True)["initial_velocity"].cpu().numpy().astype(np.float64)

    depth, width = reference.shape
    extent = (0, (width - 1) * args.spacing, (depth - 1) * args.spacing, 0)
    low, high = float(reference.min()), float(reference.max())

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
    for axis, field, title in zip(
        axes[:3], (reference, initial, recovered),
        ("Reference (evaluation only)", "Initial", f"Recovered, {fit.gaussians} Gaussians"),
    ):
        image = axis.imshow(field, cmap="viridis", vmin=low, vmax=high, extent=extent,
                            aspect="equal")
        axis.set(xlabel="x (m)", title=title)
    axes[0].set_ylabel("z (m)")
    figure.colorbar(image, ax=axes[:3], label="Velocity (m/s)", shrink=.85)

    difference = recovered - reference
    span = float(np.abs(difference).max())
    signed = axes[3].imshow(difference, cmap="RdBu_r", vmin=-span, vmax=span, extent=extent,
                            aspect="equal")
    rmse = float(np.sqrt((difference ** 2).mean()))
    axes[3].set(xlabel="x (m)", title=f"Recovered − reference, RMSE {rmse:.0f} m/s")
    figure.colorbar(signed, ax=axes[3], label="m/s", shrink=.85)

    figure.savefig(args.destination, dpi=180)
    print(f"{args.destination}: velocity RMSE {rmse:.1f} m/s, {fit.gaussians} Gaussians")


if __name__ == "__main__":
    main()
