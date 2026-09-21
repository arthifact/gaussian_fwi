"""Fit the same data at several capacities and measure what the extra freedom buys.

    python models/make_observations.py models/marmousi.npy data/marmousi_real.pt
    python models/capacity_study.py --steps-per-stage 60

A grid spends one parameter per cell everywhere, whether or not the data
constrain that cell. This field decouples capacity from the grid, so the same
survey can be fitted with a few hundred parameters or a few thousand. The study
reports, per capacity: waveform fit, velocity error inside and outside the
illuminated depth range, and a roughness measure of the recovered field.

Errors are reported by depth because a surface spread cannot constrain velocity
below roughly half its maximum offset; a single whole-model number hides that.
The reference array scores the result afterwards and never enters a fit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from survey import draw_geometry, read_geometry

import gaussian_fwi as gfwi


def roughness(field):
    """Mean absolute second difference: a scale-free speckle measure."""
    return float(np.abs(np.diff(field, 2, axis=0)).mean()
                 + np.abs(np.diff(field, 2, axis=1)).mean())


def figure(reference, initial, fits, destination, illuminated_m, spacing, geometry=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth, width = reference.shape
    extent = (0, (width - 1) * spacing, (depth - 1) * spacing, 0)
    panels = [("Reference (evaluation only)", reference), ("Initial", initial)]
    panels += [(f"{row['gaussians']} Gaussians / {row['parameters']} params", row["velocity"])
               for row in fits]
    figure, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.5),
                                layout="constrained")
    for axis, (title, field) in zip(axes, panels):
        image = axis.imshow(field, cmap="viridis", vmin=float(reference.min()),
                            vmax=float(reference.max()), extent=extent, aspect="equal")
        axis.set(xlabel="x (m)", title=title)
        if geometry is not None:
            draw_geometry(axis, *geometry, label=axis is axes[0])
        axis.axhline(illuminated_m, color="w", lw=1.1, ls=(0, (4, 3)))
    axes[0].set_ylabel("z (m)")
    if geometry is not None:
        axes[0].legend(loc="lower left", fontsize=7.5, framealpha=.85,
                       handletextpad=.2, borderpad=.3)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    figure.colorbar(image, ax=axes, label="Velocity (m/s)", shrink=.85)
    figure.suptitle("Equal waveform fit, decreasing capacity  |  "
                    "dashed line: limit of diving-wave illumination", fontsize=11)
    figure.savefig(destination, dpi=170)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observations", type=Path, default=Path("data/marmousi_real.pt"))
    parser.add_argument("--reference", type=Path, default=Path("models/marmousi.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/capacity"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 16, 32],
                        help="Seed lattice sizes to compare")
    parser.add_argument("--steps-per-stage", type=int, default=None)
    parser.add_argument("--illuminated-m", type=float, default=300.0,
                        help="Depth above which the survey has transmitted coverage")
    parser.add_argument("--spacing", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64)
    boundary = int(args.illuminated_m / args.spacing)
    args.output.mkdir(parents=True, exist_ok=True)

    def rmse(field, start=0, stop=None):
        stop = reference.shape[0] if stop is None else stop
        return float(np.sqrt(((field[start:stop] - reference[start:stop]) ** 2).mean()))

    fits, initial = [], None
    for seeds in args.seeds:
        overrides = {"field": {"bounds": [1400.0, 5000.0]}, "seed_shape": [seeds, seeds]}
        if args.steps_per_stage is not None:
            overrides["steps_per_stage"] = args.steps_per_stage
            overrides["validation_interval"] = 5
            overrides["refinement"] = {"warmup_steps": 10, "interval": 10,
                                       "stop_fraction": 0.5, "minimum_age": 10}
        destination = args.output / f"seeds_{seeds:02d}"
        fit = gfwi.run(args.observations, destination, profile=gfwi.baseline(**overrides),
                       threads=args.threads)
        velocity = fit.velocity.numpy()
        if initial is None:
            initial = torch.load(destination / "fit/predictions.pt", map_location="cpu",
                                 weights_only=True)["initial_velocity"].numpy()
        fits.append({
            "seed_shape": f"{seeds}x{seeds}", "gaussians": fit.gaussians,
            "parameters": fit.parameters,
            "parameters_per_grid_cell": round(fit.parameters / reference.size, 3),
            "rmse_all_m_s": round(rmse(velocity), 1),
            "rmse_illuminated_m_s": round(rmse(velocity, 0, boundary), 1),
            "rmse_unilluminated_m_s": round(rmse(velocity, boundary), 1),
            "roughness": round(roughness(velocity), 1),
            "train_waveform_final": [round(v, 5) for v in fit.waveform_losses["train"]["final"]],
            "velocity": velocity,
        })

    baseline_row = {
        "seed_shape": "initial", "gaussians": 0, "parameters": 0,
        "parameters_per_grid_cell": 0.0,
        "rmse_all_m_s": round(rmse(initial), 1),
        "rmse_illuminated_m_s": round(rmse(initial, 0, boundary), 1),
        "rmse_unilluminated_m_s": round(rmse(initial, boundary), 1),
        "roughness": round(roughness(initial), 1), "train_waveform_final": None,
    }
    rows = [baseline_row] + [{k: v for k, v in row.items() if k != "velocity"} for row in fits]
    (args.output / "capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
    figure(reference, initial, fits, args.output / "capacity.png", args.illuminated_m,
           args.spacing, geometry=read_geometry(args.observations, args.spacing))

    header = f"{'capacity':>10}{'params':>8}{'/cell':>7}{'RMSE all':>10}"
    header += f"{'<' + str(int(args.illuminated_m)) + 'm':>9}{'>' + str(int(args.illuminated_m)) + 'm':>10}{'rough':>8}"
    print(header)
    for row in rows:
        print(f"{row['seed_shape']:>10}{row['parameters']:>8}"
              f"{row['parameters_per_grid_cell']:>7}{row['rmse_all_m_s']:>10}"
              f"{row['rmse_illuminated_m_s']:>9}{row['rmse_unilluminated_m_s']:>10}"
              f"{row['roughness']:>8}")
    print(f"\n{args.output}/capacity.json, {args.output}/capacity.png")


if __name__ == "__main__":
    main()
