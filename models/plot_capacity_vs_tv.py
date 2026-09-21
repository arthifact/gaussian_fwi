"""Show that a smoothness prior is not a substitute for lower capacity.

    python models/plot_capacity_vs_tv.py

Reads finished runs and draws, on one page: the recovered fields for a small
population, a large one, and the same large one under a hundredfold TV weight;
and the velocity error inside and outside the illuminated depth, with the field
roughness, for each. Requires the runs named in --runs to exist already.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gaussian_fwi as gfwi  # noqa: E402


def roughness(field):
    return float(np.abs(np.diff(field, 2, axis=0)).mean()
                 + np.abs(np.diff(field, 2, axis=1)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=Path("models/marmousi.npy"))
    parser.add_argument("--small", type=Path, default=Path("results/capacity/seeds_08"))
    parser.add_argument("--large", type=Path, default=Path("results/marmousi"))
    parser.add_argument("--smoothed", type=Path, default=Path("results/tv_0.01"))
    parser.add_argument("--destination", type=Path,
                        default=Path("results/capacity_vs_tv.png"))
    parser.add_argument("--illuminated-m", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, default=10.0)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64)
    boundary = int(args.illuminated_m / args.spacing)
    depth, width = reference.shape
    extent = (0, (width - 1) * args.spacing, (depth - 1) * args.spacing, 0)

    def load(path):
        fit = gfwi.Run.open(path)
        return fit.velocity.numpy(), fit.parameters

    small, small_n = load(args.small)
    large, large_n = load(args.large)
    smoothed, smoothed_n = load(args.smoothed)
    initial = torch.load(args.large / "fit/predictions.pt", map_location="cpu",
                         weights_only=True)["initial_velocity"].numpy()

    def rmse(field, start=0, stop=None):
        stop = depth if stop is None else stop
        return float(np.sqrt(((field[start:stop] - reference[start:stop]) ** 2).mean()))

    entries = [
        (f"{small_n} params\nsmall population", small, "#1b6ca8"),
        (f"{large_n} params\nlarge population", large, "#b3532f"),
        (f"{smoothed_n} params\nlarge + 100x TV", smoothed, "#7a6a9b"),
    ]

    figure = plt.figure(figsize=(13.6, 7.6), layout="constrained")
    upper, lower = figure.subfigures(2, 1, height_ratios=[1.25, 1])

    fields = [("Reference (evaluation only)", reference)]
    fields += [(name.replace("\n", ", "), field) for name, field, _ in entries]
    axes = upper.subplots(1, 4)
    for column, ((title, field), axis) in enumerate(zip(fields, axes)):
        image = axis.imshow(field, cmap="viridis", vmin=float(reference.min()),
                            vmax=float(reference.max()), extent=extent, aspect="equal")
        axis.set(title=title, xlabel="x (m)", xticks=[0, 300, 600])
        axis.axhline(args.illuminated_m, color="w", lw=1.1, ls=(0, (4, 3)))
        if column:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("z (m)")
    upper.colorbar(image, ax=axes, label="Velocity (m/s)", shrink=.85)

    labels = [name for name, _, _ in entries]
    colors = [color for _, _, color in entries]
    metrics = [
        (f"Velocity RMSE above {args.illuminated_m:.0f} m\n(illuminated: lower is better)",
         [rmse(f, 0, boundary) for _, f, _ in entries], rmse(initial, 0, boundary), "m/s"),
        (f"Velocity RMSE below {args.illuminated_m:.0f} m\n(no data: staying at the prior is correct)",
         [rmse(f, boundary) for _, f, _ in entries], rmse(initial, boundary), "m/s"),
        ("Field roughness\n(speckle: lower is smoother)",
         [roughness(f) for _, f, _ in entries], None, ""),
    ]
    bar_axes = lower.subplots(1, 3)
    for (title, values, prior, unit), axis in zip(metrics, bar_axes):
        bars = axis.bar(range(len(values)), values, color=colors, width=.62)
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(labels, fontsize=8.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, max(values + ([prior] if prior else [])) * 1.22)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}" if unit else
                      f"{value:.0f}", ha="center", va="bottom", fontsize=9.5)
        if prior is not None:
            axis.axhline(prior, color="#444", lw=1.1, ls=(0, (4, 3)))
            axis.text(len(values) - .45, prior, " starting model", va="center",
                      fontsize=8.5, color="#444")
        if unit:
            axis.set_ylabel(unit)

    figure.suptitle("Equal waveform fit. A hundredfold TV weight does not reproduce "
                    "what a smaller population does.", fontsize=12)
    figure.savefig(args.destination, dpi=170)

    summary = {name.replace("\n", ", "): {
        "parameters": n,
        "rmse_illuminated": round(rmse(f, 0, boundary), 1),
        "rmse_unilluminated": round(rmse(f, boundary), 1),
        "roughness": round(roughness(f), 1),
    } for (name, f, _), n in zip(entries, (small_n, large_n, smoothed_n))}
    print(json.dumps(summary, indent=2))
    print(args.destination)


if __name__ == "__main__":
    main()
