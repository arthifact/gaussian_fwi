"""Draw observed against predicted waveforms for a finished run.

    python models/plot_waveforms.py results/marmousi results/marmousi/waveforms.png

Left: one training trace, as recorded and as predicted before and after fitting,
at the band the fit ended on. Right: root mean square residual over every
training trace at each time. The recorded trace carries the noise the bundle was
built with, so the residual cannot reach zero.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gaussian_fwi.core.physics import lowpass  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A finished run directory")
    parser.add_argument("destination", type=Path, help="Figure path")
    parser.add_argument("--observations", type=Path, default=Path("data/marmousi_real.pt"))
    parser.add_argument("--cutoff", type=float, default=20.0, help="Display low-pass in Hz")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--trace", type=int, default=20, help="Index within the training set")
    args = parser.parse_args()

    bundle = torch.load(args.observations, map_location="cpu", weights_only=True)
    dt = float(bundle["acquisition"]["dt"])
    train = bundle["partitions"]["train"]
    predictions = torch.load(args.run / "fit/predictions.pt", map_location="cpu",
                             weights_only=True)

    def band(tensor):
        return lowpass(tensor.double(), dt, args.cutoff)

    observed = band(bundle["traces"])
    before = band(predictions["initial_prediction"])
    after = band(predictions["final_prediction"])
    time = np.arange(observed.shape[-1]) * dt

    receiver = int(train[args.trace])
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 3.9), layout="constrained")

    axes[0].plot(time, observed[args.shot, receiver], color="#20282C", lw=1.5, label="Recorded")
    axes[0].plot(time, before[args.shot, receiver], color="#B95337", lw=1.2, ls="--",
                 label="Predicted, initial model")
    axes[0].plot(time, after[args.shot, receiver], color="#16756D", lw=1.2,
                 label="Predicted, selected model")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude (arbitrary)",
                title=f"One training trace  (shot {args.shot}, receiver {receiver})")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    def rms(prediction):
        residual = prediction[:, train] - observed[:, train]
        return residual.square().mean(dim=(0, 1)).sqrt().numpy()

    axes[1].plot(time, rms(before), color="#B95337", lw=1.4, ls="--", label="Initial model")
    axes[1].plot(time, rms(after), color="#16756D", lw=1.4, label="Selected model")
    axes[1].set(xlabel="Time (s)", ylabel="RMS residual (arbitrary)",
                title=f"Residual over all {len(train)} training traces and "
                      f"{observed.shape[0]} shots")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    figure.suptitle(f"Observed and predicted waveforms, low-passed at {args.cutoff:g} Hz. "
                    "The recorded traces carry noise, so the residual has a floor.", fontsize=10.5)
    figure.savefig(args.destination, dpi=180)
    reduction = 100 * (1 - float(rms(after).mean()) / float(rms(before).mean()))
    print(f"{args.destination}: mean RMS residual reduced {reduction:.1f}%")


if __name__ == "__main__":
    main()
