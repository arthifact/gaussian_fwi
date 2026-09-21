"""Build an observation-only bundle from a prepared velocity model.

    python models/make_observations.py models/marmousi.npy data/marmousi.pt

The model is a float32 ``(z, x)`` array in m/s on a uniform grid. Traces are
forward modelled with the same solver the inversion uses, so a fit of this
bundle is a software demonstration on synthetic data, not a field result. The
velocity array is used here and for scoring afterwards; it never enters the fit.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from gaussian_fwi import Acquisition, GridSpec, Observations, save_observations


def ricker(samples, dt, peak_hz, dtype):
    time = torch.arange(samples, dtype=dtype) * dt - 1.1 / peak_hz
    squared = (torch.pi * peak_hz * time).square()
    return (1 - 2 * squared) * torch.exp(-squared)


def build(velocity, *, spacing=10.0, dt=0.001, samples=800, peak_hz=12.0, shots=8,
          depth_index=2, pml_width=20, max_velocity=4500.0):
    """Forward model a fixed-spread surface survey over the supplied model."""
    dtype = velocity.dtype
    depth, width = velocity.shape
    grid = GridSpec((depth, width), spacing)
    columns = torch.linspace(4, width - 5, shots).round().long()
    receivers = torch.arange(1, width - 1)
    acquisition = Acquisition(
        grid=grid,
        dt=dt,
        source_amplitudes=ricker(samples, dt, peak_hz, dtype)[None, None].repeat(shots, 1, 1),
        source_locations=torch.stack(
            [torch.full((shots,), depth_index), columns], dim=-1)[:, None],
        receiver_locations=torch.stack(
            [torch.full((len(receivers),), depth_index), receivers], dim=-1)[None].repeat(
                shots, 1, 1),
        pml_width=pml_width,
        pml_frequency=peak_hz,
        max_velocity=max_velocity,
    )
    with torch.no_grad():
        traces = acquisition.simulate(velocity)
    # Disjoint receiver sets: only training traces drive the optimizer.
    index = torch.arange(len(receivers))
    partitions = {"train": index[index % 5 > 1],
                  "validation": index[index % 5 == 1],
                  "test": index[index % 5 == 0]}
    return Observations(acquisition, traces), partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="float32 (z, x) velocity array in m/s")
    parser.add_argument("destination", type=Path, help="New observation bundle path")
    parser.add_argument("--spacing", type=float, default=10.0, help="Grid spacing in meters")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step in seconds")
    parser.add_argument("--samples", type=int, default=800, help="Samples per trace")
    parser.add_argument("--peak-hz", type=float, default=12.0, help="Ricker peak frequency")
    parser.add_argument("--shots", type=int, default=8, help="Number of surface shots")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    velocity = torch.from_numpy(np.load(args.model)).to(getattr(torch, args.dtype))
    observations, partitions = build(velocity, spacing=args.spacing, dt=args.dt,
                                     samples=args.samples, peak_hz=args.peak_hz, shots=args.shots)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_observations(observations, partitions, args.destination)
    counts = {name: len(ids) for name, ids in partitions.items()}
    print(f"{args.destination}: {observations.traces.shape[0]} shots, "
          f"{observations.traces.shape[1]} receivers {counts}, "
          f"{observations.traces.shape[2]} samples at {args.dt} s")


if __name__ == "__main__":
    main()
