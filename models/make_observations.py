"""Build an observation-only bundle from a prepared velocity model.

    python models/make_observations.py models/marmousi.npy data/marmousi.pt

By default the traces are modelled under conditions the inversion does not
share, so fitting them is not an inverse crime:

* propagation runs on a finer grid with a finer time step, then the traces are
  resampled onto the inversion's grid and sampling rate;
* band-limited noise is added at a declared signal-to-noise ratio;
* the wavelet stored for the inversion differs from the one that made the data,
  standing in for an estimated source.

Pass ``--ideal`` to disable all three and reproduce the self-consistent case.
The velocity array is used here and for scoring afterwards; it never enters the
inversion, which receives waveforms, geometry and partitions only.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

from gaussian_fwi import Acquisition, GridSpec, Observations, save_observations
from gaussian_fwi.core.physics import lowpass


def ricker(samples, dt, peak_hz, dtype, delay=None):
    delay = 1.1 / peak_hz if delay is None else delay
    time = torch.arange(samples, dtype=dtype) * dt - delay
    squared = (torch.pi * peak_hz * time).square()
    return (1 - 2 * squared) * torch.exp(-squared)


def geometry(width, shots, depth_index, scale=1):
    """Source columns and receiver columns, in grid indices at the given scale."""
    columns = torch.linspace(4, width - 5, shots).round().long()
    receivers = torch.arange(1, width - 1)
    return (columns * scale, receivers * scale, depth_index * scale)


def acquisition_for(shape, spacing, dt, wavelet, shots, depth_index, *,
                    scale=1, pml_width=20, peak_hz=12.0, max_velocity=5000.0):
    columns, receivers, depth = geometry(shape[1] if scale == 1 else (shape[1] - 1) // scale + 1,
                                         shots, depth_index, scale)
    return Acquisition(
        grid=GridSpec(shape, spacing),
        dt=dt,
        source_amplitudes=wavelet[None, None].repeat(shots, 1, 1),
        source_locations=torch.stack([torch.full((shots,), depth), columns], dim=-1)[:, None],
        receiver_locations=torch.stack(
            [torch.full((len(receivers),), depth), receivers], dim=-1)[None].repeat(shots, 1, 1),
        pml_width=pml_width,
        pml_frequency=peak_hz,
        max_velocity=max_velocity,
    )


def add_noise(traces, dt, signal_to_noise, band_hz, seed):
    """Add band-limited Gaussian noise at a declared per-survey amplitude ratio."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(traces.shape, generator=generator, dtype=traces.dtype)
    noise = lowpass(noise, dt, band_hz)
    scale = traces.square().mean().sqrt() / noise.square().mean().sqrt() / signal_to_noise
    return traces + noise * scale


def build(velocity, *, spacing=10.0, dt=0.001, samples=800, peak_hz=12.0, shots=8,
          depth_index=2, pml_width=20, max_velocity=5000.0, refinement=2,
          signal_to_noise=10.0, source_error=0.08, noise_seed=0, ideal=False):
    """Model a fixed-spread surface survey, by default off the inversion's grid."""
    dtype = velocity.dtype
    depth, width = velocity.shape
    if ideal:
        refinement, signal_to_noise, source_error = 1, None, 0.0

    # Propagate on a finer grid and time step than the inversion will use.
    fine_shape = ((depth - 1) * refinement + 1, (width - 1) * refinement + 1)
    fine = velocity if refinement == 1 else torch.from_numpy(
        zoom(velocity.numpy(), (fine_shape[0] / depth, fine_shape[1] / width), order=1),
    ).to(dtype)
    fine_dt = dt / refinement
    # Deepwave injects a source as amplitude * dt^2, so a finer time step alone
    # would scale recorded amplitudes by 1/refinement^2. Compensate, or the fit
    # would chase a discretisation artifact instead of the physics.
    true_wavelet = ricker(samples * refinement, fine_dt, peak_hz, dtype) * refinement ** 2
    modelling = acquisition_for(fine_shape, spacing / refinement, fine_dt,
                                true_wavelet, shots, depth_index, scale=refinement,
                                pml_width=pml_width * refinement, peak_hz=peak_hz,
                                max_velocity=max_velocity)
    with torch.no_grad():
        traces = modelling.simulate(fine)[..., ::refinement].contiguous()

    if signal_to_noise is not None:
        traces = add_noise(traces, dt, signal_to_noise, 2.5 * peak_hz, noise_seed)

    # The inversion receives an estimated wavelet, not the one that made the data.
    recorded = ricker(samples, dt, peak_hz * (1 + source_error), dtype, delay=1.1 / peak_hz)
    inversion = acquisition_for((depth, width), spacing, dt, recorded, shots,
                                depth_index, pml_width=pml_width, peak_hz=peak_hz,
                                max_velocity=max_velocity)
    index = torch.arange(inversion.receiver_locations.shape[1])
    partitions = {"train": index[index % 5 > 1],
                  "validation": index[index % 5 == 1],
                  "test": index[index % 5 == 0]}
    return Observations(inversion, traces.to(dtype)), partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="float32 (z, x) velocity array in m/s")
    parser.add_argument("destination", type=Path, help="New observation bundle path")
    parser.add_argument("--spacing", type=float, default=10.0, help="Inversion grid spacing (m)")
    parser.add_argument("--dt", type=float, default=0.001, help="Recorded time step (s)")
    parser.add_argument("--samples", type=int, default=800, help="Samples per trace")
    parser.add_argument("--peak-hz", type=float, default=12.0, help="Source peak frequency")
    parser.add_argument("--shots", type=int, default=8, help="Number of surface shots")
    parser.add_argument("--refinement", type=int, default=2,
                        help="Modelling grid refinement; 1 reproduces the inverse crime")
    parser.add_argument("--signal-to-noise", type=float, default=10.0,
                        help="Survey amplitude ratio of band-limited noise")
    parser.add_argument("--source-error", type=float, default=0.08,
                        help="Relative peak-frequency error of the recorded wavelet")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--ideal", action="store_true",
                        help="Self-consistent case: matched grid, no noise, exact wavelet")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    velocity = torch.from_numpy(np.load(args.model)).to(getattr(torch, args.dtype))
    observations, partitions = build(
        velocity, spacing=args.spacing, dt=args.dt, samples=args.samples, peak_hz=args.peak_hz,
        shots=args.shots, refinement=args.refinement, signal_to_noise=args.signal_to_noise,
        source_error=args.source_error, noise_seed=args.noise_seed, ideal=args.ideal)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_observations(observations, partitions, args.destination)
    shots, receivers, samples = observations.traces.shape
    counts = {name: len(ids) for name, ids in partitions.items()}
    conditions = ("matched grid, noiseless, exact wavelet (inverse crime)" if args.ideal else
                  f"modelled on a {args.refinement}x finer grid, S/N {args.signal_to_noise:g}, "
                  f"{args.source_error:.0%} source error")
    print(f"{args.destination}: {shots} shots x {receivers} receivers x {samples} samples "
          f"at {args.dt} s, partitions {counts}")
    print(f"  conditions: {conditions}")


if __name__ == "__main__":
    main()
