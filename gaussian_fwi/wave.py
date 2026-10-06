"""Acoustic wave simulation for a fixed surface survey.

Grids are ``(z, x)`` tensors with uniform spacing in metres. Source and receiver
positions are integer grid indices in the same ``(z, x)`` order.
"""

import math
from dataclasses import dataclass

import deepwave
import torch
from torch import Tensor


def ricker(samples: int, dt: float, peak_hz: float, delay: float | None = None,
           dtype=torch.float32) -> Tensor:
    """Ricker wavelet, delayed by ``1.1 / peak_hz`` unless told otherwise."""
    delay = 1.1 / peak_hz if delay is None else delay
    time = torch.arange(samples, dtype=dtype) * dt - delay
    squared = (math.pi * peak_hz * time).square()
    return (1 - 2 * squared) * torch.exp(-squared)


def lowpass(traces: Tensor, dt: float, cutoff: float) -> Tensor:
    """Zero-phase fourth-order Butterworth magnitude response along the last axis."""
    samples = traces.shape[-1]
    frequency = torch.fft.rfftfreq(2 * samples, dt, dtype=traces.dtype, device=traces.device)
    response = torch.rsqrt(1 + (frequency / cutoff).pow(8))
    return torch.fft.irfft(torch.fft.rfft(traces, n=2 * samples) * response,
                           n=2 * samples)[..., :samples]


@dataclass
class Survey:
    """Shots and receivers on a grid, with the wavelet every shot fires."""

    spacing: float
    dt: float
    wavelet: Tensor            # (time,)
    sources: Tensor            # (shots, 2) int64 grid indices (z, x)
    receivers: Tensor          # (receivers, 2) int64 grid indices (z, x)
    peak_hz: float
    pml_width: int = 20
    max_velocity: float = 5000.0

    @property
    def shots(self) -> int:
        return len(self.sources)

    def simulate(self, velocity: Tensor) -> Tensor:
        """Receiver traces ``(shots, receivers, time)``, differentiable in velocity."""
        shots = self.shots
        return deepwave.scalar(
            velocity, self.spacing, self.dt,
            source_amplitudes=self.wavelet.to(velocity)[None, None].expand(shots, 1, -1),
            source_locations=self.sources[:, None].to(velocity.device),
            receiver_locations=self.receivers[None].expand(shots, -1, -1).to(velocity.device),
            accuracy=4, pml_width=self.pml_width, pml_freq=self.peak_hz,
            max_vel=self.max_velocity,
        )[-1]


def surface_survey(shape, spacing=10.0, dt=0.001, samples=800, peak_hz=12.0, shots=8,
                   depth_index=2, wavelet=None, scale=1, **options) -> Survey:
    """Evenly spaced shots and a receiver on every interior column, near the surface.

    ``scale`` places the same physical survey on a grid ``scale`` times finer.
    """
    width = (shape[1] - 1) // scale + 1
    columns = torch.linspace(4, width - 5, shots).round().long() * scale
    receiver_columns = torch.arange(1, width - 1) * scale
    depth = depth_index * scale
    if wavelet is None:
        wavelet = ricker(samples, dt, peak_hz)
    return Survey(
        spacing=spacing, dt=dt, wavelet=wavelet, peak_hz=peak_hz,
        sources=torch.stack([torch.full_like(columns, depth), columns], -1),
        receivers=torch.stack([torch.full_like(receiver_columns, depth), receiver_columns], -1),
        **options)
