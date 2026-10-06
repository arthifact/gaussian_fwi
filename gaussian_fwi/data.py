"""Synthetic observations that do not share the inversion's own simplifications.

Fitting traces made with the fitting solver, on the fitting grid, with the exact
wavelet and no noise is an inverse crime: almost any method looks strong. By
default the traces here are modelled on a finer grid, get band-limited noise,
and the inversion is handed a wavelet with the wrong peak frequency.
"""

from dataclasses import dataclass

import torch
from scipy.ndimage import zoom
from torch import Tensor

from .wave import Survey, lowpass, ricker, surface_survey


@dataclass(frozen=True)
class DataConfig:
    spacing: float = 10.0
    dt: float = 0.001
    samples: int = 800
    peak_hz: float = 12.0
    shots: int = 8
    refinement: int = 2               # modelling grid is this many times finer; 1 = same grid
    signal_to_noise: float | None = 10.0  # survey RMS ratio; None = noiseless
    wavelet_error: float = 0.08       # relative peak-frequency error of the inversion's wavelet
    seed: int = 0


def add_noise(traces: Tensor, dt: float, signal_to_noise: float, band_hz: float,
              seed: int) -> Tensor:
    """Add Gaussian noise, low-passed to the signal band, at a survey RMS ratio."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(traces.shape, generator=generator, dtype=traces.dtype)
    noise = lowpass(noise, dt, band_hz)
    return traces + noise * traces.square().mean().sqrt() / noise.square().mean().sqrt() \
        / signal_to_noise


def synthetic(velocity: Tensor, config: DataConfig = DataConfig()) -> tuple[Survey, Tensor]:
    """Return the survey the inversion sees and the traces it must explain."""
    velocity = velocity.float()
    r = config.refinement
    shape = tuple(velocity.shape)
    fine_shape = ((shape[0] - 1) * r + 1, (shape[1] - 1) * r + 1)
    fine = velocity if r == 1 else torch.from_numpy(zoom(
        velocity.numpy(), (fine_shape[0] / shape[0], fine_shape[1] / shape[1]), order=1))
    # Deepwave injects a source as amplitude * dt^2, so a finer time step alone
    # would scale recorded amplitudes by 1/r^2. Compensate.
    true_wavelet = ricker(config.samples * r, config.dt / r, config.peak_hz,
                          delay=1.1 / config.peak_hz) * r ** 2
    modelling = surface_survey(fine_shape, config.spacing / r, config.dt / r,
                               config.samples * r, config.peak_hz, config.shots,
                               wavelet=true_wavelet, scale=r, pml_width=20 * r)
    with torch.no_grad():
        traces = modelling.simulate(fine)[..., ::r].contiguous()
    if config.signal_to_noise is not None:
        traces = add_noise(traces, config.dt, config.signal_to_noise, 2.5 * config.peak_hz,
                           config.seed)
    estimated = ricker(config.samples, config.dt, config.peak_hz * (1 + config.wavelet_error),
                       delay=1.1 / config.peak_hz)
    survey = surface_survey(shape, config.spacing, config.dt, config.samples, config.peak_hz,
                            config.shots, wavelet=estimated)
    return survey, traces
