"""A pull toward the starting model that acts where the waves carry little information.

Held-out traces cannot tell whether velocity is right where the survey barely
senses it, so those regions need a prior. A uniform pull also acts where the
data are informative and costs detail there. Here its strength follows the
survey's illumination instead.

Illumination is the square root of the Gauss-Newton Hessian diagonal at the
starting model, estimated by backpropagating random residuals (each probe
gives ``J^T r``, and the mean of its square estimates ``diag(J^T J)``). It
depends on the geometry, the start and the misfit's band and time gain, not on
the recorded traces, so noise in them cannot shape it. The estimate is smoothed,
and normalized by its peak below the acquisition row, whose own sensitivity
spike would otherwise set the scale. The pull's weight in each cell is
``(1 - illumination)^2``: near zero where the survey sees well, approaching one
where it does not.
"""

import torch
from torch import Tensor

from .wave import Survey, lowpass


def smooth(image: Tensor, width_cells: float) -> Tensor:
    """Separable Gaussian blur with reflected edges."""
    radius = max(1, int(3 * width_cells))
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (offsets / width_cells).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    out = image[None, None]
    for dim in (2, 3):
        moved = out.transpose(dim, 3)
        shape = moved.shape
        flat = torch.nn.functional.pad(moved.reshape(-1, 1, shape[-1]), (radius, radius),
                                       mode="reflect")
        moved = torch.nn.functional.conv1d(flat, kernel).reshape(shape)
        out = moved.transpose(dim, 3)
    return out[0, 0]


def illumination(survey: Survey, start: Tensor, *, cutoff: float = 12.0, probes: int = 4,
                 time_gain: float = 1.5, smooth_m: float = 50.0, seed: int = 0) -> Tensor:
    """Relative sensitivity of the recorded data to each cell, between 0 and 1."""
    generator = torch.Generator().manual_seed(seed)
    gain = torch.linspace(0, 1, survey.wavelet.shape[-1]).pow(time_gain)
    total = torch.zeros_like(start)
    for _ in range(probes):
        velocity = start.clone().requires_grad_()
        predicted = survey.simulate(velocity)
        residual = lowpass(torch.randn(predicted.shape, generator=generator), survey.dt,
                           cutoff) * gain
        (lowpass(predicted, survey.dt, cutoff) * gain * residual).sum().backward()
        total += velocity.grad.square()
    sensitivity = smooth((total / probes).sqrt(), smooth_m / survey.spacing)
    below = int(survey.receivers[:, 0].max()) + int(round(smooth_m / survey.spacing)) + 1
    return (sensitivity / sensitivity[below:].max()).clamp(max=1.0)


def prior_weights(survey: Survey, start: Tensor, **options) -> Tensor:
    """Per-cell strength of the pull toward the start: ``(1 - illumination)^2``."""
    return (1 - illumination(survey, start, **options)).square()


def prior(velocity: Tensor, start: Tensor, weights: Tensor | None = None,
          scale: float = 500.0) -> Tensor:
    """Weighted mean squared departure from the start, in units of ``scale`` m/s.

    Its overall weight is a stated assumption: held-out traces cannot choose it,
    because the regions it acts on barely change the traces.
    """
    departure = ((velocity - start) / scale).square()
    return (departure if weights is None else weights * departure).mean()
