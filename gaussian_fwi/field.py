"""Velocity models: a fixed starting model plus Gaussians, or plus free pixels.

A Gaussian ``k`` adds ``amplitude_k * exp(-q/2)`` to the starting model, where
``q`` is the squared distance from its centre measured in its own rotated
widths. Every Gaussian has six parameters:

=============  =====  ==============================================
``center``     (2,)   x, z of the centre, metres
``log_width``  (2,)   log of the widths along and across its axis, metres
``angle``      ()     rotation of its long axis from horizontal, radians
``amplitude``  ()     signed velocity change at the centre, m/s
=============  =====  ==============================================

Both models apply the same soft bound, so velocity stays physical while
remaining differentiable.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PARAMETERS = ("center", "log_width", "angle", "amplitude")


def linear_start(shape, top=1500.0, bottom=3000.0) -> Tensor:
    """Velocity increasing linearly with depth: the starting model for every fit."""
    depth = torch.linspace(0, 1, shape[0])
    return (top + (bottom - top) * depth)[:, None].expand(*shape).clone()


def soft_bound(velocity: Tensor, low: float, high: float, softness: float = 20.0) -> Tensor:
    return (velocity + softness * F.softplus((low - velocity) / softness)
            - softness * F.softplus((velocity - high) / softness))


def gaussian_kernels(points: Tensor, center: Tensor, log_width: Tensor, angle: Tensor) -> Tensor:
    """Unit-peak Gaussian values ``(count, points)`` at ``(points, 2)`` x, z positions."""
    offset = points[None] - center[:, None]                                # (K, N, 2)
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    along = offset[..., 0] * cos + offset[..., 1] * sin
    across = -offset[..., 0] * sin + offset[..., 1] * cos
    width = log_width.exp()
    return torch.exp(-0.5 * ((along / width[:, :1]).square() + (across / width[:, 1:]).square()))


class GaussianField(nn.Module):
    """Starting model plus a population of anisotropic Gaussians.

    The population starts on a ``lattice x lattice`` grid with zero amplitude.
    Its size changes only through :mod:`gaussian_fwi.adapt`.
    """

    def __init__(self, start: Tensor, spacing: float, lattice: int = 8,
                 width_ratio: float = 0.65, bounds=(1400.0, 5000.0)):
        super().__init__()
        depth, width = start.shape
        self.spacing, self.bounds = float(spacing), tuple(bounds)
        self.extent = ((width - 1) * spacing, (depth - 1) * spacing)          # x, z
        self.min_width, self.max_width = 0.5 * spacing, max(self.extent)
        z, x = torch.meshgrid(torch.arange(depth) * spacing, torch.arange(width) * spacing,
                              indexing="ij")
        self.register_buffer("start", start.float())
        self.register_buffer("points", torch.stack([x, z], -1).reshape(-1, 2).float())
        axes = [torch.linspace(0, e, lattice) for e in self.extent]
        cz, cx = torch.meshgrid(axes[1], axes[0], indexing="ij")
        center = torch.stack([cx, cz], -1).reshape(-1, 2)
        widths = torch.tensor([width_ratio * e / (lattice - 1) for e in self.extent])
        self.replace(center=center, log_width=widths.log().expand(len(center), 2),
                     angle=torch.zeros(len(center)), amplitude=torch.zeros(len(center)))
        self.register_buffer("age", torch.zeros(len(center), dtype=torch.long))

    @property
    def count(self) -> int:
        return len(self.amplitude)

    def replace(self, **values: Tensor) -> None:
        """Install new parameter tensors (all four, same population size)."""
        for name in PARAMETERS:
            setattr(self, name, nn.Parameter(values[name].detach().clone().float()))

    def kernels(self, points: Tensor | None = None) -> Tensor:
        """Unit-peak Gaussian values ``(count, points)``."""
        points = self.points if points is None else points
        return gaussian_kernels(points, self.center, self.log_width, self.angle)

    def forward(self) -> Tensor:
        velocity = self.start.reshape(-1) + self.amplitude @ self.kernels()
        return soft_bound(velocity, *self.bounds).reshape(self.start.shape)

    def parameter_groups(self, amplitude=10.0, center=2.0, width=0.01, angle=0.01):
        """Adam groups; learning rates are per-step sizes in each parameter's units."""
        return [{"params": [self.center], "lr": center, "name": "center"},
                {"params": [self.log_width], "lr": width, "name": "log_width"},
                {"params": [self.angle], "lr": angle, "name": "angle"},
                {"params": [self.amplitude], "lr": amplitude, "name": "amplitude"}]

    @torch.no_grad()
    def project(self) -> None:
        """Keep widths, centres and angles in their admissible ranges."""
        self.log_width.clamp_(math.log(self.min_width), math.log(self.max_width))
        extent = self.center.new_tensor(self.extent)
        self.center.copy_(torch.minimum(torch.maximum(self.center, -0.2 * extent), 1.2 * extent))
        # A Gaussian is unchanged by a half turn; keep angles in (-pi/2, pi/2].
        self.angle.copy_(torch.remainder(self.angle + math.pi / 2, math.pi) - math.pi / 2)
        self.age += 1


class PixelField(nn.Module):
    """Starting model plus one free velocity change per grid cell: the reference method."""

    def __init__(self, start: Tensor, spacing: float, bounds=(1400.0, 5000.0)):
        super().__init__()
        self.spacing, self.bounds = float(spacing), tuple(bounds)
        self.register_buffer("start", start.float())
        self.change = nn.Parameter(torch.zeros_like(self.start))

    @property
    def count(self) -> int:
        return self.change.numel()

    def forward(self) -> Tensor:
        return soft_bound(self.start + self.change, *self.bounds)

    def parameter_groups(self, change=10.0):
        return [{"params": [self.change], "lr": change, "name": "change"}]

    def project(self) -> None:
        pass
