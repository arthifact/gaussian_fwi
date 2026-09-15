"""Differentiable spatial regularization evaluated on the physical velocity field."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Regularization:
    """Dimensionless isotropic smooth-TV and first-order Tikhonov penalties.

    Spatial derivatives are scaled by ``length_scale / velocity_scale``.
    Weights are experiment choices and should be selected independently for
    each baseline. A weighted sum of these penalties is not a decomposition
    into separate smooth and blocky models.
    """

    tv_weight: float = 0.0
    tikhonov_weight: float = 0.0
    epsilon: float = 0.1
    velocity_scale: float = 100.0
    length_scale: float = 10.0

    def __post_init__(self) -> None:
        for name in ("tv_weight", "tikhonov_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("epsilon", "velocity_scale", "length_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def __call__(self, velocity: Tensor, spacing: float) -> Tensor:
        """Return the weighted penalty with zero outward differences at boundaries."""
        if not self.tv_weight and not self.tikhonov_weight:
            return velocity.sum() * 0
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        squared_norm = torch.zeros_like(velocity)
        for axis in range(velocity.ndim):
            difference = torch.diff(velocity, dim=axis)
            padding = [0] * (2 * velocity.ndim)
            padding[2 * (velocity.ndim - axis - 1) + 1] = 1
            gradient = (
                F.pad(difference, padding) * self.length_scale / (spacing * self.velocity_scale)
            )
            squared_norm = squared_norm + gradient.square()
        tv = (torch.sqrt(squared_norm + self.epsilon**2) - self.epsilon).mean()
        smoothness = 0.5 * squared_norm.mean()
        return self.tv_weight * tv + self.tikhonov_weight * smoothness
