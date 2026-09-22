"""Differentiable spatial regularization evaluated on the physical velocity field."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Regularization:
    """Dimensionless isotropic smooth-TV, Tikhonov and second-order TGV penalties.

    Spatial derivatives are scaled by ``length_scale / velocity_scale``.
    Weights are experiment choices and should be selected independently for
    each baseline. A weighted sum of these penalties is not a decomposition
    into separate smooth and blocky models.

    TV prefers piecewise-constant velocity, so it penalizes a depth trend as
    heavily as a fault. ``tgv_weight`` selects second-order total generalized
    variation instead, which prefers piecewise-affine velocity: an exactly
    linear field costs nothing, while discontinuities still do.

    TGV's inner minimization is warm-started at the velocity gradient, where the
    fidelity term is exactly zero. Too few inner steps leave it there, and the
    penalty then contributes almost no gradient: at five steps it is some sixty
    times weaker than TV on a rough field. Check convergence on a field like the
    one being inverted, not on a smooth one, where the warm start is already
    optimal and any number of steps looks converged.
    """

    tv_weight: float = 0.0
    tikhonov_weight: float = 0.0
    epsilon: float = 0.1
    velocity_scale: float = 100.0
    length_scale: float = 10.0
    tgv_weight: float = 0.0
    tgv_ratio: float = 2.0
    tgv_steps: int = 500
    tgv_step_size: float = 2.0

    def __post_init__(self) -> None:
        for name in ("tv_weight", "tikhonov_weight", "tgv_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("epsilon", "velocity_scale", "length_scale", "tgv_ratio", "tgv_step_size"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.tgv_steps) is not int or self.tgv_steps < 1:
            raise ValueError("tgv_steps must be a positive integer")

    def _difference(self, field: Tensor, axis: int) -> Tensor:
        """Forward difference with a zero outward difference at the far edge.

        Callers apply their own scaling. The accepted penalty multiplies before
        dividing, and floating point is not associative, so that order is kept
        wherever it affects a published result.
        """
        difference = torch.diff(field, dim=axis)
        padding = [0] * (2 * field.ndim)
        padding[2 * (field.ndim - axis - 1) + 1] = 1
        return F.pad(difference, padding)

    def _gradient(self, velocity: Tensor, spacing: float) -> Tensor:
        """Dimensionless velocity gradient, stacked over the spatial axes."""
        return torch.stack([
            self._difference(velocity, axis) * self.length_scale
            / (spacing * self.velocity_scale)
            for axis in range(velocity.ndim)
        ])

    def _huber(self, stacked: Tensor, over: int) -> Tensor:
        """Smoothed magnitude of the leading ``over`` component axes, averaged."""
        squared = stacked.square().sum(dim=tuple(range(over)))
        return (torch.sqrt(squared + self.epsilon**2) - self.epsilon).mean()

    def _tgv_terms(self, gradient: Tensor, auxiliary: Tensor, spacing: float):
        """The two TGV terms: gradient minus the field, and its symmetrized derivative."""
        rows = [torch.stack([self._difference(auxiliary[j], axis) * self.length_scale / spacing
                             for j in range(auxiliary.shape[0])])
                for axis in range(auxiliary.shape[0])]
        jacobian = torch.stack(rows)
        symmetrized = 0.5 * (jacobian + jacobian.transpose(0, 1))
        return (self.tgv_ratio * self._huber(gradient - auxiliary, 1),
                self._huber(symmetrized, 2))

    def _tgv(self, velocity: Tensor, spacing: float) -> Tensor:
        """Second-order TGV, minimized over its auxiliary field.

        The inner minimum is found by a short gradient descent warm-started at
        the velocity gradient itself. By Danskin's theorem the derivative with
        respect to velocity is evaluated at that minimizer with it held fixed,
        so the inner iterations need no graph of their own.
        """
        gradient = self._gradient(velocity, spacing)
        frozen = gradient.detach()
        # The caller may be inside no_grad: the inversion evaluates this penalty
        # with gradients disabled at terminal steps and for validation. The inner
        # descent needs its own graph regardless, so enable it explicitly here.
        with torch.enable_grad():
            auxiliary = frozen.clone().requires_grad_(True)
            for _ in range(self.tgv_steps):
                fidelity, curvature = self._tgv_terms(frozen, auxiliary, spacing)
                step, = torch.autograd.grad(fidelity + curvature, auxiliary)
                auxiliary = (auxiliary - self.tgv_step_size * step).detach().requires_grad_(True)
        fidelity, curvature = self._tgv_terms(gradient, auxiliary.detach(), spacing)
        return fidelity + curvature

    def __call__(self, velocity: Tensor, spacing: float) -> Tensor:
        """Return the weighted penalty with zero outward differences at boundaries."""
        if not self.tv_weight and not self.tikhonov_weight and not self.tgv_weight:
            return velocity.sum() * 0
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        total = None
        if self.tv_weight or self.tikhonov_weight:
            # Accumulated in this order so the accepted penalty is reproduced
            # to the last bit; reassociating it moves the result by one ulp.
            squared_norm = torch.zeros_like(velocity)
            for axis in range(velocity.ndim):
                gradient = (self._difference(velocity, axis) * self.length_scale
                            / (spacing * self.velocity_scale))
                squared_norm = squared_norm + gradient.square()
            tv = (torch.sqrt(squared_norm + self.epsilon**2) - self.epsilon).mean()
            smoothness = 0.5 * squared_norm.mean()
            total = self.tv_weight * tv + self.tikhonov_weight * smoothness
        if self.tgv_weight:
            tgv = self.tgv_weight * self._tgv(velocity, spacing)
            total = tgv if total is None else total + tgv
        return total
