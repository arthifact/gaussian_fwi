"""Training-adjoint directions for finite, moment-preserving split proposals.

The matrix below contracts the raw-field adjoint with the center Hessian of
the actual tapered kernel. It suggests directions; it is neither the waveform
Hessian nor a Fisher matrix. Covariance-preserving splitting cancels its
second-order field term, so acceptance must evaluate the finite proposal.
"""

import math

import torch
from torch import Tensor

from .raster import kernel


def kernel_second_derivative(q: Tensor) -> Tensor:
    """Return d²K/dq² for the production C2 taper, including both junctions."""
    span = 6.0**2 - 5.0**2
    t = ((q - 5.0**2) / span).clamp(0, 1)
    second_window = -60 * t * (1 - t) * (1 - 2 * t) / span**2
    value, first = kernel(q)
    return torch.exp(-0.5 * q) * second_window - first - 0.25 * value


@torch.no_grad()
def center_curvature(
    points: Tensor,
    gradient: Tensor,
    center: Tensor,
    covariance: Tensor,
    amplitude: Tensor,
    *,
    chunk_size: int = 131072,
) -> Tensor:
    """Contract a sampled raw-field adjoint with one kernel's center Hessian."""
    d = center.numel()
    if (
        center.shape != (d,)
        or d not in (2, 3)
        or points.ndim != 2
        or points.shape[1] != d
        or gradient.shape != (len(points),)
        or covariance.shape != (d, d)
        or amplitude.numel() != 1
        or type(chunk_size) is not int
        or chunk_size < 1
    ):
        raise ValueError("Invalid geometry, adjoint, or chunk size for split directions")
    tensors = (points, gradient, center, covariance, amplitude)
    if any(t.device != center.device for t in tensors):
        raise ValueError("Split-direction tensors must share one device")
    if not all(torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError("Nonfinite split-direction input")
    if not torch.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-12):
        raise ValueError("Split-direction covariance must be symmetric")
    lower = torch.linalg.cholesky(covariance.double())
    precision = torch.cholesky_inverse(lower)
    result = torch.zeros((d, d), dtype=torch.float64, device=center.device)
    for start in range(0, len(points), chunk_size):
        r = points[start : start + chunk_size].double() - center.double()
        pr = r @ precision
        q = (r * pr).sum(-1)
        _, first = kernel(q)
        second = kernel_second_derivative(q)
        weight = gradient[start : start + chunk_size].double() * amplitude.double()
        result += pr.T @ ((4 * weight * second)[:, None] * pr)
        result += (2 * weight * first).sum() * precision
    result = (result + result.T) / 2
    if not torch.isfinite(result).all():
        raise FloatingPointError("Split-direction matrix overflow")
    return result


@torch.no_grad()
def adjoint_directions(
    points: Tensor,
    gradient: Tensor,
    center: Tensor,
    covariance: Tensor,
    amplitude: Tensor,
    *,
    chunk_size: int = 131072,
) -> Tensor:
    """Return physical directions ordered by curvature in whitened coordinates.

    Whitening makes direction length relative to the parent ellipsoid. Matrix
    eigenvectors are unique only for distinct eigenvalues; signs are fixed for
    reproducible child ordering on the tested numerical runtime.
    """
    matrix = center_curvature(
        points, gradient, center, covariance, amplitude, chunk_size=chunk_size
    )
    lower = torch.linalg.cholesky(covariance.double())
    whitened = lower.T @ matrix @ lower
    _, vectors = torch.linalg.eigh((whitened + whitened.T) / 2)
    directions = (lower @ vectors).T
    for direction in directions:
        if direction[direction.abs().argmax()] < 0:
            direction.neg_()
    return directions.to(center)


@torch.no_grad()
def directional_children(
    center: Tensor,
    covariance: Tensor,
    amplitude: Tensor,
    direction: Tensor,
    fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Preserve ideal signed moments along an arbitrary physical direction.

    With d normalized so d.T @ inv(covariance) @ d = fraction², the children
    have centers center ± d and covariance covariance - d d.T. The determinant
    lemma gives the same signed amplitude allocation as a principal-axis split.
    """
    dim = center.numel()
    if (
        dim not in (2, 3)
        or center.shape != (dim,)
        or covariance.shape != (dim, dim)
        or direction.shape != (dim,)
        or amplitude.numel() != 1
        or not math.isfinite(fraction)
        or not 0 < fraction < 1
    ):
        raise ValueError("Invalid directional split geometry or fraction")
    if not all(torch.isfinite(t).all() for t in (center, covariance, amplitude, direction)):
        raise FloatingPointError("Nonfinite directional split")
    if not torch.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-12):
        raise ValueError("Split covariance must be symmetric")
    lower = torch.linalg.cholesky(covariance.double())
    direction = direction.to(center.device, torch.float64)
    # Normalize before solving to avoid overflow from an arbitrary input norm.
    largest = direction.abs().max()
    if not largest > 0:
        raise ValueError("Split direction must be nonzero")
    direction = direction / largest
    length = torch.linalg.vector_norm(
        torch.linalg.solve_triangular(lower, direction[:, None], upper=False)
    )
    offset = fraction * direction / length
    matrix = covariance.double() - torch.outer(offset, offset)
    value = amplitude.double().reshape(()) / (2 * math.sqrt(1 - fraction**2))
    if not torch.isfinite(matrix).all() or not torch.isfinite(value):
        raise FloatingPointError("Directional split overflow")
    torch.linalg.cholesky(matrix)
    return (
        torch.stack((center.double() - offset, center.double() + offset)),
        matrix.expand(2, dim, dim).clone(),
        value.expand(2).clone(),
    )
