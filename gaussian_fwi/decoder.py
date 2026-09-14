"""Functional scalar-velocity decoder for tensors produced by upstream networks."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .raster import dense_sum, sparse_sum


def decode(
    points: Tensor,
    centers: Tensor,
    covariance_factors: Tensor,
    coefficients: Tensor,
    background: Tensor,
    *,
    bounds: tuple[float, float] | None = (1500.0, 4500.0),
    soft_clip: float = 20.0,
    backend: str = "sparse",
    max_pairs: int = 131072,
) -> Tensor:
    """Decode velocity at physical ``(x, [y,] z)`` points, retaining input gradients.

    Shapes are ``(P, D)``, ``(G, D)``, ``(G, D, D)``, ``(G,)`` and either
    ``()`` or ``(P,)`` respectively. Factors are lower triangular Cholesky
    factors in meters with positive diagonals: covariance is ``L @ L.T``.
    Coefficients and sampled background are in m/s. For a depth background,
    the caller supplies ``top + (bottom-top) * points[:, -1] / depth_extent``.

    No tensor is copied into a Parameter, cast or detached. The compact taper,
    signed unnormalized kernels and optional soft bounds match GaussianField.
    ``bounds=None`` returns raw velocity. Sparse evaluation supports first
    derivatives, including upstream network weights; dense is a small oracle.
    Empty populations and query sets are supported. The caller controls grid
    sampling and numerical/physical width limits; this function never projects.
    """
    tensors = (points, centers, covariance_factors, coefficients, background)
    if any(not isinstance(t, Tensor) for t in tensors):
        raise TypeError("Decoder inputs must be tensors")
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Points must have shape (P, 2) or (P, 3)")
    d = points.shape[1]
    if (centers.ndim != 2 or centers.shape[1] != d
            or covariance_factors.shape != (centers.shape[0], d, d)
            or coefficients.shape != (centers.shape[0],)
            or background.shape not in (torch.Size([]), torch.Size([len(points)]))):
        raise ValueError("Incompatible decoder tensor shapes")
    if points.dtype not in (torch.float32, torch.float64) or any(
        t.dtype != points.dtype or t.device != points.device for t in tensors
    ):
        raise ValueError("Decoder tensors must share a float32/64 dtype and device")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Decoder tensors must be finite")
    if (bool((covariance_factors.triu(1) != 0).any()) or
            bool((covariance_factors.diagonal(dim1=-2, dim2=-1) <= 0).any())):
        raise ValueError("Covariance factors must be lower triangular with positive diagonals")
    if backend not in ("sparse", "dense") or type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("Invalid decoder backend or pair budget")
    if not math.isfinite(soft_clip) or soft_clip <= 0:
        raise ValueError("soft_clip must be finite and positive")
    if bounds is not None and (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                               or not 0 < bounds[0] < bounds[1]):
        raise ValueError("Velocity bounds must be finite, positive and increasing")
    eye = torch.eye(d, dtype=points.dtype, device=points.device).expand_as(covariance_factors)
    inverse = torch.linalg.solve_triangular(covariance_factors, eye, upper=False)
    precision = inverse.transpose(-1, -2) @ inverse
    if not torch.isfinite(precision).all():
        raise ValueError("Covariance factors exceed the finite numerical range")
    evaluator = sparse_sum if backend == "sparse" else dense_sum
    kwargs = {"max_pairs": max_pairs} if backend == "sparse" else {}
    value = background + evaluator(points, centers, precision, coefficients, **kwargs)
    if bounds is not None:
        lo, hi = bounds
        value = value + soft_clip * F.softplus((lo - value) / soft_clip) - soft_clip * F.softplus(
            (value - hi) / soft_clip
        )
    return value
