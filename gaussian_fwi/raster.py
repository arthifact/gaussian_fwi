"""Memory-bounded Gaussian evaluation with an exact first-order adjoint.

The kernel equals a Gaussian through five Mahalanobis standard deviations,
then smoothly vanishes at six. The neighbor search is conservative; it does
not define a discontinuous kernel. Only the dense reference forms all pairs.
"""

from collections.abc import Iterator

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


def kernel(q: Tensor, inner: float = 5.0, outer: float = 6.0) -> tuple[Tensor, Tensor]:
    """Return compact Gaussian and its derivative with respect to squared radius."""
    span = outer * outer - inner * inner
    t = ((q - inner * inner) / span).clamp(0.0, 1.0)
    w = (1 - t).pow(3) * (1 + 3 * t + 6 * t.square())
    dw = -30 * t.square() * (1 - t).square() / span
    e = torch.exp(-0.5 * q)
    return e * w, e * (dw - 0.5 * w)


def pair_chunks(
    points: np.ndarray,
    centers: np.ndarray,
    precision: np.ndarray,
    outer: float,
    max_pairs: int,
    tree: cKDTree | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (Gaussian ids, point ids) with at most max_pairs per chunk.

    A sphere enclosing each ellipsoid is used for the spatial query. Very wide
    Gaussians stream query points directly, so even one wide kernel cannot
    create an unbounded neighbor list. Tree construction runs on the CPU.
    """
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    tree = cKDTree(points) if tree is None else tree
    sym = (precision + precision.swapaxes(-1, -2)) * 0.5
    eigmin = np.linalg.eigvalsh(sym)[:, 0]
    if not np.all(np.isfinite(eigmin) & (eigmin > 0)):
        raise ValueError("Gaussian precision must be finite and positive definite")
    radii = outer / np.sqrt(eigmin) * (1 + 1e-6)
    for start in range(0, len(centers), 512):
        stop = min(start + 512, len(centers))
        counts = tree.query_ball_point(centers[start:stop], radii[start:stop], return_length=True)
        cursor = 0
        while cursor < len(counts):
            gi = start + cursor
            if counts[cursor] > max_pairs:
                for p0 in range(0, len(points), max_pairs):
                    pi = np.arange(p0, min(p0 + max_pairs, len(points)), dtype=np.int64)
                    yield np.full(len(pi), gi, dtype=np.int64), pi
                cursor += 1
                continue
            end, total = cursor, 0
            while end < len(counts) and total + counts[end] <= max_pairs:
                total += counts[end]
                end += 1
            if total:
                lists = tree.query_ball_point(
                    centers[gi : start + end], radii[gi : start + end], return_sorted=True
                )
                ids = np.repeat(np.arange(gi, start + end), counts[cursor:end])
                yield ids, np.concatenate(lists).astype(np.int64, copy=False)
            cursor = end


def _numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


class _SparseSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, centers, precision, amplitudes, inner, outer, max_pairs):
        ctx.save_for_backward(points, centers, precision, amplitudes)
        ctx.inner, ctx.outer, ctx.max_pairs = inner, outer, max_pairs
        pp, cc, pr = _numpy(points), _numpy(centers), _numpy(precision)
        ctx.tree = cKDTree(pp)
        out = points.new_zeros(len(points))
        sym = (precision + precision.transpose(-1, -2)) * 0.5
        for gg, xx in pair_chunks(pp, cc, pr, outer, max_pairs, ctx.tree):
            g = torch.as_tensor(gg, device=points.device)
            x = torch.as_tensor(xx, device=points.device)
            delta = points[x] - centers[g]
            pd = torch.bmm(sym[g], delta.unsqueeze(-1)).squeeze(-1)
            value, _ = kernel((delta * pd).sum(-1), inner, outer)
            out.index_add_(0, x, amplitudes[g] * value)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        points, centers, precision, amplitudes = ctx.saved_tensors
        pp, cc, pr = _numpy(points), _numpy(centers), _numpy(precision)
        gp = torch.zeros_like(points) if ctx.needs_input_grad[0] else None
        gc, gpr, ga = (
            torch.zeros_like(centers),
            torch.zeros_like(precision),
            torch.zeros_like(amplitudes),
        )
        sym = (precision + precision.transpose(-1, -2)) * 0.5
        for gg, xx in pair_chunks(pp, cc, pr, ctx.outer, ctx.max_pairs, ctx.tree):
            g = torch.as_tensor(gg, device=points.device)
            x = torch.as_tensor(xx, device=points.device)
            delta = points[x] - centers[g]
            pd = torch.bmm(sym[g], delta.unsqueeze(-1)).squeeze(-1)
            value, derivative = kernel((delta * pd).sum(-1), ctx.inner, ctx.outer)
            h = upstream[x]
            factor = h * amplitudes[g] * derivative
            ddelta = (2 * factor).unsqueeze(-1) * pd
            ga.index_add_(0, g, h * value)
            gc.index_add_(0, g, -ddelta)
            gpr.index_add_(0, g, factor[:, None, None] * delta[:, :, None] * delta[:, None, :])
            if gp is not None:
                gp.index_add_(0, x, ddelta)
        return gp, gc, gpr, ga, None, None, None


def sparse_sum(
    points: Tensor,
    centers: Tensor,
    precision: Tensor,
    amplitudes: Tensor,
    inner: float = 5.0,
    outer: float = 6.0,
    max_pairs: int = 131072,
) -> Tensor:
    if not (0 < inner < outer):
        raise ValueError("Require 0 < inner < outer")
    return _SparseSum.apply(points, centers, precision, amplitudes, inner, outer, max_pairs)


def dense_sum(
    points: Tensor,
    centers: Tensor,
    precision: Tensor,
    amplitudes: Tensor,
    inner: float = 5.0,
    outer: float = 6.0,
    pure_gaussian: bool = False,
    chunk_size: int = 256,
) -> Tensor:
    """Small-problem reference using ordinary PyTorch autograd; supports Hessians."""
    out = points.new_zeros(len(points))
    sym = (precision + precision.transpose(-1, -2)) * 0.5
    for start in range(0, len(centers), chunk_size):
        delta = points[None] - centers[start : start + chunk_size, None]
        q = torch.einsum("gpd,gde,gpe->gp", delta, sym[start : start + chunk_size], delta)
        value = torch.exp(-0.5 * q) if pure_gaussian else kernel(q, inner, outer)[0]
        out = out + (amplitudes[start : start + chunk_size, None] * value).sum(0)
    return out
