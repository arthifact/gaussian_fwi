"""Explicit sampling limits and diagnostics for continuous Gaussian fields.

The optional width floor bounds the *ideal individual Gaussian's* normalized
Fourier response at the grid Nyquist frequency. It does not certify alias-free
sampling of the tapered, summed, bounded field or seismic resolving power.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from gaussian_fwi.core.geometry import GridSpec

if TYPE_CHECKING:
    from .field import GaussianField


@dataclass(frozen=True)
class SamplingConfig:
    """Numerical Gaussian bandwidth limit, independent of acquisition frequency.

    For standard deviation sigma, the ideal normalized Fourier response at
    angular frequency pi / spacing is exp(-sigma² pi² / (2 spacing²)). The
    smallest principal width bounds this response in every direction. The
    tolerance is an explicit numerical choice, not a universal optimal width.
    """

    max_nyquist_response: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_nyquist_response) or not 0 < self.max_nyquist_response < 1:
            raise ValueError("Nyquist-response tolerance must lie strictly between zero and one")

    def minimum_width(self, spacing: float) -> float:
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Sampling requires finite positive grid spacing")
        width = spacing * math.sqrt(-2 * math.log(self.max_nyquist_response)) / math.pi
        if not math.isfinite(width):
            raise ValueError("Sampling width exceeds the finite numerical range")
        return width


def sample_on_grid(field: "GaussianField", grid: GridSpec) -> torch.Tensor:
    """Evaluate an unchanged physical field on another grid of the same domain."""
    if (
        not isinstance(grid, GridSpec)
        or grid.ndim != field.grid.ndim
        or any(
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
            for a, b in zip(grid.extent, field.grid.extent)
        )
    ):
        raise ValueError("Sampling grids must cover the same physical domain")
    points = grid.points(dtype=field.background.dtype, device=field.background.device)
    return field(points).reshape(grid.shape)


@torch.no_grad()
def sampling_diagnostics(field: "GaussianField", *, refinement_factor: int = 2) -> dict:
    """Measure subgrid interpolation discrepancy without a reference Earth model.

    Compare the actual continuous decoder with multilinear interpolation of
    its coarse samples. This is a field-sampling diagnostic, not a waveform
    error estimate; propagation convergence must be checked separately.
    """
    if type(refinement_factor) is not int or refinement_factor < 2:
        raise ValueError("Sampling refinement factor must be an integer of at least two")
    grid = GridSpec(
        tuple((n - 1) * refinement_factor + 1 for n in field.grid.shape),
        field.grid.spacing / refinement_factor,
    )
    coarse = field()
    fine = sample_on_grid(field, grid)
    mode = "bilinear" if grid.ndim == 2 else "trilinear"
    interpolated = F.interpolate(
        coarse[None, None], size=grid.shape, mode=mode, align_corners=True
    )[0, 0]
    difference = (fine - interpolated).double()
    widths = [
        torch.linalg.svdvals(block.cholesky(dtype=torch.float64)).min() for block in field.blocks
    ]
    minimum = min(map(float, widths)) if widths else None
    if not torch.isfinite(difference).all() or (minimum is not None and not math.isfinite(minimum)):
        raise FloatingPointError("Nonfinite field-sampling diagnostic")
    return {
        "refinement_factor": refinement_factor,
        "coarse_shape": list(field.grid.shape),
        "fine_shape": list(grid.shape),
        "minimum_principal_width_m": minimum,
        "maximum_ideal_kernel_nyquist_response": (
            math.exp(-0.5 * (math.pi * minimum / field.grid.spacing) ** 2)
            if minimum is not None
            else None
        ),
        "interpolation_rmse_m_s": float(difference.square().mean().sqrt()),
        "interpolation_maximum_error_m_s": float(difference.abs().max()),
        "interpretation": "Continuous-field sampling diagnostic; not seismic resolution or waveform error",
    }
