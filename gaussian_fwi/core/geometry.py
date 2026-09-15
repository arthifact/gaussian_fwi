"""Grid geometry and conventions shared by both velocity parameterizations."""

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor


@dataclass(frozen=True)
class GridSpec:
    """Uniform Cartesian grid with tensor axes ``(z, [y,] x)``.

    Parameters
    ----------
    shape : tuple of int
        Number of samples along each tensor axis. Each axis needs two samples.
    spacing : float
        Sample spacing in meters. The origin is zero on every axis.

    Notes
    -----
    Continuous query points use physical coordinate order ``(x, [y,] z)``.
    Acquisition indices follow tensor order instead.
    """

    shape: tuple[int, ...]
    spacing: float = 10.0

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) not in (2, 3) or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 2 for n in shape
        ):
            raise ValueError("shape must contain two or three integers, each at least two")
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))
        object.__setattr__(self, "spacing", float(self.spacing))

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def extent(self) -> tuple[float, ...]:
        """Maximum sample coordinates in physical coordinate order, in meters."""
        return tuple((n - 1) * self.spacing for n in reversed(self.shape))

    def points(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return flattened sample coordinates with shape ``(prod(shape), ndim)``."""
        axes = [torch.arange(n, dtype=dtype, device=device) * self.spacing for n in self.shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, self.ndim)
