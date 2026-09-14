"""Continuous Gaussian velocity fields in physical coordinates (x, [y,] z)."""

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fwi_core.checkpoint import save_torch
from fwi_core.geometry import GridSpec

from .decoder import decode
from .raster import dense_sum, sparse_sum
from .sampling import SamplingConfig


class GaussianBlock(nn.Module):
    """A group of mobile Gaussian kernels with signed peak amplitudes in m/s.

    The lower covariance factor is ``diag(exp(log_scales)) @ (I + shears)``.
    Positive diagonal entries provide a unique Cholesky parameterization of
    full positive definite covariance without rotational gauge parameters.
    """

    def __init__(self, centers: Tensor, scales: Tensor) -> None:
        super().__init__()
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] not in (2, 3):
            raise ValueError("A Gaussian block must contain at least one 2D or 3D center")
        n, d = centers.shape
        if (
            scales.shape != centers.shape
            or centers.dtype not in (torch.float32, torch.float64)
            or scales.dtype != centers.dtype
            or scales.device != centers.device
            or not torch.isfinite(centers).all()
            or not torch.isfinite(scales).all()
            or not torch.all(scales > 0)
            or not torch.isfinite(scales.mean())
        ):
            raise ValueError(
                "Centers and positive scales must be finite, matching float32/64 tensors"
            )
        self.centers = nn.Parameter(centers.clone())
        self.log_scales = nn.Parameter(scales.log().clone())
        self.shears = nn.Parameter(centers.new_zeros((n, d * (d - 1) // 2)))
        self.amplitudes = nn.Parameter(centers.new_zeros(n))
        self.nominal_scale = float(scales.mean())
        self.amplitude_lr_scale = 1.0

    def cholesky(self, dtype: torch.dtype | None = None) -> Tensor:
        """Return lower covariance factors with shape ``(count, ndim, ndim)``."""
        d = self.centers.shape[-1]
        dtype = self.centers.dtype if dtype is None else dtype
        lower = (
            torch.eye(d, dtype=dtype, device=self.centers.device)
            .expand(len(self.centers), d, d)
            .clone()
        )
        row, col = torch.tril_indices(d, d, offset=-1, device=self.centers.device)
        lower[:, row, col] = self.shears.to(dtype)
        return self.log_scales.to(dtype).exp().unsqueeze(-1) * lower

    def covariance(self) -> Tensor:
        lower = self.cholesky()
        return lower @ lower.transpose(-1, -2)

    def precision(self) -> Tensor:
        lower = self.cholesky()
        eye = torch.eye(lower.shape[-1], dtype=lower.dtype, device=lower.device).expand_as(lower)
        inv = torch.linalg.solve_triangular(lower, eye, upper=False)
        return inv.transpose(-1, -2) @ inv

    def parameter_groups(
        self, amplitude_lr: float = 4.0, geometry_lr: float = 0.008, center_lr_ratio: float = 0.02
    ) -> list[dict[str, Any]]:
        """Return separate Adam groups for amplitudes, centers, and covariance."""
        _check_learning_rates(amplitude_lr, geometry_lr, center_lr_ratio)
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.nominal_scale, self.amplitude_lr_scale)
        ):
            raise ValueError("Gaussian optimization scales must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                amplitude_lr * self.amplitude_lr_scale,
                center_lr_ratio * self.nominal_scale,
            )
        ):
            raise ValueError("Gaussian learning rates exceed the finite numerical range")
        return [
            {
                "params": [self.amplitudes],
                "lr": amplitude_lr * self.amplitude_lr_scale,
                "role": "amplitude",
            },
            {
                "params": [self.centers],
                "lr": center_lr_ratio * self.nominal_scale,
                "role": "center",
            },
            {"params": [self.log_scales, self.shears], "lr": geometry_lr, "role": "covariance"},
        ]


class GaussianField(nn.Module):
    """Continuous bounded velocity from a depth background and Gaussian blocks.

    Kernel amplitudes are signed, unnormalized velocity perturbations. Physical
    points have coordinates ``(x, [y,] z)`` in meters. With no query argument,
    evaluation returns a tensor on ``grid``. The sparse backend supplies first
    derivatives; the dense backend is intended for small reference problems.
    """

    def __init__(
        self,
        grid: GridSpec,
        bounds: tuple[float, float] = (1500.0, 4500.0),
        background: tuple[float, float] = (1500.0, 3000.0),
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        soft_clip: float = 20.0,
        max_pairs: int = 131072,
        backend: str = "sparse",
        sampling: SamplingConfig | None = None,
    ) -> None:
        super().__init__()
        if (
            len(bounds) != 2
            or not all(math.isfinite(v) for v in bounds)
            or not 0 < bounds[0] < bounds[1]
            or not math.isfinite(soft_clip)
            or soft_clip <= 0
            or backend not in ("sparse", "dense", "sparse_fused")
            or type(max_pairs) is not int
            or max_pairs < 1
        ):
            raise ValueError("Invalid physical bounds, soft clip, or backend")
        if len(background) != 2 or not all(math.isfinite(v) for v in background):
            raise ValueError("background must contain two finite endpoint velocities")
        self.grid = grid
        self.bounds = tuple(map(float, bounds))
        if isinstance(sampling, dict):
            sampling = SamplingConfig(**sampling)
        if sampling is not None and not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None")
        self.sampling = sampling
        self.sigma_min = float(sigma_min if sigma_min is not None else 0.45 * grid.spacing)
        if sampling is not None:
            if not math.isfinite(self.sigma_min) or self.sigma_min <= 0:
                raise ValueError("Minimum Gaussian width must be finite and positive")
            self.sigma_min = max(self.sigma_min, sampling.minimum_width(grid.spacing))
        self.sigma_max = float(sigma_max if sigma_max is not None else max(grid.extent))
        if (
            not math.isfinite(self.sigma_min)
            or not math.isfinite(self.sigma_max)
            or not 0 < self.sigma_min < self.sigma_max
        ):
            raise ValueError("Invalid scale bounds")
        self.soft_clip, self.max_pairs, self.backend = float(soft_clip), int(max_pairs), backend
        self.background = nn.Parameter(torch.tensor(background, dtype=torch.float32))
        self.blocks = nn.ModuleList()
        self.register_buffer("_points", grid.points(), persistent=False)

    @property
    def count(self) -> int:
        """Number of kernels across all active levels."""
        return sum(len(b.centers) for b in self.blocks)

    def add_gaussians(self, centers: Tensor, scales: Tensor) -> GaussianBlock:
        """Append zero-amplitude kernels, preserving the current decoded field."""
        centers = centers.to(self.background)
        scales = scales.to(self.background)
        if centers.ndim != 2 or centers.shape[1] != self.grid.ndim:
            raise ValueError("Gaussian dimension must match grid")
        if not torch.isfinite(centers).all() or not torch.isfinite(scales).all():
            raise ValueError("Gaussian centers and scales must be finite")
        block = GaussianBlock(centers, scales)
        if self.sampling is not None and bool(
            ((scales < self.sigma_min) | (scales > self.sigma_max)).any()
        ):
            raise ValueError(
                "Initial Gaussian widths must satisfy the sampling and geometry limits"
            )
        self.blocks.append(block)
        return block

    def add_grid_level(
        self, shape: tuple[int, ...], sigma_ratio: float = 0.65, seed: int = 0, jitter: float = 0.0
    ) -> GaussianBlock:
        """Initialize a level on a regular lattice in tensor-axis order.

        Widths are ``sigma_ratio`` times the lattice spacing. Optional seeded
        jitter is measured in units of each initial principal width.
        """
        GridSpec(shape, self.grid.spacing)
        if len(shape) != self.grid.ndim or not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
            raise ValueError("Level shape must match grid dimension and each axis >= 2")
        if not math.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be finite and nonnegative")
        axes = [
            torch.linspace(0, e, n, dtype=self.background.dtype, device=self.background.device)
            for n, e in zip(shape, reversed(self.grid.extent))
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        centers = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, self.grid.ndim)
        scales = (
            centers.new_tensor(
                [sigma_ratio * e / (n - 1) for e, n in zip(self.grid.extent, reversed(shape))]
            )
            .expand_as(centers)
            .clone()
        )
        scales.clamp_(self.sigma_min, self.sigma_max)
        if jitter:
            generator = torch.Generator(device=centers.device).manual_seed(seed)
            centers = centers + jitter * scales * torch.randn(
                centers.shape, generator=generator, device=centers.device, dtype=centers.dtype
            )
        return self.add_gaussians(centers, scales)

    def raw(self, points: Tensor | None = None) -> Tensor:
        """Return unbounded velocity at flattened physical query points."""
        points = self._points if points is None else points
        depth = points[:, -1] / self.grid.extent[-1]
        value = self.background[0] + (self.background[1] - self.background[0]) * depth
        if self.backend == "sparse_fused" and self.blocks:
            # Concatenation preserves every upstream Parameter and its Adam
            # history. Only the evaluation is fused; blocks are not coalesced.
            centers = torch.cat([block.centers for block in self.blocks])
            widths = torch.cat([block.log_scales for block in self.blocks]).exp()
            shears = torch.cat([block.shears for block in self.blocks])
            lower = torch.diag_embed(widths)
            row, col = torch.tril_indices(self.grid.ndim, self.grid.ndim, -1,
                                          device=points.device)
            lower[:, row, col] = widths[:, row] * shears
            amplitudes = torch.cat([block.amplitudes for block in self.blocks])
            return decode(points, centers, lower, amplitudes, value,
                          bounds=None, backend="sparse", max_pairs=self.max_pairs)
        evaluator = sparse_sum if self.backend == "sparse" else dense_sum
        for block in self.blocks:
            kwargs = {"max_pairs": self.max_pairs} if self.backend == "sparse" else {}
            value = value + evaluator(
                points, block.centers, block.precision(), block.amplitudes, **kwargs
            )
        return value

    def bound_velocity(self, value: Tensor) -> Tensor:
        lo, hi = self.bounds
        s = self.soft_clip
        # Stable equivalent of lo + s*softplus((value-lo)/s) - s*softplus((value-hi)/s).
        return value + s * F.softplus((lo - value) / s) - s * F.softplus((value - hi) / s)

    def forward(self, points: Tensor | None = None) -> Tensor:
        result = self.bound_velocity(self.raw(points))
        return result.reshape(self.grid.shape) if points is None else result

    @torch.no_grad()
    def project_(self) -> None:
        """Project physical radii while preserving admissible covariances exactly.

        Singular values of the Cholesky factor are the principal standard
        deviations. Working in float64 avoids squaring its condition number
        during the admissibility check. Shears are not clipped independently:
        a large shear can encode a valid thin, tilted Gaussian.
        """
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise FloatingPointError("Non-finite Gaussian parameter")
        self.background.clamp_(self.bounds[0] - 1000, self.bounds[1] + 1000)
        for block in self.blocks:
            factor = block.cholesky(dtype=torch.float64)
            if not torch.isfinite(factor).all():
                raise FloatingPointError("Covariance factor is outside the finite numerical range")
            axes, radii, _ = torch.linalg.svd(factor, full_matrices=False)
            clipped = radii.clamp(self.sigma_min, self.sigma_max)
            mask = (radii != clipped).any(-1)
            if mask.any():
                projected = (axes[mask] * clipped[mask].square().unsqueeze(-2)) @ axes[
                    mask
                ].transpose(-1, -2)
                lower = torch.linalg.cholesky(projected).to(block.centers)
                diag = lower.diagonal(dim1=-2, dim2=-1)
                row, col = torch.tril_indices(
                    self.grid.ndim, self.grid.ndim, -1, device=lower.device
                )
                block.log_scales[mask] = diag.log()
                block.shears[mask] = lower[:, row, col] / diag[:, row]
            extent = block.centers.new_tensor(self.grid.extent)
            block.centers.copy_(
                torch.minimum(torch.maximum(block.centers, -0.2 * extent), 1.2 * extent)
            )

    def parameter_groups(self, **kwargs: float) -> list[dict[str, Any]]:
        """Return optimizer groups while retaining each parameter object's identity."""
        _check_learning_rates(**kwargs)
        groups = [
            {
                "params": [self.background],
                "lr": kwargs.get("amplitude_lr", 4.0),
                "role": "background",
            }
        ]
        for block in self.blocks:
            groups.extend(block.parameter_groups(**kwargs))
        return groups

    def config(self) -> dict[str, Any]:
        """Return serializable physical and numerical decoder settings."""
        config = {
            "grid": asdict(self.grid),
            "bounds": self.bounds,
            "background": self.background.detach().cpu().tolist(),
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "soft_clip": self.soft_clip,
            "max_pairs": self.max_pairs,
            "backend": self.backend,
        }
        if self.sampling is not None:
            config["sampling"] = asdict(self.sampling)
        return config

    def checkpoint(self) -> dict[str, Any]:
        """Snapshot block topology and parameter tensors without an autograd graph."""
        return {
            "format": "gaussian-fwi-field-v3"
            if self.backend == "sparse_fused"
            else "gaussian-fwi-field-v2"
            if self.sampling is not None
            else "gaussian-fwi-field-v1",
            "config": self.config(),
            "blocks": [
                {
                    "count": len(b.centers),
                    "nominal_scale": b.nominal_scale,
                    "amplitude_lr_scale": b.amplitude_lr_scale,
                }
                for b in self.blocks
            ],
            "state": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Atomically save field topology, parameters, units, and decoder settings."""
        save_torch(self.checkpoint(), path, overwrite=overwrite)

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], device: str | torch.device = "cpu"
    ) -> "GaussianField":
        """Restore field topology, dtype, and parameters from a portable snapshot."""
        if payload["format"] not in (
            "gaussian-fwi-field-v1", "gaussian-fwi-field-v2", "gaussian-fwi-field-v3"
        ):
            raise ValueError("Unknown field checkpoint format")
        fused = payload["config"].get("backend") == "sparse_fused"
        if fused != (payload["format"] == "gaussian-fwi-field-v3"):
            raise ValueError("Fused evaluation requires its versioned field format")
        if not fused and (payload["config"].get("sampling") is not None) != (
            payload["format"] == "gaussian-fwi-field-v2"
        ):
            raise ValueError("Sampling policy does not match the field checkpoint format")
        config = dict(payload["config"])
        config["grid"] = GridSpec(**config["grid"])
        dtype = payload["state"]["background"].dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Field checkpoints require float32 or float64 tensors")
        if any(
            not isinstance(value, Tensor) or value.dtype != dtype or not torch.isfinite(value).all()
            for value in payload["state"].values()
        ):
            raise ValueError("Field checkpoint tensors must be finite and share one dtype")
        field = cls(**config).to(device=device, dtype=dtype)
        if field.sampling is not None and field.sigma_min != config["sigma_min"]:
            raise ValueError("Checkpoint radius floor violates the declared sampling policy")
        for entry in payload["blocks"]:
            if type(entry["count"]) is not int or entry["count"] <= 0:
                raise ValueError("Checkpoint block counts must be positive integers")
            nominal_scale = entry["nominal_scale"]
            amplitude_lr_scale = entry.get("amplitude_lr_scale", 1.0)
            if any(
                not math.isfinite(value) or value <= 0
                for value in (nominal_scale, amplitude_lr_scale)
            ):
                raise ValueError("Invalid Gaussian optimization scale in checkpoint")
            centers = field.background.new_zeros(entry["count"], field.grid.ndim)
            block = field.add_gaussians(
                centers,
                torch.full_like(centers, field.sigma_min)
                if field.sampling is not None
                else torch.ones_like(centers),
            )
            block.nominal_scale, block.amplitude_lr_scale = nominal_scale, amplitude_lr_scale
        field.load_state_dict(payload["state"])
        for block in field.blocks:
            lower = block.cholesky()
            if not torch.isfinite(lower).all() or not (lower.diagonal(dim1=-2, dim2=-1) > 0).all():
                raise ValueError("Checkpoint covariance factors exceed the finite numerical range")
            if (
                not torch.isfinite(block.covariance()).all()
                or not torch.isfinite(block.precision()).all()
            ):
                raise ValueError(
                    "Checkpoint covariance or precision exceeds the finite numerical range"
                )
            if field.sampling is not None:
                radii = torch.linalg.svdvals(block.cholesky(dtype=torch.float64))
                tolerance = 32 * torch.finfo(dtype).eps
                if bool(
                    (
                        (radii < field.sigma_min * (1 - tolerance))
                        | (radii > field.sigma_max * (1 + tolerance))
                    ).any()
                ):
                    raise ValueError("Checkpoint widths violate the declared sampling limits")
        return field

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "GaussianField":
        """Read a field file using PyTorch's restricted weights-only loader."""
        return cls.from_checkpoint(torch.load(path, map_location="cpu", weights_only=True), device)


def _check_learning_rates(
    amplitude_lr: float = 4.0, geometry_lr: float = 0.008, center_lr_ratio: float = 0.02
) -> None:
    if any(
        not math.isfinite(value) or value < 0
        for value in (amplitude_lr, geometry_lr, center_lr_ratio)
    ):
        raise ValueError("Gaussian learning rates must be finite and nonnegative")
