"""Explicit minimum-radius continuation without changing the decoded field."""

import math

import torch

from .field import GaussianField


def normalize_radius_schedule(values, stages: int) -> tuple[float, ...] | None:
    """Validate absolute principal-radius floors, in meters, before fitting."""
    if values is None:
        return None
    values = tuple(values)
    if len(values) != stages or any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError("A finite positive spatial radius is required for every stage")
    if any(b > a for a, b in zip(values, values[1:])):
        raise ValueError("Spatial radius floors must be nonincreasing")
    return tuple(float(value) for value in values)


@torch.no_grad()
def radius_record(field: GaussianField, minimum: float | None = None) -> dict:
    """Check actual covariance radii without projecting or modifying parameters."""
    minimum = field.sigma_min if minimum is None else minimum
    if not math.isfinite(minimum) or not 0 < minimum < field.sigma_max:
        raise ValueError("Spatial radius must lie below the field maximum")
    low, high = None, None
    tolerance = 64 * torch.finfo(field.background.dtype).eps
    for block in field.blocks:
        radii = torch.linalg.svdvals(block.cholesky(dtype=torch.float64))
        if not torch.isfinite(radii).all():
            raise ValueError("Spatial continuation requires finite covariance radii")
        smallest, largest = float(radii.min()), float(radii.max())
        if smallest < minimum * (1 - tolerance) or largest > field.sigma_max * (1 + tolerance):
            raise ValueError("Gaussian radii violate the active spatial limits")
        low = smallest if low is None else min(low, smallest)
        high = largest if high is None else max(high, largest)
    return {
        "minimum_radius_m": minimum,
        "realized_minimum_radius_m": low,
        "realized_maximum_radius_m": high,
        "gaussians": field.count,
    }


def set_radius_floor(field: GaussianField, minimum: float) -> None:
    """Change only the admissible set, preserving fields and optimizer identities."""
    minimum = effective_radius_floor(field, minimum)
    radius_record(field, minimum)
    field.sigma_min = float(minimum)


def effective_radius_floor(field: GaussianField, minimum: float) -> float:
    """Intersect frequency-stage and numerical sampling constraints, in meters."""
    if field.sampling is not None:
        return max(minimum, field.sampling.minimum_width(field.grid.spacing))
    return minimum


def validate_spatial_start(field: GaussianField, schedule, resume: dict | None) -> float | None:
    """Validate a fresh or resumed schedule before any output or wave solve."""
    if schedule is None:
        return None
    schedule = tuple(effective_radius_floor(field, value) for value in schedule)
    original = field.sigma_min if resume is None else resume.get("spatial_base_sigma_min_m")
    if (
        original is None
        or isinstance(original, bool)
        or not math.isfinite(original)
        or original <= 0
        or any(value < original or value >= field.sigma_max for value in schedule)
    ):
        raise ValueError("Spatial schedule must respect the original physical radius limits")
    if resume is None:
        radius_record(field, schedule[0])
    else:
        stage = resume.get("completed_stage")
        if type(stage) is not int or not 0 <= stage < len(schedule):
            raise ValueError("Invalid spatial continuation checkpoint stage")
        if field.sigma_min != schedule[stage]:
            raise ValueError("Checkpoint radius floor differs from its saved schedule")
        radius_record(field)
        if not isinstance(resume.get("spatial_history"), list):
            raise ValueError("Spatial checkpoint is missing its radius history")
    return float(original)
