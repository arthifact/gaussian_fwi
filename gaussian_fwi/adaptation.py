"""Objective-guided Gaussian insertion and conservative binary refinement.

Proposals use the gradient of the regularized training objective with respect
to raw velocity. No reference velocity, image metric, or test receiver enters
this module. Split proposals are evaluated before the live field is changed.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from torch import Tensor, nn

from fwi_core.checkpoint import BestCheckpoint

from ._optimizer import validate_adam
from .field import GaussianBlock, GaussianField
from .raster import kernel, pair_chunks


@dataclass(frozen=True)
class AdaptationConfig:
    """Explicit refinement budget and acceptance tolerances.

    Events occur at ``warmup_steps + k * interval`` within each frequency
    stage when the remaining updates cover both the settling reserve and
    the complete comparison horizon. The horizon cannot exceed the interval.
    Insertion scales multiply that stage's nominal lattice width. The field
    change limit is in m/s; objective tolerances are relative. These defaults
    are starting settings, not an acquisition-independent optimum.

    ``comparison_steps=1`` retains the original first-update guard. Larger
    horizons evaluate future benefit but still commit only the first update,
    allowing the caller's normal loop to preserve its selection cadence.
    """

    interval: int = 25
    warmup_steps: int = 25
    settling_steps: int = 10
    insertions_per_event: int = 32
    splits_per_event: int = 8
    candidate_pool: int = 128
    insertion_scale_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
    separation_ratio: float = 1.0
    split_fraction: float = 0.5
    max_split_change: float = 10.0
    max_backtracks: int = 4
    relative_tolerance: float = 1e-6
    comparison_steps: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "insertion_scale_factors", tuple(self.insertion_scale_factors))
        for name in (
            "interval",
            "warmup_steps",
            "settling_steps",
            "candidate_pool",
            "comparison_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.comparison_steps > self.interval:
            raise ValueError("comparison_steps cannot overlap the next refinement event")
        for name in ("insertions_per_event", "splits_per_event", "max_backtracks"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.insertions_per_event + self.splits_per_event == 0:
            raise ValueError("Enable insertion, splitting, or both")
        if self.candidate_pool < max(self.insertions_per_event, self.splits_per_event):
            raise ValueError("candidate_pool must cover each event's requested operation count")
        if not self.insertion_scale_factors or any(
            not math.isfinite(s) or s <= 0 for s in self.insertion_scale_factors
        ):
            raise ValueError("Insertion scale factors must be finite and positive")
        if len(set(self.insertion_scale_factors)) != len(self.insertion_scale_factors):
            raise ValueError("Insertion scale factors must be distinct")
        for name in ("separation_ratio", "max_split_change"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.split_fraction) or not 0 < self.split_fraction < 1:
            raise ValueError("split_fraction must lie strictly between zero and one")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be finite and nonnegative")

    def due(self, step: int, steps_per_stage: int) -> bool:
        return (
            step >= self.warmup_steps
            and (step - self.warmup_steps) % self.interval == 0
            and steps_per_stage - step >= max(self.settling_steps, self.comparison_steps)
        )


@dataclass(frozen=True)
class ObjectiveValues:
    """Scores for one field at the currently active frequency bands."""

    training: float
    waveform: float
    validation: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v >= 0 for v in self.as_dict().values()):
            raise FloatingPointError("Adaptation requires finite, nonnegative objective values")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in ("training", "waveform", "validation")}

    def accepts(self, candidate: "ObjectiveValues", tolerance: float) -> bool:
        return all(
            after <= before * (1 + tolerance)
            for before, after in zip(self.as_dict().values(), candidate.as_dict().values())
        )


class GaussianCheckpoint(BestCheckpoint):
    """Restore Gaussian topology, parameter values, and Adam state together.

    References preserve parameter identities in memory; tensor values and
    optimizer moments use independent snapshots. Replaced blocks therefore
    remain recoverable even when later iterations change the number of kernels.
    Portable checkpoints continue to use ``GaussianField.checkpoint``.
    """

    def consider(self, score, step, model, optimizer=None) -> bool:
        blocks = tuple(model.blocks)
        metadata = [(b.nominal_scale, b.amplitude_lr_scale) for b in blocks]
        groups = (
            [
                {
                    **deepcopy({k: v for k, v in group.items() if k != "params"}),
                    "params": list(group["params"]),
                }
                for group in optimizer.param_groups
            ]
            if optimizer is not None
            else None
        )
        if not super().consider(score, step, model, optimizer):
            return False
        self.blocks, self.metadata, self.groups = blocks, metadata, groups
        return True

    def restore(self, model, optimizer=None) -> None:
        if self.model_state is None:
            raise RuntimeError("No finite checkpoint was evaluated")
        model.blocks = nn.ModuleList(self.blocks)
        for block, (nominal_scale, amplitude_lr_scale) in zip(self.blocks, self.metadata):
            block.nominal_scale, block.amplitude_lr_scale = nominal_scale, amplitude_lr_scale
        if optimizer is not None and self.groups is not None:
            optimizer.param_groups = [
                {**group, "params": list(group["params"])} for group in self.groups
            ]
        super().restore(model, optimizer)


def _checked_gradient(field: GaussianField, gradient: Tensor) -> Tensor:
    if gradient.numel() != math.prod(field.grid.shape):
        raise ValueError("Adaptation gradient must contain one value per simulation grid point")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite adaptation gradient")
    return gradient.detach().flatten().to(device=field.background.device, dtype=torch.float64)


def _basis(field: GaussianField, center: Tensor, covariance: Tensor) -> Tensor:
    """Evaluate one proposal in float64 without constructing all Gaussian pairs."""
    points = field._points.double()
    delta = points - center.double()
    lower = torch.linalg.cholesky(covariance.double())
    whitened = torch.linalg.solve_triangular(lower, delta.T, upper=False)
    return kernel(whitened.square().sum(0))[0]


@torch.no_grad()
def insertion_candidates(
    field: GaussianField,
    gradient: Tensor,
    nominal_scale: float,
    config: AdaptationConfig,
    *,
    exclude_covered: bool = False,
) -> tuple[Tensor, Tensor, list[float]]:
    """Rank localized descent directions by gradient correlation per basis norm.

    Gaussian filtering only screens candidate grid locations. Final scores use
    the actual tapered kernel, including domain truncation. Local suppression
    prevents one event from repeatedly selecting the same neighborhood.

    ``exclude_covered`` removes covered peaks before either shortlist budget.
    Coverage uses float64 physical distances and the radius as stored in the
    returned tensor. At most ``candidate_pool`` exact bases are evaluated per
    distinct clamped width, including evaluations yielding a zero score. Finding
    uncovered peaks can still scan every local maximum. The default preserves
    the historical selector used by scheduled adaptation and legacy refinement.
    """
    if type(exclude_covered) is not bool:
        raise TypeError("exclude_covered must be a boolean")
    g = _checked_gradient(field, gradient)
    empty = field.background.new_empty((0, field.grid.ndim))
    if config.insertions_per_event == 0 or not bool(g.any()):
        return empty, empty.clone(), []
    if not math.isfinite(nominal_scale) or nominal_scale <= 0:
        raise ValueError("nominal_scale must be finite and positive")
    image = g.cpu().numpy().reshape(field.grid.shape)
    proposals = []
    widths = sorted(
        {
            min(field.sigma_max, max(field.sigma_min, nominal_scale * f))
            for f in config.insertion_scale_factors
        }
    )
    tree = None
    if exclude_covered and field.count:
        existing_centers = torch.cat([b.centers for b in field.blocks]).double().cpu().numpy()
        existing_radii = (
            torch.cat(
                [torch.linalg.eigvalsh(b.covariance().double())[:, -1].sqrt() for b in field.blocks]
            )
            .cpu()
            .numpy()
        )
        tree = cKDTree(existing_centers)
    eye = torch.eye(field.grid.ndim, dtype=torch.float64, device=g.device)
    for width in widths:
        response = np.abs(
            gaussian_filter(image, width / field.grid.spacing, mode="constant", truncate=6)
        )
        peaks = np.flatnonzero(
            ((response == maximum_filter(response, size=3, mode="constant")) & (response > 0))
        )
        order = np.argsort(-response.ravel()[peaks], kind="stable")
        if not exclude_covered:
            order = order[: config.candidate_pool]
        evaluated = 0
        stored_radius = float(field.background.new_tensor(width))
        for index in peaks[order]:
            center = field._points[int(index)]
            if tree is not None:
                point = center.double().cpu().numpy()
                radius = config.separation_ratio * stored_radius
                # One neighborhood at a time, never grid-by-population pairs.
                neighbors = tree.query_ball_point(point, np.nextafter(radius, np.inf))
                if neighbors:
                    distances = np.linalg.norm(existing_centers[neighbors] - point, axis=1)
                    limits = config.separation_ratio * np.minimum(
                        stored_radius, existing_radii[neighbors]
                    )
                    if bool(np.any(distances < limits)):
                        continue
            values = _basis(field, center, eye * width**2)
            evaluated += 1
            norm = float(torch.linalg.vector_norm(values))
            score = abs(float(torch.dot(g, values))) / norm if norm > 0 else 0.0
            if score > 0 and math.isfinite(score):
                proposals.append((score, int(index), width))
            if evaluated == config.candidate_pool:
                break
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = []
    for score, index, width in proposals:
        center = field._points[index]
        if any(
            float(torch.linalg.vector_norm(center - field._points[other]))
            < config.separation_ratio * min(width, other_width)
            for _, other, other_width in chosen
        ):
            continue
        chosen.append((score, index, width))
        if len(chosen) == config.insertions_per_event:
            break
    if not chosen:
        return empty, empty.clone(), []
    centers = torch.stack([field._points[index] for _, index, _ in chosen])
    scales = (
        torch.tensor([width for _, _, width in chosen], dtype=centers.dtype, device=centers.device)[
            :, None
        ]
        .expand_as(centers)
        .clone()
    )
    return centers, scales, [score for score, _, _ in chosen]


@torch.no_grad()
def binary_children(
    center: Tensor, covariance: Tensor, amplitude: Tensor, axis: int, fraction: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return symmetric children preserving the ideal parent's first moments.

    Offsets are ±fraction times a principal standard deviation. Removing the
    offset outer product from each child covariance preserves the mixture's
    covariance. Peak amplitudes include the determinant ratio because our
    kernels are unnormalized and may be negative. Signed integral, centroid,
    and covariance are preserved for the untruncated Gaussian in all space;
    the sampled, tapered, bounded velocity field is checked separately.
    """
    d = center.numel()
    if (
        center.ndim != 1
        or covariance.shape != (d, d)
        or amplitude.numel() != 1
        or type(axis) is not int
        or not 0 <= axis < d
    ):
        raise ValueError("Invalid parent geometry or split axis")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Split fraction must lie strictly between zero and one")
    if not all(torch.isfinite(t).all() for t in (center, covariance, amplitude)):
        raise FloatingPointError("Non-finite split parent")
    if not torch.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-12):
        raise ValueError("Parent covariance must be symmetric")
    values, vectors = torch.linalg.eigh(covariance.double())
    if bool((values <= 0).any()):
        raise ValueError("Parent covariance must be positive definite")
    offset = fraction * values[axis].sqrt() * vectors[:, axis]
    child_covariance = covariance.double() - torch.outer(offset, offset)
    # det(child) = (1 - fraction**2) * det(parent).
    child_amplitude = amplitude.double().reshape(()) / (2 * math.sqrt(1 - fraction**2))
    if (
        not torch.isfinite(child_amplitude)
        or not torch.isfinite(child_covariance).all()
        or not (torch.linalg.eigvalsh(child_covariance) > 0).all()
    ):
        raise FloatingPointError(
            "Split children exceed the finite positive-definite numerical range"
        )
    return (
        torch.stack((center.double() - offset, center.double() + offset)),
        child_covariance.expand(2, d, d).clone(),
        child_amplitude.expand(2).clone(),
    )


def _admissible(field: GaussianField, centers: Tensor, covariances: Tensor) -> bool:
    radii = torch.linalg.eigvalsh(covariances.double()).sqrt()
    extent = centers.new_tensor(field.grid.extent)
    return bool(
        torch.isfinite(radii).all()
        and (radii >= field.sigma_min).all()
        and (radii <= field.sigma_max * (1 + 1e-12)).all()
        and (centers >= -0.2 * extent).all()
        and (centers <= 1.2 * extent).all()
    )


@torch.no_grad()
def split_candidates(
    field: GaussianField, gradient: Tensor, config: AdaptationConfig
) -> list[tuple[int, int, int, float]]:
    """Choose parent/axis pairs with a negative predicted objective change.

    A streamed absolute-gradient overlap screens parents without cancellation.
    Each principal axis is then scored using the actual signed split detail.
    This is a first-order proposal ranking, not an estimate of a velocity error.
    """
    g = _checked_gradient(field, gradient)
    if config.splits_per_event == 0 or not bool(g.any()):
        return []
    parents = []
    points = field._points.detach().cpu().double().numpy()
    for block_index, block in enumerate(field.blocks):
        centers = block.centers.detach().cpu().double().numpy()
        precision = block.precision().detach().cpu().double().numpy()
        overlap = g.new_zeros(len(centers))
        for gg, xx in pair_chunks(points, centers, precision, 6.0, field.max_pairs):
            ids = torch.as_tensor(gg, device=g.device)
            locations = torch.as_tensor(xx, device=g.device)
            delta = torch.as_tensor(points[xx] - centers[gg], device=g.device)
            matrix = torch.as_tensor(precision[gg], device=g.device)
            values = kernel(torch.einsum("ni,nij,nj->n", delta, matrix, delta))[0]
            overlap.index_add_(0, ids, g[locations].abs() * values)
        for row, score in enumerate(overlap.cpu().tolist()):
            if score > 0 and float(block.amplitudes[row].abs()) > 0:
                parents.append((score, block_index, row))
    parents.sort(key=lambda item: (-item[0], item[1], item[2]))
    proposals = []
    for _, bi, row in parents[: config.candidate_pool]:
        block = field.blocks[bi]
        covariance = block.covariance()[row].double()
        parent = block.amplitudes[row].double() * _basis(field, block.centers[row], covariance)
        best = None
        for axis in range(field.grid.ndim):
            try:
                centers, covariances, amplitudes = binary_children(
                    block.centers[row],
                    covariance,
                    block.amplitudes[row],
                    axis,
                    config.split_fraction,
                )
            except FloatingPointError:
                continue
            if not _admissible(field, centers, covariances):
                continue
            detail = (
                sum(amplitudes[k] * _basis(field, centers[k], covariances[k]) for k in range(2))
                - parent
            )
            norm = float(torch.linalg.vector_norm(detail))
            decrease = -float(torch.dot(g, detail))
            score = decrease / norm if norm > 0 else 0.0
            if score > 0 and math.isfinite(score) and (best is None or score > best[3]):
                best = (bi, row, axis, score)
        if best is not None:
            proposals.append(best)
    proposals.sort(key=lambda item: (-item[3], item[0], item[1], item[2]))
    return proposals[: config.splits_per_event]


def _block_from_tensors(centers, covariances, amplitudes, reference) -> GaussianBlock:
    lower = torch.linalg.cholesky(covariances.double()).to(reference)
    diagonal = lower.diagonal(dim1=-2, dim2=-1)
    block = GaussianBlock(centers.to(reference), diagonal)
    with torch.no_grad():
        row, col = torch.tril_indices(
            centers.shape[-1], centers.shape[-1], -1, device=centers.device
        )
        block.shears.copy_(lower[:, row, col] / diagonal[:, row])
        block.amplitudes.copy_(amplitudes.to(reference))
    if any(not torch.isfinite(p).all() for p in block.parameters()):
        raise FloatingPointError("Split parameters exceed the field's numerical precision")
    return block


@torch.no_grad()
def apply_splits(
    field: GaussianField,
    proposals: list[tuple[int, int, int, float]],
    fraction: float,
    optimizer: torch.optim.Adam | None = None,
) -> None:
    """Replace selected parents and preserve Adam state for every surviving row.

    Children retain their parent's position and covariance learning rates.
    Their amplitude rate is multiplied by the signed-mass allocation factor
    ``1 / (2 * sqrt(1 - fraction**2))``. Thus nearly coincident children do not
    double the amplitude update merely by duplicating a basis function.
    Fresh child moments remain an explicit change to the optimizer trajectory;
    :func:`adaptive_step` checks the resulting update against ordinary FWI.
    Any failure restores the original parameter identities, values and moments.
    """
    if not proposals:
        return
    if optimizer is None:
        _apply_splits(field, proposals, fraction, optimizer)
        return
    validate_adam(field, optimizer)
    checkpoint = GaussianCheckpoint()
    if not checkpoint.consider(0.0, 0, field, optimizer):
        raise FloatingPointError("Cannot split a non-finite field or optimizer")
    try:
        _apply_splits(field, proposals, fraction, optimizer)
        validate_adam(field, optimizer)
    except Exception:
        checkpoint.restore(field, optimizer)
        raise


def _apply_splits(field, proposals, fraction, optimizer) -> None:
    """Prepare replacement blocks, then transfer survivor state and commit topology."""
    selected = {}
    geometry = {}
    for bi, row, axis, _ in proposals:
        if any(type(index) is not int for index in (bi, row, axis)):
            raise ValueError("Split indices must be integers, excluding booleans")
        if not 0 <= bi < len(field.blocks) or not 0 <= row < len(field.blocks[bi].centers):
            raise ValueError("Split parent does not exist")
        if row in selected.setdefault(bi, []):
            raise ValueError("A parent can be split only once per event")
        selected[bi].append(row)
        block = field.blocks[bi]
        centers, covariances, amplitudes = binary_children(
            block.centers[row], block.covariance()[row], block.amplitudes[row], axis, fraction
        )
        if not _admissible(field, centers, covariances):
            raise ValueError("Split children exceed the field's geometry bounds")
        geometry.setdefault(bi, []).append((centers, covariances, amplitudes))
    # Keep different optimization scales separate, even within one event.
    # The metadata also preserves this choice when constructing a new optimizer.
    allocation = 1 / (2 * math.sqrt(1 - fraction**2))
    children = {}
    child_groups = []
    parameter_groups = (
        {p: group for group in optimizer.param_groups for p in group["params"]}
        if optimizer is not None
        else {}
    )
    for bi in sorted(geometry):
        old = field.blocks[bi]
        values = geometry[bi]
        child = _block_from_tensors(
            *(torch.cat([entry[k] for entry in values]) for k in range(3)), field.background
        )
        child.nominal_scale = old.nominal_scale
        child.amplitude_lr_scale = old.amplitude_lr_scale * allocation
        child.parameter_groups()  # Validate cumulative scale metadata before the live commit.
        children[bi] = child
        if optimizer is not None:
            for names, multiplier in (
                (("amplitudes",), allocation),
                (("centers",), 1.0),
                (("log_scales", "shears"), 1.0),
            ):
                origins = [parameter_groups.get(getattr(old, name)) for name in names]
                if origins[0] is None or any(group is not origins[0] for group in origins):
                    raise ValueError("Split parameters require the field's canonical Adam groups")
                child_groups.append(
                    {
                        **origins[0],
                        "params": [getattr(child, name) for name in names],
                        "lr": origins[0]["lr"] * multiplier,
                    }
                )
    replacements = {}
    blocks = []
    for bi, old in enumerate(field.blocks):
        if bi not in selected:
            blocks.append(old)
            continue
        keep = torch.ones(len(old.centers), dtype=torch.bool, device=old.centers.device)
        keep[selected[bi]] = False
        new = None
        if bool(keep.any()):
            new = GaussianBlock(old.centers[keep], old.log_scales[keep].exp())
            new.nominal_scale = old.nominal_scale
            new.amplitude_lr_scale = old.amplitude_lr_scale
            for name, parameter in new.named_parameters():
                parameter.copy_(getattr(old, name)[keep])
            blocks.append(new)
        for name, parameter in old.named_parameters():
            replacements[parameter] = (getattr(new, name) if new is not None else None, keep)
    if optimizer is not None:
        groups = []
        for group in optimizer.param_groups:
            parameters = []
            for parameter in group["params"]:
                if parameter not in replacements:
                    parameters.append(parameter)
                    continue
                replacement, keep = replacements[parameter]
                state = optimizer.state.pop(parameter, None)
                if replacement is not None:
                    parameters.append(replacement)
                    if state is not None:
                        optimizer.state[replacement] = {
                            key: value[keep].clone()
                            if isinstance(value, Tensor) and value.shape == parameter.shape
                            else deepcopy(value)
                            for key, value in state.items()
                        }
            if parameters:
                groups.append({**group, "params": parameters})
        optimizer.param_groups = groups
        for group in child_groups:
            optimizer.add_param_group(group)
    field.blocks = nn.ModuleList([*blocks, *children.values()])


@torch.no_grad()
def _adapt(
    field: GaussianField,
    optimizer: torch.optim.Adam,
    gradient: Tensor,
    nominal_scale: float,
    config: AdaptationConfig,
    learning_rates: dict[str, float],
    baseline: ObjectiveValues,
    evaluate: Callable[[GaussianField], ObjectiveValues],
) -> dict:
    """Propose, verify, and commit one refinement event without target access.

    Split trials run on detached field copies. A finite trial must respect the
    physical field-change bound and all three objective acceptance tests. New
    insertions have zero amplitude and preserve the committed velocity field.
    """
    gradient = _checked_gradient(field, gradient)
    centers, scales, insertion_scores = insertion_candidates(field, gradient, nominal_scale, config)
    proposals = split_candidates(field, gradient, config)
    record = {
        "before_count": field.count,
        "baseline": baseline.as_dict(),
        "split_candidates": [
            {"block": b, "row": r, "axis": a, "score": s} for b, r, a, s in proposals
        ],
        "trials": [],
        "accepted_splits": 0,
        "insertions": len(centers),
        "insertion_centers_m": centers.cpu().tolist(),
        "insertion_widths_m": scales.cpu().tolist(),
        "insertion_scores": insertion_scores,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
    }
    before_velocity = field().detach()
    for backtrack in range(config.max_backtracks + 1) if proposals else ():
        fraction = config.split_fraction * 0.5**backtrack
        candidate = GaussianField.from_checkpoint(
            field.checkpoint(), device=field.background.device
        )
        try:
            apply_splits(candidate, proposals, fraction)
        except FloatingPointError:
            record["trials"].append(
                {
                    "fraction": fraction,
                    "maximum_velocity_change_m_s": None,
                    "status": "nonfinite_field",
                }
            )
            continue
        candidate_velocity = candidate()
        change = float((candidate_velocity - before_velocity).abs().max())
        trial = {
            "fraction": fraction,
            "maximum_velocity_change_m_s": change if math.isfinite(change) else None,
        }
        if not math.isfinite(change):
            trial["status"] = "nonfinite_field"
        elif change > config.max_split_change:
            trial["status"] = "field_change_limit"
        else:
            try:
                values = evaluate(candidate)
            except FloatingPointError:
                trial["status"] = "nonfinite_objective"
            else:
                trial["objectives"] = values.as_dict()
                if baseline.accepts(values, config.relative_tolerance):
                    apply_splits(field, proposals, fraction, optimizer)
                    record["accepted_splits"] = len(proposals)
                    trial["status"] = "accepted"
                else:
                    trial["status"] = "objective_rejected"
        record["trials"].append(trial)
        if trial["status"] == "accepted":
            break
    for width in scales[:, 0].unique(sorted=True):
        mask = scales[:, 0] == width
        block = field.add_gaussians(centers[mask], scales[mask])
        for group in block.parameter_groups(**learning_rates):
            optimizer.add_param_group(group)
    record["after_count"] = field.count
    return record


def adapt(
    field: GaussianField,
    optimizer: torch.optim.Adam,
    gradient: Tensor,
    nominal_scale: float,
    config: AdaptationConfig,
    learning_rates: dict[str, float],
    baseline: ObjectiveValues,
    evaluate: Callable[[GaussianField], ObjectiveValues],
) -> dict:
    """Apply a refinement transaction, restoring field and Adam on any failure."""
    validate_adam(field, optimizer)
    checkpoint = GaussianCheckpoint()
    if not checkpoint.consider(baseline.validation, 0, field, optimizer):
        raise FloatingPointError("Cannot refine a non-finite field")
    try:
        event = _adapt(
            field, optimizer, gradient, nominal_scale, config, learning_rates, baseline, evaluate
        )
        validate_adam(field, optimizer)
        return event
    except Exception:
        checkpoint.restore(field, optimizer)
        raise


def _check_parameter_gradients(field: GaussianField) -> None:
    if any(p.grad is None for p in field.parameters()):
        raise ValueError("An adaptive update requires gradients for every field parameter")
    if any(not torch.isfinite(p.grad).all() for p in field.parameters()):
        raise FloatingPointError("Non-finite parameter gradient")


@dataclass
class _BranchTrace:
    """Completed optimizer work and post-update scores, including failed trials."""

    updates: int
    objectives: list[ObjectiveValues]


def _rollout(
    field: GaussianField,
    optimizer: torch.optim.Adam,
    steps: int,
    evaluate: Callable[[GaussianField], tuple[Tensor, ObjectiveValues]],
    first: GaussianCheckpoint,
    trace: _BranchTrace,
    *,
    use_existing_gradients: bool,
) -> None:
    """Evaluate a deterministic branch and snapshot its first update for commitment.

    Post-update evaluations supply the next training graph until the endpoint.
    Validation scores are scalars and never contribute to the backward pass.
    The trace is updated immediately after each optimizer step so a numerical
    failure cannot hide work already performed.
    """
    if not use_existing_gradients:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = evaluate(field)
        loss.backward()
    for step in range(1, steps + 1):
        _check_parameter_gradients(field)
        optimizer.step()
        trace.updates += 1
        field.project_()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(step < steps):
            loss, values = evaluate(field)
        validate_adam(field, optimizer)
        trace.objectives.append(values)
        if step == 1 and not first.consider(values.validation, 1, field, optimizer):
            raise FloatingPointError("Branch update produced a non-finite checkpoint")
        if step < steps:
            loss.backward()


def adaptive_step(
    field: GaussianField,
    optimizer: torch.optim.Adam,
    gradient: Tensor,
    nominal_scale: float,
    config: AdaptationConfig,
    learning_rates: dict[str, float],
    baseline: ObjectiveValues,
    evaluate: Callable[[GaussianField], tuple[Tensor, ObjectiveValues]],
) -> dict:
    """Compare equal-length refinement rollouts and commit one trajectory update.

    The caller supplies the current parameter gradients and raw-field gradient.
    Both deterministic branches evaluate ``config.comparison_steps`` updates.
    Their endpoint training, waveform, and validation scores select the branch;
    only its first update is committed. The caller's normal loop then replays
    later updates, preserving the trajectory length and selection cadence.
    The caller must keep the horizon inside one frequency stage and before
    the next refinement event, as enforced by ``AdaptationConfig.due``.
    Rejected or non-finite candidates restore the ordinary first update. An
    unexpected failure restores the pre-event state and propagates. All trial
    work is counted; no endpoint field is silently committed as a single step.
    """
    validate_adam(field, optimizer)
    gradient = _checked_gradient(field, gradient)
    _check_parameter_gradients(field)
    before = GaussianCheckpoint()
    if not before.consider(baseline.validation, 0, field, optimizer):
        raise FloatingPointError("Cannot refine a non-finite field")
    reference_trace, candidate_trace = _BranchTrace(0, []), _BranchTrace(0, [])
    try:
        reference = GaussianCheckpoint()
        _rollout(
            field,
            optimizer,
            config.comparison_steps,
            evaluate,
            reference,
            reference_trace,
            use_existing_gradients=True,
        )
        reference_values = reference_trace.objectives[-1]
        before.restore(field, optimizer)

        event = adapt(
            field,
            optimizer,
            gradient,
            nominal_scale,
            config,
            learning_rates,
            baseline,
            lambda candidate: evaluate(candidate)[1],
        )
        event["after_proposal_count"] = event["after_count"]
        guard = {"reference": reference_values.as_dict()}
        changed = event["after_count"] != event["before_count"]
        retained = False
        if changed:
            candidate_first = GaussianCheckpoint()
            try:
                _rollout(
                    field,
                    optimizer,
                    config.comparison_steps,
                    evaluate,
                    candidate_first,
                    candidate_trace,
                    use_existing_gradients=False,
                )
                candidate_values = candidate_trace.objectives[-1]
                guard["candidate"] = candidate_values.as_dict()
                retained = reference_values.accepts(candidate_values, config.relative_tolerance)
                guard["status"] = "accepted" if retained else "objective_rejected"
            except FloatingPointError:
                guard["status"] = "nonfinite_update"
        else:
            guard["status"] = "no_proposal"
        if not retained:
            reference.restore(field, optimizer)
        elif config.comparison_steps > 1:
            candidate_first.restore(field, optimizer)
        if config.comparison_steps > 1:
            guard["lookahead"] = {
                "comparison_steps": config.comparison_steps,
                "trajectory_updates_committed": 1,
                "committed_branch": "refined" if retained else "ordinary",
                "reference_updates": reference_trace.updates,
                "candidate_updates": candidate_trace.updates,
                "reference_path": [v.as_dict() for v in reference_trace.objectives],
                "candidate_path": [v.as_dict() for v in candidate_trace.objectives],
            }
        event["step_guard"] = guard
        event["optimizer_updates_evaluated"] = reference_trace.updates + candidate_trace.updates
        event["retained_insertions"] = event["insertions"] if retained else 0
        event["retained_splits"] = event["accepted_splits"] if retained else 0
        event["after_count"] = field.count
        return event
    except Exception:
        before.restore(field, optimizer)
        raise
