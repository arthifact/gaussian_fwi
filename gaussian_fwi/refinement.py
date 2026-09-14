"""Training-driven coordination of Gaussian refinement capabilities.

There is no forced growth calendar, frequency-to-kernel-count rule, or search
over operation subsets. Eligible local proposals compete by a first-order
training-objective decrease. These scores are descent proxies, not estimates
of geological truth, posterior uncertainty, or seismic resolving power.
"""

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from .adaptation import (
    AdaptationConfig,
    GaussianCheckpoint,
    _basis,
    _checked_gradient,
    insertion_candidates,
)
from .density import spatial_scores
from .directions import adjoint_directions
from .field import GaussianField
from .topology import EditRejected, GaussianTopology, TopologyEdit
from .trials import TrainingEvaluator, recovery_endpoint


@dataclass(frozen=True)
class RefinementConfig:
    """Explicit numerical, recovery, and resource limits for local decisions.

    SI-unit thresholds concern the velocity field and geometry. ``minimum_age``
    allows new or edited kernels to train before another edit. It is also the
    recovery reserve before a stage ends. Neither it nor ``max_gaussians`` is a
    target population or a prescribed time at which an operation must happen.
    """

    max_gaussians: int = 100_000
    max_edits: int = 8
    candidate_pool: int = 32
    minimum_age: int = 5
    minimum_relative_gain: float = 1e-4
    gradient_threshold: float = 0.0
    max_field_change: float = 25.0
    split_geometry: str = "moment"
    split_fraction: float = 0.45
    max_backtracks: int = 3
    clone_radius_factor: float = 2.0
    insertion_scales: tuple[float, ...] | None = None
    insertion_separation: float = 0.5
    merge_distance: float = 0.2
    merge_shape_tolerance: float = 0.2
    merge_neighbors: int = 4
    prune_amplitude: float = 0.5
    minimum_gaussians: int = 1
    reset_amplitude: float | None = None
    stagnation_window: int = 5
    stagnation_tolerance: float = 1e-3
    comparison_steps: int = 0
    acceptance_rtol: float = 1e-6
    acceptance_atol: float = 1e-12
    minimum_shot_support: float = 0.0
    directional_splits: bool = False
    relocation: bool = False
    relocation_candidates: int = 2
    insertion_screening: str = "legacy"
    operations: tuple[str, ...] | None = None

    @classmethod
    def guarded(cls, **overrides):
        """Explicit v0.8 policy: recovery trials, directional proposals, relocation.

        Ordinary construction retains v0.7 defaults for historical controls.
        These starting settings are configurable, not an empirical SOTA claim.
        """
        return cls(
            **{
                "comparison_steps": 3,
                "minimum_shot_support": 0.5,
                "directional_splits": True,
                "relocation": True,
                **overrides,
            }
        )

    @property
    def extended(self):
        return bool(self.comparison_steps or self.directional_splits or self.relocation)

    def __post_init__(self) -> None:
        if self.operations is not None:
            object.__setattr__(self, "operations", tuple(self.operations))
            allowed = {"insert", "split", "clone", "merge", "prune", "reset", "relocate"}
            if any(not isinstance(op, str) or op not in allowed for op in self.operations):
                raise ValueError("Unknown refinement operation")
            if len(set(self.operations)) != len(self.operations):
                raise ValueError("Refinement operations must be distinct")
        for name in (
            "max_gaussians",
            "max_edits",
            "candidate_pool",
            "minimum_age",
            "merge_neighbors",
            "stagnation_window",
            "relocation_candidates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stagnation_window < 2:
            raise ValueError("Stagnation requires at least two training observations")
        for name in ("max_backtracks", "minimum_gaussians", "comparison_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.minimum_gaussians > self.max_gaussians:
            raise ValueError("Minimum population exceeds the resource ceiling")
        for name in (
            "minimum_relative_gain",
            "gradient_threshold",
            "merge_distance",
            "merge_shape_tolerance",
            "prune_amplitude",
            "stagnation_tolerance",
            "acceptance_rtol",
            "acceptance_atol",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if any(type(value) is not bool for value in (self.directional_splits, self.relocation)):
            raise ValueError("Directional splitting and relocation flags must be boolean")
        if not math.isfinite(self.minimum_shot_support) or not 0 <= self.minimum_shot_support <= 1:
            raise ValueError("Minimum shot support must be a fraction in [0, 1]")
        if self.minimum_shot_support and not self.comparison_steps:
            raise ValueError("Shot support requires measured recovery trials")
        for name in ("max_field_change", "clone_radius_factor", "insertion_separation"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.split_geometry not in ("moment", "long_axis"):
            raise ValueError("Unknown split geometry")
        if self.insertion_screening not in ("legacy", "coverage_aware"):
            raise ValueError("insertion_screening must be 'legacy' or 'coverage_aware'")
        if not math.isfinite(self.split_fraction) or not 0 < self.split_fraction <= 0.5:
            raise ValueError("Split fraction must lie in (0, 0.5]")
        if self.reset_amplitude is not None and (
            not math.isfinite(self.reset_amplitude) or self.reset_amplitude <= 0
        ):
            raise ValueError("Reset amplitude must be finite and positive")
        if self.insertion_scales is not None:
            object.__setattr__(self, "insertion_scales", tuple(self.insertion_scales))
            if not self.insertion_scales or any(
                not math.isfinite(s) or s <= 0 for s in self.insertion_scales
            ):
                raise ValueError("Insertion radii must be finite and positive, in meters")


class RefinementController:
    """Coordinate local actions using the latest regularized training adjoint.

    Call ``observe`` after backward and ``after_update`` after Adam/projection.
    The gradient is therefore one ordinary update old when proposals are scored,
    as in direct density control. With ``comparison_steps > 0``, explicit
    training-only recovery trials assess the coordinated candidate. Their work
    is recorded. Validation in the inversion driver selects the stage result.

    ``allow_edits=False`` supports a caller's fixed-topology optimization phase,
    for example a future source-location update. This class does not implement
    source estimation or decide when that future problem is identifiable.
    """

    def __init__(
        self,
        field: GaussianField,
        config: RefinementConfig,
        *,
        start_step: int = 0,
        topology_state: dict | None = None,
    ):
        if (
            not isinstance(config, RefinementConfig)
            or type(start_step) is not int
            or start_step < 0
        ):
            raise ValueError("Invalid refinement configuration or starting update")
        if field.count > config.max_gaussians:
            raise ValueError("Initial population exceeds the resource ceiling")
        self.config = config
        self.topology = GaussianTopology(field, topology_state)
        if any(age > start_step for ages in self.topology.last_edit_steps for age in ages):
            raise ValueError("Topology edit history lies after the controller's starting update")
        self.start_step, self.last_update = start_step, 0
        self.losses = deque(maxlen=config.stagnation_window)
        self.scores: dict[int, tuple[float, int]] = {}
        self.gradient: Tensor | None = None
        self.pending = False
        self.recovery_until = start_step

    @torch.no_grad()
    def observe(self, field: GaussianField, gradient: Tensor, training_objective: float) -> None:
        if field is not self.topology.field or self.pending:
            raise ValueError("Observe exactly once before each update on the registered field")
        if not math.isfinite(training_objective) or training_objective < 0:
            raise ValueError("Training objective must be finite and nonnegative")
        self.topology._check()
        g = _checked_gradient(field, gradient).clone()
        scores, _ = spatial_scores(field, g)
        current = {}
        for ids, values in zip(self.topology.ids, scores):
            for kernel_id, value in zip(ids, values):
                total, count = self.scores.get(kernel_id, (0.0, 0))
                current[kernel_id] = (total + float(value), count + 1)
        if any(not math.isfinite(total) for total, _ in current.values()):
            raise FloatingPointError("Refinement sensitivity accumulation overflow")
        self.gradient, self.scores = g, current
        self.losses.append(float(training_objective))
        self.pending = True

    def _eligible(self, global_step: int) -> list[int]:
        return [
            kernel_id
            for ids, ages in zip(self.topology.ids, self.topology.last_edit_steps)
            for kernel_id, age in zip(ids, ages)
            if global_step - age >= self.config.minimum_age
            and self.scores.get(kernel_id, (0.0, 0))[1] >= self.config.minimum_age
        ]

    @torch.no_grad()
    def _proposals(self, global_step: int, learning_rates: dict[str, float]):
        field, cfg, topology = self.topology.field, self.config, self.topology
        if cfg.operations == ():
            return []
        eligible = self._eligible(global_step)
        if self.gradient is None or (not eligible and field.count > 0):
            return []
        ranked = sorted(
            eligible,
            key=lambda i: (-self.scores.get(i, (0.0, 1))[0] / self.scores.get(i, (0.0, 1))[1], i),
        )
        seeds = ranked[: cfg.candidate_pool]
        raw = field.raw().detach()
        velocity = field.bound_velocity(raw)
        threshold = cfg.minimum_relative_gain * self.losses[-1]
        proposals = []

        def consider(edit: TopologyEdit, *, insertion_gain: float | None = None):
            if cfg.operations is not None and edit.operation not in cfg.operations:
                return
            try:
                delta = topology.raw_change(edit)
            except EditRejected:
                return
            change = float((field.bound_velocity(raw + delta.to(raw)) - velocity).abs().max())
            if change > cfg.max_field_change:
                return
            gain = (
                -float(torch.dot(self.gradient, delta))
                if insertion_gain is None
                else insertion_gain
            )
            if not math.isfinite(gain):
                raise FloatingPointError("Nonfinite predicted training decrease")
            # Exact algebraic cleanup has no first-order gain to exceed a loss
            # threshold. Approximate merge/prune proposals need positive evidence.
            neutral = edit.operation == "prune" and not bool(delta.any())
            if edit.operation == "merge":
                a, b = edit.parent_values
                neutral = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            if neutral:
                gain = 0.0  # Algebraically unchanged; floating-point sums may differ.
            if gain > threshold or neutral:
                proposals.append((gain, edit))

        progress = (self.losses[0] - self.losses[-1]) / max(
            self.losses[0], self.losses[-1], torch.finfo(torch.float64).tiny
        )
        stalled = len(self.losses) == cfg.stagnation_window and progress <= cfg.stagnation_tolerance
        cap = (
            cfg.reset_amplitude
            if cfg.reset_amplitude is not None
            else (field.bounds[1] - field.bounds[0]) / 2
        )
        for kernel_id in seeds:
            center, covariance, amplitude = topology.values(kernel_id)
            score, count = self.scores.get(kernel_id, (0.0, 1))
            if score / count > cfg.gradient_threshold:
                if (
                    float(torch.linalg.eigvalsh(covariance.double())[-1].sqrt())
                    <= cfg.clone_radius_factor * field.sigma_min
                ):
                    consider(topology.clone(kernel_id))
                axes = range(field.grid.ndim) if cfg.split_geometry == "moment" else (None,)
                for axis in axes:
                    for backtrack in range(cfg.max_backtracks + 1):
                        consider(
                            topology.split(
                                kernel_id,
                                axis=axis,
                                fraction=cfg.split_fraction / 2**backtrack,
                                geometry=cfg.split_geometry,
                            )
                        )
                if cfg.directional_splits:
                    directions = adjoint_directions(
                        field._points,
                        self.gradient,
                        center,
                        covariance,
                        amplitude,
                        chunk_size=field.max_pairs,
                    )
                    for direction in directions:
                        for backtrack in range(cfg.max_backtracks + 1):
                            consider(
                                topology.split(
                                    kernel_id,
                                    direction=direction,
                                    fraction=cfg.split_fraction / 2**backtrack,
                                )
                            )
        # Weak kernels and saturated amplitudes need their own screening order;
        # limiting every operation to high-sensitivity rows would starve pruning.
        by_amplitude = sorted(eligible, key=lambda i: (abs(float(topology.values(i)[2])), i))
        for kernel_id in by_amplitude[: cfg.candidate_pool]:
            if abs(float(topology.values(kernel_id)[2])) <= cfg.prune_amplitude:
                consider(topology.prune(kernel_id))
        if stalled:
            for kernel_id in reversed(by_amplitude[-cfg.candidate_pool :]):
                if abs(float(topology.values(kernel_id)[2])) > cap:
                    consider(topology.reset(kernel_id, amplitude_cap=cap))

        # Neighborhood search is bounded in candidate count; it does not build
        # an all-pairs distance matrix when the Gaussian population is large.
        if len(eligible) > 1:
            centers = np.stack([topology.values(i)[0].double().cpu().numpy() for i in eligible])
            tree = cKDTree(centers)
            lookup = {i: j for j, i in enumerate(eligible)}
            pairs = set()
            for first in seeds:
                _, neighbors = tree.query(
                    centers[lookup[first]], k=min(len(eligible), cfg.merge_neighbors + 1)
                )
                for neighbor in np.atleast_1d(neighbors):
                    second = eligible[int(neighbor)]
                    if first != second:
                        pairs.add(tuple(sorted((first, second))))
            for first, second in sorted(pairs):
                a, b = topology.values(first), topology.values(second)
                delta = (a[0] - b[0]).double()
                average = (a[1].double() + b[1].double()) / 2
                distance = float(
                    torch.dot(delta, torch.linalg.solve(average, delta)).clamp_min(0).sqrt()
                )
                shape = float(
                    torch.linalg.matrix_norm(
                        torch.linalg.solve(average, a[1].double() - b[1].double())
                    )
                )
                if distance <= cfg.merge_distance and shape <= cfg.merge_shape_tolerance:
                    try:
                        consider(topology.merge(first, second))
                    except EditRejected:
                        continue

        scales = cfg.insertion_scales
        if scales is None:
            radii = [
                float(torch.linalg.eigvalsh(topology.values(i)[1].double())[-1].sqrt())
                for i in seeds
            ]
            reference = float(np.median(radii)) if radii else field.sigma_max
            scales = (reference / 2, reference, reference * 2)
        settings = AdaptationConfig(
            insertions_per_event=min(cfg.max_edits, cfg.candidate_pool),
            splits_per_event=0,
            candidate_pool=cfg.candidate_pool,
            insertion_scale_factors=scales,
            separation_ratio=cfg.insertion_separation,
        )
        coverage_aware = cfg.insertion_screening == "coverage_aware"
        centers, radii, _ = insertion_candidates(
            field, self.gradient, 1.0, settings, exclude_covered=coverage_aware
        )
        existing = (
            [
                (
                    topology.values(i)[0],
                    float(torch.linalg.eigvalsh(topology.values(i)[1].double())[-1].sqrt()),
                )
                for ids in topology.ids
                for i in ids
            ]
            if not coverage_aware
            else []
        )
        for center, radius in zip(centers, radii[:, 0]):
            # The new selector is the sole coverage authority for its policy;
            # repeating the legacy field-dtype check could disagree at a boundary.
            if not coverage_aware and any(
                float(torch.linalg.vector_norm(center - other))
                < cfg.insertion_separation * min(float(radius), width)
                for other, width in existing
            ):
                continue
            edit = topology.insert(center, float(radius))
            phi = _basis(field, edit.centers[0].double(), edit.covariances[0].double())
            # A zero-amplitude birth has no immediate field change. Score its
            # first possible amplitude activation at the declared Adam LR.
            gain = abs(float(torch.dot(self.gradient, phi))) * learning_rates["amplitude_lr"]
            consider(edit, insertion_gain=gain)
            if cfg.relocation:
                for kernel_id in by_amplitude[: cfg.relocation_candidates]:
                    replacement = topology.relocate(kernel_id, center, float(radius))
                    removal_gain = -float(
                        torch.dot(self.gradient, topology.raw_change(replacement))
                    )
                    consider(replacement, insertion_gain=gain + removal_gain)
        order = {
            "merge": 0,
            "prune": 1,
            "split": 2,
            "insert": 3,
            "clone": 4,
            "reset": 5,
            "relocate": 6,
        }
        return sorted(
            proposals, key=lambda item: (-item[0], order[item[1].operation], item[1].parents)
        )

    @torch.no_grad()
    def after_update(
        self,
        field: GaussianField,
        optimizer: torch.optim.Adam,
        step: int,
        *,
        learning_rates: dict[str, float],
        remaining_updates: int,
        allow_edits: bool = True,
        reserved_capacity: int = 0,
        evaluate: TrainingEvaluator | None = None,
    ) -> dict | None:
        """Apply compatible locally useful edits, or leave the population alone."""
        if (
            field is not self.topology.field
            or not self.pending
            or type(step) is not int
            or step != self.last_update + 1
        ):
            raise ValueError("Refinement requires consecutive observed optimizer updates")
        if (
            type(remaining_updates) is not int
            or remaining_updates < 0
            or type(allow_edits) is not bool
        ):
            raise ValueError("Invalid recovery reserve or edit permission")
        if (
            type(reserved_capacity) is not int
            or not 0 <= reserved_capacity < self.config.max_gaussians
        ):
            raise ValueError("Invalid reserved capacity")
        ceiling = self.config.max_gaussians - reserved_capacity
        self.topology._check()
        if field.count > ceiling:
            raise ValueError("Population exceeds the resource ceiling")
        # Validate public insertion learning rates even when no proposal wins.
        from .field import _check_learning_rates

        _check_learning_rates(**learning_rates)
        global_step = self.start_step + step
        if (
            not allow_edits
            or remaining_updates < max(self.config.minimum_age, self.config.comparison_steps)
            or global_step <= self.recovery_until
        ):
            self.last_update, self.pending = step, False
            if not allow_edits:
                self.scores.clear()
                self.losses.clear()
                self.gradient = None
            return None
        proposals = self._proposals(global_step, learning_rates)
        if not proposals:
            self.last_update, self.pending = step, False
            return None
        if self.config.comparison_steps and not callable(evaluate):
            raise ValueError("Guarded refinement requires a training-only evaluation callback")
        before = GaussianCheckpoint()
        if not before.consider(0.0, 0, field, optimizer):
            raise FloatingPointError("Cannot snapshot refinement event")
        metadata = self.topology.state_dict()
        gradients = {
            p: None if p.grad is None else p.grad.detach().clone() for p in field.parameters()
        }
        saved_scores = deepcopy(self.scores)
        saved_progress = self.last_update, self.pending, self.recovery_until
        reference = field().detach()
        accepted, rejected, used = [], [], set()
        try:
            for gain, edit in proposals:
                if any(parent in used for parent in edit.parents):
                    continue
                change = {
                    "insert": 1,
                    "clone": 1,
                    "split": 1,
                    "merge": -1,
                    "prune": -1,
                    "reset": 0,
                    "relocate": 0,
                }[edit.operation]
                if not self.config.minimum_gaussians <= field.count + change <= ceiling:
                    continue
                try:
                    event = self.topology.apply(
                        edit,
                        optimizer,
                        step=global_step,
                        max_field_change=self.config.max_field_change,
                        learning_rates=learning_rates,
                        reference_velocity=reference,
                    )
                except EditRejected as error:
                    rejected.append(
                        {
                            "operation": edit.operation,
                            "parents": list(edit.parents),
                            "reason": str(error),
                        }
                    )
                    continue
                event["predicted_training_decrease"] = gain
                accepted.append(event)
                used.update(edit.parents)
                if len(accepted) >= self.config.max_edits:
                    break
            attempted = accepted
            trial = None
            if accepted and self.config.comparison_steps:
                keep, trial = self._assess(field, optimizer, before, metadata, gradients, evaluate)
                # Replay the forecast's ordinary updates without intervening
                # topology edits, including after rejecting the candidate.
                self.recovery_until = global_step + self.config.comparison_steps
                if not keep:
                    accepted, used = [], set()
            active = {i for ids in self.topology.ids for i in ids}
            self.scores = {
                i: score for i, score in self.scores.items() if i in active and i not in used
            }
            self.last_update, self.pending = step, False
            event = {
                "step": step,
                "global_step": global_step,
                "candidates": len(proposals),
                "operations": accepted,
                "rejected": rejected,
                "after_count": field.count,
                "maximum_velocity_change_m_s": float((field() - reference).abs().max()),
                "additional_forward_calls": 0,
                "additional_adjoint_calls": 0,
            }
            if trial is not None:
                event.update(proposed_operations=attempted, trial=trial)
                event["additional_forward_calls"] = trial["work"]["evaluations"]
                event["additional_adjoint_calls"] = trial["work"]["backwards"]
            if self.config.insertion_screening == "coverage_aware" or trial is not None:
                event["objective_change"] = {
                    "predicted_decrease": sum(op["predicted_training_decrease"] for op in attempted),
                    "measured_equal_work_decrease": (
                        trial["ordinary"]["objective"] - trial["proposed"]["objective"]
                        if trial is not None and trial["proposed"] is not None else None
                    ),
                    "measurement": "training_recovery" if trial is not None else "not_measured",
                    "interpretation": (
                        "Prediction is a local descent proxy (including prospective newborn learning); "
                        "measurement compares edited and unedited recovery endpoints, even if rejected"
                    ),
                }
            return event
        except Exception:
            before.restore(field, optimizer)
            for parameter, gradient in gradients.items():
                parameter.grad = gradient
            self.topology.load_state_dict(metadata)
            self.scores = saved_scores
            self.last_update, self.pending, self.recovery_until = saved_progress
            raise

    def _assess(self, field, optimizer, before, metadata, gradients, evaluate):
        """Compare equal recovery horizons, restoring only the selected anchor.

        Evaluated updates are speculative and counted. They never advance the
        inversion's trajectory or alter checkpoint selection frequency.
        """
        candidate = GaussianCheckpoint()
        if not candidate.consider(0.0, 0, field, optimizer):
            raise FloatingPointError("Cannot snapshot a refinement candidate")
        candidate_metadata = self.topology.state_dict()
        candidate_gradients = {
            p: None if p.grad is None else p.grad.detach().clone() for p in field.parameters()
        }
        work = {"evaluations": 0, "backwards": 0, "optimizer_updates": 0}
        ordinary_anchor, proposed_anchor = [], []

        def restore(snapshot, identities, derivatives):
            snapshot.restore(field, optimizer)
            self.topology.load_state_dict(identities)
            for p, gradient in derivatives.items():
                p.grad = None if gradient is None else gradient.clone()

        restore(before, metadata, gradients)
        ordinary = recovery_endpoint(field, optimizer, evaluate, self.config.comparison_steps, work,
                                     anchor_scores=ordinary_anchor)
        ordinary_work = dict(work)
        restore(candidate, candidate_metadata, candidate_gradients)
        proposed = None
        failure = None
        try:
            proposed = recovery_endpoint(
                field, optimizer, evaluate, self.config.comparison_steps, work,
                anchor_scores=proposed_anchor,
            )
        except FloatingPointError as error:
            failure = str(error)
        support = None
        keep = False
        if proposed is not None:
            keep, support = ordinary.accepts(
                proposed,
                rtol=self.config.acceptance_rtol,
                atol=self.config.acceptance_atol,
                shot_support=self.config.minimum_shot_support,
            )
        if keep:
            restore(candidate, candidate_metadata, candidate_gradients)
        else:
            restore(before, metadata, gradients)
            # Allocated IDs remain retired even when their forecast loses.
            self.topology.next_id = max(self.topology.next_id, candidate_metadata["next_id"])
        return keep, {
            "accepted": keep,
            "recovery_updates": self.config.comparison_steps,
            "ordinary": asdict(ordinary),
            "proposed": None if proposed is None else asdict(proposed),
            "supported_shot_fraction": support,
            "candidate_numerical_failure": failure,
            "work": work,
            "branch_work": {
                "ordinary": ordinary_work,
                "proposed": {name: value - ordinary_work[name] for name, value in work.items()},
            },
            "anchors": {
                "ordinary": asdict(ordinary_anchor[0]),
                "proposed": asdict(proposed_anchor[0]) if proposed_anchor else None,
            },
            "objective_changes": {
                "ordinary_after_minus_anchor": ordinary.objective - ordinary_anchor[0].objective,
                "proposed_after_minus_anchor": (
                    proposed.objective - proposed_anchor[0].objective if proposed is not None else None
                ),
                "proposed_minus_ordinary_after_recovery": (
                    proposed.objective - ordinary.objective if proposed is not None else None
                ),
                "proposed_minus_ordinary_at_anchor": (
                    proposed_anchor[0].objective - ordinary_anchor[0].objective
                    if proposed_anchor else None
                ),
            },
        }
