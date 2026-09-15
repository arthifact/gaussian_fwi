"""One scheduled adaptive density-control policy for Gaussian FWI.

The clone/split/prune cycle follows Kerbl et al. (ACM TOG, 2023, section 5).
Physical training-objective center gradients replace projected image gradients.
Signed velocity amplitudes replace optical opacity for bounded pruning. See
docs/ALGORITHM.md for the exact translation and its empirical limitations.
"""

import math
from dataclasses import dataclass

import torch

from ._topology import EditRejected, GaussianTopology


@dataclass(frozen=True)
class RefinementConfig:
    """A warm-up, periodic density phase and final settling phase per band.

    ``gradient_threshold`` has units 1/m for the normalized training objective.
    It is not the image-space threshold from 3DGS. ``split_extent_fraction``
    compares the largest principal standard deviation with the largest domain
    extent. Resource limits are ceilings, not a target population.
    """

    warmup_steps: int = 50
    interval: int = 50
    stop_fraction: float = 0.5
    gradient_threshold: float = 0.0
    split_extent_fraction: float = 0.01
    max_gaussians: int = 8192
    max_growth: int = 128
    max_prunes: int = 128
    minimum_gaussians: int = 1
    minimum_age: int = 50
    prune_amplitude: float = 0.5
    max_field_change: float = 25.0
    seed: int = 0

    def __post_init__(self):
        for name in ("interval", "max_gaussians", "max_growth", "max_prunes", "minimum_age"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "minimum_gaussians", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**63 or self.minimum_gaussians > self.max_gaussians:
            raise ValueError("Invalid random seed or population limits")
        if not math.isfinite(self.stop_fraction) or not 0 < self.stop_fraction < 1:
            raise ValueError("stop_fraction must leave a nonempty settling phase")
        for name in ("gradient_threshold", "prune_amplitude"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("split_extent_fraction", "max_field_change"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def stop_step(self, steps_per_stage: int) -> int:
        return math.ceil(self.stop_fraction * steps_per_stage)

    def validate_stage(self, steps_per_stage: int):
        if type(steps_per_stage) is not int or steps_per_stage < 1:
            raise ValueError("A positive stage length is required")
        first_event = (self.warmup_steps // self.interval + 1) * self.interval
        if first_event >= self.stop_step(steps_per_stage):
            raise ValueError("Stage must contain warm-up, a refinement opportunity and settling")

    def due(self, completed_step: int, steps_per_stage: int) -> bool:
        return (self.warmup_steps < completed_step < self.stop_step(steps_per_stage)
                and completed_step % self.interval == 0)


class RefinementController:
    """Accumulate mean physical center-gradient norms and apply bounded edits.

    Observe once after the ordinary training backward pass, then call
    ``after_update`` after Adam and geometric projection. Topology decisions use
    no validation/test information and perform no acoustic solves. Statistics
    are reset after every event and at each frequency change.
    """

    def __init__(self, field, config, *, steps_per_stage, stage=0, topology_state=None):
        if not isinstance(config, RefinementConfig) or type(stage) is not int or stage < 0:
            raise ValueError("Invalid refinement configuration or frequency stage")
        config.validate_stage(steps_per_stage)
        if field.count > config.max_gaussians:
            raise ValueError("Initial population exceeds the Gaussian ceiling")
        self.config, self.steps_per_stage, self.stage = config, steps_per_stage, stage
        self.topology = GaussianTopology(field, topology_state)
        self.start_step = stage * steps_per_stage
        if any(age > self.start_step for ages in self.topology.last_edit_steps for age in ages):
            raise ValueError("Topology history lies after the starting stage")
        self.last_step, self.pending = 0, False
        self.scores = {}

    @torch.no_grad()
    def observe(self, field):
        if field is not self.topology.field or self.pending:
            raise ValueError("Observe exactly once before each update on the registered field")
        self.topology._check()
        if self.last_step + 1 < self.config.stop_step(self.steps_per_stage):
            current = {}
            for block, ids in zip(field.blocks, self.topology.ids, strict=True):
                gradient = block.centers.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError("Physical center gradients must be finite and present")
                norms = torch.linalg.vector_norm(gradient.detach().double(), dim=-1).cpu().tolist()
                for kernel_id, norm in zip(ids, norms, strict=True):
                    total, count = self.scores.get(kernel_id, (0.0, 0))
                    if not math.isfinite(total + norm):
                        raise FloatingPointError("Center-gradient accumulation overflow")
                    current[kernel_id] = (total + norm, count + 1)
            self.scores = current
        self.pending = True

    @torch.no_grad()
    def after_update(self, field, optimizer, completed_step):
        if (field is not self.topology.field or not self.pending
                or type(completed_step) is not int
                or completed_step != self.last_step + 1
                or completed_step > self.steps_per_stage):
            raise ValueError("Density control requires consecutive completed updates")
        event = None
        if self.config.due(completed_step, self.steps_per_stage):
            event = self._event(field, optimizer, completed_step)
            self.scores = {}
        self.last_step, self.pending = completed_step, False
        return event

    def _event(self, field, optimizer, step):
        cfg, topology = self.config, self.topology
        global_step = self.start_step + step
        eligible = [kernel_id for ids, ages in zip(topology.ids, topology.last_edit_steps)
                    for kernel_id, age in zip(ids, ages)
                    if global_step - age >= cfg.minimum_age and kernel_id in self.scores]
        values = {i: topology.values(i) for i in eligible}
        means = {i: self.scores[i][0] / self.scores[i][1] for i in eligible}
        weak = sorted((i for i in eligible if abs(float(values[i][2])) <= cfg.prune_amplitude),
                      key=lambda i: (abs(float(values[i][2])), i))
        weak = weak[:min(cfg.max_prunes, max(0, field.count - cfg.minimum_gaussians))]
        excluded = set(weak)
        candidates = sorted((i for i in eligible if i not in excluded
                             and means[i] > cfg.gradient_threshold), key=lambda i: (-means[i], i))
        available = min(cfg.max_growth, max(0, cfg.max_gaussians - field.count + len(weak)))
        threshold = cfg.split_extent_fraction * max(field.grid.extent)
        edits = [topology.prune(i) for i in weak]
        rejected_geometry = 0
        growth = 0
        for kernel_id in candidates:
            if growth >= available:
                break
            radius = float(torch.linalg.eigvalsh(values[kernel_id][1].double())[-1].sqrt())
            if radius <= threshold:
                edit = topology.clone(kernel_id)
            else:
                # A separate deterministic stream per parent/event avoids global
                # RNG mutation and makes rejected proposals replayable.
                seed = (cfg.seed + 104729 * global_step + 1000003 * kernel_id) % (2**63)
                edit = topology.split(kernel_id, seed=seed)
            try:
                topology._validate_edit(edit)
            except EditRejected:
                rejected_geometry += 1
                continue
            edits.append(edit)
            growth += 1
        before_count = field.count
        attempts, rejected_change = 0, 0
        accepted = None
        proposed = len(edits)
        # Each attempt is atomic. A rejected batch is halved in its deterministic
        # priority order; all accepted edits share one cumulative velocity bound.
        while edits:
            attempts += 1
            try:
                accepted = topology.apply_many(edits, optimizer, step=global_step,
                                               max_field_change=cfg.max_field_change)
                break
            except EditRejected:
                keep = len(edits) // 2
                rejected_change += len(edits) - keep
                edits = edits[:keep]
        return {
            "stage": self.stage, "step": step, "global_step": global_step,
            "before_count": before_count, "after_count": field.count,
            "eligible": len(eligible), "proposed": proposed, "attempts": attempts,
            "rejected_geometry": rejected_geometry, "rejected_field_change": rejected_change,
            "gradient_threshold_per_m": cfg.gradient_threshold,
            "split_radius_threshold_m": threshold,
            "operations": [] if accepted is None else accepted["operations"],
            "maximum_velocity_change_m_s": (0.0 if accepted is None
                                             else accepted["maximum_velocity_change_m_s"]),
            "additional_solver_calls": {"forward": 0, "adjoint": 0},
        }
