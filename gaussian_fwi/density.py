"""Direct density control translated from visual Gaussian splatting to FWI.

The improved strategy follows the components of Deng et al., CVPR Findings
2026. Waveform adjoints replace image-space attribution; signed integral
allocation replaces optical opacity transfer. This is an FWI adaptation,
not an RGB-rendering reproduction or a posterior sampler. See DENSITY.md.
"""

import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from fwi_core.physics import WaveformObjective, lowpass

from ._optimizer import validate_adam
from .adaptation import GaussianCheckpoint, _admissible, _block_from_tensors, _checked_gradient
from .field import GaussianBlock, GaussianField
from .raster import kernel, pair_chunks


@dataclass(frozen=True)
class DensityControlConfig:
    """Direct topology policy with SI-unit thresholds and explicit stage budgets.

    Events run after completed optimizer updates. ``strategy='improved'`` uses
    edge-weighted sampling, absolute center derivatives, long-axis splitting,
    recovery pruning and a square-root growth budget. ``'original'`` is a
    control using mean center-gradient norms, exact coefficient copying and
    sampled binary splits. Both retain broad kernels and prune signed amplitude
    magnitudes; neither has an optical opacity or a branch acceptance test.
    """

    strategy: str = "improved"
    warmup_steps: int = 25
    interval: int = 25
    settling_steps: int = 10
    gradient_threshold: float = 0.0
    max_growth: int = 32
    growth_per_stage: int = 128
    split_scale_fraction: float = 0.1
    split_fraction: float = 0.45
    prune_amplitude: float = 0.5
    max_prunes: int = 32
    max_prune_change: float = 5.0
    reset_steps: tuple[int, ...] = (25, 50)
    recovery_steps: int = 5
    reset_amplitude: float = 10.0
    recovery_fraction: float = 0.2
    edge_strength: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "reset_steps", tuple(self.reset_steps))
        if self.strategy not in ("original", "improved"):
            raise ValueError("Unknown direct density-control strategy")
        for name in ("warmup_steps", "interval", "settling_steps", "recovery_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_growth", "growth_per_stage", "max_prunes", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**63:
            raise ValueError("seed must fit a nonnegative signed 64-bit integer")
        for name in ("gradient_threshold", "prune_amplitude", "edge_strength"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("split_scale_fraction", "max_prune_change", "reset_amplitude"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.split_fraction) or not 0 < self.split_fraction <= 0.5:
            raise ValueError("split_fraction must lie in (0, 0.5]")
        if not math.isfinite(self.recovery_fraction) or not 0 <= self.recovery_fraction < 1:
            raise ValueError("recovery_fraction must lie in [0, 1)")
        if any(type(s) is not int or s < 1 for s in self.reset_steps) or any(
            b - a <= self.recovery_steps for a, b in zip(self.reset_steps, self.reset_steps[1:])
        ):
            raise ValueError("Reset steps must increase with non-overlapping recovery windows")

    def growth_due(self, step: int, steps_per_stage: int) -> bool:
        return (
            step >= self.warmup_steps
            and (step - self.warmup_steps) % self.interval == 0
            and step <= steps_per_stage - self.settling_steps
        )

    def validate_stage(self, steps_per_stage: int) -> None:
        """Reject unusable event windows before an inversion mutates its field."""
        if type(steps_per_stage) is not int or steps_per_stage < 1:
            raise ValueError("Density control requires a positive integer stage length")
        stop = steps_per_stage - self.settling_steps
        if stop < self.warmup_steps:
            raise ValueError("Stage must allow density warmup and settling")
        if self.strategy == "improved" and any(
            s < self.warmup_steps or s + self.recovery_steps > stop for s in self.reset_steps
        ):
            raise ValueError("Reset and recovery windows must fit inside the density phase")


class WaveformEdges:
    """A training-only detail objective used exclusively to rank density proposals.

    Temporal second differences of each filtered observation define bounded
    nonnegative weights. The extra adjoint of this objective is counted by the
    acquisition. It is never added to the optimized or validation objective.
    """

    def __init__(self, objective: WaveformObjective, strength: float):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Edge strength must be finite and nonnegative")
        self.objective = objective
        self.weights = {}
        ids = objective.split["train"]
        for cutoff in objective.cutoffs:
            target = objective.targets[cutoff][:, ids]
            edge = torch.zeros_like(target)
            edge[..., 1:-1] = (target[..., 2:] - 2 * target[..., 1:-1] + target[..., :-2]).abs()
            maximum = edge.max()
            if maximum > 0:
                edge = edge / maximum
            self.weights[cutoff] = (1 + strength * edge).detach()

    def loss(self, prediction: Tensor, active) -> Tensor:
        objective = self.objective
        ids = objective.split["train"]
        values = []
        for cutoff in active:
            residual = lowpass(prediction[:, ids], objective.dt, cutoff)
            residual = (residual - objective.targets[cutoff][:, ids]) * objective.weights[cutoff][
                :, ids
            ]
            values.append(
                (self.weights[cutoff] * residual.square()).mean()
                / objective.denominators[cutoff]["train"]
            )
        value = torch.stack(values).mean()
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite waveform detail objective")
        return value


@torch.no_grad()
def spatial_scores(field: GaussianField, gradient: Tensor, edge_gradient: Tensor | None = None):
    """Stream absolute cellwise center derivatives and detail-contribution scores.

    Absolute values precede spatial accumulation. This prevents opposite cell
    contributions from hiding a large center sensitivity. Derivatives use the
    actual tapered decoder kernel, not an untapered Gaussian approximation.
    Wave-equation contributions have already been summed by the adjoint; this
    does not remove cancellation between individual shots or receiver samples.
    """
    g = _checked_gradient(field, gradient)
    e = None if edge_gradient is None else _checked_gradient(field, edge_gradient).abs()
    points = field._points.detach().cpu().double().numpy()
    gradients, edges = [], []
    for block in field.blocks:
        centers = block.centers.detach().cpu().double().numpy()
        precision = block.precision().detach().cpu().double().numpy()
        amplitude = block.amplitudes.detach().double()
        accumulated = g.new_zeros((len(centers), field.grid.ndim))
        importance = g.new_zeros(len(centers))
        for gg, xx in pair_chunks(points, centers, precision, 6.0, field.max_pairs):
            ids = torch.as_tensor(gg, device=g.device)
            locations = torch.as_tensor(xx, device=g.device)
            delta = torch.as_tensor(points[xx] - centers[gg], device=g.device)
            matrix = torch.as_tensor(precision[gg], device=g.device)
            metric = torch.einsum("nij,nj->ni", matrix, delta)
            values, derivative = kernel((delta * metric).sum(-1))
            terms = -2 * (g[locations] * amplitude[ids] * derivative)[:, None] * metric
            accumulated.index_add_(0, ids, terms.abs())
            if e is not None:
                importance.index_add_(0, ids, amplitude[ids].abs() * values * e[locations])
        gradients.append(torch.linalg.vector_norm(accumulated, dim=-1))
        edges.append(importance)
    if any(not torch.isfinite(t).all() for t in [*gradients, *edges]):
        raise FloatingPointError("Non-finite density-control score")
    return gradients, edges


@torch.no_grad()
def long_axis_children(center: Tensor, covariance: Tensor, amplitude: Tensor, fraction: float):
    """LAS geometry with signed integral and centroid preservation in 2D or 3D.

    Offsets use the reference implementation's three-sigma support radius.
    Long radii shrink by 1-f; transverse radii by sqrt(1-f²). Unlike the earlier
    moment-preserving split, LAS does not preserve covariance. Optical opacity
    scaling is replaced by the determinant ratio for an additive velocity field.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 0.5:
        raise ValueError("LAS fraction must lie in (0, 0.5]")
    d = center.numel()
    if center.ndim != 1 or d not in (2, 3) or covariance.shape != (d, d) or amplitude.ndim != 0:
        raise ValueError("Invalid long-axis parent tensor shapes")
    if any(not torch.isfinite(t).all() for t in (center, covariance, amplitude)):
        raise FloatingPointError("Non-finite long-axis parent")
    matrix = covariance.double()
    if not torch.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-12):
        raise ValueError("Parent covariance must be symmetric")
    values, axes = torch.linalg.eigh(matrix)
    if not (values > 0).all():
        raise ValueError("Parent covariance must be positive definite")
    direction = axes[:, -1]
    # Fix the eigenvector sign so row ordering is reproducible.
    if direction[direction.abs().argmax()] < 0:
        direction = -direction
    # The official LAS geometry defines its ellipsoid at three standard
    # deviations. The radius convention affects offsets, not scale ratios.
    offset = 3.0 * fraction * values[-1].sqrt() * direction
    shrink = values.new_full((d,), 1 - fraction**2)
    shrink[-1] = (1 - fraction) ** 2
    child_covariance = (axes * (values * shrink)[None, :]) @ axes.T
    allocation = 1 / (2 * (1 - fraction) * (1 - fraction**2) ** ((d - 1) / 2))
    return (
        torch.stack((center.double() - offset, center.double() + offset)),
        child_covariance.expand(2, d, d).clone(),
        amplitude.double().expand(2).clone() * allocation,
        allocation,
    )


@torch.no_grad()
def _replace_rows(field, optimizer, removed, children):
    """Prepare topology and survivor Adam history before one coordinated commit.

    ``children`` maps a parent block to (centers, covariances, amplitudes, LR
    allocation). Newly created parameters have independent, empty Adam state.
    Unaffected blocks retain identities; survivor rows retain every Adam option,
    moment, and step. Empty blocks are removed, including the last block.
    """
    replacement, blocks, appended, child_groups = {}, [], [], []
    origins = {p: group for group in optimizer.param_groups for p in group["params"]}
    for bi, old in enumerate(field.blocks):
        rows = removed.get(bi, [])
        if not rows:
            blocks.append(old)
        else:
            if len(set(rows)) != len(rows) or any(
                type(r) is not int or not 0 <= r < len(old.centers) for r in rows
            ):
                raise ValueError("Removal rows must be unique existing integer indices")
            keep = torch.ones(len(old.centers), dtype=torch.bool, device=old.centers.device)
            keep[rows] = False
            new = None
            if keep.any():
                new = GaussianBlock(old.centers[keep], old.log_scales[keep].exp())
                new.nominal_scale, new.amplitude_lr_scale = (
                    old.nominal_scale,
                    old.amplitude_lr_scale,
                )
                for name, parameter in new.named_parameters():
                    previous = getattr(old, name)
                    parameter.copy_(previous[keep])
                    if previous.grad is not None:
                        parameter.grad = previous.grad[keep].clone()
                blocks.append(new)
            for name, parameter in old.named_parameters():
                replacement[parameter] = (None if new is None else getattr(new, name), keep)
        if bi in children:
            centers, covariance, amplitude, allocation = children[bi]
            child = _block_from_tensors(centers, covariance, amplitude, field.background)
            child.nominal_scale = old.nominal_scale
            child.amplitude_lr_scale = old.amplitude_lr_scale * allocation
            child.parameter_groups()
            appended.append(child)
            for names, factor in (
                (("amplitudes",), allocation),
                (("centers",), 1.0),
                (("log_scales", "shears"), 1.0),
            ):
                group = origins[getattr(old, names[0])]
                child_groups.append(
                    {
                        **deepcopy({k: v for k, v in group.items() if k != "params"}),
                        "params": [getattr(child, name) for name in names],
                        "lr": group["lr"] * factor,
                    }
                )
    if any(type(bi) is not int or not 0 <= bi < len(field.blocks) for bi in {*removed, *children}):
        raise ValueError("Topology change references an unknown block")
    state = defaultdict(dict)
    groups = []
    for group in optimizer.param_groups:
        parameters = []
        for old in group["params"]:
            new, keep = replacement.get(old, (old, None))
            if new is None:
                continue
            parameters.append(new)
            if old in optimizer.state:
                state[new] = {
                    key: value[keep].clone()
                    if keep is not None and isinstance(value, Tensor) and value.shape == old.shape
                    else deepcopy(value)
                    for key, value in optimizer.state[old].items()
                }
        if parameters:
            groups.append({**group, "params": parameters})
    field.blocks = nn.ModuleList([*blocks, *appended])
    optimizer.state = state
    optimizer.param_groups = groups
    for group in child_groups:
        optimizer.add_param_group(group)


class DensityController:
    """State for one frequency stage; stage checkpoints restart with fresh windows.

    Random sampling uses local CPU generators keyed by stage and update. Global
    random state is untouched. This driver supports stage-boundary restart,
    matching the inversion API; mid-stage restart is not implied.
    """

    def __init__(self, field, config, steps_per_stage, stage=0):
        if (
            type(steps_per_stage) is not int
            or steps_per_stage < 1
            or type(stage) is not int
            or stage < 0
        ):
            raise ValueError("Invalid density-control stage")
        if not isinstance(config, DensityControlConfig):
            raise TypeError("Expected DensityControlConfig")
        config.validate_stage(steps_per_stage)
        self.config, self.steps, self.stage = config, steps_per_stage, stage
        self.initial_count, self.last_update = field.count, 0
        self.born = [torch.zeros(len(b.centers), dtype=torch.int64) for b in field.blocks]
        self._clear(field)

    def _clear(self, field):
        self.blocks = tuple(field.blocks)
        self.sums = [torch.zeros(len(b.centers), dtype=torch.float64) for b in field.blocks]
        self.samples = 0
        self.edge_scores = None

    def _check(self, field):
        if len(self.blocks) != len(field.blocks) or any(
            a is not b for a, b in zip(self.blocks, field.blocks)
        ):
            raise ValueError("Density controller topology changed outside its transaction")

    @torch.no_grad()
    def observe(self, field, gradient, edge_gradient=None):
        self._check(field)
        if self.config.strategy == "improved":
            scores, edges = spatial_scores(field, gradient, edge_gradient)
        else:
            if any(
                b.centers.grad is None or not torch.isfinite(b.centers.grad).all()
                for b in field.blocks
            ):
                raise FloatingPointError(
                    "Original density control requires finite center gradients"
                )
            scores = [
                torch.linalg.vector_norm(b.centers.grad.double(), dim=-1) for b in field.blocks
            ]
            edges = None
        updated = [total + score.detach().cpu() for total, score in zip(self.sums, scores)]
        if any(not torch.isfinite(t).all() for t in updated):
            raise FloatingPointError("Density score accumulation overflow")
        self.sums, self.samples = updated, self.samples + 1
        self.edge_scores = (
            [value.detach().cpu() for value in edges]
            if edge_gradient is not None and edges is not None
            else None
        )

    def needs_edge(self, step):
        return self.config.strategy == "improved" and self.config.growth_due(step, self.steps)

    @torch.no_grad()
    def after_update(self, field, optimizer, step):
        """Apply all operations due after this update, restoring state on failure."""
        self._check(field)
        if type(step) is not int or step != self.last_update + 1 or step > self.steps:
            raise ValueError("Density updates must be consecutive completed stage steps")
        cfg = self.config
        grow = cfg.growth_due(step, self.steps)
        reset = cfg.strategy == "improved" and step in cfg.reset_steps
        recovery = cfg.strategy == "improved" and step - cfg.recovery_steps in cfg.reset_steps
        if not (grow or reset or recovery):
            self.last_update = step
            return None
        validate_adam(field, optimizer)
        before = GaussianCheckpoint()
        if not before.consider(0.0, 0, field, optimizer):
            raise FloatingPointError("Non-finite field before density control")
        saved = (
            self.blocks,
            self.sums,
            self.samples,
            self.born,
            self.last_update,
            self.edge_scores,
        )
        try:
            before_velocity = field().detach()
            event = self._event(field, optimizer, step, grow, reset, recovery)
            validate_adam(field, optimizer)
            after_velocity = field().detach()
            if not torch.isfinite(after_velocity).all():
                raise FloatingPointError("Non-finite decoded field after density control")
            event["maximum_velocity_change_m_s"] = float(
                (after_velocity - before_velocity).abs().max()
            )
            self.last_update = step
            self._clear(field)
            return event
        except Exception:
            before.restore(field, optimizer)
            self.blocks, self.sums, self.samples, self.born, self.last_update, self.edge_scores = (
                saved
            )
            raise

    def _event(self, field, optimizer, step, grow, reset, recovery):
        cfg = self.config
        key = f"{cfg.seed}:{self.stage}:{step}".encode("ascii")
        event_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(event_seed)
        rows = [(bi, row) for bi, b in enumerate(field.blocks) for row in range(len(b.centers))]
        magnitude = {(bi, row): float(field.blocks[bi].amplitudes[row].abs()) for bi, row in rows}
        removable = {key for key in rows if grow and magnitude[key] < cfg.prune_amplitude}
        if recovery:
            cohort = [
                key for key in rows if int(self.born[key[0]][key[1]]) <= step - cfg.recovery_steps
            ]
            cohort.sort(key=lambda key: (magnitude[key], *key))
            removable.update(cohort[: math.floor(len(cohort) * cfg.recovery_fraction)])
        chosen_prunes, bound = [], 0.0
        for key in sorted(removable, key=lambda key: (magnitude[key], *key)):
            if (
                len(chosen_prunes) >= cfg.max_prunes
                or bound + magnitude[key] > cfg.max_prune_change
            ):
                break
            chosen_prunes.append(key)
            bound += magnitude[key]
        removed = defaultdict(list)
        for bi, row in chosen_prunes:
            removed[bi].append(row)
        selected, geometry = [], defaultdict(list)
        candidates = []
        if grow:
            if self.samples == 0:
                raise ValueError("Density growth requires an observed gradient window")
            for key in rows:
                score = float(self.sums[key[0]][key[1]]) / self.samples
                if key not in chosen_prunes and score > cfg.gradient_threshold:
                    candidates.append((*key, score))
        last = self.steps - cfg.settling_steps
        progress = min(
            1.0,
            max(
                0.0,
                (step - cfg.warmup_steps + cfg.interval) / (last - cfg.warmup_steps + cfg.interval),
            ),
        )
        budget = self.initial_count + math.floor(
            cfg.growth_per_stage * (math.sqrt(progress) if cfg.strategy == "improved" else 1)
        )
        available = max(0, min(cfg.max_growth, budget - (field.count - len(chosen_prunes))))
        if candidates and available:
            if cfg.strategy == "improved":
                if self.edge_scores is None:
                    raise ValueError(
                        "Improved growth requires the training waveform detail adjoint"
                    )
                weights = torch.tensor(
                    [float(self.edge_scores[bi][row]) for bi, row, _ in candidates],
                    dtype=torch.float64,
                )
                if not bool(weights.any()):
                    weights = torch.tensor(
                        [score for _, _, score in candidates], dtype=torch.float64
                    )
                positive = torch.nonzero(weights > 0).flatten()
                order = positive[
                    torch.multinomial(
                        weights[positive] / weights[positive].max(),
                        len(positive),
                        replacement=False,
                        generator=generator,
                    )
                ].tolist()
            else:
                order = sorted(
                    range(len(candidates)),
                    key=lambda i: (-candidates[i][2], candidates[i][0], candidates[i][1]),
                )
            for index in order:
                bi, row, score = candidates[index]
                block = field.blocks[bi]
                center, covariance, amplitude = (
                    block.centers[row],
                    block.covariance()[row],
                    block.amplitudes[row],
                )
                if cfg.strategy == "improved":
                    centers, covariances, amplitudes, allocation = long_axis_children(
                        center, covariance, amplitude, cfg.split_fraction
                    )
                    operation = "split"
                elif torch.linalg.eigvalsh(covariance.double())[
                    -1
                ].sqrt() <= cfg.split_scale_fraction * max(field.grid.extent):
                    centers, covariances, amplitudes, allocation = (
                        center[None],
                        covariance[None],
                        amplitude[None],
                        1.0,
                    )
                    operation = "clone"
                else:
                    normal = torch.randn(
                        (2, field.grid.ndim), dtype=torch.float64, generator=generator
                    ).to(center.device)
                    centers = (
                        center.double() + normal @ torch.linalg.cholesky(covariance.double()).T
                    )
                    covariances = (
                        (covariance.double() / 1.6**2)
                        .expand(2, field.grid.ndim, field.grid.ndim)
                        .clone()
                    )
                    amplitudes, allocation = amplitude.double().expand(2).clone(), 1.0
                    operation = "split"
                if not _admissible(field, centers, covariances):
                    continue
                if operation == "split":
                    removed[bi].append(row)
                geometry[bi].append((centers, covariances, amplitudes, allocation))
                selected.append({"block": bi, "row": row, "operation": operation, "score": score})
                if len(selected) == available:
                    break
        children = {
            bi: (*[torch.cat([entry[k] for entry in entries]) for k in range(3)], entries[0][3])
            for bi, entries in geometry.items()
        }
        born = []
        for bi, original in enumerate(self.born):
            keep = torch.ones(len(original), dtype=torch.bool)
            keep[removed.get(bi, [])] = False
            if keep.any():
                born.append(original[keep].clone())
        born.extend(
            torch.full((len(children[bi][0]),), step, dtype=torch.int64) for bi in sorted(children)
        )
        event = {
            "strategy": cfg.strategy,
            "rng_seed": event_seed,
            "step": step,
            "before_count": field.count,
            "gradient_samples": self.samples,
            "growth_due": grow,
            "growth_budget": budget,
            "candidates": len(candidates),
            "operations": selected,
            "clones": sum(item["operation"] == "clone" for item in selected),
            "splits": sum(item["operation"] == "split" for item in selected),
            "pruned": [{"block": bi, "row": row} for bi, row in chosen_prunes],
            "prune_change_bound_m_s": bound,
            "recovery_due": recovery,
            "reset_due": reset,
        }
        if removed or children:
            _replace_rows(field, optimizer, removed, children)
        self.born = born
        reset_count = 0
        if reset:
            for block in field.blocks:
                reset_count += int((block.amplitudes.abs() > cfg.reset_amplitude).sum())
                block.amplitudes.clamp_(-cfg.reset_amplitude, cfg.reset_amplitude)
                state = optimizer.state.get(block.amplitudes, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key].zero_()
        event.update(after_count=field.count, reset_amplitudes=reset_count)
        return event
