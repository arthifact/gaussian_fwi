"""Gaussian velocity inversion with staged capacity and frequency continuation."""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from fwi_core.checkpoint import BestCheckpoint, prepare_output, save_json, save_torch
from fwi_core.physics import Observations, Preprocessing, WaveformObjective
from fwi_core.regularization import Regularization

from ._optimizer import validate_adam
from .adaptation import AdaptationConfig, GaussianCheckpoint, ObjectiveValues, adaptive_step
from .density import DensityControlConfig, DensityController, WaveformEdges
from .field import GaussianField
from .refinement import RefinementConfig, RefinementController
from .sampling import sampling_diagnostics
from .spatial import (
    normalize_radius_schedule,
    radius_record,
    set_radius_floor,
    validate_spatial_start,
)
from .topology import GaussianTopology
from .trials import TrainingScores

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class InversionConfig:
    """Gaussian optimization settings; acquisition-dependent schedules are explicit.

    ``levels`` optionally lists lattice shapes in tensor-axis order. A ``None``
    entry leaves capacity unchanged at that frequency transition; omitting
    ``levels`` uses the caller's seed field throughout. All kernels remain
    trainable, independently of the frequency schedule.
    Center learning rates are scaled by each block's initial mean width.
    """

    cutoffs: tuple[float, ...]
    levels: tuple[tuple[int, ...] | None, ...] | None = None
    steps_per_stage: int = 100
    validation_interval: int = 10
    amplitude_lr: float = 4.0
    geometry_lr: float = 0.008
    center_lr_ratio: float = 0.02
    sigma_ratio: float = 0.65
    raw_bounds_weight: float = 1e-3
    adaptation: AdaptationConfig | None = None
    density_control: DensityControlConfig | None = None
    refinement: RefinementConfig | None = None
    spatial_radius_schedule: tuple[float, ...] | None = None
    sampling_refinement_factors: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        factors = tuple(self.sampling_refinement_factors)
        if any(type(n) is not int or n < 2 for n in factors) or len(set(factors)) != len(factors):
            raise ValueError("Sampling refinement factors must be distinct integers >= 2")
        object.__setattr__(self, "sampling_refinement_factors", factors)
        object.__setattr__(
            self,
            "levels",
            tuple(None for _ in self.cutoffs)
            if self.levels is None
            else tuple(None if s is None else tuple(s) for s in self.levels),
        )
        if isinstance(self.adaptation, dict):
            object.__setattr__(self, "adaptation", AdaptationConfig(**self.adaptation))
        if self.adaptation is not None and not isinstance(self.adaptation, AdaptationConfig):
            raise TypeError("adaptation must be an AdaptationConfig or None")
        if isinstance(self.density_control, dict):
            object.__setattr__(
                self, "density_control", DensityControlConfig(**self.density_control)
            )
        if self.density_control is not None and not isinstance(
            self.density_control, DensityControlConfig
        ):
            raise TypeError("density_control must be a DensityControlConfig or None")
        if isinstance(self.refinement, dict):
            object.__setattr__(self, "refinement", RefinementConfig(**self.refinement))
        if self.refinement is not None and not isinstance(self.refinement, RefinementConfig):
            raise TypeError("refinement must be a RefinementConfig or None")
        object.__setattr__(
            self,
            "spatial_radius_schedule",
            normalize_radius_schedule(self.spatial_radius_schedule, len(self.cutoffs)),
        )
        if self.spatial_radius_schedule is not None and (
            self.adaptation is not None
            or self.density_control is not None
            or (self.refinement is not None and (
                self.refinement.directional_splits or self.refinement.relocation
            ))
            or any(level is not None for level in self.levels[1:])
        ):
            raise ValueError("Spatial continuation requires a fixed seed and direct updates")
        if (
            sum(
                policy is not None
                for policy in (self.adaptation, self.density_control, self.refinement)
            )
            > 1
        ):
            raise ValueError("Choose one topology controller for an inversion")
        if len(self.cutoffs) != len(self.levels) or not self.cutoffs:
            raise ValueError("Each frequency stage needs a level entry or None")
        if any(not math.isfinite(f) or f <= 0 for f in self.cutoffs):
            raise ValueError("Cutoffs must be finite and positive")
        if any(b <= a for a, b in zip(self.cutoffs, self.cutoffs[1:])):
            raise ValueError("Cutoffs must increase")
        if any(
            type(n) is not int or n < 1 for n in (self.steps_per_stage, self.validation_interval)
        ):
            raise ValueError("Positive step counts are required")
        if self.density_control is not None:
            self.density_control.validate_stage(self.steps_per_stage)
        for name in ("amplitude_lr", "geometry_lr", "center_lr_ratio", "sigma_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.raw_bounds_weight) or self.raw_bounds_weight < 0:
            raise ValueError("raw_bounds_weight must be finite and nonnegative")

    def learning_rates(self) -> dict[str, float]:
        """Return block learning-rate arguments in physical parameter units."""
        return {
            name: getattr(self, name) for name in ("amplitude_lr", "geometry_lr", "center_lr_ratio")
        }


def add_capacity(
    field: GaussianField,
    optimizer: torch.optim.Optimizer,
    shape: tuple[int, ...],
    config: InversionConfig,
):
    """Insert inactive kernels without replacing old parameters or Adam states."""
    block = field.add_grid_level(shape, sigma_ratio=config.sigma_ratio)
    for group in block.parameter_groups(**config.learning_rates()):
        optimizer.add_param_group(group)
    return block


def invert(
    model: GaussianField,
    observations: Observations,
    config: InversionConfig,
    output: str | Path,
    *,
    partitions: Mapping[str, Tensor],
    regularization: Regularization = Regularization(),
    preprocessing: Preprocessing = Preprocessing(),
    _resume: dict | None = None,
) -> dict:
    """Fit a Gaussian velocity field using training waveforms and validation selection.

    The model is updated in place. Output must be a new or empty directory.
    Each completed stage saves a selected field and its matching Adam state.
    Use :func:`resume` to continue from a completed stage; mid-stage resume is
    intentionally not advertised. No target Earth model enters this API.
    """
    observation_identity = observations.content_identity()
    if _resume is not None:
        if _resume.get("observation_identity") is None:
            raise ValueError(
                "Legacy checkpoint has no verified observation content identity; "
                "resume with its frozen source or start a new fit. The saved field remains loadable."
            )
        if _resume["observation_identity"] != observation_identity:
            raise ValueError("Checkpoint observations do not match actual waveform and acquisition content")
    if model.grid != observations.acquisition.grid:
        raise ValueError(
            "Field and acquisition must use the same physical grid (shape and spacing); "
            "explicitly sample the continuous field on a same-domain grid for evaluation"
        )
    if model.bounds[1] > observations.acquisition.max_velocity:
        raise ValueError("The solver maximum must cover the entire allowed velocity range")
    if any(
        len(shape) != model.grid.ndim or any(type(n) is not int or n < 2 for n in shape)
        for shape in config.levels
        if shape is not None
    ):
        raise ValueError("Every Gaussian level must match the model dimension")
    if _resume is None and model.count == 0 and config.levels[0] is None:
        raise ValueError("Provide a seeded Gaussian field or an initial Gaussian level")
    future_levels = (
        config.levels if _resume is None else config.levels[_resume["completed_stage"] + 1 :]
    )
    if (
        config.refinement is not None
        and model.count + sum(math.prod(s) for s in future_levels if s is not None)
        > config.refinement.max_gaussians
    ):
        raise ValueError("Seed and scheduled capacity exceed the refinement resource ceiling")
    if any(
        policy is not None
        for policy in (config.adaptation, config.density_control, config.refinement)
    ) and (
        model.checkpoint()["format"] not in (
            "gaussian-fwi-field-v1", "gaussian-fwi-field-v2", "gaussian-fwi-field-v3"
        )
        or any(not p.requires_grad for p in model.parameters())
    ):
        raise ValueError("Adaptation requires a fully trainable GaussianField")
    spatial_base_minimum = validate_spatial_start(model, config.spatial_radius_schedule, _resume)
    if model.sampling is not None:
        radius_record(model)
    objective = WaveformObjective(observations, config.cutoffs, partitions, preprocessing)
    optimizer = torch.optim.Adam(model.parameter_groups(**config.learning_rates()))
    saved_config = asdict(config)
    if config.spatial_radius_schedule is None:
        saved_config.pop("spatial_radius_schedule")
    if not config.sampling_refinement_factors:
        saved_config.pop("sampling_refinement_factors")
    specification = {
        "config": saved_config,
        "regularization": asdict(regularization),
        "preprocessing": asdict(preprocessing),
        "partitions": {name: ids.tolist() for name, ids in objective.split.items()},
        "observation_sha256": observations.sha256,
    }
    history, stages, adaptation_history, density_history, topology_history = [], [], [], [], []
    spatial_history = []
    topology_state = None
    first_stage, prior_elapsed = 0, 0.0
    prior_counts = {"forward": 0, "adjoint": 0}
    if _resume is not None:
        saved_specification = dict(_resume["specification"])
        saved_specification["config"] = dict(saved_specification["config"])
        saved_specification["config"].setdefault("adaptation", None)
        saved_specification["config"].setdefault("density_control", None)
        saved_specification["config"].setdefault("refinement", None)
        if saved_specification["config"].get("spatial_radius_schedule") is None:
            saved_specification["config"].pop("spatial_radius_schedule", None)
        if not saved_specification["config"].get("sampling_refinement_factors"):
            saved_specification["config"].pop("sampling_refinement_factors", None)
        if saved_specification["config"]["refinement"] is not None:
            saved_specification["config"]["refinement"] = asdict(
                RefinementConfig(**saved_specification["config"]["refinement"])
            )
        if saved_specification["config"]["adaptation"] is not None:
            saved_specification["config"]["adaptation"] = dict(
                saved_specification["config"]["adaptation"]
            )
            saved_specification["config"]["adaptation"].setdefault("comparison_steps", 1)
        if saved_specification != specification:
            raise ValueError("Checkpoint observations or inversion specification do not match")
        optimizer.load_state_dict(_resume["optimizer"])
        if any(
            policy is not None
            for policy in (config.adaptation, config.density_control, config.refinement)
        ):
            validate_adam(model, optimizer)
        history, stages = _resume["history"], _resume["stages"]
        adaptation_history = list(_resume.get("adaptation_history", []))
        density_history = list(_resume.get("density_history", []))
        topology_history = list(_resume.get("topology_history", []))
        spatial_history = list(_resume.get("spatial_history", []))
        topology_state = _resume.get("topology_state")
        if config.refinement is not None and topology_state is None:
            raise ValueError("Refinement checkpoint is missing its Gaussian identity state")
        if config.refinement is not None:
            GaussianTopology(model, topology_state)
        first_stage = _resume["completed_stage"] + 1
        prior_elapsed, prior_counts = _resume["elapsed_s"], _resume["solver_calls"]
    # Validate the objective and any restart state before creating an artifact.
    # A continuation is a new run: it must never overwrite its source evidence.
    output = prepare_output(output)
    starting_counts = observations.acquisition.counts
    start = time.perf_counter()

    def elapsed() -> float:
        return prior_elapsed + time.perf_counter() - start

    def counts() -> dict[str, int]:
        current = observations.acquisition.counts
        return {
            name: prior_counts[name] + current[name] - starting_counts[name] for name in current
        }

    with torch.no_grad():
        if _resume is None:
            initial_velocity = model().clone()
            initial_prediction = observations.acquisition.simulate(initial_velocity)
        else:
            initial_velocity = _resume["initial_velocity"].to(model.background)
            initial_prediction = _resume["initial_prediction"].to(model.background)

    for stage_index in range(first_stage, len(config.cutoffs)):
        cutoff = config.cutoffs[stage_index]
        refinement = (
            RefinementController(
                model,
                config.refinement,
                start_step=stage_index * config.steps_per_stage,
                topology_state=topology_state,
            )
            if config.refinement is not None
            else None
        )
        # Verify the identity registry against the previous stage's complete
        # physical configuration before releasing its radius constraint.
        if config.spatial_radius_schedule is not None:
            set_radius_floor(model, config.spatial_radius_schedule[stage_index])
        shape = config.levels[stage_index]
        stage_block = add_capacity(model, optimizer, shape, config) if shape is not None else None
        if config.spatial_radius_schedule is not None:
            spatial_history.append({"stage": stage_index, "step": 0, **radius_record(model)})
        if refinement is not None:
            refinement.topology.register_appended(step=stage_index * config.steps_per_stage)
        nominal_scale = None
        if config.adaptation is not None:
            nominal_scale = (
                stage_block.nominal_scale
                if stage_block is not None
                else sum(b.nominal_scale * len(b.centers) for b in model.blocks)
                / max(1, model.count)
            )
        active = config.cutoffs[: stage_index + 1]
        dynamic = any(
            policy is not None
            for policy in (config.adaptation, config.density_control, config.refinement)
        )
        best = GaussianCheckpoint() if dynamic else BestCheckpoint()
        best_topology_state = None
        controller = (
            DensityController(model, config.density_control, config.steps_per_stage, stage_index)
            if config.density_control is not None
            else None
        )
        edges = (
            WaveformEdges(objective, config.density_control.edge_strength) if controller else None
        )

        def evaluate_field(field: GaussianField):
            raw = field.raw()
            velocity = field.bound_velocity(raw).reshape(field.grid.shape)
            prediction = observations.acquisition.simulate(velocity)
            bands = objective.losses(prediction, active)
            train = bands.mean()
            # Preserve the original accumulation order for ordinary and adaptive
            # runs. At refinement events both raw-field derivatives are combined.
            bounds_raw = field.raw()
            bounds_loss = ((bounds_raw - bounds_raw.clamp(*field.bounds)) / 1000).square().mean()
            penalty = regularization(velocity, field.grid.spacing)
            penalty = penalty + config.raw_bounds_weight * bounds_loss
            loss = train + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            return raw, bounds_raw, prediction, bands, train, penalty, loss

        def evaluate_update(field: GaussianField) -> tuple[Tensor, ObjectiveValues]:
            _, _, prediction, _, train, _, loss = evaluate_field(field)
            with torch.no_grad():
                validation = objective.losses(prediction.detach(), active, "validation").mean()
            return loss, ObjectiveValues(
                float(loss.detach()), float(train.detach()), float(validation)
            )

        def evaluate_training(field: GaussianField):
            _, _, prediction, _, train, _, loss = evaluate_field(field)
            with torch.no_grad():
                shots = objective.losses_by_shot(prediction.detach(), active, "train").mean(-1)
            return loss, TrainingScores(
                float(loss.detach()), float(train.detach()), tuple(shots.tolist())
            )

        for step in range(config.steps_per_stage + 1):
            terminal = step == config.steps_per_stage
            refine = config.adaptation is not None and config.adaptation.due(
                step, config.steps_per_stage
            )
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.set_grad_enabled(not terminal):
                    raw, bounds_raw, prediction, train_bands, train, penalty, loss = evaluate_field(
                        model
                    )
                    if refine or (
                        (controller is not None or refinement is not None) and not terminal
                    ):
                        raw.retain_grad()
                        bounds_raw.retain_grad()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training objective")
                selection_due = step % config.validation_interval == 0 or terminal
                if selection_due or refine:
                    with torch.no_grad():
                        validation_bands = objective.losses(
                            prediction.detach(), active, "validation"
                        )
                        validation = float(validation_bands.mean())
                    if not math.isfinite(validation):
                        raise FloatingPointError("Non-finite validation objective")
                    if selection_due:
                        selected = best.consider(validation, step, model, optimizer)
                        if selected and refinement is not None:
                            best_topology_state = refinement.topology.state_dict()
                    history.append(
                        {
                            "stage": stage_index,
                            "cutoff_hz": cutoff,
                            "step": step,
                            "selection_eligible": selection_due,
                            "gaussians": model.count,
                            "train": float(train.detach()),
                            "validation": validation,
                            "train_bands": train_bands.detach().tolist(),
                            "validation_bands": validation_bands.tolist(),
                            "regularizer": float(penalty.detach()),
                            "elapsed_s": elapsed(),
                            "solver_calls": counts(),
                        }
                    )
                    _LOG.info(
                        "Gaussian stage=%d step=%d kernels=%d train=%.6g validation=%.6g",
                        stage_index,
                        step,
                        model.count,
                        float(train.detach()),
                        validation,
                    )
                if terminal:
                    break
                edge_gradient = None
                if controller is not None and controller.needs_edge(step + 1):
                    edge_gradient = torch.autograd.grad(
                        edges.loss(prediction, active), raw, retain_graph=True
                    )[0].detach()
                    # retain_grad also records the auxiliary adjoint. Clear it so
                    # topology eligibility uses only the optimized objective.
                    raw.grad = None
                loss.backward()
                if refine:
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("Non-finite parameter gradient before refinement")
                    before_calls = counts()
                    event = adaptive_step(
                        model,
                        optimizer,
                        raw.grad + bounds_raw.grad,
                        nominal_scale,
                        config.adaptation,
                        config.learning_rates(),
                        ObjectiveValues(float(loss.detach()), float(train.detach()), validation),
                        evaluate_update,
                    )
                    event.update(stage=stage_index, step=step, cutoff_hz=cutoff)
                    after_calls = counts()
                    event["additional_solver_calls"] = {
                        name: after_calls[name] - before_calls[name] for name in after_calls
                    }
                    adaptation_history.append(event)
                    save_json(adaptation_history, output / "adaptation.json", overwrite=True)
                    _LOG.info(
                        "Gaussian refinement stage=%d step=%d inserted=%d split=%d kernels=%d",
                        stage_index,
                        step,
                        event["retained_insertions"],
                        event["retained_splits"],
                        model.count,
                    )
                else:
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("Non-finite parameter gradient")
                    if controller is not None:
                        controller.observe(model, raw.grad + bounds_raw.grad, edge_gradient)
                    if refinement is not None:
                        refinement.observe(model, raw.grad + bounds_raw.grad, float(loss.detach()))
                    optimizer.step()
                    model.project_()
                    if controller is not None:
                        event = controller.after_update(model, optimizer, step + 1)
                        if event is not None:
                            event.update(
                                stage=stage_index,
                                cutoff_hz=cutoff,
                                additional_adjoint_calls=int(edge_gradient is not None),
                            )
                            density_history.append(event)
                            save_json(density_history, output / "density.json", overwrite=True)
                    if refinement is not None:
                        # Reserve explicitly scheduled future seeds without
                        # making their existence a prerequisite for adaptation.
                        future = sum(
                            math.prod(s) for s in config.levels[stage_index + 1 :] if s is not None
                        )
                        before_calls = counts()
                        event = refinement.after_update(
                            model,
                            optimizer,
                            step + 1,
                            learning_rates=config.learning_rates(),
                            remaining_updates=config.steps_per_stage - step - 1,
                            reserved_capacity=future,
                            evaluate=evaluate_training,
                        )
                        if event is not None:
                            after_calls = counts()
                            event["additional_forward_calls"] = (
                                after_calls["forward"] - before_calls["forward"]
                            )
                            event["additional_adjoint_calls"] = (
                                after_calls["adjoint"] - before_calls["adjoint"]
                            )
                            event.update(stage=stage_index, cutoff_hz=cutoff)
                            topology_history.append(event)
                    if config.spatial_radius_schedule is not None:
                        spatial_history.append(
                            {"stage": stage_index, "step": step + 1, **radius_record(model)}
                        )
            except (FloatingPointError, RuntimeError):
                if best.model_state is not None:
                    best.restore(model, optimizer)
                raise
        best.restore(model, optimizer)
        if refinement is not None:
            if best_topology_state is None:
                raise RuntimeError("No topology state accompanies the selected field")
            # IDs from discarded trajectories remain retired in the event log.
            best_topology_state["next_id"] = max(
                best_topology_state["next_id"], refinement.topology.next_id
            )
            refinement.topology.load_state_dict(best_topology_state)
            topology_state = refinement.topology.state_dict()
            for event in topology_history:
                if event["stage"] == stage_index:
                    event["retained_in_selected_stage_trajectory"] = event["step"] <= best.step
        stages.append(
            {
                "cutoff_hz": cutoff,
                "selected_step": best.step,
                "validation": best.score,
                "gaussians": model.count,
            }
        )
        payload = {
            "format": (
                "gaussian-fwi-training-v10"
                if config.spatial_radius_schedule is not None
                else "gaussian-fwi-training-v9"
                if (config.refinement is not None and config.refinement.extended)
                or model.sampling is not None
                else "gaussian-fwi-training-v8"
                if config.refinement is not None or any(s is None for s in config.levels)
                else "gaussian-fwi-training-v7"
                if config.density_control is not None
                else "gaussian-fwi-training-v5"
                if config.adaptation and config.adaptation.comparison_steps > 1
                else "gaussian-fwi-training-v4"
                if config.adaptation
                else "gaussian-fwi-training-v2"
            ),
            "specification": specification,
            "observation_identity": observation_identity,
            "completed_stage": stage_index,
            "optimizer": optimizer.state_dict(),
            "field": model.checkpoint(),
            "stages": stages,
            "history": history,
            "adaptation_history": adaptation_history,
            "initial_velocity": initial_velocity.detach().cpu(),
            "initial_prediction": initial_prediction.detach().cpu(),
            "elapsed_s": elapsed(),
            "solver_calls": counts(),
        }
        if controller is not None:
            payload["density_history"] = density_history
        if refinement is not None:
            payload["topology_history"] = topology_history
            payload["topology_state"] = topology_state
        if config.spatial_radius_schedule is not None:
            payload["spatial_base_sigma_min_m"] = spatial_base_minimum
            payload["spatial_history"] = spatial_history
        save_torch(payload, output / f"stage_{stage_index:02d}.pt", overwrite=_resume is not None)
        save_json(history, output / "history.json", overwrite=True)

    with torch.no_grad():
        final_velocity = model().clone()
        final_prediction = observations.acquisition.simulate(final_velocity)
        waveform_scores = {
            name: {
                "initial": objective.losses(initial_prediction, config.cutoffs, name).tolist(),
                "final": objective.losses(final_prediction, config.cutoffs, name).tolist(),
                "receiver_indices": ids.tolist(),
            }
            for name, ids in objective.split.items()
        }
    model.save(output / "field.pt", overwrite=_resume is not None)
    save_torch(
        {
            "initial_velocity": initial_velocity.cpu(),
            "final_velocity": final_velocity.cpu(),
            "initial_prediction": initial_prediction.cpu(),
            "final_prediction": final_prediction.cpu(),
        },
        output / "predictions.pt",
        overwrite=_resume is not None,
    )
    report = {
        **specification,
        "observation_identity": observation_identity,
        "method": "gaussian_adam",
        "stages": stages,
        "waveforms": waveform_scores,
        "parameters": sum(p.numel() for p in model.parameters()),
        "gaussians": model.count,
        "elapsed_s": elapsed(),
        "solver_calls": counts(),
        "optimizer_updates": {
            "trajectory": len(config.cutoffs) * config.steps_per_stage,
            "refinement_trials": sum(
                event["optimizer_updates_evaluated"] - 1 for event in adaptation_history
            )
            + sum(
                event.get("trial", {}).get("work", {}).get("optimizer_updates", 0)
                for event in topology_history
            ),
        },
        "truth_used_for_training_or_selection": False,
        "adaptation_history": adaptation_history,
    }
    save_json(history, output / "history.json", overwrite=True)
    if config.adaptation is not None:
        save_json(adaptation_history, output / "adaptation.json", overwrite=True)
    if config.density_control is not None:
        report["density_history"] = density_history
        report["density_additional_adjoints"] = sum(
            event["additional_adjoint_calls"] for event in density_history
        )
        save_json(density_history, output / "density.json", overwrite=True)
    if config.refinement is not None:
        report["topology_history"] = topology_history
        save_json(
            {"state": topology_state, "history": topology_history},
            output / "topology.json",
            overwrite=_resume is not None,
        )
    if model.sampling is not None:
        report["sampling_policy"] = asdict(model.sampling)
    factors = config.sampling_refinement_factors or ((2, 4) if model.sampling else ())
    if factors:
        # Read the delivered checkpoint, so evidence describes the saved field.
        saved_field = type(model).load(output / "field.pt", device=model.background.device)
        report["sampling_diagnostics"] = [
            sampling_diagnostics(saved_field, refinement_factor=factor) for factor in factors
        ]
        save_json({"field": "field.pt", "solver_calls": {"forward": 0, "adjoint": 0},
                   "diagnostics": report["sampling_diagnostics"]},
                  output / "sampling.json", overwrite=_resume is not None)
    if config.spatial_radius_schedule is not None:
        report["spatial_continuation"] = {
            "original_minimum_radius_m": spatial_base_minimum,
            "history": spatial_history,
        }
    save_json(report, output / "report.json", overwrite=_resume is not None)
    return report


def resume(
    checkpoint: str | Path,
    observations: Observations,
    output: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GaussianField, dict]:
    """Resume a completed stage with its verified data identity and Adam state.

    Unbound legacy checkpoints require their frozen source; their saved fields
    remain loadable independently of restart compatibility.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") == "gaussian-fwi-training-v3":
        raise ValueError(
            "Adaptive v3 checkpoints use the v0.3.0 update rule; resume them with v0.3.0. "
            "The contained field remains loadable with GaussianField.from_checkpoint."
        )
    if payload.get("format") == "gaussian-fwi-training-v6":
        raise ValueError(
            "Pre-correction density v6 checkpoints use the one-sigma LAS displacement. "
            "Resume with their frozen source; current density checkpoints use version seven. "
            "The contained field remains loadable with GaussianField.from_checkpoint."
        )
    if payload.get("format") not in (
        "gaussian-fwi-training-v2",
        "gaussian-fwi-training-v4",
        "gaussian-fwi-training-v5",
        "gaussian-fwi-training-v7",
        "gaussian-fwi-training-v8",
        "gaussian-fwi-training-v9",
        "gaussian-fwi-training-v10",
    ):
        raise ValueError(
            "Expected a supported Gaussian training checkpoint (versions 2, 4, 5, 7, 8, 9, or 10)"
        )
    specification = payload["specification"]
    config = InversionConfig(**specification["config"])
    spatial = config.spatial_radius_schedule is not None
    if spatial != (payload["format"] == "gaussian-fwi-training-v10"):
        raise ValueError("Spatial continuation does not match the checkpoint format")
    extended = (config.refinement is not None and config.refinement.extended) or payload["field"][
        "config"
    ].get("sampling") is not None
    if not spatial and extended != (payload["format"] == "gaussian-fwi-training-v9"):
        raise ValueError("Extended refinement or sampling does not match the checkpoint format")
    modern = config.refinement is not None or any(s is None for s in config.levels)
    if not spatial and not extended and modern != (payload["format"] == "gaussian-fwi-training-v8"):
        raise ValueError("Dynamic capacity does not match the checkpoint format")
    if (
        not spatial
        and not extended
        and not modern
        and (config.density_control is not None)
        != (payload["format"] == "gaussian-fwi-training-v7")
    ):
        raise ValueError("Direct density control does not match the checkpoint format")
    maturity = config.adaptation is not None and config.adaptation.comparison_steps > 1
    if (
        not spatial
        and not extended
        and not modern
        and maturity != (payload["format"] == "gaussian-fwi-training-v5")
    ):
        raise ValueError("Refinement comparison horizon does not match the checkpoint format")
    model = GaussianField.from_checkpoint(payload["field"], device=device)
    partitions = {
        name: torch.tensor(ids, dtype=torch.int64, device=device)
        for name, ids in specification["partitions"].items()
    }
    report = invert(
        model,
        observations,
        config,
        output,
        partitions=partitions,
        regularization=Regularization(**specification["regularization"]),
        preprocessing=Preprocessing(**specification["preprocessing"]),
        _resume=payload,
    )
    return model, report
