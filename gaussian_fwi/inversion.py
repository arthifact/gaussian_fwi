"""Gaussian FWI with one periodic adaptive density-control training cycle."""

import logging
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from ._optimizer import validate_adam
from ._structure import GaussianCheckpoint
from ._topology import GaussianTopology
from .core import Observations, Preprocessing, Regularization
from .core.checkpoint import prepare_output, save_json, save_torch
from .core.physics import WaveformObjective
from .field import GaussianField
from .refinement import RefinementConfig, RefinementController
from .sampling import sampling_diagnostics

_LOG = logging.getLogger(__name__)
TRAINING_FORMAT = "gaussian-fwi-adc-training-v1"
METHOD = "gaussian_fwi_adc"


@dataclass(frozen=True)
class InversionConfig:
    """One seed population and one optimize/refine/settle cycle per frequency band.

    Every Gaussian parameter remains trainable throughout. Validation selects
    checkpoints in the settling phase only. Frequency transitions add no fixed
    lattices. A new training horizon requires a new fit.
    """

    cutoffs: tuple[float, ...]
    seed_shape: tuple[int, ...] | None = None
    steps_per_stage: int = 1000
    validation_interval: int = 25
    amplitude_lr: float = 4.0
    geometry_lr: float = 0.008
    center_lr_ratio: float = 0.02
    sigma_ratio: float = 0.65
    raw_bounds_weight: float = 1e-3
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    sampling_refinement_factors: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        if (not self.cutoffs or any(not math.isfinite(f) or f <= 0 for f in self.cutoffs)
                or any(b <= a for a, b in zip(self.cutoffs, self.cutoffs[1:]))):
            raise ValueError("Frequency cutoffs must be positive, finite and increasing")
        if self.seed_shape is not None:
            object.__setattr__(self, "seed_shape", tuple(self.seed_shape))
            if (len(self.seed_shape) not in (2, 3)
                    or any(type(n) is not int or n < 2 for n in self.seed_shape)):
                raise ValueError("Seed shape must contain two or three integer sizes >= 2")
        if isinstance(self.refinement, dict):
            object.__setattr__(self, "refinement", RefinementConfig(**self.refinement))
        if not isinstance(self.refinement, RefinementConfig):
            raise TypeError("Use the Gaussian FWI RefinementConfig")
        for name in ("steps_per_stage", "validation_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.refinement.validate_stage(self.steps_per_stage)
        for name in ("amplitude_lr", "geometry_lr", "center_lr_ratio", "sigma_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.raw_bounds_weight) or self.raw_bounds_weight < 0:
            raise ValueError("raw_bounds_weight must be finite and nonnegative")
        factors = tuple(self.sampling_refinement_factors)
        if any(type(n) is not int or n < 2 for n in factors) or len(set(factors)) != len(factors):
            raise ValueError("Sampling refinement factors must be distinct integers >= 2")
        object.__setattr__(self, "sampling_refinement_factors", factors)

    def learning_rates(self):
        return {name: getattr(self, name)
                for name in ("amplitude_lr", "geometry_lr", "center_lr_ratio")}


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
    """Optimize training waveforms and select a settling-phase validation checkpoint.

    The supplied field is updated in place. Output must be new or empty. No
    reference velocity or test waveform enters optimization or density control.
    Checkpoints resume completed frequency stages with the identical policy.
    """
    if not isinstance(config, InversionConfig):
        raise TypeError("Use gaussian_fwi.InversionConfig")
    if model.grid != observations.acquisition.grid:
        raise ValueError("Field and acquisition must use the same physical grid")
    if (model.background.dtype != observations.traces.dtype
            or model.background.device != observations.traces.device):
        raise ValueError("Field and observations must use the same dtype and device")
    if model.bounds[1] > observations.acquisition.max_velocity:
        raise ValueError("The solver maximum must cover the field velocity bounds")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("The baseline requires a fully trainable Gaussian field")
    identity = observations.content_identity()
    objective = WaveformObjective(observations, config.cutoffs, partitions, preprocessing)
    specification = {
        "method": METHOD, "config": asdict(config),
        "regularization": asdict(regularization), "preprocessing": asdict(preprocessing),
        "partitions": {name: ids.tolist() for name, ids in objective.split.items()},
        "observation_sha256": observations.sha256,
    }
    seed = None
    if _resume is None:
        if config.seed_shape is None:
            if model.count == 0:
                raise ValueError("Provide a seed shape or an already seeded field")
        else:
            if model.count or len(config.seed_shape) != model.grid.ndim:
                raise ValueError("A seed shape requires an empty field of matching dimension")
            seed = GaussianField.from_checkpoint(model.checkpoint(), device=model.background.device)
            seed.add_grid_level(config.seed_shape, sigma_ratio=config.sigma_ratio)
        count = model.count if seed is None else seed.count
        if not config.refinement.minimum_gaussians <= count <= config.refinement.max_gaussians:
            raise ValueError("Initial population must lie within the configured limits")
    history, stages, events = [], [], []
    first_stage, prior_elapsed = 0, 0.0
    prior_counts = {"forward": 0, "adjoint": 0}
    topology_state = None
    optimizer = None
    if _resume is not None:
        if _resume.get("format") != TRAINING_FORMAT:
            raise ValueError("Training checkpoint uses a different algorithm; use its frozen source")
        if _resume.get("observation_identity") != identity:
            raise ValueError("Checkpoint observation content does not match")
        if _resume.get("specification") != specification:
            raise ValueError("Checkpoint observations or inversion specification do not match")
        completed = _resume.get("completed_stage")
        if type(completed) is not int or not 0 <= completed < len(config.cutoffs):
            raise ValueError("Invalid completed frequency stage")
        first_stage = completed + 1
        history, stages, events = deepcopy((_resume["history"], _resume["stages"],
                                           _resume["topology_history"]))
        if len(stages) != first_stage:
            raise ValueError("Checkpoint stage history is incomplete")
        topology_state = _resume["topology_state"]
        topology = GaussianTopology(model, topology_state)
        if (model.count > config.refinement.max_gaussians
                or any(age > first_stage * config.steps_per_stage
                       for ages in topology.last_edit_steps for age in ages)):
            raise ValueError("Checkpoint population or topology ages are invalid")
        optimizer = torch.optim.Adam(model.parameter_groups(**config.learning_rates()))
        optimizer.load_state_dict(deepcopy(_resume["optimizer"]))
        validate_adam(model, optimizer)
        prior_elapsed, prior_counts = _resume["elapsed_s"], _resume["solver_calls"]
        if (not math.isfinite(prior_elapsed) or prior_elapsed < 0
                or set(prior_counts) != {"forward", "adjoint"}
                or any(type(n) is not int or n < 0 for n in prior_counts.values())):
            raise ValueError("Invalid recorded computational work")
        for name, shape in (("initial_velocity", model.grid.shape),
                            ("initial_prediction", observations.traces.shape)):
            value = _resume[name]
            if (not isinstance(value, Tensor) or value.shape != shape
                    or not torch.isfinite(value).all() or value.dtype != model.background.dtype):
                raise ValueError("Invalid initial field or waveform in checkpoint")
    output = prepare_output(output)
    if seed is not None:
        model.blocks = seed.blocks
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameter_groups(**config.learning_rates()))
    validate_adam(model, optimizer)
    starting_counts = observations.acquisition.counts
    start = time.perf_counter()

    def elapsed():
        return prior_elapsed + time.perf_counter() - start

    def counts():
        current = observations.acquisition.counts
        return {name: prior_counts[name] + current[name] - starting_counts[name] for name in current}

    with torch.no_grad():
        if _resume is None:
            initial_velocity = model().clone()
            initial_prediction = observations.acquisition.simulate(initial_velocity)
        else:
            initial_velocity = _resume["initial_velocity"].to(model.background)
            initial_prediction = _resume["initial_prediction"].to(model.background)
    for stage_index in range(first_stage, len(config.cutoffs)):
        active = config.cutoffs[:stage_index + 1]
        controller = RefinementController(model, config.refinement,
                                         steps_per_stage=config.steps_per_stage, stage=stage_index,
                                         topology_state=topology_state)
        best, best_topology = GaussianCheckpoint(), None
        stop = config.refinement.stop_step(config.steps_per_stage)
        for step in range(config.steps_per_stage + 1):
            terminal = step == config.steps_per_stage
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.set_grad_enabled(not terminal):
                    raw = model.raw()
                    velocity = model.bound_velocity(raw).reshape(model.grid.shape)
                    prediction = observations.acquisition.simulate(velocity)
                    train_bands = objective.losses(prediction, active, "train")
                    train = train_bands.mean()
                    # Preserve the established two-decoder accumulation order.
                    bounds_raw = model.raw()
                    bounds_loss = ((bounds_raw - bounds_raw.clamp(*model.bounds)) / 1000).square().mean()
                    penalty = regularization(velocity, model.grid.spacing)
                    penalty = penalty + config.raw_bounds_weight * bounds_loss
                    loss = train + penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training objective")
                inspect = step % config.validation_interval == 0 or terminal
                validation = None
                eligible = inspect and step >= stop
                if inspect:
                    with torch.no_grad():
                        validation = float(objective.losses(prediction.detach(), active,
                                                            "validation").mean())
                    if not math.isfinite(validation):
                        raise FloatingPointError("Non-finite validation objective")
                    if eligible and best.consider(validation, step, model, optimizer):
                        best_topology = controller.topology.state_dict()
                    _LOG.info("Gaussian FWI stage=%d step=%d kernels=%d train=%.6g validation=%.6g",
                              stage_index, step, model.count, float(train.detach()), validation)
                history.append({"stage": stage_index, "step": step,
                                "cutoff_hz": config.cutoffs[stage_index],
                                "phase": "settling" if step >= stop else "refining",
                                "selection_eligible": eligible, "gaussians": model.count,
                                "train": float(train.detach()), "objective": float(loss.detach()),
                                "regularizer": float(penalty.detach()), "validation": validation,
                                "train_bands": train_bands.detach().tolist(),
                                "solver_calls": counts(), "elapsed_s": elapsed()})
                if terminal:
                    break
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError("Non-finite parameter gradient")
                controller.observe(model)
                optimizer.step()
                model.project_()
                event = controller.after_update(model, optimizer, step + 1)
                if event is not None:
                    event["cutoff_hz"] = config.cutoffs[stage_index]
                    event["solver_calls"] = counts()
                    events.append(event)
            except (FloatingPointError, RuntimeError):
                if best.model_state is not None:
                    best.restore(model, optimizer)
                raise
        best.restore(model, optimizer)
        if best_topology is None:
            raise RuntimeError("No finite settling-phase checkpoint was selected")
        controller.topology.load_state_dict(best_topology)
        topology_state = controller.topology.state_dict()
        stages.append({"cutoff_hz": config.cutoffs[stage_index], "selected_step": best.step,
                       "validation": best.score, "gaussians": model.count,
                       "refinement_stop_step": stop})
        payload = {
            "format": TRAINING_FORMAT, "specification": specification,
            "observation_identity": identity, "completed_stage": stage_index,
            "optimizer": optimizer.state_dict(), "field": model.checkpoint(),
            "stages": stages, "history": history, "topology_history": events,
            "topology_state": topology_state, "initial_velocity": initial_velocity.detach().cpu(),
            "initial_prediction": initial_prediction.detach().cpu(),
            "solver_calls": counts(), "elapsed_s": elapsed(),
        }
        save_torch(payload, output / f"stage_{stage_index:02d}.pt")
        save_json(history, output / "history.json", overwrite=True)
        save_json(events, output / "refinement.json", overwrite=True)
    with torch.no_grad():
        final_velocity = model().clone()
        final_prediction = observations.acquisition.simulate(final_velocity)
        waveform_scores = {
            name: {"initial": objective.losses(initial_prediction, config.cutoffs, name).tolist(),
                   "final": objective.losses(final_prediction, config.cutoffs, name).tolist(),
                   "receiver_indices": ids.tolist()} for name, ids in objective.split.items()
        }
    model.save(output / "field.pt")
    save_torch({"initial_velocity": initial_velocity.cpu(), "final_velocity": final_velocity.cpu(),
                "initial_prediction": initial_prediction.cpu(),
                "final_prediction": final_prediction.cpu()}, output / "predictions.pt")
    report = {
        **specification, "observation_identity": identity, "stages": stages,
        "waveforms": waveform_scores, "parameters": sum(p.numel() for p in model.parameters()),
        "gaussians": model.count, "elapsed_s": elapsed(), "solver_calls": counts(),
        "optimizer_updates": {"trajectory": len(config.cutoffs) * config.steps_per_stage,
                              "refinement_trials": 0},
        "truth_used_for_training_or_selection": False, "topology_history": events,
        "selection_window": "settling",
    }
    save_json(history, output / "history.json", overwrite=True)
    save_json(events, output / "refinement.json", overwrite=True)
    save_json({"state": topology_state, "history": events}, output / "topology.json")
    if model.sampling is not None:
        report["sampling_policy"] = asdict(model.sampling)
    factors = config.sampling_refinement_factors or ((2, 4) if model.sampling else ())
    if factors:
        saved = GaussianField.load(output / "field.pt", device=model.background.device)
        report["sampling_diagnostics"] = [sampling_diagnostics(saved, refinement_factor=f)
                                           for f in factors]
        save_json({"field": "field.pt", "solver_calls": {"forward": 0, "adjoint": 0},
                   "diagnostics": report["sampling_diagnostics"]}, output / "sampling.json")
    save_json(report, output / "report.json")
    return report


def resume(checkpoint, observations, output, *, device="cpu"):
    """Resume the same algorithm after a completed stage into a fresh directory.

    Older algorithm checkpoints require their archived source. Their field
    exports remain readable through GaussianField.from_checkpoint/load.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != TRAINING_FORMAT:
        raise ValueError("Training checkpoint uses a different algorithm; use its frozen source")
    if payload.get("observation_identity") != observations.content_identity():
        raise ValueError("Checkpoint observation content does not match")
    spec = payload["specification"]
    config = InversionConfig(**spec["config"])
    model = GaussianField.from_checkpoint(payload["field"], device=device)
    partitions = {name: torch.tensor(ids, dtype=torch.int64, device=device)
                  for name, ids in spec["partitions"].items()}
    report = invert(model, observations, config, output, partitions=partitions,
                    regularization=Regularization(**spec["regularization"]),
                    preprocessing=Preprocessing(**spec["preprocessing"]), _resume=payload)
    return model, report
