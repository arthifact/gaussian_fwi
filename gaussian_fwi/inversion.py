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
from .core.footprints import FootprintAcquisition
from .core.physics import WaveformObjective
from .core.wavefields import wavefield_directory
from .field import GaussianField
from .refinement import RefinementConfig, RefinementController
from .sampling import sampling_diagnostics

_LOG = logging.getLogger(__name__)
TRAINING_FORMAT = "gaussian-fwi-adc-training-v1"
UPDATE_FORMAT = "gaussian-fwi-adc-update-v1"
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
    checkpoint_interval: int | None = None,
    max_updates: int | None = None,
    max_seconds: float | None = None,
    diagnostics: bool = False,
    accumulate_shots: bool = False,
    wavefield_storage: str = "device",
    _resume: dict | None = None,
) -> dict:
    """Optimize training waveforms and select a settling-phase validation checkpoint.

    The supplied field is updated in place. Output must be new or empty. No
    reference velocity or test waveform enters optimization or density control.
    Optional execution limits pause after a complete update without changing
    the optimization horizon. Resume always writes into a fresh directory.
    """
    if not isinstance(config, InversionConfig):
        raise TypeError("Use gaussian_fwi.InversionConfig")
    for name, value in (("checkpoint_interval", checkpoint_interval), ("max_updates", max_updates)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    if type(diagnostics) is not bool:
        raise ValueError("diagnostics must be boolean")
    if type(accumulate_shots) is not bool:
        raise ValueError("accumulate_shots must be boolean")
    if wavefield_storage not in ("device", "cpu", "disk"):
        raise ValueError("wavefield_storage must be device, cpu or disk (uncompressed)")
    if accumulate_shots and not isinstance(observations.acquisition, FootprintAcquisition):
        raise ValueError("Shot accumulation requires a finite-footprint acquisition")
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
    shot_batch_size = getattr(observations.acquisition, "shot_batch_size", 1)
    if shot_batch_size != 1:
        specification["shot_batch_size"] = shot_batch_size
    if accumulate_shots:
        specification["accumulate_shots"] = True
    if wavefield_storage != "device":
        specification["wavefield_storage"] = wavefield_storage
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
    first_stage, first_step, prior_elapsed = 0, 0, 0.0
    prior_counts = {"forward": 0, "adjoint": 0}
    topology_state = None
    saved_controller = None
    saved_best = None
    optimizer = None
    if _resume is not None:
        if _resume.get("format") not in (TRAINING_FORMAT, UPDATE_FORMAT):
            raise ValueError("Training checkpoint uses a different algorithm; use its frozen source")
        if _resume.get("observation_identity") != identity:
            raise ValueError("Checkpoint observation content does not match")
        if _resume.get("specification") != specification:
            raise ValueError("Checkpoint observations or inversion specification do not match")
        if _resume["format"] == TRAINING_FORMAT:
            completed = _resume.get("completed_stage")
            if type(completed) is not int or not 0 <= completed < len(config.cutoffs):
                raise ValueError("Invalid completed frequency stage")
            first_stage = completed + 1
        else:
            first_stage, first_step = _resume.get("active_stage"), _resume.get("completed_step")
            if (type(first_stage) is not int or not 0 <= first_stage < len(config.cutoffs)
                    or type(first_step) is not int or not 1 <= first_step <= config.steps_per_stage
                    or _resume.get("device_type") != model.background.device.type):
                raise ValueError("Invalid completed-update cursor or device")
            saved_controller = RefinementController(
                model, config.refinement, steps_per_stage=config.steps_per_stage,
                stage=first_stage, state=_resume.get("controller_state"))
            if (saved_controller.last_step != first_step
                    or saved_controller.topology.state_dict() != _resume["topology_state"]):
                raise ValueError("Checkpoint controller and cursor disagree")
            saved_best = _resume.get("best")
        history, stages, events = deepcopy((_resume["history"], _resume["stages"],
                                           _resume["topology_history"]))
        if len(stages) != first_stage:
            raise ValueError("Checkpoint stage history is incomplete")
        if len(history) != first_stage * (config.steps_per_stage + 1) + first_step:
            raise ValueError("Checkpoint update history is incomplete")
        topology_state = _resume["topology_state"]
        topology = GaussianTopology(model, topology_state)
        if (model.count > config.refinement.max_gaussians
                or any(age > first_stage * config.steps_per_stage + first_step
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
        if _resume["format"] == UPDATE_FORMAT:
            stop = config.refinement.stop_step(config.steps_per_stage)
            inspected = [row for row in history if row["stage"] == first_stage
                         and row["selection_eligible"]]
            if bool(inspected) != (saved_best is not None):
                raise ValueError("Checkpoint omitted its settling selection state")
            if saved_best is not None:
                best_step, score = saved_best["step"], saved_best["score"]
                if (type(best_step) is not int or not stop <= best_step < first_step
                        or not math.isfinite(score)):
                    raise ValueError("Invalid settling selection in update checkpoint")
                selected = min(inspected, key=lambda row: (row["validation"], row["step"]))
                if selected["step"] != best_step or selected["validation"] != score:
                    raise ValueError("Checkpoint selected score disagrees with history")
                selected_field = GaussianField.from_checkpoint(
                    saved_best["field"], device=model.background.device)
                selected_optimizer = torch.optim.Adam(selected_field.parameter_groups(
                    **config.learning_rates()))
                selected_optimizer.load_state_dict(deepcopy(saved_best["optimizer"]))
                validate_adam(selected_field, selected_optimizer)
                GaussianTopology(selected_field, saved_best["topology"])
                restored_best = GaussianCheckpoint()
                # Keep live parameter identities; load selected values/moments separately.
                # A portable best constructed on another field must not redirect Adam's
                # background group to that temporary field when it is restored later.
                if not restored_best.consider(score, best_step, model, optimizer):
                    raise ValueError("Invalid selected update checkpoint")
                restored_best.model_state = deepcopy(selected_field.state_dict())
                restored_best.optimizer_state = deepcopy(selected_optimizer.state_dict())
            rng = _resume.get("rng")
            if (not isinstance(rng, dict) or set(rng) != {"cpu", "cuda"}
                    or not isinstance(rng["cpu"], Tensor) or rng["cpu"].dtype != torch.uint8
                    or rng["cpu"].shape != torch.random.get_rng_state().shape):
                raise ValueError("Invalid saved random state")
            if model.background.is_cuda and (not isinstance(rng["cuda"], Tensor)
                    or rng["cuda"].dtype != torch.uint8
                    or rng["cuda"].shape != torch.cuda.get_rng_state(model.background.device).shape):
                raise ValueError("Invalid saved CUDA random state")
    output = prepare_output(output)
    with wavefield_directory(output, wavefield_storage) as storage_path:
        if seed is not None:
            model.blocks = seed.blocks
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameter_groups(**config.learning_rates()))
        validate_adam(model, optimizer)
        starting_counts = observations.acquisition.counts
        start = time.perf_counter()
        segment_updates = 0
        if _resume is not None and _resume["format"] == UPDATE_FORMAT:
            torch.random.set_rng_state(_resume["rng"]["cpu"])
            if model.background.is_cuda:
                torch.cuda.set_rng_state(_resume["rng"]["cuda"], model.background.device)

        def elapsed():
            return prior_elapsed + time.perf_counter() - start

        def counts():
            current = observations.acquisition.counts
            return {name: prior_counts[name] + current[name] - starting_counts[name] for name in current}

        def save_update(stage, completed_step, controller, best, best_topology):
            selected = None
            if best.model_state is not None:
                selected_field = GaussianField.from_checkpoint(model.checkpoint())
                selected_field.load_state_dict(best.model_state)
                selected = {"field": selected_field.checkpoint(), "optimizer": best.optimizer_state,
                            "score": best.score, "step": best.step, "topology": best_topology}
            path = output / f"update_{stage * config.steps_per_stage + completed_step:08d}.pt"
            save_torch({"format": UPDATE_FORMAT, "specification": specification,
                        "observation_identity": identity, "active_stage": stage,
                        "completed_step": completed_step, "device_type": model.background.device.type,
                        "optimizer": optimizer.state_dict(), "field": model.checkpoint(),
                        "stages": stages, "history": history, "topology_history": events,
                        "topology_state": controller.topology.state_dict(),
                        "controller_state": controller.state_dict(), "best": selected,
                        "initial_velocity": initial_velocity.detach().cpu(),
                        "initial_prediction": initial_prediction.detach().cpu(),
                        "solver_calls": counts(), "elapsed_s": elapsed(),
                        "rng": {"cpu": torch.random.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state(model.background.device)
                                if model.background.is_cuda else None}}, path)
            return path

        with torch.no_grad():
            if _resume is None:
                initial_velocity = model().clone()
                initial_prediction = observations.acquisition.simulate(initial_velocity)
            else:
                initial_velocity = _resume["initial_velocity"].to(model.background)
                initial_prediction = _resume["initial_prediction"].to(model.background)
        if accumulate_shots and model.background.is_cuda:
            # Initialization uses large temporary FFT buffers. Release unused blocks
            # once before the first acoustic workspace, including on resumed runs.
            torch.cuda.empty_cache()
        for stage_index in range(first_stage, len(config.cutoffs)):
            active = config.cutoffs[:stage_index + 1]
            controller = (saved_controller if stage_index == first_stage and saved_controller is not None
                          else RefinementController(model, config.refinement,
                                                    steps_per_stage=config.steps_per_stage,
                                                    stage=stage_index, topology_state=topology_state))
            best, best_topology = GaussianCheckpoint(), None
            if stage_index == first_stage and saved_best is not None:
                best, best_topology = restored_best, saved_best["topology"]
            stop = config.refinement.stop_step(config.steps_per_stage)
            stage_first_step = first_step if stage_index == first_stage else 0
            for step in range(stage_first_step, config.steps_per_stage + 1):
                terminal = step == config.steps_per_stage
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.set_grad_enabled(not terminal):
                        raw = model.raw()
                        velocity = model.bound_velocity(raw).reshape(model.grid.shape)
                        if accumulate_shots and not terminal:
                            prediction, train_bands, waveform_gradient = objective.accumulate_shot_gradients(
                                observations.acquisition, velocity, active, wavefield_storage=wavefield_storage,
                                storage_path=storage_path)
                        else:
                            prediction = observations.acquisition.simulate(velocity, wavefield_storage=wavefield_storage,
                                                                          storage_path=storage_path)
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
                    if accumulate_shots:
                        torch.autograd.backward((velocity, penalty),
                                                (waveform_gradient, torch.ones_like(penalty)))
                    else:
                        loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.parameters()):
                        raise FloatingPointError("Non-finite parameter gradient")
                    controller.observe(model)
                    measured = diagnostics and (inspect or (checkpoint_interval is not None
                                                             and (step + 1) % checkpoint_interval == 0))
                    if measured:
                        before = {id(p): p.detach().clone() for p in model.parameters()}
                        gradient_squares = {}
                        for group in optimizer.param_groups:
                            role = group["role"]
                            gradient_squares[role] = gradient_squares.get(role, 0.) + sum(
                                float(p.grad.detach().double().square().sum())
                                for p in group["params"] if p.grad is not None)
                        history[-1]["gradient_l2_by_role"] = {
                            role: math.sqrt(value) for role, value in gradient_squares.items()}
                    optimizer.step()
                    model.project_()
                    if measured:
                        update_squares = {}
                        for group in optimizer.param_groups:
                            role = group["role"]
                            update_squares[role] = update_squares.get(role, 0.) + sum(
                                float((p.detach().double() - before[id(p)].double()).square().sum())
                                for p in group["params"])
                        history[-1]["parameter_update_l2_by_role"] = {
                            role: math.sqrt(value) for role, value in update_squares.items()}
                    event = controller.after_update(model, optimizer, step + 1)
                    if event is not None:
                        event["cutoff_hz"] = config.cutoffs[stage_index]
                        event["solver_calls"] = counts()
                        events.append(event)
                    if measured:
                        with torch.no_grad():
                            change = model().double() - velocity.detach().double()
                        history[-1]["completed_update_field_change"] = {
                            "rmse_m_s": float(change.square().mean().sqrt()),
                            "maximum_m_s": float(change.abs().max()),
                            "includes_density_event": event is not None}
                    segment_updates += 1
                    pause = ((max_updates is not None and segment_updates >= max_updates)
                             or (max_seconds is not None and time.perf_counter() - start >= max_seconds))
                    if pause or (checkpoint_interval is not None and (step + 1) % checkpoint_interval == 0):
                        checkpoint = save_update(stage_index, step + 1, controller, best, best_topology)
                    if pause:
                        save_json(history, output / "history.json", overwrite=True)
                        save_json(events, output / "refinement.json", overwrite=True)
                        report = {**specification, "status": "paused", "checkpoint": checkpoint.name,
                                  "observation_identity": identity, "active_stage": stage_index,
                                  "completed_step": step + 1, "stages": stages,
                                  "gaussians": model.count, "parameters": sum(p.numel() for p in model.parameters()),
                                  "solver_calls": counts(), "elapsed_s": elapsed(),
                                  "optimizer_updates": {"trajectory": stage_index * config.steps_per_stage + step + 1,
                                                        "refinement_trials": 0},
                                  "truth_used_for_training_or_selection": False,
                                  "topology_history": events, "selection_window": "settling"}
                        save_json(report, output / "report.json")
                        return report
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


def resume(checkpoint, observations, output, *, device="cpu", checkpoint_interval=None,
           max_updates=None, max_seconds=None, diagnostics=False, accumulate_shots=None,
           wavefield_storage=None):
    """Resume the same algorithm after a completed stage or update into a fresh directory.

    Older algorithm checkpoints require their archived source. Their field
    exports remain readable through GaussianField.from_checkpoint/load.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") not in (TRAINING_FORMAT, UPDATE_FORMAT):
        raise ValueError("Training checkpoint uses a different algorithm; use its frozen source")
    if payload.get("observation_identity") != observations.content_identity():
        raise ValueError("Checkpoint observation content does not match")
    spec = payload["specification"]
    if accumulate_shots is None:
        accumulate_shots = spec.get("accumulate_shots", False)
    if wavefield_storage is None:
        wavefield_storage = spec.get("wavefield_storage", "device")
    config = InversionConfig(**spec["config"])
    model = GaussianField.from_checkpoint(payload["field"], device=device)
    partitions = {name: torch.tensor(ids, dtype=torch.int64, device=device)
                  for name, ids in spec["partitions"].items()}
    report = invert(model, observations, config, output, partitions=partitions,
                    regularization=Regularization(**spec["regularization"]),
                    preprocessing=Preprocessing(**spec["preprocessing"]),
                    checkpoint_interval=checkpoint_interval, max_updates=max_updates,
                    max_seconds=max_seconds, diagnostics=diagnostics,
                    accumulate_shots=accumulate_shots, wavefield_storage=wavefield_storage, _resume=payload)
    return model, report
