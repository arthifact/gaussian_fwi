"""Fit an explicit Gaussian FWI profile to a portable observation-only bundle."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

import gaussian_fwi
from gaussian_fwi.core import Preprocessing, Regularization, WaveformObjective
from gaussian_fwi.core.checkpoint import prepare_output, save_json
from gaussian_fwi.core.footprints import FootprintAcquisition
from gaussian_fwi.core.io import load_observations


def configuration(profile, grid):
    """Validate method and field settings against the actual acquisition grid."""
    if not isinstance(profile, dict) or set(profile) != {
        "field", "inversion", "regularization", "preprocessing",
    }:
        raise ValueError("Profile requires field, inversion, regularization, preprocessing")
    config = gaussian_fwi.InversionConfig(**profile["inversion"])
    settings = profile["field"]
    if (not {"background", "bounds"} <= set(settings)
            or not set(settings) <= {"background", "bounds", "sampling", "backend"}):
        raise ValueError("Invalid field settings")
    gaussian_fwi.GaussianField(grid, **settings)
    return config, Regularization(**profile["regularization"]), Preprocessing(
        **profile["preprocessing"],
    )


def fit_observations(observations, partitions, profile, output, **execution):
    """Fit with training waveforms and validation selection; no velocity target."""
    config, regularization, preprocessing = configuration(profile, observations.acquisition.grid)
    field = gaussian_fwi.GaussianField(
        observations.acquisition.grid, **profile["field"],
    ).to(observations.traces)
    report = gaussian_fwi.invert(
        field, observations, config, output, partitions=partitions,
        regularization=regularization, preprocessing=preprocessing,
        **execution,
    )
    return field, report


def verify_fit(field, observations, report, fit_output, *, metrics=None):
    """Reload and repropagate on the fitting device, recording numerical agreement."""
    fit_output = Path(fit_output)
    saved = torch.load(fit_output / "predictions.pt", map_location="cpu", weights_only=True)
    device = observations.traces.device
    restored = gaussian_fwi.GaussianField.load(fit_output / "field.pt", device=device)
    checks = {}

    def compare(name, actual, expected):
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError("Replay tensor shape or precision changed")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("Replay contains nonfinite values")
        exact = torch.equal(actual, expected)
        difference = float(torch.linalg.vector_norm(actual.double()-expected.double()))
        reference = float(torch.linalg.vector_norm(expected.double()))
        relative, absolute = ((0., 0.) if device.type == "cpu" else
                              (1e-10, 1e-10) if actual.dtype == torch.float64 else (5e-5, 1e-6))
        if difference > relative*reference+absolute:
            raise AssertionError(f"{name} replay error exceeds its declared tolerance")
        checks[name] = {"exact": exact, "absolute_l2": difference,
                        "relative_l2": difference/max(reference, 1e-30),
                        "relative_tolerance": relative, "absolute_tolerance": absolute}

    with torch.no_grad():
        velocity = restored()
        compare("live_field", velocity, field())
        compare("saved_field", velocity, saved["final_velocity"])
        prediction = observations.acquisition.simulate(velocity)
        compare("prediction", prediction, saved["final_prediction"])
        for block in restored.blocks:
            torch.linalg.cholesky(block.covariance())
    if restored.count != report["gaussians"]:
        raise AssertionError("Saved population disagrees with the fit report")
    if metrics is not None:
        metrics.update(checks)
    return saved


def run_case(observation_path, profile, output, *, device="cpu", dtype=None,
             resume_checkpoint=None, checkpoint_interval=None, max_updates=None,
             max_seconds=None, diagnostics=False, shot_batch_size=None, accumulate_shots=None,
             wavefield_storage=None):
    """Run the explicit baseline on the requested device, preserving provenance."""
    device = torch.device(device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if dtype != torch.float64:
            raise ValueError("The CUDA runner currently requires explicit float64 precision")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(.70, device=device)
    observation_path = Path(observation_path)
    if resume_checkpoint is not None and shot_batch_size is None:
        parent_run = json.loads((Path(resume_checkpoint).parent.parent / "run.json").read_text())
        shot_batch_size = parent_run.get("execution", {}).get("shot_batch_size", 1)
    file_identity = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    observations, partitions = load_observations(observation_path, device=device, dtype=dtype,
                                                 shot_batch_size=shot_batch_size)
    config, _, preprocessing = configuration(profile, observations.acquisition.grid)
    WaveformObjective(observations, config.cutoffs, partitions, preprocessing)
    prior_counts = {"forward": 0, "adjoint": 0}
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint).resolve()
        original_profile = json.loads((resume_checkpoint.parent.parent / "config.json").read_text())
        if original_profile != json.loads(json.dumps(profile)):
            raise ValueError("Resume requires the original complete runner profile")
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        saved_accumulation = payload["specification"].get("accumulate_shots", False)
        if accumulate_shots is None:
            accumulate_shots = saved_accumulation
        elif accumulate_shots != saved_accumulation:
            raise ValueError("Resume cannot change shot accumulation")
        saved_storage = payload["specification"].get("wavefield_storage", "device")
        if wavefield_storage is None:
            wavefield_storage = saved_storage
        elif wavefield_storage != saved_storage:
            raise ValueError("Resume cannot change wavefield storage")
        if payload["specification"]["config"] != asdict(config):
            raise ValueError("Resume checkpoint and profile disagree")
        prior_counts = payload["solver_calls"]
    if accumulate_shots is None:
        accumulate_shots = False
    if wavefield_storage is None:
        wavefield_storage = "device"
    if wavefield_storage not in ("device", "cpu", "disk"):
        raise ValueError("wavefield_storage must be device, cpu or disk (uncompressed)")
    if type(accumulate_shots) is not bool:
        raise ValueError("accumulate_shots must be boolean")
    if accumulate_shots and not isinstance(observations.acquisition, FootprintAcquisition):
        raise ValueError("Shot accumulation requires a finite-footprint acquisition")
    if hashlib.sha256(observation_path.read_bytes()).hexdigest() != file_identity:
        raise RuntimeError("Observation bundle changed while loading")
    output = prepare_output(output)
    source_root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve()]
    for package in (gaussian_fwi,):
        sources.extend(Path(package.__file__).resolve().parent.glob("*.py"))
    import gaussian_fwi.core as fwi_core

    sources.extend(Path(fwi_core.__file__).resolve().parent.glob("*.py"))
    identities = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(sources)}
    save_json(profile, output / "config.json")
    save_json({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "device": str(observations.traces.device), "dtype": str(observations.traces.dtype),
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_allocator_fraction": .70 if device.type == "cuda" else None,
        "amp": False, "tf32": False if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_checkpoint_sha256": hashlib.sha256(resume_checkpoint.read_bytes()).hexdigest()
        if resume_checkpoint else None,
        "execution": {"checkpoint_interval": checkpoint_interval, "max_updates": max_updates,
                      "max_seconds": max_seconds, "diagnostics": diagnostics,
                      "shot_batch_size": getattr(observations.acquisition, "shot_batch_size", 1),
                      "accumulate_shots": accumulate_shots, "wavefield_storage": wavefield_storage},
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "deepwave", "numpy", "scipy")},
        "source_sha256": identities, "observation_file_sha256": file_identity,
        "observation_content_identity": observations.content_identity(),
        "scope": "Observation-only baseline; acquisition supplied by the caller",
    }, output / "run.json")
    phase = "fit"
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        execution = {"checkpoint_interval": checkpoint_interval, "max_updates": max_updates,
                     "max_seconds": max_seconds, "diagnostics": diagnostics,
                     "accumulate_shots": accumulate_shots, "wavefield_storage": wavefield_storage}
        if resume_checkpoint is None:
            field, report = fit_observations(observations, partitions, profile, output / "fit", **execution)
        else:
            field, report = gaussian_fwi.resume(resume_checkpoint, observations, output / "fit",
                                                device=device, **execution)
        segment_counts = dict(observations.acquisition.counts)
        segment_batches = dict(getattr(observations.acquisition, "batch_counts", segment_counts))
        fit_counts = {key: value + prior_counts[key] for key, value in segment_counts.items()}
        if fit_counts != report["solver_calls"]:
            raise AssertionError("Reported work disagrees with measured acoustic calls")
        phase = "verification"
        replay = {}
        paused = report.get("status") == "paused"
        if not paused:
            verify_fit(field, observations, report, output / "fit", metrics=replay)
        verification_counts = {
            key: observations.acquisition.counts[key] - segment_counts[key] for key in segment_counts
        }
        shots = observations.acquisition.source_amplitudes.shape[0]
        shot_multiplier = 1 if isinstance(observations.acquisition, FootprintAcquisition) else shots
        if hashlib.sha256(observation_path.read_bytes()).hexdigest() != file_identity:
            raise RuntimeError("Observation bundle changed during the fit")
        if any(hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest
               for name, digest in identities.items()):
            raise RuntimeError("Runtime source changed during the fit")
        result = {
            "status": "paused" if paused else "complete", "method": report["method"],
            "gaussians": report["gaussians"], "parameters": report["parameters"],
            "fit_solver_calls": fit_counts, "verification_solver_calls": verification_counts,
            "segment_solver_calls": segment_counts,
            "segment_acoustic_batch_calls": segment_batches,
            "fit_shot_solves": {key: value * shot_multiplier for key, value in fit_counts.items()},
            "verification_shot_solves": {
                key: value * shot_multiplier for key, value in verification_counts.items()
            },
            "segment_shot_solves": {key: value * shot_multiplier for key, value in segment_counts.items()},
            "solver_call_unit": (
                "individual physical shot solves" if shot_multiplier == 1 else f"batches of {shots} shots"
            ),
            "independent_prediction_exact": replay["prediction"]["exact"] if replay else None,
            "replay_verification": replay,
            "device": str(observations.traces.device), "dtype": str(observations.traces.dtype),
            "reference_velocity_loaded": False,
        }
        if paused:
            result["checkpoint"] = str(Path("fit") / report["checkpoint"])
            result["completed_updates"] = report["optimizer_updates"]["trajectory"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["cuda_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "driver_free_bytes": torch.cuda.mem_get_info(device)[0],
            }
        save_json(result, output / "result.json")
    except BaseException as error:
        save_json({"status": "failed", "phase": phase, "error": type(error).__name__,
                   "message": str(error), "notes": getattr(error, "__notes__", []),
                   "observed_solver_calls": observations.acquisition.counts},
                  output / "failure.json")
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="Observation-only .pt bundle")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--config", type=Path, help="Explicit baseline JSON profile")
    start.add_argument("--resume", type=Path, help="Runner checkpoint; reuse its unchanged profile")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--steps-per-stage", type=int, help="Override and record the update allowance")
    parser.add_argument("--threads", type=int, default=2, help="Positive number of CPU threads")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Explicit compute device; CUDA requires --dtype float64")
    parser.add_argument("--dtype", choices=("float32", "float64"),
                        help="Explicit arithmetic precision; default preserves the bundle dtype")
    parser.add_argument("--checkpoint-interval", type=int, help="Save immutable completed-update snapshots")
    parser.add_argument("--max-updates", type=int, help="Pause after this many updates in this process")
    parser.add_argument("--max-seconds", type=float, help="Pause at an update boundary after this interval")
    parser.add_argument("--diagnostics", action="store_true", help="Record gradient and update sizes")
    parser.add_argument("--shot-batch-size", type=int, help="Group compatible finite-footprint shots")
    parser.add_argument("--accumulate-shots", action="store_true", default=None,
                        help="Free each finite-footprint batch's wavefields before the next")
    parser.add_argument("--wavefield-storage", choices=("device", "cpu", "disk"),
                        help="Explicit uncompressed intermediate wavefield storage (default: device)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.resume is not None and args.steps_per_stage is not None:
        parser.error("A resumed trajectory cannot change its optimization horizon")
    profile_path = args.config if args.config is not None else args.resume.parent.parent / "config.json"
    profile = json.loads(profile_path.read_text())
    if args.steps_per_stage is not None:
        profile["inversion"]["steps_per_stage"] = args.steps_per_stage
    result = run_case(args.observations, profile, args.output, device=args.device,
                      dtype=getattr(torch, args.dtype) if args.dtype else None,
                      resume_checkpoint=args.resume, checkpoint_interval=args.checkpoint_interval,
                      max_updates=args.max_updates, max_seconds=args.max_seconds, diagnostics=args.diagnostics,
                      shot_batch_size=args.shot_batch_size, accumulate_shots=args.accumulate_shots,
                      wavefield_storage=args.wavefield_storage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
