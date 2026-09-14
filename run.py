"""Fit an explicit Gaussian FWI profile to a portable observation-only bundle."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

import dynamic_refinement
import gaussian_fwi
from fwi_core import Preprocessing, Regularization, WaveformObjective
from fwi_core.checkpoint import prepare_output, save_json
from fwi_core.footprints import FootprintAcquisition
from fwi_core.io import load_observations


def configuration(profile, grid):
    """Validate method and field settings against the actual acquisition grid."""
    if not isinstance(profile, dict) or set(profile) != {
        "method", "field", "inversion", "regularization", "preprocessing",
    }:
        raise ValueError("Profile requires method, field, inversion, regularization, preprocessing")
    if profile["method"] == "direct":
        config = dynamic_refinement.InversionConfig(**profile["inversion"])
    elif profile["method"] == "scheduled":
        config = gaussian_fwi.InversionConfig(**profile["inversion"])
        if (any(x is not None for x in (config.refinement, config.adaptation, config.density_control))
                or config.spatial_radius_schedule is not None
                or any(level is None for level in config.levels)):
            raise ValueError("Scheduled mode requires explicit lattices and no topology controller")
    else:
        raise ValueError("Method must be direct or scheduled")
    settings = profile["field"]
    if (not {"background", "bounds"} <= set(settings)
            or not set(settings) <= {"background", "bounds", "sampling", "backend"}):
        raise ValueError("Invalid field settings")
    gaussian_fwi.GaussianField(grid, **settings)
    return config, Regularization(**profile["regularization"]), Preprocessing(
        **profile["preprocessing"],
    )


def fit_observations(observations, partitions, profile, output):
    """Fit with training waveforms and validation selection; no velocity target."""
    config, regularization, preprocessing = configuration(profile, observations.acquisition.grid)
    field = gaussian_fwi.GaussianField(
        observations.acquisition.grid, **profile["field"],
    ).to(observations.traces)
    engine = dynamic_refinement if profile["method"] == "direct" else gaussian_fwi
    report = engine.invert(
        field, observations, config, output, partitions=partitions,
        regularization=regularization, preprocessing=preprocessing,
    )
    return field, report


def verify_fit(field, observations, report, fit_output):
    """Independently reload and repropagate a CPU fit; count the additional work."""
    fit_output = Path(fit_output)
    saved = torch.load(fit_output / "predictions.pt", map_location="cpu", weights_only=True)
    restored = gaussian_fwi.GaussianField.load(fit_output / "field.pt")
    with torch.no_grad():
        velocity = restored()
        torch.testing.assert_close(velocity, field(), rtol=0, atol=0)
        torch.testing.assert_close(velocity, saved["final_velocity"], rtol=0, atol=0)
        prediction = observations.acquisition.simulate(velocity)
        torch.testing.assert_close(prediction, saved["final_prediction"], rtol=0, atol=0)
        for block in restored.blocks:
            torch.linalg.cholesky(block.covariance())
    if restored.count != report["gaussians"]:
        raise AssertionError("Saved population disagrees with the fit report")
    return saved


def run_case(observation_path, profile, output):
    """Run the explicit CPU baseline, preserving identities and failure/work records."""
    observation_path = Path(observation_path)
    file_identity = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    observations, partitions = load_observations(observation_path)
    config, _, preprocessing = configuration(profile, observations.acquisition.grid)
    WaveformObjective(observations, config.cutoffs, partitions, preprocessing)
    if hashlib.sha256(observation_path.read_bytes()).hexdigest() != file_identity:
        raise RuntimeError("Observation bundle changed while loading")
    output = prepare_output(output)
    source_root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve()]
    for package in (dynamic_refinement, gaussian_fwi):
        sources.extend(Path(package.__file__).resolve().parent.glob("*.py"))
    import fwi_core

    sources.extend(Path(fwi_core.__file__).resolve().parent.glob("*.py"))
    identities = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(sources)}
    save_json(profile, output / "config.json")
    save_json({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "device": "cpu", "dtype": str(observations.traces.dtype),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "deepwave", "numpy", "scipy")},
        "source_sha256": identities, "observation_file_sha256": file_identity,
        "observation_content_identity": observations.content_identity(),
        "scope": "Observation-only baseline; acquisition supplied by the caller",
    }, output / "run.json")
    phase = "fit"
    try:
        field, report = fit_observations(observations, partitions, profile, output / "fit")
        fit_counts = dict(observations.acquisition.counts)
        if fit_counts != report["solver_calls"]:
            raise AssertionError("Reported work disagrees with measured acoustic calls")
        phase = "verification"
        verify_fit(field, observations, report, output / "fit")
        verification_counts = {
            key: observations.acquisition.counts[key] - fit_counts[key] for key in fit_counts
        }
        shots = observations.acquisition.source_amplitudes.shape[0]
        shot_multiplier = 1 if isinstance(observations.acquisition, FootprintAcquisition) else shots
        if hashlib.sha256(observation_path.read_bytes()).hexdigest() != file_identity:
            raise RuntimeError("Observation bundle changed during the fit")
        if any(hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest
               for name, digest in identities.items()):
            raise RuntimeError("Runtime source changed during the fit")
        result = {
            "status": "complete", "method": profile["method"],
            "gaussians": report["gaussians"], "parameters": report["parameters"],
            "fit_solver_calls": fit_counts, "verification_solver_calls": verification_counts,
            "fit_shot_solves": {key: value * shot_multiplier for key, value in fit_counts.items()},
            "verification_shot_solves": {
                key: value * shot_multiplier for key, value in verification_counts.items()
            },
            "solver_call_unit": (
                "individual shot calls" if shot_multiplier == 1 else f"batches of {shots} shots"
            ),
            "independent_prediction_exact": True,
            "reference_velocity_loaded": False,
        }
        save_json(result, output / "result.json")
    except BaseException as error:
        save_json({"status": "failed", "phase": phase, "error": type(error).__name__,
                   "message": str(error), "observed_solver_calls": observations.acquisition.counts},
                  output / "failure.json")
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="Observation-only .pt bundle")
    parser.add_argument("--config", type=Path, required=True, help="Explicit baseline JSON profile")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--steps-per-stage", type=int, help="Override and record the update allowance")
    parser.add_argument("--threads", type=int, default=2, help="Positive number of CPU threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    profile = json.loads(args.config.read_text())
    if args.steps_per_stage is not None:
        profile["inversion"]["steps_per_stage"] = args.steps_per_stage
    result = run_case(args.observations, profile, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
