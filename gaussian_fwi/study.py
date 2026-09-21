"""Research-ready runs: one verified profile, recorded provenance, one result object.

This is the high-level entry point. :func:`run` performs the same work as the
``gaussian-fwi`` command line: it loads an observation-only bundle, fits the
declared profile, records configuration, source and data identities, and
independently repropagates the saved field before reporting success. :class:`Run`
reads a finished output directory back into Python.

The numerical method is unchanged. This module assembles and records it.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .core import Preprocessing, Regularization
from .core.checkpoint import prepare_output, save_json
from .core.io import load_observations
from .core.physics import WaveformObjective
from .field import GaussianField
from .inversion import InversionConfig, invert, resume

SECTIONS = ("field", "inversion", "regularization", "preprocessing")

# The accepted production profile. ``configs/baseline.json`` is the canonical
# copy in a source checkout; ``tests/unit/test_study.py`` asserts they agree, so
# an installed wheel needs no data files to reproduce the baseline.
BASELINE = {
    "field": {
        "bounds": [1500.0, 4500.0],
        "background": [1500.0, 3000.0],
        "sampling": {"max_nyquist_response": 0.001},
        "backend": "sparse_fused",
    },
    "regularization": {
        "tv_weight": 0.0001,
        "tikhonov_weight": 0.0,
        "epsilon": 0.1,
        "velocity_scale": 100.0,
        "length_scale": 10.0,
    },
    "preprocessing": {"time_gain_power": 1.5, "trace_balance_cap": 5.0},
    "inversion": {
        "cutoffs": [4.0, 7.0, 12.0, 20.0],
        "seed_shape": [32, 32],
        "steps_per_stage": 1000,
        "validation_interval": 25,
        "amplitude_lr": 4.0,
        "geometry_lr": 0.008,
        "center_lr_ratio": 0.02,
        "sigma_ratio": 0.65,
        "raw_bounds_weight": 0.001,
        "refinement": {
            "warmup_steps": 50,
            "interval": 50,
            "stop_fraction": 0.5,
            "gradient_threshold": 0.0,
            "split_extent_fraction": 0.01,
            "max_gaussians": 8192,
            "max_growth": 128,
            "max_prunes": 128,
            "minimum_gaussians": 1,
            "minimum_age": 50,
            "prune_amplitude": 0.5,
            "max_field_change": 25.0,
            "seed": 0,
        },
        "sampling_refinement_factors": [2, 4],
    },
}


def baseline(*, field=None, inversion=None, regularization=None, preprocessing=None,
             refinement=None, **overrides):
    """Return the accepted profile, with any declared deviation merged in.

    Every keyword is an explicit, recorded departure from the accepted baseline.
    Bare keywords name ``inversion`` settings, so a shorter development fit is
    ``baseline(steps_per_stage=200)``. Section dictionaries are merged key by
    key. Settings are validated here, not part way through a long fit.
    """
    profile = deepcopy(BASELINE)
    for name, section in (("field", field), ("inversion", inversion),
                          ("regularization", regularization), ("preprocessing", preprocessing)):
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"{name} must be a dictionary of settings")
        unknown = set(section) - set(profile[name])
        if unknown:
            raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
        profile[name].update(deepcopy(section))
    if refinement is not None:
        if not isinstance(refinement, dict):
            raise TypeError("refinement must be a dictionary of settings")
        unknown = set(refinement) - set(profile["inversion"]["refinement"])
        if unknown:
            raise ValueError(f"Unknown refinement settings: {', '.join(sorted(unknown))}")
        profile["inversion"]["refinement"].update(deepcopy(refinement))
    unknown = set(overrides) - set(profile["inversion"])
    if unknown:
        raise ValueError(
            f"Unknown inversion settings: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(profile['inversion']))}. "
            "Use field=, regularization= or preprocessing= for other sections.",
        )
    profile["inversion"].update(deepcopy(overrides))
    # Fail now on anything checkable without an acquisition grid.
    InversionConfig(**profile["inversion"])
    Regularization(**profile["regularization"])
    Preprocessing(**profile["preprocessing"])
    return profile


def configuration(profile, grid):
    """Validate method and field settings against the actual acquisition grid."""
    if not isinstance(profile, dict) or set(profile) != set(SECTIONS):
        raise ValueError("Profile requires field, inversion, regularization, preprocessing")
    config = InversionConfig(**profile["inversion"])
    settings = profile["field"]
    if (not {"background", "bounds"} <= set(settings)
            or not set(settings) <= {"background", "bounds", "sampling", "backend"}):
        raise ValueError("Invalid field settings")
    GaussianField(grid, **settings)
    return config, Regularization(**profile["regularization"]), Preprocessing(
        **profile["preprocessing"],
    )


def fit_observations(observations, partitions, profile, output, **execution):
    """Fit with training waveforms and validation selection; no velocity target."""
    config, regularization, preprocessing = configuration(profile, observations.acquisition.grid)
    field = GaussianField(
        observations.acquisition.grid, **profile["field"],
    ).to(observations.traces)
    report = invert(
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
    restored = GaussianField.load(fit_output / "field.pt", device=device)
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


def source_identities():
    """Hash the executing method source, keyed relative to the install root."""
    package = Path(__file__).resolve().parent
    root = package.parent
    sources = list(package.glob("*.py")) + list((package / "core").glob("*.py"))
    runner = root / "run.py"
    if runner.is_file():
        sources.append(runner)
    return root, {str(path.resolve().relative_to(root)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in sorted(set(sources))}


def execute(observation_path, profile, output, *, device="cpu", dtype=None,
            resume_checkpoint=None, checkpoint_interval=None, max_updates=None,
            max_seconds=None, diagnostics=False):
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
    file_identity = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    observations, partitions = load_observations(observation_path, device=device, dtype=dtype)
    config, _, preprocessing = configuration(profile, observations.acquisition.grid)
    WaveformObjective(observations, config.cutoffs, partitions, preprocessing)
    prior_counts = {"forward": 0, "adjoint": 0}
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint).resolve()
        original_profile = json.loads((resume_checkpoint.parent.parent / "config.json").read_text())
        if original_profile != json.loads(json.dumps(profile)):
            raise ValueError("Resume requires the original complete runner profile")
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        if payload["specification"]["config"] != asdict(config):
            raise ValueError("Resume checkpoint and profile disagree")
        prior_counts = payload["solver_calls"]
    if hashlib.sha256(observation_path.read_bytes()).hexdigest() != file_identity:
        raise RuntimeError("Observation bundle changed while loading")
    output = prepare_output(output)
    source_root, identities = source_identities()
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
                      "max_seconds": max_seconds, "diagnostics": diagnostics},
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
                     "max_seconds": max_seconds, "diagnostics": diagnostics}
        if resume_checkpoint is None:
            field, report = fit_observations(observations, partitions, profile, output / "fit",
                                             **execution)
        else:
            field, report = resume(resume_checkpoint, observations, output / "fit",
                                   device=device, **execution)
        segment_counts = dict(observations.acquisition.counts)
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
            "fit_shot_solves": {key: value * shots for key, value in fit_counts.items()},
            "verification_shot_solves": {
                key: value * shots for key, value in verification_counts.items()
            },
            "segment_shot_solves": {
                key: value * shots for key, value in segment_counts.items()
            },
            "solver_call_unit": f"batches of {shots} shots",
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


def run(observations, output, *, profile=None, device="cpu", dtype=None, threads=None,
        checkpoint_interval=None, max_updates=None, max_seconds=None, diagnostics=False):
    """Fit an observation bundle with a recorded profile and return the finished run.

        import gaussian_fwi as gfwi

        fit = gfwi.run("data/observations.pt", output="results/fit01")
        print(fit.summary())
        velocity = fit.velocity            # (z, x) or (z, y, x) in m/s

    ``observations`` is a path to an observation-only bundle; target velocities
    are never accepted. ``profile`` defaults to the accepted baseline; pass
    :func:`baseline` with keywords to declare a deviation. ``output`` must be a
    new directory. The run writes its configuration, source and data identities,
    loss history, density edits and measured acoustic work, then repropagates
    the saved field independently before reporting success.
    """
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    profile = deepcopy(BASELINE) if profile is None else profile
    execute(observations, profile, output, device=device, dtype=dtype,
            checkpoint_interval=checkpoint_interval, max_updates=max_updates,
            max_seconds=max_seconds, diagnostics=diagnostics)
    return Run(output)


def continue_run(checkpoint, observations, output, **execution):
    """Continue a paused or stage checkpoint into a new directory and return the run.

    The recorded profile, optimization horizon and execution grouping are reused;
    changing them requires a new experiment.
    """
    checkpoint = Path(checkpoint)
    profile = json.loads((checkpoint.parent.parent / "config.json").read_text())
    execute(observations, profile, output, resume_checkpoint=checkpoint, **execution)
    return Run(output)


class Run:
    """A finished run directory, read back as Python objects.

    Files stay on disk and load on first use, so opening a run is cheap.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        if not (self.directory / "result.json").is_file():
            raise FileNotFoundError(
                f"{self.directory} has no result.json. A failed run records failure.json; "
                "an unfinished one has neither.",
            )
        self._cache = {}

    @classmethod
    def open(cls, directory):
        """Open a completed run directory written by :func:`run` or the CLI."""
        return cls(directory)

    def _json(self, name):
        if name not in self._cache:
            self._cache[name] = json.loads((self.directory / name).read_text())
        return self._cache[name]

    @property
    def result(self):
        """Verification summary: status, population, measured work, replay checks."""
        return self._json("result.json")

    @property
    def config(self):
        """The complete profile this run executed."""
        return self._json("config.json")

    @property
    def provenance(self):
        """Platform, dependency versions, source hashes and observation identity."""
        return self._json("run.json")

    @property
    def report(self):
        """The inversion report: stages, selection, losses and population."""
        return self._json("fit/report.json")

    @property
    def history(self):
        """Per-update loss history."""
        return self._json("fit/history.json")

    @property
    def refinement(self):
        """Accepted and rejected density-control events."""
        return self._json("fit/refinement.json")

    @property
    def status(self):
        """``"complete"`` or ``"paused"``."""
        return self.result["status"]

    @property
    def gaussians(self):
        """Number of Gaussian components in the selected field."""
        return self.result["gaussians"]

    @property
    def parameters(self):
        """Number of trainable scalars in the selected field."""
        return self.result["parameters"]

    @property
    def verified(self):
        """True when an independent propagation reproduced the saved prediction exactly.

        None for a paused run, which exports no selected field.
        """
        return self.result["independent_prediction_exact"]

    @property
    def solver_calls(self):
        """Measured forward and adjoint acoustic calls used by the fit."""
        return self.result["fit_solver_calls"]

    @property
    def waveform_losses(self):
        """Normalized initial and final waveform losses per partition and band.

        These are waveform mismatches, not velocity errors.
        """
        return self.report["waveforms"]

    @property
    def field(self):
        """The selected :class:`~gaussian_fwi.field.GaussianField`."""
        if "field" not in self._cache:
            path = self.directory / "fit/field.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"This run is {self.status} and exported no selected field. "
                    "Continue it with gaussian_fwi.continue_run before reading a velocity.",
                )
            self._cache["field"] = GaussianField.load(path)
        return self._cache["field"]

    @property
    def velocity(self):
        """The recovered velocity on the acquisition grid, ``(z, x)`` or ``(z, y, x)``, in m/s."""
        if "velocity" not in self._cache:
            with torch.no_grad():
                self._cache["velocity"] = self.field().detach()
        return self._cache["velocity"]

    def summary(self):
        """A short, printable description of what this run established."""
        stages = self.report.get("stages", [])
        bands = " / ".join(f"{stage['cutoff_hz']:g}" for stage in stages)
        lines = [
            f"Gaussian FWI run: {self.directory}",
            f"  status            {self.status}"
            + ("" if self.verified is None else
               f"; independent replay {'exact' if self.verified else 'within tolerance'}"),
            f"  population        {self.gaussians} Gaussians, {self.parameters} trainable scalars",
            f"  frequency bands   {bands} Hz" if bands else "  frequency bands   none recorded",
            f"  acoustic work     {self.solver_calls['forward']} forward,"
            f" {self.solver_calls['adjoint']} adjoint",
            f"  device            {self.result['device']} {self.result['dtype']}",
        ]
        train = self.waveform_losses.get("train", {})
        if train.get("initial") and train.get("final"):
            first, last = train["initial"], train["final"]
            lines.append(
                f"  training waveform {sum(first)/len(first):.6g} -> {sum(last)/len(last):.6g}"
                " (band mean, normalized)",
            )
        return "\n".join(lines)

    def plot(self, destination=None, *, title=None):
        """Draw the recovered velocity on its physical grid.

        Requires the optional plotting dependency: ``pip install gaussian-fwi[plot]``.
        Returns the Matplotlib figure; saves it when ``destination`` is given.
        """
        try:
            import matplotlib
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotting needs Matplotlib: pip install gaussian-fwi[plot]",
            ) from error
        if destination is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        velocity = self.velocity
        if velocity.ndim != 2:
            raise ValueError("plot draws 2D sections; slice a 3D field before plotting")
        spacing = self.field.grid.spacing
        depth, width = velocity.shape
        figure, axes = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
        image = axes.imshow(velocity.cpu().numpy(), cmap="viridis", aspect="equal",
                            extent=(0, (width-1)*spacing, (depth-1)*spacing, 0))
        axes.set(xlabel="Horizontal position (m)", ylabel="Depth (m)")
        axes.set_title(title or f"Recovered velocity, {self.gaussians} Gaussians", loc="left")
        figure.colorbar(image, ax=axes, label="Velocity (m/s)")
        if destination is not None:
            figure.savefig(destination, dpi=200)
        return figure

    def __repr__(self):
        return (f"Run({str(self.directory)!r}, status={self.status!r}, "
                f"gaussians={self.gaussians})")
