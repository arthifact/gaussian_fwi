"""Compare tiny CPU/CUDA acoustic derivatives; this is not a research fit.

Run after installing the project: python tools/check_cuda.py --output results/cuda/NAME
Requires a working CUDA device and Deepwave CUDA kernels. Failures are recorded
and return a nonzero exit status. No production configuration is modified.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

import gaussian_fwi
from gaussian_fwi import GaussianField
from gaussian_fwi.core import Acquisition, GridSpec, Observations, Preprocessing, Regularization
from gaussian_fwi.core.checkpoint import save_json
from gaussian_fwi.core.physics import WaveformObjective

TOLERANCES = {
    "float64_relative_l2": 1e-8,
    "float32_relative_l2": 5e-3,
    "absolute_l2": 1e-10,
    "adam_relative_l2": 2e-2,
    "adam_absolute_l2": 1e-7,
    "finite_difference_relative": 2e-5,
    "finite_difference_absolute": 2e-11,
}


def fixture(dimension, dtype):
    """Known-source software fixture; nonzero parameters exercise derivatives."""
    grid = GridSpec((12,) * dimension, 10.0)
    field = GaussianField(grid, background=(2100, 2600), backend="sparse_fused").to(dtype)
    t = torch.arange(100, dtype=dtype) * .001 - .04
    q = (torch.pi * 25 * t).square()
    source = ((1 - 2 * q) * torch.exp(-q))[None, None].repeat(2, 1, 1)
    receivers = [[2] + ([5] if dimension == 3 else []) + [i] for i in range(1, 11)]
    sources = [[2] + [5] * (dimension - 1), [2] + ([5] if dimension == 3 else []) + [7]]
    acquisition = Acquisition(
        grid, .001, source, torch.tensor(sources)[:, None],
        torch.tensor([receivers]).repeat(2, 1, 1),
        pml_width=6, pml_frequency=25, max_velocity=4500,
    )
    with torch.no_grad():
        reference = field() + 45
        reference[6:] += 100
        traces = acquisition.simulate(reference)
    partitions = {
        "train": torch.tensor([1, 2, 3, 6, 7, 8]),
        "validation": torch.tensor([4, 9]), "test": torch.tensor([0, 5]),
    }
    block = field.add_grid_level((2,) * dimension, sigma_ratio=.4)
    assert torch.count_nonzero(block.amplitudes) == 0
    # These perturbations test local derivatives, not a truth-supervised fit.
    with torch.no_grad():
        block.amplitudes.copy_(torch.linspace(-25, 35, field.count, dtype=dtype))
        block.shears.fill_(.2)
    return field.checkpoint(), Observations(acquisition, traces), partitions


def compare(actual, reference, relative, absolute):
    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    if actual.shape != reference.shape or not torch.isfinite(actual).all():
        raise AssertionError("Non-finite result or different tensor shapes")
    error = float(torch.linalg.vector_norm(actual - reference))
    norm = float(torch.linalg.vector_norm(reference))
    limit = relative * norm + absolute
    if error > limit:
        raise AssertionError(f"L2 difference {error:.9g} exceeds {limit:.9g}")
    return {"absolute_l2": error, "reference_l2": norm,
            "relative_l2": error / max(norm, absolute), "limit": limit}


def measure(checkpoint, observations, partitions, device, finite_difference=False):
    field = GaussianField.from_checkpoint(checkpoint, device=device)
    acquisition = replace(
        observations.acquisition,
        source_amplitudes=observations.acquisition.source_amplitudes.to(device),
        source_locations=observations.acquisition.source_locations.to(device),
        receiver_locations=observations.acquisition.receiver_locations.to(device),
    )
    data = Observations(acquisition, observations.traces.to(device))
    assert data.content_identity() == observations.content_identity()
    objective = WaveformObjective(data, (10., 20.), partitions, Preprocessing(1.5, 5))
    regularization = Regularization(tv_weight=1e-4)
    parameters = dict(field.named_parameters())
    optimizer = torch.optim.Adam(field.parameter_groups())
    originals = {name: p.detach().clone() for name, p in parameters.items()}
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def evaluate():
        raw = field.raw()
        velocity = field.bound_velocity(raw).reshape(field.grid.shape)
        prediction = acquisition.simulate(velocity)
        bounds_raw = field.raw()
        penalty = ((bounds_raw - bounds_raw.clamp(*field.bounds)) / 1000).square().mean()
        loss = (objective.losses(prediction, (10., 20.), "train").mean()
                + regularization(velocity, field.grid.spacing) + 1e-3 * penalty)
        return velocity, prediction, loss

    velocity, prediction, loss = evaluate()
    loss.backward()
    gradients = {name: p.grad.detach().cpu().clone() for name, p in parameters.items()}
    values = {"velocity": velocity.detach().cpu(), "prediction": prediction.detach().cpu(),
              "loss": loss.detach().cpu(), **{f"gradient/{k}": v for k, v in gradients.items()}}
    optimizer.step()
    field.project_()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    updates = {name: (p.detach() - originals[name]).cpu() for name, p in parameters.items()}
    for block in field.blocks:
        torch.linalg.cholesky(block.covariance())
    memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "driver_free_bytes": torch.cuda.mem_get_info()[0]} if device == "cuda" else {})
    differences = {}
    if finite_difference:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(originals[name])
            for name, parameter in parameters.items():
                direction = torch.cos(torch.arange(parameter.numel(), dtype=parameter.dtype,
                                                   device=device)).reshape_as(parameter)
                direction /= torch.linalg.vector_norm(direction)
                adjoint = float((gradients[name].to(device) * direction).sum())
                if abs(adjoint) <= 1e-12:
                    raise AssertionError(f"Uninformative directional derivative: {name}")
                estimates = []
                try:
                    for h in (1e-3, 1e-4):
                        parameter.copy_(originals[name] + h * direction)
                        positive = float(evaluate()[-1])
                        parameter.copy_(originals[name] - h * direction)
                        negative = float(evaluate()[-1])
                        estimate = (positive - negative) / (2 * h)
                        compare(torch.tensor(estimate, dtype=torch.float64),
                                torch.tensor(adjoint, dtype=torch.float64),
                                TOLERANCES["finite_difference_relative"],
                                TOLERANCES["finite_difference_absolute"])
                        estimates.append({"step": h, "derivative": estimate})
                finally:
                    parameter.copy_(originals[name])
                differences[name] = {"adjoint": adjoint, "central_differences": estimates}
    return values, updates, {
        "device": device, "synchronized_update_seconds": elapsed, **memory,
        "solver_calls": acquisition.counts,
        "shot_solves": {k: v * 2 for k, v in acquisition.counts.items()},
        "finite_differences": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), *Path(gaussian_fwi.__file__).parent.rglob("*.py")]
    root = Path(__file__).resolve().parents[1]
    record = {
        "scope": "Tiny software fixture; no GPU campaign or inversion/restart certification",
        "tolerances": TOLERANCES, "amp": False, "tf32": False,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "deepwave", "numpy", "scipy")},
        "torch_cuda_runtime": torch.version.cuda,
        "source_sha256": {str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "cases": [],
    }
    save_json(record, args.output / "manifest.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        record["gpu"] = torch.cuda.get_device_name()
        for dimension in (2, 3):
            for dtype in (torch.float64, torch.float32):
                checkpoint, data, partitions = fixture(dimension, dtype)
                case = {"dimension": dimension, "dtype": str(dtype),
                        "observation_identity": data.content_identity(),
                        "observation_generation_solver_calls": data.acquisition.counts,
                        "observation_generation_shot_solves": {"forward": 2, "adjoint": 0}}
                record["cases"].append(case)
                print(f"Checking {dimension}D {dtype} CPU/CUDA derivatives", flush=True)
                cpu, cpu_updates, case["cpu"] = measure(checkpoint, data, partitions, "cpu")
                cuda, cuda_updates, case["cuda"] = measure(
                    checkpoint, data, partitions, "cuda", finite_difference=dtype == torch.float64,
                )
                relative = TOLERANCES[f"{str(dtype).split('.')[-1]}_relative_l2"]
                case["comparisons"] = {name: compare(cuda[name], reference, relative,
                                                     TOLERANCES["absolute_l2"])
                                       for name, reference in cpu.items()}
                case["adam_updates"] = {
                    name: compare(cuda_updates[name], reference, TOLERANCES["adam_relative_l2"],
                                  TOLERANCES["adam_absolute_l2"])
                    for name, reference in cpu_updates.items()
                }
                case["passed"] = True
        record["passed"] = True
    except BaseException as error:
        record.update(passed=False, error=type(error).__name__, message=str(error))
        save_json(record, args.output / "result.json")
        raise
    save_json(record, args.output / "result.json")
    print(json.dumps({"passed": True, "cases": len(record["cases"]),
                      "evidence": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
