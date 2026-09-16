"""Explicit CUDA float64 state/restart acceptance; requires the source test fixtures."""
import argparse
import hashlib
import json
import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

import gaussian_fwi as fwi
import run
from gaussian_fwi._topology import GaussianTopology
from gaussian_fwi.core import Observations
from gaussian_fwi.core.checkpoint import save_json
from gaussian_fwi.core.io import load_observations

ROOT = Path(__file__).resolve().parents[1]


def near(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=1e-8, atol=1e-10)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            near(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            near(a, b)
    elif isinstance(first, float):
        assert abs(first-second) <= 1e-8*abs(second)+1e-10
    else:
        assert first == second


def gpu_prepared(dimension=2):
    field = fwi.GaussianField(fwi.GridSpec((21,)*dimension, 10),
                              background=(2400, 2700), backend="sparse_fused").to(
                                  device="cuda", dtype=torch.float64)
    centers = torch.tensor([[50.]*dimension, [100.]*dimension, [150.]*dimension],
                           device="cuda", dtype=torch.float64)
    scales = torch.tensor([[8.]*dimension, [40.]*dimension, [12.]*dimension],
                          device="cuda", dtype=torch.float64)
    block = field.add_gaussians(centers, scales)
    optimizer = torch.optim.Adam(field.parameter_groups(), amsgrad=True)
    for parameter in field.parameters():
        parameter.grad = torch.linspace(.1, .3, parameter.numel(), device="cuda",
                                        dtype=torch.float64).reshape_as(parameter)
    optimizer.step()
    with torch.no_grad():
        block.amplitudes.copy_(torch.tensor([10., -12., .1], device="cuda"))
        block.shears.fill_(.2)
    return field, optimizer, GaussianTopology(field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot-batch-size", type=int, default=1)
    parser.add_argument("--accumulate-shots", action="store_true")
    parser.add_argument("--wavefield-storage", choices=("device", "cpu", "disk"), default="device")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA acceptance requires a working CUDA device")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.70)
    sys.path.insert(0, str(ROOT / "tests/unit"))
    import test_refinement

    # Reuse independent invariant checks with explicitly CUDA-backed fixtures.
    test_refinement.prepared = gpu_prepared
    names = [
        "test_single_event_selects_small_clone_large_split_and_signed_prune",
        "test_batch_preserves_survivor_adam_and_gives_every_child_fresh_state",
        "test_failed_batch_restores_parameter_identities_gradients_moments_and_ids",
        "test_batch_checks_the_cumulative_change_instead_of_individual_limits",
        "test_unexpected_failure_rolls_back_the_whole_event_and_can_be_retried",
        "test_small_sampling_floor_rejects_split_without_mutation",
        "test_zero_gradients_do_not_force_growth_and_settling_forbids_all_edits",
    ]
    with (args.output / "topology.log").open("x", encoding="utf-8") as log:
        checked = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(
            test_refinement.RefinementTests(name) for name in names))
    if not checked.wasSuccessful() or checked.skipped:
        raise AssertionError("CUDA topology checks failed; see topology.log")

    profile = json.loads((ROOT / "configs/baseline.json").read_text())
    cfg = fwi.InversionConfig((10., 20.), seed_shape=(2, 2), steps_per_stage=4,
                              validation_interval=1, refinement=fwi.RefinementConfig(
                                  warmup_steps=0, interval=1, stop_fraction=.75,
                                  minimum_age=1, max_gaussians=64, max_growth=2,
                                  max_prunes=2, split_extent_fraction=1., prune_amplitude=0.))
    profile["inversion"] = asdict(cfg)
    result = run.run_case(args.observations, profile, args.output / "fixture",
                          device="cuda", dtype=torch.float64, shot_batch_size=args.shot_batch_size,
                          accumulate_shots=args.accumulate_shots, wavefield_storage=args.wavefield_storage)
    observations, partitions = load_observations(args.observations, device="cuda", dtype=torch.float64,
                                                  shot_batch_size=args.shot_batch_size)
    original = json.loads((args.output / "fixture/fit/report.json").read_text())
    initial_counts = dict(observations.acquisition.counts)
    resumed, repeated = fwi.resume(args.output / "fixture/fit/stage_00.pt", observations,
                                    args.output / "resumed", device="cuda")
    original_field = fwi.GaussianField.load(args.output / "fixture/fit/field.pt", device="cuda")
    near(original_field.checkpoint(), resumed.checkpoint())
    for key in ("stages", "topology_history", "solver_calls", "optimizer_updates", "waveforms"):
        near(original[key], repeated[key])
    a = torch.load(args.output / "fixture/fit/stage_01.pt", weights_only=True)
    b = torch.load(args.output / "resumed/stage_01.pt", weights_only=True)
    near(a["optimizer"], b["optimizer"])
    for key in ("ids", "last_edit_steps", "next_id"):
        near(a["topology_state"][key], b["topology_state"][key])
    resume_work = {key: observations.acquisition.counts[key]-initial_counts[key]
                   for key in initial_counts}

    segment_work = {"forward": 0, "adjoint": 0}
    checkpoint = None
    for index, allowance in enumerate((3, 2, 3, None)):
        output = args.output / f"segment_{index}"
        part = run.run_case(args.observations, profile, output, device="cuda", dtype=torch.float64,
                            resume_checkpoint=checkpoint, checkpoint_interval=1,
                            max_updates=allowance, diagnostics=True, shot_batch_size=args.shot_batch_size,
                            accumulate_shots=args.accumulate_shots, wavefield_storage=args.wavefield_storage)
        for key in segment_work:
            segment_work[key] += part["segment_shot_solves"][key]
        if part["status"] == "paused":
            checkpoint = output / part["checkpoint"]
        else:
            assert allowance is None
    segmented_field = fwi.GaussianField.load(output / "fit/field.pt", device="cuda")
    segmented_report = json.loads((output / "fit/report.json").read_text())
    near(original_field.checkpoint(), segmented_field.checkpoint())
    for key in ("stages", "topology_history", "solver_calls", "optimizer_updates", "waveforms"):
        near(original[key], segmented_report[key])
    segmented_state = torch.load(output / "fit/stage_01.pt", weights_only=True)
    near(a["optimizer"], segmented_state["optimizer"])
    assert segment_work == {"forward": 36, "adjoint": 24}

    changed = observations.traces.clone()
    changed[:, partitions["test"]] *= -7
    altered = Observations(observations.acquisition, changed)
    field, isolated = run.fit_observations(altered, partitions, profile, args.output / "test_isolation",
                                          accumulate_shots=args.accumulate_shots,
                                          wavefield_storage=args.wavefield_storage)
    near(original_field.checkpoint(), field.checkpoint())
    for key in ("stages", "topology_history", "solver_calls"):
        near(original[key], isolated[key])
    assert original["waveforms"]["test"] != isolated["waveforms"]["test"]
    expected = {"forward": 36, "adjoint": 24}
    assert original["solver_calls"] == repeated["solver_calls"] == isolated["solver_calls"] == expected
    assert resume_work == {"forward": 18, "adjoint": 12}
    record = {"passed": True, "device": torch.cuda.get_device_name(), "dtype": "float64",
              "scope": "CUDA state, rollback, completed-stage/update restart, test isolation",
              "topology_tests": names, "fixture": result, "resume_additional_shot_solves": resume_work,
              "isolation_shot_solves": isolated["solver_calls"],
              "segmented_shot_solves": segment_work,
              "shot_batch_size": args.shot_batch_size,
              "accumulate_shots": args.accumulate_shots,
              "wavefield_storage": args.wavefield_storage,
              "reference_velocity_loaded": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(record, args.output / "result.json")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
