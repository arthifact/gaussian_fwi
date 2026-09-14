"""Self-contained interface, acoustic equivalence, and restart checks."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

import dynamic_refinement as fwi
from fwi_core import Acquisition, GridSpec, Observations, Preprocessing, Regularization
from fwi_core.checkpoint import save_torch
from gaussian_fwi.inversion import InversionConfig as EngineConfig
from gaussian_fwi.inversion import invert as engine_invert
from gaussian_fwi.refinement import RefinementConfig as EngineRefinementConfig


def assert_exact(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError("Different checkpoint keys")
        for name in first:
            assert_exact(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError("Different checkpoint sequence lengths")
        for a, b in zip(first, second, strict=True):
            assert_exact(a, b)
    elif first != second:
        raise AssertionError(f"Different values: {first!r} != {second!r}")


def problem(dimension=2, dtype=torch.float64):
    grid = GridSpec((12,) * dimension, 10.0)
    field = fwi.GaussianField(grid, background=(2100, 2600)).to(dtype=dtype)
    time = torch.arange(100, dtype=dtype) * 0.001 - 0.04
    squared = (torch.pi * 25 * time).square()
    wavelet = ((1 - 2 * squared) * torch.exp(-squared))[None, None].repeat(2, 1, 1)
    receivers = [[2] + ([5] if dimension == 3 else []) + [i] for i in range(1, 11)]
    sources = [[2] + [5] * (dimension - 1), [2] + ([5] if dimension == 3 else []) + [7]]
    acquisition = Acquisition(
        grid,
        0.001,
        wavelet,
        torch.tensor(sources)[:, None],
        torch.tensor([receivers]).repeat(2, 1, 1),
        pml_width=6,
        pml_frequency=25,
        max_velocity=4500,
    )
    with torch.no_grad():
        velocity = field() + 45
        velocity[6:] += 100
        traces = acquisition.simulate(velocity)
    partitions = {
        "train": torch.tensor([1, 2, 3, 6, 7, 8]),
        "validation": torch.tensor([4, 9]),
        "test": torch.tensor([0, 5]),
    }
    return field, Observations(acquisition, traces, {}, "method-interface-test"), partitions


def configuration(dimension=2):
    return fwi.InversionConfig(
        cutoffs=(10.0, 20.0),
        steps_per_stage=8,
        validation_interval=2,
        seed_shape=(2,) * dimension,
        refinement=fwi.RefinementConfig(
            insertion_screening="legacy",  # The release oracle uses the frozen direct method.
            max_gaussians=64,
            max_edits=2,
            candidate_pool=4,
            minimum_age=1,
            minimum_relative_gain=0,
            max_backtracks=0,
        ),
    )


class DynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_matches_the_frozen_benchmark_without_external_files(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/benchmark_config.json"
        reference = json.loads(fixture.read_text())["config"]
        config = fwi.InversionConfig((4, 7, 12, 20), seed_shape=(10, 10),
                                     refinement=fwi.RefinementConfig.legacy())
        self.assertEqual(
            json.loads(json.dumps(asdict(config.engine_config()))),
            json.loads(json.dumps(asdict(EngineConfig(**reference)))),
        )
        self.assertEqual(config.refinement.insertion_screening, "legacy")
        self.assertEqual(
            asdict(fwi.RefinementConfig.legacy().engine_config()), asdict(EngineRefinementConfig())
        )
        self.assertEqual(fwi.RefinementConfig().insertion_screening, "coverage_aware")

    def test_configuration_validation_and_normalization(self):
        config = fwi.InversionConfig([4, 7], [2, 3], refinement={"insertion_scales": [10, 20]})
        self.assertEqual(config.seed_shape, (2, 3))
        self.assertEqual(config.refinement.insertion_scales, (10, 20))
        for make in (
            lambda: fwi.InversionConfig(()),
            lambda: fwi.RefinementConfig(max_gaussians=0),
            lambda: fwi.InversionConfig((4, 4)),
            lambda: fwi.RefinementConfig(insertion_screening="unknown"),
            lambda: fwi.RefinementConfig(operations=("unknown",)),
            lambda: fwi.RefinementConfig(operations=("insert", "insert")),
            lambda: fwi.RefinementConfig(operations=("relocate",)),
        ):
            with self.assertRaises(ValueError):
                make()
        with self.assertRaises(TypeError):
            fwi.InversionConfig((4,), refinement=EngineRefinementConfig.guarded())
        self.assertEqual(fwi.RefinementConfig(comparison_steps=3).engine_config().comparison_steps, 3)
        with self.assertRaises(ValueError):
            fwi.RefinementConfig(minimum_shot_support=0.5)
        with self.assertRaises(TypeError):
            fwi.InversionConfig((4,), levels=((2, 2),))

    def test_new_selector_preserves_historical_positional_arguments(self):
        # The first fourteen arguments formerly ended at merge_distance.
        args = (64, 2, 4, 1, 0.0, 0.0, 25.0, "moment", 0.45, 0, 2.0, None, 0.5, 0.2)
        for factory in (fwi.RefinementConfig, EngineRefinementConfig):
            config = factory(*args)
            self.assertEqual(config.merge_distance, 0.2)
            self.assertEqual(config.insertion_screening,
                             "coverage_aware" if factory is fwi.RefinementConfig else "legacy")

    def test_other_controller_checkpoints_are_rejected(self):
        field, data, _ = problem()
        variants = (
            EngineConfig((10,), ((2, 2),)),
            EngineConfig((10,), ((2, 2),), refinement=EngineRefinementConfig.guarded()),
            EngineConfig((10, 20), ((2, 2), (3, 3)), refinement=EngineRefinementConfig()),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, config in enumerate(variants):
                path = Path(temporary) / f"variant_{index}.pt"
                save_torch(
                    {"specification": {"config": asdict(config)}, "field": field.checkpoint()}, path
                )
                before = data.acquisition.counts
                destination = Path(temporary) / "wrong"
                with self.assertRaisesRegex(ValueError, "different Gaussian method"):
                    fwi.resume(path, data, destination)
                self.assertEqual(data.acquisition.counts, before)
                self.assertFalse(destination.exists())

    def test_caller_seed_and_float32_run(self):
        field, data, partitions = problem(dtype=torch.float32)
        field.add_grid_level((2, 2))
        config = replace(configuration(), seed_shape=None, steps_per_stage=2)
        with tempfile.TemporaryDirectory() as temporary:
            report = fwi.invert(
                field, data, config, Path(temporary) / "seeded", partitions=partitions
            )
            self.assertEqual(report["config"]["levels"], (None, None))
            self.assertTrue(torch.isfinite(field()).all())

    def test_real_fits_and_restarts_match_the_engine_exactly(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as temporary:
                field, data, partitions = problem(dimension)
                config = configuration(dimension)
                reference = fwi.GaussianField.from_checkpoint(field.checkpoint())
                root = Path(temporary)
                kwargs = dict(
                    partitions=partitions,
                    regularization=Regularization(tv_weight=1e-4),
                    preprocessing=Preprocessing(time_gain_power=1.5, trace_balance_cap=5),
                )
                report = fwi.invert(field, data, config, root / "selected", **kwargs)
                original = engine_invert(
                    reference, data, config.engine_config(), root / "engine", **kwargs
                )
                assert_exact(field.checkpoint(), reference.checkpoint())
                for filename in ("field.pt", "predictions.pt"):
                    assert_exact(
                        torch.load(root / "selected" / filename, weights_only=True),
                        torch.load(root / "engine" / filename, weights_only=True),
                    )
                for stage in range(2):
                    a = torch.load(root / "selected" / f"stage_{stage:02d}.pt", weights_only=True)
                    b = torch.load(root / "engine" / f"stage_{stage:02d}.pt", weights_only=True)
                    for key in (
                        "specification",
                        "field",
                        "optimizer",
                        "stages",
                        "solver_calls",
                        "topology_state",
                        "topology_history",
                        "initial_velocity",
                        "initial_prediction",
                    ):
                        assert_exact(a.get(key), b.get(key))
                self.assertEqual(report["solver_calls"], {"forward": 20, "adjoint": 16})
                for key in ("waveforms", "stages", "parameters", "gaussians", "optimizer_updates"):
                    assert_exact(report[key], original[key])
                restored, repeated = fwi.resume(
                    root / "selected/stage_00.pt", data, root / "resumed"
                )
                assert_exact(restored.checkpoint(), field.checkpoint())
                self.assertEqual(repeated["solver_calls"], report["solver_calls"])
                for filename in ("field.pt", "predictions.pt"):
                    assert_exact(
                        torch.load(root / "selected" / filename, weights_only=True),
                        torch.load(root / "resumed" / filename, weights_only=True),
                    )
                a = torch.load(root / "selected/stage_01.pt", weights_only=True)
                b = torch.load(root / "resumed/stage_01.pt", weights_only=True)
                for key in (
                    "field",
                    "optimizer",
                    "stages",
                    "solver_calls",
                    "topology_state",
                    "topology_history",
                ):
                    assert_exact(a.get(key), b.get(key))

    def test_insertion_screening_survives_restart_and_missing_key_means_legacy(self):
        for policy in ("legacy", "coverage_aware"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temporary:
                field, data, partitions = problem()
                config = configuration()
                config = replace(
                    config, refinement=replace(config.refinement, insertion_screening=policy)
                )
                self.assertFalse(config.refinement.engine_config().extended)
                root = Path(temporary)
                report = fwi.invert(field, data, config, root / "fit", partitions=partitions)
                checkpoint = root / "fit/stage_00.pt"
                payload = torch.load(checkpoint, weights_only=True)
                self.assertEqual(
                    payload["specification"]["config"]["refinement"]["insertion_screening"], policy
                )
                if policy == "legacy":
                    del payload["specification"]["config"]["refinement"]["insertion_screening"]
                    del payload["specification"]["config"]["refinement"]["operations"]
                    checkpoint = root / "historical_stage_00.pt"
                    save_torch(payload, checkpoint)
                restored, repeated = fwi.resume(checkpoint, data, root / "resumed")
                assert_exact(restored.checkpoint(), field.checkpoint())
                self.assertEqual(repeated["solver_calls"], report["solver_calls"])
                a = torch.load(root / "fit/stage_01.pt", weights_only=True)
                b = torch.load(root / "resumed/stage_01.pt", weights_only=True)
                for key in ("field", "optimizer", "topology_state", "topology_history"):
                    assert_exact(a[key], b[key])

    def test_invalid_policy_is_rejected_before_solving_or_writing(self):
        field, data, partitions = problem()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "invalid"
            before = data.acquisition.counts
            with self.assertRaises(TypeError):
                fwi.invert(
                    field, data, EngineConfig((10,), ((2, 2),)), destination, partitions=partitions
                )
            self.assertFalse(destination.exists())
            self.assertEqual(before, data.acquisition.counts)


if __name__ == "__main__":
    unittest.main()
