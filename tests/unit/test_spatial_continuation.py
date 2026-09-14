"""Independent radius, continuity, cancellation and acoustic restart oracles."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from test_dynamic import assert_exact, configuration, problem

import dynamic_refinement as fwi
from fwi_core import GridSpec, Observations
from fwi_core.checkpoint import save_torch
from gaussian_fwi.spatial import radius_record, set_radius_floor


class SpatialContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_invalid_settings_and_initial_radii_have_no_side_effects(self):
        for values in ((10,), (10, 20), (0, 0), (10, float("nan")), (True, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(configuration(), spatial_radius_schedule=values)
        for dimension in (2, 3):
            field, data, partitions = problem(dimension)
            field.add_gaussians(
                torch.full((1, dimension), 50.0, dtype=torch.float64),
                torch.full((1, dimension), 10.0, dtype=torch.float64),
            )
            cfg = replace(
                configuration(dimension), seed_shape=None, spatial_radius_schedule=(50, 25)
            )
            before, counts = field.checkpoint(), data.acquisition.counts
            with tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "invalid"
                with self.assertRaisesRegex(ValueError, "radii violate"):
                    fwi.invert(field, data, cfg, destination, partitions=partitions)
                self.assertFalse(destination.exists())
            assert_exact(before, field.checkpoint())
            self.assertEqual(counts, data.acquisition.counts)

    def test_radius_release_preserves_field_parameters_gradients_and_adam(self):
        for dimension in (2, 3):
            field, _, _ = problem(dimension)
            field.add_grid_level((2,) * dimension)
            optimizer = torch.optim.Adam(field.parameter_groups())
            field().square().mean().backward()
            optimizer.step()
            field.project_()
            field().mean().backward()
            set_radius_floor(field, 50)
            before = field().detach().clone()
            parameters = list(field.parameters())
            values = [p.detach().clone() for p in parameters]
            gradients = [p.grad.clone() if p.grad is not None else None for p in parameters]
            moments = copy.deepcopy(optimizer.state_dict())
            set_radius_floor(field, 25)
            self.assertTrue(all(a is b for a, b in zip(parameters, field.parameters())))
            assert_exact(before, field().detach())
            assert_exact(moments, optimizer.state_dict())
            for p, value, gradient in zip(parameters, values, gradients):
                assert_exact(value, p.detach())
                assert_exact(gradient, p.grad)

    def test_full_covariance_floor_matches_independent_eigenvalue_oracle(self):
        for dimension in (2, 3):
            field, _, _ = problem(dimension)
            block = field.add_gaussians(
                torch.full((1, dimension), 50.0, dtype=torch.float64),
                torch.tensor([[12.0, 44.0] + ([55.0] if dimension == 3 else [])]),
            )
            with torch.no_grad():
                block.shears.fill_(0.9)
                block.amplitudes.fill_(40)
            field.sigma_min = 30.0
            field.project_()
            covariance = block.covariance().detach().numpy()[0]
            eigenvalues = np.linalg.eigvalsh(covariance)
            self.assertGreaterEqual(eigenvalues.min(), 30.0**2 * (1 - 1e-12))
            self.assertLessEqual(eigenvalues.max(), field.sigma_max**2 * (1 + 1e-12))
            self.assertGreater(abs(covariance[0, 1]), 1)
            radius_record(field)

    def test_signed_broad_atoms_can_increase_relative_high_wavenumber_energy(self):
        # Independent Fourier identity for equal-area Gaussians: the DC terms
        # cancel although both physical widths remain at least 60 m.
        angular = np.linspace(0, 0.15, 5000)
        first = np.exp(-0.5 * (60 * angular) ** 2)
        difference = first - np.exp(-0.5 * (90 * angular) ** 2)
        self.assertEqual(difference[0], 0)
        high = angular > 2 / 60
        def fraction(a):
            return float(np.sum(a[high] ** 2) / np.sum(a**2))

        self.assertGreater(fraction(difference), fraction(first))
        self.assertGreater(fraction(difference), 0)

        field = fwi.GaussianField(GridSpec((33, 65), 10), background=(2500, 2500)).double()
        block = field.add_gaussians(
            torch.tensor([[320.0, 160.0], [320.0, 160.0]], dtype=torch.float64),
            torch.tensor([[60.0, 80.0], [90.0, 80.0]], dtype=torch.float64),
        )
        x = torch.linspace(-580, 1220, 4096, dtype=torch.float64)
        points = torch.stack((x, torch.full_like(x, 160)), -1)
        with torch.no_grad():
            block.amplitudes.copy_(torch.tensor([100.0, 0.0]))
            single = (field(points) - 2500).numpy()
            block.amplitudes[1] = -100 * 60 / 90
            combined = (field(points) - 2500).numpy()
        def spectrum(signal):
            return np.abs(np.fft.rfft(signal)) ** 2

        k = 2 * np.pi * np.fft.rfftfreq(len(x), float(x[1] - x[0]))

        def tail(signal):
            return float(spectrum(signal)[k > 2 / 60].sum() / spectrum(signal).sum())

        self.assertGreater(tail(combined), tail(single))
        self.assertLess(abs(combined.sum()), 1e-5 * np.abs(single).sum())
        set_radius_floor(field, 60)

    def test_acoustic_counts_and_stage_restart_are_exact(self):
        for dimension in (2, 3):
            field, data, partitions = problem(dimension)
            cfg = replace(
                configuration(dimension), steps_per_stage=3, spatial_radius_schedule=(50, 25)
            )
            initial = field().detach().clone()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = fwi.invert(field, data, cfg, root / "fit", partitions=partitions)
                self.assertEqual(report["solver_calls"], {"forward": 10, "adjoint": 6})
                predictions = torch.load(root / "fit/predictions.pt", weights_only=True)
                assert_exact(initial, predictions["initial_velocity"])
                self.assertEqual(len(report["spatial_continuation"]["history"]), 8)
                for row in report["spatial_continuation"]["history"]:
                    self.assertGreaterEqual(
                        row["realized_minimum_radius_m"], row["minimum_radius_m"] * (1 - 1e-12)
                    )
                stage = root / "fit/stage_00.pt"
                payload = torch.load(stage, weights_only=True)
                self.assertEqual(payload["format"], "gaussian-fwi-training-v10")
                restored, repeated = fwi.resume(stage, data, root / "resumed")
                assert_exact(restored.checkpoint(), field.checkpoint())
                assert_exact(report["spatial_continuation"], repeated["spatial_continuation"])
                for name in ("field", "optimizer", "topology_state", "topology_history"):
                    a = torch.load(root / "fit/stage_01.pt", weights_only=True)
                    b = torch.load(root / "resumed/stage_01.pt", weights_only=True)
                    assert_exact(a[name], b[name])

    def test_no_forced_growth_and_test_traces_cannot_change_the_fit(self):
        field, data, partitions = problem()
        cfg = configuration()
        cfg = replace(
            cfg,
            steps_per_stage=3,
            spatial_radius_schedule=(50, 25),
            refinement=replace(cfg.refinement, operations=()),
        )
        other = fwi.GaussianField.from_checkpoint(field.checkpoint())
        traces = data.traces.clone()
        traces[:, partitions["test"]] += 100
        changed = Observations(data.acquisition, traces, {}, "changed-test-traces")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = fwi.invert(field, data, cfg, root / "base", partitions=partitions)
            fwi.invert(other, changed, cfg, root / "changed", partitions=partitions)
            self.assertEqual(field.count, 4)
            assert_exact(field.checkpoint(), other.checkpoint())
            self.assertTrue(
                all(row["gaussians"] == 4 for row in report["spatial_continuation"]["history"])
            )

    def test_corrupt_spatial_restart_rejects_before_solving_or_writing(self):
        field, data, partitions = problem()
        cfg = replace(configuration(), steps_per_stage=1, spatial_radius_schedule=(50, 25))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fwi.invert(field, data, cfg, root / "fit", partitions=partitions)
            saved = torch.load(root / "fit/stage_00.pt", weights_only=True)
            for index in range(4):
                payload = copy.deepcopy(saved)
                if index == 0:
                    payload["format"] = "gaussian-fwi-training-v8"
                elif index == 1:
                    payload["field"]["config"]["sigma_min"] = 49.0
                elif index == 2:
                    del payload["spatial_base_sigma_min_m"]
                else:
                    payload["field"]["state"]["blocks.0.log_scales"].fill_(1.0)
                checkpoint = root / f"invalid_{index}.pt"
                save_torch(payload, checkpoint)
                counts = data.acquisition.counts
                destination = root / f"rejected_{index}"
                with self.assertRaises(ValueError):
                    fwi.resume(checkpoint, data, destination)
                self.assertEqual(counts, data.acquisition.counts)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
