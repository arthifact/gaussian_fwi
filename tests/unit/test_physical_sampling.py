"""Physical-grid rejection and independent continuous sampling oracles."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_dynamic import assert_exact, configuration, problem

import dynamic_refinement as fwi
from fwi_core import GridSpec
from gaussian_fwi import SamplingConfig, sample_on_grid, sampling_diagnostics


class PhysicalSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mismatched_field_grid_rejects_before_writes_or_solves(self):
        for dimension in (2, 3):
            _, data, partitions = problem(dimension)
            for grid in (GridSpec((12,) * dimension, 11), GridSpec((13,) * dimension, 10)):
                field = fwi.GaussianField(grid).double()
                counts = data.acquisition.counts
                with tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "invalid"
                    with self.assertRaisesRegex(ValueError, "physical grid"):
                        fwi.invert(field, data, configuration(dimension), output,
                                   partitions=partitions)
                    self.assertFalse(output.exists())
                self.assertEqual(data.acquisition.counts, counts)
                self.assertEqual(field.count, 0)

    def test_nested_sampling_matches_analytic_physical_gaussian(self):
        for dimension in (2, 3):
            grid = GridSpec((7,) * dimension, 10)
            field = fwi.GaussianField(grid, background=(2200, 2600)).double()
            block = field.add_gaussians(torch.full((1, dimension), 30.),
                                        torch.full((1, dimension), 20.))
            with torch.no_grad():
                block.amplitudes.fill_(-75)
            fine = GridSpec((13,) * dimension, 5)
            points = fine.points(dtype=torch.float64)
            expected = 2200 + 400 * points[:, -1] / 60
            expected -= 75 * torch.exp(-((points - 30) ** 2).sum(-1) / 800)
            # All raw values lie >500 m/s from the bounds; clipping <3e-10 m/s.
            actual = sample_on_grid(field, fine)
            torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=3e-10)
            torch.testing.assert_close(actual[(slice(None, None, 2),) * dimension],
                                       field(), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "same physical domain"):
                sample_on_grid(field, GridSpec((13,) * dimension, 6))

    def test_sampling_floor_intersects_stage_limits_and_restart(self):
        for dimension in (2, 3):
            original, data, partitions = problem(dimension)
            field = fwi.GaussianField(original.grid, background=(2100, 2600),
                                      sampling=SamplingConfig()).double()
            cfg = replace(configuration(dimension), steps_per_stage=2,
                          spatial_radius_schedule=(25, 5))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = fwi.invert(field, data, cfg, root / "fit", partitions=partitions)
                # Independent Fourier formula, not the policy helper.
                expected = 10 * (-2 * torch.log(torch.tensor(1e-3, dtype=torch.float64))).sqrt()
                expected = float(expected / torch.pi)
                self.assertAlmostEqual(field.sigma_min, expected)
                self.assertEqual(report["solver_calls"], {"forward": 8, "adjoint": 4})
                self.assertEqual([r["refinement_factor"] for r in report["sampling_diagnostics"]],
                                 [2, 4])
                restored, repeated = fwi.resume(root / "fit/stage_00.pt", data, root / "resume")
                assert_exact(restored.checkpoint(), field.checkpoint())
                assert_exact(report["sampling_diagnostics"], repeated["sampling_diagnostics"])

    def test_width_floor_does_not_certify_small_interpolation_error(self):
        field = fwi.GaussianField(GridSpec((7, 7), 10), background=(2500, 2500),
                                  sampling=SamplingConfig()).double()
        block = field.add_gaussians(torch.tensor([[25., 25.]]), torch.full((1, 2), 12.))
        with torch.no_grad():
            block.amplitudes.fill_(1000)
        diagnostic = sampling_diagnostics(field, refinement_factor=2)
        self.assertLess(diagnostic["maximum_ideal_kernel_nyquist_response"], 1e-3)
        self.assertGreater(diagnostic["interpolation_maximum_error_m_s"], 100)


if __name__ == "__main__":
    unittest.main()
