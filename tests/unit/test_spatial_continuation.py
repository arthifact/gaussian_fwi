"""Independent radius, continuity, cancellation and acoustic restart oracles."""

import copy
import unittest

import numpy as np
import torch
from _problems import assert_exact, problem

import gaussian_fwi as fwi
from gaussian_fwi.core import GridSpec
from gaussian_fwi.spatial import radius_record, set_radius_floor


class SpatialContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


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





if __name__ == "__main__":
    unittest.main()
