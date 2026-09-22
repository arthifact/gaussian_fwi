"""Spatial penalties: the accepted ones are unchanged, and TGV prefers affine fields."""

import unittest

import torch
import torch.nn.functional as F

from gaussian_fwi.core import Regularization


def legacy(velocity, spacing, tv_weight, tikhonov_weight,
           epsilon=0.1, velocity_scale=100.0, length_scale=10.0):
    """The penalty exactly as it was written before TGV was added."""
    squared = torch.zeros_like(velocity)
    for axis in range(velocity.ndim):
        difference = torch.diff(velocity, dim=axis)
        padding = [0] * (2 * velocity.ndim)
        padding[2 * (velocity.ndim - axis - 1) + 1] = 1
        gradient = F.pad(difference, padding) * length_scale / (spacing * velocity_scale)
        squared = squared + gradient.square()
    tv = (torch.sqrt(squared + epsilon**2) - epsilon).mean()
    return tv_weight * tv + tikhonov_weight * 0.5 * squared.mean()


def ramp(depth=40, width=40, gradient=30.0):
    rows = torch.arange(depth, dtype=torch.float64)[:, None].expand(depth, width)
    return 1500.0 + gradient * rows.clone()


class RegularizationTests(unittest.TestCase):
    def test_accepted_penalties_are_bitwise_unchanged(self):
        torch.manual_seed(0)
        velocity = 1500 + 2000 * torch.rand(30, 30, dtype=torch.float64)
        for tv, tikhonov in ((1e-4, 0.0), (0.0, 1e-3), (2e-4, 5e-4)):
            with self.subTest(tv=tv, tikhonov=tikhonov):
                penalty = Regularization(tv_weight=tv, tikhonov_weight=tikhonov)
                torch.testing.assert_close(penalty(velocity, 10.0),
                                           legacy(velocity, 10.0, tv, tikhonov),
                                           rtol=0, atol=0)

    def test_an_unpenalised_field_still_returns_a_connected_zero(self):
        velocity = ramp().requires_grad_(True)
        value = Regularization()(velocity, 10.0)
        self.assertEqual(float(value), 0.0)
        value.backward()
        self.assertTrue(torch.equal(velocity.grad, torch.zeros_like(velocity)))

    def test_tgv_prefers_an_affine_field_where_tv_penalises_the_trend(self):
        field = ramp()
        tgv = float(Regularization(tgv_weight=1.0)(field, 10.0))
        tv = float(Regularization(tv_weight=1.0)(field, 10.0))
        self.assertLess(tgv, tv / 10, "TGV should barely notice a linear trend")
        # The residual is the boundary convention, not an unconverged inner solve:
        # the outward difference is zero at the far edge, so the gradient steps
        # there and TGV correctly charges for it. More iterations do not remove it.
        longer = float(Regularization(tgv_weight=1.0, tgv_steps=40)(field, 10.0))
        self.assertAlmostEqual(tgv, longer, delta=0.2 * tgv)

    def test_tgv_still_charges_for_a_discontinuity(self):
        field = ramp()
        faulted = field.clone()
        faulted[field.shape[0] // 2:] += 800.0
        smooth = float(Regularization(tgv_weight=1.0)(field, 10.0))
        broken = float(Regularization(tgv_weight=1.0)(faulted, 10.0))
        self.assertGreater(broken, 10 * smooth, "a fault must cost more than a trend")

    def test_tgv_is_differentiable_through_its_inner_minimisation(self):
        field = ramp().requires_grad_(True)
        Regularization(tgv_weight=1.0)(field, 10.0).backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(bool(torch.isfinite(field.grad).all()))

    def test_invalid_tgv_settings_are_rejected(self):
        for kwargs in ({"tgv_weight": -1.0}, {"tgv_ratio": 0.0},
                       {"tgv_steps": 0}, {"tgv_step_size": -0.1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Regularization(**kwargs)


if __name__ == "__main__":
    unittest.main()
