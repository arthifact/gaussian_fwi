"""Fast checks of the pieces the baseline depends on.

    python -m unittest discover -s tests
"""

import math
import unittest

import torch

import gaussian_fwi as g
from gaussian_fwi.adapt import AdaptConfig, Adapter, covariance, from_covariance

START = g.linear_start((40, 40))


def populated(lattice=4, seed=0):
    torch.manual_seed(seed)
    field = g.GaussianField(START, 10.0, lattice=lattice)
    with torch.no_grad():
        field.amplitude.copy_(torch.randn(field.count) * 300)
        field.angle.copy_(torch.rand(field.count) * 3 - 1.5)
        field.log_width[:, 0] += 0.3
    return field


def with_state(field):
    optimizer = torch.optim.Adam(field.parameter_groups())
    field().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


class FieldTests(unittest.TestCase):
    def test_single_gaussian_peak_and_rotation(self):
        field = g.GaussianField(torch.full((40, 40), 2000.0), 10.0, lattice=2)
        with torch.no_grad():
            field.center.copy_(torch.tensor([[200.0, 100.0]] * 4))
            field.log_width.copy_(torch.tensor([[60.0, 20.0]] * 4).log())
            field.amplitude.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))
            field.angle.fill_(math.pi / 2)            # long axis now vertical
        velocity = field().detach()
        self.assertAlmostEqual(float(velocity[10, 20]), 2100.0, delta=0.5)
        # One width (60 m) down the long axis, now z, is exp(-1/2) of the peak.
        self.assertAlmostEqual(float(velocity[16, 20]) - 2000.0, 100 * math.exp(-0.5), delta=0.5)
        # Four short widths (80 m) across the axis it has all but vanished.
        self.assertLess(float(velocity[10, 28]) - 2000.0, 0.1)

    def test_soft_bounds(self):
        field = g.GaussianField(START, 10.0, lattice=2)
        with torch.no_grad():
            field.amplitude.fill_(1e5)
        # float32 rounding of a 1e5 m/s input, not a bound violation
        self.assertLessEqual(float(field().max()), 5000.0 + 0.05)

    def test_covariance_round_trip(self):
        log_width = torch.tensor([[4.0, 3.0], [3.5, 3.4]])
        angle = torch.tensor([0.7, -1.2])
        back_width, back_angle = from_covariance(covariance(log_width, angle))
        self.assertTrue(torch.allclose(back_width, log_width, atol=1e-4))
        self.assertTrue(torch.allclose(back_angle, angle, atol=1e-4))


class AdaptTests(unittest.TestCase):
    def test_split_preserves_velocity(self):
        field = populated()
        optimizer = with_state(field)
        before = field().detach()
        adapter = Adapter(AdaptConfig(split_factor=0.0, max_splits=16, prune_amplitude=0.0,
                                      merge_distance=0.0))
        adapter.observe(field, torch.randn(START.numel()))
        adapter.edit(field, optimizer, cutoff_hz=4.0, step=1)
        self.assertEqual(field.count, 32)
        change = (field().detach() - before).norm() / (before - START).norm()
        self.assertLess(float(change), 0.05)

    def test_merge_and_prune(self):
        field = g.GaussianField(START, 10.0, lattice=2)
        with torch.no_grad():
            field.center.copy_(torch.tensor([[200.0, 200.0], [215.0, 205.0],
                                             [50.0, 350.0], [350.0, 50.0]]))
            field.amplitude.copy_(torch.tensor([200.0, 150.0, 0.0, -300.0]))
            field.log_width.copy_(torch.tensor([[60.0, 40.0]] * 4).log())
        field.age[:] = 100
        optimizer = with_state(field)
        with torch.no_grad():
            field.amplitude[2] = 0.0       # the Adam step above moved it off zero
        before = field().detach()
        adapter = Adapter(AdaptConfig(split_factor=1e9))
        adapter.observe(field, torch.zeros(START.numel()))
        adapter.edit(field, optimizer, cutoff_hz=4.0, step=1)
        self.assertEqual((adapter.log[-1]["merged"], adapter.log[-1]["pruned"]), (1, 1))
        self.assertEqual(field.count, 2)
        change = (field().detach() - before).norm() / (before - START).norm()
        self.assertLess(float(change), 0.05)

    def test_children_inherit_adam_moments(self):
        field = populated(lattice=2)
        optimizer = with_state(field)
        parent = optimizer.state[field.amplitude]["exp_avg"].clone()
        adapter = Adapter(AdaptConfig(split_factor=0.0, prune_amplitude=0.0,
                                      merge_distance=0.0))
        adapter.observe(field, torch.randn(START.numel()))
        optimizer = adapter.edit(field, optimizer, cutoff_hz=4.0, step=1)
        moved = optimizer.state[field.amplitude]["exp_avg"]
        self.assertEqual(len(moved), 8)
        self.assertTrue(torch.allclose(torch.sort(moved).values,
                                       torch.sort(parent.repeat_interleave(2)).values))

    def test_no_split_below_band_resolution(self):
        field = populated()
        with torch.no_grad():
            field.log_width.fill_(math.log(15.0))
        optimizer = with_state(field)
        adapter = Adapter(AdaptConfig(split_factor=0.0, prune_amplitude=0.0,
                                      merge_distance=0.0))
        adapter.observe(field, torch.randn(START.numel()))
        adapter.edit(field, optimizer, cutoff_hz=4.0, step=1)   # 4 Hz resolves ~90 m
        self.assertEqual(adapter.log[-1]["split"], 0)


    def test_split_children_are_not_merged_straight_back(self):
        field = populated(lattice=2)
        field.age[:] = 100
        optimizer = with_state(field)
        adapter = Adapter(AdaptConfig(split_factor=0.0, prune_amplitude=0.0))
        adapter.start_band(steps=40)                 # four edits, nine steps apart
        adapter.observe(field, torch.randn(START.numel()))
        optimizer = adapter.edit(field, optimizer, 4.0, 9)
        self.assertEqual(field.count, 8)
        field.age += adapter.interval                # one edit later, still one band young
        adapter.observe(field, torch.zeros(START.numel()))
        adapter.config = AdaptConfig(split_factor=1e9, prune_amplitude=0.0)
        adapter.edit(field, optimizer, 4.0, 18)
        self.assertEqual(adapter.log[-1]["merged"], 0)

    def test_growth_stops_when_held_out_fit_stops_improving(self):
        field = populated()
        optimizer = with_state(field)
        adapter = Adapter(AdaptConfig(split_factor=0.0, max_splits=2, prune_amplitude=0.0,
                                      merge_distance=0.0, min_improvement=0.01))
        for misfit, expected in ((0.50, 2), (0.40, 2), (0.399, 0), (0.30, 0)):
            adapter.observe(field, torch.randn(START.numel()))
            optimizer = adapter.edit(field, optimizer, 4.0, 1, validation_misfit=misfit)
            self.assertEqual(adapter.log[-1]["split"], expected)
        adapter.start_band()            # a new band may grow again
        adapter.observe(field, torch.randn(START.numel()))
        adapter.edit(field, optimizer, 7.0, 1, validation_misfit=0.30)
        self.assertEqual(adapter.log[-1]["split"], 2)


class WaveTests(unittest.TestCase):
    def test_finer_modelling_grid_keeps_amplitudes(self):
        reference = torch.full((40, 40), 2500.0)
        config = dict(samples=400, shots=2, signal_to_noise=None, wavelet_error=0.0)
        _, coarse = g.synthetic(reference, g.DataConfig(refinement=1, **config))
        _, fine = g.synthetic(reference, g.DataConfig(refinement=2, **config))
        ratio = fine.norm() / coarse.norm()
        self.assertAlmostEqual(float(ratio), 1.0, delta=0.1)

    def test_start_recovers_a_depth_trend(self):
        reference = g.linear_start((40, 40), 1800.0, 3600.0)
        survey, observed = g.synthetic(reference, g.DataConfig(
            samples=400, shots=3, signal_to_noise=None, wavelet_error=0.0, refinement=1))
        start, info = g.estimate_start(survey, observed, (40, 40), top=(1500.0, 2100.0, 7), bottom=(2400.0, 4800.0, 7), refinements=1)
        # A 380 m spread over a 0.4 s record constrains the shallow part best;
        # below that the trend is an extrapolation and is not checked.
        error = (start[:10] - reference[:10]).abs().mean()
        self.assertLess(float(error), 100.0)
        self.assertAlmostEqual(info["top_m_s"], 1800.0, delta=100.0)

    def test_fit_reduces_misfit_and_adapts(self):
        torch.manual_seed(0)
        reference = START.clone()
        reference[15:25, 10:30] += 400.0
        survey, observed = g.synthetic(reference, g.DataConfig(
            samples=400, shots=3, signal_to_noise=None, wavelet_error=0.0, refinement=1))
        field = g.GaussianField(START, 10.0, lattice=3)
        result = g.fit(field, survey, observed, cutoffs=(6.0, 10.0), steps=60,
                       adapter=Adapter(),
                       validation=g.holdout(observed.shape[1]))
        last = result["history"][-1]
        self.assertLess(last["train_misfit"], 0.5)
        self.assertLess(last["validation_misfit"], 0.5)
        self.assertTrue(result["edits"])
        error = g.velocity_errors(field().detach(), reference)["rmse"]
        start = g.velocity_errors(START, reference)["rmse"]
        self.assertLess(error, start)


if __name__ == "__main__":
    unittest.main()
