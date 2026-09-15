"""Independent selection, sampling, state and event-boundary checks for density control."""

import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
from _problems import assert_exact

from gaussian_fwi import GaussianField, GridSpec, RefinementConfig
from gaussian_fwi._optimizer import validate_adam
from gaussian_fwi._topology import EditRejected, GaussianTopology
from gaussian_fwi.refinement import RefinementController


def prepared(dimension=2):
    field = GaussianField(GridSpec((21,) * dimension, 10), background=(2400, 2700),
                          backend="sparse_fused").double()
    centers = torch.tensor([[50.] * dimension, [100.] * dimension, [150.] * dimension])
    scales = torch.tensor([[8.] * dimension, [40.] * dimension, [12.] * dimension])
    block = field.add_gaussians(centers, scales)
    optimizer = torch.optim.Adam(field.parameter_groups(), amsgrad=True)
    for parameter in field.parameters():
        parameter.grad = torch.linspace(.1, .3, parameter.numel()).to(parameter).reshape_as(parameter)
    optimizer.step()
    with torch.no_grad():
        block.amplitudes.copy_(torch.tensor([10., -12., .1]))
        block.shears.fill_(.2)
    return field, optimizer, GaussianTopology(field)


def policy(**overrides):
    return RefinementConfig(**{
        "warmup_steps": 0, "interval": 1, "stop_fraction": .75, "minimum_age": 1,
        "split_extent_fraction": .1, "max_growth": 4, "max_prunes": 4,
        "max_field_change": 500., **overrides,
    })


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_schedule_has_explicit_warmup_periodicity_and_settling(self):
        config = RefinementConfig()
        self.assertEqual([s for s in range(1001) if config.due(s, 1000)],
                         [100, 150, 200, 250, 300, 350, 400, 450])
        self.assertEqual(config.stop_step(1000), 500)
        self.assertFalse(config.due(500, 1000))
        self.assertFalse(config.due(1000, 1000))
        self.assertTrue(config.due(950, 2000))
        for kwargs in ({"stop_fraction": 1.}, {"stop_fraction": 0.}, {"interval": 0},
                       {"warmup_steps": -1}, {"gradient_threshold": float("nan")},
                       {"max_field_change": 0}, {"seed": 2**63}):
            with self.assertRaises(ValueError):
                RefinementConfig(**kwargs)
        with self.assertRaises(ValueError):
            config.validate_stage(50)

    def test_center_scores_match_independent_physical_finite_differences(self):
        for dimension in (2, 3):
            field, optimizer, _ = prepared(dimension)
            weights = torch.linspace(-.01, .02, field().numel()).to(field.background)
            optimizer.zero_grad(set_to_none=True)
            (field().flatten() * weights).sum().backward()
            block = field.blocks[0]
            numerical = torch.zeros_like(block.centers)
            h = 1e-3
            for row in range(3):
                for axis in range(dimension):
                    original = float(block.centers[row, axis])
                    with torch.no_grad():
                        block.centers[row, axis] = original + h
                        plus = (field().flatten() * weights).sum()
                        block.centers[row, axis] = original - h
                        minus = (field().flatten() * weights).sum()
                        block.centers[row, axis] = original
                    numerical[row, axis] = (plus - minus) / (2 * h)
            torch.testing.assert_close(block.centers.grad, numerical, rtol=2e-5, atol=2e-7)
            controller = RefinementController(field, policy(), steps_per_stage=8)
            controller.observe(field)
            actual = torch.tensor([controller.scores[i][0] for i in range(3)], dtype=torch.float64)
            torch.testing.assert_close(actual, numerical.norm(dim=1), rtol=2e-5, atol=2e-7)

    def test_mean_gradient_norms_do_not_cancel_across_updates(self):
        field, optimizer, _ = prepared()
        controller = RefinementController(field, policy(warmup_steps=2), steps_per_stage=8)
        for step, sign in ((1, 1.), (2, -1.)):
            field.blocks[0].centers.grad = torch.tensor([[sign, 0.]]).double().repeat(3, 1)
            controller.observe(field)
            self.assertIsNone(controller.after_update(field, optimizer, step))
        self.assertEqual(controller.scores, {i: (2., 2) for i in range(3)})

    def test_single_event_selects_small_clone_large_split_and_signed_prune(self):
        for dimension in (2, 3):
            field, optimizer, _ = prepared(dimension)
            config = policy(max_gaussians=4)
            controller = RefinementController(field, config, steps_per_stage=8)
            field.blocks[0].centers.grad = torch.ones_like(field.blocks[0].centers)
            before = field().detach().clone()
            controller.observe(field)
            event = controller.after_update(field, optimizer, 1)
            selected = {e["parents"][0]: e["operation"] for e in event["operations"]}
            self.assertEqual(selected, {0: "clone", 1: "split", 2: "prune"})
            self.assertEqual(field.count, 4)
            self.assertEqual(event["additional_solver_calls"], {"forward": 0, "adjoint": 0})
            self.assertLessEqual(float((field() - before).abs().max()), config.max_field_change)
            self.assertEqual(controller.scores, {})
            validate_adam(field, optimizer)

    def test_exact_ties_and_resource_ceiling_have_stable_priority(self):
        field, optimizer, _ = prepared()
        config = policy(split_extent_fraction=1., max_growth=1, max_gaussians=4,
                        prune_amplitude=0.)
        controller = RefinementController(field, config, steps_per_stage=8)
        field.blocks[0].centers.grad = torch.ones_like(field.blocks[0].centers)
        controller.observe(field)
        event = controller.after_update(field, optimizer, 1)
        self.assertEqual([e["parents"] for e in event["operations"]], [[0]])
        self.assertEqual(field.count, 4)
        for step in (2, 3):
            for block in field.blocks:
                block.centers.grad = torch.ones_like(block.centers)
            controller.observe(field)
            controller.after_update(field, optimizer, step)
            self.assertEqual(field.count, 4)

    def test_zero_gradients_do_not_force_growth_and_settling_forbids_all_edits(self):
        field, optimizer, _ = prepared()
        config = policy(prune_amplitude=0.)
        controller = RefinementController(field, config, steps_per_stage=8)
        for step in range(1, 9):
            field.blocks[0].centers.grad = torch.zeros_like(field.blocks[0].centers)
            if step >= 6:
                with torch.no_grad():
                    field.blocks[0].amplitudes.zero_()
            controller.observe(field)
            event = controller.after_update(field, optimizer, step)
            if step >= 6:
                self.assertIsNone(event)
            else:
                self.assertEqual(event["operations"], [])
            self.assertEqual(field.count, 3)

    def test_sampled_split_geometry_is_reproducible_and_rng_is_private(self):
        for dimension in (2, 3):
            field, _, topology = prepared(dimension)
            before = torch.random.get_rng_state().clone()
            a = topology.split(1, seed=71)
            b = topology.split(1, seed=71)
            assert_exact(a.centers, b.centers)
            assert_exact(torch.random.get_rng_state(), before)
            _, covariance, amplitude = topology.values(1)
            torch.testing.assert_close(a.covariances * 2.56, covariance.expand(2, -1, -1),
                                       rtol=2e-15, atol=1e-12)
            assert_exact(a.amplitudes, amplitude.expand(2))
            self.assertTrue((torch.linalg.eigvalsh(a.covariances) > 0).all())

    def test_sampled_child_centers_follow_the_parent_distribution(self):
        for dimension in (2, 3):
            _, _, topology = prepared(dimension)
            center, covariance, _ = topology.values(1)
            samples = torch.cat([topology.split(1, seed=i).centers for i in range(2048)])
            whitened = torch.linalg.solve_triangular(torch.linalg.cholesky(covariance),
                                                    (samples - center).T, upper=False)
            self.assertLess(float(whitened.mean(dim=1).abs().max()), .06)
            torch.testing.assert_close(torch.cov(whitened), torch.eye(dimension).double(),
                                       rtol=.08, atol=.06)

    def test_batch_preserves_survivor_adam_and_gives_every_child_fresh_state(self):
        field, optimizer, topology = prepared()
        old = field.blocks[0]
        values = {n: p.detach().clone() for n, p in old.named_parameters()}
        gradients = {n: p.grad.clone() for n, p in old.named_parameters()}
        states = {n: deepcopy(optimizer.state[p]) for n, p in old.named_parameters()}
        background_state = deepcopy(optimizer.state[field.background])
        edits = [topology.clone(0), topology.split(1, seed=7), topology.prune(2)]
        event = topology.apply_many(edits, optimizer, step=3, max_field_change=500.)
        self.assertEqual(topology.ids, [[0], [3, 4, 5]])
        self.assertEqual(topology.last_edit_steps, [[3], [3, 3, 3]])
        self.assertEqual(event["operations"][0]["children"], [3])
        self.assertEqual(event["operations"][1]["children"], [4, 5])
        assert_exact(optimizer.state[field.background], background_state)
        for name, p in field.blocks[0].named_parameters():
            assert_exact(p.detach(), values[name][:1])
            assert_exact(p.grad, gradients[name][:1])
            assert_exact(optimizer.state[p], {k: v if k == "step" else v[:1]
                                             for k, v in states[name].items()})
        for p in field.blocks[1].parameters():
            self.assertFalse(optimizer.state.get(p))
        self.assertTrue(all(group["amsgrad"] for group in optimizer.param_groups))
        optimizer.zero_grad(set_to_none=True)
        field().square().mean().backward()
        optimizer.step()
        self.assertEqual(int(optimizer.state[field.blocks[1].amplitudes]["step"]), 1)
        self.assertEqual(int(optimizer.state[field.blocks[0].amplitudes]["step"]), 2)

    def test_failed_batch_restores_parameter_identities_gradients_moments_and_ids(self):
        field, optimizer, topology = prepared()
        parameters = tuple(field.parameters())
        gradients = [p.grad.clone() for p in parameters]
        before = field.checkpoint(), deepcopy(optimizer.state_dict()), topology.state_dict()
        with self.assertRaises(EditRejected):
            topology.apply_many([topology.clone(0), topology.clone(1)], optimizer,
                                step=3, max_field_change=1e-12)
        self.assertTrue(all(a is b for a, b in zip(parameters, field.parameters(), strict=True)))
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
        assert_exact(topology.state_dict(), before[2])
        for p, g in zip(parameters, gradients, strict=True):
            assert_exact(p.grad, g)

    def test_batch_checks_the_cumulative_change_instead_of_individual_limits(self):
        field, optimizer, _ = prepared()
        with torch.no_grad():
            block = field.blocks[0]
            block.centers[:] = 100.
            block.log_scales[:] = torch.tensor(15.).log()
            block.amplitudes[:] = 10.
        topology = GaussianTopology(field)
        raw = field.raw().detach()
        velocity = field().detach().flatten()
        edits = [topology.clone(0), topology.clone(1)]
        changes = [topology.raw_change(e) for e in edits]
        singles = [float((field.bound_velocity(raw + d) - velocity).abs().max()) for d in changes]
        joint = float((field.bound_velocity(raw + sum(changes)) - velocity).abs().max())
        limit = (max(singles) + joint) / 2
        self.assertLess(max(singles), limit)
        self.assertGreater(joint, limit)
        before = field.checkpoint(), deepcopy(optimizer.state_dict())
        with self.assertRaises(EditRejected):
            topology.apply_many(edits, optimizer, step=3, max_field_change=limit)
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])

    def test_unexpected_failure_rolls_back_the_whole_event_and_can_be_retried(self):
        field, optimizer, _ = prepared()
        controller = RefinementController(field, policy(), steps_per_stage=8)
        field.blocks[0].centers.grad = torch.ones_like(field.blocks[0].centers)
        controller.observe(field)
        before = field.checkpoint(), deepcopy(optimizer.state_dict()), controller.topology.state_dict()

        def validate_then_fail(model, adam):
            validate_adam(model, adam)
            if model.count != 3:
                raise RuntimeError("Injected failure after mutation")

        with patch("gaussian_fwi._topology.validate_adam", side_effect=validate_then_fail):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                controller.after_update(field, optimizer, 1)
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
        assert_exact(controller.topology.state_dict(), before[2])
        self.assertTrue(controller.pending)
        self.assertEqual(controller.last_step, 0)
        self.assertTrue(controller.after_update(field, optimizer, 1)["operations"])

    def test_small_sampling_floor_rejects_split_without_mutation(self):
        field, optimizer, _ = prepared()
        with torch.no_grad():
            field.blocks[0].log_scales.fill_(torch.tensor(field.sigma_min).log().item() + .01)
            field.blocks[0].shears.zero_()
            field.blocks[0].amplitudes.fill_(10.)
        controller = RefinementController(field, policy(split_extent_fraction=.001), steps_per_stage=8)
        field.blocks[0].centers.grad = torch.ones_like(field.blocks[0].centers)
        before = field.checkpoint()
        controller.observe(field)
        event = controller.after_update(field, optimizer, 1)
        self.assertEqual(event["rejected_geometry"], 3)
        self.assertEqual(event["operations"], [])
        assert_exact(field.checkpoint(), before)

    def test_invalid_or_duplicate_observation_cannot_advance_the_controller(self):
        field, optimizer, _ = prepared()
        controller = RefinementController(field, policy(), steps_per_stage=8)
        with self.assertRaises(ValueError):
            controller.after_update(field, optimizer, 1)
        field.blocks[0].centers.grad.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            controller.observe(field)
        self.assertFalse(controller.pending)
        field.blocks[0].centers.grad.zero_()
        controller.observe(field)
        with self.assertRaises(ValueError):
            controller.observe(field)
        with self.assertRaises(ValueError):
            controller.after_update(field, optimizer, 2)
