"""Independent decision and state-restoration oracles for recovery trials."""

import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from test_dynamic import assert_exact, configuration, problem
from test_topology import initialized_field

import dynamic_refinement as fwi
from fwi_core import Observations
from gaussian_fwi import SamplingConfig
from gaussian_fwi.adaptation import GaussianCheckpoint
from gaussian_fwi.refinement import RefinementConfig, RefinementController
from gaussian_fwi.trials import TrainingScores


class RecoveryTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_positive_first_order_gain_does_not_imply_lower_actual_loss(self):
        # Independently known quadratic: loss(0)=1, loss(3)=4, predicted gain=6.
        start = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss = (start - 1).square()
        gradient, = torch.autograd.grad(loss, start)
        delta = 3.0
        self.assertGreater(float(-gradient * delta), 0)
        proposed = float((start.detach() + delta - 1).square())
        self.assertGreater(proposed, float(loss.detach()))
        keep, _ = TrainingScores(1.0, 1.0).accepts(
            TrainingScores(proposed, proposed), rtol=1e-6, atol=1e-12, shot_support=0,
        )
        self.assertFalse(keep)

    def test_zero_amplitude_birth_only_helps_after_equal_recovery_work(self):
        field, optimizer, _ = initialized_field()
        controller = RefinementController(field, RefinementConfig(comparison_steps=2))
        before = GaussianCheckpoint()
        before.consider(0, 0, field, optimizer)
        metadata = controller.topology.state_dict()
        gradients = {p: p.grad.clone() for p in field.parameters()}
        controller.topology.apply(
            controller.topology.insert(field.background.new_tensor([40., 40.]), 12),
            optimizer, step=1, max_field_change=25, learning_rates={"amplitude_lr": 4.},
        )

        def evaluate(model):
            zero = sum(p.sum()*0 for p in model.parameters())
            amplitude = model.blocks[-1].amplitudes[0] if model.count == 4 else zero
            loss = (amplitude - 8).square() + zero
            return loss, TrainingScores(float(loss.detach()), float(loss.detach()))

        with torch.no_grad():
            keep, record = controller._assess(field, optimizer, before, metadata, gradients, evaluate)
        self.assertTrue(keep)
        self.assertEqual(record["anchors"]["ordinary"]["objective"], 64.)
        self.assertEqual(record["anchors"]["proposed"]["objective"], 64.)
        self.assertEqual(record["ordinary"]["objective"], 64.)
        self.assertLess(record["proposed"]["objective"], 1.)
        self.assertEqual(float(field.blocks[-1].amplitudes[0].detach()), 0.)
        self.assertEqual(record["branch_work"]["ordinary"], record["branch_work"]["proposed"])
        self.assertEqual(record["work"], {"evaluations": 6, "backwards": 4, "optimizer_updates": 4})
        self.assertEqual(record["objective_changes"]["proposed_minus_ordinary_at_anchor"], 0.)

    def test_direct_sampling_recovery_counts_restart_and_held_out_isolation(self):
        field, data, partitions = problem()
        field = fwi.GaussianField(field.grid, background=(2100, 2600), sampling=SamplingConfig()).double()
        other = fwi.GaussianField.from_checkpoint(field.checkpoint())
        cfg = configuration()
        cfg = replace(cfg, refinement=replace(cfg.refinement, insertion_screening="coverage_aware",
                                             comparison_steps=2, operations=("insert",)))
        changed_traces = data.traces.clone()
        changed_traces[:, partitions["test"]] += 100
        changed = Observations(data.acquisition, changed_traces, {}, "different-test")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = fwi.invert(field, data, cfg, root / "fit", partitions=partitions)
            fwi.invert(other, changed, cfg, root / "changed", partitions=partitions)
            assert_exact(field.checkpoint(), other.checkpoint())
            resumed, _ = fwi.resume(root / "fit/stage_00.pt", data, root / "resume")
            assert_exact(resumed.checkpoint(), field.checkpoint())
            events = [e for e in report["topology_history"] if "trial" in e]
            self.assertTrue(events)
            self.assertEqual(report["solver_calls"], {"forward": 20 + 6*len(events),
                                                    "adjoint": 16 + 4*len(events)})
            for event in events:
                trial = event["trial"]
                self.assertAlmostEqual(event["objective_change"]["measured_equal_work_decrease"],
                                       trial["ordinary"]["objective"] - trial["proposed"]["objective"])
                self.assertEqual(trial["branch_work"]["ordinary"], trial["branch_work"]["proposed"])

    def test_trials_restore_selected_anchor_moments_gradients_and_retire_rejected_ids(self):
        for choose_candidate in (False, True):
            with self.subTest(choose_candidate=choose_candidate):
                field, optimizer, _ = initialized_field()
                controller = RefinementController(field, RefinementConfig(comparison_steps=3))
                before = GaussianCheckpoint()
                self.assertTrue(before.consider(0, 0, field, optimizer))
                metadata = controller.topology.state_dict()
                gradients = {p: p.grad.detach().clone() for p in field.parameters()}
                original = (field.checkpoint(), deepcopy(optimizer.state_dict()),
                            {name: p.grad.detach().clone() for name, p in field.named_parameters()})
                controller.topology.apply(
                    controller.topology.insert(field.background.new_tensor([40.0, 40.0]), 12),
                    optimizer, step=1, max_field_change=25, learning_rates={"amplitude_lr": 4.0},
                )
                candidate_metadata = controller.topology.state_dict()
                candidate = (field.checkpoint(), deepcopy(optimizer.state_dict()),
                             {name: None if p.grad is None else p.grad.detach().clone()
                              for name, p in field.named_parameters()})
                calls = []

                def evaluate(model):
                    # Branch offsets force a known decision, independently of
                    # the optimizer. The differentiable term exercises moments.
                    offset = 0 if (model.count == 4) == choose_candidate else 10
                    value = sum(p.square().sum() * 1e-10 for p in model.parameters()) + offset
                    score = float(value.detach())
                    calls.append(model.count)
                    return value, TrainingScores(score, score)

                with torch.no_grad():
                    keep, evidence = controller._assess(
                        field, optimizer, before, metadata, gradients, evaluate,
                    )
                self.assertEqual(keep, choose_candidate)
                self.assertEqual(calls, [3] * 4 + [4] * 4)
                self.assertEqual(evidence["work"],
                                 {"evaluations": 8, "backwards": 6, "optimizer_updates": 6})
                expected = candidate if choose_candidate else original
                assert_exact(field.checkpoint(), expected[0])
                assert_exact(optimizer.state_dict(), expected[1])
                assert_exact({name: p.grad for name, p in field.named_parameters()}, expected[2])
                identities = deepcopy(candidate_metadata if choose_candidate else metadata)
                identities["next_id"] = candidate_metadata["next_id"]
                assert_exact(controller.topology.state_dict(), identities)

    def test_failed_candidate_forecast_restores_the_unedited_state(self):
        field, optimizer, _ = initialized_field()
        controller = RefinementController(field, RefinementConfig(comparison_steps=2))
        before = GaussianCheckpoint()
        before.consider(0, 0, field, optimizer)
        metadata = controller.topology.state_dict()
        gradients = {p: p.grad.clone() for p in field.parameters()}
        original_field, original_adam = field.checkpoint(), deepcopy(optimizer.state_dict())
        controller.topology.apply(
            controller.topology.insert(field.background.new_tensor([40., 40.]), 12),
            optimizer, step=1, max_field_change=25, learning_rates={"amplitude_lr": 4.},
        )

        def evaluate(model):
            if model.count == 4:
                with torch.no_grad():
                    model.background.fill_(float("nan"))
                raise FloatingPointError("Independent failed-candidate witness")
            value = sum(p.square().sum()*1e-10 for p in model.parameters())
            return value, TrainingScores(float(value.detach()), float(value.detach()))

        with torch.no_grad():
            keep, record = controller._assess(field, optimizer, before, metadata, gradients, evaluate)
        self.assertFalse(keep)
        self.assertIsNone(record["proposed"])
        self.assertIsNone(record["objective_changes"]["proposed_minus_ordinary_after_recovery"])
        self.assertEqual(record["work"], {"evaluations": 4, "backwards": 2, "optimizer_updates": 2})
        assert_exact(field.checkpoint(), original_field)
        assert_exact(optimizer.state_dict(), original_adam)
        for parameter, gradient in gradients.items():
            assert_exact(parameter.grad, gradient)
