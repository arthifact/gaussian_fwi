"""Full-survey gradients, unequal batch weights, and completed-update recovery."""
import json
import tempfile
import unittest
import weakref
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from _problems import assert_exact, configuration, problem

import gaussian_fwi as fwi
import run
from gaussian_fwi.core import (
    Acquisition,
    Observations,
    Preprocessing,
    Regularization,
    WaveformObjective,
)
from gaussian_fwi.core.footprints import FootprintAcquisition, GaussianFootprint
from gaussian_fwi.core.io import save_observations
from gaussian_fwi.core.physics import lowpass
from gaussian_fwi.refinement import RefinementController


def fixture(dimension=2):
    field, data, partitions = problem(dimension)
    base = data.acquisition
    indices = torch.tensor([0, 1, 0, 1, 0])
    amplitudes = base.source_amplitudes[indices] * torch.tensor(
        [1., .05, 3., .6, 1.4], dtype=torch.float64)[:, None, None]
    base = Acquisition(base.grid, base.dt, amplitudes, base.source_locations[indices],
                       base.receiver_locations[indices], pml_width=6, pml_frequency=25)
    acq = FootprintAcquisition(base, GaussianFootprint(1.5, 1.5), shot_batch_size=2)
    with torch.no_grad():
        traces = acq.simulate(field() + 80)
    return field, Observations(acq, traces), partitions


class ShotAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_filtered_targets_release_padding_at_construction_without_changing_values(self):
        _, data, partitions = fixture()
        cutoffs = configuration().cutoffs
        before = dict(data.acquisition.counts)
        objective = WaveformObjective(data, cutoffs, partitions)
        self.assertEqual(before, data.acquisition.counts)
        for cutoff in cutoffs:
            expected = lowpass(data.traces, objective.dt, cutoff).detach()
            target = objective.targets[cutoff]
            self.assertGreater(expected.untyped_storage().nbytes(), expected.numel()*expected.element_size())
            self.assertTrue(torch.equal(target, expected))
            self.assertFalse(target.requires_grad)
            self.assertEqual(target.untyped_storage().nbytes(), target.numel()*target.element_size())

    def test_completed_batch_releases_nodal_traces_before_next_forward(self):
        model, data, _ = fixture()
        acquisition = data.acquisition._batches[0][1]
        original = acquisition.simulate
        references = []

        def observe(*args, **kwargs):
            traces = original(*args, **kwargs)
            references.append(weakref.ref(traces))
            return traces

        with patch.object(acquisition, "simulate", observe):
            batches = data.acquisition.simulate_batches(model())
            _, logical_prediction = next(batches)
            logical_prediction.square().sum().backward()
            self.assertIsNone(references[0](), "Completed nodal traces still occupy the prior batch's storage")

    def test_uncompressed_storage_identity_gradient_and_restart_policy(self):
        model, data, parts = fixture()
        identity = data.content_identity()
        values = []
        with tempfile.TemporaryDirectory() as scratch:
            for storage in ("device", "cpu", "disk"):
                velocity = model().detach().requires_grad_()
                prediction = data.acquisition.simulate(velocity, wavefield_storage=storage,
                                                       storage_path=scratch if storage == "disk" else None)
                gradient, = torch.autograd.grad(prediction.square().sum(), velocity)
                values.append((prediction.detach(), gradient))
        assert_exact(values[0], values[1])
        assert_exact(values[0], values[2])
        self.assertEqual(identity, data.content_identity())
        before = dict(data.acquisition.counts)
        with self.assertRaises(ValueError):
            data.acquisition.simulate(model(), wavefield_storage="none")
        with self.assertRaises(ValueError):
            data.acquisition.simulate(model(), wavefield_storage="disk")
        self.assertEqual(before, data.acquisition.counts)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"field": {"background": [2100, 2600], "bounds": [1500, 4500]},
                       "inversion": asdict(configuration()), "regularization": {}, "preprocessing": {}}
            bundle = root / "observations.pt"
            save_observations(data, parts, bundle)
            partial = run.run_case(bundle, profile, root / "part", max_updates=3,
                                    accumulate_shots=True, wavefield_storage="disk")
            checkpoint = root / "part" / partial["checkpoint"]
            before = dict(data.acquisition.counts)
            with self.assertRaises(ValueError):
                fwi.resume(checkpoint, data, root / "wrong", wavefield_storage="device")
            self.assertFalse((root / "wrong").exists())
            self.assertEqual(before, data.acquisition.counts)
            with self.assertRaises(ValueError):
                run.run_case(bundle, profile, root / "wrong_runner", resume_checkpoint=checkpoint,
                             wavefield_storage="device")
            self.assertFalse((root / "wrong_runner").exists())
            complete = run.run_case(bundle, profile, root / "final", resume_checkpoint=checkpoint)
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(json.loads((root / "final/run.json").read_text())["execution"]["wavefield_storage"], "disk")
            self.assertFalse(list((root / "final/fit/wavefields").iterdir()))

    def test_unequal_batches_match_full_loss_every_gradient_density_score_and_adam(self):
        for dimension in (2, 3):
            field, data, partitions = fixture(dimension)
            field.add_grid_level((2,) * dimension)
            with torch.no_grad():
                for block in field.blocks:
                    block.amplitudes.copy_(torch.linspace(-55, 95, len(block.amplitudes)))
                    block.shears.fill_(.13)
            cfg = replace(configuration(dimension), seed_shape=None)
            preprocessing = Preprocessing(time_gain_power=1.3, trace_balance_cap=3.)
            objective = WaveformObjective(data, cfg.cutoffs, partitions, preprocessing)
            regularization = Regularization(tv_weight=.004, tikhonov_weight=.001)
            outcomes = []
            for accumulated in (False, True):
                model = fwi.GaussianField.from_checkpoint(field.checkpoint())
                optimizer = torch.optim.Adam(model.parameter_groups(**cfg.learning_rates()))
                controller = RefinementController(model, cfg.refinement,
                                                  steps_per_stage=cfg.steps_per_stage, stage=0)
                before = dict(data.acquisition.counts)
                before_batches = dict(data.acquisition.batch_counts)
                velocity = model()
                # A second decoder contribution ensures its gradient is added once.
                penalty = regularization(velocity, model.grid.spacing) + 1e-6*model.raw().square().mean()
                self.assertGreater(float(penalty.detach()), 0)
                if accumulated:
                    prediction, bands, gradient = objective.accumulate_shot_gradients(
                        data.acquisition, velocity, cfg.cutoffs)
                    torch.autograd.backward((velocity, penalty), (gradient, torch.ones_like(penalty)))
                else:
                    prediction = data.acquisition.simulate(velocity)
                    bands = objective.losses(prediction, cfg.cutoffs)
                    (bands.mean() + penalty).backward()
                controller.observe(model)
                gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
                optimizer.step()
                outcomes.append((prediction.detach(), bands.detach(), gradients, controller.scores,
                                 {name: p.detach().clone() for name, p in model.named_parameters()}))
                self.assertEqual({key: data.acquisition.counts[key]-before[key] for key in before},
                                 {"forward": 5, "adjoint": 5})
                self.assertEqual({key: data.acquisition.batch_counts[key]-before_batches[key]
                                  for key in before_batches}, {"forward": 3, "adjoint": 3})
            for actual, expected in zip(outcomes[1][:4], outcomes[0][:4], strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
            torch.testing.assert_close(outcomes[1][4], outcomes[0][4], rtol=2e-2, atol=1e-7)

            # Independent directional derivative of the ordinary full objective.
            gradient = outcomes[1][2]
            directions = {name: torch.sin(torch.arange(p.numel(), dtype=p.dtype)+.2).reshape_as(p)*.01
                          for name, p in field.named_parameters()}
            predicted = sum((gradient[name]*direction).sum() for name, direction in directions.items())
            values = []
            with torch.no_grad():
                for sign in (-1, 1):
                    shifted = fwi.GaussianField.from_checkpoint(field.checkpoint())
                    for name, p in shifted.named_parameters():
                        p.add_(directions[name], alpha=sign*1e-4)
                    velocity = shifted()
                    loss = objective.losses(data.acquisition.simulate(velocity), cfg.cutoffs).mean()
                    values.append(loss + regularization(velocity, field.grid.spacing)
                                  + 1e-6*shifted.raw().square().mean())
            torch.testing.assert_close(predicted, (values[1]-values[0])/2e-4, rtol=2e-5, atol=2e-11)

    def test_accumulated_fit_restarts_exactly_and_test_data_cannot_change_it(self):
        model, data, partitions = fixture()
        cfg = configuration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = fwi.invert(model, data, cfg, root / "whole", partitions=partitions,
                                  accumulate_shots=True)
            first = fwi.GaussianField(model.grid, background=(2100, 2600)).double()
            partial = fwi.invert(first, data, cfg, root / "partial", partitions=partitions,
                                 accumulate_shots=True, max_updates=3)
            checkpoint = root / "partial" / partial["checkpoint"]
            first, partial = fwi.resume(checkpoint, data, root / "settling", max_updates=4)
            checkpoint = root / "settling" / partial["checkpoint"]
            first, complete = fwi.resume(checkpoint, data, root / "final")
            assert_exact(model.checkpoint(), first.checkpoint())
            self.assertEqual(original["solver_calls"], {"forward": 100, "adjoint": 80})
            for key in ("stages", "topology_history", "solver_calls", "waveforms"):
                assert_exact(original[key], complete[key])
            for key in ("field", "optimizer", "topology_state"):
                assert_exact(torch.load(root / "whole/stage_01.pt", weights_only=True)[key],
                             torch.load(root / "final/stage_01.pt", weights_only=True)[key])
            changed = data.traces.clone()
            changed[:, partitions["test"]] *= -17
            other = fwi.GaussianField(model.grid, background=(2100, 2600)).double()
            isolated = fwi.invert(other, Observations(data.acquisition, changed), cfg, root / "isolated",
                                  partitions=partitions, accumulate_shots=True)
            assert_exact(model.checkpoint(), other.checkpoint())
            assert_exact(original["topology_history"], isolated["topology_history"])
            self.assertNotEqual(original["waveforms"]["test"], isolated["waveforms"]["test"])
            history = json.loads((root / "whole/history.json").read_text())
            for row in history:
                n = row["stage"] * cfg.steps_per_stage + row["step"]
                self.assertEqual(row["solver_calls"]["adjoint"],
                                 5 * (n + (row["step"] != cfg.steps_per_stage)))

    def test_runtime_identity_rejects_changed_accumulation_before_writes_or_solves(self):
        model, data, partitions = fixture()
        profile = {"field": {"background": [2100, 2600], "bounds": [1500, 4500]},
                   "inversion": asdict(configuration()), "regularization": {}, "preprocessing": {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "observations.pt"
            save_observations(data, partitions, bundle)
            partial = run.run_case(bundle, profile, root / "partial", accumulate_shots=True, max_updates=3)
            checkpoint = root / "partial" / partial["checkpoint"]
            before = dict(data.acquisition.counts)
            with self.assertRaises(ValueError):
                fwi.resume(checkpoint, data, root / "wrong", accumulate_shots=False)
            self.assertFalse((root / "wrong").exists())
            self.assertEqual(before, data.acquisition.counts)
            with self.assertRaises(ValueError):
                run.run_case(bundle, profile, root / "wrong_runner", resume_checkpoint=checkpoint,
                             accumulate_shots=False)
            self.assertFalse((root / "wrong_runner").exists())
            complete = run.run_case(bundle, profile, root / "final", resume_checkpoint=checkpoint)
            self.assertEqual(complete["fit_shot_solves"], {"forward": 100, "adjoint": 80})
            execution = json.loads((root / "final/run.json").read_text())["execution"]
            self.assertTrue(execution["accumulate_shots"])
            plain_model, plain_data, plain_parts = problem()
            with self.assertRaisesRegex(ValueError, "finite-footprint"):
                fwi.invert(plain_model, plain_data, configuration(), root / "point",
                           partitions=plain_parts, accumulate_shots=True)
            self.assertFalse((root / "point").exists())
