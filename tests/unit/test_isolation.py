"""Data isolation and fail-before-write contracts for inversion and restart."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_dynamic import assert_exact, configuration, problem

import dynamic_refinement as fwi
from fwi_core import Observations, Preprocessing
from fwi_core.physics import WaveformObjective


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_held_out_changes_cannot_change_training_objective_or_gradient(self):
        field, data, partitions = problem()
        changed = data.traces.clone()
        held_out = torch.cat((partitions["validation"], partitions["test"]))
        changed[:, held_out] = -17 * changed[:, held_out] + 0.01
        altered = Observations(data.acquisition, changed)
        values, gradients = [], []
        for observations in (data, altered):
            objective = WaveformObjective(
                observations, (10.0, 20.0), partitions, Preprocessing(1.5, 5)
            )
            prediction = (0.8 * data.traces).detach().clone().requires_grad_()
            loss = objective.losses(prediction, (10.0, 20.0)).mean()
            values.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, prediction)[0])
        assert_exact(values[0], values[1])
        assert_exact(gradients[0], gradients[1])
        self.assertTrue(gradients[0][:, partitions["train"]].any())
        self.assertFalse(gradients[0][:, held_out].any())

    def test_test_waveforms_cannot_change_the_selected_field_or_topology(self):
        field, data, partitions = problem()
        changed = data.traces.clone()
        changed[:, partitions["test"]] *= -7
        altered = Observations(data.acquisition, changed, sha256="changed-test-data")
        config = replace(configuration(), steps_per_stage=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = fwi.GaussianField.from_checkpoint(field.checkpoint())
            a = fwi.invert(field, data, config, root / "a", partitions=partitions)
            b = fwi.invert(reference, altered, config, root / "b", partitions=partitions)
            assert_exact(field.checkpoint(), reference.checkpoint())
            for key in ("stages", "topology_history", "solver_calls"):
                assert_exact(a[key], b[key])
            for split in ("train", "validation"):
                assert_exact(a["waveforms"][split], b["waveforms"][split])
            self.assertNotEqual(a["waveforms"]["test"], b["waveforms"]["test"])

    def test_invalid_objective_is_rejected_before_creating_output_or_solving(self):
        field, data, partitions = problem()
        variants = [
            (data, {**partitions, "validation": partitions["train"]}, configuration()),
            (
                Observations(data.acquisition, torch.zeros_like(data.traces)),
                partitions,
                configuration(),
            ),
            (data, partitions, replace(configuration(), cutoffs=(500.0,))),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for index, (observations, split, config) in enumerate(variants):
                with self.subTest(case=index):
                    destination = Path(temporary) / str(index)
                    before = data.acquisition.counts
                    with self.assertRaises(ValueError):
                        fwi.invert(field, observations, config, destination, partitions=split)
                    self.assertEqual(data.acquisition.counts, before)
                    self.assertFalse(destination.exists())

    def test_resume_rejects_observation_mismatch_without_creating_output(self):
        field, data, partitions = problem()
        config = replace(configuration(), steps_per_stage=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fwi.invert(field, data, config, root / "fit", partitions=partitions)
            altered = Observations(data.acquisition, data.traces, sha256="different-survey")
            before = data.acquisition.counts
            with self.assertRaisesRegex(ValueError, "do not match"):
                fwi.resume(root / "fit/stage_00.pt", altered, root / "invalid")
            self.assertEqual(data.acquisition.counts, before)
            self.assertFalse((root / "invalid").exists())

    def test_resume_cannot_overwrite_an_existing_run(self):
        field, data, partitions = problem()
        config = replace(configuration(), steps_per_stage=2)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "fit"
            fwi.invert(field, data, config, destination, partitions=partitions)
            before = {p.name: p.read_bytes() for p in destination.iterdir()}
            counts = data.acquisition.counts
            with self.assertRaises(FileExistsError):
                fwi.resume(destination / "stage_00.pt", data, destination)
            self.assertEqual(counts, data.acquisition.counts)
            self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})
