"""Interrupted continuation must preserve the numerical trajectory and counted work."""
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from _problems import assert_exact, configuration, problem

import gaussian_fwi as fwi
import run
from gaussian_fwi.core.io import save_observations


class UpdateRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_boundaries_preserve_fields_adam_selection_gradients_and_work(self):
        for dimension in (2, 3):
            reference, data, partitions = problem(dimension)
            segmented = fwi.GaussianField.from_checkpoint(reference.checkpoint())
            cfg = configuration(dimension)
            rng = torch.random.get_rng_state().clone()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = fwi.invert(reference, data, cfg, root / "whole", partitions=partitions)
                start = dict(data.acquisition.counts)
                checkpoint = None
                # Before/after edits, within settling with a selected best, and across stages.
                for index, allowance in enumerate((1, 3, 3, 3, 6)):
                    output = root / f"segment_{index}"
                    execution = dict(max_updates=allowance, checkpoint_interval=3, diagnostics=True)
                    if checkpoint is None:
                        partial = fwi.invert(segmented, data, cfg, output, partitions=partitions, **execution)
                    else:
                        segmented, partial = fwi.resume(checkpoint, data, output, **execution)
                    self.assertEqual(partial["status"], "paused")
                    checkpoint = output / partial["checkpoint"]
                segmented, completed = fwi.resume(checkpoint, data, root / "finished", diagnostics=True)
                assert_exact(reference.checkpoint(), segmented.checkpoint())
                for key in ("solver_calls", "stages", "topology_history", "optimizer_updates", "waveforms"):
                    assert_exact(original[key], completed[key])
                actual = {key: data.acquisition.counts[key]-start[key] for key in start}
                self.assertEqual(actual, original["solver_calls"])
                a = torch.load(root / "whole/stage_01.pt", weights_only=True)
                b = torch.load(root / "finished/stage_01.pt", weights_only=True)
                for key in ("field", "optimizer", "topology_state"):
                    assert_exact(a[key], b[key])
                measured = [row for row in b["history"] if "gradient_l2_by_role" in row]
                self.assertTrue(measured)
                self.assertTrue(all(row["completed_update_field_change"]["maximum_m_s"] >= 0 for row in measured))
                assert_exact(torch.random.get_rng_state(), rng)

    def test_corrupt_boundary_is_rejected_before_output_or_propagation(self):
        model, data, partitions = problem()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paused = fwi.invert(model, data, configuration(), root / "fit", partitions=partitions, max_updates=7)
            saved = torch.load(root / "fit" / paused["checkpoint"], weights_only=True)
            for name in ("cursor", "scores", "best", "optimizer", "rng"):
                payload = deepcopy(saved)
                if name == "cursor":
                    payload["completed_step"] = 999
                elif name == "scores":
                    payload["controller_state"]["scores"] = {0: (-1., 2)}
                elif name == "best":
                    payload["best"]["step"] = 0
                elif name == "optimizer":
                    next(iter(payload["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
                else:
                    payload["rng"]["cpu"] = torch.ones(2)
                path = root / f"{name}.pt"
                torch.save(payload, path)
                before = dict(data.acquisition.counts)
                with self.assertRaises((ValueError, TypeError, FloatingPointError)):
                    fwi.resume(path, data, root / name)
                self.assertFalse((root / name).exists())
                self.assertEqual(data.acquisition.counts, before)

    def test_runner_pause_is_not_completion_and_resume_counts_only_new_work(self):
        model, data, partitions = problem()
        profile = {"field": {"background": [2100, 2600], "bounds": [1500, 4500]},
                   "inversion": asdict(configuration()), "regularization": {}, "preprocessing": {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "observations.pt"
            save_observations(data, partitions, bundle)
            paused = run.run_case(bundle, profile, root / "first", max_updates=7, diagnostics=True)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["verification_solver_calls"], {"forward": 0, "adjoint": 0})
            self.assertFalse((root / "first/fit/field.pt").exists())
            checkpoint = root / "first" / paused["checkpoint"]
            completed = run.run_case(bundle, profile, root / "final", resume_checkpoint=checkpoint)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["fit_solver_calls"], {"forward": 20, "adjoint": 16})
            for key in ("forward", "adjoint"):
                self.assertEqual(paused["segment_solver_calls"][key]+completed["segment_solver_calls"][key],
                                 completed["fit_solver_calls"][key])
            self.assertEqual(completed["verification_solver_calls"], {"forward": 1, "adjoint": 0})
            provenance = json.loads((root / "final/run.json").read_text())
            self.assertEqual(provenance["resume_checkpoint"], str(checkpoint.resolve()))
