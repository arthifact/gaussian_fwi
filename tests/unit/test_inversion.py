"""One-method integration, settling selection, work accounting and compatibility."""

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
from _problems import assert_exact, configuration, problem

import gaussian_fwi as fwi
import run
from gaussian_fwi.inversion import METHOD, TRAINING_FORMAT


class InversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_one_configuration_rejects_removed_method_switches(self):
        for removed in ({"levels": ((2, 2),)}, {"adaptation": {}}, {"density_control": {}},
                        {"spatial_radius_schedule": (50, 25)}):
            with self.assertRaises(TypeError):
                fwi.InversionConfig((10., 20.), **removed)
        for removed in ({"operations": ()}, {"comparison_steps": 3}, {"strategy": "original"}):
            with self.assertRaises(TypeError):
                fwi.RefinementConfig(**removed)
        field, _, _ = problem()
        profile = {"method": "scheduled", "field": {"bounds": [1500, 4500],
                                                       "background": [2100, 2600]},
                   "inversion": asdict(configuration()), "regularization": {}, "preprocessing": {}}
        with self.assertRaises(ValueError):
            run.configuration(profile, field.grid)
        self.assertFalse(hasattr(fwi, "AdaptationConfig"))
        self.assertFalse(hasattr(fwi, "DensityControlConfig"))

    def test_real_2d_3d_fits_have_one_seed_settling_selection_and_exact_work(self):
        for dimension in (2, 3):
            field, data, partitions = problem(dimension)
            cfg = configuration(dimension)
            before_calls = data.acquisition.counts
            before_rng = torch.random.get_rng_state().clone()
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "fit"
                report = fwi.invert(field, data, cfg, output, partitions=partitions)
                self.assertEqual(report["method"], METHOD)
                self.assertEqual(report["selection_window"], "settling")
                expected = {"forward": 2 + 2 * (8 + 1), "adjoint": 2 * 8}
                self.assertEqual(report["solver_calls"], expected)
                self.assertEqual({k: data.acquisition.counts[k] - before_calls[k] for k in expected},
                                 expected)
                self.assertEqual(report["optimizer_updates"], {"trajectory": 16, "refinement_trials": 0})
                assert_exact(torch.random.get_rng_state(), before_rng)
                history = json.loads((output / "history.json").read_text())
                self.assertEqual(history[0]["gaussians"], 2**dimension)
                for stage, summary in enumerate(report["stages"]):
                    self.assertGreaterEqual(summary["selected_step"], 6)
                    rows = [r for r in history if r["stage"] == stage]
                    settled = [r for r in rows if r["step"] >= 6]
                    self.assertEqual(len({r["gaussians"] for r in settled}), 1)
                    self.assertTrue(all(r["phase"] == "settling" for r in settled))
                    self.assertTrue(all(not r["selection_eligible"] for r in rows if r["step"] < 6))
                    if stage:
                        self.assertEqual(rows[0]["gaussians"], report["stages"][stage-1]["gaussians"])
                self.assertTrue(report["topology_history"])
                self.assertTrue(all(e["step"] < 6 for e in report["topology_history"]))
                payload = torch.load(output / "stage_00.pt", weights_only=True)
                self.assertEqual(payload["format"], TRAINING_FORMAT)
                restored = fwi.GaussianField.load(output / "field.pt")
                saved = torch.load(output / "predictions.pt", weights_only=True)
                assert_exact(restored(), saved["final_velocity"])
                with torch.no_grad():
                    assert_exact(data.acquisition.simulate(restored()), saved["final_prediction"])

    def test_completed_stage_restart_preserves_fields_adam_history_and_policy(self):
        field, data, partitions = problem()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = fwi.invert(field, data, configuration(), root / "fit", partitions=partitions)
            resumed, repeated = fwi.resume(root / "fit/stage_00.pt", data, root / "resumed")
            assert_exact(field.checkpoint(), resumed.checkpoint())
            for key in ("solver_calls", "stages", "topology_history", "optimizer_updates", "waveforms"):
                assert_exact(original[key], repeated[key])
            a = torch.load(root / "fit/stage_01.pt", weights_only=True)
            b = torch.load(root / "resumed/stage_01.pt", weights_only=True)
            for key in ("field", "optimizer", "specification", "topology_state", "topology_history"):
                assert_exact(a[key], b[key])
            self.assertEqual(a["specification"]["config"]["refinement"],
                             asdict(configuration().refinement))

    def test_old_training_algorithms_are_rejected_before_solves_or_output(self):
        _, data, _ = problem()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version in range(2, 11):
                checkpoint = root / f"v{version}.pt"
                torch.save({"format": f"gaussian-fwi-training-v{version}"}, checkpoint)
                counts = data.acquisition.counts
                output = root / f"output_{version}"
                with self.assertRaisesRegex(ValueError, "different algorithm"):
                    fwi.resume(checkpoint, data, output)
                self.assertFalse(output.exists())
                self.assertEqual(data.acquisition.counts, counts)

    def test_corrupt_stage_state_is_rejected_without_solving_or_writing(self):
        field, data, partitions = problem()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fwi.invert(field, data, configuration(), root / "fit", partitions=partitions)
            original = torch.load(root / "fit/stage_00.pt", weights_only=True)
            for name in ("policy", "stage", "ids", "count", "initial_waveforms"):
                payload = deepcopy(original)
                if name == "policy":
                    payload["specification"]["config"]["refinement"]["strategy"] = "other"
                elif name == "stage":
                    payload["completed_stage"] = 5
                elif name == "ids":
                    payload["topology_state"]["field_sha256"] = "0" * 64
                elif name == "count":
                    payload["solver_calls"]["adjoint"] = -1
                else:
                    payload["initial_prediction"] = torch.ones(1)
                checkpoint = root / f"{name}.pt"
                torch.save(payload, checkpoint)
                counts = data.acquisition.counts
                with self.assertRaises((ValueError, TypeError)):
                    fwi.resume(checkpoint, data, root / name)
                self.assertFalse((root / name).exists())
                self.assertEqual(data.acquisition.counts, counts)

    def test_float32_and_float64_complete_the_same_cycle_with_finite_outputs(self):
        velocities = []
        for dtype in (torch.float32, torch.float64):
            field, data, partitions = problem(dtype=dtype)
            with tempfile.TemporaryDirectory() as tmp:
                report = fwi.invert(field, data, configuration(), Path(tmp) / "fit", partitions=partitions)
                self.assertTrue(torch.isfinite(field()).all())
                self.assertTrue(all(s["selected_step"] >= 6 for s in report["stages"]))
                self.assertEqual(report["solver_calls"], {"forward": 20, "adjoint": 16})
                velocities.append(field().detach().double())
        # A declared cross-dtype integration tolerance, not bitwise trajectory equivalence.
        relative = torch.linalg.vector_norm(velocities[0] - velocities[1]) / velocities[1].norm()
        self.assertLess(float(relative), 5e-3)

    def test_validation_targets_cannot_change_density_decisions_within_a_stage(self):
        field, data, partitions = problem()
        changed = deepcopy(data)
        changed.traces[:, partitions["validation"]] *= -5
        cfg = replace(configuration(), cutoffs=(10.,))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = fwi.GaussianField.from_checkpoint(field.checkpoint())
            a = fwi.invert(field, data, cfg, root / "a", partitions=partitions)
            b = fwi.invert(other, changed, cfg, root / "b", partitions=partitions)
            assert_exact(a["topology_history"], b["topology_history"])
            first = json.loads((root / "a/history.json").read_text())
            second = json.loads((root / "b/history.json").read_text())
            self.assertEqual([r["train"] for r in first], [r["train"] for r in second])
            self.assertNotEqual([r["validation"] for r in first], [r["validation"] for r in second])
