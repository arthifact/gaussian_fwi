"""Observation-only runner checks, without bundled geological inputs."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch
from _problems import configuration, problem

import run
from gaussian_fwi.core import Preprocessing, Regularization
from gaussian_fwi.core.io import load_observations, save_observations


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_runner_exposes_refinement_and_saved_field_diagnostics(self):
        field, data, partitions = problem()
        cfg = configuration()
        profile = {"field": {"background": [2100, 2600], "bounds": [1500, 4500],
                                                   "backend": "sparse_fused",
                                                   "sampling": {"max_nyquist_response": .001}},
                   "inversion": asdict(cfg), "regularization": asdict(Regularization()),
                   "preprocessing": asdict(Preprocessing())}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit"
            invalid = {**profile, "field": {**profile["field"], "backend": "unknown"}}
            before = data.acquisition.counts
            with self.assertRaises(ValueError):
                run.fit_observations(data, partitions, invalid, path)
            self.assertFalse(path.exists())
            self.assertEqual(data.acquisition.counts, before)
            fitted, report = run.fit_observations(data, partitions, profile, path)
            self.assertEqual(fitted.backend, "sparse_fused")
            self.assertTrue(bool(report["topology_history"]))
            self.assertTrue((path / "sampling.json").is_file())
            before = data.acquisition.counts
            run.verify_fit(fitted, data, report, path)
            self.assertEqual(data.acquisition.counts["forward"]-before["forward"], 1)

    def test_portable_bundle_identity_and_partition_rejection(self):
        _, data, partitions = problem()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.pt"
            save_observations(data, partitions, path)
            loaded, split = load_observations(
                path, expected_content_sha256=data.content_identity()["sha256"],
            )
            self.assertEqual(loaded.content_identity(), data.content_identity())
            torch.testing.assert_close(loaded.traces, data.traces, rtol=0, atol=0)
            for name in partitions:
                torch.testing.assert_close(split[name], partitions[name], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_observations(data, partitions, path)
            bundle = torch.load(path, weights_only=True)
            for changes, message in (
                ({"reference_velocity": torch.zeros(1)}, "bundle keys"),
                ({"partitions": {**partitions, "validation": partitions["train"]}}, "disjoint"),
            ):
                payload = io.BytesIO()
                torch.save({**bundle, **changes}, payload)
                payload.seek(0)
                with self.assertRaisesRegex(ValueError, message):
                    load_observations(payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_observations(path, expected_content_sha256="0" * 64)

    def test_cli_completes_without_native_data_and_preserves_existing_outputs(self):
        _, data, partitions = problem()
        cfg = replace(configuration(), steps_per_stage=2, validation_interval=1)
        profile = {
            "field": {"background": [2100, 2600], "bounds": [1500, 4500],
                      "backend": "sparse_fused"},
            "inversion": asdict(cfg),
            "regularization": asdict(Regularization()),
            "preprocessing": asdict(Preprocessing()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_observations(data, partitions, root / "observations.pt")
            (root / "profile.json").write_text(json.dumps(profile))
            command = [
                sys.executable, str(Path(run.__file__).resolve()),
                "--observations", str(root / "observations.pt"),
                "--config", str(root / "profile.json"),
                "--output", str(root / "output"),
            ]
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root / "output/result.json").read_text())
            self.assertTrue(result["independent_prediction_exact"])
            self.assertEqual(result["fit_solver_calls"]["adjoint"], 4)
            self.assertEqual(result["verification_solver_calls"], {"forward": 1, "adjoint": 0})
            self.assertFalse(result["reference_velocity_loaded"])
            before = {p.relative_to(root): p.read_bytes()
                      for p in (root / "output").rglob("*") if p.is_file()}
            rejected = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("FileExistsError", rejected.stderr)
            after = {p.relative_to(root): p.read_bytes()
                     for p in (root / "output").rglob("*") if p.is_file()}
            self.assertEqual(before, after)
