"""High-level run API: profile integrity, declared deviations and a verified fit."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from _problems import problem

import gaussian_fwi as fwi
from gaussian_fwi.core.io import save_observations

CONFIGS = Path(__file__).resolve().parents[2] / "configs/baseline.json"


def small_profile():
    """The accepted profile, shortened to a few updates for an integration check."""
    return fwi.baseline(
        field={"background": [2100.0, 2600.0]},
        cutoffs=[10.0, 20.0], seed_shape=[2, 2], steps_per_stage=8, validation_interval=2,
        sampling_refinement_factors=[],
        refinement={"warmup_steps": 0, "interval": 1, "stop_fraction": 0.75, "minimum_age": 1,
                    "max_gaussians": 64, "max_growth": 2, "max_prunes": 2,
                    "split_extent_fraction": 1.0, "prune_amplitude": 0.0},
    )


class ProfileTests(unittest.TestCase):
    @unittest.skipUnless(CONFIGS.is_file(), "canonical profile is absent from an installed wheel")
    def test_packaged_baseline_matches_the_canonical_profile(self):
        self.assertEqual(fwi.BASELINE, json.loads(CONFIGS.read_text()))

    def test_baseline_returns_an_independent_copy(self):
        first = fwi.baseline()
        first["inversion"]["steps_per_stage"] = 1
        self.assertEqual(fwi.baseline()["inversion"]["steps_per_stage"], 1000)
        self.assertEqual(fwi.BASELINE["inversion"]["steps_per_stage"], 1000)

    def test_declared_deviation_changes_only_the_named_settings(self):
        profile = fwi.baseline(steps_per_stage=400, refinement={"seed": 7})
        self.assertEqual(profile["inversion"]["steps_per_stage"], 400)
        self.assertEqual(profile["inversion"]["refinement"]["seed"], 7)
        self.assertEqual(profile["inversion"]["cutoffs"], fwi.BASELINE["inversion"]["cutoffs"])
        self.assertEqual(profile["regularization"], fwi.BASELINE["regularization"])

    def test_unknown_or_invalid_settings_are_rejected_before_any_fit(self):
        for call in (lambda: fwi.baseline(step_per_stage=200),
                     lambda: fwi.baseline(refinement={"warmup": 1}),
                     lambda: fwi.baseline(regularization={"tv": 1.0})):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(ValueError):
            fwi.baseline(cutoffs=[20.0, 4.0])
        # A horizon too short for warm-up, a refinement opportunity and settling
        # fails here rather than part way through the fit it would have started.
        with self.assertRaises(ValueError):
            fwi.baseline(steps_per_stage=200)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_run_verifies_its_own_result_and_reads_back_from_disk(self):
        _, data, partitions = problem()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "observations.pt"
            save_observations(data, partitions, bundle)
            fit = fwi.run(bundle, root / "fit01", profile=small_profile(), threads=2)

            self.assertEqual(fit.status, "complete")
            self.assertTrue(fit.verified, "independent propagation must reproduce the saved field")
            self.assertGreater(fit.gaussians, 0)
            self.assertEqual(fit.velocity.shape, data.acquisition.grid.shape)
            self.assertTrue(torch.isfinite(fit.velocity).all())
            self.assertGreater(fit.solver_calls["forward"], 0)
            self.assertIn("train", fit.waveform_losses)
            self.assertIn("Gaussian FWI run", fit.summary())

            # Provenance is recorded, and the profile is stored exactly as executed.
            self.assertEqual(fit.config, small_profile())
            self.assertFalse(fit.provenance["observation_content_identity"] == {})
            self.assertIn("gaussian_fwi/inversion.py", fit.provenance["source_sha256"])

            reopened = fwi.Run.open(root / "fit01")
            self.assertEqual(reopened.gaussians, fit.gaussians)
            torch.testing.assert_close(reopened.velocity, fit.velocity, rtol=0, atol=0)

    def test_opening_an_unfinished_directory_reports_the_missing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                fwi.Run.open(temporary)


if __name__ == "__main__":
    unittest.main()
