"""The prepared-model workflow: array in, valid observation-only bundle out."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

from gaussian_fwi.core.io import load_observations

MODELS = Path(__file__).resolve().parents[2] / "models"
if MODELS.is_dir():
    sys.path.insert(0, str(MODELS))


@unittest.skipUnless(MODELS.is_dir(), "model helpers are absent from an installed wheel")
class ObservationBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, depth=24, width=24):
        # A depth trend with a faster wedge, in the prepared 1500-4500 m/s range.
        velocity = torch.linspace(1500.0, 3500.0, depth, dtype=torch.float64)[:, None]
        velocity = velocity.repeat(1, width).clone()
        velocity[depth // 2:, width // 2:] += 400.0
        return velocity

    def test_bundle_is_observation_only_with_disjoint_partitions(self):
        from make_observations import build

        observations, partitions = build(self.model(), samples=200, shots=2, peak_hz=15.0)
        shots, receivers, samples = observations.traces.shape
        self.assertEqual((shots, samples), (2, 200))
        self.assertEqual(receivers, 22)
        self.assertTrue(torch.isfinite(observations.traces).all())
        self.assertGreater(float(observations.traces.abs().max()), 0.0)

        seen = torch.cat([ids for ids in partitions.values()])
        self.assertEqual(len(seen), len(set(seen.tolist())), "partitions must be disjoint")
        self.assertTrue(set(partitions) == {"train", "validation", "test"})
        for name, ids in partitions.items():
            self.assertGreater(len(ids), 0, name)
            self.assertTrue(bool(((ids >= 0) & (ids < receivers)).all()), name)

    def test_saved_bundle_reloads_with_matching_content_identity(self):
        from make_observations import build

        from gaussian_fwi import save_observations

        observations, partitions = build(self.model(), samples=200, shots=2, peak_hz=15.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.pt"
            save_observations(observations, partitions, path)
            restored, restored_partitions = load_observations(path)
            self.assertEqual(restored.content_identity(), observations.content_identity())
            torch.testing.assert_close(restored.traces, observations.traces, rtol=0, atol=0)
            for name, ids in partitions.items():
                torch.testing.assert_close(restored_partitions[name], ids, rtol=0, atol=0)
            # No reference velocity may travel with the observations.
            payload = torch.load(path, weights_only=True)
            self.assertEqual(set(payload), {"acquisition", "traces", "partitions"})


if __name__ == "__main__":
    unittest.main()
