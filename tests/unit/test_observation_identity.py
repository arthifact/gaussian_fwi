import unittest

import torch

from fwi_core import Acquisition, GridSpec, Observations
from fwi_core.footprints import FootprintAcquisition, GaussianFootprint


class ObservationIdentityTests(unittest.TestCase):
    def fixture(self):
        wave = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8)/8
        acquisition = Acquisition(GridSpec((25, 25), 5.), .001, wave, torch.tensor([[[12, 12]]]),
                                  torch.tensor([[[12, 10], [12, 12], [12, 14]]]))
        return Observations(acquisition, torch.arange(24, dtype=torch.float64).reshape(1, 3, 8)/24)

    def test_historical_point_and_finite_bundle_hashes_are_preserved(self):
        # Digests captured using the independent, archived pre-fix bundle hasher.
        data = self.fixture()
        self.assertEqual(data.content_identity(), {"format": "gaussian-fwi-observation-content-v1",
                                                  "sha256": "01808a69322af7a6fee3b987df7e9d73375e05826733f9bcfec7f1edaa939507"})
        data.acquisition = FootprintAcquisition(data.acquisition, GaussianFootprint(5., 0.))
        self.assertEqual(data.content_identity()["sha256"], "73941de8296e86bdab18c7066834d228b5a3049eef690f03c8aedfbe03cc3380")

    def test_identity_is_recomputed_and_excludes_provenance_counters_and_layout(self):
        data = self.fixture()
        initial = data.content_identity()
        data.sha256 = "a stale user label"
        data.metadata["description"] = "external provenance"
        data.acquisition.forward_calls = 100
        data.acquisition.adjoint_calls = 50
        data.traces = data.traces.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertEqual(data.content_identity(), initial)
        self.assertEqual(data.acquisition.counts, {"forward": 100, "adjoint": 50})
        point = data.traces[0, 0, 0]
        point.copy_(torch.nextafter(point, torch.full_like(point, float("inf"))))
        self.assertNotEqual(data.content_identity(), initial)
        point.zero_()
        self.assertEqual(data.content_identity(), initial)
        data.traces = data.traces.float()
        data.acquisition.source_amplitudes = data.acquisition.source_amplitudes.float()
        self.assertNotEqual(data.content_identity(), initial)
