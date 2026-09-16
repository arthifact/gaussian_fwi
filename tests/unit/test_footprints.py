import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from _problems import assert_exact
from _support import refined_acquisition, verify_waveform_sampling

from gaussian_fwi import GaussianField
from gaussian_fwi.core import Acquisition, GridSpec
from gaussian_fwi.core.footprints import FootprintAcquisition, GaussianFootprint, footprint_weights
from gaussian_fwi.core.identity import observation_sha256
from gaussian_fwi.core.io import load_observations


class FootprintTests(unittest.TestCase):
    def test_shot_batches_preserve_unequal_geometry_gradients_identity_and_work(self):
        torch.set_num_threads(2)
        grid = GridSpec((25, 25), 5.)
        time = torch.arange(100, dtype=torch.float64) * .0005
        pulse = torch.exp(-((time-.02)/.006)**2)[None, None]
        wave = pulse * torch.tensor([.7, 1.1, 1.6], dtype=torch.float64)[:, None, None]
        base = Acquisition(grid, .0005, wave,
                           torch.tensor([[[12, 9]], [[12, 12]], [[12, 15]]]),
                           torch.tensor([[[12, 9], [12, 15]], [[12, 9], [12, 15]],
                                         [[12, 12], [12, 12]]]), pml_width=8)
        outputs, gradients = [], []
        for size in (1, 2, 3):
            acq = FootprintAcquisition(base, GaussianFootprint(5., 5.), shot_batch_size=size)
            velocity = (2300 + 30*torch.sin(torch.arange(625, dtype=torch.float64)/40)).reshape(25, 25)
            velocity.requires_grad_()
            prediction = acq.simulate(velocity)
            weights = torch.tensor([1., 2., 4.], dtype=torch.float64)[:, None, None]
            gradient, = torch.autograd.grad((weights*prediction.square()).mean(), velocity)
            outputs.append(prediction.detach())
            gradients.append(gradient)
            self.assertEqual(acq.counts, {"forward": 3, "adjoint": 3})
            self.assertEqual(acq.batch_counts, {"forward": 3 if size == 1 else 2,
                                              "adjoint": 3 if size == 1 else 2})
            self.assertEqual(sum(s.counts["forward"] for s in acq.shots), 3)
            restored = FootprintAcquisition.from_checkpoint(acq.checkpoint())
            self.assertEqual(restored.shot_batch_size, size)
            self.assertEqual(observation_sha256(acq, outputs[0]),
                             observation_sha256(restored, outputs[0]))
        for output, gradient in zip(outputs[1:], gradients[1:], strict=True):
            torch.testing.assert_close(output, outputs[0], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(gradient, gradients[0], rtol=1e-8, atol=1e-12)

    def owned_fixture(self):
        torch.set_num_threads(2)
        grid = GridSpec((25, 25), 5.)
        time = torch.arange(100, dtype=torch.float64)*.0005
        wave = torch.exp(-((time-.02)/.006)**2)[None, None]
        base = Acquisition(grid, .0005, wave, torch.tensor([[[12, 10]]]),
                           torch.tensor([[[12, 9], [12, 15]]]), pml_width=8)
        return base, FootprintAcquisition(base, GaussianFootprint(5., 0.))

    def test_template_and_checkpoint_tensors_are_owned(self):
        base, acq = self.owned_fixture()
        velocity = torch.full(base.grid.shape, 2300., dtype=torch.float64)
        with torch.no_grad():
            before = acq.simulate(velocity)
            original = acq.checkpoint()
            digest = observation_sha256(acq, before)
            base.source_amplitudes.mul_(2)
            base.source_locations.add_(1)
            base.receiver_locations.add_(1)
            base.dt *= .5
            base.pml_width += 2
            assert_exact(acq.checkpoint(), original)
            self.assertEqual(observation_sha256(acq, before), digest)
            assert_exact(acq.simulate(velocity), before)
            payload = acq.checkpoint()
            restored = FootprintAcquisition.from_checkpoint(payload)
            payload["base"]["source_amplitudes"].mul_(3)
            payload["base"]["source_locations"].add_(1)
            payload["base"]["receiver_locations"].add_(1)
            payload["base"]["dt"] *= .5
            payload["footprint"]["source_sigma_m"] = 7.
            assert_exact(acq.checkpoint(), original)
            assert_exact(restored.checkpoint(), original)
            assert_exact(restored.simulate(velocity), before)
        self.assertEqual(acq.counts, {"forward": 2, "adjoint": 0})
        self.assertEqual(restored.counts, {"forward": 1, "adjoint": 0})
        self.assertEqual(base.counts, {"forward": 0, "adjoint": 0})

    def test_public_inspection_cannot_change_live_operators(self):
        _, acq = self.owned_fixture()
        velocity = torch.full(acq.grid.shape, 2300., dtype=torch.float64)
        with torch.no_grad():
            before = acq.simulate(velocity)
            original = acq.checkpoint()
            acq.base.source_amplitudes.mul_(2)
            acq.source_amplitudes.mul_(2)
            acq.source_locations.add_(1)
            acq.receiver_locations.add_(1)
            acq.shots[0].source_amplitudes.mul_(2)
            acq.shots[0].dt *= .5
            acq.source_operators[0].values().zero_()
            acq.receiver_operators[0].values().zero_()
            acq.measurement_metadata["source_sigma_m"] = 7.
            for name in ("dt", "grid", "source_amplitudes", "base", "footprint", "shots"):
                with self.subTest(property=name), self.assertRaises(AttributeError):
                    setattr(acq, name, getattr(acq, name))
            assert_exact(acq.checkpoint(), original)
            assert_exact(acq.simulate(velocity), before)
            self.assertEqual(acq.shots[0].counts, acq.counts)

    def test_physical_mass_centroid_and_adjoint(self):
        for ndim in (2, 3):
            grid = GridSpec((25,)*ndim, 2.)
            centers = torch.tensor([[12]*ndim, [13]*ndim])
            nodes, sparse = footprint_weights(grid, centers, 3., dtype=torch.float64)
            weights = sparse.to_dense().numpy()
            # Independent continuous kernel evaluated at the returned nodes.
            q = np.sum(((nodes.numpy()[None]-centers.numpy()[:, None])*2/3)**2, axis=-1)
            t = np.clip((q-9)/7, 0, 1)
            reference = np.exp(-q/2)*(1-10*t**3+15*t**4-6*t**5)
            reference[q >= 16] = 0
            reference /= reference.sum(1, keepdims=True)
            np.testing.assert_allclose(weights, reference, rtol=1e-11, atol=1e-16)
            np.testing.assert_allclose(weights.sum(1), 1, atol=2e-15)
            np.testing.assert_allclose(weights@nodes.numpy()*2, centers.numpy()*2, atol=1e-12)
            values = np.sin(np.arange(len(nodes)))
            logical = np.array([.3, -.7])
            self.assertAlmostEqual(float(logical@(weights@values)), float((weights.T@logical)@values), places=14)

    def test_clipping_is_not_silent(self):
        with self.assertRaisesRegex(ValueError, "buffer"):
            footprint_weights(GridSpec((20, 20), 2), torch.tensor([[0, 10]]), 3., dtype=torch.float64)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GaussianFootprint(value)

    def test_acoustic_gradient_identity_and_checkpoint(self):
        torch.set_num_threads(2)
        grid = GridSpec((25, 25), 5.)
        time = torch.arange(100, dtype=torch.float64)*.0005
        wave = torch.exp(-((time-.02)/.006)**2).reshape(1, 1, -1).repeat(2, 1, 1)
        base = Acquisition(grid, .0005, wave, torch.tensor([[[12, 9]], [[12, 15]]]),
                           torch.tensor([[[12, 9], [12, 15]], [[12, 9], [12, 15]]]), pml_width=8)
        acq = FootprintAcquisition(base, GaussianFootprint(5., 5.))
        velocity = torch.full(grid.shape, 2300., dtype=torch.float64, requires_grad=True)
        result = acq.simulate(velocity)
        loss = result.square().sum()
        gradient, = torch.autograd.grad(loss, velocity)
        direction = torch.sin(torch.arange(625, dtype=torch.float64).reshape(grid.shape)/40)
        with torch.no_grad():
            plus = acq.simulate(velocity+.05*direction).square().sum()
            minus = acq.simulate(velocity-.05*direction).square().sum()
        torch.testing.assert_close((gradient*direction).sum(), (plus-minus)/.1, rtol=1e-6, atol=1e-12)
        self.assertEqual(acq.counts, {"forward": 6, "adjoint": 2})
        restored = FootprintAcquisition.from_checkpoint(acq.checkpoint())
        with torch.no_grad():
            torch.testing.assert_close(restored.simulate(velocity), result, rtol=0, atol=0)
        digest = observation_sha256(acq, result.detach())
        self.assertNotEqual(digest, observation_sha256(base, result.detach()))
        data = io.BytesIO()
        torch.save({"acquisition": acq.checkpoint(), "traces": result.detach(),
                    "partitions": {"train": torch.tensor([0]), "validation": torch.tensor([1])}}, data)
        observations, _ = load_observations(io.BytesIO(data.getvalue()), expected_content_sha256=digest)
        self.assertEqual(observation_sha256(observations.acquisition, observations.traces), digest)
        for source, shot in zip(acq.source_operators, acq.shots):
            torch.testing.assert_close(source.to_dense().sum(1), torch.ones(1, dtype=torch.float64))
            torch.testing.assert_close(shot.source_amplitudes.sum(1), base.source_amplitudes[0])

    def test_refinement_keeps_physical_footprint_and_counts_actual_shots(self):
        grid = GridSpec((25, 25), 5.)
        wave = torch.sin(torch.arange(100, dtype=torch.float64)*.1)[None, None].repeat(2, 1, 1)
        base = Acquisition(grid, .0005, wave, torch.tensor([[[12, 9]], [[12, 15]]]),
                           torch.tensor([[[12, 9], [12, 15]], [[12, 9], [12, 15]]]), pml_width=8)
        acq = FootprintAcquisition(base, GaussianFootprint(5., 5.))
        fine = refined_acquisition(acq, 2)
        self.assertIsInstance(fine, FootprintAcquisition)
        self.assertEqual(fine.measurement_metadata, acq.measurement_metadata)
        torch.testing.assert_close(fine.source_locations*fine.grid.spacing, acq.source_locations*acq.grid.spacing)
        for shot, original in zip(fine.shots, acq.shots):
            torch.testing.assert_close(shot.source_amplitudes.sum(1)*fine.grid.spacing**2,
                                       original.source_amplitudes.sum(1)*acq.grid.spacing**2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"field.pt"
            GaussianField(grid, background=(2100., 2300.)).double().save(path)
            report = verify_waveform_sampling(path, acq)
            self.assertEqual(report["solver_calls"], {"forward": 6, "adjoint": 0})
            self.assertEqual(acq.counts, {"forward": 0, "adjoint": 0})
