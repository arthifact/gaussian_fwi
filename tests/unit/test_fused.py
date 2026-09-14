"""Independent field oracle and state/physics contracts for fused sparse decoding."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from test_dynamic import assert_exact, configuration, problem
from test_topology import initialized_field

import dynamic_refinement as fwi
from gaussian_fwi import GaussianField, GridSpec
from gaussian_fwi.topology import EditRejected


def numpy_values(field, points):
    """Independent covariance solve, scalar taper and stable soft bounds."""
    p = points.detach().numpy()
    b = field.background.detach().numpy()
    value = b[0] + (b[1]-b[0])*p[:, -1]/field.grid.extent[-1]
    for block in field.blocks:
        for center, logs, shear, amplitude in zip(
            block.centers.detach().numpy(), block.log_scales.detach().numpy(),
            block.shears.detach().numpy(), block.amplitudes.detach().numpy(), strict=True
        ):
            lower = np.eye(field.grid.ndim)
            lower[np.tril_indices(field.grid.ndim, -1)] = shear
            lower = np.diag(np.exp(logs)) @ lower
            covariance = lower @ lower.T
            delta = p-center
            q = np.sum(delta*np.linalg.solve(covariance, delta.T).T, axis=1)
            t = np.clip((q-25)/11, 0, 1)
            value += amplitude*np.exp(-q/2)*(1-10*t**3+15*t**4-6*t**5)
    lo, hi = field.bounds
    s = field.soft_clip
    return value+s*np.logaddexp(0, (lo-value)/s)-s*np.logaddexp(0, (value-hi)/s)


class FusedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_multiblock_values_and_all_family_derivatives_match_independent_oracle(self):
        rng = np.random.default_rng(291)
        for d in (2, 3):
            field = GaussianField(GridSpec((6,)*d, 10), background=(2300, 2700),
                                  backend="sparse_fused", max_pairs=17).double()
            for count in (3, 1, 4, 2):
                block = field.add_gaussians(torch.tensor(rng.uniform(0, 50, (count, d))),
                                            torch.tensor(rng.uniform(5, 14, (count, d))))
                with torch.no_grad():
                    block.shears.copy_(torch.tensor(rng.uniform(-.6, .6, block.shears.shape)))
                    block.amplitudes.copy_(torch.tensor(rng.uniform(-150, 150, count)))
            points = torch.tensor(rng.uniform(-10, 60, (37, d)), requires_grad=True)
            weights = torch.tensor(rng.normal(size=37))
            torch.testing.assert_close(field(points), torch.tensor(numpy_values(field, points)),
                                       rtol=0, atol=3e-12)
            parameters = tuple(field.parameters())
            before_ids = tuple(id(p) for p in parameters)
            gradients = torch.autograd.grad(field(points) @ weights, (*parameters, points))
            families = [(0,)] + [tuple(i for i, name in enumerate(dict(field.named_parameters()))
                                      if name.endswith(suffix)) for suffix in
                                     ("centers", "log_scales", "shears", "amplitudes")]
            h = 1e-4
            for family in families:
                directions = {i: torch.tensor(rng.normal(size=parameters[i].shape))*.1
                              for i in family}
                originals = {i: parameters[i].detach().clone() for i in family}
                endpoints = []
                for sign in (1, -1):
                    with torch.no_grad():
                        for i in family:
                            parameters[i].copy_(originals[i]+sign*h*directions[i])
                    endpoints.append(float(numpy_values(field, points) @ weights.numpy()))
                with torch.no_grad():
                    for i in family:
                        parameters[i].copy_(originals[i])
                derivative = sum(float((gradients[i]*directions[i]).sum()) for i in family)
                self.assertAlmostEqual(derivative, (endpoints[0]-endpoints[1])/(2*h), delta=3e-6)
            direction = torch.tensor(rng.normal(size=points.shape))*.1
            plus = numpy_values(field, points.detach()+h*direction) @ weights.numpy()
            minus = numpy_values(field, points.detach()-h*direction) @ weights.numpy()
            self.assertAlmostEqual(float((gradients[-1]*direction).sum()),
                                   float((plus-minus)/(2*h)), delta=3e-6)
            self.assertEqual(before_ids, tuple(id(p) for p in field.parameters()))
            with torch.no_grad():
                points.add_(.3)
                field.blocks[-1].amplitudes.add_(3)
            torch.testing.assert_close(field(points), torch.tensor(numpy_values(field, points)),
                                       rtol=0, atol=3e-12)

    def test_birth_survivor_states_and_atomic_rejection(self):
        for d in (2, 3):
            field, optimizer, topology = initialized_field(d)
            field.backend = "sparse_fused"
            parameters = tuple(field.parameters())
            states = {p: deepcopy(optimizer.state[p]) for p in parameters}
            before = field().detach()
            topology.apply(topology.insert(field.background.new_full((d,), 40), 12), optimizer,
                           step=5, max_field_change=1000, learning_rates={"amplitude_lr": 4.0})
            torch.testing.assert_close(field(), before, rtol=0, atol=0)
            self.assertTrue(all(a is b for a, b in zip(parameters, field.parameters())))
            for p in parameters:
                assert_exact(optimizer.state[p], states[p])
            for p in field.blocks[-1].parameters():
                self.assertFalse(optimizer.state.get(p))
            before = field.checkpoint(), deepcopy(optimizer.state_dict()), topology.state_dict()
            with self.assertRaises(EditRejected):
                topology.apply(topology.clone(0), optimizer, step=6, max_field_change=1e-12)
            assert_exact((field.checkpoint(), optimizer.state_dict(), topology.state_dict()), before)

    def test_versioned_export_and_empty_field_queries(self):
        for sampling in (None, {"max_nyquist_response": 1e-3}):
            field = GaussianField(GridSpec((8, 9), 10), backend="sparse_fused", sampling=sampling)
            self.assertEqual(field(torch.empty(0, 2)).shape, (0,))
            field.add_grid_level((3, 4))
            field.add_grid_level((2, 2))
            with torch.no_grad():
                field.blocks[0].amplitudes.fill_(23)
                field.blocks[1].amplitudes.fill_(-17)
            snapshot = field.checkpoint()
            self.assertEqual(snapshot["format"], "gaussian-fwi-field-v3")
            restored = GaussianField.from_checkpoint(snapshot)
            assert_exact(restored.checkpoint(), snapshot)
            torch.testing.assert_close(restored(), field(), rtol=0, atol=0)
            snapshot["format"] = "gaussian-fwi-field-v2"
            with self.assertRaisesRegex(ValueError, "versioned"):
                GaussianField.from_checkpoint(snapshot)

    def test_real_2d_3d_fits_and_restart_preserve_fused_fields_and_work(self):
        for d in (2, 3):
            field, data, partitions = problem(d)
            field.backend = "sparse_fused"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = fwi.invert(field, data, configuration(d), root/"fit", partitions=partitions)
                restored, continued = fwi.resume(root/"fit/stage_00.pt", data, root/"resume")
                self.assertEqual(report["solver_calls"], continued["solver_calls"])
                self.assertTrue(report["topology_history"])
                assert_exact(restored.checkpoint(), field.checkpoint())
                torch.testing.assert_close(restored(), field(), rtol=0, atol=0)
                saved = GaussianField.load(root/"fit/field.pt")
                torch.testing.assert_close(saved(), field(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
