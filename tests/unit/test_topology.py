"""Mathematical and transactional invariants of the active topology engine."""

import unittest
from copy import deepcopy

import torch
from _problems import assert_exact

from gaussian_fwi import GaussianField
from gaussian_fwi._topology import GaussianTopology
from gaussian_fwi.core import GridSpec


def initialized_field(dimension=2):
    field = GaussianField(GridSpec((11,) * dimension, 10), background=(2400, 2700)).double()
    centers = torch.tensor(
        [[30.0] * dimension, [50.0] * dimension, [70.0] * dimension], dtype=torch.float64
    )
    block = field.add_gaussians(centers, torch.full_like(centers, 15.0))
    with torch.no_grad():
        block.amplitudes.copy_(torch.tensor([20.0, -10.0, 5.0]))
        block.shears.fill_(0.2)
    optimizer = torch.optim.Adam(field.parameter_groups(), amsgrad=True)
    for parameter in field.parameters():
        parameter.grad = torch.linspace(
            0.1, 0.3, parameter.numel(), dtype=parameter.dtype
        ).reshape_as(parameter)
    optimizer.step()
    return field, optimizer, GaussianTopology(field)


class TopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_each_density_edit_matches_its_independent_raw_field_change(self):
        for dimension in (2, 3):
            for operation, count_change in (("split", 1), ("clone", 1), ("prune", -1)):
                with self.subTest(dimension=dimension, operation=operation):
                    field, optimizer, topology = initialized_field(dimension)
                    edit = (topology.split(0, seed=7) if operation == "split"
                            else topology.clone(1) if operation == "clone" else topology.prune(1))
                    before = field.raw().detach()
                    expected = topology.raw_change(edit)
                    topology.apply_many([edit], optimizer, step=5, max_field_change=1000)
                    torch.testing.assert_close(field.raw() - before, expected,
                                               rtol=1e-10, atol=2e-10)
                    self.assertEqual(field.count, 3 + count_change)
                    ids = [i for block in topology.ids for i in block]
                    self.assertEqual(len(ids), len(set(ids)))

    def test_stale_proposal_and_wrong_field_identity_are_rejected(self):
        field, optimizer, topology = initialized_field()
        edit = topology.split(0, seed=7)
        identity = topology.state_dict()
        with torch.no_grad():
            field.blocks[0].amplitudes[0] += 1
        before = field.checkpoint(), deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(ValueError, "parents changed"):
            topology.apply_many([edit], optimizer, step=5, max_field_change=1000)
        with self.assertRaisesRegex(ValueError, "different field"):
            GaussianTopology(field, identity)
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
