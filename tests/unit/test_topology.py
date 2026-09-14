"""Mathematical and transactional invariants of the active topology engine."""

import unittest
from copy import deepcopy

import torch
from test_dynamic import assert_exact

from dynamic_refinement import GaussianField
from fwi_core import GridSpec
from gaussian_fwi.topology import EditRejected, GaussianTopology


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

    def test_binary_split_preserves_signed_mass_centroid_and_covariance(self):
        for dimension in (2, 3):
            field, _, topology = initialized_field(dimension)
            for parent in (0, 1):
                for axis in range(dimension):
                    with self.subTest(dimension=dimension, parent=parent, axis=axis):
                        center, covariance, amplitude = topology.values(parent)
                        edit = topology.split(parent, axis=axis, fraction=0.4)
                        # Common (2*pi)**(d/2) cancels for ideal untruncated Gaussians.
                        mass = amplitude * torch.linalg.det(covariance).sqrt()
                        child_mass = edit.amplitudes * torch.linalg.det(edit.covariances).sqrt()
                        torch.testing.assert_close(child_mass.sum(), mass, rtol=2e-12, atol=1e-9)
                        weights = child_mass / mass
                        centroid = (weights[:, None] * edit.centers).sum(0)
                        offset = edit.centers - center
                        mixture = (
                            weights[:, None, None]
                            * (edit.covariances + offset[:, :, None] * offset[:, None, :])
                        ).sum(0)
                        torch.testing.assert_close(centroid, center, rtol=2e-12, atol=1e-11)
                        torch.testing.assert_close(mixture, covariance, rtol=2e-12, atol=1e-10)
                        self.assertTrue((torch.linalg.eigvalsh(edit.covariances) > 0).all())

    def test_each_direct_edit_matches_its_predicted_raw_field_change(self):
        count_changes = {"insert": 1, "split": 1, "clone": 1, "merge": -1, "prune": -1, "reset": 0}
        for dimension in (2, 3):
            for operation, count_change in count_changes.items():
                with self.subTest(dimension=dimension, operation=operation):
                    field, optimizer, topology = initialized_field(dimension)
                    edits = {
                        "insert": lambda: topology.insert(
                            field.background.new_full((dimension,), 40), 12
                        ),
                        "split": lambda: topology.split(0),
                        "clone": lambda: topology.clone(1),
                        "merge": lambda: topology.merge(0, 2),
                        "prune": lambda: topology.prune(1),
                        "reset": lambda: topology.reset(0, amplitude_cap=2),
                    }
                    edit = edits[operation]()
                    before = field.raw().detach()
                    predicted = topology.raw_change(edit)
                    event = topology.apply(
                        edit,
                        optimizer,
                        step=5,
                        max_field_change=1000,
                        learning_rates={"amplitude_lr": 4.0},
                    )
                    torch.testing.assert_close(
                        field.raw() - before, predicted, rtol=1e-10, atol=2e-10
                    )
                    self.assertEqual(field.count, 3 + count_change)
                    self.assertEqual(event["after_count"], field.count)
                    ids = [i for block in topology.ids for i in block]
                    self.assertEqual(len(ids), len(set(ids)))
                    if operation == "insert":
                        torch.testing.assert_close(field.raw(), before, rtol=0, atol=0)
                        self.assertFalse(field.blocks[-1].amplitudes.any())

    def test_split_keeps_survivor_adam_history_and_gives_children_fresh_state(self):
        field, optimizer, topology = initialized_field()
        block = field.blocks[0]
        old_parameters = dict(block.named_parameters())
        states = {name: deepcopy(optimizer.state[p]) for name, p in old_parameters.items()}
        gradients = {name: p.grad.clone() for name, p in old_parameters.items()}
        background = field.background
        background_state = deepcopy(optimizer.state[background])
        topology.apply(topology.split(0), optimizer, step=5, max_field_change=1000)
        self.assertIs(field.background, background)
        assert_exact(optimizer.state[background], background_state)
        self.assertEqual(topology.ids, [[1, 2], [3, 4]])
        self.assertEqual(topology.last_edit_steps, [[0, 0], [5, 5]])
        for name, parameter in field.blocks[0].named_parameters():
            assert_exact(parameter.detach(), old_parameters[name].detach()[1:])
            assert_exact(parameter.grad, gradients[name][1:])
            expected = {
                key: value if key == "step" else value[1:] for key, value in states[name].items()
            }
            assert_exact(optimizer.state[parameter], expected)
            self.assertNotIn(old_parameters[name], optimizer.state)
        for parameter in field.blocks[1].parameters():
            self.assertFalse(optimizer.state.get(parameter))
        self.assertTrue(all(group["amsgrad"] for group in optimizer.param_groups))

    def test_field_limit_rejection_restores_parameters_gradients_optimizer_and_ids(self):
        field, optimizer, topology = initialized_field()
        parameters = tuple(field.parameters())
        gradients = [p.grad.clone() for p in parameters]
        before = field.checkpoint(), deepcopy(optimizer.state_dict()), topology.state_dict()
        with self.assertRaisesRegex(EditRejected, "trust region"):
            topology.apply(topology.clone(0), optimizer, step=5, max_field_change=1e-12)
        self.assertTrue(all(a is b for a, b in zip(parameters, field.parameters(), strict=True)))
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
        assert_exact(topology.state_dict(), before[2])
        for parameter, gradient in zip(parameters, gradients, strict=True):
            assert_exact(parameter.grad, gradient)

    def test_multiple_edits_share_the_cumulative_field_limit(self):
        field, optimizer, topology = initialized_field()
        reference = field().detach()
        edit = topology.clone(0)
        proposed = field.bound_velocity(field.raw() + topology.raw_change(edit))
        limit = 1.5 * float((proposed - reference.flatten()).abs().max().detach())
        topology.apply(
            edit, optimizer, step=5, max_field_change=limit, reference_velocity=reference
        )
        before = field.checkpoint(), deepcopy(optimizer.state_dict()), topology.state_dict()
        with self.assertRaisesRegex(EditRejected, "trust region"):
            topology.apply(
                topology.clone(0),
                optimizer,
                step=6,
                max_field_change=limit,
                reference_velocity=reference,
            )
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
        assert_exact(topology.state_dict(), before[2])

    def test_stale_proposal_and_wrong_field_identity_are_rejected(self):
        field, optimizer, topology = initialized_field()
        edit = topology.split(0)
        identity = topology.state_dict()
        with torch.no_grad():
            field.blocks[0].amplitudes[0] += 1
        before = field.checkpoint(), deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(ValueError, "parents changed"):
            topology.apply(edit, optimizer, step=5, max_field_change=1000)
        with self.assertRaisesRegex(ValueError, "different field"):
            GaussianTopology(field, identity)
        assert_exact(field.checkpoint(), before[0])
        assert_exact(optimizer.state_dict(), before[1])
