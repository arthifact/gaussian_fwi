"""Independent witnesses for coverage-aware Gaussian insertion screening."""

import unittest
from unittest.mock import patch

import torch

from dynamic_refinement import GaussianField, RefinementConfig
from fwi_core import GridSpec
from gaussian_fwi import adaptation
from gaussian_fwi.adaptation import AdaptationConfig, insertion_candidates
from gaussian_fwi.refinement import RefinementController


def covered_peaks(dimension):
    field = GaussianField(GridSpec((31,) * dimension, 1)).double()
    centers = torch.tensor(
        [[x, z] if dimension == 2 else [x, 15, z] for z in (5, 15, 25) for x in (5, 15, 25)],
        dtype=torch.float64,
    )
    field.add_gaussians(centers[:8], torch.ones_like(centers[:8]))
    gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
    for i, center in enumerate(centers):
        gradient[tuple(reversed(center.long().tolist()))] = 100 - 10 * i
    return field, gradient, centers[-1]


def settings(**kwargs):
    return AdaptationConfig(
        **{
            "insertions_per_event": 2,
            "splits_per_event": 0,
            "candidate_pool": 2,
            "insertion_scale_factors": (1.0,),
            "separation_ratio": 0.5,
            **kwargs,
        }
    )


class InsertionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_backfills_beyond_both_covered_budgets_in_2d_and_3d(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                field, gradient, witness = covered_peaks(dimension)
                before = field().detach().clone()
                centers, radii, scores = insertion_candidates(
                    field, gradient, 1, settings(), exclude_covered=True
                )
                torch.testing.assert_close(centers, witness[None], rtol=0, atol=0)
                torch.testing.assert_close(radii, torch.ones_like(radii), rtol=0, atol=0)
                # Independent closed-form oracle for the known uncovered peak.
                d2 = (field.grid.points(dtype=torch.float64) - witness).square().sum(-1)
                t = ((d2 - 25) / 11).clamp(0, 1)
                basis = torch.exp(-d2 / 2) * (1 - t).pow(3) * (1 + 3 * t + 6 * t.square())
                expected = abs(float(gradient.flatten() @ basis)) / float(basis.norm())
                self.assertAlmostEqual(scores[0], expected, places=12)
                torch.testing.assert_close(field(), before, rtol=0, atol=0)

    def test_legacy_shortlist_and_controller_are_preserved(self):
        for dimension in (2, 3):
            field, gradient, witness = covered_peaks(dimension)
            centers, _, _ = insertion_candidates(field, gradient, 1, settings())
            self.assertEqual(len(centers), 2)
            self.assertFalse(bool((centers == witness).all(-1).any()))
            controller = RefinementController(
                field,
                RefinementConfig.legacy(
                    minimum_age=1, max_edits=2, candidate_pool=2, insertion_scales=(1.0,)
                ).engine_config(),
            )
            controller.observe(field, gradient, 1)
            self.assertFalse(
                [
                    e
                    for _, e in controller._proposals(1, {"amplitude_lr": 4})
                    if e.operation == "insert"
                ]
            )

    def test_controller_proposes_zero_amplitude_witness_without_wave_solves(self):
        for dimension in (2, 3):
            field, gradient, witness = covered_peaks(dimension)
            controller = RefinementController(
                field,
                RefinementConfig(
                    minimum_age=1,
                    max_edits=2,
                    candidate_pool=2,
                    insertion_scales=(1.0,),
                ).engine_config(),
            )
            with patch("fwi_core.Acquisition.simulate", side_effect=AssertionError("Wave solve")):
                controller.observe(field, gradient, 1)
                proposals = controller._proposals(1, {"amplitude_lr": 4})
            edits = [e for _, e in proposals if e.operation == "insert"]
            self.assertEqual(len(edits), 1)
            torch.testing.assert_close(edits[0].centers, witness[None], rtol=0, atol=0)
            self.assertFalse(bool(edits[0].amplitudes.any()))

    def test_coverage_checks_all_neighbors_and_largest_principal_radius(self):
        field = GaussianField(GridSpec((31, 31), 1), sigma_min=0.01).double()
        field.add_gaussians(torch.tensor([[14.9, 15.0]]), torch.tensor([[0.05, 0.05]]))
        gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
        gradient[15, 15] = 1
        self.assertEqual(
            len(insertion_candidates(field, gradient, 1, settings(), exclude_covered=True)[0]), 1
        )
        # A farther, anisotropic neighbor covers the candidate under the declared
        # max-radius rule, even though the nearest small kernel does not.
        block = field.add_gaussians(
            torch.tensor([[14.497, 15.0]], dtype=torch.float64),
            torch.tensor([[0.1, 0.1]], dtype=torch.float64),
        )
        with torch.no_grad():
            block.shears.fill_(10.0)
        # Distance .503 lies between .5*sqrt(max(diag(cov))) ~= .50249
        # and .5*sqrt(max(eigenvalues(cov))) ~= .50495. The candidate's
        # radius 2 does not cap either threshold; this distinguishes the rule.
        self.assertEqual(
            len(insertion_candidates(field, gradient, 2, settings(), exclude_covered=True)[0]), 0
        )

    def test_existing_separation_boundary_is_strict(self):
        for distance, expected_count in ((0.5, 1), (0.49, 0)):
            field = GaussianField(GridSpec((31, 31), 1)).double()
            field.add_gaussians(
                torch.tensor([[15 - distance, 15.0]], dtype=torch.float64),
                torch.tensor([[2.0, 2.0]], dtype=torch.float64),
            )
            gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
            gradient[15, 15] = 1
            centers, _, _ = insertion_candidates(
                field, gradient, 1, settings(), exclude_covered=True
            )
            self.assertEqual(len(centers), expected_count)

    def test_mutual_separation_keeps_a_distant_candidate(self):
        field = GaussianField(GridSpec((41, 41), 1)).double()
        gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
        for x, value in ((10, 100), (14, 90), (30, 80)):
            gradient[20, x] = value
        centers, _, _ = insertion_candidates(
            field,
            gradient,
            1,
            settings(insertions_per_event=3, candidate_pool=3, separation_ratio=5),
            exclude_covered=True,
        )
        torch.testing.assert_close(
            centers, torch.tensor([[10, 20], [30, 20]], dtype=torch.float64), rtol=0, atol=0
        )

    def test_exact_ties_use_grid_index_then_radius(self):
        field = GaussianField(GridSpec((41, 41), 1), sigma_min=0.01).double()
        gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
        gradient[20, 10] = gradient[20, 30] = 1
        centers, radii, _ = insertion_candidates(
            field,
            gradient,
            1,
            settings(insertions_per_event=1, insertion_scale_factors=(0.1, 0.05)),
            exclude_covered=True,
        )
        torch.testing.assert_close(centers, torch.tensor([[10, 20]]).double(), rtol=0, atol=0)
        torch.testing.assert_close(radii, torch.full_like(radii, 0.05), rtol=0, atol=0)

    def test_multiple_clamped_radii_have_a_bounded_exact_scoring_budget(self):
        field = GaussianField(GridSpec((41, 41), 1), sigma_min=0.5, sigma_max=2).double()
        gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
        gradient[10, 10], gradient[30, 30], gradient[10, 30] = 3, 2, 1
        with patch.object(adaptation, "_basis", wraps=adaptation._basis) as basis:
            centers, _, _ = insertion_candidates(
                field,
                gradient,
                1,
                settings(insertion_scale_factors=(0.1, 0.5, 1, 2, 10)),
                exclude_covered=True,
            )
        self.assertEqual(len(centers), 2)
        self.assertEqual(basis.call_count, 6)  # Two candidates at each of .5, 1, 2 m.
        self.assertTrue(all(call.args[1].shape == (2,) for call in basis.call_args_list))

    def test_zero_gradient_and_disabled_insertion_do_no_exact_scoring(self):
        field = GaussianField(GridSpec((8, 8), 1)).double()
        with patch.object(adaptation, "_basis", side_effect=AssertionError("Unneeded basis")):
            for gradient, config in (
                (torch.zeros(8, 8), settings()),
                (torch.ones(8, 8), settings(insertions_per_event=0, splits_per_event=1)),
            ):
                centers, radii, scores = insertion_candidates(
                    field, gradient, 1, config, exclude_covered=True
                )
                self.assertEqual(centers.shape, (0, 2))
                self.assertEqual(radii.shape, (0, 2))
                self.assertEqual(scores, [])

    def test_zero_exact_scores_still_spend_the_candidate_budget(self):
        field = GaussianField(GridSpec((41, 41), 1)).double()
        gradient = torch.zeros(field.grid.shape, dtype=torch.float64)
        gradient[10, 10], gradient[30, 30], gradient[10, 30] = 3, 2, 1
        with patch.object(
            adaptation, "_basis", return_value=torch.zeros(41 * 41, dtype=torch.float64)
        ) as basis:
            centers, _, _ = insertion_candidates(
                field, gradient, 1, settings(), exclude_covered=True
            )
        self.assertEqual(basis.call_count, 2)
        self.assertEqual(len(centers), 0)

    def test_operation_subset_and_fixed_count_relocation(self):
        field, gradient, _ = covered_peaks(2)
        from gaussian_fwi.refinement import RefinementConfig as EngineConfig

        for operations in (("insert",), ("relocate",), ()):
            with self.subTest(operations=operations):
                model = GaussianField.from_checkpoint(field.checkpoint())
                controller = RefinementController(
                    model,
                    EngineConfig(
                        minimum_age=1,
                        max_edits=2,
                        candidate_pool=2,
                        insertion_scales=(1.0,),
                        insertion_screening="coverage_aware",
                        relocation=True,
                        operations=operations,
                    ),
                )
                controller.observe(model, gradient, 1)
                proposals = controller._proposals(1, {"amplitude_lr": 4})
                self.assertTrue(all(edit.operation in operations for _, edit in proposals))
                self.assertEqual(bool(proposals), bool(operations))
                if operations == ("relocate",):
                    before = model.count
                    optimizer = torch.optim.Adam(model.parameter_groups())
                    event = controller.after_update(
                        model,
                        optimizer,
                        1,
                        learning_rates={
                            "amplitude_lr": 4,
                            "geometry_lr": 0.008,
                            "center_lr_ratio": 0.02,
                        },
                        remaining_updates=2,
                    )
                    self.assertIsNotNone(event)
                    self.assertEqual(model.count, before)


if __name__ == "__main__":
    unittest.main()
