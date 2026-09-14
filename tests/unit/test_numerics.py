"""Independent numerical checks of the decoder and acoustic chain rule."""

import unittest

import torch
from test_dynamic import problem

from dynamic_refinement import GaussianField
from fwi_core import GridSpec, Preprocessing, Regularization
from fwi_core.physics import WaveformObjective
from gaussian_fwi.raster import dense_sum, kernel, pair_chunks, sparse_sum


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kernel_matches_piecewise_formula_and_finite_difference(self):
        q = torch.tensor([0.0, 4.0, 24.0, 25.0, 28.0, 32.0, 35.0, 36.0, 49.0], dtype=torch.float64)
        expected = []
        for value in q.tolist():
            if value <= 25:
                taper = 1.0
            elif value >= 36:
                taper = 0.0
            else:
                t = (value - 25) / 11
                taper = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
            expected.append(torch.exp(torch.tensor(-value / 2, dtype=q.dtype)) * taper)
        values, derivatives = kernel(q)
        torch.testing.assert_close(values, torch.stack(expected), rtol=1e-11, atol=1e-20)
        h = 1e-4
        finite_difference = (kernel(q + h)[0] - kernel(q - h)[0]) / (2 * h)
        torch.testing.assert_close(derivatives, finite_difference, rtol=1e-7, atol=1e-18)

    def test_sparse_values_and_all_input_gradients_match_dense_autograd(self):
        for dimension in (2, 3):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dimension=dimension, dtype=dtype):
                    generator = torch.Generator().manual_seed(713 + dimension)
                    points = torch.randn(37, dimension, generator=generator, dtype=dtype) * 3
                    centers = torch.randn(3, dimension, generator=generator, dtype=dtype)
                    lower = torch.eye(dimension, dtype=dtype).repeat(3, 1, 1)
                    lower[:, 1, 0] = torch.tensor([0.4, -0.3, 0.2], dtype=dtype)
                    lower[:, -1, -1] = torch.tensor([0.6, 1.5, 0.9], dtype=dtype)
                    precision = lower @ lower.transpose(-1, -2)
                    amplitudes = torch.tensor([2.0, -3.0, 0.0], dtype=dtype)
                    inputs = (points, centers, precision, amplitudes)
                    upstream = torch.randn(37, generator=generator, dtype=dtype)
                    dense_inputs = tuple(t.clone().requires_grad_() for t in inputs)
                    expected = dense_sum(*dense_inputs)
                    expected_grad = torch.autograd.grad(expected @ upstream, dense_inputs)
                    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
                    # Small chunks exercise the wide-kernel streaming branch too.
                    for max_pairs in (5, 1000):
                        sparse_inputs = tuple(t.clone().requires_grad_() for t in inputs)
                        actual = sparse_sum(*sparse_inputs, max_pairs=max_pairs)
                        gradients = torch.autograd.grad(actual @ upstream, sparse_inputs)
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                        for observed, reference in zip(gradients, expected_grad, strict=True):
                            torch.testing.assert_close(
                                observed, reference, rtol=tolerance, atol=tolerance
                            )

    def test_pair_stream_covers_every_supported_pair_with_bounded_chunks(self):
        grid = GridSpec((7, 9), 1.0)
        points = grid.points(dtype=torch.float64).numpy()
        centers = torch.tensor([[2.0, 3.0], [6.0, 4.0]], dtype=torch.float64).numpy()
        precision = torch.tensor(
            [[[2.0, 0.5], [0.5, 1.0]], [[0.04, 0.0], [0.0, 0.04]]], dtype=torch.float64
        ).numpy()
        visited = set()
        for gaussian_ids, point_ids in pair_chunks(points, centers, precision, 6.0, 7):
            self.assertLessEqual(len(point_ids), 7)
            for pair in zip(gaussian_ids.tolist(), point_ids.tolist(), strict=True):
                self.assertNotIn(pair, visited)
                visited.add(pair)
        for gi, (center, matrix) in enumerate(zip(centers, precision, strict=True)):
            for pi, point in enumerate(points):
                delta = point - center
                if delta @ matrix @ delta < 36:
                    self.assertIn((gi, pi), visited)

    def test_full_field_parameter_gradients_match_dense_reference(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                field = GaussianField(
                    GridSpec((7,) * dimension, 10), background=(2200, 2800), max_pairs=13
                ).double()
                block = field.add_grid_level((2,) * dimension, sigma_ratio=0.3)
                with torch.no_grad():
                    block.amplitudes.copy_(torch.linspace(-80, 90, field.count))
                    block.shears.fill_(0.35)
                reference = GaussianField.from_checkpoint(field.checkpoint())
                reference.backend = "dense"
                weight = torch.linspace(-1, 2, field._points.shape[0], dtype=torch.float64)
                gradients = torch.autograd.grad(
                    field().flatten() @ weight, tuple(field.parameters())
                )
                expected = torch.autograd.grad(
                    reference().flatten() @ weight, tuple(reference.parameters())
                )
                for actual, desired in zip(gradients, expected, strict=True):
                    torch.testing.assert_close(actual, desired, rtol=2e-11, atol=2e-10)

    def test_acoustic_objective_directional_derivative_for_each_parameter_family(self):
        field, observations, partitions = problem()
        block = field.add_grid_level((2, 2), sigma_ratio=0.4)
        with torch.no_grad():
            block.amplitudes.copy_(torch.tensor([35.0, -25.0, 15.0, -10.0]))
            block.shears.fill_(0.2)
        objective = WaveformObjective(observations, (10.0, 20.0), partitions, Preprocessing(1.5, 5))
        regularization = Regularization(tv_weight=1e-4)

        def loss():
            velocity = field()
            prediction = observations.acquisition.simulate(velocity)
            return objective.losses(prediction, (10.0, 20.0)).mean() + regularization(
                velocity, field.grid.spacing
            )

        parameters = dict(field.named_parameters())
        gradients = torch.autograd.grad(loss(), tuple(parameters.values()))
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
            with self.subTest(parameter=name):
                # An arbitrary fixed direction, independent of the adjoint being tested.
                direction = torch.cos(torch.arange(parameter.numel(), dtype=parameter.dtype))
                direction = direction.reshape_as(parameter)
                direction /= torch.linalg.vector_norm(direction)
                expected = float((gradient * direction).sum())
                self.assertGreater(abs(expected), 1e-12)
                original = parameter.detach().clone()
                estimates = []
                try:
                    for h in (1e-3, 1e-4):
                        with torch.no_grad():
                            parameter.copy_(original + h * direction)
                            positive = float(loss())
                            parameter.copy_(original - h * direction)
                            negative = float(loss())
                        estimates.append((positive - negative) / (2 * h))
                finally:
                    with torch.no_grad():
                        parameter.copy_(original)
                # float64 central differences; the larger step checks cancellation.
                for estimate in estimates:
                    self.assertAlmostEqual(estimate, expected, delta=2e-5 * abs(expected) + 2e-11)
