"""Independent scalar and upstream-network finite differences for tensor decoding."""

import math
import unittest

import numpy as np
import torch

from gaussian_fwi import GaussianField, GridSpec, decode


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_covariance_values_match_scalar_solve_and_container(self):
        for d in (2, 3):
            field = GaussianField(GridSpec((5,) * d, 10), background=(2300, 2700)).double()
            block = field.add_grid_level((2,) * d, sigma_ratio=0.4)
            with torch.no_grad():
                block.shears.fill_(0.45)
                block.amplitudes.copy_(torch.linspace(-80, 130, field.count))
            p = field.grid.points(dtype=torch.float64)
            bg = field.background[0] + (field.background[1] - field.background[0]) * p[:, -1] / 40
            oracle = []
            centers = block.centers.detach().numpy()
            factors = block.cholesky().detach().numpy()
            for point, background in zip(p.numpy(), bg.detach().numpy()):
                value = background
                for center, lower, coefficient in zip(centers, factors,
                                                        block.amplitudes.detach().numpy()):
                    white = np.linalg.solve(lower, point - center)
                    q = float(white @ white)
                    t = min(1., max(0., (q - 25) / 11))
                    taper = 1 - 10*t**3 + 15*t**4 - 6*t**5
                    value += coefficient * math.exp(-q / 2) * taper
                value += 20 * np.logaddexp(0, (1500-value)/20) - 20 * np.logaddexp(0, (value-4500)/20)
                oracle.append(value)
            for backend in ("sparse", "dense"):
                actual = decode(p, block.centers, block.cholesky(), block.amplitudes, bg,
                                backend=backend, max_pairs=11)
                torch.testing.assert_close(actual, torch.tensor(oracle), rtol=0, atol=2e-12)
                torch.testing.assert_close(actual, field().flatten(), rtol=0, atol=2e-12)

    def test_nonleaf_network_outputs_and_points_preserve_all_upstream_gradients(self):
        for d in (2, 3):
            # The network weights, not wrapped Parameters in a GaussianBlock, are
            # the differentiated inputs. Check each parameter family independently.
            size = d + d*(d+1)//2 + 2
            weights = torch.linspace(-.3, .4, size, dtype=torch.float64, requires_grad=True)
            points = torch.linspace(-1, 2, 5*d, dtype=torch.float64).reshape(5, d).requires_grad_()

            def evaluate(w, p, backend):
                centers = (2*w[:d]).unsqueeze(0)
                triangle = w[d:-2]
                row, col = torch.tril_indices(d, d)
                lower = w.new_zeros(d, d).index_put((row, col), triangle)
                lower = lower.tril(-1) + torch.diag(lower.diag().exp())
                return decode(p, centers, lower[None], (20*w[-2: -1]), 2500 + 30*w[-1],
                              backend=backend) @ torch.linspace(.2, 1, 5, dtype=w.dtype)

            for backend in ("sparse", "dense"):
                gradients = torch.autograd.grad(evaluate(weights, points, backend), (weights, points))
                self.assertTrue(bool((gradients[0].abs() > 1e-8).all()))
                h = 1e-4  # Scalar m/s output: double precision cancellation ~1e-8.
                for tensor_index, variable in enumerate((weights, points)):
                    for index in range(variable.numel()):
                        inputs = [weights.detach().clone(), points.detach().clone()]
                        inputs[tensor_index].view(-1)[index] += h
                        plus = evaluate(*inputs, backend)
                        inputs[tensor_index].view(-1)[index] -= 2*h
                        minus = evaluate(*inputs, backend)
                        self.assertAlmostEqual(float(gradients[tensor_index].flatten()[index]),
                                               float((plus-minus)/(2*h)), delta=2e-6)

    def test_invalid_covariances_and_empty_queries(self):
        p = torch.zeros(2, 2, dtype=torch.float64)
        c = torch.ones(1, 2, dtype=p.dtype)
        lower = torch.eye(2, dtype=p.dtype)[None]
        a, bg = torch.ones(1, dtype=p.dtype), torch.tensor(2500., dtype=p.dtype)
        for bad in (lower*0, lower.transpose(1, 2) + .1, lower*-1):
            with self.assertRaisesRegex(ValueError, "lower triangular"):
                decode(p, c, bad, a, bg)
        with self.assertRaisesRegex(ValueError, "dtype"):
            decode(p.float(), c, lower, a, bg)
        for backend in ("sparse", "dense"):
            self.assertEqual(decode(p[:0], c, lower, a, bg, backend=backend).shape, (0,))
            torch.testing.assert_close(decode(p, c[:0], lower[:0], a[:0], bg, backend=backend),
                                       torch.full((2,), 2500., dtype=p.dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
