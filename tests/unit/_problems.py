"""Deterministic physical integration fixtures, not research benchmarks."""
import torch

import gaussian_fwi as fwi
from gaussian_fwi.core import Acquisition, GridSpec, Observations


def assert_exact(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError("Different checkpoint keys")
        for name in first:
            assert_exact(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError("Different checkpoint sequence lengths")
        for a, b in zip(first, second, strict=True):
            assert_exact(a, b)
    elif first != second:
        raise AssertionError(f"Different values: {first!r} != {second!r}")

def problem(dimension=2, dtype=torch.float64):
    grid = GridSpec((12,) * dimension, 10.0)
    field = fwi.GaussianField(grid, background=(2100, 2600)).to(dtype=dtype)
    time = torch.arange(100, dtype=dtype) * 0.001 - 0.04
    squared = (torch.pi * 25 * time).square()
    wavelet = ((1 - 2 * squared) * torch.exp(-squared))[None, None].repeat(2, 1, 1)
    receivers = [[2] + ([5] if dimension == 3 else []) + [i] for i in range(1, 11)]
    sources = [[2] + [5] * (dimension - 1), [2] + ([5] if dimension == 3 else []) + [7]]
    acquisition = Acquisition(
        grid,
        0.001,
        wavelet,
        torch.tensor(sources)[:, None],
        torch.tensor([receivers]).repeat(2, 1, 1),
        pml_width=6,
        pml_frequency=25,
        max_velocity=4500,
    )
    with torch.no_grad():
        velocity = field() + 45
        velocity[6:] += 100
        traces = acquisition.simulate(velocity)
    partitions = {
        "train": torch.tensor([1, 2, 3, 6, 7, 8]),
        "validation": torch.tensor([4, 9]),
        "test": torch.tensor([0, 5]),
    }
    return field, Observations(acquisition, traces, {}, "method-interface-test"), partitions

def configuration(dimension=2):
    return fwi.InversionConfig(
        cutoffs=(10.0, 20.0), seed_shape=(2,) * dimension,
        steps_per_stage=8, validation_interval=2,
        refinement=fwi.RefinementConfig(
            warmup_steps=0, interval=1, stop_fraction=0.75,
            minimum_age=1, max_gaussians=64, max_growth=2, max_prunes=2,
            split_extent_fraction=1.0, prune_amplitude=0.0,
        ),
    )
