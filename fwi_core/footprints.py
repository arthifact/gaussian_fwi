"""Explicit finite physical acoustic sources and spatial receiver averages.

Legacy Acquisition is unchanged. This wrapper implements a different physical
measurement model and keeps its logical centers in tensor-axis order. The
Gaussian density has a C2 taper between three and four sigma; its discrete
quadrature masses sum to one. A source's integrated strength is therefore the
logical source amplitude times grid cell volume. Receiver weights average the
pressure, without changing its units. Footprints crossing the grid are rejected.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch

from .geometry import GridSpec
from .physics import Acquisition

_ACQUISITION_FIELDS = tuple(item.name for item in fields(Acquisition) if item.init)


def _acquisition_values(base):
    """Detached snapshot of template settings; counters are not configuration."""
    return {name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name in _ACQUISITION_FIELDS for value in (getattr(base, name),)}


def _acquisition_copy(base):
    result = Acquisition(**_acquisition_values(base))
    result.forward_calls, result.adjoint_calls = base.forward_calls, base.adjoint_calls
    return result


@dataclass(frozen=True)
class GaussianFootprint:
    source_sigma_m: float = 10.
    receiver_sigma_m: float = 0.

    def __post_init__(self):
        for value in (self.source_sigma_m, self.receiver_sigma_m):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Footprint sigmas must be finite and nonnegative, in meters")
        if self.source_sigma_m == self.receiver_sigma_m == 0:
            raise ValueError("Use ordinary Acquisition for two ideal point operators")


def footprint_weights(grid, centers, sigma_m, *, dtype):
    """Return unique grid nodes and sparse (logical centers, nodes) mass weights."""
    if centers.ndim != 2 or centers.shape[1] != grid.ndim or centers.dtype != torch.int64:
        raise ValueError("Footprint centers must be int64 (centers, tensor axes)")
    if not math.isfinite(sigma_m) or sigma_m < 0:
        raise ValueError("Finite nonnegative footprint width required")
    radius = 4*sigma_m
    physical = centers.to(dtype)*grid.spacing
    extent = physical.new_tensor(tuple(reversed(grid.extent)))
    if ((physical-radius < -1e-10) | (physical+radius > extent+1e-10)).any():
        raise ValueError("Footprint leaves the grid: provide an explicit physical buffer")
    n = math.ceil(radius/grid.spacing)
    axis = torch.arange(-n, n+1, device=centers.device)
    offsets = torch.stack(torch.meshgrid(*([axis]*grid.ndim), indexing="ij"), -1).reshape(-1, grid.ndim)
    if sigma_m == 0:
        mass = torch.ones(1, dtype=dtype, device=centers.device)
    else:
        q = (offsets.to(dtype)*grid.spacing/sigma_m).square().sum(-1)
        keep = q < 16
        offsets, q = offsets[keep], q[keep]
        t = ((q-9)/7).clamp(0, 1)
        mass = torch.exp(-q/2)*(1-10*t**3+15*t**4-6*t**5)
        mass = mass/mass.sum()
    points = (centers[:, None]+offsets[None]).reshape(-1, grid.ndim)
    unique, inverse = torch.unique(points, dim=0, sorted=True, return_inverse=True)
    rows = torch.arange(len(centers), device=centers.device).repeat_interleave(len(offsets))
    weights = torch.sparse_coo_tensor(torch.stack((rows, inverse)), mass.repeat(len(centers)),
                                      (len(centers), len(unique)), check_invariants=True).coalesce()
    return unique, weights


class FootprintAcquisition:
    """Acquisition-compatible finite-footprint operator with actual per-shot work.

    Each physical shot is propagated once. Using separate shot calls supports
    arbitrary overlapping source/receiver footprints without duplicate grid
    receiver locations or padded dummy sources. Backward counts come from the
    underlying acoustic solves, not from the linear quadrature operators.
    Construction owns detached inputs. Public tensors, templates, operators and
    checkpoints are inspection copies; create a new wrapper to change physics.
    """

    def __init__(self, base: Acquisition, footprint=GaussianFootprint()):
        if type(base) is not Acquisition:
            raise TypeError("A plain point-center Acquisition is required as the template")
        if isinstance(footprint, dict):
            footprint = GaussianFootprint(**footprint)
        if not isinstance(footprint, GaussianFootprint):
            raise TypeError("Expected a GaussianFootprint policy")
        self._base = base = Acquisition(**_acquisition_values(base))
        self._footprint = footprint
        self._shots, self._source_operators, self._receiver_operators = [], [], []
        for shot in range(base.source_amplitudes.shape[0]):
            nodes, source = footprint_weights(base.grid, base.source_locations[shot],
                                              footprint.source_sigma_m, dtype=base.source_amplitudes.dtype)
            receivers, receiver = footprint_weights(base.grid, base.receiver_locations[shot],
                                                    footprint.receiver_sigma_m, dtype=base.source_amplitudes.dtype)
            amplitudes = torch.sparse.mm(source.transpose(0, 1), base.source_amplitudes[shot])
            self._shots.append(Acquisition(
                base.grid, base.dt, amplitudes[None].contiguous(), nodes[None], receivers[None],
                accuracy=base.accuracy, pml_width=base.pml_width,
                pml_frequency=base.pml_frequency, max_velocity=base.max_velocity,
            ))
            self._source_operators.append(source)
            self._receiver_operators.append(receiver)

    def __setattr__(self, name, value):
        if name in _ACQUISITION_FIELDS:
            raise AttributeError("Finite acquisition settings are read-only; construct a new wrapper")
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in _ACQUISITION_FIELDS:
            value = getattr(self._base, name)
            return value.detach().clone() if isinstance(value, torch.Tensor) else value
        raise AttributeError(name)

    @property
    def base(self):
        return _acquisition_copy(self._base)

    @property
    def footprint(self):
        return self._footprint

    @property
    def shots(self):
        return tuple(_acquisition_copy(shot) for shot in self._shots)

    @property
    def source_operators(self):
        return tuple(operator.detach().clone() for operator in self._source_operators)

    @property
    def receiver_operators(self):
        return tuple(operator.detach().clone() for operator in self._receiver_operators)

    @property
    def measurement_metadata(self):
        return {"format": "gaussian-fwi-footprint-acquisition-v1", "kernel": "isotropic-gaussian-c2-taper-q9-q16",
                "normalization": "unit quadrature mass; source strength includes cell volume",
                **asdict(self._footprint)}

    @property
    def counts(self):
        return {key: sum(shot.counts[key] for shot in self._shots) for key in ("forward", "adjoint")}

    def simulate(self, velocity):
        return torch.stack([torch.sparse.mm(weights, shot.simulate(velocity)[0])
                            for shot, weights in zip(self._shots, self._receiver_operators)])

    def checkpoint(self):
        base = _acquisition_values(self._base)
        base["grid"] = asdict(self.grid)
        return {"format": "gaussian-fwi-footprint-acquisition-v1", "base": base,
                "footprint": asdict(self._footprint)}

    @classmethod
    def from_checkpoint(cls, payload):
        if payload.get("format") != "gaussian-fwi-footprint-acquisition-v1":
            raise ValueError("Unknown footprint acquisition format")
        base = dict(payload["base"])
        base["grid"] = GridSpec(**base["grid"])
        return cls(Acquisition(**base), GaussianFootprint(**payload["footprint"]))
