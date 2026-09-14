"""Differentiable acoustic propagation and observation-only waveform objectives.

This module contains no dataset loaders, acquisition presets, or target models.
Coordinates and physical units are defined by :class:`fwi_core.GridSpec`.
"""

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import deepwave
import torch
from torch import Tensor

from .geometry import GridSpec
from .identity import OBSERVATION_IDENTITY_FORMAT, observation_sha256


@dataclass
class Acquisition:
    """Known-source scalar acoustic acquisition in SI units.

    Source amplitudes have shape ``(shots, sources, time)``. Source and receiver
    locations contain integer grid indices in tensor-axis order. A fixed
    ``max_velocity`` keeps CFL and PML settings consistent as the model changes.
    """

    grid: GridSpec
    dt: float
    source_amplitudes: Tensor
    source_locations: Tensor
    receiver_locations: Tensor
    accuracy: int = 4
    pml_width: int = 20
    pml_frequency: float = 15.0
    max_velocity: float = 4500.0
    forward_calls: int = field(default=0, init=False)
    adjoint_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for name in ("dt", "pml_frequency", "max_velocity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.accuracy not in (2, 4, 6, 8) or self.pml_width < 0:
            raise ValueError("Invalid finite difference order or PML width")
        source = self.source_amplitudes
        if source.ndim != 3 or min(source.shape) < 1 or not source.is_floating_point():
            raise ValueError("source_amplitudes must have shape (shots, sources, time)")
        if not torch.isfinite(source).all():
            raise ValueError("Source amplitudes must be finite")
        if self.receiver_locations.ndim != 3 or self.receiver_locations.shape[1] < 1:
            raise ValueError("Receiver locations must have shape (shots, receivers, ndim)")
        for locations, count in (
            (self.source_locations, source.shape[1]),
            (self.receiver_locations, self.receiver_locations.shape[1]),
        ):
            expected = (source.shape[0], count, self.grid.ndim)
            if locations.shape != expected or locations.dtype != torch.int64:
                raise ValueError("Acquisition locations must be int64 (shots, points, ndim)")
            if locations.device != source.device:
                raise ValueError("Acquisition tensors must be on the same device")
            if not torch.all(
                (locations >= 0) & (locations < locations.new_tensor(self.grid.shape))
            ):
                raise ValueError("Acquisition location lies outside the physical grid")

    @property
    def counts(self) -> dict[str, int]:
        """Batched solver calls, including line searches and validation evaluations."""
        return {"forward": self.forward_calls, "adjoint": self.adjoint_calls}

    def simulate(self, velocity: Tensor) -> Tensor:
        """Return receiver traces, retaining the gradient connection to velocity."""
        if tuple(velocity.shape) != self.grid.shape:
            raise ValueError("Velocity shape does not match acquisition grid")
        if (
            velocity.dtype != self.source_amplitudes.dtype
            or velocity.device != self.source_amplitudes.device
        ):
            raise ValueError("Velocity and source amplitudes must share dtype and device")
        if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
            raise FloatingPointError("Wave propagation requires finite, positive velocity")
        if torch.any(velocity > self.max_velocity * (1 + 1e-6)):
            raise ValueError("Velocity exceeds the fixed solver maximum")
        self.forward_calls += 1
        traces = deepwave.scalar(
            velocity,
            self.grid.spacing,
            self.dt,
            source_amplitudes=self.source_amplitudes,
            source_locations=self.source_locations,
            receiver_locations=self.receiver_locations,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            pml_freq=self.pml_frequency,
            max_vel=self.max_velocity,
        )[-1]
        if traces.requires_grad:
            traces.register_hook(self._record_adjoint)
        return traces

    def _record_adjoint(self, gradient: Tensor) -> Tensor:
        self.adjoint_calls += 1
        return gradient


@dataclass
class Observations:
    """Recorded waveforms and acquisition, without a target velocity model.

    ``sha256`` is an optional caller-supplied provenance label. Restart checks
    use :meth:`content_identity`, recomputed from actual arrays and settings.
    """

    acquisition: Acquisition
    traces: Tensor
    metadata: dict = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self) -> None:
        acq = self.acquisition
        expected = (*acq.receiver_locations.shape[:2], acq.source_amplitudes.shape[-1])
        if self.traces.shape != expected or not torch.isfinite(self.traces).all():
            raise ValueError("Observation shape is inconsistent or contains non-finite values")
        if (
            self.traces.device != acq.source_amplitudes.device
            or self.traces.dtype != acq.source_amplitudes.dtype
        ):
            raise ValueError("Observations and acquisition must share dtype and device")
        if self.traces.requires_grad:
            raise ValueError("Observed waveforms must be detached from autograd")

    def content_identity(self) -> dict[str, str]:
        """Identify current inputs, excluding labels, metadata and solver counters."""
        return {"format": OBSERVATION_IDENTITY_FORMAT,
                "sha256": observation_sha256(self.acquisition, self.traces)}


@dataclass(frozen=True)
class Preprocessing:
    """Optional observation-derived time gain and capped trace balancing."""

    time_gain_power: float = 0.0
    trace_balance_cap: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_gain_power) or self.time_gain_power < 0:
            raise ValueError("time_gain_power must be finite and nonnegative")
        if self.trace_balance_cap is not None and (
            not math.isfinite(self.trace_balance_cap) or self.trace_balance_cap < 1
        ):
            raise ValueError("trace_balance_cap must be finite and at least one")


def lowpass(traces: Tensor, dt: float, cutoff: float) -> Tensor:
    """Apply a zero-padded, fourth-order Butterworth magnitude filter."""
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(cutoff) or not 0 < cutoff < 0.5 / dt:
        raise ValueError("Cutoff must be finite, positive, and below Nyquist")
    nt = traces.shape[-1]
    nfft = 2 * nt
    frequency = torch.fft.rfftfreq(nfft, dt, device=traces.device, dtype=traces.dtype)
    transfer = torch.rsqrt(1 + (frequency / cutoff).pow(8))
    return torch.fft.irfft(torch.fft.rfft(traces, n=nfft) * transfer, n=nfft)[..., :nt]


class WaveformObjective:
    """Normalized waveform residuals with explicit, disjoint receiver partitions.

    ``partitions`` must include nonempty ``train`` and ``validation`` entries;
    an optional ``test`` entry is only evaluated after model selection. Trace
    balancing uses a reference amplitude derived from training data alone.
    """

    def __init__(
        self,
        observations: Observations,
        cutoffs: Sequence[float],
        partitions: Mapping[str, Tensor],
        preprocessing: Preprocessing = Preprocessing(),
    ):
        self.dt = observations.acquisition.dt
        self.cutoffs = tuple(cutoffs)
        self.split = {
            name: ids.to(device=observations.traces.device) for name, ids in partitions.items()
        }
        if not {"train", "validation"}.issubset(self.split):
            raise ValueError("Explicit train and validation receiver partitions are required")
        combined = []
        for ids in self.split.values():
            if ids.ndim != 1 or ids.dtype != torch.int64 or not len(ids):
                raise ValueError("Each receiver partition must be a nonempty int64 vector")
            if torch.any(ids < 0) or torch.any(ids >= observations.traces.shape[1]):
                raise ValueError("Receiver partition contains an invalid index")
            combined.extend(ids.tolist())
        if len(combined) != len(set(combined)):
            raise ValueError("Receiver partitions must not overlap or repeat indices")
        self.targets: dict[float, Tensor] = {}
        self.weights: dict[float, Tensor] = {}
        self.denominators: dict[float, dict[str, Tensor]] = {}
        relative_time = torch.linspace(
            0,
            1,
            observations.traces.shape[-1],
            device=observations.traces.device,
            dtype=observations.traces.dtype,
        )
        gain = relative_time.pow(preprocessing.time_gain_power)
        for cutoff in self.cutoffs:
            target = lowpass(observations.traces, self.dt, cutoff).detach()
            rms = (target * gain).square().mean(-1).sqrt()
            train_rms = rms[:, self.split["train"]]
            positive = train_rms[train_rms > 0]
            if not len(positive):
                raise ValueError("Training observations contain no signal")
            trace_weight = torch.ones_like(rms)
            if preprocessing.trace_balance_cap is not None:
                reference = positive.median()
                cap = preprocessing.trace_balance_cap
                trace_weight = (reference / rms.clamp_min(reference * 1e-3)).clamp(1 / cap, cap)
            weight = (trace_weight[..., None] * gain).detach()
            self.targets[cutoff], self.weights[cutoff] = target, weight
            self.denominators[cutoff] = {
                name: (target[:, ids] * weight[:, ids]).square().mean().clamp_min(1e-20)
                for name, ids in self.split.items()
            }

    def losses(self, prediction: Tensor, active: Sequence[float], split: str = "train") -> Tensor:
        """Return one dimensionless squared-residual objective per active cutoff."""
        ids = self.split[split]
        values = []
        for cutoff in active:
            residual = lowpass(prediction[:, ids], self.dt, cutoff) - self.targets[cutoff][:, ids]
            weighted = residual * self.weights[cutoff][:, ids]
            values.append(weighted.square().mean() / self.denominators[cutoff][split])
        return torch.stack(values)

    def losses_by_shot(
        self, prediction: Tensor, active: Sequence[float], split: str = "train"
    ) -> Tensor:
        """Return (shots, bands) scores with the ordinary objective's normalization.

        Their mean over shots equals ``losses`` up to floating-point reduction
        order. A shared denominator preserves the survey objective; this method
        does not independently balance shots or change their statistical weight.
        """
        ids = self.split[split]
        values = []
        for cutoff in active:
            residual = lowpass(prediction[:, ids], self.dt, cutoff) - self.targets[cutoff][:, ids]
            weighted = residual * self.weights[cutoff][:, ids]
            values.append(weighted.square().mean((-1, -2)) / self.denominators[cutoff][split])
        return torch.stack(values, dim=-1)
