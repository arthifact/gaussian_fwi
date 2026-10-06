"""Fit a velocity model to recorded traces, low frequencies first.

The same loop fits the Gaussian field and the pixel reference, so any
difference between them comes from the representation alone. Each band adds a
higher low-pass cutoff to the misfit; earlier bands stay in it. A fixed set of
receivers is held out of the misfit and only scored, as a check on overfitting
that needs no true model.
"""

import time

import torch
from torch import Tensor

from .adapt import Adapter
from .field import GaussianField
from .wave import Survey, lowpass


def holdout(receivers: int, every: int = 5) -> Tensor:
    """Every ``every``-th receiver, kept out of the misfit for validation."""
    return torch.arange(receivers) % every == every // 2


def total_variation(velocity: Tensor, scale: float = 100.0) -> Tensor:
    """Mean smoothed gradient magnitude, per ``scale`` m/s per cell."""
    dz = velocity[1:, :-1] - velocity[:-1, :-1]
    dx = velocity[:-1, 1:] - velocity[:-1, :-1]
    return ((dz.square() + dx.square() + 1.0).sqrt() / scale).mean()


class Misfit:
    """Normalized, time-gained L2 waveform misfit over a set of bands."""

    def __init__(self, observed: Tensor, dt: float, cutoffs, time_gain: float = 1.5,
                 validation: Tensor | None = None):
        self.dt = dt
        self.gain = torch.linspace(0, 1, observed.shape[-1]).pow(time_gain)
        self.validation = (torch.zeros(observed.shape[1], dtype=torch.bool)
                           if validation is None else validation)
        self.targets = {c: lowpass(observed, dt, c) * self.gain for c in cutoffs}

    def __call__(self, predicted: Tensor, bands, split: str = "train") -> Tensor:
        receivers = ~self.validation if split == "train" else self.validation
        values = []
        for cutoff in bands:
            target = self.targets[cutoff][:, receivers]
            residual = lowpass(predicted[:, receivers], self.dt, cutoff) * self.gain - target
            values.append(residual.square().sum() / target.square().sum())
        return torch.stack(values).mean()


def fit(model, survey: Survey, observed: Tensor, *, cutoffs=(4.0, 7.0, 12.0, 20.0),
        steps: int = 100, learning_rates: dict | None = None, adapter: Adapter | None = None,
        tv_weight: float = 0.0, validation: Tensor | None = None, verbose: bool = False):
    """Fit ``model`` in place. Returns a history of losses and population edits."""
    misfit = Misfit(observed, survey.dt, cutoffs, validation=validation)
    optimizer = torch.optim.Adam(model.parameter_groups(**(learning_rates or {})))
    adaptive = adapter is not None and isinstance(model, GaussianField)
    history, start = [], time.perf_counter()
    for stage, cutoff in enumerate(cutoffs):
        bands = cutoffs[:stage + 1]
        if adaptive:
            adapter.reset()
            adapter.start_band()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            velocity = model()
            velocity.retain_grad()
            predicted = survey.simulate(velocity)
            loss = misfit(predicted, bands)
            if tv_weight:
                loss = loss + tv_weight * total_variation(velocity)
            loss.backward()
            if adaptive:
                adapter.observe(model, velocity.grad)
            optimizer.step()
            model.project()
            if adaptive and adapter.due(step + 1, steps):
                held_out = (float(misfit(predicted.detach(), bands, "validation"))
                            if validation is not None else None)
                optimizer = adapter.edit(model, optimizer, cutoff, stage * steps + step + 1,
                                         validation_misfit=held_out)
        with torch.no_grad():
            predicted = survey.simulate(model())
            history.append({
                "cutoff_hz": cutoff, "parameters": parameter_count(model),
                "train_misfit": float(misfit(predicted, bands)),
                "validation_misfit": float(misfit(predicted, bands, "validation")),
                "seconds": round(time.perf_counter() - start, 1)})
        if verbose:
            print(history[-1], flush=True)
    return {"history": history, "edits": adapter.log if adaptive else [],
            "seconds": round(time.perf_counter() - start, 1)}


def parameter_count(model) -> int:
    return 6 * model.count if isinstance(model, GaussianField) else model.count
