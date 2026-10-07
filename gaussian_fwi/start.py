"""Choose the starting model from the data: the best-fitting depth trend.

Every fit starts from velocity increasing linearly with depth. Its two
end-point velocities are not assumed: they are the pair whose low-frequency
waveforms best match the training traces. Only the lowest band is used:
adding 7 Hz made the choice consistent but wrong on Overthrust, the signature
of cycle skipping.

Two details keep the estimate stable when the data are noisy:

* The first grid is fine in the top velocity (100 m/s steps by default). The
  misfit valley is narrow in that direction, and a coarser grid let noise pick
  the cell to refine.
* Recorded and predicted traces are both averaged over a few neighbouring
  receivers before they are compared (``receiver_smoothing``, in receivers).
  At 4 Hz a wavelength spans dozens of receivers, so this keeps the signal and
  averages out noise, which is independent from receiver to receiver. On
  Overthrust at SNR 5 it removed a near-constant trend that one noise seed
  otherwise chose. Only training receivers are used; held-out ones stay unseen.
"""

import itertools

import torch
from torch import Tensor

from .field import linear_start
from .wave import Survey, lowpass


def smooth_receivers(traces: Tensor, sigma: float) -> Tensor:
    """Gaussian average along the receiver axis of ``(shots, receivers, time)``."""
    if sigma <= 0:
        return traces
    radius = int(3 * sigma)
    kernel = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=traces.dtype) / sigma)
                       .square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    shots, receivers, samples = traces.shape
    flat = traces.permute(0, 2, 1).reshape(-1, 1, receivers)
    flat = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
    return torch.nn.functional.conv1d(flat, kernel).reshape(shots, samples, receivers) \
        .permute(0, 2, 1)


def estimate_start(survey: Survey, observed: Tensor, shape, *, cutoff: float = 4.0,
                   top=(1300.0, 2500.0, 13), bottom=(2000.0, 5000.0, 13),
                   refinements: int = 2, receiver_smoothing: float = 2.0,
                   time_gain: float = 1.5, validation: Tensor | None = None):
    """Return the starting model and a record of the search.

    ``top`` and ``bottom`` are (lowest, highest, points) of the first grid.
    Each refinement searches 5 x 5 points spanning one step either side of the
    best so far.
    """
    train = (torch.ones(observed.shape[1], dtype=torch.bool) if validation is None
             else ~validation)
    gain = torch.linspace(0, 1, observed.shape[-1]).pow(time_gain)

    def prepare(traces):
        return smooth_receivers(lowpass(traces[:, train], survey.dt, cutoff) * gain,
                                receiver_smoothing)

    target = prepare(observed)
    energy = target.square().sum()
    tried = {}

    def score(v_top, v_bottom):
        key = (round(v_top, 1), round(v_bottom, 1))
        if key not in tried:
            with torch.no_grad():
                predicted = prepare(survey.simulate(linear_start(shape, v_top, v_bottom)))
                tried[key] = float((predicted - target).square().sum() / energy)
        return tried[key]

    (low_top, high_top, top_points), (low_bottom, high_bottom, bottom_points) = top, bottom
    best = None
    for _ in range(refinements + 1):
        tops = torch.linspace(low_top, high_top, top_points).tolist()
        bottoms = torch.linspace(low_bottom, high_bottom, bottom_points).tolist()
        for v_top, v_bottom in itertools.product(tops, bottoms):
            if v_bottom < v_top or v_bottom > survey.max_velocity:
                continue
            value = score(v_top, v_bottom)
            if best is None or value < best[0]:
                best = (value, v_top, v_bottom)
        step_top = (high_top - low_top) / (top_points - 1)
        step_bottom = (high_bottom - low_bottom) / (bottom_points - 1)
        low_top, high_top = best[1] - step_top, best[1] + step_top
        low_bottom, high_bottom = best[2] - step_bottom, best[2] + step_bottom
        top_points = bottom_points = 5
    misfit_value, v_top, v_bottom = best
    return linear_start(shape, v_top, v_bottom), {
        "top_m_s": round(v_top, 1), "bottom_m_s": round(v_bottom, 1),
        "misfit": misfit_value, "evaluations": len(tried)}
