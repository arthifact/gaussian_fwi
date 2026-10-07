"""Choose the starting model from the data: the best two-parameter depth trend.

Every fit starts from velocity increasing linearly with depth. Its two
end-point velocities are not assumed; they are the pair whose low-frequency
waveforms best match the training traces, found by a coarse grid search
refined around its best cell. The lowest band is used because it is the least
prone to cycle skipping. Both representations start from the same estimate.
"""

import itertools

import torch
from torch import Tensor

from .field import linear_start
from .fit import Misfit
from .wave import Survey


def estimate_start(survey: Survey, observed: Tensor, shape, *, cutoff: float = 4.0,
                   top=(1400.0, 3000.0), bottom=(1800.0, 5000.0), points: int = 9,
                   refinements: int = 2, validation: Tensor | None = None):
    """Return the starting model and a record of the search."""
    misfit = Misfit(observed, survey.dt, (cutoff,), validation=validation)
    tried = {}

    def score(v_top, v_bottom):
        key = (round(v_top, 1), round(v_bottom, 1))
        if key not in tried:
            with torch.no_grad():
                predicted = survey.simulate(linear_start(shape, v_top, v_bottom))
                tried[key] = float(misfit(predicted, (cutoff,)))
        return tried[key]

    low_top, high_top = top
    low_bottom, high_bottom = bottom
    best = None
    for _ in range(refinements + 1):
        tops = torch.linspace(low_top, high_top, points).tolist()
        bottoms = torch.linspace(low_bottom, high_bottom, points).tolist()
        for v_top, v_bottom in itertools.product(tops, bottoms):
            if v_bottom < v_top or v_bottom > survey.max_velocity:
                continue
            value = score(v_top, v_bottom)
            if best is None or value < best[0]:
                best = (value, v_top, v_bottom)
        # Zoom in on the best cell, one grid step either side.
        step_top = (high_top - low_top) / (points - 1)
        step_bottom = (high_bottom - low_bottom) / (points - 1)
        low_top, high_top = best[1] - step_top, best[1] + step_top
        low_bottom, high_bottom = best[2] - step_bottom, best[2] + step_bottom
    misfit_value, v_top, v_bottom = best
    return linear_start(shape, v_top, v_bottom), {
        "top_m_s": round(v_top, 1), "bottom_m_s": round(v_bottom, 1),
        "misfit": misfit_value, "evaluations": len(tried)}
