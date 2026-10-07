"""Choose the starting model from the data: the best-fitting depth trend.

Every fit starts from velocity increasing linearly with depth. Its two
end-point velocities are not assumed: they are the pair whose low-frequency
waveforms best match the training traces. The lowest band is used because it
is the least prone to cycle skipping.

The search is a grid refined around its best cell. The first grid must be fine
in the top velocity (100 m/s steps by default): the misfit valley is narrow in
that direction, and a coarser grid lets noise pick the cell to refine, which
made the estimate jump between distant trends from one noise seed to the next.
"""

import itertools

import torch
from torch import Tensor

from .field import linear_start
from .fit import Misfit
from .wave import Survey


def estimate_start(survey: Survey, observed: Tensor, shape, *, cutoff: float = 4.0,
                   top=(1300.0, 2500.0, 13), bottom=(2000.0, 5000.0, 13),
                   refinements: int = 2, validation: Tensor | None = None):
    """Return the starting model and a record of the search.

    ``top`` and ``bottom`` are (lowest, highest, points) of the first grid.
    Each refinement searches 5 x 5 points spanning one step either side of the
    best so far.
    """
    misfit = Misfit(observed, survey.dt, (cutoff,), validation=validation)
    tried = {}

    def score(v_top, v_bottom):
        key = (round(v_top, 1), round(v_bottom, 1))
        if key not in tried:
            with torch.no_grad():
                predicted = survey.simulate(linear_start(shape, v_top, v_bottom))
                tried[key] = float(misfit(predicted, (cutoff,)))
        return tried[key]

    (low_top, high_top, top_points), (low_bottom, high_bottom, bottom_points) = top, bottom
    best = None
    for level in range(refinements + 1):
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
