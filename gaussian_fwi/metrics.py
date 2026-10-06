"""Scores against a known model. Used after a fit, never inside one."""

import numpy as np


def velocity_errors(velocity, reference, spacing: float = 10.0, illuminated_m: float = 300.0):
    """RMSE overall, above and below the depth the survey illuminates, and roughness.

    A surface spread constrains velocity only to roughly half its offset, so one
    number for the whole model hides where the data actually say something.
    Roughness is the mean absolute second difference, a speckle measure.
    """
    velocity, reference = np.asarray(velocity, float), np.asarray(reference, float)
    split = int(round(illuminated_m / spacing))

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "rmse": rmse(velocity, reference),
        "rmse_illuminated": rmse(velocity[:split], reference[:split]),
        "rmse_deep": rmse(velocity[split:], reference[split:]),
        "roughness": float(np.abs(np.diff(velocity, 2, 0)).mean()
                           + np.abs(np.diff(velocity, 2, 1)).mean()),
    }
