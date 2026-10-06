"""Full-waveform inversion with an adaptive population of anisotropic Gaussians."""

from .adapt import AdaptConfig, Adapter
from .data import DataConfig, synthetic
from .field import GaussianField, PixelField, linear_start
from .fit import fit, holdout
from .metrics import velocity_errors
from .wave import Survey, lowpass, ricker, surface_survey

__all__ = [
    "AdaptConfig", "Adapter", "DataConfig", "GaussianField", "PixelField", "Survey",
    "fit", "holdout", "linear_start", "lowpass", "ricker", "surface_survey", "synthetic",
    "velocity_errors",
]
