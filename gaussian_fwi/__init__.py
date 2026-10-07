"""Full-waveform inversion with an adaptive population of anisotropic Gaussians."""

from .adapt import AdaptConfig, Adapter
from .data import DataConfig, synthetic
from .field import GaussianField, PixelField, linear_start
from .fit import fit, holdout
from .metrics import velocity_errors
from .prior import illumination, noise_fraction, prior_weights
from .start import estimate_start
from .wave import Survey, lowpass, ricker, surface_survey

__all__ = [
    "AdaptConfig", "Adapter", "DataConfig", "GaussianField", "PixelField", "Survey",
    "estimate_start", "fit", "holdout", "illumination", "linear_start", "lowpass",
    "noise_fraction", "prior_weights", "ricker", "surface_survey", "synthetic",
    "velocity_errors",
]
