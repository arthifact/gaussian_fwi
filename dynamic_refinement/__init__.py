"""Gaussian FWI with the direct, waveform-gradient-driven refinement policy."""

from gaussian_fwi.decoder import decode
from gaussian_fwi.field import GaussianField, GridSpec
from gaussian_fwi.sampling import SamplingConfig, sample_on_grid, sampling_diagnostics

from .inversion import InversionConfig, invert, resume
from .policy import RefinementConfig

__all__ = ["GaussianField", "GridSpec", "InversionConfig", "RefinementConfig", "invert", "resume", "decode",
           "SamplingConfig", "sample_on_grid", "sampling_diagnostics"]
