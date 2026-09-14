"""Trainable anisotropic Gaussian fields for wave-equation inversion."""

from .adaptation import AdaptationConfig
from .decoder import decode
from .density import DensityControlConfig
from .field import GaussianField, GridSpec
from .inversion import InversionConfig, invert, resume
from .refinement import RefinementConfig, RefinementController
from .sampling import SamplingConfig, sample_on_grid, sampling_diagnostics
from .topology import EditRejected, GaussianTopology, TopologyEdit
from .trials import TrainingScores

__all__ = [
    "AdaptationConfig",
    "DensityControlConfig",
    "decode",
    "GaussianField",
    "GaussianTopology",
    "TopologyEdit",
    "EditRejected",
    "GridSpec",
    "InversionConfig",
    "RefinementConfig",
    "RefinementController",
    "SamplingConfig",
    "TrainingScores",
    "sample_on_grid",
    "sampling_diagnostics",
    "invert",
    "resume",
]
