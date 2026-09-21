"""Full-waveform inversion with an adaptive, anisotropic Gaussian velocity field.

    import gaussian_fwi as gfwi

    fit = gfwi.run("data/observations.pt", output="results/fit01")
    print(fit.summary())
    velocity = fit.velocity

:func:`run` executes the accepted profile, records configuration, source and data
identities, and independently repropagates the saved field before returning a
:class:`Run`. :func:`baseline` declares a deviation from that profile. The lower
level :func:`invert` and :class:`GaussianField` remain available for custom work.
"""

from .core import Acquisition, GridSpec, Observations, Preprocessing, Regularization
from .core.io import load_observations, save_observations
from .decoder import decode
from .field import GaussianField
from .inversion import InversionConfig, invert, resume
from .refinement import RefinementConfig
from .sampling import SamplingConfig, sample_on_grid, sampling_diagnostics
from .study import BASELINE, Run, baseline, continue_run, run

__all__ = [
    "run", "baseline", "Run", "continue_run", "BASELINE",
    "Acquisition", "GridSpec", "Observations", "Preprocessing", "Regularization",
    "load_observations", "save_observations", "GaussianField", "InversionConfig",
    "RefinementConfig", "invert", "resume", "decode", "SamplingConfig",
    "sample_on_grid", "sampling_diagnostics",
]
