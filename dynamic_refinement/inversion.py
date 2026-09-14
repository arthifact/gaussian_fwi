"""A dedicated interface to direct dynamic Gaussian refinement."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from fwi_core import Observations, Preprocessing, Regularization
from gaussian_fwi.field import GaussianField
from gaussian_fwi.inversion import InversionConfig as EngineConfig
from gaussian_fwi.inversion import invert as run_inversion
from gaussian_fwi.inversion import resume as resume_inversion

from .policy import RefinementConfig


@dataclass(frozen=True)
class InversionConfig:
    """Start from one seed population and let the direct controller change it.

    ``seed_shape`` creates the initial lattice inside the driver. Leave it as
    ``None`` when the supplied field is already seeded. Frequency transitions
    introduce no further prescribed lattices through this interface.
    """

    cutoffs: tuple[float, ...]
    seed_shape: tuple[int, ...] | None = None
    steps_per_stage: int = 100
    validation_interval: int = 10
    amplitude_lr: float = 4.0
    geometry_lr: float = 0.008
    center_lr_ratio: float = 0.02
    sigma_ratio: float = 0.65
    raw_bounds_weight: float = 1e-3
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    spatial_radius_schedule: tuple[float, ...] | None = None
    sampling_refinement_factors: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        if self.seed_shape is not None:
            object.__setattr__(self, "seed_shape", tuple(self.seed_shape))
        if isinstance(self.refinement, dict):
            object.__setattr__(self, "refinement", RefinementConfig(**self.refinement))
        if not isinstance(self.refinement, RefinementConfig):
            raise TypeError("Use dynamic_refinement.RefinementConfig for direct refinement")
        engine = self.engine_config()
        object.__setattr__(self, "spatial_radius_schedule", engine.spatial_radius_schedule)
        object.__setattr__(self, "sampling_refinement_factors", engine.sampling_refinement_factors)

    def engine_config(self) -> EngineConfig:
        """Map one initial seed and the direct policy to the unchanged driver."""
        settings = asdict(self)
        settings.pop("seed_shape")
        settings["levels"] = (
            (self.seed_shape, *(None for _ in self.cutoffs[1:])) if self.cutoffs else ()
        )
        settings["refinement"] = self.refinement.engine_config()
        return EngineConfig(**settings)


def invert(
    model: GaussianField,
    observations: Observations,
    config: InversionConfig,
    output: str | Path,
    *,
    partitions: Mapping[str, Tensor],
    regularization: Regularization = Regularization(),
    preprocessing: Preprocessing = Preprocessing(),
) -> dict:
    """Fit the direct dynamic method, retaining its existing topology semantics."""
    if not isinstance(config, InversionConfig):
        raise TypeError("Use dynamic_refinement.InversionConfig for this entry point")
    return run_inversion(
        model,
        observations,
        config.engine_config(),
        output,
        partitions=partitions,
        regularization=regularization,
        preprocessing=preprocessing,
    )


def resume(
    checkpoint: str | Path,
    observations: Observations,
    output: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GaussianField, dict]:
    """Continue a completed direct stage into a new or empty output directory.

    A verified observation content identity is required. Unbound historical
    fits require their frozen source. Existing runs cannot be overwritten.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = EngineConfig(**payload["specification"]["config"])
    if (
        config.refinement is None
        or config.refinement.directional_splits
        or config.refinement.relocation
        or any(level is not None for level in config.levels[1:])
        or config.adaptation is not None
        or config.density_control is not None
    ):
        raise ValueError("Checkpoint belongs to a different Gaussian method")
    return resume_inversion(checkpoint, observations, output, device=device)
