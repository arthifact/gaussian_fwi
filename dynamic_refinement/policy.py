"""Direct refinement settings with explicit historical compatibility."""

from dataclasses import asdict, dataclass

from gaussian_fwi.refinement import RefinementConfig as EngineRefinementConfig


@dataclass(frozen=True)
class RefinementConfig:
    """Configure local edits using the regularized waveform gradient.

    New runs screen covered peaks before the shortlist budget. Use ``legacy()``
    for the frozen direct configuration. Optional equal-work recovery trials use
    training waveforms; neither predicted nor measured waveform improvement
    guarantees lower velocity-model RMSE.
    """

    max_gaussians: int = 100_000
    max_edits: int = 8
    candidate_pool: int = 32
    minimum_age: int = 5
    minimum_relative_gain: float = 1e-4
    gradient_threshold: float = 0.0
    max_field_change: float = 25.0
    split_geometry: str = "moment"
    split_fraction: float = 0.45
    max_backtracks: int = 3
    clone_radius_factor: float = 2.0
    insertion_scales: tuple[float, ...] | None = None
    insertion_separation: float = 0.5
    merge_distance: float = 0.2
    merge_shape_tolerance: float = 0.2
    merge_neighbors: int = 4
    prune_amplitude: float = 0.5
    minimum_gaussians: int = 1
    reset_amplitude: float | None = None
    stagnation_window: int = 5
    stagnation_tolerance: float = 1e-3
    insertion_screening: str = "coverage_aware"
    operations: tuple[str, ...] | None = None
    comparison_steps: int = 0
    acceptance_rtol: float = 1e-6
    acceptance_atol: float = 1e-12
    minimum_shot_support: float = 0.0

    @classmethod
    def legacy(cls, **overrides):
        """Explicit frozen direct selector; old engine checkpoints retain this default."""
        return cls(**{"insertion_screening": "legacy", **overrides})

    def __post_init__(self):
        normalized = self.engine_config()
        if normalized.operations is not None and "relocate" in normalized.operations:
            raise ValueError("The direct interface does not enable relocation")
        object.__setattr__(self, "insertion_scales", normalized.insertion_scales)
        object.__setattr__(self, "operations", normalized.operations)

    def engine_config(self) -> EngineRefinementConfig:
        """Map the direct policy and its explicit selector to the shared backend."""
        return EngineRefinementConfig(**asdict(self))
