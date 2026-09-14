"""Physical-resampling helpers used only by independent unit checks."""

from dataclasses import asdict

import torch

from fwi_core import Acquisition, GridSpec
from fwi_core.checkpoint import prepare_output, save_torch
from fwi_core.footprints import FootprintAcquisition
from gaussian_fwi import GaussianField, sample_on_grid


def refined_acquisition(acquisition: Acquisition, factor: int) -> Acquisition:
    """Preserve domain, source/receiver policy and physical PML on a nested grid."""
    if type(factor) is not int or factor < 1:
        raise ValueError("Refinement factor must be a positive integer")
    grid = acquisition.grid
    refined = Acquisition(
        GridSpec(tuple((n-1)*factor+1 for n in grid.shape), grid.spacing/factor),
        acquisition.dt,
        acquisition.source_amplitudes * factor**grid.ndim,
        acquisition.source_locations * factor,
        acquisition.receiver_locations * factor,
        accuracy=acquisition.accuracy,
        pml_width=acquisition.pml_width*factor,
        pml_frequency=acquisition.pml_frequency,
        max_velocity=acquisition.max_velocity,
    )
    if isinstance(acquisition, FootprintAcquisition):
        return FootprintAcquisition(refined, acquisition.footprint)
    return refined


def verify_waveform_sampling(path, acquisition, *, output=None):
    """Report continuous saved-field propagation differences and all additional work."""
    field = GaussianField.load(path, device=acquisition.source_amplitudes.device)
    if field.grid != acquisition.grid:
        raise ValueError("Saved field and acquisition physical grids differ")
    if output is not None:
        output = prepare_output(output)
    predictions, records = [], []
    for factor in (1, 2, 4):
        fine = refined_acquisition(acquisition, factor)
        velocity = sample_on_grid(field, fine.grid)
        prediction = fine.simulate(velocity)
        predictions.append(prediction)
        record = {"factor": factor, "grid": asdict(fine.grid), "solver_calls": fine.counts}
        if isinstance(fine, FootprintAcquisition):
            record["measurement"] = fine.measurement_metadata
        if factor > 1:
            previous = predictions[-2]
            record["relative_trace_change_from_previous"] = float(
                torch.linalg.vector_norm(prediction-previous) / torch.linalg.vector_norm(prediction)
            )
        records.append(record)
    if output is not None:
        save_torch({"predictions": [p.cpu() for p in predictions], "records": records},
                   output / "predictions.pt")
    return {"grids": records, "solver_calls": {key: sum(row["solver_calls"][key] for row in records)
                                               for key in ("forward", "adjoint")},
            "interpretation": "Discretization diagnostic; no universal convergence threshold"}
