"""Portable observation bundles containing acquisition and measured traces only."""

from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import torch

from .checkpoint import save_torch
from .footprints import FootprintAcquisition
from .geometry import GridSpec
from .physics import Acquisition, Observations


def _partitions(partitions, receivers):
    if not isinstance(partitions, dict) or not {"train", "validation"} <= set(partitions):
        raise ValueError("Partitions require train and validation")
    seen = set()
    for name, indices in partitions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Partition names must be nonempty strings")
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
                or indices.ndim != 1 or not indices.numel()):
            raise ValueError("Partition indices must be nonempty int64 vectors")
        values = indices.cpu().tolist()
        if (min(values) < 0 or max(values) >= receivers
                or len(set(values)) != len(values) or seen.intersection(values)):
            raise ValueError("Partition indices must be in range, disjoint and unique")
        seen.update(values)


def load_observations(
    source: str | Path | BinaryIO, *, expected_content_sha256: str | None = None,
) -> tuple[Observations, dict[str, torch.Tensor]]:
    """Load a strict CPU bundle, optionally checking its physical content identity."""
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or set(bundle) != {"acquisition", "traces", "partitions"}:
        raise ValueError("Observation bundle keys must be acquisition, traces and partitions")
    values = dict(bundle["acquisition"])
    if "format" in values:
        acquisition = FootprintAcquisition.from_checkpoint(values)
    else:
        values["grid"] = GridSpec(**values["grid"])
        acquisition = Acquisition(**values)
    observations = Observations(acquisition, bundle["traces"])
    _partitions(bundle["partitions"], observations.traces.shape[1])
    identity = observations.content_identity()["sha256"]
    if expected_content_sha256 is not None and identity != expected_content_sha256:
        raise ValueError("Observation content hash mismatch")
    observations.sha256 = identity
    return observations, bundle["partitions"]


def save_observations(observations, partitions, destination: str | Path) -> None:
    """Save detached CPU tensors atomically without replacing an existing file."""
    _partitions(partitions, observations.traces.shape[1])
    acquisition = observations.acquisition
    if isinstance(acquisition, FootprintAcquisition):
        values = acquisition.checkpoint()
    else:
        values = asdict(acquisition)
        values.pop("forward_calls")
        values.pop("adjoint_calls")

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(item) for item in value)
        return value

    save_torch(cpu({"acquisition": values, "traces": observations.traces,
                    "partitions": partitions}), destination)
