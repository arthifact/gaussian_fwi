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
    device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
    shot_batch_size: int | None = None,
) -> tuple[Observations, dict[str, torch.Tensor]]:
    """Validate a portable bundle, then explicitly move/cast its runtime tensors.

    The optional expected hash checks the stored CPU bundle before conversion.
    ``sha256`` retains that source identity; ``content_identity`` identifies the
    actual runtime precision. Device movement alone preserves content identity.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Observation device must be CPU or CUDA")
    if dtype not in (None, torch.float32, torch.float64):
        raise ValueError("Observation precision must be float32 or float64")
    if shot_batch_size is not None and (type(shot_batch_size) is not int or shot_batch_size < 1):
        raise ValueError("shot_batch_size must be a positive integer or None")
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
    if device.type != "cpu" or dtype not in (None, observations.traces.dtype):
        def convert(value):
            if isinstance(value, torch.Tensor):
                precision = dtype if value.is_floating_point() and dtype is not None else value.dtype
                return value.to(device=device, dtype=precision)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        values = convert(dict(bundle["acquisition"]))
        if "format" in values:
            acquisition = FootprintAcquisition.from_checkpoint(values)
        else:
            values["grid"] = GridSpec(**values["grid"])
            acquisition = Acquisition(**values)
        observations = Observations(acquisition, convert(bundle["traces"]))
        bundle["partitions"] = convert(bundle["partitions"])
    if shot_batch_size is not None:
        if isinstance(observations.acquisition, FootprintAcquisition):
            acquisition = FootprintAcquisition(observations.acquisition.base,
                                               observations.acquisition.footprint,
                                               shot_batch_size=shot_batch_size)
            observations = Observations(acquisition, observations.traces)
        elif shot_batch_size != 1:
            raise ValueError("Explicit shot grouping requires a finite-footprint acquisition")
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
