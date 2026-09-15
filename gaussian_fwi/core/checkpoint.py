"""Atomic persistence and coordinated restoration of model and optimizer state."""

import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


def _atomic_write(path: Path, writer: Callable, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link creates the destination atomically without replacing it.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_torch(payload: Any, path: str | Path, *, overwrite: bool = False) -> None:
    """Write a checkpoint atomically; replacing an existing file is opt-in."""
    _atomic_write(Path(path), lambda stream: torch.save(payload, stream), overwrite=overwrite)


def save_json(payload: Any, path: str | Path, *, overwrite: bool = False) -> None:
    """Write finite, human-readable JSON atomically."""
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    _atomic_write(Path(path), lambda stream: stream.write(encoded), overwrite=overwrite)


def prepare_output(path: str | Path, *, resume: bool = False) -> Path:
    """Create an output directory and reject accidental reuse of existing runs."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}; choose a new run directory"
        )
    return output


class BestCheckpoint:
    """Select finite validation scores and restore model and Adam moments together."""

    def __init__(self) -> None:
        self.score = math.inf
        self.step = -1
        self.model_state: dict | None = None
        self.optimizer_state: dict | None = None

    def consider(
        self,
        score: float,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> bool:
        if not math.isfinite(score) or score >= self.score:
            return False
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            return False
        # Build the complete snapshot before replacing a previously selected state.
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict()) if optimizer is not None else None
        if not _finite_tensors((model_state, optimizer_state)):
            return False
        self.score, self.step = float(score), int(step)
        self.model_state, self.optimizer_state = model_state, optimizer_state
        return True

    def restore(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Restore the selected finite state; raise if selection never succeeded."""
        if self.model_state is None:
            raise RuntimeError("No finite checkpoint was evaluated")
        model.load_state_dict(self.model_state)
        if optimizer is not None and self.optimizer_state is not None:
            # PyTorch can reuse same-device tensor storage during load_state_dict.
            # The live moments must never alias a reusable checkpoint's tensors.
            optimizer.load_state_dict(deepcopy(self.optimizer_state))
            optimizer.zero_grad(set_to_none=True)


def _finite_tensors(value: Any) -> bool:
    """Check nested tensor state without imposing a particular optimizer layout."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tensors(item) for item in value)
    return True
