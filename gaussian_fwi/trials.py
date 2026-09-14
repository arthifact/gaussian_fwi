"""Training-only recovery trials with explicit work and acceptance criteria."""

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .field import GaussianField


@dataclass(frozen=True)
class TrainingScores:
    """Regularized objective, waveform misfit, and optional training-shot scores.

    A caller must not supply validation, test, or reference-model scores here.
    Shot scores use the same preprocessing and active bands as the waveform
    objective. Agreement across shots is a consistency diagnostic, not a
    statistical independence assumption or an uncertainty estimate.
    """

    objective: float
    waveform: float
    shots: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shots", tuple(self.shots))
        if not all(
            math.isfinite(v) and v >= 0 for v in (self.objective, self.waveform, *self.shots)
        ):
            raise FloatingPointError("Recovery trials require finite nonnegative training scores")

    def accepts(
        self,
        candidate: "TrainingScores",
        *,
        rtol: float,
        atol: float,
        shot_support: float,
    ) -> tuple[bool, float | None]:
        if (
            not all(math.isfinite(v) and v >= 0 for v in (rtol, atol))
            or not math.isfinite(shot_support)
            or not 0 <= shot_support <= 1
        ):
            raise ValueError("Invalid recovery acceptance tolerance or shot-support fraction")
        if not isinstance(candidate, TrainingScores) or len(self.shots) != len(candidate.shots):
            raise ValueError("Recovery trials must use matching training-score definitions")
        if shot_support and not self.shots:
            raise ValueError("Shot-consistency acceptance requires training-shot scores")
        supported = sum(b <= a + atol + rtol * a for a, b in zip(self.shots, candidate.shots))
        fraction = supported / len(self.shots) if self.shots else None
        accepted = (
            candidate.objective <= self.objective + atol + rtol * self.objective
            and candidate.waveform <= self.waveform + atol + rtol * self.waveform
            and (not shot_support or fraction >= shot_support)
        )
        return bool(accepted), fraction


TrainingEvaluator = Callable[[GaussianField], tuple[Tensor, TrainingScores]]


def recovery_endpoint(
    field: GaussianField,
    optimizer: torch.optim.Adam,
    evaluate: TrainingEvaluator,
    steps: int,
    work: dict[str, int],
    *,
    anchor_scores: list[TrainingScores] | None = None,
) -> TrainingScores:
    """Run a fixed-topology forecast; the caller restores its chosen anchor.

    This routine deliberately does not select checkpoints, edit topology, or
    access held-out data. The two alternatives must start from independent
    snapshots and receive the same number of ordinary Adam updates.
    """
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            work["evaluations"] += 1
            loss, scores = evaluate(field)
            _validate_evaluation(loss, scores)
            if anchor_scores is not None and not anchor_scores:
                anchor_scores.append(scores)
            if not loss.requires_grad:
                raise ValueError("Recovery objective must retain its parameter derivatives")
            work["backwards"] += 1
            loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in field.parameters()):
            raise FloatingPointError("Nonfinite recovery-trial gradient")
        optimizer.step()
        work["optimizer_updates"] += 1
        field.project_()
    with torch.no_grad():
        work["evaluations"] += 1
        loss, scores = evaluate(field)
        _validate_evaluation(loss, scores)
    return scores


def _validate_evaluation(loss, scores):
    if not isinstance(loss, Tensor) or loss.numel() != 1 or not isinstance(scores, TrainingScores):
        raise ValueError("A training evaluator must return a scalar loss and TrainingScores")
    value = float(loss.detach())
    if not math.isfinite(value) or value < 0:
        raise FloatingPointError("Nonfinite recovery-trial objective")
    if not math.isclose(value, scores.objective, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("Recovery objective tensor and reported score disagree")
