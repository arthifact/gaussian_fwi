"""Split, merge and prune: a population that grows where the data ask for detail.

Every edit approximately preserves the current velocity, so optimization
continues from where it was rather than recovering from a jump:

* **split** replaces a Gaussian by two along its long axis, at +/- half a width,
  with widths and amplitudes chosen to keep its mean, covariance and integral
  (moment matching). The peak changes by about 2%.
* **merge** replaces two overlapping same-sign Gaussians by the one with the
  same integral, mean and covariance, and only when that one reproduces their
  sum to within ``merge_tolerance``.
* **prune** removes a Gaussian whose amplitude stayed negligible, so regions the
  data do not constrain return to the starting model.

Whether the population grows at all is decided by receivers the fit never
sees. Growth continues while the misfit on those held-out receivers keeps
improving by at least ``min_improvement`` between edits. When it stops
improving, further Gaussians would only fit noise in the training traces, so
splitting stops for the rest of that band; a new band carries new information
and may grow again. More noise or less data therefore stops growth earlier, and
the population levels off by itself instead of approaching one per cell.

Which Gaussians split is decided by the data too. For each Gaussian, the velocity
gradient over its footprint is split into the part a change of its amplitude
could follow and the rest. A large remainder means the footprint holds
structure one Gaussian cannot represent. A Gaussian splits only while it is
wider than the current band can resolve, ``resolution * v_min / f``, so detail
appears as the frequency rises and never below what the waves can see.

Each surviving Gaussian keeps its Adam moments; children inherit their
parent's, and a merged Gaussian takes the mean of its parents'.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .field import PARAMETERS, GaussianField, gaussian_kernels


@dataclass(frozen=True)
class AdaptConfig:
    interval: int = 20              # steps between edits
    stop_fraction: float = 0.6      # no edits in the last 40% of each band
    min_age: int = 20               # steps a Gaussian must exist before prune or merge
    max_gaussians: int = 400
    max_splits: int = 24            # per edit
    max_merges: int = 24            # per edit
    split_factor: float = 1.0       # split if score exceeds this multiple of the median
    min_improvement: float = 0.01   # relative held-out improvement that keeps growth going
    resolution: float = 0.25        # smallest split child width, in wavelengths
    prune_amplitude: float = 10.0   # m/s
    merge_distance: float = 1.0     # Mahalanobis distance between centres
    merge_tolerance: float = 0.03   # relative L2 error of a merge on the grid


def covariance(log_width: Tensor, angle: Tensor) -> Tensor:
    width = log_width.exp()
    cos, sin = angle.cos(), angle.sin()
    rotation = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    return rotation @ torch.diag_embed(width.square()) @ rotation.transpose(-1, -2)


def from_covariance(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Log widths (long, short) and long-axis angle of 2x2 covariances."""
    values, vectors = torch.linalg.eigh(matrix)                    # ascending
    major = vectors[..., :, 1]
    angle = torch.atan2(major[..., 1], major[..., 0])
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    return values.flip(-1).clamp_min(1e-6).sqrt().log(), angle


class Adapter:
    """Accumulate split scores during a band and apply edits on schedule."""

    def __init__(self, config: AdaptConfig = AdaptConfig()):
        self.config = config
        self.reset()
        self.start_band()
        self.log: list[dict] = []

    def start_band(self) -> None:
        """New frequencies may justify new detail: allow growth again."""
        self.growing, self.previous = True, None

    def reset(self) -> None:
        self.score, self.samples = None, 0

    @torch.no_grad()
    def observe(self, field: GaussianField, velocity_gradient: Tensor) -> None:
        kernels = field.kernels()
        gradient = velocity_gradient.reshape(-1)
        mass = kernels.sum(1).clamp_min(1e-12)
        unexplained = (kernels * gradient.abs()).sum(1) - (kernels @ gradient).abs()
        score = unexplained / mass
        self.score = score if self.score is None else self.score + score
        self.samples += 1

    def due(self, step: int, steps: int) -> bool:
        return step > 0 and step % self.config.interval == 0 \
            and step < self.config.stop_fraction * steps

    @torch.no_grad()
    def edit(self, field: GaussianField, optimizer, cutoff_hz: float, step: int,
             validation_misfit: float | None = None):
        """Prune, merge, then split. Returns the optimizer to use from now on.

        ``validation_misfit`` is the current misfit on held-out receivers. Without
        it the population grows on the score alone, with no data-driven stop.
        """
        cfg = self.config
        if validation_misfit is not None:
            if self.previous is not None and self.growing:
                gain = (self.previous - validation_misfit) / max(self.previous, 1e-12)
                self.growing = gain >= cfg.min_improvement
            self.previous = validation_misfit
        values = {name: getattr(field, name).detach().clone() for name in PARAMETERS}
        old_age = field.age.clone()
        count = field.count
        score = self.score / max(self.samples, 1)
        self.reset()

        established = old_age >= cfg.min_age
        prune = established & (values["amplitude"].abs() < cfg.prune_amplitude)
        alive = ~prune

        merged_pairs = self._merges(field, values, alive & established)
        for i, j in merged_pairs:
            alive[i] = alive[j] = False

        resolution = cfg.resolution * float(field().min()) / cutoff_hz
        long_width = values["log_width"][:, 0].exp()
        candidates = alive & (long_width * math.sqrt(0.75) >= resolution)
        threshold = cfg.split_factor * score[alive].median() if alive.any() else math.inf
        candidates &= score > threshold
        order = torch.argsort(score.masked_fill(~candidates, -math.inf), descending=True)
        room = cfg.max_gaussians - (int(alive.sum()) + len(merged_pairs))
        allowed = min(cfg.max_splits, room, int(candidates.sum())) if self.growing else 0
        splits = order[:max(0, allowed)].tolist()
        for i in splits:
            alive[i] = False

        rows, origin, ages = {name: [] for name in PARAMETERS}, [], []

        def add(row, parents, age):
            for name in PARAMETERS:
                rows[name].append(row[name])
            origin.append(parents)
            ages.append(age)

        for i in torch.nonzero(alive).flatten().tolist():
            add({name: values[name][i] for name in PARAMETERS}, [i], int(old_age[i]))
        for i, j in merged_pairs:
            add(self._merged(values, i, j), [i, j], 0)
        for i in splits:
            for child in self._children(values, i):
                add(child, [i], 0)

        new_values = {name: torch.stack(rows[name]) for name in PARAMETERS}
        optimizer = _transfer(field, optimizer, new_values, origin)
        field.age = torch.tensor(ages, dtype=torch.long, device=field.age.device)
        event = {"step": step, "cutoff_hz": cutoff_hz, "before": count, "after": field.count,
                 "pruned": int(prune.sum()), "merged": len(merged_pairs),
                 "split": len(splits), "split_resolution_m": round(resolution, 1),
                 "growing": self.growing, "validation_misfit": validation_misfit}
        self.log.append(event)
        return optimizer

    def _merges(self, field, values, eligible):
        cfg = self.config
        index = torch.nonzero(eligible).flatten()
        if len(index) < 2:
            return []
        center = values["center"][index]
        cov = covariance(values["log_width"][index], values["angle"][index])
        amplitude = values["amplitude"][index]
        offset = center[:, None] - center[None]
        joint = cov[:, None] + cov[None]
        distance = (offset[..., None, :] @ torch.linalg.solve(joint, offset[..., None]))
        distance = distance.squeeze(-1).squeeze(-1).sqrt()
        same_sign = (amplitude[:, None] * amplitude[None]) > 0
        upper = torch.triu(torch.ones_like(same_sign), 1)
        close = same_sign & upper & (distance < cfg.merge_distance)
        pairs = torch.nonzero(close)
        pairs = pairs[torch.argsort(distance[close])]
        used, chosen = set(), []
        for a, b in pairs.tolist():
            if len(chosen) >= cfg.max_merges:
                break
            i, j = int(index[a]), int(index[b])
            if i in used or j in used:
                continue
            merged = self._merged(values, i, j)
            pair = (values["amplitude"][[i, j]] @ _kernels(field, {
                name: values[name][[i, j]] for name in PARAMETERS}))
            single = merged["amplitude"] * _kernels(field, {
                name: merged[name][None] for name in PARAMETERS})[0]
            error = (single - pair).norm() / pair.norm().clamp_min(1e-12)
            if error < cfg.merge_tolerance:
                chosen.append((i, j))
                used.update((i, j))
        return chosen

    @staticmethod
    def _merged(values, i, j):
        width = values["log_width"][[i, j]].exp()
        weight = values["amplitude"][[i, j]] * width.prod(-1)                # integrals / 2pi
        center = values["center"][[i, j]]
        mean = (weight[:, None] * center).sum(0) / weight.sum()
        spread = covariance(values["log_width"][[i, j]], values["angle"][[i, j]])
        offset = center - mean
        cov = (weight[:, None, None] * (spread + offset[:, :, None] * offset[:, None, :])).sum(0) \
            / weight.sum()
        log_width, angle = from_covariance(cov)
        return {"center": mean, "log_width": log_width, "angle": angle,
                "amplitude": weight.sum() / log_width.exp().prod()}

    @staticmethod
    def _children(values, i):
        log_width, angle = values["log_width"][i], values["angle"][i]
        direction = torch.stack([angle.cos(), angle.sin()])
        step = 0.5 * log_width[0].exp()
        child_width = log_width.clone()
        child_width[0] = child_width[0] + 0.5 * math.log(0.75)
        amplitude = values["amplitude"][i] / (2 * math.sqrt(0.75))
        return [{"center": values["center"][i] + sign * step * direction,
                 "log_width": child_width, "angle": angle, "amplitude": amplitude}
                for sign in (-1.0, 1.0)]


def _kernels(field, values):
    return gaussian_kernels(field.points, values["center"], values["log_width"], values["angle"])


def _transfer(field, optimizer, new_values, origin):
    """Install the new population and carry each row's Adam moments across."""
    old_state = {}
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        old_state[group["name"]] = optimizer.state.get(parameter, {})
    field.replace(**new_values)
    groups = []
    for group in optimizer.param_groups:
        settings = {k: v for k, v in group.items() if k != "params"}
        groups.append({**settings, "params": [getattr(field, group["name"])]})
    fresh = torch.optim.Adam(groups)
    for group in fresh.param_groups:
        state = old_state[group["name"]]
        if not state:
            continue
        parameter = group["params"][0]
        moved = {"step": state["step"].clone()}
        for key in ("exp_avg", "exp_avg_sq"):
            moved[key] = torch.stack([state[key][parents].mean(0) for parents in origin])
        fresh.state[parameter] = moved
    return fresh
