"""Numerical geometry and coordinated optimizer-state primitives.

These utilities do not choose a refinement policy or run an inversion.
"""

from collections import defaultdict
from copy import deepcopy

import torch
from torch import Tensor, nn

from .core.checkpoint import BestCheckpoint
from .field import GaussianBlock, GaussianField
from .raster import kernel


class GaussianCheckpoint(BestCheckpoint):
    """Restore Gaussian topology, parameter values, and Adam state together.

    References preserve parameter identities in memory; tensor values and
    optimizer moments use independent snapshots. Replaced blocks therefore
    remain recoverable even when later iterations change the number of kernels.
    Portable checkpoints continue to use ``GaussianField.checkpoint``.
    """

    def consider(self, score, step, model, optimizer=None) -> bool:
        blocks = tuple(model.blocks)
        metadata = [(b.nominal_scale, b.amplitude_lr_scale) for b in blocks]
        groups = (
            [
                {
                    **deepcopy({k: v for k, v in group.items() if k != "params"}),
                    "params": list(group["params"]),
                }
                for group in optimizer.param_groups
            ]
            if optimizer is not None
            else None
        )
        if not super().consider(score, step, model, optimizer):
            return False
        self.blocks, self.metadata, self.groups = blocks, metadata, groups
        return True

    def restore(self, model, optimizer=None) -> None:
        if self.model_state is None:
            raise RuntimeError("No finite checkpoint was evaluated")
        model.blocks = nn.ModuleList(self.blocks)
        for block, (nominal_scale, amplitude_lr_scale) in zip(self.blocks, self.metadata):
            block.nominal_scale, block.amplitude_lr_scale = nominal_scale, amplitude_lr_scale
        if optimizer is not None and self.groups is not None:
            optimizer.param_groups = [
                {**group, "params": list(group["params"])} for group in self.groups
            ]
        super().restore(model, optimizer)




def _basis(field: GaussianField, center: Tensor, covariance: Tensor) -> Tensor:
    """Evaluate one proposal in float64 without constructing all Gaussian pairs."""
    points = field._points.double()
    delta = points - center.double()
    lower = torch.linalg.cholesky(covariance.double())
    whitened = torch.linalg.solve_triangular(lower, delta.T, upper=False)
    return kernel(whitened.square().sum(0))[0]




def _admissible(field: GaussianField, centers: Tensor, covariances: Tensor) -> bool:
    radii = torch.linalg.eigvalsh(covariances.double()).sqrt()
    extent = centers.new_tensor(field.grid.extent)
    return bool(
        torch.isfinite(radii).all()
        and (radii >= field.sigma_min).all()
        and (radii <= field.sigma_max * (1 + 1e-12)).all()
        and (centers >= -0.2 * extent).all()
        and (centers <= 1.2 * extent).all()
    )


def _block_from_tensors(centers, covariances, amplitudes, reference) -> GaussianBlock:
    lower = torch.linalg.cholesky(covariances.double()).to(reference)
    diagonal = lower.diagonal(dim1=-2, dim2=-1)
    block = GaussianBlock(centers.to(reference), diagonal)
    with torch.no_grad():
        row, col = torch.tril_indices(
            centers.shape[-1], centers.shape[-1], -1, device=centers.device
        )
        block.shears.copy_(lower[:, row, col] / diagonal[:, row])
        block.amplitudes.copy_(amplitudes.to(reference))
    if any(not torch.isfinite(p).all() for p in block.parameters()):
        raise FloatingPointError("Split parameters exceed the field's numerical precision")
    return block




@torch.no_grad()
def _replace_rows(field, optimizer, removed, children):
    """Prepare topology and survivor Adam history before one coordinated commit.

    ``children`` maps a parent block to (centers, covariances, amplitudes, LR
    allocation). Newly created parameters have independent, empty Adam state.
    Unaffected blocks retain identities; survivor rows retain every Adam option,
    moment, and step. Empty blocks are removed, including the last block.
    """
    replacement, blocks, appended, child_groups = {}, [], [], []
    origins = {p: group for group in optimizer.param_groups for p in group["params"]}
    for bi, old in enumerate(field.blocks):
        rows = removed.get(bi, [])
        if not rows:
            blocks.append(old)
        else:
            if len(set(rows)) != len(rows) or any(
                type(r) is not int or not 0 <= r < len(old.centers) for r in rows
            ):
                raise ValueError("Removal rows must be unique existing integer indices")
            keep = torch.ones(len(old.centers), dtype=torch.bool, device=old.centers.device)
            keep[rows] = False
            new = None
            if keep.any():
                new = GaussianBlock(old.centers[keep], old.log_scales[keep].exp())
                new.nominal_scale, new.amplitude_lr_scale = (
                    old.nominal_scale,
                    old.amplitude_lr_scale,
                )
                for name, parameter in new.named_parameters():
                    previous = getattr(old, name)
                    parameter.copy_(previous[keep])
                    if previous.grad is not None:
                        parameter.grad = previous.grad[keep].clone()
                blocks.append(new)
            for name, parameter in old.named_parameters():
                replacement[parameter] = (None if new is None else getattr(new, name), keep)
        if bi in children:
            centers, covariance, amplitude, allocation = children[bi]
            child = _block_from_tensors(centers, covariance, amplitude, field.background)
            child.nominal_scale = old.nominal_scale
            child.amplitude_lr_scale = old.amplitude_lr_scale * allocation
            child.parameter_groups()
            appended.append(child)
            for names, factor in (
                (("amplitudes",), allocation),
                (("centers",), 1.0),
                (("log_scales", "shears"), 1.0),
            ):
                group = origins[getattr(old, names[0])]
                child_groups.append(
                    {
                        **deepcopy({k: v for k, v in group.items() if k != "params"}),
                        "params": [getattr(child, name) for name in names],
                        "lr": group["lr"] * factor,
                    }
                )
    if any(type(bi) is not int or not 0 <= bi < len(field.blocks) for bi in {*removed, *children}):
        raise ValueError("Topology change references an unknown block")
    state = defaultdict(dict)
    groups = []
    for group in optimizer.param_groups:
        parameters = []
        for old in group["params"]:
            new, keep = replacement.get(old, (old, None))
            if new is None:
                continue
            parameters.append(new)
            if old in optimizer.state:
                state[new] = {
                    key: value[keep].clone()
                    if keep is not None and isinstance(value, Tensor) and value.shape == old.shape
                    else deepcopy(value)
                    for key, value in optimizer.state[old].items()
                }
        if parameters:
            groups.append({**group, "params": parameters})
    field.blocks = nn.ModuleList([*blocks, *appended])
    optimizer.state = state
    optimizer.param_groups = groups
    for group in child_groups:
        optimizer.add_param_group(group)
