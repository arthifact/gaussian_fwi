"""Gaussian topology edits with stable identities and coordinated Adam state.

Geometry proposals are independent of the wave equation and the decision to
apply them. The registry belongs to one field. Its portable state is bound to
the exact field values, so a restored identity map cannot silently label a
different reconstruction. Full training restart remains stage-boundary only.
"""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor

from ._optimizer import validate_adam
from .adaptation import GaussianCheckpoint, _admissible, _basis, binary_children
from .density import _replace_rows, long_axis_children
from .directions import directional_children
from .field import GaussianField


class EditRejected(Exception):
    """A finite proposal violates an explicit geometry or field-change limit."""


def _fingerprint(field: GaussianField) -> str:
    digest = hashlib.sha256(json.dumps(field.config(), sort_keys=True).encode())
    for block in field.blocks:
        digest.update(repr((block.nominal_scale, block.amplitude_lr_scale)).encode())
    for name, parameter in field.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TopologyEdit:
    """A proposal produced by :class:`GaussianTopology`; tensors are snapshots.

    ``parents`` are stable IDs, never mutable row indices. ``parent_values``
    detects an edit prepared before a subsequent change to its parents.
    """

    operation: str
    parents: tuple[int, ...]
    centers: Tensor
    covariances: Tensor
    amplitudes: Tensor
    allocation: float
    parent_values: tuple[tuple[Tensor, Tensor, Tensor], ...]


class GaussianTopology:
    """Prepare insertion, split, clone, merge, prune, reset, and relocation edits.

    Unaffected rows retain their optimizer history. Replacement kernels receive
    fresh histories; children inherit learning rates from their first parent.
    A merge lists the parent with greatest absolute ideal mass first. Literal
    cloning retains the original parent and copies its signed contribution;
    it therefore changes the field, unlike a coincident half-amplitude split.
    Relocation starts a fresh inactive kernel with explicit insertion rates.
    """

    def __init__(self, field: GaussianField, state: dict | None = None):
        self.field = field
        self.ids: list[list[int]] = []
        self.last_edit_steps: list[list[int]] = []
        self.next_id = 0
        self.blocks = ()
        if state is None:
            self.register_appended(step=0)
        else:
            self.load_state_dict(state)

    def _check(self) -> None:
        if len(self.blocks) != len(self.field.blocks) or any(
            a is not b for a, b in zip(self.blocks, self.field.blocks)
        ):
            raise ValueError("Field topology changed outside its registry")

    def _refresh(self) -> None:
        self.blocks = tuple(self.field.blocks)
        self._locations = {
            kernel_id: (bi, row)
            for bi, ids in enumerate(self.ids)
            for row, kernel_id in enumerate(ids)
        }

    def register_appended(self, *, step: int) -> list[int]:
        """Register externally appended seed blocks; existing rows cannot change."""
        if type(step) is not int or step < 0:
            raise ValueError("Edit step must be a nonnegative integer")
        if len(self.blocks) > len(self.field.blocks) or any(
            a is not b for a, b in zip(self.blocks, self.field.blocks)
        ):
            raise ValueError("Only appending complete seed blocks can be registered")
        added = []
        for block in self.field.blocks[len(self.blocks) :]:
            ids = list(range(self.next_id, self.next_id + len(block.centers)))
            self.next_id += len(ids)
            self.ids.append(ids)
            self.last_edit_steps.append([step] * len(ids))
            added.extend(ids)
        self._refresh()
        return added

    def locate(self, kernel_id: int) -> tuple[int, int]:
        if type(kernel_id) is not int:
            raise ValueError("Kernel IDs must be integers")
        if kernel_id not in self._locations:
            raise ValueError(f"Unknown or retired kernel ID: {kernel_id}")
        bi, row = self._locations[kernel_id]
        if (
            len(self.blocks) != len(self.field.blocks)
            or self.blocks[bi] is not self.field.blocks[bi]
        ):
            raise ValueError("Field topology changed outside its registry")
        return bi, row

    @torch.no_grad()
    def values(self, kernel_id: int) -> tuple[Tensor, Tensor, Tensor]:
        bi, row = self.locate(kernel_id)
        block = self.field.blocks[bi]
        d = self.field.grid.ndim
        lower = torch.eye(d, dtype=block.centers.dtype, device=block.centers.device)
        rr, cc = torch.tril_indices(d, d, offset=-1, device=lower.device)
        lower[rr, cc] = block.shears[row]
        lower = block.log_scales[row].exp()[:, None] * lower
        return tuple(
            t.detach().clone() for t in (block.centers[row], lower @ lower.T, block.amplitudes[row])
        )

    def _proposal(self, operation, parents, centers, covariances, amplitudes, allocation=1.0):
        reference = self.field.background
        return TopologyEdit(
            operation,
            tuple(parents),
            centers.detach().to(reference).clone(),
            covariances.detach().to(reference).clone(),
            amplitudes.detach().to(reference).clone(),
            float(allocation),
            tuple(self.values(i) for i in parents),
        )

    @torch.no_grad()
    def insert(self, center: Tensor, radius: float) -> TopologyEdit:
        """Create an inactive isotropic kernel without changing the current field."""
        if not math.isfinite(radius) or radius <= 0 or center.shape != (self.field.grid.ndim,):
            raise ValueError("Insertion requires one center and a positive physical radius")
        d = self.field.grid.ndim
        return self._proposal(
            "insert",
            (),
            center[None],
            torch.eye(d, device=center.device, dtype=center.dtype)[None] * radius**2,
            center.new_zeros(1),
        )

    @torch.no_grad()
    def split(
        self,
        kernel_id: int,
        *,
        fraction: float = 0.45,
        axis: int | None = None,
        geometry: str = "moment",
        direction: Tensor | None = None,
    ) -> TopologyEdit:
        """Prepare binary moment-preserving or Improved-GS long-axis children."""
        center, covariance, amplitude = self.values(kernel_id)
        if direction is not None and (axis is not None or geometry != "moment"):
            raise ValueError(
                "A physical direction requires moment geometry without a principal-axis index"
            )
        if geometry == "moment":
            if direction is None:
                axis = self.field.grid.ndim - 1 if axis is None else axis
                centers, matrices, amplitudes = binary_children(
                    center, covariance, amplitude, axis, fraction
                )
            else:
                centers, matrices, amplitudes = directional_children(
                    center, covariance, amplitude, direction, fraction
                )
            allocation = 1 / (2 * math.sqrt(1 - fraction**2))
        elif geometry == "long_axis":
            if axis is not None:
                raise ValueError("Long-axis geometry chooses its own principal axis")
            centers, matrices, amplitudes, allocation = long_axis_children(
                center, covariance, amplitude, fraction
            )
        else:
            raise ValueError("Unknown split geometry")
        return self._proposal("split", (kernel_id,), centers, matrices, amplitudes, allocation)

    @torch.no_grad()
    def relocate(self, kernel_id: int, center: Tensor, radius: float) -> TopologyEdit:
        """Replace a parent with an inactive kernel at a training-selected location.

        Removal and birth are one atomic proposal with zero net population change.
        The replacement receives a fresh ID and insertion learning rates/history.
        It is not a probability-preserving MCMC transition.
        """
        birth = self.insert(center, radius)
        return self._proposal(
            "relocate", (kernel_id,), birth.centers, birth.covariances, birth.amplitudes
        )

    @torch.no_grad()
    def clone(self, kernel_id: int) -> TopologyEdit:
        """Copy a kernel with fresh history, retaining the existing parent."""
        center, covariance, amplitude = self.values(kernel_id)
        return self._proposal(
            "clone", (kernel_id,), center[None], covariance[None], amplitude[None]
        )

    @torch.no_grad()
    def merge(self, first: int, second: int) -> TopologyEdit:
        """Match ideal signed moments; opposite signs require identical geometry.

        Moment matching does not preserve the field pointwise. Identical kernels
        combine by exact coefficient addition, including signed cancellation.
        """
        if first == second:
            raise ValueError("A merge requires two distinct kernels")
        parents = [first, second]
        values = [self.values(i) for i in parents]
        centers = torch.stack([v[0] for v in values]).double()
        matrices = torch.stack([v[1] for v in values]).double()
        amplitudes = torch.stack([v[2] for v in values]).double()
        volumes = torch.linalg.det(matrices).sqrt()
        masses = amplitudes * volumes
        if not torch.isfinite(masses).all() or not (volumes > 0).all():
            raise FloatingPointError("Nonfinite or singular parent moments")
        if abs(float(masses[1])) > abs(float(masses[0])):
            parents.reverse()
        identical = torch.equal(centers[0], centers[1]) and torch.equal(matrices[0], matrices[1])
        if identical:
            center, matrix, amplitude = centers[0], matrices[0], amplitudes.sum()
        else:
            if float(amplitudes[0] * amplitudes[1]) <= 0:
                raise EditRejected(
                    "Separated opposite-sign or inactive kernels cannot be moment-merged"
                )
            weights = masses.abs() / masses.abs().sum()
            center = (weights[:, None] * centers).sum(0)
            offsets = centers - center
            matrix = (
                weights[:, None, None] * (matrices + offsets[:, :, None] * offsets[:, None, :])
            ).sum(0)
            amplitude = masses.sum() / torch.linalg.det(matrix).sqrt()
        return self._proposal("merge", parents, center[None], matrix[None], amplitude[None])

    @torch.no_grad()
    def prune(self, kernel_id: int) -> TopologyEdit:
        """Propose removing one kernel; the physical field-change guard still applies."""
        d = self.field.grid.ndim
        empty = self.field.background.new_empty
        return self._proposal("prune", (kernel_id,), empty((0, d)), empty((0, d, d)), empty((0,)))

    @torch.no_grad()
    def reset(self, kernel_id: int, *, amplitude_cap: float) -> TopologyEdit:
        """Cap one signed amplitude, clearing only that row's amplitude moments."""
        if not math.isfinite(amplitude_cap) or amplitude_cap <= 0:
            raise ValueError("The reset amplitude cap must be finite and positive")
        center, matrix, amplitude = self.values(kernel_id)
        return self._proposal(
            "reset",
            (kernel_id,),
            center[None],
            matrix[None],
            amplitude.clamp(-amplitude_cap, amplitude_cap)[None],
        )

    def _validate_edit(self, edit: TopologyEdit) -> None:
        self._check()
        arity = {
            "insert": (0, 1),
            "split": (1, 2),
            "clone": (1, 1),
            "merge": (2, 1),
            "prune": (1, 0),
            "reset": (1, 1),
            "relocate": (1, 1),
        }
        if not isinstance(edit, TopologyEdit) or edit.operation not in arity:
            raise ValueError("Expected a supported topology proposal")
        parents, children = arity[edit.operation]
        d = self.field.grid.ndim
        if (
            len(edit.parents) != parents
            or len(set(edit.parents)) != parents
            or len(edit.parent_values) != parents
        ):
            raise ValueError("Invalid proposal parent identities")
        for value, shape in (
            (edit.centers, (children, d)),
            (edit.covariances, (children, d, d)),
            (edit.amplitudes, (children,)),
        ):
            if (
                value.shape != shape
                or value.dtype != self.field.background.dtype
                or value.device != self.field.background.device
            ):
                raise ValueError("Proposal tensors must match field shape, dtype, and device")
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite topology proposal")
        if not math.isfinite(edit.allocation) or edit.allocation <= 0:
            raise ValueError("Invalid child learning-rate allocation")
        for kernel_id, expected in zip(edit.parents, edit.parent_values):
            if (
                not isinstance(expected, tuple)
                or len(expected) != 3
                or any(not isinstance(t, Tensor) for t in expected)
            ):
                raise ValueError("Invalid parent parameter snapshot")
            if any(not torch.equal(a, b) for a, b in zip(self.values(kernel_id), expected)):
                raise ValueError("Proposal parents changed after preparation")
        if children:
            if (
                not torch.allclose(
                    edit.covariances, edit.covariances.transpose(-1, -2), rtol=1e-6, atol=1e-12
                )
                or not (torch.linalg.eigvalsh(edit.covariances.double()) > 0).all()
            ):
                raise ValueError("Proposal covariance must be symmetric positive definite")
            if not _admissible(self.field, edit.centers, edit.covariances):
                raise EditRejected("Proposed geometry exceeds the field constraints")
        if edit.operation in ("insert", "relocate") and bool(edit.amplitudes.any()):
            raise ValueError("Insertion must start with zero amplitude")
        if edit.operation in ("insert", "relocate") and not torch.equal(
            edit.covariances, torch.diag_embed(torch.diagonal(edit.covariances, dim1=-2, dim2=-1))
        ):
            raise ValueError(
                "Inserted covariance must be diagonal; geometry can subsequently adapt"
            )
        if edit.operation == "clone" and any(
            not torch.equal(expected, actual[0])
            for expected, actual in zip(
                edit.parent_values[0], (edit.centers, edit.covariances, edit.amplitudes)
            )
        ):
            raise ValueError("A clone must copy its parent's physical parameters")
        if edit.operation == "reset":
            center, matrix, amplitude = edit.parent_values[0]
            if (
                not torch.equal(center, edit.centers[0])
                or not torch.equal(matrix, edit.covariances[0])
                or abs(float(edit.amplitudes[0])) > abs(float(amplitude))
                or float(edit.amplitudes[0] * amplitude) < 0
            ):
                raise ValueError("Reset may only reduce an amplitude while preserving its sign")

    @torch.no_grad()
    def raw_change(self, edit: TopologyEdit) -> Tensor:
        """Evaluate the proposal's additive change with the production kernel."""
        self._validate_edit(edit)
        change = self.field.background.new_zeros(
            math.prod(self.field.grid.shape), dtype=torch.float64
        )
        for center, matrix, amplitude in zip(edit.centers, edit.covariances, edit.amplitudes):
            change += amplitude.double() * _basis(self.field, center.double(), matrix.double())
        if edit.operation not in ("insert", "clone"):
            for center, matrix, amplitude in edit.parent_values:
                change -= amplitude.double() * _basis(self.field, center.double(), matrix.double())
        if not torch.isfinite(change).all():
            raise FloatingPointError("Nonfinite proposed field change")
        return change

    @torch.no_grad()
    def apply(
        self,
        edit: TopologyEdit,
        optimizer: torch.optim.Adam,
        *,
        step: int,
        max_field_change: float,
        learning_rates: dict[str, float] | None = None,
        reference_velocity: Tensor | None = None,
    ) -> dict:
        """Commit one edit or restore the field, gradients, Adam, and identity map.

        ``reference_velocity`` can bind several edits to one cumulative field
        trust region. This is a geometry safeguard, not a waveform acceptance
        test or a guarantee about reconstruction error.
        """
        self._validate_edit(edit)
        if (
            type(step) is not int
            or step < 0
            or not math.isfinite(max_field_change)
            or max_field_change <= 0
        ):
            raise ValueError("A nonnegative edit step and positive field-change limit are required")
        validate_adam(self.field, optimizer)
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in self.field.parameters()
        ):
            raise FloatingPointError("Nonfinite gradient before a topology edit")
        reference = self.field().detach() if reference_velocity is None else reference_velocity
        if reference.shape != self.field.grid.shape or not torch.isfinite(reference).all():
            raise ValueError("Invalid field-change reference")
        before = GaussianCheckpoint()
        if not before.consider(0.0, 0, self.field, optimizer):
            raise FloatingPointError("Cannot snapshot a nonfinite field or optimizer")
        gradients = {
            p: None if p.grad is None else p.grad.detach().clone() for p in self.field.parameters()
        }
        saved = deepcopy((self.ids, self.last_edit_steps, self.next_id))
        locations = [self.locate(i) for i in edit.parents]
        if any(step < self.last_edit_steps[bi][row] for bi, row in locations):
            raise ValueError("An edit cannot precede its parent's last edit")
        count_before = self.field.count
        exact_copy = edit.operation == "clone" or (
            edit.operation == "merge"
            and torch.equal(edit.parent_values[0][0], edit.parent_values[1][0])
            and torch.equal(edit.parent_values[0][1], edit.parent_values[1][1])
        )
        coordinates = None
        if exact_copy:
            bi, row = locations[0]
            coordinates = {
                name: getattr(self.field.blocks[bi], name)[row].detach().clone()
                for name in ("log_scales", "shears")
            }
        try:
            removed: dict[int, list[int]] = {}
            if edit.operation not in ("insert", "clone", "reset"):
                for bi, row in locations:
                    removed.setdefault(bi, []).append(row)
            ids, ages = [], []
            for bi, (block_ids, block_ages) in enumerate(zip(self.ids, self.last_edit_steps)):
                keep = [r for r in range(len(block_ids)) if r not in removed.get(bi, ())]
                if keep:
                    ids.append([block_ids[r] for r in keep])
                    ages.append([block_ages[r] for r in keep])
            new_ids = []
            if edit.operation == "reset":
                bi, row = locations[0]
                block = self.field.blocks[bi]
                block.amplitudes[row] = edit.amplitudes[0]
                for key, value in optimizer.state.get(block.amplitudes, {}).items():
                    if key != "step":
                        value[row].zero_()
                ages[bi][row] = step
            else:
                children = {}
                if edit.operation in ("split", "clone", "merge"):
                    children[locations[0][0]] = (
                        edit.centers,
                        edit.covariances,
                        edit.amplitudes,
                        edit.allocation,
                    )
                if removed or children:
                    _replace_rows(self.field, optimizer, removed, children)
                    if coordinates is not None:
                        # Exact copies must not acquire a different geometry
                        # through an unnecessary covariance/Cholesky round trip.
                        for name, value in coordinates.items():
                            getattr(self.field.blocks[-1], name)[0].copy_(value)
                if edit.operation in ("insert", "relocate"):
                    if learning_rates is None:
                        raise ValueError("Insertion requires explicit learning rates")
                    scales = torch.diagonal(edit.covariances, dim1=-2, dim2=-1).sqrt()
                    block = self.field.add_gaussians(edit.centers, scales)
                    for group in block.parameter_groups(**learning_rates):
                        optimizer.add_param_group(group)
                if edit.operation != "prune":
                    new_ids = list(range(self.next_id, self.next_id + len(edit.centers)))
                    self.next_id += len(new_ids)
                    ids.append(new_ids)
                    ages.append([step] * len(new_ids))
                if edit.operation == "clone":
                    bi, row = locations[0]
                    ages[bi][row] = step
            self.ids, self.last_edit_steps = ids, ages
            self._refresh()
            validate_adam(self.field, optimizer)
            change = float((self.field() - reference).abs().max())
            if not math.isfinite(change):
                raise FloatingPointError("Nonfinite decoded field after a topology edit")
            if change > max_field_change:
                raise EditRejected("Cumulative sampled velocity change exceeds its trust region")
            return {
                "operation": edit.operation,
                "parents": list(edit.parents),
                "children": new_ids,
                "step": step,
                "before_count": count_before,
                "after_count": self.field.count,
                "maximum_velocity_change_m_s": change,
            }
        except Exception:
            before.restore(self.field, optimizer)
            for parameter, gradient in gradients.items():
                parameter.grad = gradient
            self.ids, self.last_edit_steps, self.next_id = saved
            self._refresh()
            raise

    def state_dict(self) -> dict:
        self._check()
        return {
            "format": "gaussian-fwi-topology-v1",
            "field_sha256": _fingerprint(self.field),
            "ids": deepcopy(self.ids),
            "last_edit_steps": deepcopy(self.last_edit_steps),
            "next_id": self.next_id,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {"format", "field_sha256", "ids", "last_edit_steps", "next_id"}
        if (
            not isinstance(state, dict)
            or set(state) != expected
            or state["format"] != "gaussian-fwi-topology-v1"
        ):
            raise ValueError("Invalid Gaussian topology checkpoint")
        if state["field_sha256"] != _fingerprint(self.field):
            raise ValueError("Topology checkpoint belongs to a different field")
        ids, ages, next_id = state["ids"], state["last_edit_steps"], state["next_id"]
        if (
            not isinstance(ids, list)
            or not isinstance(ages, list)
            or len(ids) != len(self.field.blocks)
            or len(ages) != len(ids)
        ):
            raise ValueError("Topology block counts do not match the field")
        for block, block_ids, block_ages in zip(self.field.blocks, ids, ages):
            if (
                not isinstance(block_ids, list)
                or not isinstance(block_ages, list)
                or len(block_ids) != len(block.centers)
                or len(block_ages) != len(block_ids)
            ):
                raise ValueError("Topology rows do not match the field")
        flat_ids = [i for block in ids for i in block]
        if (
            type(next_id) is not int
            or next_id < 0
            or any(type(i) is not int or not 0 <= i < next_id for i in flat_ids)
            or len(set(flat_ids)) != len(flat_ids)
        ):
            raise ValueError("Topology IDs must be unique nonnegative integers below next_id")
        if any(type(s) is not int or s < 0 for block in ages for s in block):
            raise ValueError("Topology edit ages must be nonnegative integers")
        self.ids, self.last_edit_steps, self.next_id = deepcopy((ids, ages, next_id))
        self._refresh()
