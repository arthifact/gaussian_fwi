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
from ._structure import GaussianCheckpoint, _admissible, _basis, _replace_rows
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
    """Stable Gaussian identities and atomic clone/split/prune transactions.

    Unaffected rows retain optimizer history. Children inherit parent learning
    rates and start with fresh Adam state. Clones copy the signed contribution;
    sampled splits replace one parent by two smaller children.
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
    def clone(self, kernel_id: int) -> TopologyEdit:
        """Copy a kernel with fresh history, retaining the existing parent."""
        center, covariance, amplitude = self.values(kernel_id)
        return self._proposal(
            "clone", (kernel_id,), center[None], covariance[None], amplitude[None]
        )


    @torch.no_grad()
    def prune(self, kernel_id: int) -> TopologyEdit:
        """Propose removing one kernel; the physical field-change guard still applies."""
        d = self.field.grid.ndim
        empty = self.field.background.new_empty
        return self._proposal("prune", (kernel_id,), empty((0, d)), empty((0, d, d)), empty((0,)))


    def _validate_edit(self, edit: TopologyEdit) -> None:
        self._check()
        arity = {
            "split": (1, 2),
            "clone": (1, 1),
            "prune": (1, 0),
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
        if edit.operation == "clone" and any(
            not torch.equal(expected, actual[0])
            for expected, actual in zip(
                edit.parent_values[0], (edit.centers, edit.covariances, edit.amplitudes)
            )
        ):
            raise ValueError("A clone must copy its parent's physical parameters")
    @torch.no_grad()
    def raw_change(self, edit: TopologyEdit) -> Tensor:
        """Evaluate the proposal's additive change with the production kernel."""
        self._validate_edit(edit)
        change = self.field.background.new_zeros(
            math.prod(self.field.grid.shape), dtype=torch.float64
        )
        for center, matrix, amplitude in zip(edit.centers, edit.covariances, edit.amplitudes):
            change += amplitude.double() * _basis(self.field, center.double(), matrix.double())
        if edit.operation != "clone":
            for center, matrix, amplitude in edit.parent_values:
                change -= amplitude.double() * _basis(self.field, center.double(), matrix.double())
        if not torch.isfinite(change).all():
            raise FloatingPointError("Nonfinite proposed field change")
        return change


    def state_dict(self) -> dict:
        self._check()
        return {
            "format": "gaussian-fwi-topology-v1",
            "field_sha256": _fingerprint(self.field),
            "ids": deepcopy(self.ids),
            "last_edit_steps": deepcopy(self.last_edit_steps),
            "next_id": self.next_id,
        }

    @torch.no_grad()
    def split(self, kernel_id: int, *, seed: int) -> TopologyEdit:
        """Two sampled children with scales divided by 1.6, following 3DGS.

        Children inherit the signed parent amplitude. This proposal does not
        preserve signed integral or covariance moments. Its actual sampled
        velocity change must pass the same guard as every other density edit.
        The local CPU generator leaves the caller's random stream untouched.
        """
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Split seed must be a nonnegative signed 64-bit integer")
        center, covariance, amplitude = self.values(kernel_id)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((2, self.field.grid.ndim), generator=generator, dtype=torch.float64)
        lower = torch.linalg.cholesky(covariance.detach().cpu().double())
        centers = (center.detach().cpu().double() + noise @ lower.T).to(center)
        matrices = (covariance / 1.6**2).expand(2, -1, -1).clone()
        return self._proposal("split", (kernel_id,), centers, matrices, amplitude.expand(2).clone())

    @torch.no_grad()
    def apply_many(self, edits, optimizer, *, step, max_field_change):
        """Commit a clone/split/prune batch with one field decode and atomic rollback.

        Stable survivor IDs, gradients, Adam moments/steps/options and group
        learning rates are retained. Every child starts with empty Adam state.
        All edits are measured against the same pre-event sampled velocity.
        """
        if not edits or type(step) is not int or step < 0:
            raise ValueError("A nonempty edit batch and nonnegative update are required")
        if not math.isfinite(max_field_change) or max_field_change <= 0:
            raise ValueError("Field-change limit must be finite and positive")
        parent_ids = []
        for edit in edits:
            self._validate_edit(edit)
            if edit.operation not in ("clone", "split", "prune"):
                raise ValueError("Density batches support clone, split and prune only")
            if edit.allocation != 1.0:
                raise ValueError("Density children inherit their parent's learning rates")
            parent_ids.extend(edit.parents)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("A parent can participate in only one edit per event")
        validate_adam(self.field, optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in self.field.parameters()):
            raise FloatingPointError("Nonfinite gradient before a topology batch")
        reference = self.field().detach().clone()
        if not torch.isfinite(reference).all():
            raise FloatingPointError("Nonfinite pre-event velocity")
        snapshot = GaussianCheckpoint()
        if not snapshot.consider(0.0, 0, self.field, optimizer):
            raise FloatingPointError("Cannot snapshot the field and optimizer")
        gradients = {p: None if p.grad is None else p.grad.detach().clone()
                     for p in self.field.parameters()}
        saved = deepcopy((self.ids, self.last_edit_steps, self.next_id))
        old_blocks = tuple(self.field.blocks)
        removed, grouped = {}, {}
        for index, edit in enumerate(edits):
            bi, row = self.locate(edit.parents[0])
            if step < self.last_edit_steps[bi][row]:
                raise ValueError("An edit cannot precede its parent")
            if edit.operation != "clone":
                removed.setdefault(bi, []).append(row)
            if edit.operation != "prune":
                grouped.setdefault(bi, []).append((index, row, edit))
        children = {bi: (torch.cat([e.centers for _, _, e in group]),
                         torch.cat([e.covariances for _, _, e in group]),
                         torch.cat([e.amplitudes for _, _, e in group]), 1.0)
                    for bi, group in grouped.items()}
        before_count = self.field.count
        try:
            ids, ages = [], []
            cloned = {e.parents[0] for e in edits if e.operation == "clone"}
            for bi, (block_ids, block_ages) in enumerate(zip(self.ids, self.last_edit_steps)):
                keep = [r for r in range(len(block_ids)) if r not in removed.get(bi, ())]
                if keep:
                    ids.append([block_ids[r] for r in keep])
                    ages.append([step if block_ids[r] in cloned else block_ages[r] for r in keep])
            survivor_blocks = len(ids)
            _replace_rows(self.field, optimizer, removed, children)
            events = [{"operation": e.operation, "parents": list(e.parents), "children": []}
                      for e in edits]
            for position, bi in enumerate(sorted(grouped)):
                block = self.field.blocks[survivor_blocks + position]
                block_ids, offset = [], 0
                for index, row, edit in grouped[bi]:
                    count = len(edit.centers)
                    new_ids = list(range(self.next_id, self.next_id + count))
                    self.next_id += count
                    events[index]["children"] = new_ids
                    block_ids.extend(new_ids)
                    if edit.operation == "clone":
                        # Preserve exact clone geometry without a Cholesky round trip.
                        for name in ("log_scales", "shears"):
                            getattr(block, name)[offset].copy_(getattr(old_blocks[bi], name)[row])
                    offset += count
                ids.append(block_ids)
                ages.append([step] * len(block_ids))
            self.ids, self.last_edit_steps = ids, ages
            self._refresh()
            validate_adam(self.field, optimizer)
            change = float((self.field() - reference).abs().max())
            if not math.isfinite(change):
                raise FloatingPointError("Nonfinite decoded field after density control")
            if change > max_field_change:
                raise EditRejected("Cumulative sampled velocity change exceeds its limit")
            return {"operations": events, "before_count": before_count,
                    "after_count": self.field.count, "maximum_velocity_change_m_s": change}
        except Exception:
            snapshot.restore(self.field, optimizer)
            for parameter, gradient in gradients.items():
                parameter.grad = gradient
            self.ids, self.last_edit_steps, self.next_id = saved
            self._refresh()
            raise

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
