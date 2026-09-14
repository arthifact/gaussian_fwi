"""Validate the Adam layout required for reversible Gaussian topology changes."""

import math

import torch

from .field import GaussianField


def validate_adam(field: GaussianField, optimizer: torch.optim.Adam) -> None:
    """Reject unsupported groups or corrupt history before changing live state.

    Group order must match ``field.parameter_groups()`` so a portable field
    checkpoint can reconstruct the optimizer. Learning rates, Adam options,
    and empty history for newly inserted or split parameters are preserved.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("Adaptive state transfer currently supports torch.optim.Adam")
    expected = field.parameter_groups()
    if len(optimizer.param_groups) != len(expected):
        raise ValueError("Adaptation requires the field's canonical Adam groups")
    parameters = set(field.parameters())
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("Adaptation requires a fully trainable Gaussian field")
    if any(parameter not in parameters for parameter in optimizer.state):
        raise ValueError("Adam history contains parameters outside the field")
    for group, canonical in zip(optimizer.param_groups, expected):
        if (
            group.get("role") != canonical["role"]
            or len(group["params"]) != len(canonical["params"])
            or any(p is not q for p, q in zip(group["params"], canonical["params"]))
        ):
            raise ValueError("Adaptation requires the field's canonical Adam groups")
        if group.get("differentiable", False):
            raise ValueError("Differentiable Adam is unsupported for discrete topology changes")
        if any(
            not math.isfinite(group[name]) or group[name] < 0
            for name in ("lr", "eps", "weight_decay")
        ) or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("Invalid Adam hyperparameters")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            moments = {"exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                moments.add("max_exp_avg_sq")
            if set(state) != moments | {"step"}:
                raise ValueError("Incomplete or unsupported Adam history")
            step = state["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.ndim != 0
                or not torch.isfinite(step)
                or step < 0
                or float(step) != math.floor(float(step))
            ):
                raise ValueError("Adam step must be a finite nonnegative integer scalar")
            for name in moments:
                value = state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                ):
                    raise ValueError(
                        f"Adam {name} must match its parameter's shape, dtype and device"
                    )
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite Adam {name}")
                if name != "exp_avg" and (value < 0).any():
                    raise ValueError(f"Adam {name} must be nonnegative")
            if group["amsgrad"] and (state["max_exp_avg_sq"] < state["exp_avg_sq"]).any():
                raise ValueError("AMSGrad maximum cannot be smaller than the second moment")
