"""Fit an explicit Gaussian FWI profile to a portable observation-only bundle.

The command line is a thin front end. Its behaviour lives in
:mod:`gaussian_fwi.study`, so Python callers reach the same recorded,
independently verified run through :func:`gaussian_fwi.run`.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from gaussian_fwi.study import configuration, execute, fit_observations, verify_fit

run_case = execute

__all__ = ["configuration", "fit_observations", "verify_fit", "run_case", "main"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True,
                        help="Observation-only .pt bundle")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--config", type=Path, help="Explicit baseline JSON profile")
    start.add_argument("--resume", type=Path, help="Runner checkpoint; reuse its unchanged profile")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--steps-per-stage", type=int,
                        help="Override and record the update allowance")
    parser.add_argument("--threads", type=int, default=2, help="Positive number of CPU threads")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Explicit compute device; CUDA requires --dtype float64")
    parser.add_argument("--dtype", choices=("float32", "float64"),
                        help="Explicit arithmetic precision; default preserves the bundle dtype")
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Save immutable completed-update snapshots")
    parser.add_argument("--max-updates", type=int, help="Pause after this many updates in this process")
    parser.add_argument("--max-seconds", type=float,
                        help="Pause at an update boundary after this interval")
    parser.add_argument("--diagnostics", action="store_true", help="Record gradient and update sizes")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.resume is not None and args.steps_per_stage is not None:
        parser.error("A resumed trajectory cannot change its optimization horizon")
    profile_path = args.config if args.config is not None else args.resume.parent.parent / "config.json"
    profile = json.loads(profile_path.read_text())
    if args.steps_per_stage is not None:
        profile["inversion"]["steps_per_stage"] = args.steps_per_stage
    result = run_case(args.observations, profile, args.output, device=args.device,
                      dtype=getattr(torch, args.dtype) if args.dtype else None,
                      resume_checkpoint=args.resume, checkpoint_interval=args.checkpoint_interval,
                      max_updates=args.max_updates, max_seconds=args.max_seconds,
                      diagnostics=args.diagnostics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
