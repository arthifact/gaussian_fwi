"""Fit the prepared Marmousi model with the accepted profile and score the result.

    python models/make_observations.py models/marmousi.npy data/marmousi.pt
    python models/run_marmousi.py

Traces are forward modelled with the same solver and grid the inversion uses, so
this is a software demonstration on synthetic data under favourable conditions,
not a field-data result. The reference array makes the data and scores the fit
afterwards; it never enters the inversion.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import gaussian_fwi as gfwi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=Path("data/marmousi.pt"))
    parser.add_argument("--reference", type=Path, default=Path("models/marmousi.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/marmousi_baseline_4000"))
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--steps-per-stage", type=int, default=None,
                        help="Declared deviation from the accepted 1000 updates per stage")
    args = parser.parse_args()

    profile = None if args.steps_per_stage is None else gfwi.baseline(
        steps_per_stage=args.steps_per_stage)

    start = time.perf_counter()
    fit = gfwi.run(args.observations, args.output, profile=profile, threads=args.threads)
    elapsed = time.perf_counter() - start

    reference = torch.from_numpy(np.load(args.reference)).to(fit.velocity)
    predictions = torch.load(args.output / "fit/predictions.pt", map_location="cpu",
                             weights_only=True)

    def rmse(field):
        return float((field.cpu() - reference.cpu()).square().mean().sqrt())

    summary = {
        "scope": "Synthetic demonstration: traces modelled with the fitting solver and grid.",
        "wall_seconds": elapsed,
        "status": fit.status,
        "independent_replay_exact": fit.verified,
        "gaussians": fit.gaussians,
        "parameters": fit.parameters,
        "solver_calls": fit.solver_calls,
        "velocity_rmse_initial_m_s": rmse(predictions["initial_velocity"]),
        "velocity_rmse_final_m_s": rmse(fit.velocity),
        "waveform_losses": fit.waveform_losses,
    }
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print()
    print(fit.summary())


if __name__ == "__main__":
    main()
