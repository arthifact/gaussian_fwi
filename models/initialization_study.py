"""Fit the same data from several starting models, at two capacities.

    python models/make_observations.py models/marmousi.npy data/marmousi_real.pt
    python models/initialization_study.py --steps-per-stage 60

Initialization here is otherwise deterministic: a regular lattice of
zero-amplitude components, so two runs of one profile are bitwise identical.
The starting model is varied instead, through the depth background, which is
what a practitioner actually has uncertainty about.

The question is not only which start gives the lowest error. It is whether the
fits converge towards each other. Spread is therefore reported before and after
fitting: if the final models are closer together than the initial ones, the data
are pulling different starts to a common answer, which is the behaviour a
well-conditioned parameterization should show.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import gaussian_fwi as gfwi

BACKGROUNDS = [[1500.0, 3000.0], [1500.0, 3500.0], [1600.0, 2600.0], [1800.0, 3200.0]]


def spread(fields):
    """Mean pairwise RMS difference between recovered models, in m/s."""
    pairs = list(itertools.combinations(fields, 2))
    return float(np.mean([np.sqrt(((a - b) ** 2).mean()) for a, b in pairs]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observations", type=Path, default=Path("data/marmousi_real.pt"))
    parser.add_argument("--reference", type=Path, default=Path("models/marmousi.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/initialization"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--steps-per-stage", type=int, default=60)
    parser.add_argument("--illuminated-m", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64)
    boundary = int(args.illuminated_m / args.spacing)
    args.output.mkdir(parents=True, exist_ok=True)

    def rmse(field, start=0, stop=None):
        stop = reference.shape[0] if stop is None else stop
        return float(np.sqrt(((field[start:stop] - reference[start:stop]) ** 2).mean()))

    summary = []
    for lattice in args.seeds:
        finals, initials, rows = [], [], []
        for index, background in enumerate(BACKGROUNDS):
            profile = gfwi.baseline(
                field={"bounds": [1400.0, 5000.0], "background": background},
                seed_shape=[lattice, lattice], steps_per_stage=args.steps_per_stage,
                validation_interval=5,
                refinement={"warmup_steps": 10, "interval": 10, "stop_fraction": 0.5,
                            "minimum_age": 10})
            destination = args.output / f"seeds_{lattice:02d}_start_{index}"
            fit = gfwi.run(args.observations, destination, profile=profile,
                           threads=args.threads)
            velocity = fit.velocity.numpy()
            start = torch.load(destination / "fit/predictions.pt", map_location="cpu",
                               weights_only=True)["initial_velocity"].numpy()
            finals.append(velocity)
            initials.append(start)
            rows.append({
                "background": background, "parameters": fit.parameters,
                "rmse_all": round(rmse(velocity), 1),
                "rmse_illuminated": round(rmse(velocity, 0, boundary), 1),
                "rmse_unilluminated": round(rmse(velocity, boundary), 1),
                "initial_rmse_all": round(rmse(start), 1),
                "train_final": [round(v, 5) for v in fit.waveform_losses["train"]["final"]],
            })
            print(f"  {lattice}x{lattice} start {background} -> "
                  f"RMSE {rows[-1]['rmse_all']} "
                  f"(illuminated {rows[-1]['rmse_illuminated']})", flush=True)

        summary.append({
            "seed_shape": f"{lattice}x{lattice}",
            "parameters": rows[0]["parameters"],
            "runs": rows,
            "initial_spread_m_s": round(spread(initials), 1),
            "final_spread_m_s": round(spread(finals), 1),
            "illuminated_spread_m_s": round(spread([f[:boundary] for f in finals]), 1),
            "rmse_mean": round(float(np.mean([r["rmse_illuminated"] for r in rows])), 1),
            "rmse_std": round(float(np.std([r["rmse_illuminated"] for r in rows])), 1),
        })

    (args.output / "initialization.json").write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(f"{'capacity':>10}{'params':>8}{'start spread':>14}{'final spread':>14}"
          f"{'illum. spread':>15}{'illum. RMSE':>13}")
    for row in summary:
        print(f"{row['seed_shape']:>10}{row['parameters']:>8}{row['initial_spread_m_s']:>14}"
              f"{row['final_spread_m_s']:>14}{row['illuminated_spread_m_s']:>15}"
              f"{row['rmse_mean']:>9} ±{row['rmse_std']:<4}")
    print(f"\n{args.output}/initialization.json")


if __name__ == "__main__":
    main()
