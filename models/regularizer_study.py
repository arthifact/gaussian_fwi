"""Can a better smoothness prior on many parameters match a few parameters?

    python models/make_observations.py models/marmousi.npy data/marmousi_real.pt
    python models/regularizer_study.py --steps-per-stage 60

The capacity study shows a small population beating a large one at equal
waveform fit. The obvious objection is that a well-chosen prior on the large
population would do the same. TV is the wrong prior to test that with: it
favours piecewise-constant velocity, so it charges for a depth trend exactly as
it charges for a fault. Second-order TGV favours piecewise-affine velocity
instead, which is the prior the literature identifies as the right one for
velocity, and is the comparison this study runs.

Errors are reported inside and outside the depth the survey illuminates,
because a surface spread constrains neither equally.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import gaussian_fwi as gfwi


def roughness(field):
    return float(np.abs(np.diff(field, 2, axis=0)).mean()
                 + np.abs(np.diff(field, 2, axis=1)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observations", type=Path, default=Path("data/marmousi_real.pt"))
    parser.add_argument("--reference", type=Path, default=Path("models/marmousi.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/regularizer"))
    parser.add_argument("--large", type=int, default=32, help="Large seed lattice")
    parser.add_argument("--small", type=int, default=8, help="Reference small lattice")
    parser.add_argument("--tv", type=float, nargs="+", default=[1e-4, 1e-2])
    parser.add_argument("--tgv", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--steps-per-stage", type=int, default=60)
    parser.add_argument("--illuminated-m", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64)
    boundary = int(args.illuminated_m / args.spacing)
    args.output.mkdir(parents=True, exist_ok=True)

    def rmse(field, start=0, stop=None):
        stop = reference.shape[0] if stop is None else stop
        return float(np.sqrt(((field[start:stop] - reference[start:stop]) ** 2).mean()))

    cases = [(f"{args.small}x{args.small}, tv={args.tv[0]:g}", args.small,
              {"tv_weight": args.tv[0]})]
    cases += [(f"{args.large}x{args.large}, tv={w:g}", args.large, {"tv_weight": w})
              for w in args.tv]
    cases += [(f"{args.large}x{args.large}, tgv={w:g}", args.large,
               {"tv_weight": 0.0, "tgv_weight": w}) for w in args.tgv]

    rows = []
    for index, (label, lattice, regularization) in enumerate(cases):
        profile = gfwi.baseline(
            field={"bounds": [1400.0, 5000.0]}, seed_shape=[lattice, lattice],
            regularization=regularization, steps_per_stage=args.steps_per_stage,
            validation_interval=5,
            refinement={"warmup_steps": 10, "interval": 10, "stop_fraction": 0.5,
                        "minimum_age": 10})
        fit = gfwi.run(args.observations, args.output / f"case_{index:02d}",
                       profile=profile, threads=args.threads)
        velocity = fit.velocity.numpy()
        rows.append({
            "case": label, "parameters": fit.parameters,
            "rmse_all": round(rmse(velocity), 1),
            "rmse_illuminated": round(rmse(velocity, 0, boundary), 1),
            "rmse_unilluminated": round(rmse(velocity, boundary), 1),
            "roughness": round(roughness(velocity), 1),
            "train_final": [round(v, 5) for v in fit.waveform_losses["train"]["final"]],
        })
        print(f"  {label:<22} RMSE {rows[-1]['rmse_all']:>6}  "
              f"illuminated {rows[-1]['rmse_illuminated']:>6}  "
              f"below {rows[-1]['rmse_unilluminated']:>6}  "
              f"rough {rows[-1]['roughness']:>6}", flush=True)

    (args.output / "regularizer.json").write_text(json.dumps(rows, indent=2) + "\n")
    print()
    print(f"{'case':>22}{'params':>8}{'RMSE all':>10}{'illum.':>9}{'below':>9}{'rough':>8}")
    for row in rows:
        print(f"{row['case']:>22}{row['parameters']:>8}{row['rmse_all']:>10}"
              f"{row['rmse_illuminated']:>9}{row['rmse_unilluminated']:>9}"
              f"{row['roughness']:>8}")
    print(f"\n{args.output}/regularizer.json")


if __name__ == "__main__":
    main()
