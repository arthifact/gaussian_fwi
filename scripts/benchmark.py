"""Adaptive Gaussians against pixel FWI, on every model and data condition.

    python scripts/benchmark.py tune      # choose settings on Marmousi only
    python scripts/benchmark.py run       # frozen settings, all models and conditions
    python scripts/benchmark.py report    # table and figure

Tuning sees one model (Marmousi, default condition, noise seed 0) and selects
each method's settings by the waveform misfit on held-out receivers. The true
model scores fits afterwards and is never used to choose anything. Settings
selected there are frozen in ``results/baseline/settings.json`` and reused
unchanged on all five models, so four of them are untouched by tuning.

Every fit uses the same solver, data, frequency bands, steps and loss. The
methods differ only in how velocity is represented.
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MODELS = ("marmousi", "overthrust", "bp2004", "seam", "sigsbee2a")
# name: (signal-to-noise or None, noise seeds)
CONDITIONS = {"snr10": (10.0, (0, 1, 2)), "snr5": (5.0, (0, 1, 2)), "noiseless": (None, (0,))}
OUTPUT = Path("results/baseline")
# Illumination-weighted pull toward the start, applied to the Gaussian and the
# regularized pixel fits; plain pixel FWI stays unregularized as a reference.
# The weight was chosen on SEAM, Sigsbee2A and BP 2004 against their true
# models (scripts/prior_test.py), so Marmousi and Overthrust are the clean
# check of it.
PRIOR = {"prior_weight": 0.01, "prior_map": True}
RUNS = "runs_v4"
STARTS = "starts_v3"        # fine first grid; "starts" holds the coarse-grid ones

GAUSS_GRID = [{"method": "gauss", "learning_rates": {"amplitude": lr},
               "adapt": {"split_factor": split}}
              for lr, split in itertools.product((5.0, 10.0, 20.0), (1.0, 1.5))]
PIXEL_GRID = [{"method": "pixel", "learning_rates": {"change": lr}, "tv_weight": tv}
              for lr, tv in itertools.product((2.0, 5.0, 10.0, 20.0), (0.0, 0.03, 0.3))]


def run_one(job):
    """Fit one (model, condition, seed, settings) and save scores and velocity."""
    import torch

    import gaussian_fwi as g

    torch.set_num_threads(job["threads"])
    path = Path(job["path"])
    if path.with_suffix(".json").exists():
        return json.loads(path.with_suffix(".json").read_text())
    reference = torch.from_numpy(np.load(f"models/{job['model']}.npy"))
    survey, observed = g.synthetic(reference, g.DataConfig(
        signal_to_noise=job["snr"], seed=job["seed"]))
    start = starting_model(job, survey, observed, reference.shape)
    settings = job["settings"]
    prior_map = (g.prior_weights(survey, start) if settings.get("prior_map") else None)
    if settings["method"] == "gauss":
        model = g.GaussianField(start, survey.spacing)
        adapter = g.Adapter(g.AdaptConfig(**settings.get("adapt", {})))
    else:
        model, adapter = g.PixelField(start, survey.spacing), None
    result = g.fit(model, survey, observed, steps=job["steps"], adapter=adapter,
                   learning_rates=settings.get("learning_rates"),
                   tv_weight=settings.get("tv_weight", 0.0),
                   prior_weight=settings.get("prior_weight", 0.0), prior_map=prior_map,
                   validation=g.holdout(observed.shape[1]))
    velocity = model().detach()
    row = {key: job[key] for key in ("model", "condition", "seed", "label")}
    row.update(settings=settings, **result["history"][-1], wall_seconds=result["seconds"],
               edits=result["edits"], **g.velocity_errors(velocity, reference),
               start={key: value for key, value in g.velocity_errors(start, reference).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), velocity.numpy())
    path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def starting_model(job, survey, observed, shape):
    """The data-estimated start for this dataset, computed once and shared by all methods."""
    import gaussian_fwi as g

    path = OUTPUT / STARTS / f"{job['model']}_{job['condition']}_s{job['seed']}.json"
    if path.exists():
        info = json.loads(path.read_text())
    else:
        _, info = g.estimate_start(survey, observed, shape,
                                   validation=g.holdout(observed.shape[1]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2) + "\n")
    return g.linear_start(shape, info["top_m_s"], info["bottom_m_s"])


def prepare_starts(datasets, workers, threads):
    """Estimate every dataset's start in parallel before any fit needs it."""
    jobs = [{"model": m, "condition": c, "snr": snr, "seed": s, "threads": threads}
            for m, c, snr, s in datasets]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_start_job, jobs))


def _start_job(job):
    import torch

    import gaussian_fwi as g

    torch.set_num_threads(job["threads"])
    reference = torch.from_numpy(np.load(f"models/{job['model']}.npy"))
    survey, observed = g.synthetic(reference, g.DataConfig(
        signal_to_noise=job["snr"], seed=job["seed"]))
    starting_model(job, survey, observed, reference.shape)


def execute(jobs, workers):
    with ProcessPoolExecutor(workers) as pool:
        for row in pool.map(run_one, jobs):
            print(f"{row['model']:10s} {row['condition']:9s} s{row['seed']} {row['label']:22s} "
                  f"rmse {row['rmse']:6.1f} ({row['rmse_illuminated']:5.1f}/"
                  f"{row['rmse_deep']:5.1f})  val {row['validation_misfit']:.4f}  "
                  f"params {row['parameters']}  {row['wall_seconds']:.0f}s", flush=True)


def label(settings):
    if settings["method"] == "gauss":
        return (f"gauss lr{settings['learning_rates']['amplitude']:g} "
                f"split{settings['adapt']['split_factor']:g}")
    return f"pixel lr{settings['learning_rates']['change']:g} tv{settings['tv_weight']:g}"


def slug(text):
    """File-safe name; a dot would be read as a file extension."""
    return text.replace(" ", "_").replace(".", "p")


def tune(args):
    prepare_starts([("marmousi", "snr10", 10.0, 0)], 1, args.threads * args.workers)
    jobs = [{"model": "marmousi", "condition": "snr10", "snr": 10.0, "seed": 0,
             "settings": s, "label": label(s), "steps": args.steps, "threads": args.threads,
             "path": str(OUTPUT / "tune" / slug(label(s)))}
            for s in GAUSS_GRID + PIXEL_GRID]
    execute(jobs, args.workers)
    chosen = {}
    for method in ("gauss", "pixel"):
        rows = [json.loads(p.read_text()) for p in sorted((OUTPUT / "tune").glob("*.json"))]
        rows = [r for r in rows if r["settings"]["method"] == method]
        best = min(rows, key=lambda r: r["validation_misfit"])
        chosen[method] = best["settings"]
        print(f"selected {method}: {best['label']} (validation misfit "
              f"{best['validation_misfit']:.4f})")
    # The pixel reference is also run without regularization, as plain FWI.
    plain = dict(chosen["pixel"], tv_weight=0.0)
    chosen["pixel_plain"] = plain
    (OUTPUT / "settings.json").write_text(json.dumps(chosen, indent=2) + "\n")


def run(args):
    chosen = json.loads((OUTPUT / "settings.json").read_text())
    prepare_starts([(m, c, snr, s) for m, (c, (snr, seeds)) in
                    itertools.product(MODELS, CONDITIONS.items()) for s in seeds],
                   args.workers, args.threads)
    jobs = []
    for model, (condition, (snr, seeds)) in itertools.product(MODELS, CONDITIONS.items()):
        for seed, (name, settings) in itertools.product(seeds, chosen.items()):
            if name != "pixel_plain":
                settings = dict(settings, **PRIOR)
            jobs.append({"model": model, "condition": condition, "snr": snr, "seed": seed,
                         "settings": settings, "label": name, "steps": args.steps,
                         "threads": args.threads,
                         "path": str(OUTPUT / RUNS / f"{model}_{condition}_s{seed}_{name}")})
    execute(jobs, args.workers)


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted((OUTPUT / RUNS).glob("*.json"))]
    methods = sorted({r["label"] for r in rows})
    keys = ("rmse", "rmse_illuminated", "rmse_deep", "roughness")
    lines = ["| Model | Condition | Method | RMSE | RMSE <300 m | RMSE >300 m | Roughness | "
             "Params | Held-out misfit |", "|" + "---|" * 9]
    summary = {}
    for model, condition in itertools.product(MODELS, CONDITIONS):
        group = [r for r in rows if (r["model"], r["condition"]) == (model, condition)]
        if group:
            start = {k: np.mean([r["start"][k] for r in group]) for k in keys}
            lines.append(f"| {model} | {condition} | start (from data) | "
                         + " | ".join(f"{start[k]:.0f}" for k in keys) + " | 2 | — |")
        for method in methods:
            group = [r for r in rows if (r["model"], r["condition"], r["label"])
                     == (model, condition, method)]
            if not group:
                continue
            mean = {k: np.mean([r[k] for r in group]) for k in keys}
            spread = {k: np.std([r[k] for r in group]) for k in keys}
            summary[(model, condition, method)] = mean
            cells = [f"{mean[k]:.0f} ± {spread[k]:.0f}" if len(group) > 1 else f"{mean[k]:.0f}"
                     for k in keys]
            lines.append(f"| {model} | {condition} | {method} | " + " | ".join(cells)
                         + f" | {np.mean([r['parameters'] for r in group]):.0f} | "
                         f"{np.mean([r['validation_misfit'] for r in group]):.3f} |")
    wins = {m: 0 for m in methods}
    for model, condition in itertools.product(MODELS, CONDITIONS):
        cases = {m: summary.get((model, condition, m)) for m in methods}
        if all(cases.values()):
            wins[min(cases, key=lambda m: cases[m]["rmse"])] += 1
    lines += ["", "Lowest overall RMSE, out of model x condition cells: "
              + ", ".join(f"{m} {n}" for m, n in wins.items())]
    for method in methods:
        own = [r for r in rows if r["label"] == method]
        worse = [r for r in own if r["rmse_deep"] > r["start"]["rmse_deep"] + 1]
        lines.append(f"{method}: deep zone worse than its start in {len(worse)} of {len(own)} fits"
                     + (": " + ", ".join(sorted({f"{r['model']}/{r['condition']}" for r in worse}))
                        if worse else ""))
    text = "\n".join(lines)
    (OUTPUT / f"report_{RUNS}.md").write_text(text + "\n")
    print(text)
    figure(rows, methods)


def figure(rows, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import gaussian_fwi as g

    fig, axes = plt.subplots(len(MODELS), 2 + len(methods), figsize=(2.6 * (2 + len(methods)),
                             2.5 * len(MODELS)), layout="constrained")
    for i, model in enumerate(MODELS):
        reference = np.load(f"models/{model}.npy")
        panels = [("Reference", reference), ("Start", g.linear_start(reference.shape).numpy())]
        for method in methods:
            path = OUTPUT / RUNS / f"{model}_snr10_s0_{method}.npy"
            if path.exists():
                row = json.loads(path.with_suffix(".json").read_text())
                panels.append((f"{method}: {row['rmse']:.0f} m/s", np.load(path)))
        for axis, (title, field) in zip(axes[i], panels):
            axis.imshow(field, cmap="viridis", vmin=1500, vmax=4500, extent=(0, 690, 690, 0))
            axis.set_title(f"{model}\n{title}" if title == "Reference" else title, fontsize=8)
            axis.axhline(300, color="w", lw=.8, ls="--")
            axis.tick_params(labelsize=6)
    fig.savefig(OUTPUT / f"baseline_{RUNS}.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("tune", "run", "report"))
    parser.add_argument("--steps", type=int, default=60, help="Updates per frequency band")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--threads", type=int, default=2, help="Torch threads per worker")
    args = parser.parse_args()
    {"tune": tune, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
