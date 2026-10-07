"""Does a weak pull toward the starting model stop deep-zone damage?

    python scripts/prior_test.py run
    python scripts/prior_test.py report

Three models whose deep zone the 2026-10-06 benchmark damaged most, SNR 10,
two noise seeds, both methods at three prior weights. Settings are the
benchmark's frozen ones; only the prior weight varies.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from benchmark import CONDITIONS, execute, prepare_starts

OUTPUT = Path("results/prior_test")
MODELS = ("seam", "sigsbee2a", "bp2004")
SEEDS = (0, 1)
WEIGHTS = (0.0, 0.01, 0.1)


def run():
    chosen = json.loads(Path("results/baseline/settings.json").read_text())
    snr = CONDITIONS["snr10"][0]
    prepare_starts([(m, "snr10", snr, s) for m in MODELS for s in SEEDS], 6, 2)
    jobs = []
    for model, seed, method, weight in itertools.product(MODELS, SEEDS, ("gauss", "pixel"),
                                                         WEIGHTS):
        label = f"{method} prior{weight:g}"
        jobs.append({"model": model, "condition": "snr10", "snr": snr, "seed": seed,
                     "settings": dict(chosen[method], prior_weight=weight), "label": label,
                     "steps": 60, "threads": 2,
                     "path": str(OUTPUT / f"{model}_s{seed}_{label.replace(' ', '_')}"
                                 .replace(".", "p"))})
    execute(jobs, 7)


def report():
    rows = [json.loads(p.read_text()) for p in sorted(OUTPUT.glob("*.json"))]
    print("| Model | Method | RMSE | <300 m | >300 m | start >300 m | Gaussians | stops |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for label in sorted({r["label"] for r in rows}):
            group = [r for r in rows if (r["model"], r["label"]) == (model, label)]
            if not group:
                continue

            def mean(key):
                return np.mean([r[key] for r in group])

            stops = sum(not e["growing"] for r in group for e in r["edits"])
            count = (f"{np.mean([r['parameters'] // 6 for r in group]):.0f}"
                     if label.startswith("gauss") else "-")
            print(f"| {model} | {label} | {mean('rmse'):.0f} | {mean('rmse_illuminated'):.0f} | "
                  f"{mean('rmse_deep'):.0f} | "
                  f"{np.mean([r['start']['rmse_deep'] for r in group]):.0f} | {count} | "
                  f"{stops} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
