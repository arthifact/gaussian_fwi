"""Exercise every public path end to end and report pass/fail per feature.

    python models/make_observations.py models/marmousi.npy data/marmousi.pt
    python models/verify.py

A smoke test over the shipped interface: profiles, fitting, replay verification,
provenance, reload, pause/continue, restart identity, data separation, both
precisions, 2D and 3D, plotting and the command line. It is not a benchmark.
"""

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "tests/unit")
sys.path.insert(0, "models")

import gaussian_fwi as gfwi  # noqa: E402

RESULTS = []
FIT = None   # the reference fit, set by the first run check


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append(("PASS", name, detail or ""))
        except Exception as error:
            RESULTS.append(("FAIL", name, f"{type(error).__name__}: {error}"))
            traceback.print_exc(file=sys.stderr)
        return fn
    return wrap


def small_profile(**over):
    settings = dict(
        field={"background": [2100.0, 2600.0]},
        cutoffs=[10.0, 20.0], seed_shape=[2, 2], steps_per_stage=8, validation_interval=2,
        sampling_refinement_factors=[],
        refinement={"warmup_steps": 0, "interval": 1, "stop_fraction": 0.75, "minimum_age": 1,
                    "max_gaussians": 64, "max_growth": 2, "max_prunes": 2,
                    "split_extent_fraction": 1.0, "prune_amplitude": 0.0},
    )
    settings.update(over)
    return gfwi.baseline(**settings)


root = Path(tempfile.mkdtemp(prefix="gfwi_verify_"))
from _problems import problem  # noqa: E402

_, data, partitions = problem()
gfwi.save_observations(data, partitions, root / "obs.pt")
(root / "profile.json").write_text(json.dumps(small_profile(), indent=2))


@check("baseline() returns the accepted profile")
def _():
    profile = gfwi.baseline()
    canonical = json.loads(Path("configs/baseline.json").read_text())
    assert profile == canonical, "packaged baseline drifted from configs/baseline.json"
    return f"{profile['inversion']['steps_per_stage']} updates/stage, cutoffs {profile['inversion']['cutoffs']}"


@check("baseline() rejects unknown and invalid settings")
def _():
    for bad in (lambda: gfwi.baseline(nope=1), lambda: gfwi.baseline(cutoffs=[20.0, 4.0]),
                lambda: gfwi.baseline(steps_per_stage=200)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid setting was accepted")
    return "unknown key, decreasing cutoffs, impossible horizon all rejected"


@check("run() completes and self-verifies by independent replay")
def _():
    fit = gfwi.run(root / "obs.pt", root / "fit", profile=small_profile(), threads=2)
    assert fit.status == "complete"
    assert fit.verified is True, "independent propagation did not reproduce the saved prediction"
    global FIT
    FIT = fit
    return f"{fit.gaussians} Gaussians, replay exact, {fit.solver_calls}"


@check("Run accessors expose every recorded artifact")
def _():
    fit = FIT
    for name in ("result", "config", "provenance", "report", "history", "refinement"):
        assert isinstance(getattr(fit, name), (dict, list)), name
    assert fit.velocity.shape == data.acquisition.grid.shape
    assert torch.isfinite(fit.velocity).all()
    assert fit.parameters > 0 and "train" in fit.waveform_losses
    assert "Gaussian FWI run" in fit.summary()
    return f"result/config/provenance/report/history/refinement, velocity {tuple(fit.velocity.shape)}"


@check("provenance records source hashes, dependencies and data identity")
def _():
    p = FIT.provenance
    assert p["observation_content_identity"]["sha256"]
    assert p["observation_file_sha256"]
    assert "gaussian_fwi/inversion.py" in p["source_sha256"]
    assert set(p["dependencies"]) == {"torch", "deepwave", "numpy", "scipy"}
    return f"{len(p['source_sha256'])} source files, deps {p['dependencies']}"


@check("Run.open() reopens a finished directory identically")
def _():
    again = gfwi.Run.open(root / "fit")
    torch.testing.assert_close(again.velocity, FIT.velocity, rtol=0, atol=0)
    return f"reopened {again.gaussians} Gaussians, bitwise-identical velocity"


@check("GaussianField.load() reproduces the velocity bitwise")
def _():
    field = gfwi.GaussianField.load(root / "fit/fit/field.pt")
    with torch.no_grad():
        torch.testing.assert_close(field(), FIT.velocity, rtol=0, atol=0)
    return f"{field.count} components reloaded, velocity identical"


@check("field supports off-grid physical queries")
def _():
    field = gfwi.GaussianField.load(root / "fit/fit/field.pt")
    points = torch.tensor([[35.0, 45.0], [61.5, 22.25]], dtype=field.background.dtype)
    with torch.no_grad():
        values = field(points)
    assert values.shape == (2,) and torch.isfinite(values).all()
    return f"queried 2 off-grid points -> {[round(float(v), 1) for v in values]} m/s"


@check("pause at an update boundary, then continue into a new directory")
def _():
    paused = gfwi.run(root / "obs.pt", root / "paused", profile=small_profile(),
                      threads=2, max_updates=3, checkpoint_interval=1)
    assert paused.status == "paused"
    resumed = gfwi.continue_run(root / "paused" / paused.result["checkpoint"],
                                root / "obs.pt", root / "continued")
    assert resumed.status == "complete" and resumed.verified is True
    return f"paused at {paused.result['checkpoint']}, continued to {resumed.gaussians} Gaussians"


@check("paused run refuses to hand back a velocity")
def _():
    paused = gfwi.run(root / "obs.pt", root / "paused2", profile=small_profile(),
                      threads=2, max_updates=2, checkpoint_interval=1)
    try:
        paused.velocity
    except FileNotFoundError as error:
        return str(error)[:70] + "..."
    raise AssertionError("paused run returned a velocity")


@check("resume() continues from a completed stage checkpoint")
def _():
    # Resume with the same observation source the fit used; the checkpoint pins
    # the caller-supplied label as well as the recomputed content identity.
    same, _ = gfwi.load_observations(root / "obs.pt")
    field, report = gfwi.resume(root / "fit/fit/stage_00.pt", same, root / "stage_resume")
    assert report["stages"], "resumed report has no stages"
    return f"resumed stage_00 -> {field.count} components"


@check("resume() rejects observations with a different provenance label")
def _():
    try:
        gfwi.resume(root / "fit/fit/stage_00.pt", data, root / "mislabelled")
    except ValueError as error:
        return str(error)[:64]
    raise AssertionError("resume accepted a differently labelled observation set")


@check("existing output directories are never overwritten")
def _():
    try:
        gfwi.run(root / "obs.pt", root / "fit", profile=small_profile(), threads=2)
    except FileExistsError:
        return "FileExistsError raised; existing results untouched"
    raise AssertionError("an existing output directory was reused")


@check("observation bundles carrying a reference velocity are rejected")
def _():
    payload = torch.load(root / "obs.pt", weights_only=True)
    payload["reference_velocity"] = torch.zeros(12, 12)
    torch.save(payload, root / "leaky.pt")
    try:
        gfwi.load_observations(root / "leaky.pt")
    except ValueError as error:
        return str(error)[:70]
    raise AssertionError("a bundle containing a target velocity was accepted")


@check("test waveforms cannot influence the selected model")
def _():
    changed = torch.load(root / "obs.pt", weights_only=True)
    changed["traces"][:, changed["partitions"]["test"]] *= 100
    torch.save(changed, root / "tampered.pt")
    other = gfwi.run(root / "tampered.pt", root / "tampered_fit", profile=small_profile(), threads=2)
    torch.testing.assert_close(other.velocity, FIT.velocity, rtol=0, atol=0)
    return "100x test-trace perturbation left the selected model bitwise identical"


@check("float32 precision runs end to end")
def _():
    fit = gfwi.run(root / "obs.pt", root / "fit32", profile=small_profile(), threads=2,
                   dtype="float32")
    assert fit.status == "complete" and fit.verified is True
    return f"float32 fit verified, {fit.gaussians} Gaussians"


@check("3D fields fit end to end")
def _():
    from _problems import problem as build
    _, data3d, split3d = build(dimension=3)
    gfwi.save_observations(data3d, split3d, root / "obs3d.pt")
    profile = small_profile(seed_shape=[2, 2, 2])
    fit = gfwi.run(root / "obs3d.pt", root / "fit3d", profile=profile, threads=2)
    assert fit.status == "complete" and fit.verified is True
    return f"3D {tuple(fit.velocity.shape)} grid, {fit.gaussians} Gaussians, replay exact"


@check("Run.plot() renders a figure")
def _():
    FIT.plot(root / "velocity.png")
    size = (root / "velocity.png").stat().st_size
    assert size > 5000
    return f"{size} byte PNG"


@check("gaussian-fwi CLI runs from the installed entry point")
def _():
    done = subprocess.run([".venv/bin/gaussian-fwi", "--observations", str(root / "obs.pt"),
                           "--config", str(root / "profile.json"),
                           "--output", str(root / "cli_fit")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-400:]
    payload = json.loads(done.stdout)
    assert payload["independent_prediction_exact"] is True
    return f"exit 0, status {payload['status']}, replay exact, JSON on stdout"


@check("Marmousi bundle loads with the acquisition it was built with")
def _():
    observations, split = gfwi.load_observations("data/marmousi.pt")
    shots, receivers, samples = observations.traces.shape
    reference = np.load("models/marmousi.npy")
    assert observations.acquisition.grid.shape == reference.shape
    overlap = set(split["train"].tolist()) & set(split["validation"].tolist())
    assert not overlap, "train and validation receivers overlap"
    return (f"{shots} shots x {receivers} receivers x {samples} samples, "
            f"grid {observations.acquisition.grid.shape}, partitions disjoint")


print()
width = max(len(name) for _, name, _ in RESULTS)
for status, name, detail in RESULTS:
    mark = "ok  " if status == "PASS" else "FAIL"
    print(f"[{mark}] {name:<{width}}  {detail}")
failed = [name for status, name, _ in RESULTS if status == "FAIL"]
print()
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
if failed:
    print("FAILED:", ", ".join(failed))
sys.exit(1 if failed else 0)
