"""Build a clean wheel and verify it outside the checkout, without other FWI projects."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

WORKER = r"""
import importlib
import importlib.abc
import importlib.metadata
import json
import platform
import sys
import unittest
from pathlib import Path

installed, tests, output = (Path(value).resolve() for value in sys.argv[1:4])
settings = json.loads(sys.argv[4])
reference = Path(sys.argv[5]) if sys.argv[5] else None
blocked = tuple(settings['blocked-imports'])

class OtherProjectBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('Other FWI project is unavailable during isolation check: ' + fullname)

assert sys.flags.isolated
assert not any(name in sys.modules for name in blocked)
sys.meta_path.insert(0, OtherProjectBlocker())
sys.path.insert(0, str(installed))
for name in settings['packages']:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), name
suite = unittest.defaultTestLoader.discover(str(tests / 'unit'))
def identities(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from identities(test)
        else:
            yield test.id()
test_ids = list(identities(suite))
assert test_ids and len(test_ids) == len(set(test_ids)), 'Empty or duplicate test discovery'
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert (result.wasSuccessful() and not result.skipped and not result.expectedFailures
        and result.testsRun == len(test_ids)), 'Installed tests failed, skipped, or incomplete'
replays = []
if reference is not None:
    import torch
    from gaussian_fwi import GaussianField
    from _problems import problem
    torch.set_num_threads(2)
    for dimension in (2, 3):
        before = reference / str(dimension)
        field = GaussianField.load(before / 'field.pt')
        saved = torch.load(before / 'predictions.pt', weights_only=True)
        _, data, _ = problem(dimension)
        counts = dict(data.acquisition.counts)
        with torch.no_grad():
            velocity = field()
            torch.testing.assert_close(velocity, saved['final_velocity'], rtol=0, atol=0)
            prediction = data.acquisition.simulate(velocity)
            torch.testing.assert_close(prediction, saved['final_prediction'], rtol=0, atol=0)
            for block in field.blocks:
                torch.linalg.cholesky(block.covariance())
        work = {name: data.acquisition.counts[name] - counts[name] for name in counts}
        assert work == {'forward': 1, 'adjoint': 0}
        replays.append({'dimension': dimension, 'field_and_waveform_exact': True,
                        'additional_solver_calls': work,
                        'scope': 'Saved field compatibility; no old training trajectory replay'})
record = {
    'passed': True, 'tests': result.testsRun, 'isolated_python': True,
    'test_ids': test_ids,
    'all_packages_from_wheel': True, 'blocked_imports': list(blocked),
    'historical_field_replays': replays,
    'environment': {
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'packages': {name: importlib.metadata.version(name) for name in
                     (settings['project-name'], 'torch', 'deepwave', 'numpy', 'scipy', 'setuptools', 'ruff')},
        'torch_threads': importlib.import_module('torch').get_num_threads(),
    },
}
(output / 'installed_checks.json').write_text(json.dumps(record, indent=2) + '\n')
"""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(output: Path, reference: Path | None = None) -> dict:
    """Keep only the wheel, logs, and evidence; build copies stay in temporary storage."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = project["tool"]["setuptools"]["packages"]
    mapping = project["tool"]["setuptools"].get("package-dir", {})
    source_roots = []
    for name in sorted(packages, key=lambda value: (value.count("."), value)):
        path = Path(mapping.get(name, name.replace(".", "/")))
        if not any(path.is_relative_to(parent) for parent in source_roots):
            source_roots.append(path)
    source_hashes = {
        str(path.relative_to(ROOT)): digest(path)
        for directory in source_roots
        for path in sorted((ROOT / directory).rglob("*.py"))
    }
    evidence_inputs = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / ".python-version",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        *sorted((ROOT / "configs").glob("*.json")),
        ROOT / "run.py",
        Path(__file__),
        ROOT / "tools/validate.py",
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "examples").glob("*.py")),
        *sorted((ROOT / ".github/workflows").glob("*.yml")),
        *sorted((TESTS / "unit").glob("*.py")),
    ]
    verification_hashes = {
        str(path.relative_to(ROOT)): digest(path) for path in evidence_inputs if path.is_file()
    }
    output.mkdir(parents=True, exist_ok=False)
    settings = {**project["tool"]["fwi-verification"], "packages": packages,
                "project-name": project["project"]["name"]}
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="fwi-release-") as temporary:
        temporary = Path(temporary)
        source, installed, tests = (temporary / name for name in ("source", "installed", "tests"))
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "requirements.txt", "run.py", "LICENSE",
                     "CITATION.cff", "MANIFEST.in"):
            if (ROOT / name).is_file():
                shutil.copyfile(ROOT / name, source / name)
        for directory in source_roots:
            shutil.copytree(
                ROOT / directory,
                source / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for name in ("unit",):
            shutil.copytree(
                TESTS / name,
                tests / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        with (output / "build.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import setuptools.build_meta; setuptools.build_meta.build_wheel('../dist')",
                ],
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        wheels = list((temporary / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError("Expected exactly one wheel")
        wheel = output / wheels[0].name
        shutil.copyfile(wheels[0], wheel)
        distribution = project["project"]["name"].replace("-", "_")
        metadata = f"{distribution}-{project['project']['version']}.dist-info"
        allowed = {name.split(".")[0] for name in packages} | {metadata, "run.py"}
        installed.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or path.parts[0] not in allowed:
                    raise AssertionError(f"Unexpected wheel member: {name}")
            archive.extractall(installed)
        print(f"Testing isolated {settings['method']} wheel", flush=True)
        with (output / "tests.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    WORKER,
                    str(installed),
                    str(tests),
                    str(output),
                    json.dumps(settings),
                    str(reference) if reference is not None else "",
                ],
                cwd=temporary,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    result = json.loads((output / "installed_checks.json").read_text())
    result.update(
        method=settings["method"],
        version=project["project"]["version"],
        wheel=wheel.name,
        wheel_sha256=digest(wheel),
        source_sha256=source_hashes,
        verification_inputs_sha256=verification_hashes,
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "method", "tests", "historical_field_replays")}
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory under results/validation"
    )
    parser.add_argument("--reference", type=Path, help="Optional external 2D/3D saved-field references (not training restart)")
    args = parser.parse_args()
    output = args.output.resolve()
    results = ROOT / "results/validation"
    if output == results or not output.is_relative_to(results):
        parser.error("Choose a new directory under results/validation")
    reference = args.reference.resolve() if args.reference is not None else None
    if reference is not None and not all((reference / str(d)).is_dir() for d in (2, 3)):
        parser.error("Reference must contain both the 2 and 3 dimensional saved-field fixtures")
    verify(output, reference)
