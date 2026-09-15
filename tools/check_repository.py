"""Check publishable source files, local documentation links and baseline profiles."""

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("gaussian_fwi", "configs", "docs", "tests", "tools", ".github", "models")
MODEL_NAMES = ("bp2004", "marmousi", "overthrust", "seam", "sigsbee2a")
REMOVED_PATHS = ("dynamic_refinement", "fwi_core", "gaussian_fwi/adaptation.py",
                 "gaussian_fwi/density.py", "gaussian_fwi/trials.py",
                 "gaussian_fwi/directions.py", "configs/direct.json", "configs/scheduled.json")
ROOT_FILES = ("README.md", "AGENTS.md", "CONTRIBUTING.md", "CITATION.cff", "LICENSE",
              "pyproject.toml", "requirements.txt", "requirements-dev.txt", "MANIFEST.in",
              "run.py", ".gitignore", ".gitattributes", ".editorconfig", ".python-version")
SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".cff", ".typed"}


def source_files():
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts)
    files.extend(ROOT / "models" / f"{name}.npy" for name in MODEL_NAMES)
    return sorted(set(files))


def check_models():
    """Verify preserved model bytes and array metadata against their provenance."""
    directory = ROOT / "models"
    expected = {f"{name}.npy" for name in MODEL_NAMES} | {"README.md", "manifest.json"}
    assert {p.name for p in directory.iterdir()} == expected, "Unexpected model files"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == "portable-gaussian-fwi-models-v1"
    assert set(manifest["models"]) == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        record = manifest["models"][name]
        path = directory / f"{name}.npy"
        assert not path.is_symlink() and record["prepared_file"] == path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["prepared_file_sha256"]
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == record["prepared_shape_zx"] == [70, 70]
        assert str(array.dtype) == record["prepared_dtype"] == "float32"
        assert np.isfinite(array).all() and record["velocity_units"] == "m/s"
        assert [float(array.min()), float(array.max())] == record["prepared_range_m_s"]
    return len(MODEL_NAMES)


def main():
    files = source_files()
    failures, link_count = [], 0
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    blocked = project["tool"]["fwi-verification"]["blocked-imports"]
    for name in REMOVED_PATHS:
        if (ROOT / name).exists():
            failures.append(f"Obsolete public method remains: {name}")
    for path in files:
        if path.is_symlink():
            failures.append(f"Symlink cannot be published: {path.relative_to(ROOT)}")
            continue
        if path.suffix == ".npy":
            continue  # Binary arrays are checked by identity and safe NumPy loading below.
        text = path.read_text()
        if "/" + "Users/" in text or re.search(r"[A-Z]:\\Users\\", text):
            failures.append(f"Machine-local path: {path.relative_to(ROOT)}")
        if re.search(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
                     r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", text):
            failures.append(f"Credential-like content: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(text)):
                modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                           else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(name == excluded or name.startswith(excluded + ".")
                       for name in modules for excluded in blocked):
                    failures.append(f"Removed/external dependency: {path.relative_to(ROOT)}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = target.strip("<>")
                if urlsplit(target).scheme or target.startswith("#"):
                    continue
                local = unquote(target.split("#", 1)[0])
                if local and not (path.parent / local).exists():
                    failures.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")
                link_count += 1
    assert project["project"]["name"] == "gaussian-fwi"
    assert set(project["tool"]["setuptools"]["packages"]) == {
        "gaussian_fwi", "gaussian_fwi.core",
    }
    assert {p.name for p in (ROOT / "configs").glob("*.json")} == {"baseline.json"}
    profile = json.loads((ROOT / "configs/baseline.json").read_text())
    assert set(profile) == {"field", "inversion", "regularization", "preprocessing"}
    config = profile["inversion"]
    assert profile["field"]["backend"] == "sparse_fused"
    assert config["steps_per_stage"] == 1000 and config["validation_interval"] == 25
    assert len(config["cutoffs"]) * config["steps_per_stage"] == 4000
    assert "refinement" in config
    assert not set(config).intersection({"levels", "adaptation", "density_control"})
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for directory in ("results/", "data/", "archive/", "experiments/"):
        assert directory in ignored, directory
    assert "/models/*" in ignored
    for name in MODEL_NAMES:
        assert f"!/models/{name}.npy" in ignored
    model_count = check_models()
    if failures:
        raise AssertionError("\n".join(failures))
    print(json.dumps({"passed": True, "source_files": len(files), "local_links": link_count,
                      "profile_total_updates": {"baseline": 4000},
                      "preserved_development_models": model_count,
                      "scientific_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
