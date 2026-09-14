"""Check publishable source files, local documentation links and baseline profiles."""

import ast
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("dynamic_refinement", "gaussian_fwi", "fwi_core", "configs", "docs", "tests",
               "tools", ".github")
ROOT_FILES = ("README.md", "AGENTS.md", "CONTRIBUTING.md", "CITATION.cff", "LICENSE",
              "pyproject.toml", "requirements.txt", "requirements-dev.txt", "MANIFEST.in",
              "run.py", ".gitignore", ".gitattributes", ".editorconfig", ".python-version")
SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".cff", ".typed"}


def source_files():
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts)
    return sorted(set(files))


def main():
    files = source_files()
    failures, link_count = [], 0
    for path in files:
        if path.is_symlink():
            failures.append(f"Symlink cannot be published: {path.relative_to(ROOT)}")
            continue
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
                if any(name.split(".")[0] == "fwi_experiments" for name in modules):
                    failures.append(f"Experimental dependency: {path.relative_to(ROOT)}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = target.strip("<>")
                if urlsplit(target).scheme or target.startswith("#"):
                    continue
                local = unquote(target.split("#", 1)[0])
                if local and not (path.parent / local).exists():
                    failures.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")
                link_count += 1
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "gaussian-fwi"
    assert set(project["tool"]["setuptools"]["packages"]) == {
        "gaussian_fwi", "dynamic_refinement", "fwi_core",
    }
    profiles = {}
    for method in ("direct", "scheduled"):
        profile = json.loads((ROOT / "configs" / f"{method}.json").read_text())
        config = profile["inversion"]
        assert profile["method"] == method and profile["field"]["backend"] == "sparse_fused"
        assert config["steps_per_stage"] == 1000 and config["validation_interval"] == 25
        profiles[method] = len(config["cutoffs"]) * config["steps_per_stage"]
        assert profiles[method] == 4000
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for directory in ("results/", "data/", "models/", "archive/", "experiments/"):
        assert directory in ignored, directory
    if failures:
        raise AssertionError("\n".join(failures))
    print(json.dumps({"passed": True, "source_files": len(files), "local_links": link_count,
                      "profile_total_updates": profiles, "scientific_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
