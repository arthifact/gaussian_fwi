"""Run the same numerical and packaging gates locally and in CI."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Also verify the isolated wheel")
    parser.add_argument("--output", type=Path, help="New directory inside results/validation")
    parser.add_argument("--reference", type=Path, help="Optional historical 2D/3D reference fits")
    args = parser.parse_args()
    if (args.output is not None or args.reference is not None) and not args.release:
        parser.error("--output and --reference require --release")

    commands = [
        [sys.executable, "tools/check_repository.py"],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "gaussian_fwi",
            "tests/unit",
            "tools",
            "run.py",
        ],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit", "-v"],
    ]
    if args.release:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / "results/validation" / f"validation_{stamp}"
        command = [sys.executable, "tools/verify_release.py", "--output", str(output)]
        if args.reference is not None:
            command.extend(("--reference", str(args.reference)))
        commands.append(command)
    for command in commands:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
