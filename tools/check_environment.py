"""Inspect dependency versions and CPU/CUDA visibility without fitting a model."""

import argparse
import importlib.metadata
import json
import platform

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "deepwave", "numpy", "scipy")},
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda, "devices": [],
        "scope": "Environment inventory; no GPU numerical or performance acceptance",
    }
    if cuda:
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            report["devices"].append({
                "index": index, "name": device.name, "total_memory_bytes": device.total_memory,
                "compute_capability": [device.major, device.minor],
            })
    print(json.dumps(report, indent=2))
    if args.require_cuda and not cuda:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
