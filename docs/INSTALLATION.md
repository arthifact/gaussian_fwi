# Installation

Use Python 3.12 and the pinned dependencies. Record the exact interpreter,
packages and platform with each run. Install from a source checkout; this
repository does not publish or require a package on PyPI.

## CPU development

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -e .
    python tools/check_environment.py
    python tools/validate.py --release

The requirements include Matplotlib for downstream figure generation. Core
library installation uses the dependencies declared in pyproject.toml.

For a CPU-only Linux environment, install the matching PyTorch CPU wheel first:

    python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements-dev.txt
    python -m pip install --no-deps --no-build-isolation -e .

## Windows 11 and the planned RTX 4060 campaign

The reference campaign machine has a Ryzen 9, 16 GB system RAM and an RTX 4060
with 8 GB VRAM. The proposed environment is Ubuntu under WSL 2. NVIDIA's guide
describes the supported Windows driver route; do not install a Linux NVIDIA
display driver inside WSL.
[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)

In PowerShell, inspect the current setup:

    nvidia-smi
    wsl --status
    wsl --list --verbose

If WSL is not installed, follow the
[Microsoft installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
Create a fresh Linux virtual environment inside WSL. Choose a CUDA-enabled
PyTorch 2.13.0 wheel compatible with the installed driver using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then
install the pinned requirements and this project. Do not copy a macOS or
Windows virtual environment into Linux.

    python -m pip install -r requirements-dev.txt
    python -m pip install --no-deps --no-build-isolation -e .
    python tools/check_environment.py --require-cuda

Deepwave can use prebuilt wheels; a source build additionally requires CMake,
a C compiler and, for CUDA support, a CUDA compiler.
[Deepwave installation](https://ausargeo.com/deepwave/)

CUDA visibility is an environment check. Acceptance still requires actual
field/propagation gradients, batching and restart checks on the GPU.
The current observation-only CLI executes CPU fits; do not launch it expecting
automatic GPU acceleration. Follow [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md) for the
remaining implementation and measurement steps.

WSL 2 defaults to half the host RAM, approximately 8 GB on this machine.
Measure both Windows and guest usage before changing that allocation or
offloading wavefields. [WSL memory settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

## Dependency changes

Keep the pinned scientific stack for initial acceptance. If a compatible wheel
is unavailable, record the proposed version change, declare numerical
compatibility and run the independent source/wheel checks before fitting.
Never silently relax a failed numerical tolerance.
