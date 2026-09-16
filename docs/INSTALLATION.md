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

### Native Windows first pass

The pinned stack also has native Windows wheels. A project-local environment
can run the CPU release gate and the separate CUDA derivative diagnostic without
installing WSL. In PowerShell, from this checkout:

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
    .venv/Scripts/python.exe -m pip install -r requirements-dev.txt
    .venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
    .venv/Scripts/python.exe tools/check_environment.py --require-cuda
    .venv/Scripts/python.exe tools/validate.py --release
    .venv/Scripts/python.exe tools/check_cuda.py --output results/cuda/first_check

The CUDA 13.0 build is an official option for the pinned PyTorch release.
[Pinned PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)
Use a fresh output name each time. The CUDA tool fails if the runtime or kernels
are unavailable and preserves a result record on numerical failure. It checks
2D/3D float32/float64 waveforms, every parameter-family gradient, one Adam update
and float64 finite differences. It does not establish large-model memory limits,
shot accumulation, GPU topology/restart or research performance. An additional
runtime acceptance tool checks CUDA float64 topology, state rollback,
completed-stage/update restart and test-data isolation using an observation bundle:

    .venv/Scripts/python.exe tools/check_cuda_runtime.py --observations data/observations.pt --output results/cuda/runtime_check

For an existing observation bundle:

    .venv/Scripts/python.exe run.py --observations data/observations.pt --config configs/baseline.json --output results/my_fit --device cuda --dtype float64

The explicit CUDA path requires float64: the supplied Marmousi developed-state
float32 gradient comparison failed its original tolerance. Float64 passed.
The runner enables deterministic operations, disables AMP/TF32, records actual
device/precision and caps its CUDA allocator at 70% of device memory. A requested
GPU must be available; there is no CPU fallback. Omitting `--device` preserves
the existing CPU default. Field evaluation, propagation and optimization use
CUDA; sparse neighbor search, checkpoint I/O and plotting still use the CPU.
Measure the full update on each acquisition before choosing its allowance.

For a bounded GPU segment with saved updates and explicit shot batching:

    .venv/Scripts/python.exe run.py --observations data/observations.pt --config configs/baseline.json --output results/segment_001 --device cuda --dtype float64 --shot-batch-size 3 --checkpoint-interval 100 --max-seconds 780 --diagnostics

If the result is paused, use its reported checkpoint in the next segment:

    .venv/Scripts/python.exe run.py --observations data/observations.pt --resume results/segment_001/fit/update_00001200.pt --output results/segment_002 --device cuda --dtype float64 --checkpoint-interval 100 --max-seconds 780 --diagnostics

Replace the example checkpoint with the path actually reported. Resume reuses
the original profile and shot grouping. The time limit is checked at completed
updates and leaves the declared full fitting budget unchanged. Output directories
must be new. Without `--accumulate-shots`, batching keeps all shot graphs until
the full backward pass. For finite-footprint acquisitions, add
`--accumulate-shots --shot-batch-size 1` to release each batch's saved wavefields
after backward while retaining one update over the complete training survey.

The pinned Deepwave 0.0.27 provides the accepted uncompressed storage options.
Explicit `--wavefield-storage cpu` moves intermediate wavefield storage to host
RAM; `--wavefield-storage disk` uses temporary files inside the fresh fit output.
Measure host-memory headroom or disk space and I/O cost for the intended workload.
Propagation remains on CUDA, with unchanged precision and temporal sampling.
Resume inherits accumulation and storage settings. See the [API](API.md) for
normalization, physical work accounting and scratch-directory behavior.

### Proposed WSL campaign environment

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

CUDA visibility is an environment check. Native Windows short-run acceptance
does not certify a different WSL stack or the planned larger acquisition.
Follow [GPU_CAMPAIGN.md](GPU_CAMPAIGN.md) for measured accumulation/storage
evidence and the remaining intended-scale research requirements.

WSL 2 defaults to half the host RAM, approximately 8 GB on this machine.
Measure both Windows and guest usage before changing that allocation or
offloading wavefields. [WSL memory settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

## Dependency changes

Keep the pinned scientific stack for initial acceptance. If a compatible wheel
is unavailable, record the proposed version change, declare numerical
compatibility and run the independent source/wheel checks before fitting.
Never silently relax a failed numerical tolerance.
