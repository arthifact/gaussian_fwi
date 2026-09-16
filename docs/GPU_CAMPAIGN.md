# RTX 4060 campaign

Reference machine: Windows 11, Ryzen 9, 16 GB host RAM, RTX 4060 with 8 GB VRAM,
available for several days continuously. Follow [INSTALLATION.md](INSTALLATION.md)
for the proposed Ubuntu/WSL 2 environment.

## Implementation status

The single observation-only runner supports explicit CUDA float64 execution
with deterministic algorithms, AMP/TF32 disabled and a 70% allocator cap. CPU
defaults and file formats remain compatible. The sparse decoder retains CPU
cKDTree search and NumPy transfers.

Native Windows acceptance on the RTX 4060 includes tiny 2D/3D derivatives,
full-acquisition Marmousi gradient/Adam comparisons, GPU topology/state/rollback,
completed-stage/update restart and test-data isolation. See
[ENGINEERING_SPEC.md](ENGINEERING_SPEC.md), `tools/check_cuda.py` and
`tools/check_cuda_runtime.py`. This accepts short development runs at the tested
scale. It does not establish a larger research campaign. Full-gradient unequal
shot accumulation is now checked independently, along with optional uncompressed
host/disk wavefield storage. A 256x512 float64 software fixture with 20 shots,
4,001 samples, 256 receivers and 8,192 fragmented components completed one
accumulated update and density event using disk storage within the declared
GPU memory margin. Device-only storage failed the single-shot memory limit;
initial host-storage attempts lacked sufficient free RAM. See the dated
engineering evidence for all attempts, measured costs and limitations.
Five additional small development fits completed 40,000 total updates with
independent post-fit audits. Longer Marmousi fits reduced waveform loss while
worsening evaluation-only velocity RMSE; longer Overthrust fits improved both.
All five retain unfinished-convergence indicators. Historical and same-source
long-prefix state comparisons failed their strict tolerances, although local
gradient and short restart checks passed. These results do not justify a
larger-scale superiority or bitwise-repeatability claim.

The post-campaign source compacts filtered targets during construction and releases
unused CUDA allocator blocks once after accumulated-run initialization.
Fresh 72-test source and wheel gates and CUDA disk-state checks passed. These
post-campaign memory changes are archived separately from the research source.
The subsequent 20-shot analytic fixture completed 12 updates and final replay
across four processes, with exactly 360 fitting forwards, 240 adjoints and
20 verification forwards. Maximum reserved memory was 5.39 GiB and the minimum
recorded post-segment free memory was 1.39 GiB. Its first four updates used the
archived research source; the later processes used the memory correction.
This accepts that source-transition recovery case, not an uninterrupted
trajectory, fragmented-field sustained run or geological convergence. The
same-source large prefix probe was omitted when its source prerequisite failed.
The I/O probe also established that virtual-environment launcher counters miss
worker writes; instrument the actual worker before budgeting wavefield traffic.
Optional completed-update snapshots support bounded execution segments. Explicit
shot batching groups compatible quadrature shapes and has passed serial/batched
gradient, unequal-batch count and interrupted CUDA checks on the development
acquisition. It improves throughput but does not reduce retained wavefield memory.

The final closeout adds invocation-owned disk scratch cleanup through completion,
pause and ordinary exceptions. Fresh source and isolated-wheel gates each pass
76 tests, with native return code zero; CUDA disk restart/state/isolation and
retained-graph failure/recovery checks also pass. The [acceptance matrix](BASELINE_ACCEPTANCE.md)
defines the frozen support contract. These checks close the declared engineering
change; larger future applications receive their own resource budgets.

## Memory strategy

Start with one GPU experiment worker and one shot per batch. Accumulate the full
training gradient before one optimizer step, retaining the global loss
normalization and training-derived trace weights. Weight unequal batches
correctly, add regularization once and rank topology from the complete gradient.
Deepwave documents this memory-reduction approach.
[Shot accumulation](https://ausargeo.com/deepwave/example_rtm)

The supplied Marmousi float32 developed-state gradient comparison failed its
original tolerance; float64 passed. Use float64 for the current CUDA runner.
Disable AMP, TF32, lossy compression and reduced gradient sampling. CPU/GPU reductions can differ;
predeclare tolerances and assess repeatability rather than assuming bitwise
trajectory equality. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

Measure peak reserved/allocated VRAM, driver-free memory and host/WSL RAM through
forward, backward, topology and checkpoint writes. Aim initially below 6 GiB
reserved with at least 1 GiB free, adjusted for desktop usage. Try two-shot
batches only after the worst planned workload meets the margin.

With 20 PML cells per side, a 256x512 grid has 296x552 samples before extra
stencil halos. One float32 scalar stack at 4,001 stored times is about 2.44 GiB
per shot. Actual memory depends on internal sampling, additional states and
overhead. Gaussian parameter storage does not predict wavefield memory.

If one shot cannot fit, assess temporal checkpointing or uncompressed CPU
storage within measured host-memory headroom. Checkpointing preserves internal
time sampling, complete source resampling and PML states; assemble and filter
traces consistently. Count recomputed forward work.
[Deepwave checkpointing](https://ausargeo.com/deepwave/example_checkpointing)

## Proposed workload

The starting planning grid is 256x512 at 10 m spacing: 2,550 m depth and
5,110 m width between endpoint nodes. A candidate acquisition has 20 shots,
256 receivers, a four-second record at 1 ms sampling and cumulative 4/7/12/20 Hz
bands. These are planning values pending physical convergence and memory tests.
Same-domain 5 m and 2.5 m grids have shapes 511x1023 and 1021x2045.

Representation allowances are 5,000 / 10,000 / 20,000 full-data updates.
Four-stage FWI allowances are 4,000 / 8,000 / 16,000 total. One FWI update
includes every training shot; a batch is not an epoch. At 20 shots, the initial
4,000-update allowance alone costs 80,000 forward and 80,000 adjoint shot
solves, before all extra categories.

## Execution order

1. Retain memory-saving unequal shot accumulation acceptance. Recheck data
   movement, gradients, optimizer state, isolation and rollback whenever that
   implementation or the numerical execution policy changes.
2. Extend the recorded 1,024/4,096/8,192-component software stress cases to the
   intended actual observations and sustained updates, including fragmented
   fields, dense overlap, topology and checkpoint writes. The declared analytic
   20-shot case checks one update, not a complete fit or universal worst case.
3. Lead with the [macro model and local detail application](RESEARCH_DIRECTION.md).
   Keep constructed diagnostics and supervised representation fits separate from
   complete waveform-driven development fits. Use inclined layers and then
   faulted/curved structures with controls for geometry, scale and population.
   Include an independently accepted external reference when a particular
   comparative claim requires it.
4. Freeze the [baseline protocol](BASELINE_PROTOCOL.md) and measured resource
   budget before the larger independent comparison matrix.

Save immutable completed-update snapshots every 100 updates or 15 minutes,
whichever comes first at a consistent boundary. Include field, Adam, identities,
controller history, selection state, RNG, stage/update and complete elapsed work.
Test interrupted continuation across a topology event before long sessions.
This capability is available through `checkpoint_interval`, `max_updates` and
`max_seconds`, with exact CPU and tolerance-checked CUDA continuation. Repeat
acceptance for any changed device/numerical execution policy or larger workload.

Estimate each fit from measured full-update time by frequency stage, plus
validation, topology, checkpoint I/O and audits. Include failed runs and
development extensions in the total campaign cost. Continuous multi-day access
supports sequential long runs; it does not guarantee that the entire campaign
finishes within one reservation.

## One density policy across budgets

Use configs/baseline.json and the same optimize/refine/settle algorithm in every
run. For convergence extensions, declare whether the absolute density window
stays fixed or scales with the update allowance. The controlled settings are in
[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md). No second production controller is
required. Tune all FWI-specific thresholds on development data before evaluation.
