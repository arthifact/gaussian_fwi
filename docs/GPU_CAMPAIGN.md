# RTX 4060 campaign

Reference machine: Windows 11, Ryzen 9, 16 GB host RAM, RTX 4060 with 8 GB VRAM,
available for several days continuously. Follow [INSTALLATION.md](INSTALLATION.md)
for the proposed Ubuntu/WSL 2 environment.

## Implementation status

The numerical baseline and observation-only CPU runner are available. The core
uses tensor devices explicitly, but this repository does not yet certify CUDA
execution. The sparse decoder retains CPU cKDTree search and NumPy transfers.
Shot accumulation, GPU numerical acceptance and reliable mid-stage snapshots
are the next bounded implementation. Existing restart resumes completed stages.

## Memory strategy

Start with one experiment worker and one shot per batch. Accumulate the full
training gradient before one optimizer step, retaining the global loss
normalization and training-derived trace weights. Weight unequal batches
correctly, add regularization once and rank topology from the complete gradient.
Deepwave documents this memory-reduction approach.
[Shot accumulation](https://ausargeo.com/deepwave/example_rtm)

Start with float32 and independent float64 checks. Initially disable AMP, TF32,
lossy compression and reduced gradient sampling. CPU/GPU reductions can differ;
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

1. Implement and independently check CUDA data movement, per-parameter
   gradients, unequal shot accumulation, an optimizer step, held-out isolation,
   survivor/newborn state, rollback and completed-update restart.
2. Measure 1,024/4,096/8,192 Gaussians on the intended domain, including
   fragmented fields, dense overlap and topology events. Use synchronized timing.
   Short timing/stress probes are engineering measurements, not research fits.
3. Run complete long development pairs: fixed/adaptive representation of
   inclined thin layers, then adaptive/grid waveform inversion. Expand to
   faulted/curved structures and a second FWI development case.
4. Freeze the [baseline protocol](BASELINE_PROTOCOL.md) and measured resource
   budget before the larger independent comparison matrix.

Save immutable completed-update snapshots every 100 updates or 15 minutes,
whichever comes first at a consistent boundary. Include field, Adam, identities,
controller history, selection state, RNG, stage/update and complete elapsed work.
Test interrupted continuation across a topology event before long sessions.
This capability remains to be implemented.

Estimate each fit from measured full-update time by frequency stage, plus
validation, topology/trials, checkpoint I/O and audits. Include failed runs and
development extensions in the total campaign cost. Continuous multi-day access
supports sequential long runs; it does not guarantee that the entire campaign
finishes within one reservation.
