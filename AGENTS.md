# Working on Gaussian FWI

Read docs/ENGINEERING_SPEC.md before changing the baseline. Keep changes bounded
and declare an independent acceptance check and compatibility before editing
numerical policy. Core packages are dynamic_refinement/, gaussian_fwi/ and
fwi_core/. The observation-only entry point is run.py; tests live in tests/unit/.

Training waveforms drive FWI and topology; validation selects. Test waveforms
and true velocities cannot influence optimization, tuning or ranking.
Supervised representation fitting is a separately declared diagnostic.

Tensor axes are (z,[y,]x); physical points are (x,[y,]z) in meters. Velocity and
signed amplitudes are m/s; covariance is full SPD. Grid spacing must agree.
Topology changes preserve survivor Adam state, initialize newborn state fresh,
insert at zero amplitude, obey cumulative sampled field-change limits and roll
back atomically. Count actual forward/adjoint solves. Predicted gain is not a
measured waveform reduction; waveform fit is not velocity RMSE.

Keep generated results, datasets, archives and unrelated experimental modules
out of Git. Never overwrite a fit or import another Gaussian-FWI checkout.
Use docs/BASELINE_PROTOCOL.md and docs/GPU_CAMPAIGN.md for research budgets.
Small numerical fixtures are software checks, not paper-scale fits.

Run relevant independent checks and python tools/validate.py --release in a
fresh directory. Record commands, evidence, limitations and next steps in the
engineering specification. Use additional agents only when the owner requests
them. Passing tests does not establish convergence or publication readiness.
