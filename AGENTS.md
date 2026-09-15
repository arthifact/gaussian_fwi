# Working on Gaussian FWI

Read docs/ENGINEERING_SPEC.md before changing the baseline. Declare a bounded
change, compatibility and an independent acceptance check before numerical
implementation. There is one method in gaussian_fwi/, with physical support in
gaussian_fwi/core/, one configs/baseline.json and one observation-only run.py.
Do not reintroduce competing public inversion engines or compatibility wrappers.

Training waveforms drive gradients and topology. Validation selects only within
the fixed-population settling phase. Test waveforms and true velocities cannot
influence optimization, tuning or topology ranking. A supervised representation
study must be explicitly separated from FWI.

Tensor axes are (z,[y,]x); physical points are (x,[y,]z) in meters. Velocity and
signed amplitudes are m/s; covariance is full SPD. Grid spacing must agree.
Seed at zero amplitude. Clone/split children inherit signed amplitudes and fresh
Adam state; survivor state is preserved. Density events obey a cumulative
sampled velocity-change limit and roll back atomically. Sampled splits do not
preserve moments or pointwise fields. Count actual forward/adjoint solves;
a gradient score is not a measured loss reduction.

Keep results, datasets, archives, credentials and unrelated experimental modules
out of Git. Never overwrite a fit or import another Gaussian-FWI checkout.
Follow docs/BASELINE_PROTOCOL.md and docs/GPU_CAMPAIGN.md for research budgets.
Small numerical fixtures are software checks, not paper-scale fits.

Run relevant checks and python tools/validate.py --release into a fresh output
directory. Record evidence, limitations and next steps in the engineering spec.
Use additional agents only when requested. Passing tests is not a SOTA result.
