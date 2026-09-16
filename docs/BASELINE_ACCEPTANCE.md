# Foundation acceptance and extension boundary

The accepted implementation is one physical field, one density controller and
one inversion loop. Its completion boundary is [BASELINE_CLOSEOUT.md](BASELINE_CLOSEOUT.md).
This matrix identifies evidence that later applications can reuse.

| Contract | Independent acceptance | Current result |
|---|---|---|
| Physical field and full SPD covariance | Scalar/NumPy covariance solves, dense references, physical coordinate and sampling checks | Passed |
| First derivatives | Parameter-family finite differences, dense autograd and acoustic directional derivatives | Passed |
| Bounded topology changes | Cumulative sampled guard, atomic rollback, signed child amplitudes, survivor/fresh-child Adam checks | Passed |
| Training/validation/test separation | Perturbed held-out traces, stage settling-selection rules, observation-only input schema | Passed |
| Saved fields and completed-update continuation | Exact CPU state comparisons, fresh waveform replay, CUDA tolerance checks | Passed at the checked workloads |
| Unequal shot accumulation | Independent globally normalized losses, all parameter gradients, density scores, work and restart | Passed |
| Disk failure lifetime | Retained real acoustic graphs and injected adjoint failure on CPU/CUDA; recovery and unrelated-file checks | Passed |
| Release packaging | Ruff, repository/link checks, 76 source tests and 76 isolated-wheel tests | Passed |
| Arithmetic compatibility of closeout | Normalized inversion AST and unchanged baseline profile hash | Passed |
| Completed representation study | Independent saved-field decoding, paired starts, selection/continuation decisions, all 18 controls/seeds and guarded edits | Passed; 320,000 supervised updates, zero acoustic solves |
| Long CUDA trajectory equality | Historical and same-source strict prefix comparisons | Failed; not part of the exact-repetition contract |

The fresh release evidence is
`results/validation/baseline_closeout_20260916T155000Z_002/`. The matching resource,
compatibility and paper records are in
`results/baseline_closeout_20260916T155000Z/`. These are local ignored artifacts;
their exact identities are archived for handoff. Research fits retain their
original source identities. The [engineering specification](ENGINEERING_SPEC.md)
records tests, failures, measured resource limits and changes in full.

The final local record includes a review PDF, self-contained HTML, exact
editable manuscript, accepted source archive/wheel and a complete artifact
hash manifest. The representation study improves macro and local detail through
geometry learning; a single accepted split benefits one paired curved-model fit.
Five archived acoustic fits provide separate waveform-driven application
evidence. This acceptance closes the declared foundation stage; it does not
imply universal comparative superiority or resolution/uncertainty calibration.

## What stable means here

A stable foundation has specified physical units and axes, explicit numerical
limits, checked derivatives, validated optimizer/topology state transitions,
protected outputs and reproducible saved-field interpretation. The field and
objective can support a new application without silently changing those rules.
An optimizer can still follow a nonmonotone path, recover an imperfect model,
or differ across long floating-point GPU trajectories. Those outcomes are
measured scientific behavior, rather than a reason to keep redesigning the
baseline indefinitely.

The sparse backend supplies first derivatives. Population selection, geometry
projection and checkpoint selection are discrete operations outside that
derivative contract. The velocity-change guard is sampled; component density
and width are not calibrated uncertainty or seismic-resolution measures.
Memory support applies to recorded workloads. Forced process termination and
persistent OS file locks can leave disk scratch for manual inspection.

## How a new application uses the foundation

Start with the existing [field and inversion API](API.md). Specify the new
objective or model operation, the data it may use, its physical units and an
independent acceptance check. Preserve the field's coordinate conventions,
saved-state meaning and topology/optimizer invariants. A geological penalty,
macro constraint or measured model proposal is a separately named research
extension with its own evaluation; it is not an additional public baseline
engine here.

Reopen a numerical check when a change affects its contract or a concrete
counterexample is found. Reuse unchanged accepted evidence otherwise. Complete
editorial release work for the completed representation manuscript independently
of new applications. They can begin against the frozen accepted source without
waiting for every future scientific question to be resolved.
