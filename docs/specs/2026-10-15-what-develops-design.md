# What develops with training: the physical trace or the memory?

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

The budget curve (FINDINGS 17.10, `artifacts/budget_curve.json`) scores the survival
snapshots at 100, 200, 300 and 450 updates of both corrected runs on the registered readout.
Decodability rises with return in both arms: 0.548, 0.680, 0.733 and 0.821 with the
auxiliary, 0.528, 0.565, 0.613 and 0.685 without. The curve reads the target only. The
control battery on `C1` at 300 updates (FINDINGS 17.8 and 18.4) splits that reading into
what the observation stream carries, what the state keeps beyond the stream, and what it
keeps beyond the physical trace of the agent's motion, with the best-fitting control leaving
0.620. Whether training sharpens a trace the senses already carry or builds a stored
component beyond it has not been read, and neither has whether the decoded direction is one
axis sharpening or a reorganization. The snapshots exist locally for both runs, 90 agent
files each.

## Design

Agents: the drift-0.45 survival snapshots at 100, 200, 300 (the base file) and 450 updates
of the `C1` rerun (`fullruns/corrected_l3_h8_wm/agents`) and the `C2` rerun
(`fullruns/corrected_l3_h8_nowm/agents`), seeds 0 to 9, with each run's drift-0.45 untrained
arm as the zero-update floor.

Integrity gate: the own-episode target of every snapshot on the registered pools (seed bases
800000 and 850000, 110 episodes of 24 steps, ray steps 5) must equal the per-seed value in
`artifacts/budget_curve.json` to 0.001 for every seed. A difference stops the readout and is
reported with its size.

Per snapshot, run and seed, on the snapshot's own episodes: the control battery of
`scripts/run_control_diagnostics.py`, unchanged in its computation, with a snapshot selector
added so the runner can read the `_u100`, `_u200` and `_u450` files. The quantities read are
the target, the observation-stream decoder, the state beyond the stream (`resid_obs`), the
state beyond the physical trace (`resid_trace`), and the best-fitting control. Matched input
beside it: the 300-update agent's recorded episodes replayed through every snapshot of the
same seed (`eval_protocols.replay_states`), with the target on the replayed states, so the
development is also read with the walk held fixed.

Direction stability, per seed: `u_300` from `mechanism_readouts.probe_direction` on the
300-update pools, as the nudge readout fits it, and `u_b` for the other budgets. Snapshots of
one seed share their hidden units, so the cosine between `u_300` and `u_b` in raw state space
is meaningful. Beside it, the registered probe fit on the 300-update pooled features, frozen,
scores each other snapshot's pooled features of the same seed (transfer AUROC).

## Readings, stated before any number

Seed-paired t-based 90% intervals, n = 10, for each run.

1. Memory grows beyond the senses: `resid_obs` at 450 minus `resid_obs` at 100 has a lower
   bound above 0.05. If it does not while the target at 450 minus at 100 has a lower bound
   above 0.05, the reading is "only the trace sharpens".
2. A stored component beyond the physical trace develops: `resid_trace` at 450 minus at 100
   has a lower bound above 0.05.
3. One axis sharpens: the cosine between `u_300` and `u_450` has a mean of at least 0.7 and a
   lower bound above 0.5, and the frozen 300-update probe reads the 450-update states at 0.65
   or above. "Reorganizes" if the cosine's upper bound is below 0.5. Otherwise "partly".
4. The auxiliary: `C1` minus `C2` in `resid_obs` at each budget is reported; a lower bound
   above 0 at 450 reads "the auxiliary builds the stored component".

Descriptive, with no threshold reading: the first budget at which the own-episode target's
lower bound exceeds the untrained floor's mean by 0.05, per run.

Each reading is reported as met or not met. None is a registered verdict.

## Output

`artifacts/development/corrected_l3_h8_wm.json` and
`artifacts/development/corrected_l3_h8_nowm.json`, each with provenance, the integrity check,
per-seed values per budget, the cosines and transfer readings, and the readings.

## Amendment 2026-10-15, before any code or number

Found while writing the code. The two sequence decoders of the control battery (the
supervised GRU and the flattened-sequence probe) cost about 20 minutes per cell on this
workstation, and this readout has 80 cells. No reading above uses them. The battery is run
without them: target, `resid_trace`, `resid_obs`, the joint MLP control (`resid_joint_mlp`,
the best-fitting control) and the summary-feature stream comparator (`obs_summary_only`),
each computed by the battery's own functions. "The observation-stream decoder" above means
that summary-feature comparator. The cells are computed by a script that calls those
functions on the snapshot files directly, so the runner itself is not changed. Nothing else
changes.

## Second amendment 2026-10-15, after a one-seed setup check and before the full run

The integrity gate's 0.001 tolerance was set without accounting for the stack. The budget
curve was recorded on the cloud container (torch 2.14.1, NumPy 2.4.6, scikit-learn 1.9.1);
this readout runs on the workstation (torch 2.7.0, NumPy 1.26.4, scikit-learn 1.5.2). On seed
0 the 100- and 200-update targets equal the recorded values exactly, and the 300- and
450-update targets differ by 0.0008 and 0.0012, two and three single-pair steps of the
mean-of-folds AUROC. Exact agreement at two budgets shows the right agents and pools are
read. The gate's tolerance is 0.005, changed after this check showed 0.0012, and the largest
difference per budget is recorded and reported. No reading changes. The one-seed check is
not part of the result.

## Note 2026-10-15, after the deep revision review

Reading 1's name, "memory grows beyond the senses", is a label for the increase in
decodability after the linear observation control `history_basis(lags=1)`. It does not
identify memory: a memoryless nonlinear function of the current observation can read high
after that control. Reading 2 is likewise a label for the increase after the physical-trace
basis. The two readings are separate controls, not shares of one signal, and are read as
labels with that meaning. The frozen names and thresholds stay.

