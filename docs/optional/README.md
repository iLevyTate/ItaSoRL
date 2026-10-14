# Optional readouts

These are sensitivity readouts on saved agents. They were run after the October 13
review plan closed, outside that plan. They are not an adjudication. No number here
replaces a recorded control, a frozen threshold, or a verdict, and none enters the
manuscript. Each JSON file carries a `provenance` block naming the script, the
command, the inputs, the stack, and this standing.

## O1. Fold-local speed median

The speed positive control labels an episode fast when its mean speed is above a
threshold, then asks whether the state reads that label. The recorded call takes the
median of the whole pool, held-out episodes included. The fold-local call takes the
median on the outer training fold only and applies it to the training and held-out
episodes. Pair groups stay in one fold when the data are paired. Code:
`itasorl/optional_readouts.py`, `speed_threshold_probes`.

| File | Population | Cells | Result |
|---|---|---|---|
| `o1_fold_local_speed.json` | pooled worlds, bases 800000 and 850000, both drifts, three arms, B1 and B2 | 120 | recomputed global median equals the recorded `pool.speed` exactly; zero calls change; fold-local minus global +0.00162, t90 [0.00015, 0.00309] |
| `o1_fold_local_speed_matched.json` | retained pairs at base 910000, drift 0.45, three arms, B1 and B2 | 60 | two calls change, both untrained seed 19: recorded 0.7493, fold-local 0.7609; mean shift near zero, t90 [-0.00273, 0.00266] |

The matched change does not make B1 or B2 informative. The untrained floor fails on
its own.

```
python scripts/run_optional_o1.py --pooled --out docs/optional/o1_fold_local_speed.json
python scripts/run_optional_o1.py --matched --device cuda --out docs/optional/o1_fold_local_speed_matched.json
```

Inputs: `fullruns/balanced_confirmation_wm` and `fullruns/balanced_confirmation_touch`
(cells, state dumps, saved agents, and `matched_gates.json`). These bundles are local
and in the supplement, not in git.

## O2. Long-history ridge penalty chosen inside each outer training fold

The long-history controls in the paper used one ridge penalty, 0.001, and their held-out
fit was negative. Here the penalty is chosen inside each outer training fold by an
episode-grouped inner split, so a held-out episode never informs its own penalty. The
bases are the two the paper reports as failed fits: observation history through t-7
plus EMA traces and actions (1470 columns), and that basis plus the behavior trace
(1484 columns). Agents: the C1 survival arm, drift 0.45, seeds 0 to 9, pools at
bases 800000 and 850000. Code: `itasorl/optional_readouts.py`, `long_history_ridge`.

| Basis | Fixed 0.001: residual AUROC, held-out R2 | Selected penalty: residual AUROC, held-out R2 |
|---|---|---|
| history + traces + actions | 0.640 [0.608, 0.671], -0.128 [-0.234, -0.021] | 0.737 [0.690, 0.785], 0.906 [0.904, 0.908] |
| the same plus behavior | 0.606 [0.581, 0.630], -0.177 [-0.292, -0.063] | 0.691 [0.650, 0.732], 0.909 [0.907, 0.910] |

The fixed penalty reproduces the recorded control. The selected penalty is 100 or 1000
on every fold. With a positive fit the residual AUROC rises rather than falling to
chance, so the negative fit was depressing the residual, and a negative held-out R2 is
a failed fit, not a direction for the residual. One SVD of the wide design did not
converge on seed 4; the factor routine retries with the `gesvd` driver.

```
python scripts/run_optional_o2.py --device cuda
```

Per-seed checkpoints are written under `fullruns/optional/o2/` (local, ignored by git)
and the summary to `o2_long_history_ridge.json`. A rerun reads any checkpoint already
present instead of refitting it.
