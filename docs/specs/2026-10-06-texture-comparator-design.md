# Texture comparators: three questions, kept apart

Date: 2026-10-06
Status: frozen before any comparator result on real agents (only quick-scale smoke runs on
the diagnostic agents existed)
Plan: `docs/REVISION_2026-10.md`, step 9.

## What the published texture knockout answered

FINDINGS 14.5 fit the world-identity direction on the agent's own learned-law pools and
scored it, frozen, on a Gaussian-jitter comparator (`G_gn`, sigma_v = 0.01, gate-0 oracle
0.865). Survival read 0.539 against an untrained 0.542. That answers one question: **does
the original L3 direction transfer to the Gaussian perturbation?** It does not. It does not
show that the Gaussian condition cannot be encoded, and it does not show that the L3 signal
is about the perturbation being learned.

## The three questions

| # | Question | Measurement | Code |
|---|---|---|---|
| 1 | Does the original direction transfer to the comparator? | `transfer`: probe fit on the agent's learned-law pools, scored frozen on authentic vs comparator pools | `scripts/run_texture_fresh_probe.py` |
| 2 | Can a fresh probe decode the comparator from the existing agent? | `fresh`: grouped-CV probe fit on authentic vs comparator pools of the same agent (and `fresh_paired`, the balanced paired-seed version) | `scripts/run_texture_fresh_probe.py` |
| 3 | Do agents trained under the comparator develop decodable states? | the standard primary readout of a full run whose agents live in the comparator | `scripts/run_expB2.py --l3-family gn|qd` |

## Comparators

- `gn`: the authentic law plus iid isotropic velocity noise, sigma_v = 0.01 (the published
  gate-0 value). White: lag-1 autocorrelation of its deviation 0.02, RMS one-step deviation
  0.0143 on held-out authentic transitions (`perturbation_profile`).
- `qd`: the authentic law minus a hand-authored quadratic drag, `eps * |v| v dt`.
  Deterministic, smooth, state-dependent, temporally coherent (lag-1 autocorrelation 0.94 at
  eps 0.4). `eps` is calibrated through the same gate 0 as every family
  (`scripts/run_expA_l3.py --family qd`, sweep 0.5 to 16, first in-band candidate with clean
  mechanical leakage and untrained floor), so its detectability to the privileged detector is
  matched to the band the learned law and `gn` were calibrated into. Magnitude and temporal
  structure of each comparator are reported beside its result (`perturbation_profile`), since
  oracle-matched is not the same as structure-matched.

## Runs

1. Readout-only on the saved agents of the corrected run `C1`: questions 1 and 2 for `gn` and
   for `qd` (after its gate 0).
2. Comparator-trained runs `T-gn` and `T-qd`: the declared protocol of `C1` (corrected
   trainer, auxiliary on, hidden 8 trunk, 300 updates, n = 10, drifts 0 and 0.45) with the
   agents living in the comparator. Engagement, leakage, floor, L0, and speed gates apply
   unchanged. These run only if compute allows after `C1` and `C2`; if they do not run, the
   conclusion stays at question 1.

## Frozen wording rules (survival arm, n = 10)

- Q1 passes if `transfer` mean is at least 0.65 and at least 0.05 above the untrained arm:
  "the L3 direction reads the comparator". Otherwise: "the original direction does not
  transfer to the comparator", and nothing more.
- Q2 passes by the same rule on `fresh`: "the existing agent's state separates the comparator
  when a probe is fit to it". Otherwise: "a fresh probe did not meet the registered criterion
  on the comparator".
- Q3 uses the run's primary rule (`decide_h_b2`).
- A **learned-texture** claim ("decodability is specific to learned dynamics among the tested
  perturbation classes") needs all of: `C1` meets its primary rule, and `T-gn` and `T-qd` both
  fail theirs with every gate passing and detectability matched at gate 0. Anything less, and
  the paper says which classes were tested and what each showed, with no mechanism claim.

## The graded blend (FINDINGS 14)

The convex blend from the learned law to the authentic law shows decoding falls monotonically
as the dynamics difference shrinks, to chance at the authentic end. That is evidence that the
decoding depends on the dynamics difference. It does not by itself identify a representation
of learnedness, because every point of the blend is also a smaller perturbation.

## Amendment (2026-10-06, after the qd gate 0 and before any qd readout on agents)

Gate 0 for `qd` (`artifacts/texture/gate0_qd.json`, sweep eps 0.5 to 16, sigma 0.02, three
untrained floor seeds) found **no passing eps**. The privileged oracle rises slowly (0.601 at
eps 0.5, 0.656 at 4, 0.697 at 8, 0.768 at 16) and the untrained floor leaves its tolerance
first (0.560 at eps 4, 0.682 at 8, 0.814 at 16). A coherent coefficient-like perturbation is
felt by an untrained recurrent state before the handicapped oracle finds it in band, the same
empty window FINDINGS 10.7 recorded for constant drag.

Fallback, fixed here: `qd` is matched to `gn` on **one-step RMS magnitude** instead of oracle
detectability. On 60 held-out authentic episodes `gn` at sigma 0.01 deviates by 0.01419 RMS
and `qd` at eps 1 by 0.002356, and the deviation is linear in eps, so the matched value is
**eps = 6.0**. At that eps the untrained floor is expected outside tolerance, so questions 1
and 2 for `qd` are reported with the untrained arm beside the survival arm, and the
comparison that carries meaning is the survival-minus-untrained margin. The comparator-trained
run `T-qd` is **not run**: its family fails gate 0, and a run on an uncalibrated family could
not support the matched-detectability condition of the learned-texture rule. That rule
therefore cannot be met in this revision, and the paper keeps the narrow wording.
