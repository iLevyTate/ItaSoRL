# Whisper or shout: how prominent is world identity in the state?

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

Every reading of world identity in the record is supervised: a probe with labels finds a
direction, and its held-out AUROC is the number (`C1` 0.733; the untrained trunk on
identical input 0.636, FINDINGS 22.2; the balanced readout 0.767). None of them says how large that
component is against everything else the state carries. The fake world could be a major
axis of the agent's state, or a faint trace that only a labeled probe can find. Label-free
measures separate the two, and that is the first question a reader outside the field asks.

## Design

Agents: the `C1` rerun (`fullruns/corrected_l3_h8_wm/agents`, seeds 0 to 9): survival,
predictor and untrained at drift 0.45, and survival at drift 0 as the L0 analogue. Second
block: the `C2` rerun's survival arm (`fullruns/corrected_l3_h8_nowm/agents`), no auxiliary.

States: each arm's own-policy episodes on the registered pools (seed bases 800000 authentic
and 850000 surrogate, 110 episodes of 24 steps, ray steps 5), the pools the registered probe
reads. The features `X` are the registered pooled features [mean h, final h], standardized
over the pooled episodes, which is what the probe sees after its scaler. Secondary: the same
measures on the drift-0.45 survival agent's episodes replayed through every arm
(`eval_protocols.record_trajectories` and `replay_states`), so the floor is read on
identical input.

Per arm and seed:

- Variance share: the fraction of the total variance of `X` that lies along the registered
  probe's direction in standardized feature space (the unit coefficient vector of the probe
  fit on all episodes), beside the share of the first principal component, and the index of
  the principal component most aligned with the probe direction.
- Effect size: Cohen's d between worlds of the out-of-fold projection onto the probe
  direction. The probe is fit on the training folds of the registered grouped five-fold
  split and the projection taken on the held-out fold, so the d is not inflated by fitting.
- Label-free recovery: k-means with two clusters on `X` (ten restarts from a fixed seed), its
  agreement with the world label as the accuracy of the better of the two labelings, and the
  adjusted Rand index. The sign of the first principal component is scored the same way as
  a second label-free split.
- L0 analogue: the drift-0 survival arm on two authentic pools (seed bases 800000 and
  850000, both at drift 0). Its k-means agreement should sit within 0.1 of 0.5; if it does
  not, every agreement here is reported as carrying the world-sample draw.

## Readings, stated before any number

Seed means with t-based 90% intervals, n = 10, for the drift-0.45 survival arm of each run.

1. Shout: k-means agreement at least 0.65 with a lower bound above 0.60, or a variance share
   along the probe direction of at least 0.25.
2. Whisper: k-means agreement with an upper bound below 0.60 and a variance share with an
   upper bound below 0.10.
3. Between: neither, reported as "audible" with its numbers.
4. Floor: the untrained arm's k-means agreement on identical input. If the untrained arm
   meets reading 1 there, the survival reading is reported as carried by the input, not the
   trunk.

Each reading is reported as met or not met. None is a registered verdict.

## Output

`artifacts/salience/corrected_l3_h8_wm.json` and `artifacts/salience/corrected_l3_h8_nowm.json`,
each with provenance, per-seed values for every arm, and the readings.
