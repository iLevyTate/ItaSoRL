# Controlled persistence test

Date: 2026-10-06
Status: frozen before any agent was scored with it (only a quick-scale smoke run on the
diagnostic agents existed)
Plan: `docs/REVISION_2026-10.md`, step 7. Code: `itasorl/persistence.py`,
`scripts/run_persistence_readout.py`.

## Why

The common-garden readout (FINDINGS 10.6.1; survival tail 0.666 forward, 0.684 reverse) runs
each branch's tail under authentic dynamics but lets it keep both its prefix hidden state and
its prefix physical state. The tail can then be decoded from internal memory, from the
external footprint the prefix left (where the agent is, how fast it moves, how much energy it
has, which pellets are gone), or from both. "Persistent internal representation" claims the
first; the readout cannot tell them apart.

## Conditions (all tails under authentic dynamics, every member of a pair in one CV group)

| Condition | Hidden state at tail start | Physical state at tail start | Tail input | Reads |
|---|---|---|---|---|
| `replay` (primary) | own prefix state | not used | the identical recorded tail of one branch (source alternates A, S by pair), open loop, same first previous action | retention under common input |
| `common_state` | own prefix state | the same snapshot for both (the source branch's) | closed loop, same first previous action | memory plus its behavioral consequences |
| `factorial` | crossed with physical: (A, A), (S, S), (A, S), (S, A) | crossed | closed loop, previous action standardized to branch A's | the hidden-origin label and the physical-origin label, decoded separately on the same tails |
| `reset_hidden` | zero, previous action zero | own prefix state | closed loop | the external footprint alone |

Prefix 20 steps, tail 24 steps, 110 pairs, world seeds 980000 + p (disjoint from every other
readout), the standard probe on h_t at each tail step and on [mean h, final h] over the tail
and its last 8 steps, explicit-v1 partition.

## Frozen decision rule (survival arm, strongest drift, n = 10 seeds)

- **Retained under identical input:** `replay` window AUROC mean at least 0.65, t-based 90%
  CI lower bound at least 0.65, and at least 0.05 above the untrained arm's `replay` window.
  Wording licensed: "the recurrent state retains the prefix condition under identical
  input", with the decay curve and the first tail step at which the per-step AUROC falls
  below 0.55.
- **Not shown:** anything else. Wording: "prefix condition remains decodable after restoring
  authentic dynamics", and the `factorial` and `reset_hidden` results say how much of that is
  carried by the physical footprint.
- The drift-0 survival agents are scored too; every condition must read exactly 0.5 there
  (`tests/test_persistence.py`), or the run is invalid.

## Scope

The historical agents behind 0.666 and 0.684 are not in this repository, so the test runs on
the saved agents of the corrected run `C1` (`docs/specs/2026-10-06-corrected-trainer-confirmation-design.md`).
The historical common-garden numbers stay as recorded, under the narrow wording, whatever the
test shows; a claim about memory in the published agents would need those agents.
