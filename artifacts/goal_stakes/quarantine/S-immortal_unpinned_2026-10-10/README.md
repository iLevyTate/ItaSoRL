# Quarantined: S-immortal first cloud step, unpinned torch (2026-10-10)

These cells came from the first cloud step of `S-immortal` (2026-10-10, 01:04 to 02:44 UTC). That
step installed the newest torch from the CPU wheel index instead of the version `C1` used
(torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1).

Integrity check 2 of `docs/specs/2026-10-07-goal-and-stakes-design.md` requires the
predictor and untrained pooled targets to equal `C1`'s at the same (drift, seed) to the bit.
On the first four drift-0 cells the untrained target was bit-equal and the predictor target
was not (C1 / this step: s0 0.4314 / 0.4698, s1 0.5537 / 0.4971, s2 0.6074 / 0.4921,
s3 0.4831 / 0.4764). `T-touch` and `S-immortal` agreed with each other to the bit, so
the step was deterministic and the gap is environmental. A mismatch stops the comparison, so
nothing here enters any result. The run restarts from zero cells on the pinned versions.
Kept for the record only.
