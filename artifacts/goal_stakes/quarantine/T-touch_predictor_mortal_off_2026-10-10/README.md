# Quarantined: T-touch, complete run with the predictor arm in immortal worlds (2026-10-10)

All 20 cells of `T-touch` from the pinned cloud steps (torch 2.14.1+cpu, numpy 2.4.6,
scikit-learn 1.9.1), finished 2026-10-10. The `--mortal off` switch reached the predictor
arm's own scripted training episodes, so its pooled target cannot equal `C1`'s.
Integrity check 2: untrained bit-equal to `C1` in 20 of 20 cells, predictor in 0 of 20.
A mismatch stops the comparison, so nothing here enters any result. Fixed in PR #135;
see the 2026-10-10 PREREGISTRATION_L3 amendment. Kept for the record only.
