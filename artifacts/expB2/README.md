# Experiment B-v2 published artifacts

Canonical committed outputs from pre-registered confirmatory runs (not ephemeral
`fullruns/` bundles from `scripts/run_e2e.py`).

| File | Description |
|------|-------------|
| `expB2_results.json` | **Canonical** full-scale metrics from independent Colab replication (`fullruns/06302026`, commit `4c16be6`, 300 updates, 3 seeds, Tesla T4). Includes CIs, drag ceiling, manipulation-check fields. |
| `expB2_survival.png` | Figure matching the canonical JSON |
| `expB2_results_confirmatory_n3.json` | Archived initial lab confirmatory run (pre-rigor-hardening; survival @ drift 0.45 ≈ 0.595) |
| `expB2_survival_confirmatory_n3.png` | Figure for the archived initial run |

New runs write to `fullruns/<MMDDYYYY>/artifacts/` or here when promoted manually.
Promote a run by copying `expB2_results.json` and `expB2_survival.png` from the run
folder and updating this README if provenance changes.

## Promotion history

| Date | Source | Commit | Notes |
|------|--------|--------|-------|
| 2026-07-01 | `fullruns/06302026` | `4c16be6` | Independent Colab full run (T4, 237 min). Survival @ drift 0.45 = **0.523 ± 0.045** (seeds 0.586, 0.495, 0.488). Replaces prior `expB2_results.json`; initial lab run (≈ 0.595) kept as `expB2_results_confirmatory_n3.json`. Documented in `docs/FINDINGS.md` §9. |
