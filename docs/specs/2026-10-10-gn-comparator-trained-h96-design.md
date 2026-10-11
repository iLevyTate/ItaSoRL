# Comparator-trained run on the white-jitter family, at the registered trunk: frozen design (2026-10-10)

Frozen before the run. This is the run `docs/specs/2026-10-09-gn-comparator-trained-design.md`
specified, executed as that spec states it. The 2026-10-09 execution (`T-GN-TRAINED`) passed
`--hidden 8` alongside `--l3-hidden 8`, so its agent trunk was 8 units instead of the registered 96,
and it is recorded as uninformative for that reason (PREREGISTRATION_L3, second 2026-10-09
correction). That record stands unchanged. Nothing in the 2026-10-09 spec is re-decided here except
the two items below.

## What is unchanged

The question, gate 0 (`G_gn` at `sigma_v = 0.01`, oracle 0.8653 in band, floor 0.4479, recorded in
`artifacts/expH2/texture_knockout_h8.json`), the decision rule (survival pooled target at least
0.65 and at least 0.05 above both the predictor and the untrained arm, adjudicated at the mean with
the t-based 90% interval beside it), every gate of PREREGISTRATION_L3 section 7, and the text
written in each of the three outcomes are those of the 2026-10-09 spec, verbatim. A gate failure
routes to uninformative exactly as the registered matrix says. In no case does this run change a
margin, a bar, or the verdict of any other run.

## The two changes, both to execution only

1. **The registered trunk.** `--hidden` is not passed, so the agent trunk takes the registered
   default of 96. `--l3-hidden 8` keeps its meaning: the surrogate machinery's width, not the agent's.
2. **The device.** The 2026-10-09 spec fixed CPU so the run would share `C1`'s device class. `C1`
   ran in a Linux CPU container (4 vCPU, torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1, 4
   workers), and the Claude Code cloud container pinned to that stack reproduces `C1`'s cells to the
   bit (the goal-and-stakes runs and the C1 rerun, 2026-10-10). This run executes there, which
   matches `C1` more closely than the owner's laptop did.

## What is run

    bash scripts/cloud_run_step.sh T-gn-h96 100 --l3-family gn --l3-family-param 0.01

which runs `scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --drifts 0.0 0.45
--seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --workers 4 --device cpu --save-agents --resume
--l3-family gn --l3-family-param 0.01` with `ITASORL_FOLDS=explicit`, in steps of at most 100
minutes from a scheduled routine, committing cells to branch `run/T-gn-h96` under
`artifacts/goal_stakes/T-gn-h96/` as they land. The objective and mortality flags take their
registered defaults (survival, mortal). The directory name follows the cloud runner; the result is
promoted to `artifacts/texture/` by `scripts/promote_texture_trained.py` and registered as its own
manifest row, as the 2026-10-09 spec's provenance section describes.

## Integrity before reading

Every cell records `gae_bootstrap = successor`, `objective = survival`, `mortal = true`, and an
agent trunk of 96 (the saved agents' `cell.weight_hh` is `(288, 96)`). A cell that fails any of
these stops the run from being read.

## Cost

Ten seeds, two drift levels, three arms at 300 updates, on four workers: about 4 to 6 hours of
cloud time over two or three scheduled steps.
