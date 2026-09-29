# L3 skill-matched model-free baseline

Date: 2026-09-29
Status: FROZEN 2026-09-29 on the owner's instruction ("run whatever is left"),
before any run. Executes on the owner's GPU machine after the hidden-8 new-seed
chain releases the card.

## Purpose

FINDINGS 10.8 removed the next-observation decoder from the survival arm and the
world-identity signal fell from 0.752 (GPU) or 0.730 (CPU, same protocol) to 0.601.
The verdict was "auxiliary-conditional". One confound is open
(`docs/STATUS_2026-09-27.md`, open gap 2): the decoder-carrying agents also forage
better. At train drift 0.45, eval drift 0.45, the published GPU agents return
-0.219 on average (per seed -0.098, -0.312, 0.027, -0.194, -0.152, -0.455, -0.141,
-0.189, -0.285, -0.391; sd 0.144), the CPU device-control agents -0.355, and the
model-free agents -0.474. So 10.8 does not separate the decoder's direct effect on
the recurrent state from an indirect effect through a stronger survival policy.

This run trains the model-free survival arm longer, until its return matches the
decoder-carrying agents', then probes it with the unchanged protocol. If a
skill-matched model-free agent encodes world identity, the decoder mattered
through skill; if it does not, the decoder's effect on the state is direct.

## Runner change

`scripts/run_expB2.py --survival-updates N` sets the actor-critic update budget
for the survival arm only; the predictor and untrained arms keep `--updates`.
The knob enters the config fingerprint (`None` is identical to the key being
absent, so every existing checkpoint stays valid). Tests:
`tests/test_reviewer_gap_flags.py`.

## Stage 1: skill calibration (frozen)

Model-free survival arm (`--no-world-model`), hidden 8, G seed 0, drift 0.45 only,
seeds 0, 1, 2, `--updates 300` for the other arms, `--survival-updates` in the
frozen ascending order **450, 600, 900**. Readout: the survival arm's mean
train@0.45 eval@0.45 return over the three seeds.

Match rule: the FIRST budget whose three-seed mean return is within 0.05 of the
published GPU decoder-carrying mean (-0.219), that is, at least -0.269, is
selected and the sweep stops. If 900 does not reach it, 900 is selected and the
run is reported as "skill not matched at three times the budget"; stage 2 still
runs, and the confound is then reported as unresolved by this design rather
than adjudicated.

Three seeds is a calibration, not an inference: the selection reads the mean and
nothing else. The n = 10 return of stage 2 is what the write-up quotes.

## Stage 2: organism run (frozen)

Identical to the 10.8 architecture-baseline run (PREREGISTRATION_L3 section 9,
`--no-world-model`, drifts [0.0, 0.45], seeds 0..9, `--dump-states`,
`--save-agents`) with `--survival-updates <selected>`, on the owner's GPU
(`--device cuda`, two parallel workers with `--resume`), explicit fold split.
Then the behavior audit on the dumps.

## Gates and decision rules

PREREGISTRATION_L3 sections 7 and 8 as usual (engagement, L0, speed, leakage,
floor, deaths). Skill match is confirmed on the n = 10 return: the stage 2
survival arm's mean eval@0.45 return must be within 0.05 of -0.219; if it is not,
the run is "skill not matched" whatever the probe reads.

Primary comparison, drift 0.45, survival arm, t-based 90% CI over seeds, against
the explicit-split headline (0.774 [0.727, 0.821], from the 2026-09-29 fold
re-score; the same partition this run is scored on) and the bar:

| Outcome | Reading |
|---|---|
| Survival >= 0.65 with the t-CI above the bar, and leads predictor and untrained by more than 0.05 | Skill-mediated: a model-free agent with the decoder agents' survival skill encodes the fingerprint. The 10.8 "auxiliary-conditional" verdict becomes "skill-conditional"; the decoder was one way to reach that skill. |
| Survival below the bar with skill matched | Decoder-direct: matched survival skill does not restore the encoding, so the auxiliary's effect on the state does not run through the policy. The 10.8 verdict stands, strengthened. |
| Intermediate (above both baselines by more than 0.05, under the bar, or t-CI straddling it) | Reported as such; a partial contribution of skill, sized by the survival minus 0.601 lead. |
| Skill not matched | No verdict on the confound; the readout is recorded. |

Secondary, reported but not adjudicated: the paired-by-seed difference against
the 10.8 model-free per-seed values (different device; the CPU device control
puts the device shift near 0.02) and against the published GPU per-seed values.

## Cost

Stage 1: 9 cells at 1.5 to 3 times the survival budget, about 3 to 4 hours with
two workers. Stage 2: 20 cells, about 8 hours. Both queue behind the hidden-8
new-seed chain on the same GPU.

## Out of scope

Matching skill by any other lever (a different learning rate, a longer horizon,
reward shaping), and matching the predictor arm's budget.
