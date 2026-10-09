# Comparator-trained run on the white-jitter family: frozen design (2026-10-09)

Frozen before the run. This one is **confirmatory, not diagnostic**: it tests the registered
encoding criterion on a new artifact family, under the rule already registered in
`docs/PREREGISTRATION_L3.md` sections 5 to 7, with no clause relaxed. It can produce a negative or
a positive, and both are reportable.

## The question

The paper's sufficiency claim, which the title names, is that in-band detectability under a
calibrated oracle does not guarantee encoding. Two cases carry it today and the adversarial review
of 2026-10-09 found both weaker than the claim:

- The L1 quantization seam is in band under **its own** observation-level oracle at the world's
  sensor noise, which is a different detector from the velocity-residual oracle the L3 band is
  defined by. It is matched by number, not by handicap.
- The white velocity jitter `G_gn` is in band under the **same** dynamics oracle, but the only
  evidence that it is "not read" is a readout on agents raised under the **learned fingerprint**.
  No agent has ever been raised inside the jitter, so the paper cannot say what an agent that lived
  in it would carry.

This run closes the second gap. It raises agents inside the in-band white jitter and applies the
registered rule to them. It is the one case on the project's roadmap that would let the sufficiency
claim rest on a trained, same-oracle, gate-0-clean negative.

The preregistration already anticipates it: the 2026-10-08 texture entry in section 12 records that
"no `gn`-trained run was executed" and that "the learned-texture rule remains unmet in this
revision". This is that run.

## Gate 0, already calibrated and unchanged

`G_gn` at `sigma_v = 0.01` was calibrated on world `P` before this spec and is recorded in
`artifacts/expH2/texture_knockout_h8.json`: oracle \auroc 0.8653, in the registered band
[0.85, 0.95]; mechanical leakage passes; untrained floor 0.4479, inside the 0.1 tolerance. Nothing
about gate 0 is re-decided here. This is the property that distinguishes this family from the
hand-authored quadratic drag, whose gate 0 failed and whose trained run is therefore uninformative
and unpromoted.

## What is run

`scripts/run_expB2.py` at the registered L3 configuration, with the surrogate family switched from
the learned `G_motion` to `G_gn` and nothing else changed:

    --drift-mode l3 --l3-family gn --l3-family-param 0.01 --l3-hidden 8 --hidden 8
    --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --drifts 0.0 0.45 --save-agents --resume

Three arms (untrained, predictor, survival), ten seeds, both drift levels, the registered 300
updates, the registered pooled readout at 110 episodes per class and 24 steps. The corrected
successor-value trainer, which is the trainer the manuscript describes.

**Device, fixed here before the run: CPU, four workers** (`--device cpu --workers 4`). The run
exists to be compared against the corrected learned-fingerprint run `C1`, which executed on CPU,
and the project has a recorded CPU-versus-GPU shift of about 0.022 for a full retrain. Running this
one on the idle GPU would be faster and would put a device difference inside the one cross-run
comparison the result is for. The within-run margins the registered rule adjudicates are unaffected
by the choice, since all three arms share the device; the cross-run comparison is the reason the
slower option is taken.

## The decision rule: the registered one, unchanged

From `docs/PREREGISTRATION_L3.md` section 6, applied verbatim: encoding is claimed only if the
survival pooled target is at least **0.65** and exceeds both the predictor and the untrained arm by
at least **0.05**, adjudicated at the mean with the t-based 90% interval reported beside it. Every
gate of section 7 applies as written (engagement, L0 at drift zero, speed positive control, reward
leakage, survivorship, untrained floor). A gate failure routes to **uninformative** exactly as the
registered matrix says, and an L0 failure is reported as the open gate it is, as in C1.

## What is written in each case, committed in advance

- **Rule met.** The sufficiency claim as the title states it is **withdrawn** in its current form:
  an in-band unstructured perturbation is encoded when an agent is raised in it, so detectability
  under the common handicap would be sufficient in this case and the contrast the paper draws is
  not about coherence. The abstract, the title gloss, Contribution 3 and the Discussion are
  rewritten around the surviving contrast, and the result is reported as the primary finding of
  this run whatever it does to the story.
- **Rule not met, gates clean.** The sufficiency claim gains the trained, same-oracle case it
  currently lacks: an in-band perturbation an agent was raised in is not encoded, while an in-band
  learned fingerprint an agent was raised in is. The abstract's sufficiency sentence is rewritten
  to cite this case instead of the readout-only one, and the limitation that no agent was raised
  under the jitter is removed.
- **Any gate fails.** The run is **uninformative** under the registered matrix. It is reported as
  such, it is not promoted, it does not strengthen or weaken the sufficiency claim, and the
  limitation that no comparator-trained run supports the claim stays in the paper.

In no case does this run change a margin, a bar, or the verdict of any other run.

## Provenance

Output is a run bundle under `fullruns/T_gn_l3_h8_wm/` (gitignored), promoted to a committed
artifact under `artifacts/texture/` by the project's promote path, owned by a run registered in
`scripts/build_results_manifest.py`, with per-seed values and the gate battery recorded.
`scripts/audit_stats_recheck.py` binds the prose to it. The agents are saved (`--save-agents`) so
the readout-only batteries can be repeated on them.

## Cost, recorded so the decision is informed

Ten seeds, two drift levels, three arms at 300 updates. The comparable historical run took about
6.5 hours on the owner's laptop GPU; on CPU with two workers expect roughly 8 to 12 hours. The run
is resumable (`--resume` skips finished cells), so an interruption costs at most one cell.
