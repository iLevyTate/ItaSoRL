# ITASORL, In Plain Language

This is the explainer. Every term, experiment, and process in ordinary English,
then the research it sits on (APA 7 at the end). Written for an adult, not a
child. Numbers here are a snapshot as of 2026-07-14. Living sources:
`docs/FINDINGS.md` and `docs/PREREGISTRATION_L3.md`.

Each idea gets a comparison first, then a precise sentence. If you can walk from
one to the other, you understand the project.

---

## 1. The Big Question

Picture raising a digital creature inside a computer world. It is born knowing
nothing: no labels, no instructions, no idea what a "simulation" or an
"experiment" is. It simply lives there. It looks around, gets hungry, and slowly
learns to find food.

Now play a trick on it. On some runs its world uses the true physics of the
program underneath. On other runs we quietly swap in a near-perfect copy, a
stand-in that behaves almost the same but carries tiny manufacturing flaws. Think
of it as a gentle Truman Show, or a quiet "is this the Matrix?" test aimed at a
very small mind. The philosophical version of that question, whether a world can
in principle be a simulation and whether anyone inside could tell, is an old one
(Bostrom, 2003), and physicists have even asked what measurable fingerprints a
simulated universe might leave (Beane et al., 2014). This project turns that idea
into a controlled experiment.

The project asks two questions and keeps them strictly apart:

1. **Can the copy be caught at all?** If an all-seeing inspector with the full
   records examines how the world behaves, can it spot the fake? This is called
   **detectability**.
2. **Does the creature notice on its own?** We never reward it for noticing. We
   never even tell it that a fake exists. Afterward, we only check whether the
   knowledge ended up in its head as a side effect of ordinary living. This is
   called **incidental encoding**.

The gap between those two questions is the whole point. Something can be easy to
catch from the outside and still never get recorded on the inside, simply because
nothing in the creature's daily life depended on it. So the sharp version of the
question becomes: under what conditions does a mind start to represent something
it was never asked to care about?

Four core terms, defined up front:

- **ITASORL** (written "ItaSoRL" in the README): the project's name. The
  documents never spell out the acronym; the "RL" stands for reinforcement
  learning. Treat it as a proper name for "the artificial-life
  simulation-detection project."
- **Tabula rasa**: Latin for "blank slate." The creature is a newborn with no
  inherited knowledge. In technical terms: no pre-trained weights and no imported
  meaning; everything it knows comes from its own experience in this world.
- **Readout, not reward**: we read its diary, but we never pay it for answers.
  In technical terms: whether the world is real or fake never appears in the
  observation, the reward, or the training loss. It is only ever decoded from the
  creature's internal state by an outside tool after training.
- **Substrate-grounded**: the fake's flaws come from how it is computed, not from
  any label we attach. In technical terms: the only differences between the two
  worlds are computational artifacts (rounding, drift, learned-model error),
  never task structure, rewards, episode length, or metadata.

The four hypotheses (from `README.md`):

- **H1 (detectability)**: a privileged outside inspector can tell the two worlds apart.
- **H2 (substrate-grounding)**: the signal really lives in the computational seams, not in some side effect.
- **H3 (emergence under selection)**: across generations, creatures whose survival depends on the difference get better at representing it. (Tested in Experiment C: thirty generations did not raise what the pooled population readout measures. That readout cannot see detectors that individuals carry in their own private directions, so it does not show there were none. See Act 9's caveats in section 8.)
- **H4 (legibility, or incidental encoding)**: a from-scratch creature encodes the difference on its own, without reward. (The main event.)

---

## 2. The World: A Patch of Earth

The creature lives in "A Patch of Earth" v0 (`itasorl/patch_of_earth.py`, class
`PatchOfEarthV0`). Think of it as a sealed terrarium built by hand: small,
self-contained, and completely predictable if you know the starting conditions.

- **2.5D**: the terrarium is a flat 2D table with a sculpted landscape glued on
  top. Precisely: a continuous 2D plane plus a fixed terrain-height field, which
  gives slopes and different media without the chaos of full 3D physics.
- **Deterministic**: replay a chess game from the same opening and you get the
  identical game. Precisely: given the same seeds, every run reproduces exactly,
  which is what makes clean comparisons possible.
- **Dissipative**: the world has friction everywhere, like a ball rolling through
  honey. Precisely: drag, decay, and a pull back toward equilibrium keep
  trajectories from flying apart, so two identical worlds stay identical over the
  test window. Chaos is deliberately tamed.
- **Seeds**: the recipe card for the randomness. Precisely: separate random
  streams for the world, weather, ecology, and policy (a `SeedBundle`), so each
  source of randomness can be held fixed or varied on its own.

What the creature senses, about 146 numbers per moment (the observation):

- **Vision**: 24 range-finding beams fanned across a 300-degree view, like
  whiskers made of light. Each beam reports how far it reached, the rough color
  of what it hit, and how fast that thing is moving. 24 beams times 5 values is
  120 numbers.
- **Smell**: chemical strengths and which direction they grow, the way you can
  follow the smell of baking to the kitchen. This channel can be switched off,
  and it is off in the first setup.
- **Interoception**: sensing your own body from the inside, such as hunger,
  speed, heading, and temperature. Precisely: about 14 internal values like
  velocity, energy, and hydration.

Two mechanical terms that come up constantly:

- **obs_spec**: every message to the creature arrives in the exact same shaped
  envelope, no matter which world sent it. Precisely: the observation format
  (length, data type, channel layout) is identical between the real and fake
  worlds, checked with a hash, so nothing can be given away by the packaging. All
  signal must live in the values, never the format.
- **Drag, metabolism, and food**: the daily grind. Movement follows Newton's
  rules, thrust plus friction. Energy constantly drains (metabolism) and is
  refilled by eating pellets. In the survival experiments the metabolism is
  turned up so a creature that does not forage starves in about 50 steps.

---

## 3. The Fake Worlds: The Surrogate Ladder (L0 to L4)

A **surrogate** is the stand-in world, the counterfeit. The **surrogate ladder**
is a series of counterfeits, from "photocopy" to "master forger," each with a
different kind of flaw (the **tell**). Code: `itasorl/world.py` wraps the real
world, and `itasorl/surrogate_l3.py` builds the learned one.

| Rung | Counterfeit comparison | Precisely | Tell |
|------|------------------------|-----------|------|
| **L0** | A second genuine bill from the same mint | An identical rebuild of the real world | None. Detection must sit at chance (0.5). This is the honesty check: if the apparatus "detects" L0, the apparatus is broken. |
| **L1** | A pixelated printout of a photo | Observations snapped to a grid (rounding in space or time; grid spacing 0.06 in the headline) | Staircase patterns and reduced variety, because values can only land on grid points. Visible in a single frame. |
| **L2** | Ink that slowly smears the longer you copy | The drag coefficient drifts during the episode (a slow random walk pulled toward home, called AR(1); strength set by `drift_sigma`) | The physics stops adding up over time: energy and momentum budgets slowly break. You need a window of time to see it, not one frame. |
| **L2-regime (B-v3)** | Every copy is smeared by one fixed, random amount | The drag offset is constant within an episode but drawn fresh each episode (`drag_mode="regime"`) | A stable handicap you could, in principle, identify and adapt to, rather than a moving wobble. |
| **L3** | A forger who studied thousands of real bills and now draws them from memory | A small neural network (**G_motion**, or just **G**) is trained to imitate the real motion law, then frozen and swapped in as the world's physics | The **generative fingerprint**: slight blur, missing fine detail, a characteristic learned noise, and **mode collapse** (the forger never draws the rare serial numbers, so rare events stop happening). |
| **L4** | A forger with a detective on the team | An adversarially trained surrogate that actively works to look identical to the real world | Whatever small invariant the adversary fails to preserve. Not started. |

Key L3 ideas:

- **G_motion (G)**: the forger itself. Precisely: a small recurrent network
  trained on real motion transitions, then frozen; during L3 runs it replaces the
  velocity-update rule entirely. A learned world model standing in for true
  physics is exactly the DreamerV3 style of model this project borrows from
  (Hafner et al., 2023).
- **Capacity calibration**: choosing how skilled the forger is. Too skilled and
  even the inspector cannot catch it; too clumsy and it is obvious. Precisely: the
  network size of G is the only difficulty knob, tuned so the Experiment A
  inspector scores in the band 0.85 to 0.95 AUROC. That band is **gate-0** (see
  section 6).
- **hidden=8 and hidden=7**: two forger skill levels that both landed in the
  band. hidden=8 is the headline fake (inspector 0.928); hidden=7 is the **second
  capacity** (inspector 0.922), run to check whether the conclusion depends on the
  particular forger. Careful: these are the forger's brain size, not the
  creature's (the creature's memory is hidden=96). Same word, two different knobs.
- **Artifact-conditional**: a conclusion that holds for one forger but not
  automatically for every forger. This caveat turned out to matter (section 8).

---

## 4. The Creature: Brains and Hidden States

Two creature designs share the same skeleton: an encoder, a recurrent memory
cell, and heads on top. Code: `itasorl/agent.py` and `itasorl/agent_ac.py`.

**The daydreamer (RSSM-lite, `RecurrentWorldModel`)**, used in Experiment B:

- **Encoder**: the eyes. Precisely: a small network that compresses the roughly
  146-number observation into a compact code.
- **GRU (Gated Recurrent Unit)**: a short-term-memory notebook the creature
  rewrites every moment, deciding what to keep and what to erase. Precisely: a
  recurrent cell whose state vector `h_t` carries information forward through
  time.
- **h_t (the hidden state)**: the notebook page at time t. This is the object the
  whole project studies. If the world's identity is anywhere in the creature's
  head, it should be readable from `h_t`.
- **hidden=N**: the notebook size (lines per page). The creature's memory is
  usually hidden=96.
- **Decoder**: the imagination. Precisely: a network that tries to reconstruct
  the next observation from `h_t`. The daydreamer's only job is predicting the
  next moment of its own senses (self-supervised prediction).
- **RSSM (Recurrent State-Space Model)**: the architecture family this follows,
  which is encode, remember, predict. "Dreamer" is the well-known full version
  (Hafner et al., 2023); this is a deliberately compact "lite" cut.

**The hungry survivor (`RecurrentActorCritic`)**, used in B-v2, B-v3, and L3:

- Same eyes and notebook, plus two new heads.
- **Actor**: the player who chooses moves (thrust, turn, eat).
- **Critic**: the coach who scores situations ("this spot is worth about this
  much future food"). Precisely: a value head that estimates expected return.
- **A2C (Advantage Actor-Critic)**: the training method where the coach's scores
  tell the player which moves turned out better than expected (Mnih et al., 2016).
  "Advantage" means "better than the coach predicted."
- **GAE (Generalized Advantage Estimation)**: a way of smoothing the coach's
  feedback so it is neither too jumpy nor too sluggish (Schulman et al., 2016;
  lambda is 0.95 here).
- **Potential-based shaping**: a "warmer, colder" hint toward food, computed the
  same way in both worlds, designed not to change what the best strategy is.
  Precisely: reward shaping with a potential function (negative distance to the
  nearest pellet) that leaves the optimal policy unchanged when a terminal state has
  potential zero and an episode cut off by the rollout limit is bootstrapped
  correctly. The historical trainer got the second part wrong (Act 10), so for those
  runs the hint was a training aid without the guarantee.
- **Harsh metabolism**: the survival pressure. Energy burns fast enough that a
  non-forager starves in about 50 steps. Combined with denser food (24 pellets
  instead of 12), this makes "learn to forage or die" the real lesson.
- **wm_coef, ent_coef, vf_coef**: dials on the training loss. wm_coef weights an
  extra "keep predicting your senses" task added to the survivor; ent_coef
  rewards exploration; vf_coef weights the coach's learning.

**Three creatures, one comparison.** Every survival-era experiment trains three
creatures with the identical skeleton, differing only in their goal:

- **untrained**: a newborn with random weights, never trained. This is the
  mechanical floor: whatever a reader gets from it is what the wiring gives away
  for free.
- **predictor**: the daydreamer (prediction only) on this skeleton.
- **survival**: the hungry survivor (A2C plus GAE plus a prediction side task).

If only the survival creature encodes the world's identity, then survival
pressure is doing the work. That three-way comparison is the experiment's engine.

One wrinkle, found later and worth flagging here rather than letting it ambush you
in section 8: the survival arm carries that prediction side task too, so the
three-way comparison on its own cannot separate "survival did it" from "survival
plus self-prediction did it". A fourth arm, survival with the side task removed,
settles it, and the answer is that both are needed. See Act 6.

---

## 5. The Experiments: A, B, B-v2, B-v3, L3, C

Each experiment is one question. The runners live in `scripts/`.

**Experiment A: the inspector with full records.**
Question: can the fake be caught at all, by anyone?
Setup: no creature is involved. An outside classifier (the **oracle**) reads full
trajectory logs from matched pairs of worlds and tries to tell real from fake. It
even gets privileged tools, such as re-applying the exact real physics rule and
measuring the leftover error. Precisely: this measures the **detectability
ceiling**, the best score anyone could reach. Code: `itasorl/experiment_a.py`;
scripts `run_expA.py` (L1), `run_expA_l2.py` (L2), `run_expA_l3.py` (L3 gate-0).

- **Matched pair**: identical twins share one morning, then at noon are split
  into two rooms, one real and one fake, starting from the exact same freeze
  frame. Precisely: a shared real prefix is snapshotted, then continued once
  through the real world and once through the surrogate from an identical state.
  Any later difference is the artifact and nothing else. This is the keystone
  control against confounds.

**Experiment B: the daydreamer's diary check.**
Question: does a creature trained only to predict its senses write "this world
feels off" into its notebook on its own?
Setup: train the daydreamer in both worlds (using the L2 drift fake on purpose,
not L1, so nobody can say "it just noticed the pixelation"), then read `h_t`.
Variants: recurrent-state probe (`run_expB_full.py`), prediction-error channel
(`run_expB_surprise.py`), longer imagination horizons (`run_expB_kstep.py`),
engagement checks (`run_expB_gap.py`), nonlinear probe (`run_expB_nonlinear.py`).

- **Surprise channel (Signal B)**: train the daydreamer only in the real world,
  drop it into the fake, and measure how wrong its predictions feel. Precisely:
  read world identity from the frozen model's per-episode prediction error.
- **Signal A versus Signal B**: two routes the knowledge could travel. Signal A:
  the raw experience differs, so the encoder maps the two worlds to different
  mental regions. Signal B: the prediction errors have a different texture in the
  fake. Both are honest "incidental" routes.

**Experiment B-v2: make it life and death.**
Question: if coping with the fake becomes necessary to survive, does the creature
start representing it?
Setup: the three-creature comparison under harsh metabolism, using the L2 drift
fake. Pre-registered in `docs/PREREGISTRATION.md`. Script: `run_expB2.py`.

**Experiment B-v3: a steady handicap instead of a wobbly one.**
Question: maybe a randomly wobbling drag simply cannot be learned; what about one
fixed, identifiable offset per episode?
Setup: the same pipeline with `--drift-mode regime`. Pre-registered in
`docs/PREREGISTRATION_Bv3.md`.

**The L3 arc (this branch's work): swap the hand-made fake for the forger's fake.**
Question: hand-tuned knobs (L1, L2) never got encoded; does a structurally
different world, a learned imitation with a generative fingerprint, change the
answer?
Setup: the B-v2 pipeline with `--drift-mode l3 --l3-hidden 8` (or 7), after G
passes gate-0. Pre-registered in `docs/PREREGISTRATION_L3.md`. It includes the
second-capacity replication and the behavior-mediation audit (section 6).

**Experiment C: evolution (not started).**
Question: across many generations of creatures that live, die, and reproduce,
does detection ability strengthen when survival depends on it (H3)?
One planned control worth knowing now: the **common-garden assay**. To compare
offspring fairly, you test them all in the same kitchen, not each in their own
home kitchen. Precisely: freeze each generation and test everyone on the same
hidden set of worlds, which separates "genuinely better at the skill" from
"merely descended from survivors" (**survivorship bias**).

---

## 6. The Referee: Gates, Probes, and Audits

This project is careful by design. Most of its machinery exists to catch
self-deception before it becomes a claim.

**The probe: a reader with strict rules.**
Think of someone who is handed only photocopies of the creature's notebook pages
and must guess which world each page came from. The reader is kept deliberately
simple, so that if it succeeds, the information must be written plainly on the
page. Precisely: a logistic-regression classifier trained on episode-level
features of `h_t` (the average and final hidden state per episode), with grouped
cross-validation. A **nonlinear probe** (a random forest) is a stronger reader,
used to confirm that a negative result is not the simple reader's weakness.
Using a weak probe plus controls, rather than a powerful one, is a deliberate
choice from the probing literature, where a strong probe can appear to "find"
structure that is not really there (Hewitt & Liang, 2019).

**Probe controls, always run as a trio:**

- **target**: read the world's identity (the real question).
- **shuffled (negative control)**: read deliberately scrambled labels. This must
  score about 0.5. If the reader can "read" nonsense labels, it is cheating
  somehow.
- **speed (positive control)**: read whether the creature was moving fast or
  slow. This must score high (about 0.9). It proves the notebook is legible at
  all. If even speed cannot be read, the creature never learned anything and the
  run means nothing.

**Gates: the pre-flight checklist.**
A rocket launch is scrubbed if any item on the checklist fails; you do not
"interpret" a scrubbed launch, you fix the rocket. Precisely: these are
pre-registered pass or fail checks, and if any gate fails the run is declared
**UNINFORMATIVE** (neither positive nor negative evidence). The gates:

- **gate-0 (L3 only)**: the forger must be good but not perfect. Precisely: the
  Experiment A inspector on G's world must score between 0.85 and 0.95 AUROC, and
  the creature-side untrained floor must be near chance. This is now a committed,
  runnable check (`scripts/run_expA_l3.py`) that validates both halves at each
  capacity.
- **Engagement gate**: the creature really did learn to eat. Precisely: the
  trained policy must beat random and scripted baselines by a set margin (0.15
  return) with a comparable lifetime. A creature that never engaged with survival
  cannot tell you anything about survival-driven encoding.
- **L0 equivalence**: the lie detector must read "calm" on an innocent person.
  Precisely: real-versus-real probing must be statistically equivalent to chance,
  not merely "not significantly different." That distinction is exactly what
  equivalence testing (the TOST procedure) was built for (Lakens, 2017).
- **Leakage audit**: make sure the answer is not written on the envelope.
  Precisely: the same probe trained on side channels (summed reward, episode
  length, lifetime, metadata) must score about 0.5. The audit has teeth:
  deliberately planting a tiny reward difference makes it correctly fail, with
  the leak jumping to about 0.98. This guards against a known trap, where a
  probe's apparent "awareness" is really tracking a surface feature rather than
  the thing you care about (Nguyen et al., 2025).
- **Survivorship check**: no missing bodies skewing the sample. Precisely: zero
  early deaths in the probe pools (110 of 110 episodes per world), so the data is
  not biased toward creatures that happened to live.
- **Untrained-floor tolerance**: the wiring itself must not give away the answer.
  Precisely: the untrained creature's target AUROC must sit within 0.1 of chance
  at the test difficulty.

**The behavior-mediation audit: thoughts, or just a limp?**
The deepest worry: the survival creature behaves differently in the two worlds,
because different physics means different walking. So maybe the reader is not
reading thoughts at all, just noticing the limp reflected in the notebook. To
test whether a handwriting analyst reads personality or only penmanship, you
first subtract everything that penmanship predicts and see what is left.
Precisely (`itasorl/behavior_audit.py`, `scripts/audit_behavior_mediation.py`):

- **behavior-only baseline**: how well does behavior alone (speed, energy, food
  distance, drag) reveal the world? This measures the size of the "limp."
- **Residualization**: fit a map from behavior to `h_t`, subtract it, and read
  what remains. This is done in-fold (fit only on the training portion) so the
  control cannot cheat by peeking at the test data.
- **Per-episode control**: subtract only each episode's average behavior. This is
  coarse and, it turned out, over-subtracts, hiding real signal.
- **Per-timestep control**: subtract behavior moment by moment (this step, the
  previous step, and the running average). This is surgical, strictly stronger,
  and the pre-registered deciding version. In the result files it is called
  `resid_trace` (with `resid_trace_quad` for the version that also removes squared
  and interaction terms).
- **Trace dumps**: keeping the full security-camera footage, not only the daily
  summary. Precisely: per-timestep behavior traces saved alongside the states
  (`--dump-states`), so audits can be rerun later without retraining anything.

---

## 7. The Statistics

- **AUROC** (Area Under the ROC Curve): the standard score in this project.
  Blindfolded, you draw one item from the "real" bag and one from the "fake" bag;
  AUROC is the chance your rule ranks them in the right order. 0.5 is a coin flip
  (knows nothing) and 1.0 is perfect. So 0.75 means "gets the pair right three
  times out of four."
- **The 0.65 bar (SESOI, Smallest Effect Size Of Interest)**: the height a claim
  must clear, fixed to the wall before anyone jumps. Precisely: incidental
  encoding is claimed only if the survival AUROC is at least 0.65 and beats the
  untrained creature by 0.05 and beats the predictor by 0.05. Choosing this in
  advance stops anyone from lowering the bar after seeing the data.
- **90% CI (confidence interval)**: the net around the true value. The reported
  number is the catch; the interval says where the true value plausibly sits. The
  pre-registered rule is that the interval must clear the bar (exclude 0.65), not
  just have its midpoint above it. Reporting intervals rather than single numbers
  is standard practice for trustworthy RL results with few runs (Agarwal et al.,
  2021).
- **TOST and ROPE**: tools for proving that something is boringly equal to
  chance, which is harder than failing to prove it is different. "I found no
  elephant in the room" is weak; "I searched every corner and can certify there
  is no elephant" is what TOST and ROPE provide (Lakens, 2017). They are used on
  the L0 control, where being at chance is a requirement.
- **GroupKFold and unit of inference**: do not count one student's 100 answers as
  100 students. Precisely: cross-validation folds respect episode boundaries, and
  statistics are computed at the episode or seed level, never the timestep, which
  avoids fake confidence from repeated measurements.
- **Seeds, n=10**: bake the same recipe in 10 different ovens. Precisely: the full
  protocol repeated under 10 random seeds; the per-seed scores give the interval
  and the "how many seeds cleared the bar" count. Reinforcement learning is
  famously noisy across seeds, so this discipline is essential, not decorative
  (Henderson et al., 2018; Chan et al., 2020).
- **Pre-registration**: sealing your predictions in an envelope before the game.
  Precisely: the hypotheses, thresholds, gates, and decision rules are written
  down (the `PREREGISTRATION*.md` files) before running, which removes the
  temptation to adjust the analysis until a result appears.
- **Pooled versus matched-pair readout**: two ways to grade the reader. Pooled
  (the headline) mixes all episodes from all worlds and reads identity; it
  measures one world-identity direction shared across independent episodes.
  Matched-pair uses the identical-twins design; it measures whether the difference
  is detectable under the tightest control. Detectable under the twins design is not
  the same as readable across independent episodes, which is why the pooled readout
  is the one that carries the claim.
- **Capacity ceiling (sysid-aux)**: how loudly could this notebook say it, if we
  forced the creature to study the answer directly? Precisely: an extra head,
  trained on the drift itself, measures the most the architecture could ever
  decode, which puts the incidental result in context. This connects to a known
  point that a model good at prediction is not automatically good at the thing you
  actually want (Lambert et al., 2020).

---

## 8. What We Found So Far (updated 2026-10-02)

The story in nine acts, with real numbers. Remember: 0.5 is blind guessing, 1.0
is perfect, and 0.65 is the pre-registered bar. Acts 1 to 5 are the original arc.
Acts 6 to 9 are the work from September 2026, which narrowed the headline claim in
one important way and tested it in four others.

**Act 1: the inspector is nearly perfect (Experiment A).**
L0 photocopy: inspector 0.523, which is chance, exactly as required (the honesty
check passes). L1 pixelation: inspector 1.000. L2 smearing ink: inspector 0.993.
The leakage audit is clean everywhere, and when a tiny reward difference is
planted on purpose the audit correctly sounds the alarm (the leak jumps to about
0.98). Verdict: the fakes are clearly catchable, and only through the substrate.
H1 is supported.

*Correction, 2026-09-27.* Those ceilings were measured with a near-perfect
detector. Give the inspector the same blurry glance the creature gets, and the
ranking of the fakes reverses. At the detector noise that defines the L3 band, the
L2 smears read 0.646 and 0.618 while the L3 forger reads 0.928; the L2 artifacts
only reach 1.000 when the detector is a hundred times sharper. So the project no
longer says "L2 is more detectable than L3 yet only L3 is encoded". The clean
matched comparison is L1 against L3: an L1 seam sitting inside the same band
(0.873 to the inspector) leaves the creature at chance (0.533), while an in-band
forger is encoded. Detectable at the creature's own resolution turns out to be
necessary but not sufficient.

**Act 2: the daydreamer does not care (Experiment B).**
The prediction-only creature, living in a world whose ink smears (L2), writes
nothing about it in its notebook: target about 0.51 at every drift strength, while
the speed control reads about 0.90 (the notebook is perfectly legible, just
silent on this topic). The prediction-error channel shows a faint 0.596 at the
strongest drift. Longer imagination horizons: no lift. The stronger reader (a
random forest): nothing. This is the headline negative: detectable does not mean
encoded. Nothing about predicting the next moment requires caring whether the drag
slowly wobbles.

**Act 3: even hunger does not make L2 matter (B-v2, B-v3).**
Make it life and death: the survival creature at L2 drift reads 0.523 (plus or
minus 0.045) in the authoritative replication, and an earlier run's 0.595 did not
reproduce. A steady per-episode handicap (B-v3) reaches 0.610, interval [0.585,
0.634], still under the bar. Even force-feeding the answer (the sysid capacity
ceiling) saturates the pooled readout at only 0.596. This is a strengthened
negative: for hand-tuned artifacts, survival pressure does not create incidental
encoding.

**Act 4: the forger's fake flips the result (L3, hidden=8).**
Swap the hand-tuned knob for G_motion, the learned forger (gate-0 passed,
inspector 0.928). Now the creature encodes the world: pooled target 0.752 (90% CI
[0.698, 0.807], which clears the bar; 8 of 10 seeds clear it), while the untrained
newborn reads 0.488 (a clean floor) and the daydreamer reads 0.573. Every referee
check passes: reward is not decodable (0.541, clean in 10 of 10 seeds), zero deaths
(all pools 110 of 110), L0 sits at 0.517 with equivalence accepted, and the result
survives the nonlinear probe.

Then comes the deflating objection: behavior alone reveals the world at 0.803,
better than the state probe, so maybe the reader is just seeing the limp. The
pre-registered per-timestep control answers it: after surgically subtracting the
behavior trace, 0.726 remains (90% CI [0.685, 0.765]; 9 of 10 seeds clear the bar;
the quadratic version is 0.721). A later audit widened that subtraction to also
remove the creature's absolute position and heading, the one covariate the two
worlds' different motion laws could smuggle in; the signal barely moved, to 0.723,
so it is not position wearing a disguise. And the control is honest on its own
tests: the untrained creature's residual reads exact chance (0.498) even though its
raw behavior reveals the world at 0.645.

A second objection arrived later and is worth its own line. Controlling for behavior
does not automatically control for input: a notebook that passively mirrors
what the creature sees would survive the behavior control, because the view under a
different motion law is itself different. A readout-only control, frozen before it
ran, subtracts the observation stream too. The signal holds at 0.731 (90% CI
[0.690, 0.772], 8 of 10 seeds). Subtract the senses and the behavior together, with
a nonlinear reader, and it falls to 0.654, right at the bar with the interval
straddling it. The honest summary is that roughly two thirds of the above-chance
signal survives a one-step mirror of everything the creature just saw and did.

**Act 5: a second forger complicates the moral (gate-0 recal plus hidden=7).**
The pre-registration required a replication with a second in-band forger. The
first attempt, hidden=4, blew through its gates (untrained floor 0.891, reward
leak clean in 0 of 10 seeds, engagement 30%), so it was UNINFORMATIVE, not a
negative. The cause: it had been calibrated on the wrong world settings before a
fix. The fix: gate-0 became a committed, runnable per-capacity check, and a fresh
recalibration sweep re-froze the second capacity at hidden=7 (inspector 0.922, in
band; floor 0.566, elevated but within tolerance). The hidden=7 run of 10 seeds
then passed every gate. Results: the survival target is 0.737 (90% CI [0.688,
0.780], 8 of 10 seeds), and the signal left after the behavior control replicates almost
exactly at 0.722 versus 0.726 at hidden=8. But the predictor now also reads 0.714,
only 0.023 behind survival, so the pre-registered "survival must beat predictor by
0.05" requirement fails. The reading: the hidden=7 forger is a coarser, more
behaviorally obvious fake, so at this capacity every trained creature picks it up.
The survival-specific part of the claim is therefore artifact-conditional; it held
for the subtler hidden=8 forger. (The sensory control was later run at hidden=7 as
well, and there the dissociation reappears: survival holds 0.684 while the
daydreamer, level with it on the raw probe, falls to 0.589.)

**Act 6: it is not survival alone. It is survival plus imagination.**
This is the act that changed the headline, and it arrived on 2026-09-27.

Every creature in Act 4 had two jobs, not one. It had to stay alive, and it also
carried a decoder: a second head trained to predict what it would see and feel
next. That was always in the architecture, but the write-up credited the result to
survival. So the obvious experiment is to take the imagination away and run the
same thing.

Take it away and the signal collapses. The survival creature with no decoder reads
**0.601** (90% CI [0.549, 0.654]), under the bar, on the same machine where the
decoder-carrying creature reads **0.730**. Prediction alone, as Act 4 already
showed, reads 0.573. So at this training budget neither job did it alone: at 300
updates the signal belongs to the two trained together. One possible account is that
the decoder makes the notebook carry information about the sensory stream, and
foraging under a flawed motion law shapes it into a world-identity direction. That
account was never tested, and a decoder-free creature trained half again as long did
read above the bar, with too much skill to count as a matched comparison.

The published wording "encoded by the survival objective, uniquely" was narrowed
accordingly, and the films were corrected to match.

One confound stayed open. The decoder-carrying creatures also forage better, so the
decoder's effect on the notebook is not separated from its effect through a better
policy. The pre-registered check for that ran on 2026-09-30: train the no-decoder
creature longer until its foraging return matches, then probe it. The probe came
back at **0.717**, above the bar. But the match missed. The arm's return landed
0.0131 past the top of its frozen window, so by the rule fixed in advance the run
yields **no verdict** on the confound, because you cannot credit encoding to
matched skill that was in fact exceeded. Worth knowing how small that miss is:
0.0131 against a standard error of 0.096 on the same quantity, about one seventh of
one standard error. The window was narrower than the noise it was gating. The
result is reported as attempted and inconclusive rather than reinterpreted after
the fact, which is what pre-registration costs when it bites.

**Act 7: other forgers reproduce the direction, not the size.**
Act 5 changed capacity but kept the same forger recipe and seed. Two later runs
trained genuinely independent forgers. A hidden=10 forger at a new seed reads 0.639
on the cloud machine and 0.612 when re-measured on the owner's GPU. A second
hidden=8 forger at another new seed reads 0.676, and misses the predictor margin by
0.0009, which under the frozen table reads as "margins fail" at the boundary.

Both land above their untrained floors and above their daydreamer arms, and both
land under the 0.65 bar once the interval is taken into account. So the direction
replicates and the magnitude does not travel: 0.752 belongs to the specific forger
the claim came from. The machine explains about 0.03 of the gap and not the rest.

**Act 8: is it really the seam? (substrate grounding).**
If the creature is reading the forger's brushstroke, then sanding the brushstroke
off should silence it, and sanding half of it off should half-silence it. Blend the
seam out in graded steps and the signal falls monotonically from 0.752 to 0.506,
perfect rank correlation, while the untrained arm stays flat throughout. A texture
knockout with a dose response and a sensory lesion fit the same picture. This shows
the reading depends on the difference in the motion law, not on something the
apparatus manufactures. It does not show the creature represents that the law was
learned: the texture knockout only shows that the original reading direction does not
carry over to random jitter, and no creature was ever raised in a jittered world.

**Act 9: an audit of the bookkeeping, and what it moved.**
At the very end of the work a dependency surfaced: the way episodes were split into
cross-validation folds depended on the installed software versions. Two valid
splits existed, and GPU and CPU runs had been scored on different ones. A
pre-registered re-score of the saved dumps under a stack-independent split
reproduced every published number under the original split, and moved the hidden=8
readouts up by about 0.02 on every arm (survival 0.774, behavior-controlled 0.750).

No adjudicated verdict changed, and the contrasts the claims rest on moved by at
most 0.007, because the shift lifts every arm together. Two clauses written against
fixed absolute thresholds did flip, and both are reported as open rather than
quietly re-run: the drift-zero chance control now reads 0.539 and its equivalence
test does not accept at ten seeds, and the hidden=7 untrained floor, marginal from
the start, sits at its tolerance. Buying the first one would take about forty extra
drift-zero seeds, and the quantity the claim actually rests on does not move with
it, so it was left open and said out loud.

**Honest caveats, current scoreboard:**

- The positive needs both jobs. Survival alone reads 0.601 and prediction alone
  0.573; together they read 0.752 (Act 6). Whether the decoder acts on the notebook
  directly or through the better policy it buys is still open.
- The survival-specific dissociation depends on the artifact (Act 5), and the size
  of the effect depends on the individual forger (Act 7).
- Behavior is highly world-distinguishing on its own (0.803 at hidden=8). The 0.72
  figure is what survives a per-timestep control, 0.731 survives a sensory control,
  and 0.654 survives both at once with a nonlinear reader. A full-history control
  could in principle remove more.
- There are 10 seeds, one world family, one trunk (a 96-unit GRU), and one frozen
  oracle band. No other architecture or width has been tested.
- Detectability to a privileged inspector is not sufficient, and the matched
  comparison that shows it is L1 against L3, not L2 against L3 (Act 1 correction).
  The creature's own senses carry every tested flaw almost perfectly: an inspector
  that reads only those senses scores 0.99 (FINDINGS 10.1.1).
- The held-out and cross-recipe probes are done. The world signal transfers to a
  fingerprint the creature has never seen (0.773, rule passes) and to a forger built
  on a different recipe (0.684, rule passes). A frozen reverse test fails the bar
  (0.638), so transfer runs from subtle training artifacts and bidirectionality is
  unresolved. A common-garden control shows that which world the creature came from
  stays readable, weakly, after the motion law is put back to normal (0.666 forward,
  0.684 reverse, both clearing the bar, and decaying across the tail). Each creature
  keeps its own position and energy from the first part of the episode, so this does
  not separate memory from the traces the first part left behind; a test that does
  is specified and runs on the corrected creatures.
- L4, the forger with a detective on the team, remains unbuilt. Experiment C's
  emergence pilot resolved negative under its pooled population readout (FINDINGS
  13.D and 13.E): thirty generations of selection did not raise that readout.

Where results live: `docs/FINDINGS.md` (the narrative), `docs/PREREGISTRATION_L3.md`
section 12 (the dated lab log), `artifacts/expB2/*.json` (the committed numbers),
and `fullruns/` (the full bundles, ignored by git, with the latest pointer in
`results/LATEST_RUN.txt`). A verification script checks the numbers quoted in the
main documents against values recomputed from the committed per-seed results, and it
runs in CI on every change. It cannot catch an error inside the results themselves:
the training defect of Act 10 passed it for months.

---

## 9. Quick-Reference Glossary (A to Z)

| Term | Plain-English meaning | Where it lives |
|------|----------------------|----------------|
| A2C | Advantage Actor-Critic: player plus coach RL training; moves better than the coach expected get reinforced | `itasorl/agent_ac.py` |
| Actor | The policy head; the player choosing thrust, turn, and eat | `agent_ac.py` |
| Advantage | How much better an action turned out than the critic predicted | A2C training |
| AR(1) | A random walk with a pull back toward home; how L2 drag drifts within an episode | L2 surrogate |
| Artifact | The intended flaw that marks a fake world (the counterfeit's tell) | surrogate ladder |
| Artifact-conditional | A finding that depends on which specific fake was used (held at hidden=8, weaker at hidden=7) | PREREGISTRATION_L3 sec.12 |
| AUROC | 0.5 is a coin flip, 1.0 is perfect; the chance of ranking a real and fake pair correctly | every result |
| Authentic world | The real physics; the genuine bill | `patch_of_earth.py` |
| B-v2 | Experiment B version 2: survival pressure added (the three-creature comparison) | `run_expB2.py`, PREREGISTRATION.md |
| B-v3 | B-v2 with a constant per-episode drag offset instead of a wobble | `--drift-mode regime`, PREREGISTRATION_Bv3.md |
| Behavior audit (mediation) | Checking whether the probe reads thoughts or just the limp (different behavior per world) | `behavior_audit.py` |
| Behavior trace (bta/bts) | The per-timestep record of speed, energy, food, and drag saved with the states | trace dumps |
| Capacity | Network size; for G_motion it is the only difficulty knob of the L3 fake | `surrogate_l3.py` |
| Capacity ceiling (sysid-aux) | Force-feed the answer to measure the loudest the architecture could ever say it | `--sysid-aux` |
| Common-garden assay | Test every generation in the same kitchen to separate skill from survivorship | Experiment C (planned) |
| Confound | Any accidental clue that is not the substrate artifact (reward, length, metadata) | leakage audit |
| Contamination test | Deliberately plant a clue to prove the leakage audit would catch it (it does) | Experiment A |
| Critic | The value head; the coach scoring situations | `agent_ac.py` |
| Decoder | The imagination: rebuilds the next observation from the hidden state | RSSM |
| Detectability ceiling | The best possible detection score, measured by the privileged oracle | Experiment A |
| Determinism | Same seeds, identical replay; a chess game from a recorded opening | world design |
| Drag | Motion friction; the physics knob L2 perturbs and L3 replaces | dynamics |
| Drift (rollout drift) | The L2 artifact: drag slowly wanders during an episode | L2 |
| drift_sigma | How hard the L2 drag wanders (0.45 in the survival tests; inspector calibrated at 0.30) | configs |
| Dreamer | The well-known full RSSM-based agent; this repo uses a deliberately "lite" cut | context |
| Encoder | The eyes: compresses about 146 sensor numbers into a compact code | RSSM |
| Engagement gate | Proof the creature actually learned to survive; otherwise the run is uninformative | gates |
| ent_coef | The entropy bonus weight; keeps the player exploring | A2C config |
| Episode | One life segment (about 80 to 110 steps); the unit for probing statistics | everywhere |
| Experiment A | The agent-free inspector: how catchable is each fake? | `experiment_a.py` |
| Experiment B | The daydreamer's diary check: prediction-only incidental encoding | `experiment_b.py` |
| Experiment C | Evolution across generations (H3); not started | ITASORL.md |
| Fingerprint (generative) | The forger's signature flaws: blur, missing detail, mode collapse, learned noise | L3 |
| fullruns/ | Git-ignored folders holding complete run bundles; pointer in `results/LATEST_RUN.txt` | outputs |
| G (G_motion) | The forger: a small frozen network imitating real motion, swapped in as L3 physics | `surrogate_l3.py` |
| GAE | Generalized Advantage Estimation; smooths the coach's feedback (lambda 0.95) | A2C |
| Gate | A pre-registered pass or fail checkpoint; a failure means the run is UNINFORMATIVE, never "negative" | all prereg docs |
| gate-0 | The L3 entry gate: the forger's inspector score must land in 0.85 to 0.95 and the untrained floor near chance | `run_expA_l3.py` |
| GRU | Gated Recurrent Unit; the memory-notebook cell that produces h_t | agents |
| h_t | The notebook page at time t; the hidden state every probe reads | agents |
| Harsh metabolism | Energy burn set so a non-forager starves in about 50 steps; makes survival pressure real | B-v2 config |
| hidden=N (agent) | The creature's notebook size (memory usually 96) | agents |
| hidden=N (G_motion) | The forger's skill level; 8 is the headline fake, 7 is the second capacity | L3 |
| Incidental encoding | Learning to represent something you were never rewarded for; the H4 question | core concept |
| Interoception | Sensing your own body: hunger, speed, temperature (about 14 values) | observation |
| ITASORL (ItaSoRL) | The project name; the acronym is never spelled out; RL is reinforcement learning | README |
| L0 | The photocopy control: an identical world; all detection must sit at chance | ladder |
| L1 | The pixelated fake: observations snapped to a grid | ladder |
| L2 | The smearing-ink fake: drag drifts during the episode | ladder |
| L3 | The forger's fake: a learned network replaces the physics | ladder |
| L4 | The adversarial forger (with a detective on its team); not started | ladder |
| Leakage audit | Prove the answer is not written on the envelope (reward, length, metadata all at chance) | gates |
| Linear probe | The deliberately simple reader: logistic regression on episode features of h_t | probes |
| Matched pair | Identical twins split at noon into real and fake rooms from the same freeze frame | Experiment A, secondary readouts |
| Mode collapse | The forger never draws the rare serial numbers: rare events vanish from the fake | L3 fingerprint |
| Nonlinear probe | The stronger reader (a random forest); checks that a negative is not probe weakness | `run_expB_nonlinear.py` |
| obs_spec | The identical envelope: the observation format is the same across all worlds | `world.py` |
| Oracle | The privileged outside inspector of Experiment A | `experiment_a.py` |
| Patch of Earth | The terrarium: a 2.5D deterministic dissipative world with about 146 senses | `patch_of_earth.py` |
| Per-episode control | Subtract the average behavior per episode before probing (coarse; over-subtracts) | behavior audit |
| Per-timestep control | Subtract behavior moment by moment before probing (surgical; the deciding test) | behavior audit |
| Pooled readout | Mix all episodes and read world identity; measures one direction shared across independent episodes (the headline) | readouts |
| Positive control (speed) | The reader must at least read speed (about 0.9), or the notebook is illegible and the run is void | probes |
| Potential-based shaping | Warmer, colder food hints that provably never change the best strategy | B-v2 reward |
| Pre-registration | Predictions sealed in an envelope before running; removes after-the-fact knob tweaking | PREREGISTRATION*.md |
| Predictor (agent) | The daydreamer arm of the three-creature comparison | B-v2/L3 |
| Probe | The reader: an outside classifier reading world identity from h_t | probes |
| Readout, not reward | Read the diary, never pay for answers: identity is probed, never trained on | core invariant |
| Recal (recalibration) | Rerunning gate-0 across forger sizes on the corrected world; re-froze the second capacity at hidden=7 | `fullruns/l3_gate0_recal` |
| Regime drift | B-v3's version of L2: one constant drag offset per episode | `--drift-mode regime` |
| Residualization | Subtract everything behavior predicts, then read what remains | behavior audit |
| resid_trace | The result-file key for the per-timestep-controlled probe score (the ~0.72 numbers) | audit artifacts |
| ROPE | Region of Practical Equivalence: a Bayesian "certified boring, equals chance" test | `stats.py` |
| RSSM | Recurrent State-Space Model: encode, remember, predict; the creature's skeleton | `agent.py` |
| Second capacity | The pre-registered replication with a different in-band forger (hidden=7) | PREREGISTRATION_L3 sec.11 |
| Seed (SeedBundle) | The recipe card for randomness; separate streams for world, weather, ecology, and policy | reproducibility |
| SESOI | Smallest Effect Size Of Interest: the 0.65 bar plus margins over untrained and predictor, fixed in advance | prereg |
| Shuffled control | The reader must fail on scrambled labels (about 0.5), or it is cheating | probes |
| Signal A | Route 1 to noticing: the raw experience differs between worlds | theory |
| Signal B | Route 2: your predictions feel wrong in the fake (prediction-error texture) | theory |
| Substrate | The computational machinery under the world; the only allowed source of tells | core invariant |
| Surrogate | The stand-in world; the counterfeit | ladder |
| Surrogate ladder | The series of counterfeits, from L0 photocopy to L4 master forger | `world.py` |
| Survival (agent) | The hungry survivor arm (A2C plus GAE plus a prediction side task) | B-v2/L3 |
| Survivorship check | No missing bodies: zero early deaths so pools are unbiased (110 of 110) | gates |
| Tabula rasa | Blank slate: no pre-trained knowledge, everything learned from experience | core invariant |
| Tell | The specific flaw by which a counterfeit could be caught | ladder |
| TOST | Two One-Sided Tests: the frequentist "certified boring, equals chance" test | `stats.py` |
| Trace dumps | The full security-camera footage: per-timestep states plus behavior, saved for later audits | `--dump-states` |
| Unit of inference | Count students, not their answers: statistics at the episode or seed level, never the timestep | methodology |
| Untrained (agent) | The newborn arm with random weights; measures what the wiring gives away for free | B-v2/L3 |
| vf_coef, wm_coef | Loss weights: the coach's learning weight, and the "keep predicting your senses" weight | A2C config |
| World identity | The secret label (real or fake) that only the probe ever sees | everywhere |

---

## 10. References (APA 7)

These are the established, verifiable works the project's methods are built on.
The full and most current list, including recent preprints on evaluation
awareness and world-model probing, lives in `docs/ITASORL.md`.

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G.
(2021). Deep reinforcement learning at the edge of the statistical precipice. In
*Advances in Neural Information Processing Systems 34* (pp. 29304-29320).
https://arxiv.org/abs/2108.13264

Beane, S. R., Davoudi, Z., & Savage, M. J. (2014). Constraints on the universe as
a numerical simulation. *The European Physical Journal A, 50*(9), 148.
https://doi.org/10.1140/epja/i2014-14148-0

Bostrom, N. (2003). Are you living in a computer simulation? *The Philosophical
Quarterly, 53*(211), 243-255. https://doi.org/10.1111/1467-9213.00309

Chan, S. C. Y., Fishman, S., Canny, J., Korattikara, A., & Guadarrama, S. (2020).
Measuring the reliability of reinforcement learning algorithms. In *International
Conference on Learning Representations*. https://arxiv.org/abs/1912.05663

Demski, A., & Garrabrant, S. (2019). *Embedded agency* (arXiv:1902.09469). arXiv.
https://arxiv.org/abs/1902.09469

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering diverse
domains through world models* (arXiv:2301.04104). arXiv.
https://arxiv.org/abs/2301.04104

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018).
Deep reinforcement learning that matters. In *Proceedings of the 32nd AAAI
Conference on Artificial Intelligence* (pp. 3207-3214).
https://arxiv.org/abs/1709.06560

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control
tasks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP)* (pp. 2733-2743). Association for Computational
Linguistics. https://doi.org/10.18653/v1/D19-1275

Lakens, D. (2017). Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses. *Social Psychological and Personality Science,
8*(4), 355-362. https://doi.org/10.1177/1948550617697177

Lambert, N., Amos, B., Yadan, O., & Calandra, R. (2020). Objective mismatch in
model-based reinforcement learning. In *Proceedings of the 2nd Conference on
Learning for Dynamics and Control* (PMLR Vol. 120, pp. 761-770).
https://arxiv.org/abs/2002.04523

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver,
D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement
learning. In *Proceedings of the 33rd International Conference on Machine
Learning* (PMLR Vol. 48, pp. 1928-1937). https://arxiv.org/abs/1602.01783

Nguyen, J., Hoang, K., Attubato, S., & Hofstatter, F. (2025). *Probing (and
steering) evaluation awareness of language models* (arXiv:2507.01786). arXiv.
https://arxiv.org/abs/2507.01786

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
High-dimensional continuous control using generalized advantage estimation. In
*International Conference on Learning Representations*.
https://arxiv.org/abs/1506.02438

---

*If one sentence had to sum up the project: an outside inspector can always catch
the fake, but a creature only starts carrying that knowledge in its own head when
the fake is a learned imitation and its life depends on coping with it, and even
then the "only survivors notice" part depends on which forger you hired.*
