# ItaSoRL

<img width="2560" height="1280" alt="Social card: a detector that knows the true rules ranks a fake above a real one 87% of the time; the agent's own state reads 55% against a 65% bar (53% as first scored); against a learned fake its state reads 73%, which does not meet the registered rule because with both worlds real it reads 56%; a brain trained only to predict, fed the same senses, reads 72%" src="https://raw.githubusercontent.com/iLevyTate/ItaSoRL/main/docs/social-card/social-preview.png" />

I raised a small agent from nothing and asked whether it would notice it was
living in a fake world. Nobody paid it to look. A detector that knows the true
rules ranks a hand-made fake above the real world **87%** of the time (AUROC
0.873). That is a ranking score, the chance a fake episode is placed above a
real one, not the share of episodes labeled correctly. The agent's own state reads
**55%**, barely above a coin flip and well under the 65% bar, which I set for a
different experiment and applied to this run after the fact.
A learned fake reads **73%**, in agents trained both to survive and to predict
what they will sense next. Drop the prediction task and it reads **61%**.
Both numbers come from the one pair of evaluation worlds all ten seeds share. The
73% does not meet the rule I registered, because one honesty check fails: with
both worlds real it should read a coin flip, and on the registered pair of
worlds it reads 56%.

Then I checked what was doing the noticing. A brain trained only to predict,
fed the exact senses the surviving agent lived through, reads the fake within
about a point of the survival brain (72% against 73%). What finds the fake is
where the agent goes. A brain shaped to predict keeps what it saw there.

The agent numbers above come from a corrected trainer. A review in October 2026
found a bug in how the trainer valued the last step of an episode. Fixing it
moved the headline by less than a point. The run behind the 55% was repeated on
the fixed trainer as well and reads the same as before the fix (55% on the
scoring split used now, 53% as first scored; FINDINGS §22.1). The record of what
changed, and what did not get rerun, is in
[`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

The 90-second film is the one to watch. Voice and music are in. Nothing clipped
out. [ilevytate.github.io/ItaSoRL](https://ilevytate.github.io/ItaSoRL/)

Two films play on that page. *Detectable All Along* is the 90-second story,
released 24 August 2026 with the voice and music track and made on the original
trainer. Its closing line, that the creature noticed because survival made it
matter, is withdrawn: a brain trained only to predict, fed the same senses,
reads the fake within about 0.01 of the survival brain ([FINDINGS §17.6](docs/FINDINGS.md#176-who-drives-the-evaluation-episodes-revision-step-6)).
The captions-only cut that plays live on the page tells the corrected story.
*Two Minds* goes inside the creature's head: one brain, two worlds, one flaw,
every dot named after a real sensor, unit, or motor head, and the clue cells
ringed from measured data. It was re-rendered on the corrected record in
October 2026. Its motion is illustrative, and its 52% and 73% are two different
brains, an untrained one and a trained one, not one brain before and after. The
page prints these caveats beside both players. Provenance and rebuild commands
for both: [`assets/film/README.md`](assets/film/README.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iLevyTate/ItaSoRL/blob/main/notebooks/colab_gpu.ipynb)

**Google Colab (GPU):** [Open `notebooks/colab_gpu.ipynb` in Colab](https://colab.research.google.com/github/iLevyTate/ItaSoRL/blob/main/notebooks/colab_gpu.ipynb).
Make your own copy first (`File -> Save a copy in Drive`). The first cell
enforces that. Pick your own GPU runtime. Then run all cells: it clones the repo
and runs `python scripts/run_e2e.py`. The notebook reruns Experiments A and B and the L2
runs; it does not rerun the L3 runs behind the 73% or the corrected L1 run behind the 55%,
which retrain on CPU (see "Reproduce a headline number" under How to run).

Two illustrated walkthroughs from July 2026, printed before the October
corrections. Their numbers come from the original trainer, and a dated cover page
on each lists what has changed since:

- [**Plain-English series (PDF)**](docs/itasorl-series-plain-english.pdf): the July 2026 pre-correction walkthrough
- [**Research edition (PDF)**](docs/itasorl-series-research.pdf): the July 2026 pre-correction walkthrough, in research terms

---

## The question

A creature starts with no labels and no instructions. It lives here. Sometimes
the physics are the real program. Sometimes we swap in a copy with one rule
wrong: how the body moves when it pushes. Tiny seams. The kind a stand-in leaves.

Two questions. Keep them apart.

1. **Can anyone catch the copy?** A detector that already knows the real rules
   can. How well depends on how noisy its view is, so every flaw is scored at
   the same detector noise.
2. **Does the creature's state separate the two on its own?** We never reward
   spotting the fake. Nothing in training names it. We read its state
   afterward with a probe.

At L1 and L2, the hand-made flaws, the detector catches the copy and the
creature's state stays under the 0.65 bar, which was registered for the L2 survival
run and applied after the fact to the L1 run and to the prediction-only L2 run. At L2
it reads about a coin flip. At L1 it reads 0.551, with an interval that excludes
chance, which the record does not read as an absence.

At L3 the fake is a learned-dynamics fingerprint: a small net standing in for
the velocity law. Agents trained to survive and to predict their next
observation then carry a world signal of about **0.73**. Survival training
without the prediction task reads 0.61, under the bar. The signal rides on the
paths the foraging creature takes, and a brain trained to predict holds on to
it; why those paths carry more of the difference than scripted ones has not
been measured. A hand-written fake gets read too: a drag that depends on how
fast the body is moving, as the learned law does, though an untrained brain already
reads it outside the tolerance around chance, so only the margin over that brain
counts. Random jitter in the velocity,
read from brains raised under the learned law, reaches the bar at no setting tried,
even jitter that stays correlated over time.
Rerun with its 24-step drift matched exactly at the two most correlated settings and
within 1.1 percent at the uncorrelated one (a larger drift than the drag's), the tested
contrasts did not demonstrate that correlation raises the reading on the tested worlds and
jitter draws, and they leave room for a small rise, so these tests do not say why
(FINDINGS §22.3).

### Hasn't this been done before?

Parts of it. Four lines of work come close, and each differs in a specific way.

1. **Chatbots that know when they're being tested.** Large language models can
   often tell an evaluation from a real conversation. They read most of the
   internet first, so they arrive knowing what a test and a simulation are.
   This creature starts with no language and no concepts. Whatever its state
   carries came from the world it lived in.
2. **Looking inside networks for things nobody taught them.** Probing studies
   find untaught knowledge in trained networks, such as a game-playing model
   that tracks the board. Those facts are about things inside one world: the
   pieces, the score, the map. Here the fact is which world the agent is in.
3. **Robots that flag a sim-to-real mismatch.** Some robots carry a detector
   that fires when reality stops matching the simulator they practiced in.
   Someone built that detector and told it what to watch. Nothing here is told
   what to watch.
4. **Agents that work out which world they are in.** This is the closest
   relative. Meta-reinforcement learning (RL², VariBAD, PEARL, RMA) trains an
   agent across many environments, so working out which one it is in pays, and
   its recurrent state comes to track that. Our agents each train in one world,
   no objective term targets the condition, and the condition is whether the
   physics are the real law or a learned copy of it. The meta-RL results
   predict that training across both worlds would make the condition decodable.
   That mixture has not been run here.

So the claim is a measurement: how catchable a substrate flaw is from outside,
set against whether a blank-slate agent that lives with it comes to carry it in
its state unasked. The overclaims this rules out, and the works that rule them
out, are listed in [`docs/ITASORL.md`](docs/ITASORL.md).

---

## For researchers

Same two questions, in the terms the papers use. Can a privileged detector tell
authentic base dynamics from a substrate-flawed surrogate, at a stated
detector-side noise? Under a stated training protocol, is the condition
decodable from a from-scratch agent's recurrent state, beyond named baselines
and controls, with world identity read out after the fact and never rewarded?

Three invariants. All required, or the measurement collapses:

- **Tabula rasa**: no semantics are imported; the agent starts from nothing.
- **Readout, not reward**: world identity is *probed*, never trained or rewarded.
- **Substrate-grounded**: the only tells are artifacts of the computational substrate.

Detection difficulty is dialed with a **surrogate ladder** (L0 identical control,
L1 discretization, L2 rollout drift, L3 learned-model fingerprint, L4 adversarial).

Prior work holds pieces. Evaluation-awareness papers read test-vs-deploy signals
out of models that already know what a test is. Probing papers (AtariARI,
Othello-GPT) read incidental facts *inside* one fixed world. Mismatch detectors
(GalilAI, RAPT) are told to watch for the change. Meta-RL (RL², VariBAD, PEARL,
RMA) infers a latent task because training spans the tasks; here each agent
trains in one condition. This project measures the gap between how catchable
the fake is from outside and whether a blank-slate agent's state separates it
unasked. Axis-by-axis comparison: [`docs/ITASORL.md`](docs/ITASORL.md).

### Hypotheses

- **H1 (detectability)**: a privileged detector distinguishes the surrogate from the authentic world. **Supported** at L1, L2, and L3; at one shared detector noise the L2 flaws read weaker than L3 (FINDINGS §15).
- **H2 (substrate-grounding)**: the signal depends on the substrate artifact, tested by ablation. At L3 decoding falls with the dynamics difference (FINDINGS §14). A hand-authored drag that, like the learned law, depends on the agent's own velocity is read as well, measured as a margin over an untrained arm whose floor (0.612) sits outside tolerance (§17.9), so nothing here is specific to learned dynamics. Random velocity jitter, read from agents raised under the learned law, reaches the bar at no setting tried, including strongly autocorrelated jitter matched on its 24-step deviation, so the comparison names no mechanism (§22.3); these jitter readings hold for the tested agents, worlds and jitter draws. At L1 the survival agent did not meet the encoding criterion at the in-band grid (§14.7).
- **H3 (emergence under selection)**: selection raises world decodability. **Flat contrast, registered verdict uninformative.** Over 3 lineages and 30 generations the contrast is flat under both the pooled and the per-individual readout, but the speed control (gate 5) fails, and the preregistration interprets no contrast unless every gate passes, so neither readout yields a verdict. The payoff for knowing the world (gate 1) is weak and falls short only of a draft margin that was never ratified (FINDINGS §13.F; the earlier record is §13.D, §13.E).
- **H4 (incidental encoding)**: the condition is decodable from a from-scratch agent's state without reward. At L3 the registered rule is **not met**, because the L0 (authentic-versus-authentic) gate does not pass on the registered world pair, 0.559; its three **decodability clauses are met** for survival with the next-observation auxiliary at 300 updates, corrected trainer: 0.733 (FINDINGS §17). Not met at L1 or L2.

### Current status

| Component | State |
|-----------|-------|
| World ("A Patch of Earth" v0) | built, verified deterministic, snapshot-exact |
| Ladder L0 / L1 / L2 | implemented and validated |
| Ladder L3 (learned-dynamics surrogate) | implemented, gate-calibrated |
| Experiment A (detectability ceiling, agent-free), L1 | **done** |
| Experiment A, L2 | **done** |
| Experiment B (incidental detection), L2 arc | **done: did not meet the registered encoding criterion** (its prediction-only world model received each action one step late, so that arm was a weaker predictor than designed; [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-16). Retrained with the timing corrected, it gives the same result: 0.544 at drift 0.45 against 0.517 with the old timing, three seeds each, both inside the recorded interval [0.429, 0.591] (`artifacts/expB_aligned_2026-10-18/`) |
| Experiment B, L1 organism + H2 battery | **done: did not meet the encoding criterion registered for L2 (H_B2), applied to L1 after its run ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-16), at in-band Δ=0.023 (survival 0.551 on the corrected trainer, 0 of 10 seeds at the bar, FINDINGS §22.1; 0.533 as first recorded on the original trainer, §14.7)** |
| Experiment B, L3 (learned-dynamics) | **registered rule not met (L0 gate open on the registered world pair); decodability clauses met at n=10** with survival plus the next-observation auxiliary at 300 updates, corrected trainer: 0.733 against predictor 0.589 and untrained 0.523, on the one evaluation-world draw all seeds share. Without the auxiliary 0.613, not met; the auxiliary lead compares the tested regimes at the registered budget and does not establish general necessity (FINDINGS §17). Historical: 0.752 on GPU, 0.730 and 0.601 on CPU (§10.8); an independently trained fingerprint 0.639 (§10.9); prefix condition remains decodable after restoring authentic dynamics, carried mostly by the physical footprint the prefix leaves rather than by memory, and retention under identical input is not shown (FINDINGS §17.7, §10.6.1). Details in [Key result](#key-result) |
| Experiment B, L3 matched-world confirmation (B1, B2) | **uninformative.** A confirmation frozen before it ran, on new seeds 10-19 with authentic and surrogate episodes paired on the same world seeds: survival reads 0.800 and the pellet objective 0.843, but the untrained arm reads 0.614, outside the 0.1 floor. A linear sensory control frozen after those raw outcomes and the untrained readings were known leaves 0.761 and 0.780, with the untrained arm at 0.596 and its interval [0.577, 0.615] crossing 0.60 (FINDINGS §21) |
| Experiment B, L3 prospective matched-world confirmation (PB1, PB2) | **met on both.** That sensory control and its rule, frozen before any agent was trained and amended once before any result (the speed gate moved to the trained arm and the untrained mean; under the first text one untrained seed at 0.724 leaves both runs uninformative), on fresh seeds 20-29 and fresh matched worlds: survival 0.709 [0.675, 0.743] and pellets with death off 0.743 [0.714, 0.771], untrained 0.598 [0.572, 0.624], inside the floor by 0.002. The strict interval reading is not met, and nonlinear and longer-history variants of the control leave 0.644 to 0.725 (FINDINGS §24) |
| Trainer correction (2026-10) | **done**: GAE truncation bootstrap fixed; both primary runs rerun; effect +0.003 and +0.012 ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), FINDINGS §17) |
| Experiment C (emergence under selection) | **done: flat contrast, registered verdict uninformative.** The speed control (gate 5) fails and the preregistration interprets no contrast unless every gate passes, so neither the pooled nor the per-individual readout yields a verdict; gate 1 falls short only of a draft margin that was never ratified (FINDINGS §13.F) |
| Ladder L4 (adversarially-hardened surrogate) | not started |

### Key result

*The L3 primary comparisons below were rerun with the corrected trainer
(FINDINGS §17), and so was the L1 organism run (§22.1). The matched-world
confirmation (§21), its prospective confirmation (§24) and the goal-and-stakes runs
(§18) ran on the corrected trainer from the start. Every other agent
number is historical and labeled so in
[`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md).*

Hand-authored seams (L1 discretization, L2 rollout drift): catchable from the
outside, not decodable inside at the encoding criterion registered for L2 (H_B2),
which was applied to L1 after its run ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md),
2026-10-16). An L2 artifact a
nearly noiseless external detector finds at AUROC ≈ 0.99 leaves the
prediction-only agent's state near chance (≈ 0.5). That agent's world model
received each action one step late, so it was a weaker predictor than designed;
the prediction-error and open-loop channels are the most exposed
([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-16). Retrained on one machine
under both timings, three seeds each, the drift-0.45 state target reads 0.544 ± 0.037
with the timing corrected and 0.517 ± 0.056 with the old timing, both inside the
recorded row's interval [0.429, 0.591], so the negative stands with correct timing; the
prediction-error channel reads 0.599 and 0.596
(`artifacts/expB_aligned_2026-10-18/`, `artifacts/expB_legacy_local_2026-10-18/`). Give that detector the input
noise the L3 gate uses and the L2 artifacts read 0.62 to 0.65 on one noise draw,
and 0.60 and 0.50 averaged over eight (FINDINGS §15), so
the matched case is L1: an in-band quantization grid (detector AUROC 0.873; 0.871 on
the stack-independent partition) leaves the survival state at **0.551** [0.531, 0.570]
on the corrected trainer (n = 10, 0 of 10 seeds at the bar; FINDINGS §22.1). The
original trainer scored the same way reads 0.552, and 0.533 as first recorded (§14.7).
The readout is not blind to this kind of seam:
at a coarser grid outside the band, Δ = 0.06, the same readout registers the seam, with
the untrained arm at 0.672 [0.640, 0.704] (6 of 10 seeds at 0.65) and the survival arm
at 0.643 [0.619, 0.667]. Survival minus untrained there is -0.029 [-0.062, +0.005]:
the readout registers the seam on an untrained trunk, and survival training does not
raise it. The survival arm's 0.643 is under the bar, so the criterion is not met at
either grid (FINDINGS §22.1; `artifacts/expL1/l1_coarse_grid_corrected.json`).

**Corrected L3 result (FINDINGS §17).** Survival training with the
next-observation auxiliary, 300 updates, ten seeds, on CPU: the survival state
reads **0.733** (t-based 90% CI **[0.669, 0.797]**, 8 of 10 seeds at the 0.65
bar), the predictor arm 0.589 and the untrained arm 0.523, with paired margins
of +0.144 and +0.210 whose intervals clear the registered 0.05. These
readings are on the one evaluation-world draw all seeds share. The L0 gate
(both pools authentic) is open on the registered pair of evaluation worlds
(0.559, the largest of nine draws on record). On that same pair a second set of
survival agents, the drift-zero agents of the matched-world confirmation on seeds 10
to 19, reads 0.561 [0.527, 0.595] (TOST p = 0.717), so the separation repeats on new
agents (FINDINGS §21). Across eight independent pairs
the absolute deviation from chance averages 0.037, a TOST over pairs accepts
equivalence (p = 0.022) on the signed pair means; folded about 0.5 the same eight draws
give p = 0.129, not equivalent. That TOST is a diagnostic not named in the frozen spec.
The gate accepts on two pairs individually. On the
agents of the historical hidden-8 run at drift 0.45, all eight independent
draws clear the bar (0.728 to 0.860; the registered draw's 0.774 ranks 5 of 9),
SECURE under a rule frozen before the run. The registered verdict is therefore
not met (L0 gate open), while the decodability clauses are met. Without the
auxiliary the same protocol reads **0.613** [0.552, 0.675], not met; the
decoder adds **+0.120** [+0.065, +0.174]. That compares the tested regimes at the
registered budget: the registered auxiliary rule is not met, because C1 does not meet
its primary rule, and general necessity of the auxiliary is not established. The
bootstrap fix moved these two readings by +0.003 and +0.012.

**Matched worlds (FINDINGS §21).** A confirmation frozen before it ran trains the
same corrected recipe on new seeds 10-19 and pairs authentic and surrogate
episodes on the same world seeds. Survival reads 0.800 [0.772, 0.829] and a
pellet-seeking objective with death off 0.843 [0.825, 0.860], but both runs are
**uninformative**: the untrained arm reads 0.614, outside the 0.1 floor, so the
run cannot say what training added. A later control, frozen after those raw
outcomes and the untrained readings were known and before any trained arm was
scored under it, removes an in-fold linear prediction from the current and
previous observations. The untrained arm falls to 0.596, with an interval
[0.577, 0.615] that crosses 0.60, and the trained arms keep 0.761 and 0.780.
That control does not rule out nonlinear sensory transformations, longer
history, or the previous action. A confirmation of that control frozen before
any of its agents existed and amended once before any result (FINDINGS §24) trains fresh seeds 20-29 on fresh
matched worlds and is **met** for both objectives: survival 0.709 [0.675, 0.743]
and pellets with death off 0.743 [0.714, 0.771], with the untrained arm at 0.598
[0.572, 0.624], inside the floor by 0.002. The amendment moved the speed gate
from every arm to the trained arm and the untrained mean; under the first text one
untrained seed (0.724) leaves both runs uninformative. Its strict interval reading is not
met, and nonlinear and longer-history variants of the control leave 0.644 to
0.725 on those agents and 0.697 to 0.748 on B1 and B2. B1 and B2 are new agents, so these numbers sit
beside the C1 readings above and do not replace them. The earlier pellet run read
0.796 and was recorded as meeting every gate, but its leakage gate summed the homeostatic
reward rather than pellets, so full-rule validity was not established
([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-12).

**What does the noticing.** Fed the survival agent's own observation streams,
a trunk trained only to predict reads the world at 0.721, and one trained on
exactly the survival agent's experience at 0.738, against 0.733 for the
survival trunk. Driven by one scripted policy, every arm reads 0.55 to 0.58.
A decoder that never sees the brain, only the senses and the actions, reads
0.724, and summary statistics of the senses alone read 0.715. So the
signal rides on where the survival policy takes the body, and a trunk trained
to predict keeps it; at matched input the survival objective adds nothing
measurable, and "survival-specific" is withdrawn. A replay across the two
corrected runs puts part of the decoder's +0.120 in the trunk and part in the
paths the decoder-carrying policy takes, in shares that depend on the order of
the exchange (exploratory, FINDINGS §17.10).

**Bounded by budget.** Trained to 450 updates, the arm without the decoder
reaches 0.685 and the arm with it 0.821; the return interval is too wide to
say the two forage equally. The best-fitting
nonlinear control over observations, actions, and behavior leaves 0.620, under
the bar. Under identical input the prefix condition fades within a few steps,
and with the recurrent state zeroed the body's footprint still reads 0.662. A
hand-authored drag of the same one-step size as the Gaussian jitter, which like
the learned law depends on the agent's own velocity, is read through the original
direction (0.721) and by a fresh probe (0.754); the jitter is not. An untrained network
already reads the drag at 0.612, outside the 0.1 tolerance, so the margins over it (+0.109
and +0.142) are the comparison that carries meaning (FINDINGS §17.9). Correlated jitter does not reach the bar either.
With the 24-step deviation matched exactly at rho 0.95 and 0.99 and within 1.1 percent
at rho 0, the most correlated setting minus the uncorrelated one reads -0.017
[-0.051, +0.016], and the original sweep's contrast read +0.009 [-0.027, +0.045]. The
tested contrasts did not demonstrate an increase from coherence, and the corrected interval
still admits a small one. It also misses the 0.05 equivalence band by 0.001, so equivalence
is not established either, and the comparison names no mechanism (FINDINGS §22.3).

**The historical record (pre-correction trainer).** The surrogate is a
**learned-dynamics fingerprint**: a small net replacing the velocity law, not a
hand-tuned knob. On the published GPU run the **survival** agent's state,
trained with the next-observation auxiliary, separates the worlds. Untrained
sits at chance on these pools (about 0.49, and 0.52 under a nonlinear probe, a
value with no committed source; the committed random forest reads the untrained arm at
0.51 on the original partition and 0.557 [0.545, 0.569] on the stack-independent one, so
that near-chance floor does not hold on the stack-independent partition;
`artifacts/agent_free/rf_probe_l3_n10_historical.json`,
[`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-11). On the matched world pairs
of FINDINGS §21 the untrained arm reads 0.614. Prediction-only sits near chance (about 0.57).
Survival reads **0.752** (n = 10, t-based 90% CI **[0.698, 0.807]**, excludes
the pre-registered 0.65 bar; 8 of 10 seeds clear it). World is not decodable
from summed reward (AUROC 0.541, 10 of 10 seeds). Zero early deaths, every pool
110/110. L0 authentic-vs-authentic sits at 0.517 under the original
cross-validation split (0.539 under the stack-independent split, where the
ten-seed equivalence test no longer accepts; FINDINGS §16).

**Most of the signal remains after a per-timestep behavior control.** The agent
does move and forage differently in the two worlds. The full per-timestep behavior
trace alone decodes the world at **0.803**, better than the state probe (per-episode
behavior means alone read 0.689). The cheap
reading is that the probe is just reading behavior. A pre-registered
per-timestep control dumps every step's speed, energy, food, and drag,
residualizes the recurrent state on that trace in-fold with a linear map, and
probes what is left. What is left is **0.726** (t-based 90% CI **[0.679,
0.772]**, excludes the 0.65 bar; seed-level bootstrap [0.685, 0.765]; 9 of 10
seeds; quadratic variant 0.721). Adding absolute position and heading barely
moves it, to **0.723** (t-based 90% CI [0.676, 0.769]; 8 of 10 seeds). FINDINGS
§10.4.1. Untrained state under the same control is near chance (0.498) even though
untrained *behavior* decodes 0.645: under this control an untrained trunk keeps little
residual signal, a diagnostic relative to the basis and regressor used, not a showing that
every behavior-related contribution is removed in the trained agents. Prediction-only stays
near chance (0.574).
The control removes what its channels and model can express; behavior and
position information outside that basis, nonlinear or dependent on history, are
not excluded. An earlier per-episode-mean control had
under-estimated the signal at ~0.66 by over-removing, the attenuation the
synthetic tests predicted. Code: `scripts/audit_behavior_mediation.py`.
Artifacts: `artifacts/expB2/`. On the corrected agents the best-fitting behavior and
sensory control leaves 0.620, under the bar (FINDINGS §17.8).

A second in-band fingerprint (hidden = 7, frozen fallback after hidden = 4
failed its gates) leaves about the same after the behavior control: **0.722**
(t-based 90% CI [0.672, 0.773]) vs 0.726 at hidden = 8. That coarser artifact
is one every trained agent picks up (predictor 0.714 vs survival 0.737, misses
the pre-registered +0.05 dissociation). The survival-over-predictor margin was
therefore conditional on the subtler hidden = 8 artifact (and on the corrected
agents it disappears at matched input, FINDINGS §17.6). Both capacities
leave about **0.72** after the behavior control, with reward leakage and
survivorship clean.

Two boundary checks run in September narrow it. Take away the survival agent's
next-observation predictor, an auxiliary loss it trains alongside survival, and
the signal drops to **0.601** (t-based 90% CI [0.549, 0.654]; 1 of 10 seeds),
under the bar. That run was on CPU, so a pre-registered device control reran the
published setup on the same machine: **0.730** [0.668, 0.791], 8 of 10. At the
300-update budget neither objective alone reached the bar: prediction alone
reads about 0.59, survival alone 0.601, both together 0.73 to 0.75 (FINDINGS
§10.8). Survival alone trained to 450 updates reads 0.717, above the bar, and
its skill match failed (§10.8.1), so the statement holds for the tested budget only. A second
fingerprint, trained independently, keeps the survival-over-predictor margin
(**0.639** against 0.534 for prediction-only) but lands under the bar. A third
fingerprint at the headline capacity (new seed, hidden 8) reads **0.676**
[0.636, 0.717] against 0.627 for prediction-only and 0.550 untrained: above the
bar in the mean, but the lower bound is under it and the predictor margin misses
by 0.001. Replication of the full-strength result is not claimed (FINDINGS
§10.9).

A held-out probe (n = 10, `artifacts/expB2/heldout_l3_h8_summary.json`) splits
the rest. The world-identity direction still reads a held-out capacity variant
(transfer **0.773** vs untrained floor 0.569; the pre-registered rule passes).
The variant shares the training recipe, seed, and data, so that is robustness
inside one recipe (FINDINGS §10.6). A frozen reverse probe (train on the
coarser fingerprint, read the subtler one;
`artifacts/expB2/heldout_l3_h7_reverse_summary.json`) fails its bar at
**0.638**. Transfer goes one way. Under a common-garden control (identical
authentic tail after differing prefixes), tail-only state still recovers the
prefix world above the frozen bar on both directions (**0.666** forward,
**0.684** reverse). The last-8-step late tail decays toward chance (0.586 /
0.577). Prefix condition stays decodable, weakly, after authentic dynamics are
restored. This control cannot tell memory in the state from a footprint the
prefix left in body and world. The test that can has been run, on the corrected agents:
retention under identical input is not shown (replay 0.561), and with memory zeroed the tail
still reads 0.662, so this signal is mostly the footprint (FINDINGS §17.7; spec
[`docs/specs/2026-10-06-controlled-persistence-design.md`](docs/specs/2026-10-06-controlled-persistence-design.md)).
The original common-garden number, 0.557, used a since-fixed biased estimator
and is overturned; FINDINGS §10.6.1. Transfer numbers were unaffected.

A cross-recipe probe (n = 10, `artifacts/l3_crossrecipe/summary.json`) then
reads a gate-calibrated random-Fourier-features ridge law the agent never
lived with (**0.684** vs untrained floor 0.548; rule passes, machine-checked).
Thin: the t-based 90% CI lower bound clears the bar by 0.004, and 7 of 10
seeds sit above it. The direction reads two learned function classes fit on
the same data; independent data were not tested. Details:
[`docs/FINDINGS.md`](docs/FINDINGS.md),
[`docs/PREREGISTRATION_L3.md`](docs/PREREGISTRATION_L3.md).

---

## Repository layout

```
.
|-- README.md                   this file: the map
|-- LICENSE
|-- CITATION.cff                citation metadata (name, ORCID, version)
|-- pyproject.toml              package metadata (pip install -e .)
|-- requirements.txt            runtime dependencies
|-- requirements-dev.txt        pytest + dev tooling
|-- itasorl/                    core library (world, agents, experiments)
|   |-- world.py                World protocol and obs/action specs; first ladder sketch (superseded)
|   |-- patch_of_earth.py       PatchOfEarthV0 concrete world; the L1, L2 and L3 rungs (drift_mode)
|   |-- agent.py                recurrent world model (RSSM-lite)
|   |-- agent_ac.py             survival actor-critic (Experiment B-v2)
|   |-- experiment_a.py         agent-free L1 detectability oracle
|   |-- experiment_a_l2.py      agent-free L2 detectability oracle
|   |-- experiment_a_l3.py      agent-free L3 (learned-fingerprint) oracle
|   |-- experiment_b.py         incidental-detection harness
|   |-- experiment_b2.py        survival-coupled B-v2 pipeline
|   |-- experiment_c.py         Experiment C: world-coupled seams, fitness and detection panel
|   |-- experiment_c_gate1.py   Experiment C gate 1: world-only exploitability
|   |-- neuroevolution.py       Experiment C neuroevolution primitives
|   |-- surrogate_l3.py         L3 learned-dynamics surrogate (G_motion)
|   |-- surrogate_l1_families.py  L1 surrogate families for the H2 ablations
|   |-- surrogate_l3_families.py  held-out L3 surrogate families for transfer and ablation probes
|   |-- behavior_audit.py       behavior-mediation controls (residual probes)
|   |-- control_diagnostics.py  what each behavior or sensory control removes (held-out R^2)
|   |-- folds.py                versioned grouped-CV fold partitions
|   |-- l0_audit.py             L0 gate across independent world samples
|   |-- eval_protocols.py       readouts under own, scripted, or replayed policies
|   |-- persistence.py          controlled persistence test (replay, common state, reset)
|   |-- post_review.py          post-review readouts on the saved agents of a corrected run
|   |-- mechanism_readouts.py   intervention, surprise and adaptation readouts (goal-and-stakes spec)
|   |-- novel_readouts.py       salience, world-feature, erasure and development readouts
|   |-- optional_readouts.py    optional sensitivity readouts; they change no frozen verdict
|   |-- stats.py                TOST/ROPE equivalence, bootstrap AUROC CIs
|   |-- logschema.py            logging schema (the data contract of a run log)
|   `-- results_io.py           end-to-end run recording
|-- scripts/                    deterministic reproduction runners
|   |-- run_e2e.py              pytest + Experiments A, B and B-v2, recorded (no L3 run)
|   |-- run_expA.py ...         Experiment A/B runners
|   |-- run_expB2.py            Experiment B-v2 / L3 (GPU if available)
|   |-- audit_behavior_mediation.py  behavior-mediation audit on dumped states
|   |-- audit_stats_recheck.py  CI gate: quoted numbers and watched wording vs committed artifacts
|   |-- reproduce.py            tables without training, retrain recipes, anonymized supplement
|   |-- build_results_manifest.py  which implementation produced each committed result
|   |-- build_index.py          renders index.html from index.template.html + artifacts
|   `-- dump_brain_film_data.py per-unit world-signal for the Two Minds film
|-- index.template.html         the site, hand-maintained; numbers are {{placeholders}}
|-- index.html                  GENERATED from the template (do not edit by hand)
|-- assets/film/                films the site serves + README with provenance
|-- viz/                        film tooling: scene exporter, browser player, Two Minds renderer
|   |-- collect.py              sim -> viz/data/scene.json
|   |-- player/                 the 90-second film, live in the browser (beats.json, player.js)
|   |-- player/brain/           Two Minds canvas renderer (brain.js, brain-data.js)
|   `-- player/capture/         headless capture rigs -> MP4 (build-two-minds.ps1)
|-- docs/
|   |-- ITASORL.md              research plan
|   |-- ITASORL_world_spec.md   world specification ("A Patch of Earth" v0)
|   |-- FINDINGS.md             empirical results
|   |-- CORRECTIONS.md          dated record of what was wrong and what changed
|   |-- REVISION_2026-10.md     claim decisions and status of the 2026-10 revision
|   |-- RESULTS_MANIFEST.md     GENERATED: commit, config, trainer, status per result
|   |-- GATE_TABLE.md           GENERATED: every gate per run and fold partition
|   |-- CONTRAST_INTERVALS.md   GENERATED: seed-paired margin intervals
|   |-- METHODS_ARMS.md         what each arm trained on and was evaluated on
|   |-- REPRODUCE.md            how to regenerate the tables and retrain the runs
|   |-- specs/                  frozen designs and decision rules, dated
|   |-- LEARNING.md             running lab notebook / lessons log
|   |-- PAPER_OUTLINE.md        writeup outline + claims inventory
|   |-- PREREGISTRATION.md      B-v2 pre-registration
|   |-- PREREGISTRATION_Bv3.md  B-v3 pre-registration
|   |-- PREREGISTRATION_L3.md   L3 pre-registration + deviation log
|   |-- PREREGISTRATION_C.md    Experiment C pre-registration (ran; verdict uninformative, FINDINGS 13.F)
|   |-- AUDIT_2026-07.md        research-integrity audit of July 2026 (dated snapshot)
|   |-- REVISION_AUDIT_2026-10-07.md  hand-off audit of the 2026-10 revision (dated snapshot)
|   |-- itasorl-series-plain-english.pdf  July 2026 pre-correction walkthrough (plain English)
|   |-- itasorl-series-research.pdf       July 2026 pre-correction walkthrough (research edition)
|   `-- figures/                result figures (.png) + provenance README
|-- artifacts/                  published summaries (committed); each file has one owner in docs/RESULTS_MANIFEST.md
|   |-- *.json                  verdicts, gate table, contrast intervals, manifest, budget curve, engagement, timing, surrogate diagnostics
|   |-- expA/  expB/            Experiments A and B: agent-free detectors, the L2 prediction-only arc
|   |-- expB_legacy_local_2026-10-18/  Experiment B retrained on the workstation with the original action timing
|   |-- expB_aligned_2026-10-18/       the same, with the corrected action timing
|   |-- expB2/                  B-v2, B-v3 and L3 run summaries, behavior audits, corrected-run summaries
|   |-- expC/                   Experiment C (selection)
|   |-- expH2/  expL1/          H2 batteries at L3 and L1; the L1 organism run and its readouts
|   |-- l3_crossrecipe/         cross-recipe transfer probe
|   |-- surrogates/             frozen L3 learned laws (G_motion)
|   |-- folds/  fold_rescore/   the frozen explicit partition; re-scores and gate-0 oracles under both partitions
|   |-- agent_free/             readouts that need no agent (gate-0 intervals, residual fields, forest probe, drag oracle)
|   |-- corrected_runs/         cells and post-run checks of the corrected-trainer runs
|   |-- reviewer_gaps_runs/     cells of the no-auxiliary, device-control, instance and gate-0 runs
|   |-- repro_check/  repro_check_c1_regen/  cross-stack replay checks; the C1 state dumps regenerated on the workstation
|   |-- goal_stakes/  goal_stakes_grouped_surprise/  pellet and death-off runs; their surprise readout with twins grouped
|   |-- balanced_confirmation/  balanced_pooled/  matched-world confirmation and sensory-echo controls; registered-pair readings
|   |-- l0_audit/               L0 across world samples, drift-0.45 world samples, the reverse direction
|   |-- policy_controls/  persistence/  control_diagnostics/  cross_replay/  readouts of FINDINGS 17.6 to 17.10
|   |-- population_readout/     pooled population probe on known-decodable agents, row- and world-grouped
|   |-- post_review/            post-review readouts on saved agents (FINDINGS 20)
|   |-- texture/                drag and jitter comparators, correlated-jitter sweeps
|   |-- surprise/               prediction error on identical input (FINDINGS 22.2)
|   |-- salience/  world_features/  mechanism/  development/  the four readouts of FINDINGS 23
|   |-- probe_sensitivity/  step_resolved/  probe-penalty sensitivity; within-episode decoding curve
|   |-- walk_speed/  pellet_dose/  head_overlap/  walk speed against goal direction; pellet density; head overlap
|   `-- bayes_factors/  budget_curve_dynamics/  interval-null Bayes factors (plug-in sensitivity); budget-curve slopes (agent-free)
|-- fullruns/                   e2e run bundles (gitignored; default output)
|-- results/LATEST_RUN.txt      pointer to the latest fullruns folder (gitignored; written by itasorl/results_io.py)
|-- notebooks/colab_gpu.ipynb   Colab end-to-end runner
`-- tests/                      pytest regression suite
```

### Documents

- [`docs/itasorl-series-plain-english.pdf`](docs/itasorl-series-plain-english.pdf): the July 2026 pre-correction walkthrough, illustrated, in plain English. Its numbers come from the original trainer, and a dated cover page lists what the corrected record changed.
- [`docs/itasorl-series-research.pdf`](docs/itasorl-series-research.pdf): the July 2026 pre-correction walkthrough in research terms (design, control battery, pre-registered results, open questions), with the same dated cover page.
- [`docs/ITASORL.md`](docs/ITASORL.md): the research plan, core question, prior work, hypotheses (H1 to H4), experiments (A/B/C), the surrogate ladder, validity audit, statistics, and engineering architecture.
- [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md): the world specification, "A Patch of Earth" v0, the 2.5D representation, fields and forcing, dynamics, ecology, the ~146-dim observation, ladder attachment, and confound management.
- [`docs/FINDINGS.md`](docs/FINDINGS.md): empirical results from the first build-and-test cycle.
- [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md): the dated corrections record, starting with the October 2026 trainer bootstrap fix, and what each change does not establish.
- [`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md): for every committed result, the commit, configuration, seeds, surrogate, budget, fold partition, device, and trainer, and whether it is historical or corrected.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): regenerate every table from committed artifacts without training (`python scripts/reproduce.py tables`), or retrain a run.
- [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md): the writeup outline and a claims inventory linking every headline number to its committed artifact.
- [`docs/LEARNING.md`](docs/LEARNING.md): the running lab notebook (methods lessons, dead ends, decisions).
- [`docs/PREREGISTRATION_L3.md`](docs/PREREGISTRATION_L3.md), [`PREREGISTRATION_Bv3.md`](docs/PREREGISTRATION_Bv3.md), [`PREREGISTRATION.md`](docs/PREREGISTRATION.md), [`PREREGISTRATION_C.md`](docs/PREREGISTRATION_C.md): pre-registrations (with deviation logs) for the B-v2, B-v3, L3, and C experiments (Experiment C ran; its registered verdict is uninformative, FINDINGS 13.F).
- [`docs/AUDIT_2026-07.md`](docs/AUDIT_2026-07.md): a skeptical research-integrity audit of July 2026 (numbers, statistics, pre-registration timing, citations), kept as a dated snapshot; it predates the October 2026 corrections, which are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).
- [`docs/REVISION_AUDIT_2026-10-07.md`](docs/REVISION_AUDIT_2026-10-07.md): the 2026-10-07 hand-off audit of the revision, kept as a dated snapshot; its items are closed in [`docs/REVISION_2026-10.md`](docs/REVISION_2026-10.md).
- [`assets/film/README.md`](assets/film/README.md): the two films, where each file came from, and the ffmpeg lines that rebuild the web encodes.

### Figures

Result figures live in [`docs/figures/`](docs/figures/). The Experiment A and B figures are
drawn from committed artifacts by `scripts/plot_record_figures.py`, with no training:

- `expA_ceiling.png`: L1 detectability ceiling vs grid spacing.
- `expA_L2_ceiling.png`: L2 detectability ceiling vs drift strength.
- `expB_incidental.png`: recurrent-state probe across the drift sweep (world-identity target, speed positive control, shuffled-label negative control).
- `expB_channels.png`: two incidental-detection channels (recurrent state vs prediction error).
- `expB_kstep.png`: effect of a longer-horizon objective on encoding.
- `budget_curve.png`: the corrected arms against the training budget, from
  `scripts/build_budget_curve.py` and `artifacts/budget_curve.json`.

To redraw them, run `python scripts/plot_record_figures.py` from the repo root. The
`scripts/run_exp*.py` runners and `run_e2e.py` still save a titled figure of the run they just
made over these files. Those are run outputs, so restore the committed copies afterwards with
`git checkout -- docs/figures`.
Per-figure provenance (which script, which doc section) is tracked in
[`docs/figures/README.md`](docs/figures/README.md).

---

## How to run

Get the code. A shallow clone has every file the runs need:

```bash
git clone --depth 1 https://github.com/iLevyTate/ItaSoRL
cd ItaSoRL
```

Earlier commits added film renders that later left the tree, so a full clone
downloads about 260 MB of history for the main branch, and about 690 MB with every
branch, since the run branches carry saved agents. Checking the commit hashes recorded in
[`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md) needs that history: clone
without `--depth`, or run `git fetch --unshallow` inside the shallow clone.

Dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, first: CPU-only torch (Linux, Windows)
pip install -r requirements.txt      # runtime deps
pip install -e .                     # or: install itasorl as an editable package
```

Reproduce the experiments. Each is deterministic given its seeds on the stack and machine
that wrote the result; elsewhere a replay on the pinned CPU stack agrees to a measured
tolerance, not to the bit ([`docs/REPRODUCE.md`](docs/REPRODUCE.md), "Supported-stack replay
route"; [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), 2026-10-16). Run from the repo root;
the runners write run-specific figures over `docs/figures/`, so restore the committed copies
afterwards with `git checkout -- docs/figures`:

```bash
python scripts/run_e2e.py --quick   # Experiments A and B and the L2 B-v2 runs (reduced scale), recorded; no L3 run
python scripts/run_expA.py          # Experiment A, L1
python scripts/run_expA_l2.py       # Experiment A, L2
python scripts/run_expB_full.py     # Experiment B: recurrent-state probe
python scripts/run_expB_surprise.py # Experiment B: prediction-error channel
python scripts/run_expB_kstep.py    # Experiment B: open-loop horizons
python scripts/run_expB_gap.py      # Experiment B: engagement + delta objective
python scripts/run_expB_nonlinear.py # Experiment B: nonlinear probe check
python scripts/run_expB2.py         # Experiment B-v2: survival-coupled
```

### Reproduce a headline number

None of the commands above retrains the L3 or corrected L1 runs. The three numbers the
page opens with come from these (details, stacks and tolerances in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md)):

- **0.733**, the L3 survival state on the corrected trainer (`C1`). `python
  scripts/reproduce.py tables` recomputes it and every committed table from the per-seed
  artifacts in minutes. `python scripts/reproduce.py retrain C1 --execute` retrains it on
  CPU in hours (`retrain C2 --execute` for the 0.613 arm without the predictor; without
  `--execute` each prints its steps). The saved agents are also on the public branches
  `run/C1-rerun` and `run/C2-rerun`, from which the state dumps rebuild without training
  (`docs/REPRODUCE.md`, section 1).
- **0.551**, the L1 survival state on the corrected trainer (`L1-CORRECTED`). Committed
  in `artifacts/expB2/corrected_L1-corrected.json`;
  `python scripts/reproduce.py retrain L1-CORRECTED --execute` retrains it on CPU.
- **0.873**, the detector AUROC at the in-band L1 grid. It needs no agent and runs in
  seconds, and it reproduces on the legacy fold partition; the stack-independent
  partition gives 0.871:

```bash
ITASORL_FOLDS=legacy python scripts/run_expA_l1.py --out <dir>/gate0_l1_legacy.json --deltas 0.022 0.023 0.024
ITASORL_FOLDS=explicit python scripts/run_expA_l1.py --out <dir>/gate0_l1_explicit.json --deltas 0.022 0.023 0.024
```

A quick smoke test of the Experiment B pipeline:

```bash
python -m itasorl.experiment_b
```

Run the test suite (pytest ships in the dev requirements):

```bash
pip install -r requirements-dev.txt
pytest -q
```

**Google Colab (GPU):** [Open in Colab](https://colab.research.google.com/github/iLevyTate/ItaSoRL/blob/main/notebooks/colab_gpu.ipynb) (same notebook as the badge at the top). Make your own copy first (`File -> Save a copy in Drive`; the first cell enforces this), enable a GPU runtime, then run all cells.

**Local Jupyter / VS Code:** open [`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb) from this repo; it auto-detects local mode (no Drive/download cells).

**Readout-only reruns on the saved agents:** the ten survival agents of the historical hidden-8 held-out run (96-unit trunk; 8 is the surrogate's capacity) behind the world-sample sensitivity result (FINDINGS 17.5.1) are gitignored, so git does not carry them; they ship in the anonymized review supplement, beside the hidden-8 seed-2 run's agents and the saved agents and rerun cells of the corrected C1 and C2 runs (`fullruns/corrected_l3_h8_wm/` and `_nowm/`, per-file digests in `SHA256SUMS`); the full archive, built where the runs live, carries the whole run folders (`docs/REPRODUCE.md`). `python scripts/fetch_saved_agents.py --local-dir <folder>` finds them in a folder you hold (`--drive-dir` for a mounted Drive folder, `--url` for a zip or tar of them) and checks every file's sha256 against `artifacts/l0_audit/d045_world_samples_l3_h8_heldout.json`; the Colab notebook's Extra checks section does the same from a form field and then runs `scripts/run_world_sample_sensitivity.py` and compares its per-draw means and verdict with the committed artifact.

---

## What to read first

[`docs/FINDINGS.md`](docs/FINDINGS.md) is the record. [`docs/ITASORL.md`](docs/ITASORL.md)
is the plan. [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md) is the world.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff). GitHub renders a
"Cite this repository" button from it. `cffconvert -f bibtex` if you want BibTeX.

## License

The code and the record are MIT licensed ([`LICENSE`](LICENSE)); the 17 web fonts
under `assets/` (IBM Plex Mono, Hanken Grotesk, Space Grotesk) are under the SIL
Open Font License 1.1, with their copyright lines and the full license text in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
