# ItaSoRL

<img width="2560" height="1280" alt="GitHub social preview 1280x640@2x" src="https://github.com/user-attachments/assets/e1af8c43-470d-4758-baef-7859619b24ac" />

I raised a small agent from nothing and asked whether it would notice it was
living in a fake world. Nobody paid it to look. A detector that knows the true
rules catches a hand-made fake **87%** of the time. The agent's own state reads
**53%**, a coin flip. A learned fake reads **73%**, in agents trained both to
survive and to predict what they will sense next. Drop the prediction task and
it reads **61%**.

Then I checked what was doing the noticing. A brain trained only to predict,
fed the exact senses the surviving agent lived through, reads the fake just as
well. What finds the fake is where the agent goes. A brain shaped to predict
keeps what it saw there.

Those numbers come from a corrected trainer. A review in October 2026 found a
bug in how the trainer valued the last step of an episode. Fixing it moved the
headline by less than a point, and the record of what changed, and what did not
get rerun, is in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

The 90-second film is the one to watch. Voice and music are in. Nothing clipped
out. [ilevytate.github.io/ItaSoRL](https://ilevytate.github.io/ItaSoRL/)

Two films play on that page. *Detectable All Along* is the 90-second story,
released 24 August 2026 with the voice and music track. *Two Minds* goes inside
the creature's head: one brain, two worlds, one flaw, every dot named after a
real sensor, unit, or motor head, and the clue cells ringed from measured data.
Provenance and rebuild commands for both: [`assets/film/README.md`](assets/film/README.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iLevyTate/ItaSoRL/blob/main/notebooks/colab_gpu.ipynb)

**Google Colab (GPU):** [Open `notebooks/colab_gpu.ipynb` in Colab](https://colab.research.google.com/github/iLevyTate/ItaSoRL/blob/main/notebooks/colab_gpu.ipynb).
Make your own copy first (`File -> Save a copy in Drive`). The first cell
enforces that. Pick your own GPU runtime. Then run all cells: it clones the repo
and runs `python scripts/run_e2e.py`.

Two illustrated walkthroughs, same numbers, from the committed artifacts:

- [**Plain-English series (PDF)**](docs/itasorl-series-plain-english.pdf)
- [**Research edition (PDF)**](docs/itasorl-series-research.pdf)

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
creature's state stays near a coin flip, under the bar we set in advance.

At L3 the fake is a learned-dynamics fingerprint: a small net standing in for
the velocity law. Agents trained to survive and to predict their next
observation then carry a world signal of about **0.73**. Survival training
without the prediction task reads 0.61, under the bar. The signal rides on the
paths the foraging creature takes, where the fake physics part from the real
ones, and a brain trained to predict holds on to it. A hand-written fake that
drifts the same coherent way gets read too; white noise of the same size does
not.

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
- **H2 (substrate-grounding)**: the signal depends on the substrate artifact, tested by ablation. At L3 decoding falls with the dynamics difference (FINDINGS §14). A hand-authored, temporally coherent drag of matched one-step size is read as well, white jitter is not, so nothing here is specific to learned dynamics (§17.9). At L1 the survival agent did not meet the encoding criterion at the in-band grid (§14.7).
- **H3 (emergence under selection)**: selection raises world decodability. **Negative under the pooled population readout**, 3 lineages at 30 generations (FINDINGS §13.D, §13.E).
- **H4 (incidental encoding)**: the condition is decodable from a from-scratch agent's state without reward. **Met at L3** for survival with the next-observation auxiliary at 300 updates, corrected trainer: 0.733, with the L0 gate open on the registered world pair (FINDINGS §17); not met at L1 or L2.

### Current status

| Component | State |
|-----------|-------|
| World ("A Patch of Earth" v0) | built, verified deterministic, snapshot-exact |
| Ladder L0 / L1 / L2 | implemented and validated |
| Ladder L3 (learned-dynamics surrogate) | implemented, gate-calibrated |
| Experiment A (detectability ceiling, agent-free), L1 | **done** |
| Experiment A, L2 | **done** |
| Experiment B (incidental detection), L2 arc | **done: did not meet the registered encoding criterion** |
| Experiment B, L1 organism + H2 battery | **done: did not meet it at in-band Δ=0.023 (survival 0.533; FINDINGS §14.7)** |
| Experiment B, L3 (learned-dynamics) | **met the decodability clauses at n=10** with survival plus the next-observation auxiliary at 300 updates, corrected trainer: 0.733 against predictor 0.589 and untrained 0.523, L0 gate open on the registered world pair. Without the auxiliary 0.613, not met (FINDINGS §17). Historical: 0.752 on GPU, 0.730 and 0.601 on CPU (§10.8); an independently trained fingerprint 0.639 (§10.9); prefix condition remains decodable after restoring authentic dynamics, carried mostly by the physical footprint the prefix leaves rather than by memory, and retention under identical input is not shown (FINDINGS §17.7, §10.6.1). Details in [Key result](#key-result) |
| Trainer correction (2026-10) | **done**: GAE truncation bootstrap fixed; both primary runs rerun; effect +0.003 and +0.012 ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), FINDINGS §17) |
| Experiment C (emergence under selection) | **validated null** under the pooled population readout on the fixed-code re-run (FINDINGS §13.D, §13.E) |
| Ladder L4 (adversarially-hardened surrogate) | not started |

### Key result

*The L3 primary comparisons below were rerun with the corrected trainer
(FINDINGS §17). Every other agent number is historical and labeled so in
[`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md).*

Hand-authored seams (L1 discretization, L2 rollout drift): catchable from the
outside, not decodable inside at the registered criterion. An L2 artifact a
nearly noiseless external detector finds at AUROC ≈ 0.99 leaves the
prediction-only agent's state near chance (≈ 0.5). Give that detector the input
noise the L3 gate uses and the L2 artifacts read 0.62 to 0.65 (FINDINGS §15), so
the matched case is L1: an in-band quantization grid (detector AUROC 0.873)
leaves the survival state at **0.533** (n = 10; FINDINGS §14.7).

**Corrected L3 result (FINDINGS §17).** Survival training with the
next-observation auxiliary, 300 updates, ten seeds, on CPU: the survival state
reads **0.733** (t-based 90% CI **[0.669, 0.797]**, 8 of 10 seeds at the 0.65
bar), the predictor arm 0.589 and the untrained arm 0.523, with paired margins
of +0.144 and +0.210 whose intervals clear the registered 0.05. The L0 gate
(both pools authentic) is open on the registered pair of evaluation worlds
(0.559) and equivalent to chance across eight independent pairs (0.493), so
the verdict is met on the decodability clauses, conditional on L0. Without the
auxiliary the same protocol reads **0.613** [0.552, 0.675], not met; the
decoder adds **+0.120** [+0.065, +0.174]. The bootstrap fix moved these two
readings by +0.003 and +0.012.

**What does the noticing.** Fed the survival agent's own observation streams,
a trunk trained only to predict reads the world at 0.721, and one trained on
exactly the survival agent's experience at 0.738, against 0.733 for the
survival trunk. Driven by one scripted policy, every arm reads 0.55 to 0.58.
A decoder that never sees the brain, only the senses, reads 0.724. So the
signal rides on where the survival policy takes the body, and a trunk trained
to predict keeps it; at matched input the survival objective adds nothing
measurable, and "survival-specific" is withdrawn. A replay across the two
corrected runs puts about half of the decoder's +0.120 in the trunk and half in
the paths the decoder-carrying policy takes (exploratory, FINDINGS §17.10).

**Bounded by budget.** Trained to 450 updates, the arm without the decoder
reaches 0.685 and the arm with it 0.821, at about equal return. The best-fitting
nonlinear control over observations, actions, and behavior leaves 0.620, under
the bar. Under identical input the prefix condition fades within a few steps,
and with the recurrent state zeroed the body's footprint still reads 0.662. A
hand-authored, smoothly drifting drag of the same one-step size as the Gaussian
jitter is read through the original direction (0.721); the jitter is not.

**The historical record (pre-correction trainer).** The surrogate is a
**learned-dynamics fingerprint**: a small net replacing the velocity law, not a
hand-tuned knob. On the published GPU run the **survival** agent's state,
trained with the next-observation auxiliary, separates the worlds. Untrained
sits at chance (about 0.49, still
0.52 under a nonlinear probe). Prediction-only sits near chance (about 0.57).
Survival reads **0.752** (n = 10, t-based 90% CI **[0.698, 0.807]**, excludes
the pre-registered 0.65 bar; 8 of 10 seeds clear it). World is not decodable
from summed reward (AUROC 0.541, 10 of 10 seeds). Zero early deaths, every pool
110/110. L0 authentic-vs-authentic sits at 0.517 under the original
cross-validation split (0.539 under the stack-independent split, where the
ten-seed equivalence test no longer accepts; FINDINGS §16).

**Most of the signal remains after a per-timestep behavior control.** The agent
does move and forage differently in the two worlds. The full behavior trace
alone decodes the world at **0.803**, better than the state probe. The cheap
reading is that the probe is just reading behavior. A pre-registered
per-timestep control dumps every step's speed, energy, food, and drag,
residualizes the recurrent state on that trace in-fold with a linear map, and
probes what is left. What is left is **0.726** (t-based 90% CI **[0.679,
0.772]**, excludes the 0.65 bar; seed-level bootstrap [0.685, 0.765]; 9 of 10
seeds; quadratic variant 0.721). Adding absolute position and heading barely
moves it, to **0.723** (t-based 90% CI [0.676, 0.769]; 8 of 10 seeds). FINDINGS
§10.4.1. Untrained state under the same control is chance (0.498) even though
untrained *behavior* decodes 0.645. Prediction-only stays near chance (0.574).
The control removes what its channels and model can express; behavior outside
that basis is not excluded. An earlier per-episode-mean control had
under-estimated the signal at ~0.66 by over-removing, the attenuation the
synthetic tests predicted. Code: `scripts/audit_behavior_mediation.py`.
Artifacts: `artifacts/expB2/`.

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
|   |-- world.py                World protocol, surrogate ladder, matched-pair harness
|   |-- patch_of_earth.py       PatchOfEarthV0 concrete world, incl. L1/L2 hooks
|   |-- agent.py                recurrent world model (RSSM-lite)
|   |-- agent_ac.py             survival actor-critic (Experiment B-v2)
|   |-- experiment_a.py         agent-free L1 detectability oracle
|   |-- experiment_a_l2.py      agent-free L2 detectability oracle
|   |-- experiment_a_l3.py      agent-free L3 (learned-fingerprint) oracle
|   |-- experiment_b.py         incidental-detection harness
|   |-- experiment_b2.py        survival-coupled B-v2 pipeline
|   |-- surrogate_l3.py         L3 learned-dynamics surrogate (G_motion)
|   |-- behavior_audit.py       behavior-mediation controls (residual probes)
|   |-- control_diagnostics.py  what each behavior or sensory control removes (held-out R^2)
|   |-- folds.py                versioned grouped-CV fold partitions
|   |-- l0_audit.py             L0 gate across independent world samples
|   |-- eval_protocols.py       readouts under own, scripted, or replayed policies
|   |-- persistence.py          controlled persistence test (replay, common state, reset)
|   |-- stats.py                TOST/ROPE equivalence, bootstrap AUROC CIs
|   `-- results_io.py           end-to-end run recording
|-- scripts/                    deterministic reproduction runners
|   |-- run_e2e.py              pytest + all experiments (recorded)
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
|   |-- PREREGISTRATION_C.md    Experiment C pre-registration (design-complete)
|   |-- AUDIT_2026-07.md        research-integrity audit
|   |-- itasorl-series-plain-english.pdf  illustrated walkthrough (plain English)
|   |-- itasorl-series-research.pdf       illustrated walkthrough (research edition)
|   `-- figures/                result figures (.png) + provenance README
|-- artifacts/                  published summaries (committed): expA/, expB/, expB2/, expC/, l3_crossrecipe/
|-- fullruns/                   e2e run bundles (gitignored; default output)
|-- results/LATEST_RUN.txt      pointer to latest fullruns folder
|-- notebooks/colab_gpu.ipynb   Colab end-to-end runner
`-- tests/                      pytest regression suite
```

### Documents

- [`docs/itasorl-series-plain-english.pdf`](docs/itasorl-series-plain-english.pdf): the illustrated walkthrough in plain English - the friendliest entry point to the whole project.
- [`docs/itasorl-series-research.pdf`](docs/itasorl-series-research.pdf): the same series in research terms - design, control battery, pre-registered results, open questions.
- [`docs/ITASORL.md`](docs/ITASORL.md): the research plan, core question, prior work, hypotheses (H1 to H4), experiments (A/B/C), the surrogate ladder, validity audit, statistics, and engineering architecture.
- [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md): the world specification, "A Patch of Earth" v0, the 2.5D representation, fields and forcing, dynamics, ecology, the ~146-dim observation, ladder attachment, and confound management.
- [`docs/FINDINGS.md`](docs/FINDINGS.md): empirical results from the first build-and-test cycle.
- [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md): the dated corrections record, starting with the October 2026 trainer bootstrap fix, and what each change does not establish.
- [`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md): for every committed result, the commit, configuration, seeds, surrogate, budget, fold partition, device, and trainer, and whether it is historical or corrected.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): regenerate every table from committed artifacts without training (`python scripts/reproduce.py tables`), or retrain a run.
- [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md): the writeup outline and a claims inventory linking every headline number to its committed artifact.
- [`docs/LEARNING.md`](docs/LEARNING.md): the running lab notebook (methods lessons, dead ends, decisions).
- [`docs/PREREGISTRATION_L3.md`](docs/PREREGISTRATION_L3.md), [`PREREGISTRATION_Bv3.md`](docs/PREREGISTRATION_Bv3.md), [`PREREGISTRATION.md`](docs/PREREGISTRATION.md), [`PREREGISTRATION_C.md`](docs/PREREGISTRATION_C.md): pre-registrations (with deviation logs) for the B-v2, B-v3, L3, and (design-complete) C experiments.
- [`docs/AUDIT_2026-07.md`](docs/AUDIT_2026-07.md): a skeptical research-integrity audit (numbers, statistics, pre-registration timing, citations).
- [`assets/film/README.md`](assets/film/README.md): the two films, where each file came from, and the ffmpeg lines that rebuild the web encodes.

### Figures

Result figures live in [`docs/figures/`](docs/figures/) and are regenerated by the
run scripts:

- `expA_ceiling.png`: L1 detectability ceiling vs grid spacing.
- `expA_L2_ceiling.png`: L2 detectability ceiling vs drift strength.
- `expB_incidental.png`: recurrent-state probe across the drift sweep (target vs negative control).
- `expB_channels.png`: two incidental-detection channels (recurrent state vs prediction error).
- `expB_kstep.png`: effect of a longer-horizon objective on encoding.
- `budget_curve.png`: the corrected arms against the training budget, from
  `scripts/build_budget_curve.py` and `artifacts/budget_curve.json`.

To regenerate all of them in one recorded pass, run `python scripts/run_e2e.py --quick`
from the repo root; each `scripts/run_exp*.py` runner rewrites only its own figure.
Per-figure provenance (which script, which doc section) is tracked in
[`docs/figures/README.md`](docs/figures/README.md).

---

## How to run

Dependencies:

```bash
pip install -r requirements.txt      # runtime deps
pip install -e .                     # or: install itasorl as an editable package
```

Reproduce the experiments (each is deterministic given its seeds; run from the repo
root so figures land in `docs/figures/`):

```bash
python scripts/run_e2e.py --quick   # full battery + recorded results (recommended)
python scripts/run_expA.py          # Experiment A, L1
python scripts/run_expA_l2.py       # Experiment A, L2
python scripts/run_expB_full.py     # Experiment B: recurrent-state probe
python scripts/run_expB_surprise.py # Experiment B: prediction-error channel
python scripts/run_expB_kstep.py    # Experiment B: open-loop horizons
python scripts/run_expB_gap.py      # Experiment B: engagement + delta objective
python scripts/run_expB_nonlinear.py# Experiment B: nonlinear probe check
python scripts/run_expB2.py         # Experiment B-v2: survival-coupled
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

**Readout-only reruns on the saved agents:** the ten hidden-8 survival agents behind the world-sample sensitivity result (FINDINGS 17.5.1) are gitignored and are published with the paper's archive record. `python scripts/fetch_saved_agents.py --url <archive link>` downloads them and checks every file's sha256 against `artifacts/l0_audit/d045_world_samples_l3_h8_heldout.json`; the Colab notebook's Extra checks section does the same from a form field and then runs `scripts/run_world_sample_sensitivity.py`, comparing its per-draw means and verdict against the committed artifact.

---

## What to read first

[`docs/FINDINGS.md`](docs/FINDINGS.md) is the record. [`docs/ITASORL.md`](docs/ITASORL.md)
is the plan. [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md) is the world.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff). GitHub renders a
"Cite this repository" button from it. `cffconvert -f bibtex` if you want BibTeX.
