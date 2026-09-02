# ItaSoRL

I raised a small agent from nothing and asked whether it would notice it was
living in a fake world. Nobody paid it to look. An outsider who knows the true
rules catches the fake **99%** of the time. The agent's own mind sits at **50%**
until the fake costs dinner. Then **73%**.

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
wrong: how well the ground grips. Tiny seams. The kind a stand-in leaves.

Two questions. Keep them apart.

1. **Can anyone catch the copy?** An outsider who already knows the real rules
   can. At L2 that outsider sits near **99%**.
2. **Does the creature notice on its own?** We never reward spotting the fake.
   We never mention the fake. We only read its mind afterward.

At L1 and L2 the outsider sees it and the creature does not. Coin flip. Nothing
in its day required it to care, so it did not represent the difference.

At L3 the fake is a learned-dynamics fingerprint, and survival is on the line.
The survival agent's state then carries a behavior-independent world-signal of
about **0.73**. Detectable was always there. Noticing shows up when dinner
depends on it.

### Hasn't this been done before?

No. Other projects have gotten close. Each one skipped the hard part:

1. **Chatbots that know when they're being tested.** Today's AI chatbots can often
   tell when they're being evaluated versus talking to a real person. But those AIs
   read most of the internet - they already know what "a test" is, what "a
   simulation" is, what researchers do. That's like a student who read the teacher's
   answer key. Our creature has read nothing. It is born knowing zero - no language,
   no concepts, no hints. If it figures out something is off about its world, it
   worked that out from the world itself.
2. **AI brains that learn things nobody taught them.** Scientists have looked inside
   AI systems before and found knowledge that was never explicitly taught - like a
   game-playing AI that secretly worked out the layout of the board. But those
   discoveries were always about things *inside* the AI's world - the pieces, the
   score, the map. Nobody had checked for the bigger question: does it know whether
   *the world itself* is the real one or a copy?
3. **Robots that check whether practice matched reality.** Robots trained in
   simulators are sometimes given alarm systems that go off when the real world
   doesn't match their practice world. But someone *built* that alarm and told it
   what to watch for. That's an installed smoke detector. We never install anything.
   We just let the creature live, then peek inside its memory afterward to see if
   the knowledge showed up *on its own*.

Everyone before either told their AI to look for the fake, or used an AI that
already knew what "fake" means. We hid the flaw in the physics, never mentioned
it, and checked whether the knowledge showed up anyway. Citations:
[`docs/ITASORL.md`](docs/ITASORL.md).

---

## For researchers

Same two questions, in the terms the papers use. Can a privileged discriminator
tell authentic base dynamics from a substrate-flawed surrogate? Does a
from-scratch agent encode that distinction incidentally, with world identity
read out after the fact and never rewarded?

Three invariants. All required, or the measurement collapses:

- **Tabula rasa**: no semantics are imported; the agent starts from nothing.
- **Readout, not reward**: world identity is *probed*, never trained or rewarded.
- **Substrate-grounded**: the only tells are artifacts of the computational substrate.

Detection difficulty is dialed with a **surrogate ladder** (L0 identical control,
L1 discretization, L2 rollout drift, L3 learned-model fingerprint, L4 adversarial).

Prior work holds pieces. Evaluation-awareness papers read test-vs-deploy signals
out of models that already know what a test is. Probing papers (AtariARI,
Othello-GPT) read incidental facts *inside* one fixed world. Mismatch detectors
(GalilAI, RAPT) are told to watch for the change. This project measures the gap
between those two numbers: how catchable the fake is from the outside, versus
whether a blank-slate agent encodes it unasked. Axis-by-axis comparison:
[`docs/ITASORL.md`](docs/ITASORL.md).

### Hypotheses

- **H1 (detectability)**: the surrogate is distinguishable from the authentic world by a privileged discriminator.
- **H2 (substrate-grounding)**: the signal lives in substrate artifacts, confirmed via ablations - **confirmed at L3**; L1 organism encoding is negative at the matched in-band grid (FINDINGS §14-14.7).
- **H3 (emergence under selection)**: representing the distinction emerges when survival depends on it - **resolved negative** (FINDINGS §13.D).
- **H4 (legibility / incidental encoding)**: a from-scratch agent encodes the distinction incidentally, without reward - **positive at L3**, negative at L1/L2.

### Current status

| Component | State |
|-----------|-------|
| World ("A Patch of Earth" v0) | built, verified deterministic, snapshot-exact |
| Ladder L0 / L1 / L2 | implemented and validated |
| Ladder L3 (learned-dynamics surrogate) | implemented + oracle-gated |
| Experiment A (detectability ceiling, agent-free), L1 | **done** |
| Experiment A, L2 | **done** |
| Experiment B (incidental detection), L2 arc | **done (robust negative result)** |
| Experiment B, L1 organism + H2 battery | **done (organism negative at in-band Δ=0.023: survival 0.533; FINDINGS §14.7)** |
| Experiment B, L3 (learned-dynamics) | **positive at n=10, replicated at a second capacity** - behavior-independent signal ~0.72 at both; H2 texture-specific at L3; transfer is direction-dependent but recipe-general; the re-scored common-garden control shows a modest persistent world-identity component (details in [Key result](#key-result) below and FINDINGS §10.6.1 / §14) |
| Experiment C (emergence under selection) | **validated null** on fixed-code re-run (FINDINGS §13.D); H3 resolves negative |
| Ladder L4 (adversarially-hardened surrogate) | not started |

### Key result

Hand-authored seams (L1 discretization, L2 rollout drift): catchable from the
outside, not encoded inside. An L2 artifact an external oracle detects at AUROC
≈ 0.99 leaves essentially no decodable trace in a from-scratch agent's
representation (≈ 0.5). An in-band L1 quantization grid (oracle AUROC 0.873)
leaves the survival state at **0.533** (n = 10; FINDINGS §14.7).

L3 is where that sentence breaks. The surrogate is a **learned-dynamics
fingerprint**: a small net replacing the velocity law, not a hand-tuned knob.
At this fingerprint, and only here, the **survival** agent encodes it. Untrained
sits at chance (about 0.49, still 0.52 under a nonlinear probe). Prediction-only
sits near chance (about 0.57). Survival reads **0.752** (n = 10, t-based 90% CI
**[0.698, 0.807]**, excludes the pre-registered 0.65 bar; 8 of 10 seeds clear
it). World is not decodable from summed reward (AUROC 0.541, 10 of 10 seeds).
Zero early deaths, every pool 110/110. L0 authentic-vs-authentic sits at 0.517.

**The signal survives after behavior is removed.** The agent does move and
forage differently in the two worlds. The full behavior trace alone decodes the
world at **0.803**, better than the state probe. The cheap reading is that the
probe is just reading behavior. A pre-registered per-timestep control dumps
every step's speed, energy, food, and drag, residualizes the recurrent state
on that trace in-fold, and probes what is left. What is left is **0.726**
(t-based 90% CI **[0.679, 0.772]**, excludes the 0.65 bar; seed-level bootstrap
[0.685, 0.765]; 9 of 10 seeds; quadratic variant 0.721). Adding absolute
position and heading barely moves it, to **0.723** (t-based 90% CI [0.676,
0.769]; 8 of 10 seeds). FINDINGS §10.4.1. Untrained state under the same
control is chance (0.498) even though untrained *behavior* decodes 0.645.
Prediction-only stays near chance (0.574). An earlier per-episode-mean control
had under-estimated the signal at ~0.66 by over-removing, the attenuation the
synthetic tests predicted. Code: `scripts/audit_behavior_mediation.py`.
Artifacts: `artifacts/expB2/`.

A second in-band fingerprint (hidden = 7, frozen fallback after hidden = 4
failed its gates) replicates the behavior-independent world-signal: **0.722**
(t-based 90% CI [0.672, 0.773]) vs 0.726 at hidden = 8. That coarser artifact
is one every trained agent picks up (predictor 0.714 vs survival 0.737, misses
the pre-registered +0.05 dissociation). Survival-*only* is therefore
conditional on the subtler hidden = 8 artifact. What both capacities share is
a reward-clean, survivorship-clean, behavior-independent world-signal of about
**0.72**.

A held-out probe (n = 10, `artifacts/expB2/heldout_l3_h8_summary.json`) splits
the rest. The world-identity direction still reads a held-out capacity variant
(transfer **0.773** vs untrained floor 0.569; the pre-registered rule passes).
The variant shares the training recipe, seed, and data, so that is robustness
inside one recipe (FINDINGS §10.6). A frozen reverse probe (train on the
coarser fingerprint, read the subtler one;
`artifacts/expB2/heldout_l3_h7_reverse_summary.json`) fails its bar at
**0.638**. Transfer goes one way. Under a common-garden control (identical tail
after differing prefixes), tail-only state still recovers the prefix world
above the frozen bar on both directions (**0.666** forward, **0.684** reverse).
The last-8-step late tail decays toward chance (0.586 / 0.577). Persistent, but
weak. The original common-garden number, 0.557, used a since-fixed biased
estimator and is overturned; FINDINGS §10.6.1. Transfer numbers were
unaffected.

A cross-recipe probe (n = 10, `artifacts/l3_crossrecipe/summary.json`) then
reads a gate-calibrated random-Fourier-features ridge law the agent never
lived with (**0.684** vs untrained floor 0.548; rule passes, machine-checked).
Thin: the t-based 90% CI lower bound clears the bar by 0.004, and 7 of 10
seeds sit above it. Recipe-general, not one function class. Details:
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
|   |-- stats.py                TOST/ROPE equivalence, bootstrap AUROC CIs
|   `-- results_io.py           end-to-end run recording
|-- scripts/                    deterministic reproduction runners
|   |-- run_e2e.py              pytest + all experiments (recorded)
|   |-- run_expA.py ...         Experiment A/B runners
|   |-- run_expB2.py            Experiment B-v2 / L3 (GPU if available)
|   |-- audit_behavior_mediation.py  behavior-mediation audit on dumped states
|   |-- audit_stats_recheck.py  CI gate: every published number vs its committed artifact
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

---

## What to read first

[`docs/FINDINGS.md`](docs/FINDINGS.md) is the record. [`docs/ITASORL.md`](docs/ITASORL.md)
is the plan. [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md) is the world.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff). GitHub renders a
"Cite this repository" button from it. `cffconvert -f bibtex` if you want BibTeX.
