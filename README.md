# ITASORL

*A tabula-rasa artificial-life system that asks whether a from-scratch digital
organism can tell that its world is a generative **surrogate** rather than the
authentic base dynamics of its computational substrate — using only substrate
seams, with detection **read out, not rewarded**.*

---

## The question

Take an agent with no imported semantics, drop it into a controllable world, and
make that world either an authentic simulation or a subtly-flawed surrogate of
itself. Can the distinction be detected from the substrate's seams alone? And —
the part that matters — does an agent encode that distinction *incidentally*,
without ever being rewarded for it?

The project is built on three invariants:

- **Tabula rasa** — no semantics are imported; the agent starts from nothing.
- **Readout, not reward** — world identity is *probed*, never trained or rewarded.
- **Substrate-grounded** — the only tells are artifacts of the computational substrate.

Detection difficulty is dialed with a **surrogate ladder** (L0 identical control →
L1 discretization → L2 rollout drift → L3 learned-model fingerprint → L4
adversarial).

---

## Current status

| Component | State |
|-----------|-------|
| World ("A Patch of Earth" v0) | built, verified deterministic, snapshot-exact |
| Ladder L0 / L1 / L2 | implemented and validated |
| Experiment A (detectability ceiling, agent-free) — L1 | **done** |
| Experiment A — L2 | **done** |
| Experiment B (incidental detection) — first full arc | **done (robust negative result)** |
| Experiment C (emergence under selection) | not started |
| Ladder L3 / L4 | not started |

**Key result:** *detectability does not imply incidental encoding.* An artifact an
external oracle detects at AUROC ≈ 0.99 leaves essentially no decodable trace in a
from-scratch agent's representation (≈ 0.5), across three readout channels and
surviving objective-engagement and nonlinear-probe checks. See **`FINDINGS.md`**
for the full writeup and figures.

---

## Repository contents

### Documents
- **`ITASORL.md`** — the research plan: core question, literature white-space, hypotheses (H1–H4), experiments (A/B/C), the surrogate ladder, validity audit, statistics, and engineering architecture.
- **`ITASORL_world_spec.md`** — the world specification: "A Patch of Earth" v0, the 2.5D representation, fields and forcing, dynamics, ecology, the ~146-dim observation, ladder attachment, and confound management.
- **`FINDINGS.md`** — empirical results from the first build-and-test cycle (this is where the numbers live).

### Code
- **`world.py`** — the `World` protocol, `ObsSpec` (with format-stable identity hash), the surrogate-ladder wrappers, and the matched-pair rollout harness (the keystone control).
- **`patch_of_earth.py`** — the concrete world `PatchOfEarthV0`, including the L1 (observation discretization) and L2 (drifting-drag) hooks.
- **`logschema.py`** — the step/activation logging schema and writer (the ground-truth label lives only in logs, never in observations).
- **`experiment_a.py`** — the agent-free detectability oracle for L1, plus the leakage-audit battery and grouped-AUROC evaluation.
- **`experiment_a_l2.py`** — the L2 detectability oracle (invariant / rollout-residual) and its matched-pair drift generator.
- **`agent.py`** — the recurrent world model (RSSM-lite, PyTorch) with one-step and open-loop rollout objectives, plus a numpy reservoir fallback.
- **`experiment_b.py`** — the incidental-detection harness: episode collection, world-model training, and the probe + selectivity controls.

### Figures
- `expA_ceiling.png` — L1 detectability ceiling vs grid spacing.
- `expA_L2_ceiling.png` — L2 detectability ceiling vs drift strength.
- `expB_channels.png` — two incidental-detection channels (recurrent state vs prediction error).
- `expB_kstep.png` — effect of a longer-horizon objective on encoding.

### Reproduction scripts
- `run_expA.py`, `run_expA_l2.py`, `run_expB_full.py`, `run_expB_surprise.py`, `run_expB_kstep.py`, `run_expB_gap.py`, `run_expB_nonlinear.py`.

---

## How to run

Dependencies:

```bash
pip install numpy scikit-learn matplotlib torch
```

Reproduce the experiments (each is deterministic given its seeds):

```bash
python run_expA.py          # Experiment A, L1: control / detection / contamination / calibration
python run_expA_l2.py       # Experiment A, L2: same, for rollout drift
python run_expB_full.py     # Experiment B: recurrent-state probe, drift sweep + control + selectivity
python run_expB_surprise.py # Experiment B: prediction-error (surprise) channel
python run_expB_kstep.py    # Experiment B: longer-horizon (open-loop) objective
python run_expB_gap.py      # Experiment B: objective-engagement diagnostic + delta-rollout objective
python run_expB_nonlinear.py# Experiment B: nonlinear (random-forest) probe robustness check
```

A quick smoke test of the Experiment B pipeline:

```bash
python experiment_b.py
```

---

## What to read first

1. `README.md` (this file) — the map.
2. `FINDINGS.md` — what we found and what it means.
3. `ITASORL.md` — the full research plan and rationale.
4. `ITASORL_world_spec.md` — the world, in detail.

---

## The one-paragraph takeaway

The hard part of "can an organism notice its world is fake?" turned out not to be
detectability — substrate artifacts are readily detectable by a privileged
observer, with calibrated difficulty and a clean confound audit. The hard part is
**incidental encoding**: a from-scratch agent trained only to predict its senses
does not represent a dynamics artifact it is never asked to care about, even when
that artifact is strongly present and trivially detectable from the outside. That
gap — between what is detectable and what an agent spontaneously encodes — is the
real object of study, and it turns the headline hypothesis into a sharper,
conditional question that the next phase (survival-coupled reward, richer ladder
rungs) is designed to answer.
