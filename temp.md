# ITASORL — What Happened This Week (Plain English)

*Snapshot as of June 29, 2026. This is a listening doc: what existed, what changed, what's left, and where things stand now.*

---

## The idea in one sentence

We built a tiny digital creature that lives in a simulated world. Sometimes the world runs on "real" physics; sometimes we swap in a slightly fake copy with tiny flaws. The big question: **does the creature notice the fake on its own**, even though we never tell it to care and never reward it for figuring that out?

---

## What was (before ~June 27)

**Almost nothing.** The repo was brand new — initial commit landed June 27. There was no working test suite, no CI pipeline, no packaged dependencies, and several experiment scripts that would crash if you tried to run them.

Conceptually, the research plan already existed on paper:
- Build a controllable simulated world ("A Patch of Earth")
- Dial in different kinds of "fake world" flaws on a ladder (L0 = identical, L1 = quantization, L2 = drifting physics, L3/L4 = not built yet)
- Run three experiment types:
  - **Experiment A** — Can an all-seeing outside observer spot the fake? (no agent needed)
  - **Experiment B** — Does a learning agent *incidentally* encode which world it's in?
  - **Experiment C** — Does encoding emerge when survival depends on it? (not started)

The first wave of results (Experiments A and B) was already written up in spirit, but the codebase needed hardening, missing APIs, tests, and docs cleanup before it could be trusted as a reproducible research repo.

---

## What changed (last 2–3 days)

Think of this as three layers of work: **build the lab**, **run the experiments**, **stress-test the survival hypothesis**.

### 1. Built the lab from scratch (June 27)

| Area | What happened |
|------|----------------|
| **World** | A deterministic 2.5D environment (~146 sensory dimensions) with a "surrogate ladder" — ways to make the world subtly fake |
| **Agent** | A recurrent world-model (predicts what it will sense next) plus later an actor-critic survival agent |
| **Experiments** | Runnable scripts for Experiment A (L1 & L2) and Experiment B (multiple probe types) |
| **Docs** | README expanded, research plan and world spec organized under `docs/` |
| **CI** | GitHub Actions for lint + tests |
| **Tests** | pytest suite grew from zero to ~37 tests covering world dynamics, oracles, logging, and agent APIs |
| **Dependencies** | `requirements.txt` added so installs and CI are reproducible |

**Bug fixes along the way:**
- Experiment B's open-loop rollout API was missing — `run_expB_gap.py` and `run_expB_kstep.py` crashed until it was implemented
- Lint/CI was red until ruff was configured for the repo's compact research-script style
- Unused imports and other small cleanup

**Automation:** A "Ralph" bug-hunting loop was added — an autonomous senior-dev persona that reads a backlog, finds issues, and fixes them with tests.

---

### 2. Ran the core experiments and wrote up findings

#### Experiment A — "Can an outsider spot the fake?" → **Yes, almost perfectly**

- **L1 (grid quantization):** Outside oracle AUROC ≈ **1.00** (perfect detection)
- **L2 (slowly drifting physics):** Outside oracle AUROC ≈ **0.99**
- Leakage audits pass — the signal lives in the substrate artifact, not in reward tricks or metadata

**Plain English:** If you have god-mode access to the world's behavior, the fake is obvious.

#### Experiment B — "Does the agent notice on its own?" → **No (robust negative)**

Three different ways of reading the agent's internal state all came back at ~coin-flip level (~0.50 AUROC):

1. **Recurrent state probe** — what's inside the agent's "memory"?
2. **Surprise channel** — is the agent surprised when physics drift?
3. **Longer-horizon prediction** — force it to imagine further into the future

Extra checks confirmed this isn't because the agent was lazy or the probe was too weak:
- The open-loop model genuinely learned to predict (MSE well below baselines)
- A nonlinear (random forest) probe also found nothing
- Positive control (speed) reads fine at ~0.90 — the machinery works

**Plain English:** The creature learns useful stuff (like how fast it's moving) but does **not** bother to learn "am I in the real world or the fake one" when its only job is prediction.

**Headline finding:** *Detectability ≠ incidental encoding.* What you can spot from outside ≠ what the agent spontaneously represents.

---

### 3. Experiment B-v2 — "What if survival depends on the fake physics?" (June 28)

This was the natural next question: maybe prediction-only training is too weak. What if **staying alive** forces the agent to cope with drifting drag (movement cost)?

Three agents, same brain architecture, different training:

| Agent | What it does |
|-------|----------------|
| **untrained** | Random weights — baseline floor |
| **predictor** | Predicts next sensation (Experiment B again) |
| **survival** | Actor-critic: forage to not starve (~50 steps to death) |

**Results (corrected run after bug fixes):**

| Agent | Drift = 0 (control) | Drift = 0.45 (strong fake) |
|-------|---------------------|----------------------------|
| untrained | ~0.46 | ~0.44 |
| predictor | ~0.49 | ~0.49 |
| **survival** | ~0.51 | **~0.60** |

**Plain English:**
- The prediction-only agent still learns nothing about world identity (good — confirms Experiment B).
- The survival agent shows a **small bump** at strong drift (~0.60 vs ~0.50 chance).
- But that's **far below** the pre-registered bar for a real win (0.65), and still nowhere near the outsider oracle (~0.99).
- **Verdict:** Survival pressure leaves a faint trace, not a decodable "I know which world I'm in" representation.

**Bugs found and fixed during B-v2:**
- **GAE bug** — advantage estimation leaked padded timesteps at episode end; only affected survival training; conclusion unchanged after fix
- **Engagement gate** — recalibrated so we can tell the survival agent actually learned to forage (not just got lucky)
- **RunningNorm** — investigated and confirmed correct (not a bug; added characterization tests)

---

### 4. Repo hygiene (June 28)

- Docs moved into a clean `docs/` tree with figures
- README rewritten as the map (plain-language section + researcher section)
- `docs/FINDINGS.md` updated with full empirical writeup including B-v2
- Pre-registration document for B-v2 (`PREREGISTRATION.md`)
- Ralph backlog seeded with follow-up hardening items

---

## What is left (not done yet)

### Research / experiments

| Item | Status | Why it matters |
|------|--------|----------------|
| **Experiment C** (emergence under selection) | Not started | Broader evolutionary / selection framing from the original plan |
| **Ladder L3** (learned world-model surrogate fingerprint) | Not started | More interesting artifact than L2 drift; harder to dismiss as "trivial" |
| **Ladder L4** (adversarially hardened surrogate) | Not started | Stress-test whether artifacts can be hidden from both oracle and agent |
| **B-v2 follow-ups** | Open questions | More seeds (n=10 runs appear in progress), longer training, L3 artifact, "reactive vs representational" disambiguation |
| **Stronger multi-step objectives** | Partially tried | Open-loop engaged but didn't lift encoding; variants (weight dynamics dims, more capacity) remain |
| **Hypothesis H2** (substrate ablations) | Not tested | Confirm signal lives in substrate, not confounds |
| **Hypothesis H3** (emergence under selection) | Not tested | Tied to Experiment C |

### Engineering / hardening (from Ralph backlog)

| Priority | Item |
|----------|------|
| P2 | GPU/CPU parity tests for B-v2 readouts (determinism on both devices) |
| P2 | Verify early-death episode dropping doesn't silently corrupt pools |
| P3 | Extreme-value action clamping tests for survival agent |
| P3 | Optional: wrap run scripts in `if __name__ == "__main__"` guards |

### In-flight (uncommitted artifacts spotted)

- `expB2_results_n10.json` / `expB2_survival_n10.png` — looks like a higher-seed confirmatory run
- `expB2_results_confirmatory_n3.json` / `expB2_survival_confirmatory_n3.png` — confirmatory pass artifacts

These may need to be folded into findings once reviewed.

---

## Current status of the app (right now)

### What works today

```
✅ Simulated world ("A Patch of Earth" v0) — deterministic, snapshot-exact
✅ Surrogate ladder L0 / L1 / L2 — implemented and validated
✅ Experiment A (detectability ceiling) — done for L1 and L2
✅ Experiment B (incidental detection) — done; robust negative result
✅ Experiment B-v2 (survival-coupled) — done; weak trace, not a positive claim
✅ Test suite — ~37 pytest tests, CI green
✅ Reproducible runs — seeded scripts regenerate figures into docs/figures/
✅ Documentation — README + FINDINGS + research plan + world spec + pre-registration
```

### What the science says right now

1. **Fake worlds are easy to detect from the outside** (oracle ≈ 0.99).
2. **Agents don't spontaneously encode that fact** under prediction-only training (≈ 0.50).
3. **Survival pressure adds a small signal** (≈ 0.60 at strong drift) but **not enough** to claim the agent "knows" which world it's in.
4. The project's core question has **sharpened**: not "can it notice?" but **"under what conditions does a mind represent something it was never asked to care about?"**

### Branch / repo state

- **Current branch:** `expB2-survival-coupling` (aligned with `origin/main`)
- **Last commit:** corrected B-v2 sweep after GAE fix — conclusion unchanged
- **Working tree:** mostly clean; a few untracked B-v2 result files from confirmatory runs

### How to run it yourself

```powershell
pip install numpy scikit-learn matplotlib torch
pytest -q                                    # smoke the test suite
python run_expA.py                           # outsider detection, L1
python run_expA_l2.py                        # outsider detection, L2
python run_expB_full.py                      # agent incidental encoding
python run_expB2.py                          # survival-coupled version
```

Full command list lives in `README.md`.

---

## The story arc (if you're listening, not reading code)

**Day 0:** Idea on paper — can a digital organism tell its world is fake?

**Day 1–2:** Built the world, the agent, the experiments, CI, and tests. Ran Experiment A → outsiders see the fake easily. Ran Experiment B → the agent doesn't care.

**Day 3:** Asked the harder question — what if its life depends on the fake physics? Built survival agent (B-v2). Found a faint trace, fixed real bugs in training, confirmed the big picture holds: **detectability and encoding are different things.**

**Next:** Scale up (more seeds, longer training), try richer fake-world artifacts (L3), and figure out whether the survival trace is real representation or just reactive coping.

---

## Where to read more

| Doc | What's in it |
|-----|----------------|
| `README.md` | Project map + plain-language intro |
| `docs/FINDINGS.md` | All the numbers, figures, and caveats |
| `docs/ITASORL.md` | Full research plan and hypotheses |
| `PREREGISTRATION.md` | What we promised before running B-v2 |
| `ralph/BACKLOG.md` | Open engineering issues and what's already fixed |

---

*This file is temporary — a listening aid, not canonical science. For numbers and methodology, trust `docs/FINDINGS.md`.*
