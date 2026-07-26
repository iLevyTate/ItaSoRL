# ItaSoRL — X (Twitter) posts

Copy-paste-ready posts built from the nine-page illustrated series
([plain-English](../itasorl-series-plain-english.pdf),
[research edition](../itasorl-series-research.pdf)). Every number here is read
from committed artifacts and traced to a section in
[`docs/FINDINGS.md`](../FINDINGS.md); the trace table is at the bottom of this
file.

**Voice:** first person, Ben Kennedy (`@iambenkennedy`).

**Length:** every post is ≤ 280 characters, counting each URL as X does (23
chars, whatever its real length). Inside the code blocks, one line = one
paragraph; blank lines are deliberate. Copy the block whole — don't re-wrap it.

**Before you post, read the [name-consistency note](#name-consistency-note) at
the bottom** — the repo, the site URL, and `CITATION.cff` currently carry
`iLevyTate` / "Levy Tate", so anyone who clicks a link lands on a different
name than the one posting.

---

## Thread A — the flagship thread (9 posts)

One post per page of the illustrated series. This is the main artifact: post it
first, pin it.

**1/9 — the hook**

```
I raised a digital organism from scratch inside a simulated world.

Sometimes that world ran on real physics. Sometimes on a subtly flawed copy.

I never told it. I never rewarded it for noticing.

Then I read its mind to see if it figured it out anyway.

What I found: 🧵
```

**2/9 — the world**

```
The world is "A Patch of Earth": small, deterministic, Earth-like. Light, wind, scent, food, weather. The organism gets a ~146-dim sensory vector and one job — stay alive.

It starts with zero. No language, no labels, no concepts, no hint about where it lives.
```

**3/9 — the invariants**

```
Three rules I never break:

• tabula rasa — no imported semantics, ever
• readout, not reward — world identity is probed, never trained
• substrate-grounded — the only tells are artifacts of the computation itself

Break one and the result means nothing. That's the discipline.
```

**4/9 — the ladder**

```
"Fake" isn't one thing, so I built a difficulty dial — the surrogate ladder.

L0 identical copy (control, must read chance)
L1 discretization
L2 rollout drift
L3 learned-dynamics fingerprint
L4 adversarial

Same world, same organism, one knob. Frontier is L3.
```

**5/9 — the oracle**

```
First question: is the fake even detectable?

I gave a privileged outside observer the world's raw behavior and asked it to call real vs copy.

L2 drift: AUROC ≈ 0.99.
L1 quantization: 0.873.

The seams are there. An outside watcher catches the copy almost every time.
```

**6/9 — the null**

```
Second question, the one that matters: does the organism notice on its own?

No.

Same L2 world the oracle nails at 0.99 — the organism's internal state decodes it at ≈ 0.50. A coin flip. L1 replicates the null on a different primitive: 0.533, 0/10 seeds clearing the bar.
```

**7/9 — the finding**

```
Detectable ≠ learned. Nothing in its day required it to care, so it never represented it.

That gap is the object of study. It turns a vague question ("can it tell it's in the Matrix?") into a testable one: when does a mind encode something it was never asked to care about?
```

**8/9 — the reversal**

```
Then it reverses.

At L3 — where the copy is a learned fingerprint of the physics, not a hand-tuned knob — a survival-trained agent's state decodes the world at 0.752 (90% CI [0.698, 0.807], n=10).

Prediction-only agent: 0.573. Untrained: 0.488. Same trunk.

Never rewarded.
```

**9/9 — the close**

```
The honest version: survival pressure is what turned a detectable artifact into an encoded one. That's the result.

Nine illustrated pages, in two voices — the story, and the research edition:
https://ilevytate.github.io/ItaSoRL/

Code, data, pre-registrations, every number: same link.
```

---

## Standalone posts

Each stands on its own. Space them out — roughly one every 2–3 days — and don't
post one the same day as the thread.

### B — the three-number post

Highest-signal single post. Pairs with a screenshot of the three stat tiles from
the site.

```
Three numbers from a year of this:

99% — an outside observer catching the fake world
50% — the organism's own mind, idle. Coin flip.
73% — the same mind, once survival depended on it

Detectable all along. It only learned it when it had to.

https://ilevytate.github.io/ItaSoRL/
```

### C — the film

Attach `assets/film/detectable-not-learned-web.mp4` (90s, 4:5) as native video.

```
90 seconds, no jargon:

A creature is born knowing nothing. I copy its world and change one rule of physics. An outside watcher catches the fake almost every time — the creature's own mind stays at a coin flip.

Until telling real from fake decides whether it eats.
```

### D — the novelty posts (2)

The "hasn't this been done?" answer. Good for the ML-research audience.

**D 1/2**

```
"Hasn't someone done this?" Three neighbors, and what each skipped:

LLMs that clock when they're being evaluated — but they read the internet. They already knew what "a test" is.

Probes like Othello-GPT — real, but always about things *inside* one world. Never *which world*.
```

**D 2/2**

```
sim2real anomaly detectors — someone built that alarm and told it what to watch for. An installed smoke detector.

Mine reads nothing. Is told nothing. Is never paid to notice.

I raise it, then check afterward whether the knowledge showed up on its own.

https://ilevytate.github.io/ItaSoRL/
```

### E — the deflationary-reading posts (2)

The controls. This is the pair researchers respect.

**E 1/2**

```
The obvious objection to my L3 result: the agent just *moves* differently in the two worlds, so the probe reads behavior, not a representation.

Fair — the behavior trace alone decodes the world at 0.803. Better than the state probe.

So I pre-registered a control to kill it.
```

**E 2/2**

```
Dump every timestep's speed, energy, food, drag. Residualize the recurrent state on that trace in-fold. Probe what's left.

Behavior-independent world-signal: 0.726 (90% CI [0.679, 0.772], 9/10 seeds clear the bar).

Residualize position and heading too: 0.723. Barely moves.
```

### F — the limits posts (2)

Post these. They're the most credible thing on the list.

**F 1/2**

```
What my result does *not* show:

• survival-specificity holds at one of two tested fingerprints, not both
• transfer is direction-dependent — the frozen reverse probe failed at 0.638
• the stored component is modest; its late tail decays
• Experiment C came back null
```

**F 2/2**

```
Each of those is written into the findings doc, next to the pre-registration it deviated from.

A negative result you quietly drop isn't a result. Half of what I have is negative, and it's the half that makes the positive mean anything.

https://ilevytate.github.io/ItaSoRL/
```

### G — the reproducibility post

```
Everything is reproducible and nothing is hand-waved:

• deterministic given its seeds
• every headline number traced to a committed artifact
• pre-registrations with deviation logs
• a skeptical self-audit of my own stats

One GPU notebook runs the whole thing end to end.
```

### H — the estimator-bug post

Optional, and the most human one. Post it if you want the "how research actually
goes" angle.

```
A control said my effect died — 0.557, under the bar. I wrote it up as a failure.

Then I found the bug: my estimator was biased.

Re-scored the saved dumps with it fixed. 0.666 and 0.684. The effect was there all along.

The fix is in the findings doc, beside the old number.
```

### I — the invitation post

Good closer, or a reply to your own pinned thread.

```
Open question I don't have the answer to, and I'd take help on:

Under what conditions does a mind start representing something it was never asked to care about?

I have one crossing point — survival pressure at L3. One point isn't a curve.

The ladder goes to L4. Code's open.
```

---

## Posting plan

| Day | Post | Media |
|-----|------|-------|
| 1 | Thread A (9 posts), then pin it | series PDF cover as the 1/9 image |
| 3 | B — three numbers | screenshot of the three stat tiles |
| 6 | C — the film | `detectable-not-learned-web.mp4`, native |
| 9 | D — novelty (2 posts) | none |
| 13 | E — controls (2 posts) | `docs/figures/expB_incidental.png` |
| 17 | F — limits (2 posts) | none |
| 21 | G — reproducibility | Colab badge or notebook screenshot |
| 25 | H — estimator bug | none |
| 29 | I — invitation | none |

Notes:

- **Alt text on every image.** Describe the axes and what the reader should
  see, not "a chart."
- **Link placement.** The first post of a thread does better without one —
  that's why 1/9 has no link and 9/9 carries it.
- **The video is the best single asset here.** If you post only one thing, post
  C.
- Numbers are as of the current `main`. If FINDINGS changes, re-check against
  the trace table before reposting.

---

## Number trace

Every figure used above, and where it comes from.

| Number | Claim | Source |
|--------|-------|--------|
| ~146-dim | observation vector | `ITASORL_world_spec.md` |
| ≈ 0.99 | L2 oracle AUROC (Exp A tamed diagnostic config) | FINDINGS TL;DR, §3 |
| 0.873 | L1 oracle AUROC at Δ = 0.023 | FINDINGS §14.7 |
| ≈ 0.50 | L2 agent state, near chance | FINDINGS §§3, 4, 9 |
| 0.533 | L1 survival pooled world-identity, 0/10 seeds ≥ 0.65 | FINDINGS §14.7 |
| 0.752 | L3 survival state probe, 90% CI [0.698, 0.807], n = 10 | FINDINGS §10 |
| 0.573 / 0.488 | predictor / untrained baselines, identical trunk | FINDINGS §10 |
| 0.803 | full behavior trace alone decodes world | FINDINGS §10 |
| 0.726 | behavior-independent signal, CI [0.679, 0.772], 9/10 seeds | FINDINGS §10 |
| 0.723 | same control + position and heading residualized | FINDINGS §10.4.1 |
| 0.638 | frozen reverse held-out probe, misses the 0.65 bar | FINDINGS §10.6 |
| 0.557 → 0.666 / 0.684 | common-garden, biased estimator → re-scored | FINDINGS §§13.C, 10.6.1 |
| hidden = 7 | replication 0.722; survival 0.737 vs predictor 0.714 (+0.023 < +0.05) | FINDINGS §10.5 |
| validated null | Experiment C on fixed-code re-run; H3 negative | FINDINGS §13.D |
| 99 / 50 / 73 | the site's rounded headline tiles | `index.html`, film section |

---

## Name-consistency note

The posts are written for **@iambenkennedy**, but everything they link to
carries a different identity:

- the site URL is `ilevytate.github.io/ItaSoRL`
- the repo is `github.com/iLevyTate/ItaSoRL`
- `CITATION.cff` lists the author as **Tate, Levy**, ORCID
  `0009-0009-1337-0709`

A reader clicking through from a post signed Ben Kennedy lands on a page
credited to someone else — the one thing that undercuts a "this is my work"
post. Worth resolving before the thread goes out. Options, cheapest first:

1. **Add a byline only.** Put your real name in `CITATION.cff` and an "about the
   author" line on the site, leaving URLs alone. Costs nothing.
2. **Rename the GitHub account, or move the repo.** URLs change, GitHub
   redirects the old ones, the Pages URL becomes `<newname>.github.io/ItaSoRL`.
   Every link in this file, the README, and `index.html` needs updating with it.
3. **Leave it, address it in the thread.** A single reply — "iLevyTate is me" —
   works, but you'll repeat it every time.

Say the word and I'll do (1), or (2) once you've made the account change.
