"""Extract real per-unit brain data for the "Two Minds" film renderer.

Reads the SAVED heldout state pools (fullruns/l3_h8_heldout/states, the same
dumps the H2 integrity gate bit-matches against) and computes, per GRU unit,
how strongly its step-level activation separates the real world from the fake
one (rank AUROC, the dependency-light analog of the abs logistic coefficient
used by the original viz_mind_learning brain graph). No model is loaded and no
rollout is run: this is a pure numpy readout of frozen artifacts.

The film shows one representative brain: the seed whose pooled survival target
sits closest to the published 0.752 mean. The 12 displayed memory cells are the
most active units of that brain, with the top world-signal units swapped in so
the story cells are always on screen. Ring strength is the normalized signal.

Writes viz/player/brain/brain-data.js (window.BRAIN_DATA = {...}).

Usage (from repo root):
    python scripts/dump_brain_film_data.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "fullruns" / "l3_h8_heldout" / "states"
A2_AGG = ROOT / "fullruns" / "l3_h8_obs_localization" / "aggregate.json"
OUT = ROOT / "viz" / "player" / "brain" / "brain-data.js"

PUBLISHED_MEAN = 0.752   # canonical L3 headline (docs/FINDINGS.md TL;DR)
N_DISPLAY = 12           # memory cells shown in the film
N_RING = 6               # teal-ringed clue cells


def unit_auroc(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rank AUROC per unit: P(surr sample > auth sample), column-wise.

    a, s: (n_a, H) and (n_s, H) step-level pools. Returned per-unit values sit
    in [0, 1]; 0.5 means that unit's activation carries no world identity.
    """
    n_a, n_s = len(a), len(s)
    out = np.empty(a.shape[1])
    for k in range(a.shape[1]):
        both = np.concatenate([a[:, k], s[:, k]])
        # midranks (ties averaged) so equal activations contribute 0.5
        order = np.argsort(both, kind="mergesort")
        sorted_v = both[order]
        rank_sorted = np.arange(1, both.size + 1, dtype=float)
        i = 0
        while i < both.size:
            j = i
            while j + 1 < both.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            if j > i:
                rank_sorted[i:j + 1] = rank_sorted[i:j + 1].mean()
            i = j + 1
        ranks = np.empty_like(rank_sorted)
        ranks[order] = rank_sorted
        r_s = ranks[n_a:].sum()
        out[k] = (r_s - n_s * (n_s + 1) / 2.0) / (n_a * n_s)
    return out


def load_pool(seed: int, arm: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(STATES / f"states_d0.45_s{seed}_{arm}.npz")
    ha, hs = d["Ha"], d["Hs"]                    # (eps, steps, H)
    return ha.reshape(-1, ha.shape[-1]), hs.reshape(-1, hs.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    agg = json.loads(A2_AGG.read_text())
    per_seed = np.asarray(agg["none_survival_per_seed"], float)
    seed = int(np.argmin(np.abs(per_seed - PUBLISHED_MEAN)))
    print(f"representative seed = {seed} (pooled target {per_seed[seed]:.4f}, "
          f"published mean {PUBLISHED_MEAN})")

    fa, fs = load_pool(seed, "survival")
    ua, us = load_pool(seed, "untrained")
    hidden = fa.shape[1]

    auroc = unit_auroc(fa, fs)
    signal = np.abs(auroc - 0.5) * 2.0
    activity = np.abs(np.concatenate([fa, fs])).mean(0)
    activity = activity / (activity.max() + 1e-9)

    # Displayed cells: most active units, with the strongest-signal units
    # guaranteed a slot so the clue cells are always on screen.
    by_act = list(np.argsort(activity)[::-1][:N_DISPLAY])
    for mi in np.argsort(signal)[::-1][:N_RING]:
        if int(mi) not in by_act:
            weakest = min(
                (u for u in by_act if u not in np.argsort(signal)[::-1][:N_RING]),
                key=lambda u: signal[u],
            )
            by_act[by_act.index(weakest)] = int(mi)
    display = sorted(int(u) for u in by_act)

    u_auroc = unit_auroc(ua, us)
    ring_rank = sorted(display, key=lambda u: signal[u], reverse=True)[:N_RING]
    sig_max = max(signal[u] for u in ring_rank) + 1e-9

    data = {
        "source": f"fullruns/l3_h8_heldout/states d0.45 seed {seed} "
                  "(saved pools; H2 integrity-gate reference dumps)",
        "published_mean": PUBLISHED_MEAN,
        "seed": seed,
        "seed_pooled_target": round(float(per_seed[seed]), 4),
        "hidden": int(hidden),
        "display_units": display,
        "unit_activity": [round(float(activity[u]), 4) for u in display],
        "unit_auroc": [round(float(auroc[u]), 4) for u in display],
        "unit_signal": [round(float(signal[u] / sig_max), 4) for u in display],
        "untrained_unit_auroc": [round(float(u_auroc[u]), 4) for u in display],
        "ring_units": [int(u) for u in ring_rank],
        "obs_masking": {
            "none": round(agg["none_survival_mean"], 4),
            "vision_masked": round(agg["vision_survival_mean"], 4),
            "intero_masked": round(agg["intero_survival_mean"], 4),
            "all_masked": round(agg["all_survival_mean"], 4),
        },
    }

    js = ("// GENERATED by scripts/dump_brain_film_data.py - do not edit.\n"
          "// Real per-unit world-signal from the saved heldout pools.\n"
          "window.BRAIN_DATA = " + json.dumps(data, indent=1) + ";\n")
    Path(a.out).write_text(js)
    print(f"wrote {a.out}")
    print("display units:", display)
    print("ring units (by signal):", ring_rank)
    print("unit auroc:", [round(float(auroc[u]), 3) for u in display])
    print("untrained auroc:", [round(float(u_auroc[u]), 3) for u in display])


if __name__ == "__main__":
    main()
