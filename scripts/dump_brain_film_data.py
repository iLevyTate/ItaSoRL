"""Extract real per-unit brain data for the "Two Minds" film renderer.

Reads drift-0.45 state pools rebuilt from the corrected C1 agents and writes
viz/player/brain/brain-data.js (window.BRAIN_DATA = {...}). No model is loaded and no
rollout is run here: the pools come from scripts/regenerate_state_dumps.py, which replays
the standard pooled readout on saved agents and checks each rebuilt target against the
run's own cells.

Two checks run before anything is written, and either one stops the script:

1. The film's numbers. viz/player/brain/brain.js keeps every number it puts on screen in
   its FILM block. Each display must equal the committed corrected-trainer value, rounded
   to a whole percent, and each `value` must equal it to 3 dp: artifacts/corrected_verdicts.json
   (untrained, trained, no-auxiliary at 300 updates, L0 gate, bar),
   artifacts/policy_controls/corrected_l3_h8_wm.json (predict-only trunk on the survival
   agent's replayed streams) and artifacts/budget_curve.json (no-auxiliary at 450 updates).
   The claims the film makes about those numbers are checked too: the untrained interval
   includes 0.5 (it is called a coin flip), the trained arm clears the bar, the arm without
   the prediction job is under it at 300 updates, and the L0 gate did not pass (the end card
   says the preregistered test is not passed).
2. The brain. The representative brain is the drift-0.45 seed whose committed C1 survival
   target sits closest to the C1 survival mean (ties go to the lower seed). Its rebuilt
   survival and untrained pools must reproduce the committed cells' pooled targets
   (artifacts/corrected_runs/corrected_l3_h8_wm/cells) under the explicit fold partition,
   or the dumps are not C1's and nothing is written.

Per recurrent unit, the script then computes how strongly its step-level activation
separates the real world from the fake one (rank AUROC, the dependency-light analog of
the abs logistic coefficient used by the original viz_mind_learning brain graph). The 12
displayed memory cells are the most active units of that brain, with the top world-signal
units swapped in so the story cells are always on screen. Ring strength is the normalized
signal.

Usage (from repo root):
    python scripts/dump_brain_film_data.py --check-only
    python scripts/dump_brain_film_data.py --states DIR [--report REPORT.json]
DIR holds states_d0.45_s{seed}_{survival,untrained}.npz as regenerate_state_dumps.py
writes them; REPORT.json is its --report output, which must say all_reproduced.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS_REL = Path("artifacts") / "corrected_runs" / "corrected_l3_h8_wm" / "cells"
RENDERER = ROOT / "viz" / "player" / "brain" / "brain.js"
OUT = ROOT / "viz" / "player" / "brain" / "brain-data.js"

DRIFT_TAG = "0.45"
HIDDEN = 96              # the registered trunk
N_DISPLAY = 12           # memory cells shown in the film
N_RING = 6               # teal-ringed clue cells
FILM_KEYS = ("untrained", "trained", "replay", "noAux300", "noAux450", "l0", "bar")

_FILM_BLOCK = re.compile(r"^const FILM = \{\n(.*?)\n\};", re.S | re.M)
_FILM_ROW = re.compile(r'^\s*(\w+): \{ value: ([0-9.]+), display: "(\d+%)" \},', re.M)


class NumberHonesty(SystemExit):
    """Raised when a number, or a dump, disagrees with the committed corrected record."""


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise NumberHonesty("number honesty: " + msg)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def load_record(root: Path = ROOT) -> dict:
    """The corrected-trainer numbers the film shows, and C1's per-seed cells, read from the
    committed artifacts. Nothing here comes from a dump or a rerun."""
    art = root / "artifacts"
    v = json.loads((art / "corrected_verdicts.json").read_text(encoding="utf-8"))
    c1, c2 = v["runs"]["C1"], v["runs"]["C2"]
    pol = json.loads((art / "policy_controls" / "corrected_l3_h8_wm.json").read_text(encoding="utf-8"))
    bud = json.loads((art / "budget_curve.json").read_text(encoding="utf-8"))

    def budget(series: str, updates: int) -> float:
        pts = [p for p in bud["series"][series] if p["updates"] == updates]
        need(len(pts) == 1, f"budget_curve.json has {len(pts)} '{series}' points at {updates} updates")
        return float(pts[0]["target"]["mean"])

    # The budget curve's 300-update points are C1 and C2 themselves; if they disagree the
    # artifacts are out of step with each other and no number on screen can be trusted.
    need(abs(budget("decoder on", 300) - c1["primary"]["survival"]["mean"]) < 1e-5,
         "budget_curve.json decoder-on at 300 updates is not C1's survival mean")
    need(abs(budget("decoder off", 300) - c2["primary"]["survival"]["mean"]) < 1e-5,
         "budget_curve.json decoder-off at 300 updates is not C2's survival mean")

    cells = {}
    for path in sorted((root / CELLS_REL).glob(f"cell_d{DRIFT_TAG}_s*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))["cell"]
        cells[int(cell["seed"])] = {
            "gae_bootstrap": cell.get("gae_bootstrap"),
            **{arm: float(cell["agents"][arm]["pool"]["target"]) for arm in cell["agents"]},
        }
    need(len(cells) == 10, f"expected 10 committed C1 cells at drift {DRIFT_TAG}, found {len(cells)}")
    return {
        "untrained": float(c1["primary"]["untrained"]["mean"]),
        "untrained_t90": [float(x) for x in c1["primary"]["untrained"]["t90"]],
        "trained": float(c1["primary"]["survival"]["mean"]),
        "replay": float(pol["aggregate"]["predictor replay"]["mean"]),
        "noAux300": float(c2["primary"]["survival"]["mean"]),
        "noAux450": budget("decoder off", 450),
        "l0": float(c1["gates"]["l0"]["mean"]),
        "l0_pass": bool(c1["gates"]["l0"]["pass"]),
        "c1_met": bool(c1["primary"]["met"]),
        "bar": float(v["rules"]["bar"]),
        "cells": cells,
    }


def renderer_numbers(path: Path = RENDERER) -> dict:
    """Parse brain.js's FILM block into {key: (value, display)}."""
    src = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    m = _FILM_BLOCK.search(src)
    need(m is not None, f"no FILM block in {path}")
    rows = _FILM_ROW.findall(m.group(1))
    need(len(rows) == len(m.group(1).strip().splitlines()),
         "every FILM line must read `key: { value: 0.123, display: \"12%\" },`")
    return {k: (float(val), disp) for k, val, disp in rows}


def verify_film_numbers(film: dict, record: dict) -> dict:
    """Binding rule: every number Two Minds shows matches the committed corrected-trainer
    artifacts, and the sentences built on them stay true. Returns the verified values."""
    need(set(film) == set(FILM_KEYS), f"FILM keys {sorted(film)}, expected {sorted(FILM_KEYS)}")
    for k in FILM_KEYS:
        value, display = film[k]
        want = float(record[k])
        need(display == _pct(want), f"{k}: brain.js shows {display}, the record reads {want:.4f} ({_pct(want)})")
        need(abs(value - round(want, 3)) < 1e-9, f"{k}: brain.js value {value} is not the record's {want:.4f} to 3 dp")
    lo, hi = record["untrained_t90"]
    need(lo <= 0.5 <= hi, f"untrained interval [{lo:.3f}, {hi:.3f}] excludes chance, so it is not a coin flip")
    need(record["trained"] >= record["bar"], "the trained arm is under the bar; the film says it clears it")
    need(record["noAux300"] < record["bar"],
         "the arm without the prediction job clears the bar at 300 updates; the end card says it falls below")
    need(not record["l0_pass"] and not record["c1_met"],
         "the end card says the preregistered test is not passed, and the record says it is")
    return {k: round(float(record[k]), 4) for k in FILM_KEYS}


def representative_seed(cells: dict, mean: float) -> int:
    """The seed whose survival target sits closest to the arm mean; ties go to the lower seed."""
    return min(sorted(cells), key=lambda s: (abs(cells[s]["survival"] - mean), s))


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


def pool_target(ha: np.ndarray, hs: np.ndarray, features=None) -> float:
    """The pooled target exactly as itasorl.experiment_b2.pooled_readout scores it: episode
    features of (Ha ++ Hs), one CV group per episode, mean of the five explicit-fold AUROCs."""
    from itasorl import folds
    from itasorl.experiment_a import grouped_auroc
    if features is None:
        from itasorl.experiment_b import episode_features as features
    H = np.concatenate([ha, hs])
    y = np.concatenate([np.zeros(len(ha)), np.ones(len(hs))]).astype(int)
    with folds.fold_scheme("explicit"):
        return float(grouped_auroc(features(H), y, np.arange(len(y))))


def load_checked_pool(states: Path, seed: int, arm: str, recorded: float, atol: float,
                      features=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Load one rebuilt pool and refuse it unless it reproduces the committed target."""
    path = Path(states) / f"states_d{DRIFT_TAG}_s{seed}_{arm}.npz"
    need(path.exists(), f"missing {path}; rebuild the pools first (scripts/regenerate_state_dumps.py)")
    with np.load(path) as d:
        ha, hs = np.asarray(d["Ha"]), np.asarray(d["Hs"])
        drift = float(d["drift_sigma"])
    need(abs(drift - float(DRIFT_TAG)) < 1e-12, f"{path.name} was dumped at drift {drift}, not {DRIFT_TAG}")
    need(ha.ndim == 3 and hs.ndim == 3 and ha.shape[1:] == hs.shape[1:],
         f"{path.name}: pools are {ha.shape} and {hs.shape}, expected (episodes, steps, hidden)")
    got = pool_target(ha, hs, features)
    need(abs(got - recorded) <= atol,
         f"{arm} seed {seed}: the rebuilt pool reads {got!r}, the committed C1 cell reads {recorded!r}; "
         "these are not C1's dumps")
    return ha, hs, got


def brain_data(fa: np.ndarray, fs: np.ndarray, ua: np.ndarray, us: np.ndarray) -> dict:
    """Per-unit film data from step-level survival (fa, fs) and untrained (ua, us) pools."""
    auroc = unit_auroc(fa, fs)
    signal = np.abs(auroc - 0.5) * 2.0
    activity = np.abs(np.concatenate([fa, fs])).mean(0)
    activity = activity / (activity.max() + 1e-9)

    # Displayed cells: most active units, with the strongest-signal units
    # guaranteed a slot so the clue cells are always on screen.
    top_signal = [int(u) for u in np.argsort(signal)[::-1][:N_RING]]
    by_act = [int(u) for u in np.argsort(activity)[::-1][:N_DISPLAY]]
    for mi in top_signal:
        if mi not in by_act:
            weakest = min((u for u in by_act if u not in top_signal), key=lambda u: signal[u])
            by_act[by_act.index(weakest)] = mi
    display = sorted(by_act)

    u_auroc = unit_auroc(ua, us)
    ring_rank = sorted(display, key=lambda u: signal[u], reverse=True)[:N_RING]
    sig_max = max(signal[u] for u in ring_rank) + 1e-9
    return {
        "display_units": display,
        "unit_activity": [round(float(activity[u]), 4) for u in display],
        "unit_auroc": [round(float(auroc[u]), 4) for u in display],
        "unit_signal": [round(float(signal[u] / sig_max), 4) for u in display],
        "untrained_unit_auroc": [round(float(u_auroc[u]), 4) for u in display],
        "ring_units": [int(u) for u in ring_rank],
    }


def check_report(report: Path, seed: int) -> None:
    """regenerate_state_dumps.py --report output: every rebuilt cell must have reproduced."""
    rep = json.loads(Path(report).read_text(encoding="utf-8"))
    need(bool(rep.get("all_reproduced")), f"{report} says not every rebuilt target reproduced")
    have = {(round(c["drift"], 2), c["seed"], c["arm"]) for c in rep.get("cells", []) if c.get("reproduced")}
    for arm in ("survival", "untrained"):
        need((float(DRIFT_TAG), seed, arm) in have, f"{report} has no reproduced {arm} cell at seed {seed}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", help="directory of rebuilt pools, states_d0.45_s{seed}_{arm}.npz")
    ap.add_argument("--report", help="regenerate_state_dumps.py --report json; must say all_reproduced")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="largest allowed |rebuilt - committed| pooled target (default 1e-9)")
    ap.add_argument("--check-only", action="store_true",
                    help="check brain.js's numbers against the record, name the seed, write nothing")
    a = ap.parse_args(argv)

    record = load_record()
    numbers = verify_film_numbers(renderer_numbers(), record)
    print("film numbers match the corrected record: " + json.dumps(numbers), flush=True)
    seed = representative_seed(record["cells"], record["trained"])
    cell = record["cells"][seed]
    print(f"representative seed = {seed} (committed survival target {cell['survival']:.4f}, "
          f"C1 mean {record['trained']:.4f})", flush=True)
    if a.check_only:
        return
    need(a.states is not None, "--states is required unless --check-only")
    need(cell["gae_bootstrap"] == "successor", f"committed cell for seed {seed} is not a corrected-trainer cell")
    if a.report:
        check_report(Path(a.report), seed)

    fa3, fs3, t_surv = load_checked_pool(Path(a.states), seed, "survival", cell["survival"], a.atol)
    ua3, us3, t_untr = load_checked_pool(Path(a.states), seed, "untrained", cell["untrained"], a.atol)
    need(fa3.shape[-1] == HIDDEN, f"survival pool has {fa3.shape[-1]} units, the registered trunk has {HIDDEN}")
    exact = a.atol <= 1e-9
    print(("rebuilt pools reproduce the committed targets: " if exact else
           f"rebuilt pools within {a.atol} of the committed targets (cross-stack rebuild): ")
          + f"survival {t_surv!r} vs {cell['survival']!r}, untrained {t_untr!r} vs {cell['untrained']!r}",
          flush=True)

    flat = lambda h: h.reshape(-1, h.shape[-1])
    units = brain_data(flat(fa3), flat(fs3), flat(ua3), flat(us3))
    data = {
        "source": f"C1, corrected trainer, drift {DRIFT_TAG}, seed {seed}: state pools rebuilt from the "
                  "saved agents of the C1 rerun by scripts/regenerate_state_dumps.py",
        "trainer": "corrected",
        "record_mean": round(record["trained"], 4),
        "seed": seed,
        "seed_pooled_target": round(t_surv, 4),
        "untrained_pooled_target": round(t_untr, 4),
        "committed_seed_pooled_target": round(cell["survival"], 4),
        "committed_untrained_pooled_target": round(cell["untrained"], 4),
        "rebuild_atol": a.atol,
        "rebuild_note": ("the rebuilt pools reproduce the committed cells exactly" if exact else
                         "the state pools were regenerated on a different software stack from the run's, "
                         f"so the pooled readings agree with the committed cells within {a.atol}, not to the bit; "
                         "the on-screen numbers come from the committed artifacts, not from these pools"),
        "hidden": int(fa3.shape[-1]),
        **units,
        "film_numbers": numbers,
    }
    js = ("// GENERATED by scripts/dump_brain_film_data.py; do not edit.\n"
          "// Per-unit world-signal of one corrected C1 brain, from its rebuilt state pools.\n"
          "window.BRAIN_DATA = " + json.dumps(data, indent=1) + ";\n")
    Path(a.out).write_text(js, encoding="utf-8")
    print(f"wrote {a.out}")
    print("display units:", units["display_units"])
    print("ring units (by signal):", units["ring_units"])
    print("unit auroc:", units["unit_auroc"])
    print("untrained auroc:", units["untrained_unit_auroc"])


if __name__ == "__main__":
    main()
