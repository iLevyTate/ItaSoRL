"""The 95% sensitivity table (G09) and the Holm family-wise view of the bar clauses (G08) are
rendered by scripts/build_paper_tables.py from the committed per-seed lists. The rows the
manuscript typed by hand must reproduce to three decimals, the five runs of 2026-10-11 are now
listed, and the intervals agree with scipy on synthetic fixtures."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_paper_tables as bpt  # noqa: E402

# The rows of the hand-typed table (manuscript revision of 2026-10-16), label | mean | t90 | t95.
TYPED = r"""
Corrected C1, pooled target & 0.733 & \ci{0.669}{0.797} & \ci{0.654}{0.812}
Corrected C1, behavior-controlled (seven channels) & 0.723 & \ci{0.670}{0.776} & \ci{0.658}{0.788}
Corrected C1, behavior plus actions controlled & 0.681 & \ci{0.633}{0.728} & \ci{0.622}{0.739}
Corrected C1, best-fitting nonlinear control & 0.620 & \ci{0.584}{0.656} & \ci{0.575}{0.665}
Corrected C1, observation and previous-action stream decoder & 0.724 & \ci{0.670}{0.778} & \ci{0.657}{0.790}
Corrected C2, survival without auxiliary & 0.613 & \ci{0.552}{0.675} & \ci{0.537}{0.690}
Corrected L1 organism, pooled target & 0.551 & \ci{0.531}{0.570} & \ci{0.527}{0.575}
L3 hidden 8, pooled target (historical) & 0.752 & \ci{0.698}{0.807} & \ci{0.686}{0.819}
Authentic-raised readout, historical held-out agents & 0.760 & \ci{0.716}{0.804} & \ci{0.706}{0.814}
Drift-0.45 registered draw, historical held-out agents & 0.774 & \ci{0.727}{0.821} & \ci{0.716}{0.832}
L3 hidden 8, behavior-controlled (four channels) & 0.726 & \ci{0.679}{0.772} & \ci{0.668}{0.783}
L3 hidden 8, behavior-controlled (with position, heading) & 0.723 & \ci{0.676}{0.769} & \ci{0.666}{0.779}
L3 hidden 8, sensory-controlled & 0.731 & \ci{0.690}{0.772} & \ci{0.680}{0.782}
L3 hidden 7, sensory-controlled & 0.684 & \ci{0.627}{0.741} & \ci{0.613}{0.754}
L3 hidden 8, sensory and behavior jointly controlled & 0.670 & \ci{0.638}{0.702} & \ci{0.630}{0.709}
L3 hidden 8, nonlinear joint control & 0.654 & \ci{0.621}{0.687} & \ci{0.613}{0.695}
L3 hidden 8, survival-with-auxiliary arm on CPU (device control) & 0.730 & \ci{0.668}{0.791} & \ci{0.654}{0.806}
L3 hidden 8, predictor target & 0.573 & \ci{0.542}{0.604} & \ci{0.535}{0.612}
L3 hidden 8, untrained target & 0.488 & \ci{0.457}{0.518} & \ci{0.450}{0.526}
L3 hidden 7, pooled target & 0.737 & \ci{0.682}{0.791} & \ci{0.669}{0.804}
L3 hidden 7, predictor target & 0.714 & \ci{0.683}{0.745} & \ci{0.675}{0.753}
L3 hidden 7, behavior-controlled & 0.722 & \ci{0.672}{0.773} & \ci{0.660}{0.784}
L3 hidden 8, survival without auxiliary & 0.601 & \ci{0.549}{0.654} & \ci{0.537}{0.666}
L3 hidden 10, $G$ seed 1, survival & 0.639 & \ci{0.610}{0.669} & \ci{0.603}{0.675}
L3 hidden 10, $G$ seed 1, behavior-controlled & 0.658 & \ci{0.625}{0.691} & \ci{0.617}{0.698}
L3 hidden 10, $G$ seed 1, survival, GPU re-measure & 0.612 & \ci{0.582}{0.642} & \ci{0.575}{0.649}
L3 hidden 10, $G$ seed 1, behavior-controlled, GPU re-measure & 0.638 & \ci{0.612}{0.663} & \ci{0.606}{0.669}
L3 hidden 8, $G$ seed 2, survival & 0.676 & \ci{0.636}{0.717} & \ci{0.627}{0.726}
L3 hidden 8, $G$ seed 2, behavior-controlled & 0.660 & \ci{0.631}{0.689} & \ci{0.624}{0.696}
Transfer, hidden 8 to held-out hidden 7 & 0.773 & \ci{0.722}{0.824} & \ci{0.711}{0.836}
Transfer, hidden 7 to held-out hidden 8 & 0.638 & \ci{0.600}{0.676} & \ci{0.592}{0.685}
Cross-recipe transfer (RFF) & 0.684 & \ci{0.654}{0.715} & \ci{0.646}{0.722}
Common garden, hidden-8 tail & 0.666 & \ci{0.624}{0.709} & \ci{0.615}{0.718}
Common garden, hidden-7 tail & 0.684 & \ci{0.645}{0.723} & \ci{0.636}{0.733}
B-v3 regime, pooled target & 0.610 & \ci{0.581}{0.639} & \ci{0.575}{0.646}
L2 supervised reference & 0.596 & \ci{0.573}{0.619} & \ci{0.567}{0.625}
L1 organism, pooled target (historical) & 0.533 & \ci{0.506}{0.561} & \ci{0.500}{0.567}
Graded seam, $\alpha = 0$ & 0.506 & \ci{0.482}{0.531} & \ci{0.476}{0.536}
B1 balanced survival (sensitivity) & 0.800 & \ci{0.772}{0.829} & \ci{0.765}{0.836}
B2 balanced pellets (sensitivity) & 0.843 & \ci{0.825}{0.860} & \ci{0.821}{0.864}
B1 sensory residual (sensitivity) & 0.761 & \ci{0.727}{0.795} & \ci{0.719}{0.802}
B2 sensory residual (sensitivity) & 0.780 & \ci{0.757}{0.804} & \ci{0.752}{0.809}
""".strip().splitlines()


def test_every_hand_typed_row_reproduces_to_three_decimals():
    rendered = {ln.rsplit(" \\\\", 1)[0] for ln in bpt.ci95_table().splitlines() if ln.endswith(" \\\\")}
    missing = [r for r in TYPED if r not in rendered]
    assert not missing, missing
    assert len(TYPED) == 42


def test_the_runs_of_2026_10_11_are_listed():
    rows = {r["artifact"]: r for r in bpt.ci95_rows()}
    for art, lo95 in (("expB2/goal_stakes_T-touch.json", 0.754), ("expB2/goal_stakes_S-immortal.json", 0.684)):
        assert round(rows[art]["t95"][0], 3) == lo95       # the two lower bounds the appendix quotes
    for art in ("expB2/corrected_H7-corrected.json", "expB2/corrected_H10GS1-corrected.json",
                "expB2/corrected_H8GS2-corrected.json"):
        assert art in rows and rows[art]["block"] == "corrected" and rows[art]["n"] == 10


def test_the_table_is_registered_and_marked_generated():
    t = bpt.tables()
    assert "ci95.tex" in t and "do not edit" in t["ci95.tex"]
    assert t["ci95.tex"].count(r"\midrule") == 3          # header rule plus two block breaks


def _fixture(tmp_path, monkeypatch, docs: dict, rows: tuple) -> None:
    for name, doc in docs.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(bpt, "ART", str(tmp_path))
    monkeypatch.setattr(bpt, "CI95_ROWS", rows)


def test_intervals_match_scipy_on_a_fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(0.7, 0.05, 10)
    y = rng.normal(0.6, 0.08, 7)
    _fixture(tmp_path, monkeypatch, {"a.json": {"block": {"per_seed": list(x)}},
                                     "b.json": {"cells": [{"seed": s, "arm": "survival", "m": float(v)}
                                                          for s, v in reversed(list(enumerate(y)))]}},
             (("one", "x", "a.json", "block"), ("two", "y", "b.json", "cells:arm=survival:m")))
    rows = bpt.ci95_rows()
    for r, v in zip(rows, (x, y)):
        for key, level in (("t90", 0.90), ("t95", 0.95)):
            lo, hi = stats.t.interval(level, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            assert r[key] == pytest.approx((lo, hi), abs=1e-12)
        assert r["mean"] == pytest.approx(v.mean(), abs=1e-15) and r["n"] == len(v)
    t = bpt.ci95_table()
    assert t.count(r"\midrule") == 2 and f"x & {x.mean():.3f} & " in t


def test_a_cells_selector_that_repeats_a_seed_fails(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, {"b.json": {"cells": [{"seed": 0, "m": 0.5}, {"seed": 0, "m": 0.6}]}},
             (("one", "y", "b.json", "cells:seed=0:m"),))
    with pytest.raises(ValueError, match="one seed twice"):
        bpt.ci95_rows()


def test_a_non_finite_seed_value_fails(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, {"a.json": {"v": [0.5, float("nan"), 0.6]}}, (("one", "x", "a.json", "v"),))
    with pytest.raises(ValueError, match="non-finite"):
        bpt.ci95_rows()


# ---- G08: Holm, post hoc, over the bar clauses read on the t90 lower bound ----

def test_holm_step_down_on_a_known_example():
    adj, rej = bpt._holm([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06, stop), 0.04
    assert rej == [True, False, False, True]
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_rows_reproduce_the_ledger_and_c1_fails_in_its_family():
    rows = bpt.holm_rows()
    ledger = bpt.load_ledger()
    want = sum(1 for r in ledger["rows"] for c in r["clauses"]
               if c["role"] == "rule" and c["basis"] == "t90 lower bound" and c["threshold"] == 0.65)
    assert len(rows) == want
    fam_c, fam_all = (f for f, _ in bpt.HOLM_FAMILIES)
    c1 = next(r for r in rows if r["id"] == "C1-PRIMARY")
    assert round(c1["p_one_sided"], 3) == 0.021
    assert c1["holm"][fam_c]["holds"] is False and c1["holm"][fam_all]["holds"] is False
    assert all(r["holm"][fam_all] is not None for r in rows)
    assert all((r["holm"][fam_c] is None) == (r["trainer"] not in ("corrected", "successor")) for r in rows)
    # Holm never holds a clause whose own t90 lower bound is under the bar
    assert not any(h["holds"] for r in rows if r["t90_lower"] < 0.65 for h in r["holm"].values() if h)


def test_holm_changes_no_ledger_reading():
    before = json.dumps(bpt.load_ledger(), sort_keys=True)
    t = bpt.tables()
    assert "holm.tex" in t and "post hoc" in t["holm.tex"] and "do not edit" in t["holm.tex"]
    assert json.dumps(bpt.load_ledger(), sort_keys=True) == before
