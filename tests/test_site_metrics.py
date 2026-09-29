"""Contract pins for scripts/site_metrics.py (the single source of the site's numbers).

derive_metrics() reads the committed artifacts/expB2/ tree and returns the display
strings the static site quotes. The whole point of the module is that these EQUAL the
hand-verified headline numbers, so future runs regenerate the page instead of drifting
against string pins. Reads real committed artifacts (small, deterministic); no mocks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

site_metrics = importlib.import_module("site_metrics")


def test_derive_metrics_matches_findings_headline_numbers():
    m = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    # FINDINGS sec.10.2 L3 survival headline + hero (2dp) rounding of the same source.
    assert m["l3_survival"] == "0.752"
    assert m["l3_survival_hero"] == "0.75"
    # Decision interval: t-based 90% CI recomputed from the per-seed survival cells.
    assert m["l3_ci_lo"] == "0.698"
    assert m["l3_ci_hi"] == "0.807"
    # sec.10.6-10.7 transfer + re-scored common-garden (the gate-pinned five).
    assert m["transfer_same"] == "0.773"
    assert m["transfer_reverse"] == "0.638"
    assert m["cg_forward"] == "0.666"
    assert m["cg_reverse"] == "0.684"


def test_derive_metrics_values_are_display_strings():
    m = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    assert all(isinstance(v, str) for v in m.values())


def test_readout_rows_match_findings_per_seed_sources():
    # The readout chart plots these rows; each mean and t-based 90% CI is the one
    # FINDINGS reports for that readout (10.2, 10.4, 10.4.2 and addendum, 10.8).
    import json

    rows = json.loads(site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")["readout_rows"])
    expect = {
        "untrained": (0.488, None, None, 0),
        "predictor": (0.573, None, None, 0),
        "no_predictor": (0.601, 0.549, 0.654, 1),
        "no_predictor_ref": (0.730, 0.668, 0.791, 8),
        "survival": (0.752, 0.698, 0.807, 8),
        "minus_behavior": (0.726, 0.679, 0.772, 9),
        "minus_senses": (0.731, 0.690, 0.772, 8),
        "minus_both_nonlinear": (0.654, 0.621, 0.687, 6),
    }
    assert set(rows) == set(expect)
    for key, (mean, lo, hi, over) in expect.items():
        r = rows[key]
        assert len(r["seeds"]) == 10, key
        assert r["mean"] == mean, key
        assert r["over_bar"] == over, key
        if lo is not None:
            assert (r["lo"], r["hi"]) == (lo, hi), key


def test_readout_rows_literal_has_no_adjacent_braces():
    # The page runtime treats "{{" and "}}" as template bindings, even inside the
    # component script, so the injected JSON must never put two braces together.
    lit = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")["readout_rows"]
    assert "{{" not in lit and "}}" not in lit
