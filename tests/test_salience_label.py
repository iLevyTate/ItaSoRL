"""H66: the salience label carries the spec's L0 qualifier.

docs/specs/2026-10-15-salience-design.md line 44: the L0 analogue's k-means agreement should
sit within 0.1 of 0.5; if it does not, every agreement here is reported as carrying the
world-sample draw. summarize() stores that gate as a boolean; the label string carries the
qualifier so a reader of the label alone sees it. The committed artifacts both pass the
gate, so their stored readings are reproduced byte for byte from their stored rows.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_salience_readout as rsr  # noqa: E402

ARTS = ("corrected_l3_h8_wm.json", "corrected_l3_h8_nowm.json")
KEYS = ("share_probe_dir", "share_pc1", "cohen_d_oof", "kmeans_agreement", "kmeans_ari",
        "pc1_sign_agreement")


def _load(name: str) -> dict:
    with open(os.path.join(ROOT, "artifacts", "salience", name), encoding="utf-8") as fh:
        return json.load(fh)


def _rows(n: int, km: float, share: float, l0_km: float) -> list:
    """Synthetic per-seed rows: every arm and protocol whispers (k-means agreement `km`,
    probe-direction share `share`), with the L0 analogue's agreement at `l0_km`. A small
    seed-dependent wobble keeps the t interval finite."""
    rows = []
    for s in range(n):
        eps = 0.001 * s
        block = {k: 0.0 for k in KEYS}
        block["kmeans_agreement"] = km + eps
        block["share_probe_dir"] = share + eps
        l0 = {k: 0.0 for k in KEYS}
        l0["kmeans_agreement"] = l0_km + eps
        rows.append({"seed": s,
                     "own": {g: dict(block) for g in rsr.ARMS},
                     "replay": {g: dict(block) for g in rsr.ARMS},
                     "l0_survival_d0": l0})
    return rows


@pytest.mark.parametrize("name", ARTS)
def test_summarize_reproduces_the_committed_readings_from_their_rows(name):
    art = _load(name)
    got = rsr.summarize(copy.deepcopy(art["rows"]))
    assert got["readings"] == art["readings"]
    assert got["aggregate"] == art["aggregate"]
    assert got["readings"]["l0 within 0.1 of 0.5"] is True
    assert "world-sample draw" not in got["readings"]["label"]


def test_label_is_bare_when_the_l0_analogue_sits_at_chance():
    r = rsr.summarize(_rows(5, km=0.52, share=0.02, l0_km=0.51))["readings"]
    assert r["label"] == "whisper"
    assert r["l0 within 0.1 of 0.5"] is True


@pytest.mark.parametrize("l0_km", (0.61, 0.39, 0.75))
def test_label_carries_the_world_sample_qualifier_when_the_l0_analogue_is_off_chance(l0_km):
    r = rsr.summarize(_rows(5, km=0.52, share=0.02, l0_km=l0_km))["readings"]
    assert r["l0 within 0.1 of 0.5"] is False
    assert r["label"].startswith("whisper")
    assert "carrying the world-sample draw" in r["label"]


def test_qualifier_follows_the_input_clause_on_a_carried_shout():
    # Shout with the untrained floor also loud: both clauses appear, input clause first.
    r = rsr.summarize(_rows(5, km=0.90, share=0.40, l0_km=0.70))["readings"]
    assert r["1 shout"] is True
    assert r["label"] == ("shout (carried by the input) "
                          "(carrying the world-sample draw: L0 analogue not at chance)")
