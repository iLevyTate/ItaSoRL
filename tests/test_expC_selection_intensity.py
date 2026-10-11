"""Experiment C selection-intensity artifact (Q5): every derived number follows from the stored
per-generation series, the run's own fitness deltas match the committed milestone summary,
and the artifact is rebuilt from the run outputs when the gitignored bundle is present."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "artifacts" / "expC" / "selection_intensity.json"


@pytest.fixture(scope="module")
def doc():
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def test_derived_numbers_recompute(doc):
    import expC_selection_intensity as si
    n = doc["n_individuals"]
    for b in doc["per_seed"]:
        for arm in ("treatment", "control"):
            e = b[arm]
            again = si.arm_block(n, e["threshold_gen0_median"], e["n_qualifiers"], e["mean_fitness"])
            again["fit_delta_recorded"] = e["fit_delta_recorded"]
            assert again == e, (b["seed"], arm)
            assert len(e["n_qualifiers"]) == doc["generations"] == 30
            assert e["n_qualifiers"][0] >= n // 2          # the gen-0 median threshold
            assert abs(e["fitness_last_minus_gen0"] - e["fit_delta_recorded"]) < 1e-6


def test_matches_committed_milestone_summary(doc):
    with open(ROOT / "artifacts" / "expC" / "emergence_pilot_per_individual_summary.json",
              encoding="utf-8") as fh:
        cells = {c["seed"]: c for c in json.load(fh)["cells"]}
    for b in doc["per_seed"]:
        c = cells[b["seed"]]
        assert b["treatment"]["fit_delta_recorded"] == c["fit_delta_treat"]
        assert b["control"]["fit_delta_recorded"] == c["fit_delta_ctrl"]


def test_record_values(doc):
    """The values the record quotes (FINDINGS 13, selection-intensity note)."""
    s = doc["summary"]
    assert s["treatment"]["first_gen_at_least_90pct"] == [3, 7, 4]
    assert s["control"]["first_gen_at_least_90pct"] == [3, 7, 4]
    assert [round(x, 3) for x in s["treatment"]["mean_share_gens_5_to_29"]] == [0.988, 0.922, 0.951]
    assert [round(x, 3) for x in s["control"]["mean_share_gens_5_to_29"]] == [0.962, 0.973, 0.968]
    t0 = doc["per_seed"][0]["treatment"]
    assert t0["fitness_peak_gen"] == 15 and round(t0["fitness_peak"], 3) == -0.863
    assert round(t0["fitness_last"], 3) == -1.122
    assert all(doc["main_run_series_identical"].values())


def test_rebuild_when_bundle_present(doc):
    import expC_selection_intensity as si
    path = si._full(si.PER_IND)
    if not os.path.exists(path):
        pytest.skip("gitignored milestone-3 output not present")
    with open(path, encoding="utf-8") as fh:
        per_ind = json.load(fh)
    main = None
    if os.path.exists(si._full(si.MAIN)):
        with open(si._full(si.MAIN), encoding="utf-8") as fh:
            main = json.load(fh)
    body = json.loads(json.dumps(si.build(per_ind, main)))
    for k, v in body.items():
        assert doc[k] == v, k
