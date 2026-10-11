"""The gate-0 pair-clustered interval artifact (finding R26) recomputes from its own rows.

artifacts/agent_free/gate0_pair_intervals.json stores, per calibrated instance and fold
scheme, the oracle's out-of-fold scores with pair ids. These tests check that the stored
point, pooled AUROC and intervals follow from those rows, that every recorded value the
artifact quotes matches its committed source, that every recorded point reproduced, and
(one live instance, from the committed CPU export) that the stored rows are what the code
produces. No gitignored bundle is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "artifacts" / "agent_free" / "gate0_pair_intervals.json"


@pytest.fixture(scope="module")
def doc():
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _rows(doc):
    for inst in doc["instances"]:
        for scheme, block in inst["schemes"].items():
            yield inst, scheme, block


def test_points_follow_from_stored_oof(doc):
    from itasorl.stats import auroc
    for inst, scheme, b in _rows(doc):
        oof = b["oof"]
        y, s, f, pair = (np.asarray(oof[k]) for k in ("label", "score", "fold", "pair"))
        assert len(y) == 120 and len(np.unique(pair)) == 60, inst["id"]
        # every pair contributes one authentic and one surrogate branch, in one test fold
        for p in np.unique(pair):
            m = pair == p
            assert sorted(y[m].tolist()) == [0, 1] and len(set(f[m].tolist())) == 1
        fold_auc = [auroc(y[f == k], s[f == k]) for k in sorted(set(f.tolist()))]
        assert abs(np.mean(fold_auc) - b["oracle_auroc"]) < 1e-12, (inst["id"], scheme)
        assert abs(auroc(y, s) - b["pooled_oof_auroc"]) < 1e-12, (inst["id"], scheme)
        assert b["in_band_point"] == (0.85 <= b["oracle_auroc"] <= 0.95)


@pytest.mark.parametrize("iid,scheme", [("mlp_h8_s0", "legacy"), ("mlp_h7_s1", "explicit")])
def test_intervals_recompute(doc, iid, scheme):
    import gate0_pair_intervals as g0
    inst = next(i for i in doc["instances"] if i["id"] == iid)
    b = inst["schemes"][scheme]
    again = g0.intervals_from_oof(b["oof"])
    for k, v in again.items():
        if isinstance(v, list):
            assert np.allclose(v, b[k], rtol=0, atol=1e-12), k
        else:
            assert v == b[k] or abs(v - b[k]) < 1e-12, k


def test_band_call_labels():
    import gate0_pair_intervals as g0
    assert g0.band_call(0.86, 0.94) == "inside"
    assert g0.band_call(0.80, 0.84) == "below"
    assert g0.band_call(0.96, 0.99) == "above"
    assert g0.band_call(0.84, 0.96) == "straddles 0.85 and 0.95"
    assert g0.band_call(0.89, 0.97) == "straddles 0.95"


def test_recorded_values_match_committed_sources(doc):
    for inst in doc["instances"]:
        for rec in inst["recorded"]:
            src = ROOT / rec["source"]
            if rec["source"].startswith("fullruns/"):
                continue                      # gitignored local bundle; quoted, not checked
            assert src.exists(), rec["source"]
            text = src.read_text(encoding="utf-8")
            assert rec["value"] is not None
            assert repr(rec["value"]) in text or f"{rec['value']}" in text, (inst["id"], rec)


def test_every_recorded_point_reproduced(doc):
    """The finding the artifact supports: on this stack every recorded gate-0 oracle value
    reproduces under its recorded fold scheme. A future regeneration that breaks one must say
    so in the results note, not pass silently."""
    for inst in doc["instances"]:
        for rec in inst["recorded"]:
            if rec["value"] is None:
                continue
            assert rec["reproduces"], (inst["id"], rec)


def test_live_recompute_h8_s0_explicit(doc):
    pytest.importorskip("sklearn")
    from itasorl import folds
    from itasorl.experiment_a_l3 import generate_l3_pairs, run_experiment_a_l3
    from itasorl.surrogate_l3 import GMotion
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    g = GMotion.from_npz(str(ROOT / "artifacts" / "surrogates" / "gmotion_h8_s0.npz"))
    eps = generate_l3_pairs(g, n_pairs=60, branch=30, seed0=3000, params=P)
    with folds.fold_scheme("explicit"):
        res = run_experiment_a_l3(eps, sigma_meas=0.02, seed=0, return_oof=True)
    b = next(i for i in doc["instances"] if i["id"] == "mlp_h8_s0")["schemes"]["explicit"]
    assert res["oracle_auroc"] == b["oracle_auroc"]
    assert res["oracle_oof"]["pair"] == b["oof"]["pair"]
    assert res["oracle_oof"]["fold"] == b["oof"]["fold"]
    assert np.allclose(res["oracle_oof"]["score"], b["oof"]["score"], rtol=0, atol=1e-12)
