"""The residual-field correlation artifact recomputes from its stored per-episode sums.

artifacts/agent_free/residual_field_correlation.json stores, for each fresh episode and each
velocity component, the sums and cross-product sums of every fingerprint's residual field.
Every correlation and every episode-bootstrap interval follows from those alone; one live
check regenerates three fresh episodes from the committed CPU exports and compares their
sums. No gitignored bundle is needed.
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

ART = ROOT / "artifacts" / "agent_free" / "residual_field_correlation.json"


@pytest.fixture(scope="module")
def doc():
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def test_matrices_and_intervals_recompute(doc):
    import residual_field_correlation as rfc
    again = rfc.summarize(doc["episode_stats"], doc["names"])
    for view in ("pooled", "x", "y"):
        for key in ("r", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi"):
            assert np.allclose(again["matrices"][view][key], doc["matrices"][view][key],
                               rtol=0, atol=1e-12), (view, key)
    assert again["pairs"] == doc["pairs"] or all(
        np.allclose([a[k] for k in ("r_pooled", "r_x", "r_y", "r2_pooled")],
                    [b[k] for k in ("r_pooled", "r_x", "r_y", "r2_pooled")], rtol=0, atol=1e-12)
        for a, b in zip(again["pairs"], doc["pairs"]))


def test_matrix_is_a_correlation_matrix(doc):
    R = np.asarray(doc["matrices"]["pooled"]["r"])
    assert np.allclose(np.diag(R), 1.0) and np.allclose(R, R.T)
    assert np.all(np.linalg.eigvalsh(R) > -1e-10)


def test_fresh_sample_disjoint_from_training(doc):
    from itasorl.surrogate_l3 import HELDOUT_SEED
    lo, hi = doc["fresh_sample"]["episodes"]
    assert lo == HELDOUT_SEED and hi - lo == doc["fresh_sample"]["n_episodes"]
    for fp in doc["fingerprints"]:
        a, b = fp["train_episodes"]
        assert hi <= a or b <= lo, fp["name"]


def test_quoted_values_are_inside_or_near(doc):
    """The scope note's two quoted values, as the artifact records them."""
    q = {(p["a"], p["b"]): p for p in doc["pairs"] if "quoted_in_findings_10_5" in p}
    assert set(q) == {("h8_s0", "h7_s0"), ("h8_s0", "rff_D80")}
    for p in q.values():
        lo, hi = p["ci95_pooled"]
        assert abs(p["r_pooled"] - p["quoted_in_findings_10_5"]) < 0.05
        assert lo < hi


def test_live_episode_sums_from_exports(doc):
    """Three fresh episodes regenerated from the committed exports give the stored sums."""
    import residual_field_correlation as rfc
    from itasorl.surrogate_l3 import HELDOUT_SEED, GMotion, collect_authentic_transitions
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    names = doc["names"]
    exports = {n: GMotion.from_npz(str(ROOT / fp["source"]))
               for n, fp in zip(names, doc["fingerprints"]) if fp["source"].endswith(".npz")}
    X, _ = collect_authentic_transitions(n_eps=3, steps=40, params=P, seed0=HELDOUT_SEED)
    Xd = X.astype(float)
    f = (1.0 - P.k_land * P.dt) * Xd[:, :2] + Xd[:, 2:] * P.dt
    R = {n: (np.stack([np.asarray(g(x[:2], x[2:], None), float) for x in X]) - f).reshape(3, 40, 2)
         for n, g in exports.items()}
    pairs = rfc.triu_pairs(len(names))
    for c, comp in (("x", 0), ("y", 1)):
        S = np.asarray(doc["episode_stats"][c]["S"])[:3]
        Q = np.asarray(doc["episode_stats"][c]["Q"])[:3]
        for i, n in enumerate(names):
            if n in R:
                assert np.allclose(R[n][..., comp].sum(1), S[:, i], rtol=1e-10, atol=1e-12), (c, n)
        for k, (i, j) in enumerate(pairs):
            if names[i] in R and names[j] in R:
                want = (R[names[i]][..., comp] * R[names[j]][..., comp]).sum(1)
                assert np.allclose(want, Q[:, k], rtol=1e-10, atol=1e-14), (c, names[i], names[j])
