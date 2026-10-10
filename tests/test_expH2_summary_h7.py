"""The hidden-7 graded-seam (A1) column, from its committed artifact.

artifacts/expH2/summary_h7.json is promoted from the gitignored fullruns/l3_h7_ablation by
scripts/promote_expH2_summary.py --l3-hidden 7. Every reading the manuscript and FINDINGS take
from that column is recomputed here from the per-seed values, without the audit script. The
untrained and predictor floors are pinned as recorded: they rise with the seam, so nothing
here asserts that they are flat.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from itasorl.stats import rope_test

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "expH2" / "summary_h7.json"
BUNDLE = ROOT / "fullruns" / "l3_h7_ablation" / "aggregate.json"
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
BAR = 0.65


def _art() -> dict:
    return json.loads(ART.read_text(encoding="utf-8"))


def _means(d, arm):
    return [float(np.mean(d["cells"][arm][f"a{al:.2f}"]["per_seed"])) for al in ALPHAS]


def test_provenance_and_configuration():
    d = _art()
    assert d["generated_by"] == "scripts/promote_expH2_summary.py"
    assert d["source_run"] == "fullruns/l3_h7_ablation/aggregate.json"
    assert d["config"]["l3_hidden"] == 7 and d["config"]["n_seeds"] == 10
    assert d["config"]["alphas"] == ALPHAS and d["config"]["drift"] == 0.45
    assert set(d["config"]["arms"]) == {"survival", "untrained", "predictor"}
    assert "hidden=7" in d["surrogate"]
    assert d["git_commit_at_run"].startswith("not recorded in the bundle")


def test_stored_cells_are_consistent_with_their_per_seed_values():
    d = _art()
    for arm, cells in d["cells"].items():
        for key, c in cells.items():
            v = [float(x) for x in c["per_seed"]]
            assert len(v) == 10, (arm, key)
            assert abs(np.mean(v) - c["mean"]) <= 1e-4, (arm, key)
            assert sum(x >= BAR for x in v) == c["n_ge_065"], (arm, key)


def test_the_survival_column_and_its_strict_collapse():
    d = _art()
    surv = _means(d, "survival")
    # tab:h2 column 4, alpha 0 to 1
    assert [round(m, 3) for m in surv] == [0.472, 0.554, 0.650, 0.704, 0.725, 0.737]
    assert [round(m, 4) for m in surv] == d["survival_curve"]
    assert all(b > a for a, b in zip(surv, surv[1:]))
    assert d["survival_monotonicity_rho"] == 1.0


def test_the_alpha_zero_anchor_leaves_the_equivalence_region_at_the_lower_edge():
    d = _art()
    rr = rope_test(d["cells"]["survival"]["a0.00"]["per_seed"], rope=(0.45, 0.55))
    assert round(rr.mean, 3) == 0.472
    assert [round(x, 3) for x in rr.hdi] == [0.445, 0.501]
    assert not rr.accept and rr.accept == d["l0_anchor_alpha0"]["accept_equiv"]
    assert abs(rr.p_in_rope - d["l0_anchor_alpha0"]["p_in_rope"]) <= 1e-4


def test_the_integrity_gate_reproduced_the_hidden_7_headline():
    d = _art()
    assert d["integrity"]["pools_bit_match"] is True
    assert d["integrity"]["survival_mean_alpha1"] == d["integrity"]["published_target"] == 0.737
    assert round(_means(d, "survival")[-1], 3) == 0.737


def test_the_hidden_7_floors_rise_with_the_seam_as_recorded():
    d = _art()
    untr, pred = _means(d, "untrained"), _means(d, "predictor")
    assert [round(m, 3) for m in untr] == [0.474, 0.482, 0.530, 0.574, 0.578, 0.586]
    assert [d["cells"]["untrained"][f"a{al:.2f}"]["n_ge_065"] for al in ALPHAS] == [0, 0, 0, 2, 2, 2]
    assert [round(m, 3) for m in pred] == [0.500, 0.518, 0.606, 0.655, 0.692, 0.714]


@pytest.mark.skipif(not BUNDLE.exists(), reason="gitignored run bundle not present")
def test_the_promoter_reproduces_the_artifact_from_the_bundle():
    sys.path.insert(0, str(ROOT / "scripts"))
    import promote_expH2_summary as pes
    have = _art()
    with open(BUNDLE, encoding="utf-8") as fh:
        run = json.load(fh)
    fresh = pes.summarize(run, source=os.fspath(BUNDLE), l3_hidden=7,
                          commit_at_run=have["git_commit_at_run"],
                          head=have["git_commit_at_promotion"])
    assert json.loads(json.dumps(fresh)) == have
