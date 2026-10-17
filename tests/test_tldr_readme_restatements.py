"""The FINDINGS TL;DR and the README headline paragraphs restate numbers whose home sections the
audit guards; the restatements themselves were guarded by nothing, so an edit there passed every
check (H42). Each needle below is rendered from the committed artifact (never from the home
section), and must appear in the summary surfaces that restate it: the TL;DR (from "## TL;DR" to
"## 1. What was tested") and the README from its opening to "## Repository layout". Text is
compared after whitespace is collapsed and Markdown bold is dropped. Numbers the TL;DR gives
only as rough values (the about 0.99 of Experiment A's diagnostic configuration) are not
listed."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_paper_tables as bpt  # noqa: E402

from itasorl.stats import t_ci90  # noqa: E402


def _flat(s: str) -> str:
    return " ".join(s.replace("**", "").split())


def _surfaces() -> dict[str, str]:
    with open(os.path.join(ROOT, "docs", "FINDINGS.md"), encoding="utf-8") as fh:
        f = fh.read()
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        r = fh.read()
    return {"tldr": _flat(f[f.index("## TL;DR"):f.index("## 1. What was tested")]),
            "readme": _flat(r[:r.index("## Repository layout")])}


def _a(*parts) -> dict:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _ps(art: str, sel: str) -> np.ndarray:
    return np.asarray(bpt.per_seed_values(art, sel), float)


def _mean_iv(x) -> str:
    lo, hi = t_ci90(np.asarray(x, float))
    return f"{np.mean(x):.3f} [{lo:.3f}, {hi:.3f}]"


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def needles() -> list[tuple[str, tuple[str, ...], str]]:
    """(name, surfaces, text), every text rendered from an artifact."""
    v = _a("artifacts", "corrected_verdicts.json")
    c1, c2, ac = v["runs"]["C1"], v["runs"]["C2"], v["auxiliary_comparison"]
    s1 = np.asarray(c1["primary"]["survival"]["per_seed"], float)
    p1 = np.asarray(c1["primary"]["predictor"]["per_seed"], float)
    u1 = np.asarray(c1["primary"]["untrained"]["per_seed"], float)
    lo1, hi1 = t_ci90(s1)
    s2 = np.asarray(c2["primary"]["survival"]["per_seed"], float)
    diff = np.asarray(ac["paired"]["per_seed"], float) if "per_seed" in ac["paired"] else None
    aux = (f"{ac['paired']['mean']:+.3f} [{ac['paired']['t90'][0]:+.3f}, {ac['paired']['t90'][1]:+.3f}]"
           if diff is None else
           f"{diff.mean():+.3f} [{t_ci90(diff)[0]:+.3f}, {t_ci90(diff)[1]:+.3f}]")
    def eff(r):
        x = r["correction_effect"]["0.45"]["survival_target"]
        return float(np.mean(np.asarray(x["corrected"]["per_seed"]) - np.asarray(x["historical"]["per_seed"])))
    l0 = np.asarray(c1["gates"]["l0"]["per_seed"], float)
    la = _a("artifacts", "l0_audit", "corrected_l3_h8_wm.json")["l0_summary"]
    pm = np.asarray(la["tost_over_independent_pairs"] and la["independent_pair_means"], float)
    n_eq = sum(bool(p["equivalent"]) for p in la["by_pair"] if not p["standard"])
    ws = _a("artifacts", "l0_audit", "d045_world_samples_l3_h8_heldout.json")["summary"]["survival"]
    draws = np.asarray(ws["per_draw"], float)
    rank = 1 + int(np.sum(draws > ws["registered"]))
    pc = _a("artifacts", "policy_controls", "corrected_l3_h8_wm.json")["aggregate"]
    pcm = {k: float(np.mean(b["per_seed"])) for k, b in pc.items()}
    scripted = [pcm[f"{g} scripted"] for g in ("survival", "predictor", "predictor_logged", "untrained")]
    bc = _a("artifacts", "budget_curve.json")["series"]
    b450 = {k: float(np.mean(next(p for p in bc[k] if p.get("updates", p.get("u")) == 450)["target"]["per_seed"]))
            for k in ("decoder on", "decoder off")}
    cd = "control_diagnostics/corrected_l3_h8_wm.json"
    cdm = lambda m: float(_ps(cd, f"aggregate/d=0.45 survival {m}/per_seed").mean())  # noqa: E731
    pers = _a("artifacts", "persistence", "corrected_l3_h8_wm.json")["aggregate"]
    phys = float(np.mean(pers["d=0.45 survival reset_hidden:prefix"]["window_per_seed"]))
    qd = _a("artifacts", "texture", "corrected_l3_h8_wm_qd.json")["cells"]
    qm = lambda arm, k: float(np.mean([c[k] for c in qd if c["arm"] == arm]))  # noqa: E731
    fu = _a("artifacts", "texture", "ar_sweep_c_followup", "summary.json")["contrast"][
        "follow-up: M(0.99, matched) - M(0, stored)"]
    l1 = _ps("expB2/corrected_L1-corrected.json", "arms/0.023/survival/pool_target/per_seed")
    l1h = _ps("expL1/organism_summary.json", "arms/0.023/survival/pool_target")
    l1x = _a("artifacts", "fold_rescore", "l1_heldout.json")["aggregate"]["d=0.0230 survival"]["target"]["explicit"]["mean"]
    org = _a("artifacts", "expL1", "organism_summary.json")
    b1 = _ps("balanced_confirmation/b1.json", "balanced/survival/per_seed")
    b2 = _ps("balanced_confirmation/b2.json", "balanced/survival/per_seed")
    bu = _ps("balanced_confirmation/b1.json", "balanced/untrained/per_seed")
    se = "aggregate/d=0.45 {}/resid_obs/per_seed"
    e1 = _ps("balanced_confirmation/sensory_echo_b1.json", se.format("survival"))
    e2 = _ps("balanced_confirmation/sensory_echo_b2.json", se.format("survival"))
    eu = _ps("balanced_confirmation/sensory_echo_b1.json", se.format("untrained"))
    tt = _ps("expB2/goal_stakes_T-touch.json", "arms/0.45/survival/pool_target/per_seed")
    h10 = _ps("expB2/corrected_H10GS1-corrected.json", "arms/0.45/survival/pool_target/per_seed")
    rows = {r["label"]: r for r in bpt.ci95_rows()}
    hist = _ps("expB2/sensory_echo_l3_h8.json", "aggregate/d=0.45 survival/target/per_seed")
    m = lambda label: rows[label]["mean"]  # noqa: E731
    seam = m(r"Graded seam, $\alpha = 0$")
    sk = _ps("expB2/skill_matched_l3_h8_nowm_u450.json", "arms/0.45/survival/pool_target/per_seed")
    l2 = _a("artifacts", "expA", "l2_inconfig_oracle.json")["noise_seed_draws"]
    one = [l2["ar1"]["oracle_auroc"][0], l2["regime"]["oracle_auroc"][0]]
    gt = {(r["run"], r["partition"].split(" ")[0]): r["l0"]["mean"] for r in _a("artifacts", "gate_table.json")["rows"]}
    ec = _a("artifacts", "expC", "emergence_pilot_summary.json")["estimand"]
    ecc = np.asarray(ec["contrast"], float)
    ind = [c["indiv_final_treat"]["mean"] for c in _a("artifacts", "expC", "emergence_pilot_per_individual_summary.json")["cells"]]
    pop = _a("artifacts", "population_readout", "corrected_l3_h8_wm.json")["panels"]["survival"]
    both = ("tldr", "readme")
    return [
        ("C1 survival and interval", both, f"{s1.mean():.3f} (t-based 90% CI [{lo1:.3f}, {hi1:.3f}]"),
        ("C1 baselines (TL;DR)", ("tldr",), f"against {p1.mean():.3f} for the predictor arm and {u1.mean():.3f} untrained"),
        ("C1 baselines (README)", ("readme",), f"the predictor arm {p1.mean():.3f} and the untrained arm {u1.mean():.3f}"),
        ("C1 paired margins", both, f"paired margins of {np.mean(s1 - p1):+.3f} and {np.mean(s1 - u1):+.3f}"),
        ("C1 L0 on the registered pair", both, f"({l0.mean():.3f}"),
        ("L0 deviation over eight pairs", both, f"absolute deviation from chance averages {np.mean(np.abs(pm - 0.5)):.3f}"),
        ("L0 TOST over pairs", both, f"(p = {la['tost_over_independent_pairs']['p']:.3f})"),
        ("L0 pairs accepted one by one", both, f"accepts on {WORDS[n_eq]} pairs individually"),
        ("drift-0.45 draws, range and rank", both,
         f"({draws.min():.3f} to {draws.max():.3f}; the registered draw's {ws['registered']:.3f} ranks {rank} of {len(draws) + 1})"),
        ("C2 survival", both, f"{_mean_iv(s2)}, not met"),
        ("auxiliary lead", both, f"decoder adds {aux}"),
        ("trainer correction effect", both, f"by {eff(c1):+.3f} and {eff(c2):+.3f}"),
        ("matched input, prediction trunks", both,
         f"reads the world at {pcm['predictor replay']:.3f}, and one trained on exactly the survival "
         f"agent's experience at {pcm['predictor_logged replay']:.3f}"),
        ("scripted policy range", both, f"every arm reads {min(scripted):.2f} to {max(scripted):.2f}"),
        ("budget at 450 (TL;DR)", ("tldr",),
         f"reaches {b450['decoder off']:.3f} at 450 updates; at that budget the decoder arm reads {b450['decoder on']:.3f}"),
        ("budget at 450 (README)", ("readme",), f"reaches {b450['decoder off']:.3f} and the arm with it {b450['decoder on']:.3f}"),
        ("behavior control", ("tldr",), f"behavior control {cdm('resid_trace'):.3f} remains"),
        ("best-fitting nonlinear control", both, f"leaves {cdm('resid_joint_mlp'):.3f}, under"),
        ("stream decoder", both, f"reads {cdm('seq_gru'):.3f}"),
        ("observation summary", both, f"alone read {float(_ps(cd, 'aggregate/d=0.45 survival obs_summary_only/per_seed').mean()):.3f}"),
        ("physical footprint", both, f"still reads {phys:.3f}"),
        ("drag through the original direction", both, f"direction ({qm('survival', 'transfer'):.3f})"),
        ("drag fresh probe", both, f"probe ({qm('survival', 'fresh'):.3f}"),
        ("drag untrained floor (TL;DR)", ("tldr",), f"untrained floor of {qm('untrained', 'fresh'):.3f}"),
        ("drag untrained floor (README)", ("readme",), f"reads the drag at {qm('untrained', 'fresh'):.3f}"),
        ("drag margins", both, f"{qm('survival', 'transfer') - qm('untrained', 'transfer'):+.3f} and "
                                f"{qm('survival', 'fresh') - qm('untrained', 'fresh'):+.3f}"),
        ("follow-up band miss", ("tldr",), f"misses the 0.05 equivalence band by {abs(fu['t90'][0]) - 0.05:.3f}"),
        ("corrected L1", both, f"{_mean_iv(l1)}"),
        ("historical L1 on the explicit partition (TL;DR)", ("tldr",), f"against {l1x:.3f} for the historical run"),
        ("historical L1 on the explicit partition (README)", ("readme",), f"reads {l1x:.3f}, and"),
        ("historical L1 as recorded (TL;DR)", ("tldr",), f"target is {l1h.mean():.3f}, with 0 of 10"),
        ("historical L1 as recorded (README)", ("readme",), f"{l1h.mean():.3f} as first recorded"),
        ("L1 detector", ("tldr",), f"detector {org['gate0']['delta']['chosen_auroc']:.3f}"),
        ("L1 detector (README)", ("readme",), f"detector AUROC {org['gate0']['delta']['chosen_auroc']:.3f}"),
        ("L3 detector", ("tldr",), f"privileged detector {l2['l3_h8_seed0']['oracle_auroc'][0]:.3f}"),
        ("L2 detector on one draw", ("tldr",), f"at {min(one):.2f} to {max(one):.2f} on one noise draw"),
        ("L2 detector over eight", ("tldr",), f"at {l2['ar1']['mean']:.2f} and {l2['regime']['mean']:.2f} averaged over eight"),
        ("B-v3", ("tldr",), f"reaches {m('B-v3 regime, pooled target'):.3f} against the 0.65 bar"),
        ("matched worlds (TL;DR)", ("tldr",), f"reads {b1.mean():.3f} and {b2.mean():.3f} on matched world pairs"),
        ("matched worlds (README)", ("readme",), f"Survival reads {_mean_iv(b1)} and a pellet-seeking objective with "
                                                 f"death off {_mean_iv(b2)}"),
        ("matched-world untrained arm", both, f"untrained arm reads {bu.mean():.3f}, outside the 0.1 floor"),
        ("sensory echo, trained arms", both, f"{e1.mean():.3f} and {e2.mean():.3f}"),
        ("sensory echo, untrained arm (TL;DR)", ("tldr",), f"untrained arm at {_mean_iv(eu)}"),
        ("sensory echo, untrained arm (README)", ("readme",),
         f"falls to {eu.mean():.3f}, with an interval [{t_ci90(eu)[0]:.3f}, {t_ci90(eu)[1]:.3f}]"),
        ("T-touch (TL;DR)", ("tldr",), f"pellet run's {tt.mean():.3f} was recorded"),
        ("T-touch (README)", ("readme",), f"{tt.mean():.3f} and was recorded as meeting every gate"),
        ("H10GS1 corrected", ("tldr",), f"reads {_mean_iv(h10)}"),
        ("historical headline (TL;DR)", ("tldr",), f"survival {_mean_iv(hist)} on GPU"),
        ("historical headline (README)", ("readme",),
         f"Survival reads {hist.mean():.3f} (n = 10, t-based 90% CI [{t_ci90(hist)[0]:.3f}, {t_ci90(hist)[1]:.3f}]"),
        ("historical CPU", ("tldr",), f"and {m('L3 hidden 8, survival-with-auxiliary arm on CPU (device control)'):.3f} on CPU"),
        ("historical behavior control", ("tldr",), f"{m('L3 hidden 8, behavior-controlled (four channels)'):.3f} after the behavior control"),
        ("historical sensory control", ("tldr",), f"{m('L3 hidden 8, sensory-controlled'):.3f} after the sensory control"),
        ("historical without auxiliary (TL;DR)", ("tldr",), f"{m('L3 hidden 8, survival without auxiliary'):.3f} without the auxiliary"),
        ("historical without auxiliary (README)", ("readme",), f"signal drops to {m('L3 hidden 8, survival without auxiliary'):.3f}"),
        ("historical 450 updates", ("tldr",), f"and {sk.mean():.3f} without it at 450 updates"),
        ("second capacity residual", ("tldr",), f"({m('L3 hidden 7, behavior-controlled'):.3f})"),
        ("second fingerprint (TL;DR)", ("tldr",), f"under the bar ({m('L3 hidden 10, $G$ seed 1, survival'):.3f}; 10.9)"),
        ("second fingerprint (README)", ("readme",), f"({m('L3 hidden 10, $G$ seed 1, survival'):.3f} against"),
        ("historical L0, two partitions", ("tldr",),
         f"L0 reads {gt[('L3-H8-N10', 'legacy')]:.3f} under the original fold partition and "
         f"{gt[('L3-H8-N10', 'explicit')]:.3f} under the explicit one"),
        ("graded seam", ("tldr",), f"({seam:.3f} at the authentic end"),
        ("transfer", ("tldr",), f"({m('Transfer, hidden 8 to held-out hidden 7'):.3f};"),
        ("cross-recipe", ("tldr",), f"({m('Cross-recipe transfer (RFF)'):.3f}, t-based lower bound "
                                    f"{rows['Cross-recipe transfer (RFF)']['t90'][0]:.3f}"),
        ("reverse transfer", ("tldr",), f"misses the bar ({m('Transfer, hidden 7 to held-out hidden 8'):.3f})"),
        ("common garden", ("tldr",), f"({m('Common garden, hidden-8 tail'):.3f} forward, "
                                     f"{m('Common garden, hidden-7 tail'):.3f} reverse"),
        ("Experiment C contrast", ("tldr",),
         f"(contrast {bpt._sg(ecc.mean())} {bpt._iv(t_ci90(ecc), signed=True)}"),
        ("Experiment C per-individual range", ("tldr",), f"final treatment means {min(ind):.3f} to {max(ind):.3f}"),
        ("pooled population readout", ("tldr",),
         f"reads {pop['pooled_probe']:.3f} where independent known-decodable individuals average {pop['mean']:.3f}"),
    ]


@pytest.fixture(scope="module")
def surfaces():
    return _surfaces()


def test_every_restated_number_is_the_artifact_value(surfaces):
    missing = [(name, where, text) for name, where_all, text in needles()
               for where in where_all if text not in surfaces[where]]
    assert not missing, missing


def test_a_moved_restatement_is_caught(surfaces):
    """Mutation check: the needles read the artifact, so a TL;DR or README edit is caught."""
    for name, where, text in needles():
        if name == "drift-0.45 draws, range and rank":
            moved = {w: surfaces[w].replace(text, text.replace("0.728", "0.718")) for w in where}
            assert all(text not in moved[w] for w in where)
            return
    pytest.fail("the range-and-rank needle is missing")
