"""Round-trip tests for scripts/promote_h2_batteries.py on a tiny synthetic bundle set.

The promotion script's contract is copy-not-recompute: every per-seed list, mean,
n_ge_065 count, gate-0 row, and integrity receipt in the committed artifact must be
the bundle's value verbatim. These tests build a 3-seed stand-in for each of the
seven source bundles under tmp_path, run the script end to end with every path
overridden, and assert the artifacts reproduce the inputs (plus the few derived
fields: seed-level CIs, L0 equivalence tests, and canonical source labels)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_h2_batteries as ph  # noqa: E402

ARMS = ("survival", "untrained", "predictor")
SEEDS = (0, 1, 2)


def _knockout_bundle(d: Path, hidden: int, published: float, utf16_log: bool) -> None:
    d.mkdir(parents=True)
    agg = {"quick": False, "n_eps": 110, "steps": 24, "published_target_check": published,
           "gn_dropped_at_gate0": False, "ladder_oracles": {"h16": 0.7875, "h32": 0.6556, "h64": 0.6028},
           "gn_sigma_v": 0.01, "gn_rule_pass": False, "gn_rule_margin": -0.11,
           "gn_verdict": "H2_SUPPORTED", "ladder_promotion_eligible": False}
    cells = []
    for ch_i, ch in enumerate(ph.KNOCKOUT_CHANNELS):
        for a_i, arm in enumerate(ARMS):
            full = [0.5 + 0.01 * ch_i + 0.02 * a_i + 0.00123 * s for s in SEEDS]
            agg[f"{ch}_{arm}_per_seed"] = [round(v, 4) for v in full]
            agg[f"{ch}_{arm}_mean"] = round(float(np.mean(full)), 4)
            agg[f"{ch}_{arm}_n_ge_065"] = int(sum(v >= 0.65 for v in full))
            for s, v in zip(SEEDS, full):
                row = next((c for c in cells if c["seed"] == s and c["arm"] == arm), None)
                if row is None:
                    row = {"drift": 0.45, "seed": s, "arm": arm}
                    cells.append(row)
                row.update({f"transfer_{ch}_target": v, f"transfer_{ch}_n_auth": 110,
                            f"transfer_{ch}_n_surr": 110, f"transfer_{ch}_deaths_auth": 0,
                            f"transfer_{ch}_deaths_surr": 0})
    (d / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    (d / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    log = ("gn sigma_v=0.01\n" + "".join(f"  integrity ok: agent_d0.45_s{s}_{arm}.pt\n"
                                        for s in SEEDS for arm in ARMS)
           + f"integrity gate PASSED: survival mean {published:.3f} == {published:.3f} "
             "(determinism check #5)\n")
    if utf16_log:  # PowerShell `> file` redirects write UTF-16 LE with a BOM
        (d / "ablations.log").write_bytes(b"\xff\xfe" + log.encode("utf-16-le"))
    else:
        (d / "ablations.log").write_text(log, encoding="utf-8")


def _gate0_files(d: Path) -> None:
    gn_rows = [{"family": "gn", "sigma_v": sv, "oracle_auroc": oa, "in_band": 0.85 <= oa <= 0.95,
                "mech_leak_pass": True, "oracle_reward_leak": 0.48, "floor": 0.45,
                "floor_per_seed": [0.44, 0.45, 0.46], "floor_ok": True,
                "pool_reward_leak_per_seed": [0.5, 0.5, 0.5],
                "passes_gate0": 0.85 <= oa <= 0.95}
               for sv, oa in ((0.005, 0.65), (0.01, 0.865), (0.02, 0.99))]
    (d / "gate0_gn.json").write_text(json.dumps(
        {"world": "P", "sigma_meas": 0.02, "band": [0.85, 0.95], "floor_tol": 0.1, "drift": 0.45,
         "floor_seeds": [0, 1, 2], "rows": gn_rows, "regression_ok": None,
         "selected_hidden": 0.01, "selected": {"family": "gn", "sigma_v": 0.01}}), encoding="utf-8")
    lad_rows = [{"hidden": h, "oracle_auroc": oa, "in_band": False, "mech_leak_pass": True,
                 "oracle_reward_leak": 0.5, "floor": 0.47, "floor_per_seed": [0.46, 0.47, 0.48],
                 "floor_ok": True, "pool_reward_leak_per_seed": [0.5, 0.5, 0.5], "passes_gate0": False}
                for h, oa in ((16, 0.7875), (32, 0.6556), (64, 0.6028))]
    (d / "gate0_ladder.json").write_text(json.dumps(
        {"world": "P", "sigma_meas": 0.02, "band": [0.85, 0.95], "floor_tol": 0.1, "drift": 0.45,
         "floor_seeds": [0, 1, 2], "rows": lad_rows, "regression_ok": None, "selected_hidden": None}),
        encoding="utf-8")


def _obs_bundle(d: Path, extra_cfg: dict, drift: float) -> None:
    d.mkdir(parents=True)
    masks = ["none", "vision", "intero", "all"]
    dims = {"none": 0, "vision": 120, "intero": 14, "all": 146}
    agg = {"masks": masks, "obs_dim": 146, **extra_cfg}
    cells, log = [], []
    for m_i, mask in enumerate(masks):
        log.append(f"mask={mask}: {dims[mask]}/146 dimensions zeroed\n")
        for a_i, arm in enumerate(ARMS):
            full = ([0.5] * 3 if mask == "all"
                    else [0.6 + 0.03 * m_i - 0.04 * a_i + 0.00111 * s for s in SEEDS])
            agg[f"{mask}_{arm}_per_seed"] = [round(v, 4) for v in full]
            agg[f"{mask}_{arm}_mean"] = round(float(np.mean(full)), 4)
            agg[f"{mask}_{arm}_n_ge_065"] = int(sum(v >= 0.65 for v in full))
            for s, v in zip(SEEDS, full):
                cells.append({"mask": mask, "drift": drift, "seed": s, "arm": arm, "target": v,
                              "speed": 0.9, "pool_reward_leak": 0.5, "pool_leak_clean": True,
                              "deaths_auth": 0, "deaths_surr": 0})
    (d / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    (d / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    (d / "ablations.log").write_text("".join(log), encoding="utf-8")


def _l1_heldout(d: Path) -> dict:
    (d / "cells").mkdir(parents=True)
    results = {}
    for dk, base in (("0.0", 0.50), ("0.023", 0.53)):
        results[dk] = {}
        for a_i, arm in enumerate(ARMS):
            tgt = [base - 0.02 * a_i + 0.004 * s for s in SEEDS]
            results[dk][arm] = {
                "pool_target": tgt, "pool_target_lo": [v - 0.07 for v in tgt],
                "pool_target_hi": [v + 0.07 for v in tgt], "pool_speed": [0.9, 0.91, 0.92],
                "pool_shuffled": [0.5, 0.5, 0.5], "pool_anchor_energy": [0.8, 0.8, 0.8],
                "pool_anchor_food": [0.8, 0.8, 0.8], "pool_ceiling_drag": [float("nan")] * 3,
                "pool_reward_leak": [0.5, 0.5, 0.5], "pool_leak_clean": [True] * 3,
                "pool_deaths_auth": [0, 0, 0], "pool_deaths_surr": [0, 0, 0],
                "mp_target": [0.6, 0.6, 0.6], "mp_leak_clean": [True] * 3, "xeval_return": []}
    (d / "expB2_results.json").write_text(json.dumps(results), encoding="utf-8")
    for dk in results:
        for s in SEEDS:
            cell = {"fingerprint": "abc123", "git_commit": "8a71593",
                    "cell": {"drift": float(dk), "seed": s,
                             "eng": {"trained_return": -0.3, "random_return": -1.0,
                                     "scripted_return": -1.4, "trained_len": 72.0,
                                     "random_len": 69.0, "scripted_len": 63.0,
                                     "better_return": True, "not_worse_life": True, "engaged": True},
                             "xeval": {}, "agents": {}}}
            (d / "cells" / f"cell_d{float(dk):.4f}_s{s}.json").write_text(json.dumps(cell), encoding="utf-8")
    (d / "run.log").write_text("Experiment B-v2 full run (device=cpu)\n  survival metabolism\n"
                               "  drift_mode=l1: delta=0.02300\n  more\n", encoding="utf-8")
    return results


def _l1_ablations(d: Path) -> tuple[dict, dict]:
    d.mkdir(parents=True)
    deltas = [0.0, 0.00575, 0.0115, 0.01725, 0.023]
    a1 = {"deltas": deltas, "arms": list(ARMS), "quick": False, "n_eps": 110, "steps": 24,
          "headline_delta": 0.023, "sensor_sigma": 0.01,
          "integrity": {"checked": True, "pools_bit_match": True, "survival_mean_headline": 0.533}}
    curve = []
    for arm in ARMS:
        for dl in deltas:
            ps = [0.52 + 0.001 * s for s in SEEDS]
            a1[f"{arm}_d{dl:.4f}_per_seed"] = ps
            a1[f"{arm}_d{dl:.4f}_mean"] = round(float(np.mean(ps)), 4)
            a1[f"{arm}_d{dl:.4f}_tci90"] = [0.5, 0.54]
            a1[f"{arm}_d{dl:.4f}_n_ge_065"] = 0
            if arm == "survival":
                curve.append(a1[f"{arm}_d{dl:.4f}_mean"])
    a1["survival_curve"] = curve
    a1["survival_monotonicity_rho"] = 0.8
    a1["l0_anchor"] = {"mean": 0.52, "hdi": [0.5, 0.55], "p_in_rope": 0.98, "accept_equiv": True}
    noise = {"noise_sigma": 0.01, "quick": False, "n_eps": 110, "steps": 24, "gn_dropped_at_gate0": False,
             "gn_rule_pass": False, "gn_rule_margin": 0.0166, "gn_verdict": "H2_SUPPORTED"}
    for arm in ARMS:
        noise[f"{arm}_per_seed"] = [0.51, 0.52, 0.53]
        noise[f"{arm}_mean"] = 0.52
        noise[f"{arm}_n_ge_065"] = 0
        noise[f"{arm}_tci90"] = [0.5, 0.54]
    (d / "a1_aggregate.json").write_text(json.dumps(a1), encoding="utf-8")
    (d / "noise_aggregate.json").write_text(json.dumps(noise), encoding="utf-8")
    (d / "ablations.log").write_text("integrity gate PASSED: headline survival mean 0.533\n",
                                     encoding="utf-8")
    return a1, noise


def _calibs(fr: Path) -> None:
    row = lambda dl, oa: {"delta": dl, "sigma": 0.01, "oracle_auroc": oa,  # noqa: E731
                          "leakage": {"reward": 0.5, "length": 0.5, "metadata": 0.5},
                          "leakage_pass": True}
    (fr / "l1_calib.json").write_text(json.dumps(
        {"sensor_sigma": 0.01, "deltas_sweep": [row(0.06, 1.0), row(0.025, 0.97)],
         "fine_sweep": [row(0.022, 0.82), row(0.023, 0.872875), row(0.024, 0.94)],
         "chosen_delta": 0.023, "chosen_auroc": 0.872875}), encoding="utf-8")
    nrow = lambda ns, oa: {"headline_delta": 0.023, "sensor_sigma": 0.01,  # noqa: E731
                           "noise_sigma": ns, "oracle_auroc": oa,
                           "leakage": {"reward": 0.5, "length": 0.5, "metadata": 0.5},
                           "leakage_pass": True}
    (fr / "l1_noise_calib.json").write_text(json.dumps(
        {"headline_delta": 0.023, "sensor_sigma": 0.01,
         "noise_sigmas_sweep": [nrow(0.005, 0.64), nrow(0.01, 0.87275), nrow(0.015, 0.96)],
         "chosen_sigma": 0.01}), encoding="utf-8")


@pytest.fixture
def promoted(tmp_path: Path):
    fr = tmp_path / "fullruns"
    fr.mkdir()
    _knockout_bundle(fr / "l3_h2_ablations", 8, 0.752, utf16_log=True)
    _gate0_files(fr / "l3_h2_ablations")
    _knockout_bundle(fr / "l3_h7_h2_ablations", 7, 0.737, utf16_log=False)
    _obs_bundle(fr / "l3_h8_obs_localization", {"hidden": 8}, 0.45)
    _obs_bundle(fr / "l3_h7_obs_localization", {"hidden": 7}, 0.45)
    _obs_bundle(fr / "l1_obs_localization", {"drift_mode": "l1", "l1_delta": 0.023, "sensor_sigma": 0.01}, 0.023)
    results = _l1_heldout(fr / "l1_heldout")
    a1, noise = _l1_ablations(fr / "l1_h2_ablations")
    _calibs(fr)
    out = tmp_path / "artifacts"
    rc = ph.main(["--root", str(tmp_path), "--fullruns", str(fr), "--out-root", str(out)])
    assert rc == 0
    load = lambda rel: json.loads((out / rel).read_text(encoding="utf-8"))  # noqa: E731
    return {"fr": fr, "out": out, "results": results, "a1": a1, "noise": noise, "load": load}


def test_writes_all_seven_artifacts(promoted):
    out = promoted["out"]
    for rel in ("expH2/texture_knockout_h8.json", "expH2/texture_knockout_h7.json",
                "expH2/obs_localization_h8.json", "expH2/obs_localization_h7.json",
                "expL1/organism_summary.json", "expL1/h2_ablations.json",
                "expL1/obs_localization.json"):
        assert (out / rel).exists(), rel
        doc = promoted["load"](rel)
        assert doc["generated_by"] == "scripts/promote_h2_batteries.py"
        assert doc["source_run"].startswith("fullruns/"), doc["source_run"]
        assert doc["git_commit_at_promotion"]
        assert doc["bar"] == 0.65


def test_texture_knockout_round_trip(promoted):
    fr, load = promoted["fr"], promoted["load"]
    agg = json.loads((fr / "l3_h2_ablations" / "aggregate.json").read_text())
    cells = json.loads((fr / "l3_h2_ablations" / "cells.json").read_text())
    doc = load("expH2/texture_knockout_h8.json")
    assert doc["hidden"] == 8 and doc["source_run"] == "fullruns/l3_h2_ablations"
    for ch in ph.KNOCKOUT_CHANNELS:
        for arm in ARMS:
            e = doc["channels"][ch][arm]
            assert e["per_seed"] == agg[f"{ch}_{arm}_per_seed"]
            assert e["mean"] == agg[f"{ch}_{arm}_mean"]
            assert e["n_ge_065"] == agg[f"{ch}_{arm}_n_ge_065"]
            full = sorted((c["seed"], c[f"transfer_{ch}_target"]) for c in cells if c["arm"] == arm)
            assert e["per_seed_full"] == [v for _, v in full]
            assert e["n_auth"] == [110.0] * 3 and e["deaths_surr"] == [0.0] * 3
    assert doc["ladder_oracles"] == agg["ladder_oracles"]
    assert doc["gn_rule"] == {"pass": False, "margin": -0.11, "verdict": "H2_SUPPORTED"}
    assert doc["ladder_promotion_eligible"] is False
    assert doc["config"]["n_seeds"] == 3 and doc["config"]["gn_sigma_v"] == 0.01
    assert doc["cells"] == cells
    # UTF-16 log decoded: integrity receipt + 9 reloaded-agent lines
    assert doc["integrity"]["gate_passed"] is True
    assert doc["integrity"]["survival_mean_reproduced"] == 0.752
    assert doc["integrity"]["published_target"] == 0.752
    assert doc["integrity"]["determinism_check"] == 5
    assert doc["integrity"]["n_integrity_ok_lines"] == 9
    # gate-0 rows copied verbatim; the selected row is the single in-band one
    g0 = doc["gate0"]["gn"]
    assert g0["selected"] == {"family": "gn", "sigma_v": 0.01}
    assert g0["selected_row"]["oracle_auroc"] == 0.865 and g0["selected_row"]["passes_gate0"]
    assert len(g0["rows"]) == 3
    assert [r["hidden"] for r in doc["gate0"]["ladder"]["rows"]] == [16, 32, 64]
    assert doc["gate0_shared_from_hidden8"] is False


def test_hidden7_shares_gate0_and_has_own_integrity(promoted):
    load = promoted["load"]
    h7, h8 = load("expH2/texture_knockout_h7.json"), load("expH2/texture_knockout_h8.json")
    assert h7["hidden"] == 7 and h7["source_run"] == "fullruns/l3_h7_h2_ablations"
    assert h7["gate0_shared_from_hidden8"] is True
    assert h7["gate0"] == h8["gate0"]
    assert h7["source_files"]["gate0_gn"] == "fullruns/l3_h2_ablations/gate0_gn.json"
    assert h7["integrity"]["survival_mean_reproduced"] == 0.737
    assert h7["integrity"]["n_integrity_ok_lines"] == 9


def test_obs_localization_round_trip(promoted):
    fr, load = promoted["fr"], promoted["load"]
    for rel, src, hidden in (("expH2/obs_localization_h8.json", "l3_h8_obs_localization", 8),
                             ("expH2/obs_localization_h7.json", "l3_h7_obs_localization", 7)):
        agg = json.loads((fr / src / "aggregate.json").read_text())
        cells = json.loads((fr / src / "cells.json").read_text())
        doc = load(rel)
        assert doc["source_run"] == f"fullruns/{src}"
        assert doc["config"]["hidden"] == hidden and doc["config"]["obs_dim"] == 146
        assert doc["config"]["masks"] == ["none", "vision", "intero", "all"]
        for mask, zd in (("none", 0), ("vision", 120), ("intero", 14), ("all", 146)):
            assert doc["masks"][mask]["zeroed_dims"] == {"zeroed": zd, "total": 146}
            for arm in ARMS:
                e = doc["masks"][mask][arm]
                assert e["per_seed"] == agg[f"{mask}_{arm}_per_seed"]
                assert e["mean"] == agg[f"{mask}_{arm}_mean"]
                assert e["n_ge_065"] == agg[f"{mask}_{arm}_n_ge_065"]
                full = [c["target"] for c in sorted(cells, key=lambda c: c["seed"])
                        if c["mask"] == mask and c["arm"] == arm]
                assert e["per_seed_full"] == full
                assert e["pool_leak_clean"] == [True] * 3
        assert doc["cells"] == cells
    l1 = load("expL1/obs_localization.json")
    assert l1["config"]["drift_mode"] == "l1" and l1["config"]["l1_delta"] == 0.023
    assert "hidden" not in l1["config"]


def test_l1_organism_summary_round_trip(promoted):
    results, load = promoted["results"], promoted["load"]
    doc = load("expL1/organism_summary.json")
    assert doc["source_run"] == "fullruns/l1_heldout"
    assert doc["fingerprint"] == "abc123" and doc["git_commit_at_run"] == ["8a71593"]
    assert doc["drifts"] == ["0.0", "0.023"] and doc["headline_drift"] == "0.023"
    assert doc["run_log_header"][2].startswith("  drift_mode=l1")
    for dk in results:
        for arm in ARMS:
            src, row = results[dk][arm], doc["arms"][dk][arm]
            for k in ph.ORGANISM_POOL_KEYS:
                if k == "pool_ceiling_drag":
                    assert row[k] == [None] * 3  # NaN is not JSON; promoted as null
                else:
                    assert row[k] == src[k], (dk, arm, k)
            assert "xeval_return" not in row
            agg = row["aggregate"]["pool_target"]
            assert agg["mean"] == pytest.approx(float(np.mean(src["pool_target"])))
            assert agg["n_seeds"] == 3 and agg["n_ge_065"] == 0
            assert agg["boot90"][0] <= agg["mean"] <= agg["boot90"][1]
            assert agg["tci90"][0] <= agg["mean"] <= agg["tci90"][1]
    assert len(doc["engagement"]) == 6 and all(e["engaged"] for e in doc["engagement"])
    assert [e["seed"] for e in doc["engagement"]] == [0, 1, 2, 0, 1, 2]
    g = doc["gates"]
    assert g["engagement"] == {"n_cells": 6, "n_engaged": 6, "pass": True}
    assert g["l0_control"]["drift"] == "0.0"
    assert g["l0_control"]["survival_pool_target_per_seed"] == results["0.0"]["survival"]["pool_target"]
    assert g["l0_control"]["mean"] == pytest.approx(0.504)
    assert set(g["l0_control"]["tost"]) == {"margin", "p_value", "equivalent"}
    assert set(g["l0_control"]["rope"]) == {"rope", "hdi", "p_in_rope", "accept"}
    assert g["speed_positive_control"]["pass"] is True
    assert g["speed_positive_control"]["min_speed"]["d=0.023 survival"] == 0.9
    assert g["leakage"] == {"pool_leak_clean_all": True, "mp_leak_clean_all": True}
    assert g["survivorship"] == {"deaths_total": 0}
    g0 = doc["gate0"]
    assert g0["delta"]["chosen_delta"] == 0.023 and g0["delta"]["chosen_auroc"] == 0.872875
    assert g0["delta"]["chosen_row"]["delta"] == 0.023 and g0["delta"]["chosen_row"]["leakage_pass"]
    assert g0["noise"]["chosen_sigma"] == 0.01
    assert g0["noise"]["chosen_row"]["oracle_auroc"] == 0.87275


def test_l1_h2_ablations_round_trip(promoted):
    a1, noise, load = promoted["a1"], promoted["noise"], promoted["load"]
    doc = load("expL1/h2_ablations.json")
    assert doc["source_run"] == "fullruns/l1_h2_ablations"
    assert doc["a1_graded_seam"] == a1
    assert doc["a2_noise_knockout"] == noise
    assert doc["integrity"]["gate_passed"] is True
    assert doc["integrity"]["survival_mean_reproduced"] == 0.533
    assert "published_target" not in doc["integrity"]  # L1 log line carries no `== x` clause
    assert doc["gate0_noise"]["chosen_row"]["noise_sigma"] == 0.01


def test_helpers_read_text_any_and_rel_source(tmp_path: Path):
    p8 = tmp_path / "a.log"
    p8.write_bytes(b"plain\n")  # bytes: no platform newline translation
    p16 = tmp_path / "b.log"
    p16.write_bytes(b"\xff\xfe" + "wide\n".encode("utf-16-le"))
    assert ph.read_text_any(str(p8)) == "plain\n"
    assert ph.read_text_any(str(p16)) == "wide\n"
    assert ph.read_text_any(str(tmp_path / "missing.log")) is None
    fr = tmp_path / "fullruns"
    assert ph.rel_source(str(fr / "l3_h2_ablations"), str(fr)) == "fullruns/l3_h2_ablations"
    assert ph.rel_source(str(fr / "l1_calib.json"), str(fr)) == "fullruns/l1_calib.json"
    outside = str(tmp_path / "elsewhere" / "bundle")
    assert ph.rel_source(outside, str(fr)) == outside.replace("\\", "/")
    assert ph.parse_integrity(None) == {"log_found": False}
    assert ph.parse_integrity("no gate here")["gate_passed"] is None
    assert ph.parse_mask_dims("mask=vision: 120/146 dimensions zeroed\n") == {
        "vision": {"zeroed": 120, "total": 146}}


def test_mixed_fingerprints_refused(tmp_path: Path):
    fr = tmp_path / "fullruns"
    fr.mkdir()
    _l1_heldout(fr / "l1_heldout")
    bad = fr / "l1_heldout" / "cells" / "cell_d0.0000_s1.json"
    doc = json.loads(bad.read_text())
    doc["fingerprint"] = "zzz"
    bad.write_text(json.dumps(doc))
    _calibs(fr)
    with pytest.raises(ValueError, match="mixed fingerprints"):
        ph.l1_organism_summary(
            json.loads((fr / "l1_heldout" / "expB2_results.json").read_text()),
            [json.loads(p.read_text()) for p in sorted((fr / "l1_heldout" / "cells").glob("*.json"))],
            None, json.loads((fr / "l1_calib.json").read_text()),
            json.loads((fr / "l1_noise_calib.json").read_text()))
