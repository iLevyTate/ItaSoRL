"""Check quoted numbers and watched wording against the committed artifacts.

What it does: loads the committed JSONs under artifacts/ and checks that the numbers
quoted in README.md, docs/FINDINGS.md (sections 9-10, 14 to 16 and the methods notes),
docs/PAPER_OUTLINE.md, CITATION.cff, the site, and the B-v3 gate entry of
docs/PREREGISTRATION_Bv3.md section 12 equal values recomputed from the per-seed cells
they come from; that generated pages (index.html, the results manifest, the gate table,
the contrast intervals, the manuscript tables) are current; and that retired wording has
not returned to the pages it watches. Fails loudly (non-zero exit) on any mismatch
beyond rounding.

What it does not do (revision 2026-10): it reruns no experiment, it cannot detect an error
in the committed per-seed values themselves (the GAE bootstrap defect passed it for
months), and it does not read the LaTeX manuscript, which lives outside git
(`scripts/build_paper_tables.py --manuscript <dir>` checks that).

Two interval types appear in the docs, both recomputed here:
  boot: seed-level percentile bootstrap of the across-seed mean
        (itasorl.stats.mean_ci, level 0.90, seed 0)
  t:    Student-t 90% CI of the across-seed mean (decision-relevant at the
        0.65 bar per PREREGISTRATION_L3.md and FINDINGS methods note 5)

Usage:
    python scripts/audit_stats_recheck.py
"""

from __future__ import annotations

import json
import re
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import equivalence_test, mean_ci, rope_test  # noqa: E402

ARTROOT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
ART = os.path.join(ARTROOT, "expB2")
TOL = 5e-4  # docs quote 3 decimals; allow rounding slack
BAR = 0.65

failures: list[str] = []
n_checks = 0


def t_ci(vals: list[float], level: float = 0.90) -> tuple[float, float]:
    from scipy.stats import t as student_t
    x = np.asarray(vals, dtype=float)
    n = x.size
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    q = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, n - 1))
    return (m - q * se, m + q * se)


def check(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    global n_checks
    n_checks += 1
    ok = math.isfinite(actual) and abs(actual - expected) <= tol
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}: doc {expected:.3f} vs recomputed {actual:.4f}")
    if not ok:
        failures.append(label)


def check_int(label: str, actual: int, expected: int) -> None:
    global n_checks
    n_checks += 1
    ok = actual == expected
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}: doc {expected} vs recomputed {actual}")
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool) -> None:
    global n_checks
    n_checks += 1
    print(f"  [{'ok  ' if actual else 'FAIL'}] {label}")
    if not actual:
        failures.append(label)


def load(name: str) -> dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def seed_vals(doc: dict, drift: str, agent: str, metric: str) -> list[float]:
    rows = [r for r in doc["cells"]
            if r["drift"] == drift and r["agent"] == agent and metric in r]
    rows.sort(key=lambda r: r["seed"])
    return [float(r[metric]) for r in rows]


def verify_aggregate_consistency(name: str, doc: dict) -> None:
    """The stored `aggregate` block must be exactly reproducible from `cells`."""
    global n_checks
    bad = 0
    for arm, metrics in doc["aggregate"].items():
        d, agent = arm.split(" ", 1)
        drift = d.split("=")[1]
        for metric, stored in metrics.items():
            vals = [v for v in seed_vals(doc, drift, agent, metric)
                    if np.isfinite(v)]
            mean, lo, hi = mean_ci(vals)
            above = int(sum(v >= doc.get("bar", BAR) for v in vals))
            if (abs(mean - stored["mean"]) > 1e-9 or abs(lo - stored["lo"]) > 1e-9
                    or abs(hi - stored["hi"]) > 1e-9
                    or above != stored["n_above_bar"]):
                bad += 1
                failures.append(f"{name} aggregate {arm}/{metric}")
    n_checks += 1
    print(f"  [{'ok  ' if bad == 0 else 'FAIL'}] {name}: stored aggregate == recompute"
          f" from cells ({len(doc['cells'])} cells)")


def main() -> int:
    n10 = load("behavior_audit_l3_n10.json")
    h8 = load("behavior_audit_l3_h8_traces.json")
    h7 = load("behavior_audit_l3_h7_traces.json")
    covar = load("behavior_audit_l3_covar_n10.json")
    held = load("heldout_l3_h8_summary.json")
    rev = load("heldout_l3_h7_reverse_summary.json")
    cgf = load("heldout_l3_h8_cg_rescore.json")
    cgr = load("heldout_l3_h7_reverse_cg_rescore.json")
    b2 = load("expB2_results.json")
    b2c = load("expB2_results_confirmatory_n3.json")

    print("== internal consistency: stored aggregates reproduce from per-seed cells ==")
    for name, doc in [("n10", n10), ("h8_traces", h8), ("h7_traces", h7),
                      ("covar_n10", covar),
                      ("heldout_summary", held), ("h7_reverse_summary", rev)]:
        verify_aggregate_consistency(name, doc)

    print("== FINDINGS 10.2 / README / PAPER_OUTLINE: L3 headline (hidden=8) ==")
    surv = seed_vals(h8, "0.45", "survival", "target")
    check("survival target mean (0.752)", float(np.mean(surv)), 0.752)
    lo, hi = mean_ci(surv)[1:]
    check("survival target boot lo (0.704)", lo, 0.704)
    check("survival target boot hi (0.797)", hi, 0.797)
    tlo, thi = t_ci(surv)
    check("survival target t lo (0.698)", tlo, 0.698)
    check("survival target t hi (0.807)", thi, 0.807)
    check_int("survival seeds >= 0.65 (8)", sum(v >= BAR for v in surv), 8)
    doc_seeds = [0.853, 0.636, 0.841, 0.823, 0.830, 0.573, 0.705, 0.782, 0.759, 0.723]
    check_true("per-seed list in FINDINGS 10.2 matches cells",
               all(abs(a - b) <= TOL for a, b in zip(surv, doc_seeds)))
    pred = seed_vals(h8, "0.45", "predictor", "target")
    check("predictor target mean (0.573)", float(np.mean(pred)), 0.573)
    check("predictor boot lo (0.546)", mean_ci(pred)[1], 0.546)
    check("predictor boot hi (0.599)", mean_ci(pred)[2], 0.599)
    untr = seed_vals(h8, "0.45", "untrained", "target")
    check("untrained target mean (0.488)", float(np.mean(untr)), 0.488)
    check("untrained boot lo (0.461)", mean_ci(untr)[1], 0.461)
    check("untrained boot hi (0.514)", mean_ci(untr)[2], 0.514)
    check_true("dissociation: survival - predictor >= 0.05 (h8)",
               float(np.mean(surv)) - float(np.mean(pred)) >= 0.05)

    print("== FINDINGS 10.2/10.3: L0 control and leakage ==")
    l0 = seed_vals(h8, "0.00", "survival", "target")
    check("L0 survival target (0.517)", float(np.mean(l0)), 0.517)
    check_true("L0 TOST equivalent to chance (margin 0.05)",
               equivalence_test(l0, margin=0.05).equivalent)
    check_true("L0 ROPE accepts equivalence", rope_test(l0).accept)
    leak = seed_vals(held, "0.45", "survival", "pool_reward_leak")
    check("reward-leak mean (0.541)", float(np.mean(leak)), 0.541)
    check("reward-leak min (0.517)", min(leak), 0.517)
    check("reward-leak max (0.559)", max(leak), 0.559)
    check_true("reward-leak clean 10/10",
               all(seed_vals(held, "0.45", "survival", "pool_leak_clean")))
    deaths = (seed_vals(held, "0.45", "survival", "deaths_auth")
              + seed_vals(held, "0.45", "survival", "deaths_surr"))
    check_true("0 early deaths in survival pools",
               all(d == 0 for d in deaths))
    check_true("110/110 episodes per pool",
               all(n == 110 for n in seed_vals(held, "0.45", "survival", "n_auth")
                   + seed_vals(held, "0.45", "survival", "n_surr")))

    print("== FINDINGS 10.4: behavior mediation (hidden=8, four-channel control; "
          "superseded by 10.4.1) ==")
    check("behavior_only linear (0.689)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "behavior_only"))), 0.689)
    check("behavior_only nonlinear (0.705)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "behavior_only_nonlinear"))), 0.705)
    bt = seed_vals(h8, "0.45", "survival", "behavior_trace_only")
    check("behavior trace (0.803)", float(np.mean(bt)), 0.803)
    check("behavior trace boot lo (0.763)", mean_ci(bt)[1], 0.763)
    check("behavior trace boot hi (0.840)", mean_ci(bt)[2], 0.840)
    check("resid_epmean (0.676)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "resid_epmean"))), 0.676)
    check("resid_epmean_quad (0.659)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "resid_epmean_quad"))), 0.659)
    rt = seed_vals(h8, "0.45", "survival", "resid_trace")
    check("resid_trace (0.726)", float(np.mean(rt)), 0.726)
    check("resid_trace boot lo (0.685)", mean_ci(rt)[1], 0.685)
    check("resid_trace boot hi (0.765)", mean_ci(rt)[2], 0.765)
    check_int("resid_trace seeds >= 0.65 (9)", sum(v >= BAR for v in rt), 9)
    tlo, thi = t_ci(rt)
    check("resid_trace t lo (0.679)", tlo, 0.679)
    check("resid_trace t hi (0.772)", thi, 0.772)
    check_true("resid_trace t-CI excludes bar", tlo > BAR)
    rtq = seed_vals(h8, "0.45", "survival", "resid_trace_quad")
    check("resid_trace_quad (0.721)", float(np.mean(rtq)), 0.721)
    check("resid_trace_quad boot lo (0.678)", mean_ci(rtq)[1], 0.678)
    check("resid_trace_quad boot hi (0.760)", mean_ci(rtq)[2], 0.760)
    check("untrained resid_trace (0.498)",
          float(np.mean(seed_vals(h8, "0.45", "untrained", "resid_trace"))), 0.498)
    check("untrained behavior trace (0.645)",
          float(np.mean(seed_vals(h8, "0.45", "untrained", "behavior_trace_only"))), 0.645)
    check("predictor resid_trace (0.574)",
          float(np.mean(seed_vals(h8, "0.45", "predictor", "resid_trace"))), 0.574)

    print("== FINDINGS 10.4.1: position/heading covariate resolution (7-channel, n=10) ==")
    ctg = seed_vals(covar, "0.45", "survival", "target")
    check("covar target reproduces headline (0.752)", float(np.mean(ctg)), 0.752)
    check_true("covar target byte-identical to h8 (determinism receipt)",
               all(abs(a - b) <= 1e-6 for a, b in
                   zip(ctg, seed_vals(h8, "0.45", "survival", "target"))))
    cbt = seed_vals(covar, "0.45", "survival", "behavior_trace_only")
    check("covar behavior trace rises (0.832)", float(np.mean(cbt)), 0.832)
    check_true("covar behavior ceiling above four-channel (0.803)",
               float(np.mean(cbt)) > 0.803)
    crt = seed_vals(covar, "0.45", "survival", "resid_trace")
    check("covar resid_trace (0.723)", float(np.mean(crt)), 0.723)
    check("covar resid_trace boot lo (0.682)", mean_ci(crt)[1], 0.682)
    check("covar resid_trace boot hi (0.760)", mean_ci(crt)[2], 0.760)
    check_int("covar resid_trace seeds >= 0.65 (8)", sum(v >= BAR for v in crt), 8)
    ctlo, cthi = t_ci(crt)
    check("covar resid_trace t lo (0.676)", ctlo, 0.676)
    check("covar resid_trace t hi (0.769)", cthi, 0.769)
    check_true("covar resid_trace t-CI excludes bar", ctlo > BAR)
    check_true("covar resolution: control stronger, signal held (delta > -0.02)",
               float(np.mean(crt)) - float(np.mean(rt)) > -0.02)
    check("covar resid_trace_quad (0.700)",
          float(np.mean(seed_vals(covar, "0.45", "survival", "resid_trace_quad"))), 0.700)
    check("covar untrained resid_trace (0.512)",
          float(np.mean(seed_vals(covar, "0.45", "untrained", "resid_trace"))), 0.512)
    check("covar predictor resid_trace (0.565)",
          float(np.mean(seed_vals(covar, "0.45", "predictor", "resid_trace"))), 0.565)

    print("== FINDINGS 10.5: second capacity (hidden=7) ==")
    surv7 = seed_vals(h7, "0.45", "survival", "target")
    pred7 = seed_vals(h7, "0.45", "predictor", "target")
    untr7 = seed_vals(h7, "0.45", "untrained", "target")
    check("h7 survival target (0.737)", float(np.mean(surv7)), 0.737)
    check("h7 survival boot lo (0.688)", mean_ci(surv7)[1], 0.688)
    check("h7 survival boot hi (0.780)", mean_ci(surv7)[2], 0.780)
    check_int("h7 survival seeds >= 0.65 (8)", sum(v >= BAR for v in surv7), 8)
    check("h7 predictor target (0.714)", float(np.mean(pred7)), 0.714)
    check("h7 untrained target (0.586)", float(np.mean(untr7)), 0.586)
    check_true("h7 dissociation NOT met (lead < 0.05, artifact-conditional claim)",
               float(np.mean(surv7)) - float(np.mean(pred7)) < 0.05)
    rt7 = seed_vals(h7, "0.45", "survival", "resid_trace")
    check("h7 resid_trace (0.722)", float(np.mean(rt7)), 0.722)
    check("h7 resid_trace boot lo (0.678)", mean_ci(rt7)[1], 0.678)
    check("h7 resid_trace boot hi (0.763)", mean_ci(rt7)[2], 0.763)
    tlo7, thi7 = t_ci(rt7)
    check("h7 resid_trace t lo (0.672)", tlo7, 0.672)
    check("h7 resid_trace t hi (0.773)", thi7, 0.773)
    check_int("h7 resid_trace seeds >= 0.65 (8)", sum(v >= BAR for v in rt7), 8)
    check("h7 resid_trace_quad (0.704)",
          float(np.mean(seed_vals(h7, "0.45", "survival", "resid_trace_quad"))), 0.704)
    check("h7 predictor resid_trace (0.691)",
          float(np.mean(seed_vals(h7, "0.45", "predictor", "resid_trace"))), 0.691)
    check("h7 untrained resid_trace (0.579)",
          float(np.mean(seed_vals(h7, "0.45", "untrained", "resid_trace"))), 0.579)
    for agent in ("survival", "predictor", "untrained"):
        m = float(np.mean(seed_vals(h7, "0.45", agent, "behavior_trace_only")))
        check_true(f"h7 {agent} behavior trace in stated 0.762-0.796 range ({m:.3f})",
                   0.762 - TOL <= m <= 0.796 + TOL)

    print("== FINDINGS 10.6: held-out fingerprint probe ==")
    tr = seed_vals(held, "0.45", "survival", "transfer_target")
    check("transfer survival (0.773)", float(np.mean(tr)), 0.773)
    tlo, thi = t_ci(tr)
    check("transfer t lo (0.722)", tlo, 0.722)
    check("transfer t hi (0.824)", thi, 0.824)
    check_int("transfer seeds >= 0.65 (9)", sum(v >= BAR for v in tr), 9)
    trp = seed_vals(held, "0.45", "predictor", "transfer_target")
    tru = seed_vals(held, "0.45", "untrained", "transfer_target")
    check("transfer predictor (0.633)", float(np.mean(trp)), 0.633)
    check_int("transfer predictor seeds >= 0.65 (3)", sum(v >= BAR for v in trp), 3)
    check("transfer untrained floor (0.569)", float(np.mean(tru)), 0.569)
    check_true("pre-registered transfer rule passes (>=0.65 and >untrained+0.05)",
               float(np.mean(tr)) >= BAR
               and float(np.mean(tr)) > float(np.mean(tru)) + 0.05)
    # Biased pre-1633bca cg numbers, retained to verify the frozen §10.6 body
    # (the historical/invalidated record) still matches its committed artifact.
    # The corrected verdict is re-scored below and adjudicated in §10.6.1.
    cg = seed_vals(held, "0.45", "survival", "cg_tail_target")
    check("common-garden survival, biased/historical (0.557)", float(np.mean(cg)), 0.557)
    tlo, thi = t_ci(cg)
    check("common-garden t lo, biased/historical (0.492)", tlo, 0.492)
    check("common-garden t hi, biased/historical (0.622)", thi, 0.622)
    check_int("common-garden seeds >= 0.65, biased/historical (1)", sum(v >= BAR for v in cg), 1)
    check("common-garden predictor, biased/historical (0.409)",
          float(np.mean(seed_vals(held, "0.45", "predictor", "cg_tail_target"))), 0.409)
    check("common-garden untrained, biased/historical (0.377)",
          float(np.mean(seed_vals(held, "0.45", "untrained", "cg_tail_target"))), 0.377)
    check("late-tail decay, biased/historical (0.492)",
          float(np.mean(seed_vals(held, "0.45", "survival", "cg_latetail_target"))), 0.492)

    print("== FINDINGS 10.6 / PREREG 2026-07-16: reverse transfer (train h7, hold out h8) ==")
    rpool = seed_vals(rev, "0.45", "survival", "pool_target")
    check("reverse standard-half survival (0.737)", float(np.mean(rpool)), 0.737)
    check("reverse standard-half boot lo (0.688)", mean_ci(rpool)[1], 0.688)
    check("reverse standard-half boot hi (0.780)", mean_ci(rpool)[2], 0.780)
    rl0 = seed_vals(rev, "0.00", "survival", "pool_target")
    check("reverse L0 (0.517)", float(np.mean(rl0)), 0.517)
    check_true("reverse L0 TOST equivalent to chance",
               equivalence_test(rl0, margin=0.05).equivalent)
    check_true("reverse L0 ROPE accepts equivalence", rope_test(rl0).accept)
    rleak = seed_vals(rev, "0.45", "survival", "pool_reward_leak")
    check("reverse reward-leak (0.567)", float(np.mean(rleak)), 0.567)
    check_true("reverse reward-leak clean 10/10",
               all(seed_vals(rev, "0.45", "survival", "pool_leak_clean")))
    rtr = seed_vals(rev, "0.45", "survival", "transfer_target")
    check("reverse transfer survival (0.638)", float(np.mean(rtr)), 0.638)
    tlo, thi = t_ci(rtr)
    check("reverse transfer t lo (0.600)", tlo, 0.600)
    check("reverse transfer t hi (0.676)", thi, 0.676)
    check_int("reverse transfer seeds >= 0.65 (4)", sum(v >= BAR for v in rtr), 4)
    rtru = seed_vals(rev, "0.45", "untrained", "transfer_target")
    rtrp = seed_vals(rev, "0.45", "predictor", "transfer_target")
    check("reverse transfer untrained floor (0.525)", float(np.mean(rtru)), 0.525)
    check("reverse transfer predictor (0.603)", float(np.mean(rtrp)), 0.603)
    check_true("reverse frozen rule FAILS the absolute bar (0.638 < 0.65)",
               float(np.mean(rtr)) < BAR)
    check_true("reverse floor-margin clause passes (> untrained + 0.05)",
               float(np.mean(rtr)) > float(np.mean(rtru)) + 0.05)
    # Biased pre-1633bca reverse cg numbers, retained as the historical §10.6 body
    # record; corrected verdict re-scored below (§10.6.1).
    rcg = seed_vals(rev, "0.45", "survival", "cg_tail_target")
    check("reverse common-garden survival, biased/historical (0.598)", float(np.mean(rcg)), 0.598)
    tlo, thi = t_ci(rcg)
    check("reverse common-garden t lo, biased/historical (0.547)", tlo, 0.547)
    check("reverse common-garden t hi, biased/historical (0.649)", thi, 0.649)
    check_int("reverse common-garden seeds >= 0.65, biased/historical (4)", sum(v >= BAR for v in rcg), 4)
    check("reverse common-garden predictor, biased/historical (0.504)",
          float(np.mean(seed_vals(rev, "0.45", "predictor", "cg_tail_target"))), 0.504)
    check("reverse common-garden untrained, biased/historical (0.456)",
          float(np.mean(seed_vals(rev, "0.45", "untrained", "cg_tail_target"))), 0.456)
    check("reverse late-tail decay, biased/historical (0.489)",
          float(np.mean(seed_vals(rev, "0.45", "survival", "cg_latetail_target"))), 0.489)

    print("== FINDINGS 10.6.1: common-garden re-score (fixed estimator, frozen rule) ==")
    cg_cases = [
        ("forward (h8 trained, h7 held out)", cgf,
         dict(surv=0.666, untr=0.570, pred=0.588, late=0.586, margin=0.620)),
        ("reverse (h7 trained, h8 held out)", cgr,
         dict(surv=0.684, untr=0.573, pred=0.597, late=0.577, margin=0.623)),
    ]
    for tag, doc, exp in cg_cases:
        sd, adj = doc["strong_drift"], doc["adjudication"]
        surv = float(sd["survival"]["cg_tail_mean"])
        untr = float(sd["untrained"]["cg_tail_mean"])
        late = float(sd["survival"]["cg_latetail_mean"])
        check(f"cg re-score {tag} survival tail", surv, exp["surv"])
        check(f"cg re-score {tag} untrained floor", untr, exp["untr"])
        check(f"cg re-score {tag} predictor tail",
              float(sd["predictor"]["cg_tail_mean"]), exp["pred"])
        check(f"cg re-score {tag} late-tail", late, exp["late"])
        check(f"cg re-score {tag} margin threshold (untrained + 0.05)",
              float(adj["margin_threshold"]), exp["margin"])
        check_true(f"cg re-score {tag} drift-0.00 L0 floors == 0.500 (bias fix confirmed)",
                   all(abs(float(v) - 0.5) <= 1e-6 for v in doc["floor_drift0"].values()))
        check_true(f"cg re-score {tag} survival tail clears bar (>= 0.65)",
                   surv >= BAR and adj["cg_tail_pass_bar"] is True)
        check_true(f"cg re-score {tag} survival tail > untrained + 0.05",
                   surv > untr + 0.05 and adj["cg_tail_pass_margin"] is True)
        check_true(f"cg re-score {tag} frozen rule PASSES (modest persistent component)",
                   adj["cg_channel_pass"] is True)
        check_true(f"cg re-score {tag} late-tail decays below bar (decay diagnostic)",
                   late < BAR)

    print("== FINDINGS 9 / artifacts: B-v2 survival coupling (L2, n=3) ==")
    rep = [float(v) for v in b2["0.45"]["survival"]["pool_target"]]
    check("replication mean (0.523)", float(np.mean(rep)), 0.523)
    check("replication std ddof=0 (0.045)", float(np.std(rep)), 0.045)
    m, lo, hi = mean_ci(rep)
    check("replication boot lo (0.490)", lo, 0.490)
    check("replication boot hi (0.556)", hi, 0.556)
    for got, exp in zip(sorted(rep, reverse=True), [0.586, 0.495, 0.488]):
        check(f"replication per-seed ({exp})", got, exp)
    conf = [float(v) for v in b2c["0.45"]["survival"]["pool_target"]]
    check("confirmatory mean (0.595)", float(np.mean(conf)), 0.595)
    check("confirmatory std ddof=0 (0.014)", float(np.std(conf)), 0.014)

    print("== FINDINGS 2/3: Experiment A/B summaries (committed) ==")
    with open(os.path.join(ARTROOT, "expA", "summary.json"), encoding="utf-8") as fh:
        ea = json.load(fh)
    l1 = {c["label"]: c for c in ea["expA_l1"]["cells"]}
    check("L1 oracle at delta=0.06 (1.000)",
          l1["L1 discretization, delta=0.06"]["oracle_auroc"], 1.000)
    check("L1 L0 control (0.523)",
          l1["L0 control, identical world"]["oracle_auroc"], 0.523)
    check_true("L1 leakage audit passes on clean cells",
               l1["L1 discretization, delta=0.06"]["leakage_pass"]
               and l1["L0 control, identical world"]["leakage_pass"])
    check_true("L1 contaminated-reward negative control CAUGHT (leakage_pass False)",
               not l1["L1 + contaminated reward (+0.02 in surrogate)"]["leakage_pass"])
    l2 = {c["label"]: c for c in ea["expA_l2"]["cells"]}
    check("L2 oracle at drift=0.30 (0.993)",
          l2["L2 rollout drift, drift_sigma=0.30"]["oracle_auroc"], 0.993)
    check("L2 L0 control (0.440)",
          l2["L2 control, drift_sigma=0  -> identical dynamics"]["oracle_auroc"], 0.440)
    with open(os.path.join(ARTROOT, "expB", "summary.json"), encoding="utf-8") as fh:
        eb = json.load(fh)
    sweep = {r["drift"]: r for r in eb["expB_full"]["drift_sweep"]}
    check("expB recurrent-state target @0.45 (0.510)", sweep[0.45]["target_mean"], 0.510)
    check("expB recurrent-state std @0.45 (0.039)", sweep[0.45]["target_std"], 0.039)
    check("expB recurrent-state target @0.20 (0.509)", sweep[0.2]["target_mean"], 0.509)
    surp = {r["drift"]: r for r in eb["expB_surprise"]["drift_sweep_with_std"]}
    check("surprise channel @0.45 (0.596)", surp[0.45]["mean"], 0.596)
    check("surprise channel std @0.45 (0.007)", surp[0.45]["std"], 0.007)
    nl = {r["drift"]: r for r in eb["expB_nonlinear"]["drift_sweep_with_std"]}
    check("nonlinear probe @0.45 (0.482)", nl[0.45]["mean"], 0.482)
    check("nonlinear probe std @0.45 (0.031)", nl[0.45]["std"], 0.031)
    for row in eb["expB_kstep_rerun_20260713"]["horizons"]:
        check_true(f"kstep rerun horizon {row['open_horizon']} at chance "
                   f"({row['drift_045_target_mean']:.3f} in 0.48-0.51)",
                   0.48 - TOL <= row["drift_045_target_mean"] <= 0.51 + TOL)

    print("== FINDINGS 7.1: B-v3 n=10 and capacity ceiling (committed) ==")
    with open(os.path.join(ART, "bv3_n10_summary.json"), encoding="utf-8") as fh:
        bv3 = json.load(fh)["pooled_target_drift045"]["survival"]
    check("B-v3 survival n=10 (0.610)", bv3["mean"], 0.610)
    check("B-v3 boot lo (0.585)", bv3["boot90_lo"], 0.585)
    check("B-v3 boot hi (0.634)", bv3["boot90_hi"], 0.634)
    check_true("B-v3 CI entirely below 0.65 bar", bv3["boot90_hi"] < BAR)
    with open(os.path.join(ART, "sysid_ceiling_n10_summary.json"), encoding="utf-8") as fh:
        ceil = json.load(fh)["pooled_target_drift045"]["survival"]
    check("capacity ceiling n=10 (0.596)", ceil["mean"], 0.596)
    check("ceiling boot lo (0.577)", ceil["boot90_lo"], 0.577)
    check("ceiling boot hi (0.616)", ceil["boot90_hi"], 0.616)
    check_true("ceiling CI entirely below 0.65 bar", ceil["boot90_hi"] < BAR)
    ctlo, cthi = t_ci(ceil["pool_target_per_seed"])
    check("ceiling t lo (0.573)", ctlo, 0.573)
    check("ceiling t hi (0.619)", cthi, 0.619)
    check_true("ceiling t-CI also below 0.65 bar", cthi < BAR)

    print("== FINDINGS 10.7: cross-recipe transfer probe (committed) ==")
    with open(os.path.join(ARTROOT, "l3_crossrecipe", "summary.json"), encoding="utf-8") as fh:
        xr = json.load(fh)
    tr = xr["transfer_rff"]
    sv = tr["survival_per_seed"]
    m, lo, hi = mean_ci(sv, level=0.90, seed=0)
    check("cross-recipe survival mean (0.684)", m, 0.684)
    check("cross-recipe survival boot lo (0.657)", lo, 0.657)
    check("cross-recipe survival boot hi (0.710)", hi, 0.710)
    tlo, thi = t_ci(sv)
    check("cross-recipe survival t lo (0.654)", tlo, 0.654)
    check("cross-recipe survival t hi (0.715)", thi, 0.715)
    check_true("cross-recipe survival t-CI entirely above 0.65 bar", tlo > BAR)
    check_int("cross-recipe survival seeds >= 0.65 (7)",
              sum(1 for v in sv if v >= BAR), 7)
    mu, ulo, uhi = mean_ci(tr["untrained_per_seed"], level=0.90, seed=0)
    check("cross-recipe untrained floor mean (0.548)", mu, 0.548)
    check("cross-recipe untrained boot lo (0.538)", ulo, 0.538)
    check("cross-recipe untrained boot hi (0.557)", uhi, 0.557)
    mp, plo, phi = mean_ci(tr["predictor_per_seed"], level=0.90, seed=0)
    check("cross-recipe predictor mean (0.574)", mp, 0.574)
    check("cross-recipe predictor boot lo (0.554)", plo, 0.554)
    check("cross-recipe predictor boot hi (0.593)", phi, 0.593)
    check_true("cross-recipe frozen rule recomputed (survival >= 0.65 AND "
               "> untrained + 0.05)", m >= BAR and m > mu + 0.05)
    check_true("cross-recipe rule_pass recorded true", bool(tr["rule_pass"]))
    check("cross-recipe rule margin (0.034)", tr["rule_margin"], 0.034)
    check("cross-recipe integrity gate reproduced 0.752",
          xr["integrity_gate"]["drift045_survival_mean_reproduced"], 0.752)
    g0 = xr["gate0"]
    check("cross-recipe gate-0 rff oracle (0.887)", g0["rff"]["selected_oracle_auroc"], 0.887)
    check_true("cross-recipe gate-0 rff oracle in band",
               0.85 <= g0["rff"]["selected_oracle_auroc"] <= 0.95)
    check_true("cross-recipe cd dropped with empty window (floors > 0.6 "
               "wherever oracle in band)",
               g0["cd"]["dropped"] and all(
                   f > 0.6 for o, f in zip(g0["cd"]["sweep_oracle"],
                                           g0["cd"]["sweep_floor"]) if o >= 0.85))

    # Estimator sanity gate: once a corrected cg re-score is promoted
    # (artifacts/expB2/cg_rescore*.json), its drift-0 L0 floors MUST sit in the
    # chance band - a floor outside [0.4, 0.6] means a broken estimator and the
    # audit FAILS, so the July-18 bias class can never silently pass this gate
    # again. (No-op until a re-score artifact exists.)
    import glob as _glob
    for rp in sorted(_glob.glob(os.path.join(ARTROOT, "expB2", "*cg_rescore*.json"))):
        with open(rp, encoding="utf-8") as fh:
            rs = json.load(fh)
        for key, agg in rs.get("aggregate", {}).items():
            if key.startswith("d0.00_"):
                check_true(f"cg re-score L0 floor in chance band ({os.path.basename(rp)}:{key})",
                           0.4 <= agg["cg_tail_mean"] <= 0.6)

    # Same gate for the corrected mp re-scores (FINDINGS 10.6 banner resolution,
    # 2026-07-22): fixed-estimator L0 floors must sit at chance, and every cell's
    # singleton-group re-score must have reproduced its stored pre-fix value
    # (rollout-determinism gate). (No-op until an mp re-score artifact exists.)
    for rp in sorted(_glob.glob(os.path.join(ARTROOT, "expB2", "*mp_rescore*.json"))):
        with open(rp, encoding="utf-8") as fh:
            rs = json.load(fh)
        name = os.path.basename(rp)
        for agent, floor in rs["floor_drift0_fixed"].items():
            check_true(f"mp re-score L0 floor in chance band ({name}:{agent})",
                       0.4 <= floor <= 0.6)
        check_true(f"mp re-score determinism gate ({name})",
                   rs["gates"]["all_cells_reproduce_stored_prefix_value"])
    # the corrected survival means quoted in the FINDINGS 10.6 banner
    for rp, quoted in (("heldout_l3_h8_mp_rescore.json", 0.893),
                       ("heldout_l3_h7_reverse_mp_rescore.json", 0.855)):
        with open(os.path.join(ARTROOT, "expB2", rp), encoding="utf-8") as fh:
            rs = json.load(fh)
        check(f"mp re-score survival mean matches doc quote ({rp})",
              rs["strong_drift"]["survival"]["mp_fixed_mean"], quoted, tol=5e-4)

    print("== FINDINGS Exp C: emergence pilot (milestone 3, N=48 G=30 3 seeds) ==")
    # NOTE (2026-07-18): this block re-verifies the INVALIDATED pilot artifact
    # (git_commit 9758202, pre-fix; FINDINGS 13.C) as a historical record. The
    # re-run's summary will be gated as a separate artifact when promoted.
    print("  [invalidation marker] Exp C checks verify the invalidated pilot's "
          "historical record; re-run pending (FINDINGS 13.C)")
    with open(os.path.join(ARTROOT, "expC", "emergence_pilot_summary.json"),
              encoding="utf-8") as fh:
        ec = json.load(fh)
    cells = sorted(ec["cells"], key=lambda r: r["seed"])
    gen0 = np.array([c["gen0_auroc"] for c in cells])
    ft = np.array([c["final_treat_auroc"] for c in cells])
    fc = np.array([c["final_ctrl_auroc"] for c in cells])
    d_treat = ft - gen0
    d_ctrl = fc - gen0
    contrast = d_treat - d_ctrl
    est = ec["estimand"]
    # per-seed deltas/contrast reproduce the stored estimand from raw AUROCs
    check_true("Exp C per-seed delta_treat reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(d_treat, est["delta_treat"])))
    check_true("Exp C per-seed delta_ctrl reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(d_ctrl, est["delta_ctrl"])))
    check_true("Exp C per-seed contrast reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(contrast, est["contrast"])))
    for got, exp in zip(contrast, [0.002, -0.009, 0.002]):
        check(f"Exp C per-seed contrast ({exp:+.3f})", float(got), exp)
    for got, exp in zip(ft, [0.508, 0.510, 0.509]):
        check(f"Exp C per-seed final treat AUROC ({exp:.3f})", float(got), exp)
    # headline contrast + both interval types, recomputed from the per-seed cells
    check("Exp C mean contrast (-0.002)", float(contrast.mean()), -0.002)
    check_true("Exp C mean contrast reproduces stored",
               abs(float(contrast.mean()) - est["mean_contrast"]) <= 1e-9)
    tlo, thi = t_ci(list(contrast))
    check("Exp C contrast t lo (-0.013)", tlo, -0.013)
    check("Exp C contrast t hi (+0.009)", thi, 0.009)
    check_true("Exp C t-CI reproduces stored",
               abs(tlo - est["t_ci90"][0]) <= 1e-9 and abs(thi - est["t_ci90"][1]) <= 1e-9)
    blo, bhi = mean_ci(list(contrast))[1:]
    check("Exp C contrast boot lo (-0.006)", blo, -0.006)
    check("Exp C contrast boot hi (+0.002)", bhi, 0.002)
    check_true("Exp C boot-CI reproduces stored",
               abs(blo - est["boot_ci90"][0]) <= 1e-9 and abs(bhi - est["boot_ci90"][1]) <= 1e-9)
    check("Exp C mean final treat AUROC (0.509)", float(ft.mean()), 0.509)
    # the pre-registered claim is a null: all three sub-conditions fail, CI spans 0
    check_true("Exp C contrast t-CI spans 0 (ci_excludes_zero False)",
               tlo < 0.0 < thi and not est["ci_excludes_zero"])
    check_true("Exp C mean contrast below SESOI 0.05 (meets_sesoi False)",
               float(contrast.mean()) < 0.05 and not est["meets_sesoi"])
    check_true("Exp C mean final treat AUROC below floor 0.65 (meets_auroc_floor False)",
               float(ft.mean()) < BAR and not est["meets_auroc_floor"])
    check_true("Exp C emergence_claim is False", not est["emergence_claim"])
    # mechanism: selection HAD grip (fitness moved in both arms) but did not route
    # it through detection. On the world-P-fixed re-run nothing dies in either
    # arm, so grip is read off the positive fitness delta, not a survival gap.
    check_true("Exp C gate-2 fitness moved in both arms",
               ec["gates"]["gate2_fitness_moves_treat"]
               and ec["gates"]["gate2_fitness_moves_ctrl"])
    check_true("Exp C seed-0 treatment bit-reproducible",
               ec["gates"]["determinism_bit_reproducible"])
    check_true("Exp C fitness delta positive in every arm-run (selection had grip)",
               all(c["fit_delta_treat"] > 0 and c["fit_delta_ctrl"] > 0 for c in cells))
    check_true("Exp C authentic/surrogate death symmetric at gen0 (world-P fix: no ~0.58-vs-0.01 asymmetry)",
               all(abs(c["death_rate_auth_gen0"] - c["death_rate_surr_gen0"]) <= 1e-6
                   for c in cells)
               and all(c["death_rate_auth_gen0"] <= 1e-6 for c in cells))

    print("== FINDINGS 14: H2 substrate-grounding ablation (A1 graded-seam sweep) ==")
    with open(os.path.join(ARTROOT, "expH2", "summary.json"), encoding="utf-8") as fh:
        h2 = json.load(fh)
    h2_alphas = h2["config"]["alphas"]
    h2_surv = h2["cells"]["survival"]
    h2_untr = h2["cells"]["untrained"]
    # survival collapse curve: recompute each alpha mean from the per-seed cells
    curve_doc = {0.0: 0.506, 0.1: 0.538, 0.25: 0.618, 0.5: 0.683, 0.75: 0.723, 1.0: 0.752}
    surv_means = []
    for al in h2_alphas:
        m = float(np.mean(h2_surv[f"a{al:.2f}"]["per_seed"]))
        surv_means.append(m)
        # alpha=1 raw mean is 0.7525, exactly the 3dp rounding midpoint of the
        # canonical 0.752 L3 headline, so allow one extra digit of slack there.
        tol = 1e-3 if al == 1.0 else TOL
        check(f"H2 survival target at alpha={al:.2f} ({curve_doc[al]:.3f})", m, curve_doc[al], tol=tol)
    check_true("H2 survival collapse strictly monotone increasing in alpha",
               all(b > a for a, b in zip(surv_means, surv_means[1:])))
    check_true("H2 stored Spearman rho == 1.0", abs(h2["survival_monotonicity_rho"] - 1.0) <= 1e-9)
    # endpoints: alpha=1 reproduces the published 0.752 headline; alpha=0 at the L0 floor
    check("H2 alpha=1 survival == published L3 headline (0.752)", surv_means[-1], 0.752, tol=1e-3)
    check("H2 integrity gate alpha=1 survival mean (0.752)",
          h2["integrity"]["survival_mean_alpha1"], 0.752)
    check_true("H2 integrity pools bit-match saved dumps (determinism check #5)",
               h2["integrity"]["pools_bit_match"] is True)
    # L0 anchor: alpha=0 survival equivalent to chance (ROPE [0.45, 0.55])
    h2rr = rope_test(h2_surv["a0.00"]["per_seed"], rope=(0.45, 0.55))
    check("H2 L0 anchor alpha=0 survival mean (0.506)", h2rr.mean, 0.506)
    check_true("H2 L0 anchor accepts equivalence to chance (ROPE)", h2rr.accept)
    check_true("H2 L0 anchor recompute matches stored accept",
               h2rr.accept == bool(h2["l0_anchor_alpha0"]["accept_equiv"]))
    # untrained floor control: flat near chance at EVERY alpha, so the collapse is
    # specific to the learned survival signal, not an artifact of the graded world.
    untr_means = [float(np.mean(h2_untr[f"a{al:.2f}"]["per_seed"])) for al in h2_alphas]
    check("H2 untrained floor at alpha=1 (0.488)", untr_means[-1], 0.488)
    check_true("H2 untrained floor flat/below bar at every alpha (max mean < 0.55)",
               max(untr_means) < 0.55)
    check_true("H2 untrained clears the 0.65 bar in 0/10 seeds at every alpha",
               all(h2_untr[f"a{al:.2f}"]["n_ge_065"] == 0 for al in h2_alphas))

    # ---- FINDINGS 14.5 / 14.6 / 14.7 + B-v3 gates (promoted 2026-09-26) --
    def _load_art(*parts) -> dict:
        with open(os.path.join(ARTROOT, *parts), encoding="utf-8") as fh:
            return json.load(fh)

    def _stored_means_reproduce(label: str, block: dict) -> None:
        """Every {per_seed, mean, n_ge_065} entry must be self-consistent, and where the
        cell files were promoted alongside (per_seed_full), the 4-dp aggregate list must
        be the rounding of the full-precision one (copied, not retyped)."""
        bad = 0
        for arm, e in block.items():
            if not isinstance(e, dict) or "per_seed" not in e:
                continue
            ps = [float(v) for v in e["per_seed"]]
            if abs(float(np.mean(ps)) - e["mean"]) > 1e-4:
                bad += 1
            if int(sum(v >= BAR for v in ps)) != e["n_ge_065"]:
                bad += 1
            if "per_seed_full" in e and any(abs(a - b) > 5.1e-5
                                            for a, b in zip(ps, e["per_seed_full"])):
                bad += 1
            if "per_seed_full" in e and len(e["per_seed_full"]) != len(ps):
                bad += 1
        check_true(f"{label}: stored mean/n_ge_065 reproduce from per-seed lists", bad == 0)

    def _arm_mean(e: dict) -> float:
        """Doc numbers are 3-dp roundings of the full-precision across-seed mean; the
        bundle aggregate stores a 4-dp rounding, which can land on a 3-dp midpoint
        (e.g. gn untrained 0.5415 vs true 0.54151 -> 0.542). Prefer the promoted
        full-precision cell values whenever they were copied alongside."""
        if "per_seed_full" in e:
            return float(np.mean(e["per_seed_full"]))
        return float(e["mean"])

    print("== FINDINGS 14.5: H2 texture knockout + capacity ladder (hidden=8 / hidden=7) ==")
    tk = {8: _load_art("expH2", "texture_knockout_h8.json"),
          7: _load_art("expH2", "texture_knockout_h7.json")}
    tk_doc = {8: dict(gn=(0.539, 0.542, 0.556), pub=0.752,
                      ladder={"h16": (0.701, 8), "h32": (0.622, 3), "h64": (0.541, 0)}),
              7: dict(gn=(0.510, 0.520, 0.539), pub=0.737,
                      ladder={"h16": (0.574, 1), "h32": (0.516, None), "h64": (0.513, None)})}
    for hid, doc in tk.items():
        exp = tk_doc[hid]
        ch = doc["channels"]
        for name in ("gn", "h16", "h32", "h64"):
            _stored_means_reproduce(f"h{hid} texture knockout channel {name}", ch[name])
        check_int(f"h{hid} texture knockout: hidden recorded", doc["hidden"], hid)
        check_int(f"h{hid} texture knockout: n_seeds (10)", doc["config"]["n_seeds"], 10)
        gs, gu, gp = exp["gn"]
        check(f"h{hid} gn survival transfer ({gs:.3f})", _arm_mean(ch["gn"]["survival"]), gs)
        check(f"h{hid} gn untrained floor ({gu:.3f})", _arm_mean(ch["gn"]["untrained"]), gu)
        check(f"h{hid} gn predictor ({gp:.3f})", _arm_mean(ch["gn"]["predictor"]), gp)
        check_true(f"h{hid} gn frozen rule fails both clauses (recomputed)",
                   ch["gn"]["survival"]["mean"] < BAR
                   and ch["gn"]["survival"]["mean"] <= ch["gn"]["untrained"]["mean"] + 0.05)
        check_true(f"h{hid} gn rule_pass recorded False, verdict H2_SUPPORTED",
                   doc["gn_rule"]["pass"] is False and doc["gn_rule"]["verdict"] == "H2_SUPPORTED")
        check_true(f"h{hid} gn not dropped at gate 0", doc["gn_dropped_at_gate0"] is False)
        check(f"h{hid} gn sigma_v (0.010)", doc["config"]["gn_sigma_v"], 0.01)
        # integrity gate: 60 reloaded agents regenerate pools bit-identically and the
        # drift-0.45 survival mean reproduces the published headline
        integ = doc["integrity"]
        check_true(f"h{hid} integrity gate PASSED in run log", integ.get("gate_passed") is True)
        check(f"h{hid} integrity reproduces published survival ({exp['pub']:.3f})",
              integ.get("survival_mean_reproduced", float("nan")), exp["pub"])
        check(f"h{hid} published_target_check recorded ({exp['pub']:.3f})",
              doc["config"]["published_target_check"], exp["pub"])
        check_int(f"h{hid} integrity: 60 reloaded agents ok", integ.get("n_integrity_ok_lines", -1), 60)
        # capacity ladder oracles are intentionally sub-band
        lo = doc["ladder_oracles"]
        check(f"h{hid} ladder oracle h16 (0.788)", lo["h16"], 0.788)
        check(f"h{hid} ladder oracle h32 (0.656)", lo["h32"], 0.656)
        check(f"h{hid} ladder oracle h64 (0.603)", lo["h64"], 0.603)
        for name, (m, n_above) in exp["ladder"].items():
            check(f"h{hid} ladder {name} survival ({m:.3f})", _arm_mean(ch[name]["survival"]), m)
            if n_above is not None:
                check_int(f"h{hid} ladder {name} seeds >= 0.65 ({n_above})",
                          ch[name]["survival"]["n_ge_065"], n_above)
        means = [ch[n]["survival"]["mean"] for n in ("h16", "h32", "h64")]
        check_true(f"h{hid} ladder survival co-decays monotonically with oracle",
                   means[0] > means[1] > means[2])
        check_true(f"h{hid} ladder promotion rule not evaluated (eligible False)",
                   doc["ladder_promotion_eligible"] is False)
    # gate 0 for the gn comparator (hidden=8 calibration, reused verbatim at hidden=7)
    g0 = tk[8]["gate0"]["gn"]
    sel = g0["selected_row"]
    check("gate-0 gn selected sigma_v (0.010)", sel["sigma_v"], 0.01)
    check("gate-0 gn oracle AUROC (0.865)", sel["oracle_auroc"], 0.865)
    check("gate-0 gn untrained floor (0.448)", sel["floor"], 0.448)
    check_true("gate-0 gn selected row in band, mech leak pass, floor ok, passes gate 0",
               sel["in_band"] and sel["mech_leak_pass"] and sel["floor_ok"] and sel["passes_gate0"])
    check_true("gate-0 gn exactly one sigma_v passes gate 0",
               sum(1 for r in g0["rows"] if r["passes_gate0"]) == 1)
    lad = {r["hidden"]: r for r in tk[8]["gate0"]["ladder"]["rows"]}
    check_true("gate-0 ladder capacities all sub-band with clean floors and leakage",
               all(not lad[h]["in_band"] and lad[h]["mech_leak_pass"] and lad[h]["floor_ok"]
                   for h in (16, 32, 64)))
    check_true("gate-0 ladder oracles match the aggregate's ladder_oracles",
               all(abs(lad[h]["oracle_auroc"] - tk[8]["ladder_oracles"][f"h{h}"]) <= 1e-9
                   for h in (16, 32, 64)))
    check_true("h7 texture knockout shares the hidden=8 gate-0 calibration",
               tk[7]["gate0_shared_from_hidden8"] is True
               and tk[7]["gate0"]["gn"]["selected_row"] == sel)

    print("== FINDINGS 14.6: A2 observation-channel localization (hidden=8 / hidden=7) ==")
    ob = {8: _load_art("expH2", "obs_localization_h8.json"),
          7: _load_art("expH2", "obs_localization_h7.json")}
    # (survival mean, survival seeds >= 0.65, predictor mean, untrained mean)
    ob_doc = {8: {"none": (0.753, 8, 0.573, 0.488), "vision": (0.686, 7, 0.598, 0.567),
                  "intero": (0.756, 8, 0.562, 0.506), "all": (0.500, 0, 0.500, 0.500)},
              7: {"none": (0.737, 8, 0.714, 0.586), "vision": (0.764, 9, 0.674, 0.724),
                  "intero": (0.742, 8, 0.639, 0.646), "all": (0.500, 0, 0.500, 0.500)}}
    zero_dims = {"none": 0, "vision": 120, "intero": 14, "all": 146}
    traces = {8: h8, 7: h7}
    for hid, doc in ob.items():
        masks = doc["masks"]
        check_int(f"h{hid} obs localization: hidden recorded", doc["config"]["hidden"], hid)
        check_int(f"h{hid} obs localization: obs_dim (146)", doc["config"]["obs_dim"], 146)
        for mask, (sm, sn, pm, um) in ob_doc[hid].items():
            _stored_means_reproduce(f"h{hid} obs mask={mask}", masks[mask])
            # the unmasked hidden=8 survival mean is 0.75249 at full precision (the
            # 0.752 headline); the 14.6 table quotes 0.753, a round-up of the bundle's
            # 4-dp aggregate 0.7525. Allow one extra digit there; recorded 2026-09-26.
            tol = 1e-3 if mask == "none" else TOL
            check(f"h{hid} mask={mask} survival ({sm:.3f})", _arm_mean(masks[mask]["survival"]), sm, tol=tol)
            check_int(f"h{hid} mask={mask} survival seeds >= 0.65 ({sn})",
                      masks[mask]["survival"]["n_ge_065"], sn)
            check(f"h{hid} mask={mask} predictor ({pm:.3f})", _arm_mean(masks[mask]["predictor"]), pm)
            check(f"h{hid} mask={mask} untrained ({um:.3f})", _arm_mean(masks[mask]["untrained"]), um)
            zd = masks[mask]["zeroed_dims"] or {}
            check_int(f"h{hid} mask={mask} zeroed dims ({zero_dims[mask]}/146)",
                      zd.get("zeroed", -1), zero_dims[mask])
        # the unmasked pools are a determinism receipt: they must reproduce the
        # published per-seed survival targets of the traced hidden=N run
        base = masks["none"]["survival"]["per_seed"]
        ref = seed_vals(traces[hid], "0.45", "survival", "target")
        check_true(f"h{hid} mask=none survival per-seed reproduces the published run (4 dp)",
                   len(base) == len(ref) and all(abs(a - b) <= 5.1e-5 for a, b in zip(base, ref)))
        check_true(f"h{hid} mask=all collapses every arm to exactly 0.500 in every seed",
                   all(v == 0.5 for arm in ("survival", "predictor", "untrained")
                       for v in masks["all"][arm]["per_seed"]))
    check_true("h8 vision mask drops survival by > 0.05; intero mask leaves it within 0.01",
               ob[8]["masks"]["none"]["survival"]["mean"] - ob[8]["masks"]["vision"]["survival"]["mean"] > 0.05
               and abs(ob[8]["masks"]["intero"]["survival"]["mean"]
                       - ob[8]["masks"]["none"]["survival"]["mean"]) < 0.01)
    check_true("h7 neither single-channel mask collapses survival (both stay >= baseline)",
               ob[7]["masks"]["vision"]["survival"]["mean"] >= ob[7]["masks"]["none"]["survival"]["mean"]
               and ob[7]["masks"]["intero"]["survival"]["mean"] >= ob[7]["masks"]["none"]["survival"]["mean"])

    print("== FINDINGS 14.7: A3 L1 substrate-grounding battery (organism null) ==")
    org = _load_art("expL1", "organism_summary.json")
    abl = _load_art("expL1", "h2_ablations.json")
    lob = _load_art("expL1", "obs_localization.json")
    # gate 0: frozen grid and the matched-band noise comparator
    gd, gn_ = org["gate0"]["delta"], org["gate0"]["noise"]
    check("L1 gate-0 chosen delta (0.023)", gd["chosen_delta"], 0.023)
    check("L1 gate-0 oracle AUROC at delta (0.873)", gd["chosen_auroc"], 0.873)
    check_true("L1 gate-0 delta row leakage clean",
               gd["chosen_row"] is not None and gd["chosen_row"]["leakage_pass"] is True)
    check("L1 noise comparator sigma_o (0.010)", gn_["chosen_sigma"], 0.01)
    check("L1 noise comparator oracle AUROC (0.873)", gn_["chosen_row"]["oracle_auroc"], 0.873)
    check_true("L1 noise comparator leakage clean", gn_["chosen_row"]["leakage_pass"] is True)
    check_true("L1 gate-0 sensor sigma 0.01 shared by delta and noise calibrations",
               abs(gd["sensor_sigma"] - 0.01) <= 1e-12
               and abs(gn_["chosen_row"]["sensor_sigma"] - 0.01) <= 1e-12)
    # organism result at the headline grid (n = 10, three arms)
    hd = org["headline_drift"]
    check("L1 headline drift key is 0.023", float(hd), 0.023)
    l1s = [float(v) for v in org["arms"][hd]["survival"]["pool_target"]]
    l1p = [float(v) for v in org["arms"][hd]["predictor"]["pool_target"]]
    l1u = [float(v) for v in org["arms"][hd]["untrained"]["pool_target"]]
    check_int("L1 n seeds (10)", len(l1s), 10)
    check("L1 survival pooled target (0.533)", float(np.mean(l1s)), 0.533)
    m_, lo_, hi_ = mean_ci(l1s)
    check("L1 survival boot lo (0.509)", lo_, 0.509)
    check("L1 survival boot hi (0.556)", hi_, 0.556)
    check_int("L1 survival seeds >= 0.65 (0)", sum(v >= BAR for v in l1s), 0)
    stored = org["arms"][hd]["survival"]["aggregate"]["pool_target"]
    check_true("L1 stored survival aggregate reproduces from per-seed list",
               abs(stored["mean"] - m_) <= 1e-9 and abs(stored["boot90"][0] - lo_) <= 1e-9
               and abs(stored["boot90"][1] - hi_) <= 1e-9 and stored["n_ge_065"] == 0)
    check("L1 predictor pooled target (0.489)", float(np.mean(l1p)), 0.489)
    check("L1 untrained pooled target (0.494)", float(np.mean(l1u)), 0.494)
    check_true("L1 primary H_B2 not met (survival < 0.65)", float(np.mean(l1s)) < BAR)
    l0v = org["gates"]["l0_control"]["survival_pool_target_per_seed"]
    l0r = rope_test(l0v)
    check("L1 L0 ROPE p (0.989)", l0r.p_in_rope, 0.989)
    check_true("L1 L0 ROPE accepts equivalence (recomputed == stored)",
               l0r.accept and org["gates"]["l0_control"]["rope"]["accept"] is True
               and abs(l0r.p_in_rope - org["gates"]["l0_control"]["rope"]["p_in_rope"]) <= 1e-9)
    check_true("L1 L0 TOST equivalent to chance", equivalence_test(l0v, margin=0.05).equivalent
               and org["gates"]["l0_control"]["tost"]["equivalent"] is True)
    gt = org["gates"]
    check_true("L1 engagement gate passes (20/20 cells)",
               gt["engagement"]["pass"] and gt["engagement"]["n_engaged"] == 20)
    check_true("L1 leakage gates pass (pool and mp clean in every cell)",
               gt["leakage"]["pool_leak_clean_all"] and gt["leakage"]["mp_leak_clean_all"])
    check_true("L1 survivorship gate passes (0 early deaths)", gt["survivorship"]["deaths_total"] == 0)
    check_true("L1 speed positive control >= 0.75 in every pool", gt["speed_positive_control"]["pass"])
    check_true("L1 ceilings readable (energy / food anchors > 0.6 in every survival pool)",
               all(v > 0.6 for v in org["arms"][hd]["survival"]["pool_anchor_energy"]
                   + org["arms"][hd]["survival"]["pool_anchor_food"]))
    check_true("L1 run provenance: commit 8a71593, single fingerprint",
               org["git_commit_at_run"] == ["8a71593"] and bool(org["fingerprint"]))
    # A1 graded seam (delta ladder): flat near chance, integrity bit-match
    a1 = abl["a1_graded_seam"]
    check_true("L1 A1 integrity: pools bit-match saved dumps",
               a1["integrity"]["checked"] and a1["integrity"]["pools_bit_match"] is True)
    check("L1 A1 integrity reproduces headline survival (0.533)",
          a1["integrity"]["survival_mean_headline"], 0.533)
    check_true("L1 A1 integrity gate PASSED in run log",
               abl["integrity"].get("gate_passed") is True
               and abs(abl["integrity"]["survival_mean_reproduced"] - 0.533) <= TOL)
    check_true("L1 A1 deltas are {0, .25, .5, .75, 1} x headline 0.023",
               all(abs(d - f * 0.023) <= 1e-9 for d, f in zip(a1["deltas"], (0, .25, .5, .75, 1.0))))
    a1_keys = ["survival_d0.0000", "survival_d0.0057", "survival_d0.0115",
               "survival_d0.0173", "survival_d0.0230"]
    for key, exp_m in zip(a1_keys, (0.522, 0.519, 0.528, 0.547, 0.533)):
        ps = [float(v) for v in a1[f"{key}_per_seed"]]
        check(f"L1 A1 {key} survival ({exp_m:.3f})", float(np.mean(ps)), exp_m)
        check_true(f"L1 A1 {key} stored mean/n_ge_065 reproduce (0/10 >= 0.65)",
                   abs(float(np.mean(ps)) - a1[f"{key}_mean"]) <= 1e-4
                   and a1[f"{key}_n_ge_065"] == 0 and all(v < BAR for v in ps))
    check_true("L1 A1 survival_curve equals the per-delta means",
               all(abs(c - a1[f"{k}_mean"]) <= 1e-9 for c, k in zip(a1["survival_curve"], a1_keys)))
    check_true("L1 A1 headline-delta per-seed == organism run per-seed (4 dp)",
               all(abs(a - b) <= 5.1e-5 for a, b in zip(a1["survival_d0.0230_per_seed"], l1s)))
    check_true("L1 A1 L0 anchor accepts chance equivalence", a1["l0_anchor"]["accept_equiv"] is True)
    # A2 noise knockout: vacuous rule failure on a null primary
    nz = abl["a2_noise_knockout"]
    check("L1 A2 noise sigma (0.010)", nz["noise_sigma"], 0.01)
    check("L1 A2 noise survival (0.525)", float(np.mean(nz["survival_per_seed"])), 0.525)
    check("L1 A2 noise untrained (0.509)", float(np.mean(nz["untrained_per_seed"])), 0.509)
    check("L1 A2 noise predictor (0.516)", float(np.mean(nz["predictor_per_seed"])), 0.516)
    check_true("L1 A2 frozen rule fails both clauses (recomputed; verdict H2_SUPPORTED)",
               float(np.mean(nz["survival_per_seed"])) < BAR and nz["gn_rule_pass"] is False
               and nz["gn_verdict"] == "H2_SUPPORTED")
    # A2 observation-channel localization at L1
    lm = lob["masks"]
    check("L1 obs delta (0.023)", lob["config"]["l1_delta"], 0.023)
    check("L1 obs sensor sigma (0.010)", lob["config"]["sensor_sigma"], 0.01)
    for mask in ("none", "vision", "intero", "all"):
        _stored_means_reproduce(f"L1 obs mask={mask}", lm[mask])
        check_int(f"L1 obs mask={mask} zeroed dims ({zero_dims[mask]}/146)",
                  (lm[mask]["zeroed_dims"] or {}).get("zeroed", -1), zero_dims[mask])
    check("L1 obs baseline reproduces headline survival (0.533)", _arm_mean(lm["none"]["survival"]), 0.533)
    check_true("L1 obs baseline per-seed == organism run per-seed (4 dp)",
               all(abs(a - b) <= 5.1e-5 for a, b in zip(lm["none"]["survival"]["per_seed"], l1s)))
    check_true("L1 obs intero mask stays at chance (~0.51)",
               0.50 <= lm["intero"]["survival"]["mean"] <= 0.52)
    check_true("L1 obs all-mask collapses every arm to 0.500",
               all(lm["all"][arm]["mean"] == 0.5 for arm in ("survival", "predictor", "untrained")))
    check("L1 obs vision mask survival (0.696)", _arm_mean(lm["vision"]["survival"]), 0.696)
    check("L1 obs vision mask untrained (0.703)", _arm_mean(lm["vision"]["untrained"]), 0.703)
    check_true("L1 obs vision lift is non-specific (untrained >= survival)",
               lm["vision"]["untrained"]["mean"] >= lm["vision"]["survival"]["mean"])

    print("== PREREGISTRATION_Bv3 section 12: B-v3 n=10 gates (promoted 2026-09-26) ==")
    bg = _load_art("expB2", "bv3_n10_gates.json")
    check_true("B-v3 gates: run commit 820849f, single fingerprint",
               bg["git_commit_at_run"] == ["820849f"] and bool(bg["fingerprint"]))
    with open(os.path.join(ART, "bv3_n10_summary.json"), encoding="utf-8") as fh:
        bv3_seed = json.load(fh)["pooled_target_drift045"]["survival"]["pool_target_per_seed"]
    bg_seed = [float(v) for v in bg["arms"]["0.45"]["survival"]["pool_target"]]
    check_true("B-v3 gates survival per-seed byte-identical to bv3_n10_summary",
               len(bg_seed) == 10 and all(a == b for a, b in zip(bg_seed, bv3_seed)))
    check("B-v3 gates survival mean reproduces 0.610", float(np.mean(bg_seed)), 0.610)
    g = bg["gates"]
    check_true("B-v3 gate 1 engagement: 20/20 cells engaged",
               g["engagement"]["pass"] and g["engagement"]["n_engaged"] == 20
               and g["engagement"]["n_cells"] == 20)
    l0b = g["l0_control"]["survival_pool_target_per_seed"]
    tb = equivalence_test(l0b, margin=0.05)
    check("B-v3 gate 2 L0 survival mean (0.517)", tb.mean, 0.517)
    check("B-v3 gate 2 L0 TOST p (0.010)", tb.p_value, 0.010)
    check_true("B-v3 gate 2 L0 TOST equivalent (recomputed == stored)",
               tb.equivalent and g["l0_control"]["tost"]["equivalent"] is True
               and abs(tb.p_value - g["l0_control"]["tost"]["p_value"]) <= 1e-9)
    rb = rope_test(l0b)
    check("B-v3 gate 2 L0 ROPE p (0.999)", rb.p_in_rope, 0.999)
    check_true("B-v3 gate 2 L0 ROPE accepts", rb.accept and g["l0_control"]["rope"]["accept"] is True)
    check("B-v3 gate 3 min speed probe across all pools (0.784)",
          min(g["speed_positive_control"]["min_speed"].values()), 0.784)
    check_true("B-v3 gate 3 speed >= 0.75 in every pool", g["speed_positive_control"]["pass"])
    check_int("B-v3 gate 4 mp leakage clean cells (59 of 60)", g["leakage"]["n_clean"], 59)
    check_int("B-v3 gate 4 mp leakage cells audited (60)", g["leakage"]["n_cells"], 60)
    fails = g["leakage"]["failures"]
    check_true("B-v3 gate 4 exactly one exception: drift 0.45 seed 6 survival reward_sum",
               len(fails) == 1 and fails[0]["drift"] == "0.45" and fails[0]["seed"] == 6
               and fails[0]["agent"] == "survival" and fails[0]["channel"] == "reward_sum")
    check("B-v3 gate 4 exception reward_sum AUROC (0.393)", fails[0]["value"], 0.393)
    check("B-v3 gate 4 exception max_abs_dev (0.107)", fails[0]["max_abs_dev"], 0.107)
    check_true("B-v3 gate 4 exception recomputed from the per-cell mp leakage rows",
               sum(1 for r in bg["mp_leakage"] if not r["leakage_clean"]) == 1)
    check_true("B-v3 gates: leakage pass recorded False, all_pass recorded False",
               g["leakage"]["pass"] is False and g["all_pass"] is False)
    check_true("B-v3 gates: the three remaining gates recorded pass",
               g["engagement"]["pass"] and g["l0_control"]["pass"]
               and g["speed_positive_control"]["pass"])

    # ---- FINDINGS 10.4.2: sensory-echo control (promoted 2026-09-26) ------
    print("\n== FINDINGS 10.4.2: sensory-echo control (hidden 8, readout-only) ==")
    se = _load_art("expB2", "sensory_echo_l3_h8.json")
    check_true("sensory-echo integrity: every regenerated pool bit-matches the saved dumps",
               se["integrity"]["all_match"] is True)
    check_true("sensory-echo integrity: published 0.752 reproduced",
               se["integrity"]["target_reproduced"] is True)
    check_int("sensory-echo cells (60)", len(se["cells"]), 60)
    for key, block in se["aggregate"].items():
        for met, v in block.items():
            check(f"sensory-echo {key} {met}: stored mean reproduces per-seed mean",
                  float(np.mean(v["per_seed"])), float(v["mean"]))
            check_int(f"sensory-echo {key} {met}: n_ge_065 reproduces per-seed count",
                      int(sum(x >= BAR for x in v["per_seed"])), int(v["n_ge_065"]))
    sv = se["aggregate"]["d=0.45 survival"]
    check("10.4.2 survival target reproduces 0.752", sv["target"]["mean"], 0.752)
    check("10.4.2 survival resid_trace reproduces 10.4.1's 0.723", sv["resid_trace"]["mean"], 0.723)
    check("10.4.2 survival obs_trace_only (0.709)", sv["obs_trace_only"]["mean"], 0.709)
    check("10.4.2 survival resid_obs (0.731)", sv["resid_obs"]["mean"], 0.731)
    lo, hi = t_ci(sv["resid_obs"]["per_seed"])
    check("10.4.2 survival resid_obs t90 lo (0.690)", lo, 0.690)
    check("10.4.2 survival resid_obs t90 hi (0.772)", hi, 0.772)
    check_int("10.4.2 survival resid_obs seeds >= 0.65 (8)", sv["resid_obs"]["n_ge_065"], 8)
    check("10.4.2 survival resid_obs_int (0.758)", sv["resid_obs_int"]["mean"], 0.758)
    check("10.4.2 survival resid_obs_beh (0.670)", sv["resid_obs_beh"]["mean"], 0.670)
    lo, hi = t_ci(sv["resid_obs_beh"]["per_seed"])
    check("10.4.2 survival resid_obs_beh t90 lo (0.638)", lo, 0.638)
    check("10.4.2 survival resid_obs_beh t90 hi (0.702)", hi, 0.702)
    check("10.4.2 predictor resid_obs (0.542)",
          se["aggregate"]["d=0.45 predictor"]["resid_obs"]["mean"], 0.542)
    check("10.4.2 untrained resid_obs (0.534)",
          se["aggregate"]["d=0.45 untrained"]["resid_obs"]["mean"], 0.534)
    check("10.4.2 drift-0 survival resid_obs floor (0.504)",
          se["aggregate"]["d=0.00 survival"]["resid_obs"]["mean"], 0.504)
    check("10.4.2 drift-0 untrained resid_obs floor (0.472)",
          se["aggregate"]["d=0.00 untrained"]["resid_obs"]["mean"], 0.472)
    dec = se["decision"]
    check_true("10.4.2 frozen rule passes both clauses", bool(dec["pass_bar"] and dec["pass_margin"]))
    check_true("10.4.2 stored rule flags reproduce from stored means",
               (dec["survival_resid_obs"] >= BAR) == dec["pass_bar"]
               and (dec["survival_resid_obs"] > dec["untrained_resid_obs"] + 0.05) == dec["pass_margin"])
    check_true("FINDINGS carries the 10.4.2 sensory-echo subsection",
               "### 10.4.2" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"),
                                    encoding="utf-8").read())

    # ---- FINDINGS 10.8 / 10.9: cloud reviewer-gap runs (promoted 2026-09-27) --
    print("\n== FINDINGS 10.8: architecture baseline (no world-model auxiliary) ==")
    ab = _load_art("expB2", "arch_baseline_l3_h8_nowm.json")
    for arm, ref in (("survival", 0.601), ("predictor", 0.589), ("untrained", 0.529)):
        blk = ab["arms"][ab["dmax"]][arm]["pool_target"]
        check(f"10.8 {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"10.8 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv = ab["arms"][ab["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.8 survival t90 lo (0.549)", lo, 0.549)
    check("10.8 survival t90 hi (0.654)", hi, 0.654)
    check_int("10.8 survival seeds >= 0.65 (1)", sv["n_ge_065"], 1)
    check("10.8 survival resid_trace (0.646)", ab["behavior_audit"]["survival"]["resid_trace"]["mean"], 0.646)
    check_true("10.8 all gates pass", ab["gates"]["l0_tost"]["equivalent"] and ab["gates"]["l0_rope"]["accept"]
               and ab["gates"]["pool_leak_clean_all"] and ab["gates"]["untrained_floor_ok"]
               and all(v["pass"] == v["n"] for v in ab["gates"]["engagement"].values()))
    check_true("10.8 auxiliary-conditional cell: survival below bar, predictor margin not met",
               (not ab["decision"]["pass_bar"]) and (not ab["decision"]["pass_margin_predictor"]))
    print("\n== FINDINGS 10.9: second fingerprint instance (G seed 1, hidden 10) ==")
    si = _load_art("expB2", "second_instance_l3_h10_gseed1.json")
    check_int("10.9 gate 0 selected hidden (10)", int(si["gate0_calibration"]["selected_hidden"]), 10)
    rows = {int(r["hidden"]): r for r in si["gate0_calibration"]["rows"]}
    check("10.9 gate 0 hidden 8 oracle (0.928)", rows[8]["oracle_auroc"], 0.928)
    check("10.9 gate 0 hidden 8 floor dirty (0.664)", rows[8]["floor"], 0.664)
    check_true("10.9 gate 0 hidden 8 fails at seed 1", not rows[8]["passes_gate0"])
    check("10.9 gate 0 hidden 10 oracle (0.893)", rows[10]["oracle_auroc"], 0.893)
    check("10.9 gate 0 hidden 10 floor (0.484)", rows[10]["floor"], 0.484)
    check_true("10.9 gate 0 hidden 10 passes", bool(rows[10]["passes_gate0"]))
    for arm, ref in (("survival", 0.639), ("predictor", 0.534), ("untrained", 0.514)):
        blk = si["arms"][si["dmax"]][arm]["pool_target"]
        check(f"10.9 {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"10.9 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv = si["arms"][si["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.9 survival t90 lo (0.610)", lo, 0.610)
    check("10.9 survival t90 hi (0.669)", hi, 0.669)
    check_int("10.9 survival seeds >= 0.65 (5)", sv["n_ge_065"], 5)
    rt = si["behavior_audit"]["survival"]["resid_trace"]
    check("10.9 survival resid_trace (0.658)", rt["mean"], 0.658)
    lo, hi = t_ci(rt["per_seed"])
    check("10.9 survival resid_trace t90 lo (0.625)", lo, 0.625)
    check("10.9 survival resid_trace t90 hi (0.691)", hi, 0.691)
    check_true("10.9 all gates pass", si["gates"]["l0_tost"]["equivalent"] and si["gates"]["l0_rope"]["accept"]
               and si["gates"]["pool_leak_clean_all"] and si["gates"]["untrained_floor_ok"]
               and all(v["pass"] == v["n"] for v in si["gates"]["engagement"].values()))
    check_true("10.9 replication not claimed: bar missed, both margin clauses pass",
               (not si["decision"]["pass_bar"]) and si["decision"]["pass_margin_predictor"]
               and si["decision"]["pass_margin_untrained"])
    for sec in ("### 10.8", "### 10.9"):
        check_true(f"FINDINGS carries the {sec} subsection",
                   sec in open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"),
                               encoding="utf-8").read())

    print("\n== FINDINGS 10.9 addendum: second instance re-measured on GPU (2026-09-29) ==")
    sg = _load_art("expB2", "second_instance_l3_h10_gseed1_gpu.json")
    check_true("10.9-gpu execution names the owner's GPU machine", sg["execution"].startswith("owner's GPU"))
    check_int("10.9-gpu gate 0 re-run selected hidden (10)", int(sg["gate0_calibration"]["selected_hidden"]), 10)
    g10 = {int(r["hidden"]): r for r in sg["gate0_calibration"]["rows"]}[10]
    check("10.9-gpu gate 0 hidden 10 oracle on CUDA (0.893)", g10["oracle_auroc"], 0.893)
    check("10.9-gpu gate 0 hidden 10 floor on CUDA (0.521)", g10["floor"], 0.521)
    for arm, ref in (("survival", 0.612), ("predictor", 0.529), ("untrained", 0.521)):
        blk = sg["arms"][sg["dmax"]][arm]["pool_target"]
        check(f"10.9-gpu {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"10.9-gpu {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    svg = sg["arms"][sg["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(svg["per_seed"])
    check("10.9-gpu survival t90 lo (0.582)", lo, 0.582)
    check("10.9-gpu survival t90 hi (0.642)", hi, 0.642)
    check_int("10.9-gpu survival seeds >= 0.65 (3)", svg["n_ge_065"], 3)
    rtg = sg["behavior_audit"]["survival"]["resid_trace"]
    check("10.9-gpu survival resid_trace (0.638)", rtg["mean"], 0.638)
    lo, hi = t_ci(rtg["per_seed"])
    check("10.9-gpu survival resid_trace t90 lo (0.612)", lo, 0.612)
    check("10.9-gpu survival resid_trace t90 hi (0.663)", hi, 0.663)
    check("10.9-gpu lead over predictor (+0.083)", sg["decision"]["lead_over_predictor"], 0.083)
    check("10.9-gpu lead over untrained (+0.091)", sg["decision"]["lead_over_untrained"], 0.091)
    check_true("10.9-gpu rule not met: bar missed, both margin clauses pass",
               (not sg["decision"]["pass_bar"]) and sg["decision"]["pass_margin_predictor"]
               and sg["decision"]["pass_margin_untrained"])
    check_true("10.9-gpu L0 gate open as stated (TOST and ROPE both reject at n = 10)",
               (not sg["gates"]["l0_tost"]["equivalent"]) and (not sg["gates"]["l0_rope"]["accept"]))
    check("10.9-gpu L0 survival mean (0.539)", sg["gates"]["l0_survival_mean"], 0.539)
    check_true("10.9-gpu remaining gates pass (engagement, leak, floor, deaths)",
               sg["gates"]["pool_leak_clean_all"] and sg["gates"]["untrained_floor_ok"]
               and sg["gates"]["deaths_total"] == 0
               and all(v["pass"] == v["n"] for v in sg["gates"]["engagement"].values()))
    # paired device delta, recomputed from the two committed per-seed arrays
    d_surv = np.asarray(svg["per_seed"]) - np.asarray(sv["per_seed"])
    d_pred = (np.asarray(sg["arms"][sg["dmax"]]["predictor"]["pool_target"]["per_seed"])
              - np.asarray(si["arms"][si["dmax"]]["predictor"]["pool_target"]["per_seed"]))
    check("10.9-gpu paired GPU-minus-CPU survival delta (-0.027)", float(d_surv.mean()), -0.027)
    lo, hi = t_ci(list(d_surv))
    check("10.9-gpu paired delta t90 lo (-0.060)", lo, -0.060)
    check("10.9-gpu paired delta t90 hi (+0.005)", hi, 0.005)
    check("10.9-gpu paired predictor delta (-0.005)", float(d_pred.mean()), -0.005)
    check_true("10.9-gpu paired delta interval includes zero", lo < 0 < hi)

    print("\n== FINDINGS 10.4.2 addendum: sensory-echo control at hidden 7 (2026-09-29) ==")
    s7 = _load_art("expB2", "sensory_echo_l3_h7.json")
    check_true("h7 sensory-echo: artifact names the hidden 7 surrogate and agents",
               "hidden=7" in s7["surrogate"] and "l3_h7_heldout" in s7["agents"])
    check_true("h7 sensory-echo integrity: every regenerated pool bit-matches",
               s7["integrity"]["all_match"] is True)
    check_true("h7 sensory-echo integrity: published 0.737 reproduced (legacy split)",
               s7["integrity"]["target_reproduced"] is True and s7["integrity"]["fold_scheme"] == "legacy")
    check_int("h7 sensory-echo cells (60)", len(s7["cells"]), 60)
    check_int("h7 sensory-echo cells are unique (drift, seed, agent)",
              len({(c["drift"], c["seed"], c["agent"]) for c in s7["cells"]}), 60)
    for key, block in s7["aggregate"].items():
        for met, v in block.items():
            check(f"h7 sensory-echo {key} {met}: stored mean reproduces per-seed mean",
                  float(np.mean(v["per_seed"])), float(v["mean"]))
    sv7 = s7["aggregate"]["d=0.45 survival"]
    check("h7 10.4.2 survival target reproduces 0.737", sv7["target"]["mean"], 0.737)
    check("h7 10.4.2 survival obs_trace_only (0.697)", sv7["obs_trace_only"]["mean"], 0.697)
    check("h7 10.4.2 survival resid_trace, seven-channel (0.739)", sv7["resid_trace"]["mean"], 0.739)
    check("h7 10.4.2 survival resid_obs (0.684)", sv7["resid_obs"]["mean"], 0.684)
    lo, hi = t_ci(sv7["resid_obs"]["per_seed"])
    check("h7 10.4.2 survival resid_obs t90 lo (0.627)", lo, 0.627)
    check("h7 10.4.2 survival resid_obs t90 hi (0.741)", hi, 0.741)
    check_int("h7 10.4.2 survival resid_obs seeds >= 0.65 (7)", sv7["resid_obs"]["n_ge_065"], 7)
    check("h7 10.4.2 survival resid_obs_int (0.695)", sv7["resid_obs_int"]["mean"], 0.695)
    check("h7 10.4.2 survival resid_obs_beh (0.671)", sv7["resid_obs_beh"]["mean"], 0.671)
    lo, hi = t_ci(sv7["resid_obs_beh"]["per_seed"])
    check("h7 10.4.2 survival resid_obs_beh t90 lo (0.618)", lo, 0.618)
    check("h7 10.4.2 survival resid_obs_beh t90 hi (0.725)", hi, 0.725)
    p7o = s7["aggregate"]["d=0.45 predictor"]["resid_obs"]["mean"]
    check("h7 10.4.2 predictor resid_obs (0.589)", p7o, 0.589)
    check("h7 10.4.2 predictor raw target (0.714)", s7["aggregate"]["d=0.45 predictor"]["target"]["mean"], 0.714)
    check("h7 10.4.2 untrained resid_obs (0.549)", s7["aggregate"]["d=0.45 untrained"]["resid_obs"]["mean"], 0.549)
    check("h7 10.4.2 untrained obs_trace_only (0.643)", s7["aggregate"]["d=0.45 untrained"]["obs_trace_only"]["mean"], 0.643)
    check("h7 10.4.2 survival-minus-predictor lead under the sensory control (+0.095)",
          sv7["resid_obs"]["mean"] - p7o, 0.095)
    check("h7 10.4.2 drift-0 survival resid_obs floor (0.504)", s7["aggregate"]["d=0.00 survival"]["resid_obs"]["mean"], 0.504)
    check("h7 10.4.2 drift-0 untrained resid_obs floor (0.472)", s7["aggregate"]["d=0.00 untrained"]["resid_obs"]["mean"], 0.472)
    d7 = s7["decision"]
    check_true("h7 10.4.2 frozen rule passes both clauses", bool(d7["pass_bar"] and d7["pass_margin"]))
    check_true("h7 10.4.2 t-CI lower bound below the bar, as stated", d7["survival_t90"][0] < BAR)

    print("\n== FINDINGS methods note 8: explicit-fold re-score of the saved dumps (2026-09-29) ==")
    def _fold_art(name):
        with open(os.path.join(ARTROOT, "fold_rescore", name), encoding="utf-8") as fh:
            return json.load(fh)
    fr = {h: _fold_art(f"l3_h{h}_traces.json") for h in (8, 7)}
    for h in (8, 7):
        check_true(f"fold h{h}: re-scored on the owner's stack (scikit-learn 1.5.2, numpy 1.26.4)",
                   fr[h]["stack"]["sklearn"] == "1.5.2" and fr[h]["stack"]["numpy"] == "1.26.4")
        check_true(f"fold h{h}: explicit partition is five balanced 22/22 folds",
                   fr[h]["partition_110_110"]["explicit"] == [[22, 22]] * 5)
        check_true(f"fold h{h}: legacy partition on this stack is (22,22)x3, (21,23), (23,21)",
                   fr[h]["partition_110_110"]["legacy"] == [[22, 22], [22, 22], [22, 22], [21, 23], [23, 21]])
        check_int(f"fold h{h}: 60 cells re-scored", int(fr[h]["n_cells"]), 60)
    # legacy column reproduces the published record; explicit column is the re-score
    published = {8: {"target": (0.752, 0.774), "resid_trace": (0.726, 0.750)},
                 7: {"target": (0.737, 0.740), "resid_trace": (0.722, 0.725)}}
    for h, mets in published.items():
        for met, (leg, exp) in mets.items():
            blk = fr[h]["aggregate"]["d=0.45 survival"][met]
            tol = 1e-3 if (h, met) == (8, "target") else TOL   # 0.7525 is the exact 3-dp midpoint
            check(f"fold h{h} survival {met} legacy reproduces published ({leg})", blk["legacy"]["mean"], leg, tol=tol)
            check(f"fold h{h} survival {met} explicit ({exp})", blk["explicit"]["mean"], exp)
            check(f"fold h{h} survival {met}: stored shift = explicit - legacy",
                  blk["mean_shift"], blk["explicit"]["mean"] - blk["legacy"]["mean"])
            for scheme in ("legacy", "explicit"):
                per_seed = [c[scheme][met] for c in fr[h]["cells"]
                            if c["agent"] == "survival" and float(c["drift"]) == 0.45]
                check_int(f"fold h{h} survival {met} {scheme}: ten cells", len(per_seed), 10)
                check(f"fold h{h} survival {met} {scheme}: mean reproduces from cells", float(np.mean(per_seed)), blk[scheme]["mean"])
                lo, hi = t_ci(per_seed)
                check(f"fold h{h} survival {met} {scheme} t90 lo reproduces", lo, blk[scheme]["t90"][0])
                check(f"fold h{h} survival {met} {scheme} t90 hi reproduces", hi, blk[scheme]["t90"][1])
    # no pre-registered verdict moves on the two published organism runs
    s8 = fr[8]["aggregate"]["d=0.45 survival"]["target"]["explicit"]
    check_true("fold h8: explicit survival still clears the bar with the t-CI above it",
               s8["mean"] >= 0.65 and s8["t90"][0] > 0.65)
    p8 = fr[8]["aggregate"]["d=0.45 predictor"]["target"]["explicit"]["mean"]
    u8 = fr[8]["aggregate"]["d=0.45 untrained"]["target"]["explicit"]["mean"]
    check_true("fold h8: explicit survival keeps both margins (> predictor + 0.05, > untrained + 0.05)",
               s8["mean"] > p8 + 0.05 and s8["mean"] > u8 + 0.05)
    s7 = fr[7]["aggregate"]["d=0.45 survival"]["target"]["explicit"]
    p7 = fr[7]["aggregate"]["d=0.45 predictor"]["target"]["explicit"]["mean"]
    check_true("fold h7: explicit survival still clears the bar", s7["mean"] >= 0.65 and s7["t90"][0] > 0.65)
    check_true("fold h7: survival-minus-predictor lead still under the 0.05 margin (10.5 verdict unchanged)",
               (s7["mean"] - p7) < 0.05)
    check("fold h7: explicit lead over predictor (+0.037; raw 0.0365 is the 3-dp midpoint)", s7["mean"] - p7, 0.037, tol=1e-3)
    check_true("folds.py explicit references match the re-score (8: 0.774, 7: 0.740)",
               abs(round(s8["mean"], 3) - 0.774) < 1e-9 and abs(round(s7["mean"], 3) - 0.740) < 1e-9)
    # common garden, matched pair, gate 0 under both schemes
    cg = {(h, sch): _fold_art(f"cg_h{h}_{sch}.json")["aggregate"] for h in (8, 7) for sch in ("legacy", "explicit")}
    check("fold cg h8 legacy reproduces the forward tail (0.666)", cg[(8, "legacy")]["d0.45_survival"]["cg_tail_mean"], 0.666)
    check("fold cg h7 legacy reproduces the reverse tail (0.684)", cg[(7, "legacy")]["d0.45_survival"]["cg_tail_mean"], 0.684)
    check("fold cg h8 explicit forward tail (0.677)", cg[(8, "explicit")]["d0.45_survival"]["cg_tail_mean"], 0.677)
    check("fold cg h7 explicit reverse tail (0.677)", cg[(7, "explicit")]["d0.45_survival"]["cg_tail_mean"], 0.677)
    for h in (8, 7):
        a = cg[(h, "explicit")]
        check_true(f"fold cg h{h} explicit: both frozen clauses still pass",
                   a["d0.45_survival"]["cg_tail_mean"] >= 0.65
                   and a["d0.45_survival"]["cg_tail_mean"] > a["d0.45_untrained"]["cg_tail_mean"] + 0.05)
    for h, leg, exp in ((8, 0.893, 0.903), (7, 0.855, 0.848)):
        mpl = _fold_art(f"mp_h{h}_legacy.json")["aggregate"]["d0.45_survival"]
        mpe = _fold_art(f"mp_h{h}_explicit.json")["aggregate"]["d0.45_survival"]
        check_true(f"fold mp h{h} legacy reproduces every stored value", bool(mpl["all_reproduce_stored"]))
        check(f"fold mp h{h} legacy survival ({leg})", mpl["mp_fixed_mean"], leg)
        check(f"fold mp h{h} explicit survival ({exp})", mpe["mp_fixed_mean"], exp)
    g0 = {sch: {int(r["hidden"]): r for r in _fold_art(f"gate0_h8h7_{sch}.json")["rows"]} for sch in ("legacy", "explicit")}
    check("fold gate0 legacy h8 oracle reproduces (0.928)", g0["legacy"][8]["oracle_auroc"], 0.928)
    check("fold gate0 legacy h8 floor reproduces (0.482)", g0["legacy"][8]["floor"], 0.482)
    check("fold gate0 legacy h7 oracle reproduces (0.922)", g0["legacy"][7]["oracle_auroc"], 0.922)
    check("fold gate0 legacy h7 floor reproduces (0.566)", g0["legacy"][7]["floor"], 0.566)
    check_true("fold gate0 legacy: both capacities pass", g0["legacy"][8]["passes_gate0"] and g0["legacy"][7]["passes_gate0"])
    check("fold gate0 explicit h8 floor (0.508)", g0["explicit"][8]["floor"], 0.508)
    check_true("fold gate0 explicit h8 passes", bool(g0["explicit"][8]["passes_gate0"]))
    check("section 16: gate0 explicit h7 floor (0.615)", g0["explicit"][7]["floor"], 0.615)
    check("section 16: gate0 explicit h7 oracle still in band (0.918)", g0["explicit"][7]["oracle_auroc"], 0.918)
    check_true("section 16: gate0 explicit h7 FAILS on the floor clause",
               (not g0["explicit"][7]["passes_gate0"]) and (not g0["explicit"][7]["floor_ok"])
               and abs(g0["explicit"][7]["floor"] - 0.5) >= 0.1)
    u7e = fr[7]["aggregate"]["d=0.45 untrained"]["target"]["explicit"]["mean"]
    check_true("section 16: h7 organism-run floor under explicit inside the tolerance by < 0.001",
               abs(u7e - 0.5) < 0.1 and (0.1 - abs(u7e - 0.5)) < 0.001)
    u7_cells = {sch: [c[sch]["target"] for c in fr[7]["cells"] if c["agent"] == "untrained" and float(c["drift"]) == 0.45]
                for sch in ("legacy", "explicit")}
    check_int("section 16: h7 seeds with floor >= 0.6 under explicit (5)", sum(x >= 0.6 for x in u7_cells["explicit"]), 5)
    check_int("section 16: h7 seeds with floor >= 0.6 under legacy (3)", sum(x >= 0.6 for x in u7_cells["legacy"]), 3)
    check_true("FINDINGS carries section 16",
               "## 16. Fold-split sensitivity" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"),
                                                      encoding="utf-8").read())

    print("\n== FINDINGS section 16 addendum: L0 equivalence gate under the explicit split (2026-09-29) ==")
    l0 = {}
    for name in ("l3_h8_traces", "l3_h7_traces", "l3_h4_traces", "l3_n10", "l1_heldout"):
        cells0 = [c for c in _fold_art(f"{name}.json")["cells"]
                  if c["agent"] == "survival" and float(c["drift"]) == 0.0]
        check_int(f"L0 {name}: drift-0 survival cells (10)", len(cells0), 10)
        for sch in ("legacy", "explicit"):
            vals = [float(c[sch]["target"]) for c in cells0]
            l0[(name, sch)] = (vals, equivalence_test(vals, 0.5, margin=0.05), rope_test(vals))
    for name in ("l3_h8_traces", "l3_h7_traces", "l3_h4_traces", "l3_n10"):
        v_l, t_l, r_l = l0[(name, "legacy")]
        v_e, t_e, r_e = l0[(name, "explicit")]
        check(f"L0 {name} legacy mean (0.517)", float(np.mean(v_l)), 0.517)
        check(f"L0 {name} legacy TOST p (0.010)", t_l.p_value, 0.010)
        check(f"L0 {name} explicit mean (0.539)", float(np.mean(v_e)), 0.539)
        check(f"L0 {name} explicit TOST p (0.207)", t_e.p_value, 0.207)
        check(f"L0 {name} explicit ROPE P (0.816)", r_e.p_in_rope, 0.816)
        check_true(f"L0 {name}: equivalent under legacy, not shown under explicit",
                   bool(t_l.equivalent) and bool(r_l.accept) and (not t_e.equivalent) and (not r_e.accept))
    check_true("L0: the four L3 dump sets share one set of drift-0 cells (surrogate off at drift 0)",
               all(l0[(n, "explicit")][0] == l0[("l3_h8_traces", "explicit")][0]
                   for n in ("l3_h7_traces", "l3_h4_traces", "l3_n10")))
    v_l, t_l, r_l = l0[("l1_heldout", "legacy")]
    v_e, t_e, r_e = l0[("l1_heldout", "explicit")]
    check("L0 L1 organism legacy mean (0.522)", float(np.mean(v_l)), 0.522)
    check("L0 L1 organism legacy TOST p (0.029)", t_l.p_value, 0.029)
    check("L0 L1 organism explicit mean (0.546)", float(np.mean(v_e)), 0.546)
    check("L0 L1 organism explicit TOST p (0.374)", t_e.p_value, 0.374)
    check("L0 L1 organism explicit ROPE P (0.628)", r_e.p_in_rope, 0.628)

    print("\n== FINDINGS 10.9 addendum: hidden 8 new-seed run (G seed 2, 2026-09-29) ==")
    s2 = _load_art("expB2", "second_seed_l3_h8_gseed2_gpu.json")
    _fd = open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"), encoding="utf-8").read()
    check_true("FINDINGS carries the section 16 L0 equivalence addendum",
               "**Addendum (2026-09-29, found while promoting the hidden 8 new-seed run)" in _fd)
    check_true("FINDINGS carries the hidden-8 new-seed paragraph",
               "*Hidden-8 new-seed run (2026-09-29).*" in _fd)
    check_true("h8-seed2 execution names the owner's GPU machine", s2["execution"].startswith("owner's GPU"))
    check_int("h8-seed2 gate 0 G seed (2)", int(s2["gate0_calibration"]["g_seed"]), 2)
    check_int("h8-seed2 gate 0 selected hidden (8)", int(s2["gate0_calibration"]["selected_hidden"]), 8)
    g8 = s2["gate0_calibration"]["rows"][0]
    check("h8-seed2 gate 0 oracle (0.939)", g8["oracle_auroc"], 0.939)
    check("h8-seed2 gate 0 floor (0.540)", g8["floor"], 0.540)
    check_true("h8-seed2 gate 0 passes with a clean mechanical floor",
               bool(g8["passes_gate0"]) and bool(g8["mech_leak_pass"]) and bool(g8["in_band"]))
    for arm, ref in (("survival", 0.676), ("predictor", 0.627), ("untrained", 0.550)):
        blk = s2["arms"][s2["dmax"]][arm]["pool_target"]
        check(f"h8-seed2 {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"h8-seed2 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv2 = s2["arms"][s2["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv2["per_seed"])
    check("h8-seed2 survival t90 lo (0.636)", lo, 0.636)
    check("h8-seed2 survival t90 hi (0.717)", hi, 0.717)
    check_int("h8-seed2 survival seeds >= 0.65 (6)", sv2["n_ge_065"], 6)
    rt2 = s2["behavior_audit"]["survival"]["resid_trace"]
    check("h8-seed2 survival resid_trace (0.660)", rt2["mean"], 0.660)
    lo, hi = t_ci(rt2["per_seed"])
    check("h8-seed2 survival resid_trace t90 lo (0.631)", lo, 0.631)
    check("h8-seed2 survival resid_trace t90 hi (0.689)", hi, 0.689)
    check("h8-seed2 survival behavior_trace_only (0.745)", s2["behavior_audit"]["survival"]["behavior_trace_only"]["mean"], 0.745)
    ba2 = s2["behavior_audit"]
    check("h8-seed2 predictor behavior_trace_only (0.719)", ba2["predictor"]["behavior_trace_only"]["mean"], 0.719)
    check("h8-seed2 untrained behavior_trace_only (0.683)", ba2["untrained"]["behavior_trace_only"]["mean"], 0.683)
    check("h8-seed2 predictor resid_trace (0.606)", ba2["predictor"]["resid_trace"]["mean"], 0.606)
    check("h8-seed2 untrained resid_trace (0.563)", ba2["untrained"]["resid_trace"]["mean"], 0.563)
    check("h8-seed2 resid_trace lead over predictor (+0.054)", rt2["mean"] - ba2["predictor"]["resid_trace"]["mean"], 0.054)
    check("h8-seed2 resid_trace lead over untrained (+0.098)", rt2["mean"] - ba2["untrained"]["resid_trace"]["mean"], 0.098)
    check_int("h8-seed2 survival resid_trace seeds >= 0.65 (5)", sum(v >= BAR for v in rt2["per_seed"]), 5)
    d2 = s2["decision"]
    check("h8-seed2 lead over predictor (+0.0491)", d2["lead_over_predictor"], 0.0491, tol=0.0001)
    check("h8-seed2 lead over untrained (+0.126)", d2["lead_over_untrained"], 0.126)
    check_true("h8-seed2 rule: mean clears the bar, t-CI does not, predictor margin misses by < 0.001",
               bool(d2["pass_bar"]) and (not d2["t90_excludes_bar"]) and (not d2["pass_margin_predictor"])
               and bool(d2["pass_margin_untrained"]) and (0.05 - d2["lead_over_predictor"]) < 0.001)
    lead2 = np.asarray(sv2["per_seed"]) - np.asarray(s2["arms"][s2["dmax"]]["predictor"]["pool_target"]["per_seed"])
    lo, hi = t_ci(list(lead2))
    check("h8-seed2 paired lead over predictor t90 lo (-0.003)", lo, -0.003)
    check("h8-seed2 paired lead over predictor t90 hi (+0.101)", hi, 0.101)
    check_true("h8-seed2 paired lead over predictor includes zero", lo < 0 < hi)
    lead2u = np.asarray(sv2["per_seed"]) - np.asarray(s2["arms"][s2["dmax"]]["untrained"]["pool_target"]["per_seed"])
    lo, hi = t_ci(list(lead2u))
    check("h8-seed2 paired lead over untrained t90 lo (+0.074)", lo, 0.074)
    check("h8-seed2 paired lead over untrained t90 hi (+0.179)", hi, 0.179)
    check_true("h8-seed2 remaining gates pass (engagement, leak, floor, deaths)",
               s2["gates"]["pool_leak_clean_all"] and s2["gates"]["untrained_floor_ok"]
               and s2["gates"]["deaths_total"] == 0
               and all(v["pass"] == v["n"] for v in s2["gates"]["engagement"].values()))
    check_true("h8-seed2 L0 gate open, same drift-0 cells as the hidden 10 GPU run",
               (not s2["gates"]["l0_tost"]["equivalent"])
               and s2["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]
               == sg["arms"]["0.0"]["survival"]["pool_target"]["per_seed"])
    cpu0 = _load_art("expB2", "device_control_l3_h8_wm_cpu.json")
    dcs = np.asarray(sv2["per_seed"]) - np.asarray(cpu0["arms"][cpu0["dmax"]]["survival"]["pool_target"]["per_seed"])
    check("h8-seed2 paired vs CPU device control, survival (-0.053)", float(dcs.mean()), -0.053)
    lo, hi = t_ci(list(dcs))
    check("h8-seed2 paired vs CPU device control t90 lo (-0.124)", lo, -0.124)
    check("h8-seed2 paired vs CPU device control t90 hi (+0.018)", hi, 0.018)
    seed0 = {c["seed"]: float(c["explicit"]["target"]) for c in fr[8]["cells"]
             if c["agent"] == "survival" and float(c["drift"]) == 0.45}
    dse = np.asarray(sv2["per_seed"]) - np.asarray([seed0[i] for i in range(10)])
    check("h8-seed2 paired vs explicit seed-0 headline, survival (-0.097)", float(dse.mean()), -0.097)
    lo, hi = t_ci(list(dse))
    check("h8-seed2 paired vs explicit seed-0 headline t90 lo (-0.140)", lo, -0.140)
    check("h8-seed2 paired vs explicit seed-0 headline t90 hi (-0.055)", hi, -0.055)

    print("\n== FINDINGS 10.8.1: skill-matched model-free baseline (budget 450, 2026-09-30) ==")
    sk = _load_art("expB2", "skill_matched_l3_h8_nowm_u450.json")
    check_true("skill-match execution names the owner's GPU machine", sk["execution"].startswith("owner's GPU"))
    check_true("FINDINGS carries the skill-matched section",
               "### 10.8.1 Skill-matched model-free baseline: no skill-mediation verdict (2026-09-30)" in _fd)
    check_true("FINDINGS records the verdict as no skill-mediation verdict",
               "**no skill-mediation verdict.**" in _fd)
    check_true("FINDINGS keeps 10.8 auxiliary-conditional unchanged",
               "**auxiliary-conditional** verdict therefore stands unchanged" in _fd)
    for arm, ref in (("survival", 0.717), ("predictor", 0.588), ("untrained", 0.513)):
        blk = sk["arms"][sk["dmax"]][arm]["pool_target"]
        check(f"skill-match {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"skill-match {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    svk = sk["arms"][sk["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(svk["per_seed"])
    check("skill-match survival t90 lo (0.679)", lo, 0.679)
    check("skill-match survival t90 hi (0.755)", hi, 0.755)
    check_int("skill-match survival seeds >= 0.65 (8)", svk["n_ge_065"], 8)
    rtk = sk["behavior_audit"]["survival"]["resid_trace"]
    check("skill-match survival resid_trace (0.724)", rtk["mean"], 0.724)
    lo, hi = t_ci(rtk["per_seed"])
    check("skill-match survival resid_trace t90 lo (0.690)", lo, 0.690)
    check("skill-match survival resid_trace t90 hi (0.757)", hi, 0.757)
    check_int("skill-match survival resid_trace seeds >= 0.65 (9)", sum(v >= BAR for v in rtk["per_seed"]), 9)
    bak = sk["behavior_audit"]
    check("skill-match predictor resid_trace (0.590)", bak["predictor"]["resid_trace"]["mean"], 0.590)
    check("skill-match untrained resid_trace (0.539)", bak["untrained"]["resid_trace"]["mean"], 0.539)
    check("skill-match survival behavior_trace_only (0.838)", bak["survival"]["behavior_trace_only"]["mean"], 0.838)
    dk = sk["decision"]
    check("skill-match lead over predictor (+0.129)", dk["lead_over_predictor"], 0.129)
    check("skill-match lead over untrained (+0.204)", dk["lead_over_untrained"], 0.204)
    check_true("skill-match probe passes bar, t-CI excludes it, and both margins pass",
               bool(dk["pass_bar"]) and bool(dk["t90_excludes_bar"])
               and bool(dk["pass_margin_predictor"]) and bool(dk["pass_margin_untrained"]))
    # The frozen match clause: an overshoot voids the mediation inference even though
    # every probe clause above passes. Both halves must stay true together.
    sm = sk["skill_match"]
    check("skill-match reference return (-0.219)", sm["reference_return"], -0.219)
    check("skill-match window lo (-0.269)", sm["window"][0], -0.269)
    check("skill-match window hi (-0.169)", sm["window"][1], -0.169)
    check("skill-match stage-2 return R (-0.1559)", sm["match_return"], -0.1559, tol=0.0001)
    check("skill-match R reproduces the per-seed mean", float(np.mean(sm["per_seed_returns"])), sm["match_return"])
    check_int("skill-match stage-2 seeds (10)", sm["n_seeds"], 10)
    check("skill-match stage-2 return se (0.096)", sm["return_se"], 0.096)
    check("skill-match overshoot past the window (+0.0131)", sm["overshoot_past_window"], 0.0131, tol=0.0001)
    check("skill-match overshoot in standard errors (0.14)", sm["overshoot_in_se"], 0.14, tol=0.005)
    check("skill-match deviation from reference (+0.0631)", sm["deviation_from_reference"], 0.0631, tol=0.0001)
    check_true("skill-match R is outside the window and above it",
               (not sm["in_window"]) and sm["match_return"] > sm["window"][1])
    check_true("skill-match probe does not read below the bar", not sm["probe_below_bar"])
    check_true("skill-match frozen rule gives no skill-mediation verdict",
               sm["verdict"].startswith("NO SKILL-MEDIATION VERDICT"))
    check_true("skill-match gates pass except L0 (engagement, leak, floor, deaths)",
               sk["gates"]["pool_leak_clean_all"] and sk["gates"]["untrained_floor_ok"]
               and sk["gates"]["deaths_total"] == 0
               and all(v["pass"] == v["n"] for v in sk["gates"]["engagement"].values()))
    check("skill-match speed positive control min (0.837)", sk["gates"]["speed_min"], 0.837)
    check("skill-match L0 survival mean (0.539)", sk["gates"]["l0_survival_mean"], 0.539)
    check("skill-match L0 TOST p (0.285)", sk["gates"]["l0_tost"]["p_value"], 0.285)
    check("skill-match L0 ROPE P (0.736)", sk["gates"]["l0_rope"]["p_in_rope"], 0.736)
    check_true("skill-match L0 gate recorded open, not accepted",
               (not sk["gates"]["l0_tost"]["equivalent"]) and (not sk["gates"]["l0_rope"]["accept"]))
    # Determinism cross-check: at drift 0 the untrained and predictor arms are objective
    # -identical to the hidden-8 new-seed GPU run, while the survival arm must differ
    # because this arm drops the decoder and trains on a longer budget.
    for arm in ("untrained", "predictor"):
        check_true(f"skill-match drift-0 {arm} arm bit-identical to the h8-seed2 GPU run",
                   sk["arms"]["0.0"][arm]["pool_target"]["per_seed"]
                   == s2["arms"]["0.0"][arm]["pool_target"]["per_seed"])
    check_true("skill-match drift-0 survival arm differs from the h8-seed2 GPU run",
               sk["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]
               != s2["arms"]["0.0"]["survival"]["pool_target"]["per_seed"])
    check_true("skill-match survival clears the budget-300 no-auxiliary run (0.601)",
               svk["mean"] > _load_art("expB2", "arch_baseline_l3_h8_nowm.json")["decision"]["survival"])

    # ---- FINDINGS 10.8 device control (promoted 2026-09-28) --------------------
    print("\n== FINDINGS 10.8: device control (decoder-carrying arm on the CPU sandbox) ==")
    dc = _load_art("expB2", "device_control_l3_h8_wm_cpu.json")
    for arm, ref in (("survival", 0.730), ("predictor", 0.589), ("untrained", 0.523)):
        blk = dc["arms"][dc["dmax"]][arm]["pool_target"]
        check(f"10.8 device control {arm} pooled target ({ref})", blk["mean"], ref)
        check(f"10.8 device control {arm}: stored mean reproduces per-seed mean",
              float(np.mean(blk["per_seed"])), blk["mean"])
    sv = dc["arms"][dc["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.8 device control survival t90 lo (0.668)", lo, 0.668)
    check("10.8 device control survival t90 hi (0.791)", hi, 0.791)
    check_int("10.8 device control survival seeds >= 0.65 (8)", sv["n_ge_065"], 8)
    rt = dc["behavior_audit"]["survival"]["resid_trace"]
    check("10.8 device control survival resid_trace (0.710)", rt["mean"], 0.710)
    lo, hi = t_ci(rt["per_seed"])
    check("10.8 device control survival resid_trace t90 lo (0.656)", lo, 0.656)
    check("10.8 device control survival resid_trace t90 hi (0.764)", hi, 0.764)
    check_true("10.8 device control all gates pass",
               dc["gates"]["l0_tost"]["equivalent"] and dc["gates"]["l0_rope"]["accept"]
               and dc["gates"]["pool_leak_clean_all"] and dc["gates"]["untrained_floor_ok"]
               and all(v["pass"] == v["n"] for v in dc["gates"]["engagement"].values()))
    # the rule compares against the no-auxiliary run promoted above; recompute from both
    nowm = ab["arms"][ab["dmax"]]["survival"]["pool_target"]["per_seed"]
    wm = sv["per_seed"]
    check("10.8 device control lead over no-auxiliary (+0.128)", float(np.mean(wm) - np.mean(nowm)), 0.128)
    lo, hi = t_ci([a - b for a, b in zip(wm, nowm)])
    check("10.8 device control paired lead t90 lo (+0.089)", lo, 0.089)
    check("10.8 device control paired lead t90 hi (+0.168)", hi, 0.168)
    check_true("10.8 device control rule: device not the cause",
               dc["device_control"]["verdict"].startswith("DEVICE NOT THE CAUSE")
               and dc["device_control"]["t90_excludes_bar"] and dc["device_control"]["pass_lead"])
    check_true("10.8 device control predictor arm bit-identical to the no-auxiliary run",
               dc["arms"][dc["dmax"]]["predictor"]["pool_target"]["per_seed"]
               == ab["arms"][ab["dmax"]]["predictor"]["pool_target"]["per_seed"])

    # ---- FINDINGS 10.4.2 addendum: nonlinear joint control (2026-09-28) ------
    print("\n== FINDINGS 10.4.2 addendum: nonlinear joint control ==")
    nl = _load_art("expB2", "sensory_echo_l3_h8_mlp.json")
    check_true("nonlinear control integrity: 30/30 bit-match, 0.752 reproduced",
               nl["integrity"]["all_match"] is True and nl["integrity"]["target_reproduced"] is True
               and len(nl["cells"]) == 30)
    sv = nl["aggregate"]["d=0.45 survival"]
    check("nonlinear survival resid_obs_beh_mlp (0.654)", sv["resid_obs_beh_mlp"]["mean"], 0.654)
    lo, hi = t_ci(sv["resid_obs_beh_mlp"]["per_seed"])
    check("nonlinear survival resid_obs_beh_mlp t90 lo (0.621)", lo, 0.621)
    check("nonlinear survival resid_obs_beh_mlp t90 hi (0.687)", hi, 0.687)
    check_int("nonlinear survival resid_obs_beh_mlp seeds >= 0.65 (6)", sv["resid_obs_beh_mlp"]["n_ge_065"], 6)
    check("nonlinear survival resid_obs_mlp (0.653)", sv["resid_obs_mlp"]["mean"], 0.653)
    check("nonlinear untrained resid_obs_beh_mlp (0.509)",
          nl["aggregate"]["d=0.45 untrained"]["resid_obs_beh_mlp"]["mean"], 0.509)
    check("nonlinear predictor resid_obs_beh_mlp (0.542)",
          nl["aggregate"]["d=0.45 predictor"]["resid_obs_beh_mlp"]["mean"], 0.542)
    check("nonlinear run reproduces the linear joint control (0.670)", sv["resid_obs_beh"]["mean"], 0.670)
    check_true("nonlinear frozen rule passes at the mean",
               bool(nl["decision_nonlinear"]["pass_bar"] and nl["decision_nonlinear"]["pass_margin"]))
    check_true("FINDINGS carries the nonlinear joint control addendum",
               "**Nonlinear joint control (2026-09-28" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"),
                                                          encoding="utf-8").read())

    # ---- FINDINGS 15: matched-handicap oracle ceilings (promoted 2026-09-27) --
    print("\n== FINDINGS 15: matched-handicap oracle ceilings across rungs ==")
    mh = _load_art("expA", "l2_inconfig_oracle.json")
    cells = {(r["cell"], r["sigma_meas"]): r for r in mh["rows"]}
    for cell, sig, ref in (("ar1", 0.02, 0.646), ("regime", 0.02, 0.618), ("ar1", 0.0002, 1.000),
                           ("regime", 0.0002, 1.000), ("ar1", 0.01, 0.810), ("regime", 0.005, 0.854),
                           ("l3_h8_seed0", 0.02, 0.928), ("l3_h7_seed0", 0.02, 0.922),
                           ("l3_h8_seed0", 0.05, 0.674), ("ar1", 0.05, 0.588)):
        check(f"15 {cell} @ sigma {sig} ({ref})", cells[(cell, sig)]["oracle_auroc"], ref)
    check_true("15 mechanical leakage clean in every cell", all(r["leakage_pass"] for r in mh["rows"]))
    check("15 L0 anchor mean over noise seeds (0.475)", mh["l0_anchor_noise_seeds"]["mean"], 0.475)
    check_true("15 L3 at least as detectable as L2 ar1 at every sigma",
               all(cells[("l3_h8_seed0", s)]["oracle_auroc"] >= cells[("ar1", s)]["oracle_auroc"] - 1e-9
                   for s in (0.0002, 0.001, 0.005, 0.01, 0.02, 0.05)))
    check_true("FINDINGS carries the section 15 subsection",
               "## 15. Matched-handicap" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "FINDINGS.md"),
                                                 encoding="utf-8").read())

    # ---- derived-doc resolution guard -----------------------------------
    # The reactive-vs-persistent reading was PROVISIONAL until the section 10.6
    # re-score; it is now RESOLVED (FINDINGS 10.6.1, 2026-07-19): the corrected
    # common-garden control passes the frozen rule on both directions. The 2026-10
    # revision narrowed the reading to "prefix condition remains decodable after
    # restoring authentic dynamics", since the control cannot separate memory from a
    # physical footprint of the prefix. The public-facing
    # derived docs are hand-maintained and drifted stale before (audit fault:
    # index.html once stated the reading as final), so pin the resolved
    # references here and forbid regression to the provisional/reactive-only
    # wording.
    print("\n== derived-doc resolution guard (10.6.1 persistent reading) ==")
    root = os.path.join(os.path.dirname(__file__), "..")

    def _read(relpath: str) -> str:
        with open(os.path.join(root, relpath), encoding="utf-8") as fh:
            return fh.read()

    for relpath, needle, label in [
        ("index.html", "10.6.1",
         "index.html points at the 10.6.1 resolution"),
        ("index.html", "remains decodable after restoring authentic dynamics",
         "index.html carries the narrowed common-garden reading (revision 2026-10)"),
        ("CITATION.cff", "remains decodable after restoring authentic dynamics",
         "CITATION.cff carries the narrowed common-garden reading (revision 2026-10)"),
        ("CITATION.cff", "10.6.1",
         "CITATION.cff points at the 10.6.1 resolution"),
        ("README.md", "10.6.1",
         "README points at the 10.6.1 resolution"),
        ("docs/FINDINGS.md", "### 10.6.1",
         "FINDINGS carries the 10.6.1 resolution subsection"),
        ("docs/FINDINGS.md", "RE-SCORE RESOLVED",
         "FINDINGS 10.6 banner is marked RESOLVED"),
        ("docs/PAPER_OUTLINE.md", "10.6.1",
         "PAPER_OUTLINE points at the 10.6.1 resolution"),
    ]:
        check_true(label, needle in _read(relpath))
    # forbid regression: the provisional qualifier must not reappear as the
    # current verdict in the hand-maintained public pages.
    for relpath, banned in [("index.html", "PROVISIONAL"),
                            ("CITATION.cff", "provisional pending")]:
        check_true(f"{relpath} no longer carries the provisional qualifier",
                   banned not in _read(relpath))
    # Revision 2026-10 (docs/REVISION_2026-10.md, step 14): the common garden does not
    # separate memory from the prefix's physical footprint, and a behavior control removes
    # only what its basis can express. The public pages must not say otherwise.
    for relpath in ("index.html", "CITATION.cff", "README.md"):
        _t = _read(relpath)
        for banned in ("persistent stored", "behavior-independent", "behaviour-independent",
                       "H2 confirmed", "H3 resolves negative"):
            check_true(f"{relpath} no longer says '{banned}'", banned not in _t)
    # FINDINGS 17.6 withdrew the survival-specific reading; the phrase may appear on the public
    # pages only in the sentence that withdraws it.
    for relpath in ("index.html", "CITATION.cff", "README.md"):
        _t = " ".join(_read(relpath).split())
        _bad = [m.start() for m in re.finditer(r"survival-specific", _t)
                if "withdrawn" not in _t[max(0, m.start() - 120): m.start() + 120]]
        check_true(f"{relpath} uses 'survival-specific' only to withdraw it", not _bad)
    for relpath, needle, label in [
        ("README.md", "docs/CORRECTIONS.md", "README points at the corrections record"),
        ("index.html", "docs/CORRECTIONS.md", "index.html points at the corrections record"),
        ("CITATION.cff", "docs/CORRECTIONS.md", "CITATION.cff points at the corrections record"),
        ("README.md", "VariBAD", "README compares against meta-RL"),
        ("index.html", "VariBAD", "index.html compares against meta-RL"),
    ]:
        check_true(label, needle in _read(relpath))
    # and the old reactive-only claim must not reappear in index.html.
    idx = _read("index.html")
    for phrase in ("not a persistent stored representation",
                   "not a stored representation"):
        check_true(f"index.html no longer states the reactive-only claim '{phrase}'",
                   len(_find_all(idx, phrase)) == 0)

    # ---- derived-doc guard: the 2026-09 boundary checks (10.8, 10.9, 15) ------
    # The public pages once said the signal "transfers across fingerprint
    # instances" and credited survival alone; 10.8 (with its device control) and
    # 10.9 narrowed both, and section 15 withdrew the L2-versus-L3 ceiling
    # contrast. Pin the narrowed wording and forbid the old one.
    print("\n== derived-doc guard (10.8 / 10.9 boundary checks, section 15) ==")
    for relpath, needle, label in [
        ("index.html", "0.601", "index.html carries the no-auxiliary result"),
        ("index.html", "0.730", "index.html carries the device-control result"),
        ("index.html", "0.639", "index.html carries the second-instance result"),
        ("README.md", "§10.8", "README points at the 10.8 boundary check"),
        ("README.md", "§10.9", "README points at the 10.9 boundary check"),
        ("README.md", "§15", "README points at the section 15 matched ceilings"),
        ("CITATION.cff", "next-observation auxiliary", "CITATION.cff names the auxiliary"),
        ("docs/PAPER_OUTLINE.md", "device_control_l3_h8_wm_cpu.json",
         "PAPER_OUTLINE inventories the device control"),
    ]:
        check_true(label, needle in _read(relpath))
    for relpath, banned in [("index.html", "transfers across fingerprint instances"),
                            ("index.html", "subtler of the two tested fingerprints"),
                            ("CITATION.cff", "subtler of two tested fingerprints"),
                            ("docs/PAPER_OUTLINE.md", "L3 encoded by survival only"),
                            ("docs/PAPER_OUTLINE.md", "H3 open pending re-run")]:
        check_true(f"{relpath} no longer says '{banned}'", banned not in _read(relpath))

    # ---- derived-doc guard: the survival-alone phrasing, everywhere it lives ---
    # 10.8 narrowed "encoded by the survival objective, uniquely" on 2026-09-27:
    # strip the next-observation decoder and the same protocol reads 0.601 against
    # 0.730 with it, so survival is necessary and not sufficient. FINDINGS and the
    # paper were corrected the same week because their numbers are pinned here.
    # The prose artifacts were not, and the old sentence survived in three of them
    # until 2026-10-01: the film's end card, the status note's film table, and the
    # plain-language explainer. Nothing failed, because nothing was watching them.
    #
    # These guards are deliberately phrase-level rather than number-level. The
    # failure mode is not a wrong number, it is a correct number under a claim the
    # evidence no longer supports, which no arithmetic check can catch.
    print("\n== derived-doc guard (survival-alone phrasing in the prose artifacts) ==")
    for relpath, needle, label in [
        ("viz/player/brain/brain.js", "Trained to survive and to expect what comes next",
         "film end card names both objectives"),
        ("viz/player/brain/brain.js", "drop the second job and the trace falls below the bar",
         "film end card states the decoder requirement"),
        ("viz/player/brain/brain.js", "Five actions. Three predictions.",
         "film actions chapter still names the prediction heads"),
        ("docs/LEARNING.md", "it is not survival alone",
         "LEARNING.md carries the decoder requirement as its own act"),
        ("docs/LEARNING.md", "0.601",
         "LEARNING.md quotes the no-decoder result"),
        ("docs/LEARNING.md", "prediction side task",
         "LEARNING.md flags the confound where it sets up the three-way comparison"),
    ]:
        check_true(label, needle in _read(relpath))
    # What the film must not SAY, which is not the same as what its source may
    # mention. brain.js carries a comment naming the retired sentence so the next
    # editor knows why it is retired; checking the raw file would fire on that
    # comment and push the explanation out of the code. Strip line comments first
    # and check what actually renders.
    def _code_without_comments(relpath: str) -> str:
        return "\n".join(ln for ln in _read(relpath).splitlines()
                         if not ln.lstrip().startswith("//"))

    for relpath, banned in [
        ("docs/LEARNING.md", "Now, and only now, the survival creature encodes the world"),
        ("docs/LEARNING.md", "snapshot 2026-07-14"),
        ("docs/STATUS_2026-09-27.md", "Three lines credit survival alone"),
        ("index.html", "trained only to survive"),
        ("index.template.html", "trained only to survive"),
    ]:
        check_true(f"{relpath} no longer says '{banned}'", banned not in _read(relpath))
    for banned in ("trained only to survive", "survival alone", "only to survive"):
        check_true(f"the film renders no '{banned}' claim",
                   banned not in _code_without_comments("viz/player/brain/brain.js"))
    # The film ships silent. If a voiced cut is ever committed, the VO script in
    # docs/specs/2026-10-01-two-minds-vo-music.md has to be re-read against 10.8
    # first, so pin the spec's own warning rather than let it be recorded blind.
    check_true("the Two Minds VO spec records that its survival-alone line has a shelf life",
               "The one line with a shelf life" in _read("docs/specs/2026-10-01-two-minds-vo-music.md"))

    # ---- derived-doc guard: sensory echo (10.4.2), H2 (14), Exp C (13.D) -------
    # The site carried Experiment C as "next" for two months after 13.D closed it,
    # and never mentioned the sensory-echo or graded-seam controls. Pin them.
    print("\n== derived-doc guard (10.4.2 sensory echo, 14 H2, 13.D Exp C) ==")
    for relpath, needle, label in [
        ("index.html", "0.731", "index.html carries the sensory-echo control"),
        ("index.html", "0.654", "index.html carries the nonlinear joint control"),
        ("index.html", "0.506", "index.html carries the graded-seam collapse"),
        ("index.html", "0.509", "index.html carries the Exp C final AUROC"),
        ("index.html", "validated null", "index.html states the Exp C verdict"),
    ]:
        check_true(label, needle in _read(relpath))
    for relpath, banned in [("index.html", "EXP C · next"),
                            ("index.html", "no matter how loud the artifact")]:
        check_true(f"{relpath} no longer says '{banned}'", banned not in _read(relpath))

    # ---- section 16: the shared mechanism behind both flipped clauses --------
    # Recompute the legacy-to-explicit shift and the contrasts straight from the
    # re-score artifact, so the section 16 tables cannot drift from the dumps.
    print("")
    print("== section 16: fold shift and contrast invariance ==")
    from collections import defaultdict as _dd
    _h8 = _load_art("fold_rescore", "l3_h8_traces.json")
    _g = _dd(lambda: _dd(list))
    for _c in _h8["cells"]:
        for _k in ("target", "resid_trace"):
            _g[(_c["drift"], _c["agent"], _k)]["leg"].append(_c["legacy"][_k])
            _g[(_c["drift"], _c["agent"], _k)]["exp"].append(_c["explicit"][_k])
    _m = {k: (float(np.mean(v["leg"])), float(np.mean(v["exp"]))) for k, v in _g.items()}
    _deltas = [e - l for l, e in _m.values()]
    check_int("section 16 shift: 12 arm/drift/metric means at hidden 8", len(_deltas), 12)
    check_int("section 16 shift: all 12 move up", sum(d > 0 for d in _deltas), 12)
    check("section 16 shift: mean delta +0.0212", float(np.mean(_deltas)), 0.0212)
    for label, a, da, b, db, metric, expect_leg, expect_exp in [
        ("survival minus L0 floor", "survival", "0.45", "survival", "0.00", "target", 0.2358, 0.2347),
        ("survival minus untrained", "survival", "0.45", "untrained", "0.45", "target", 0.2646, 0.2610),
        ("survival minus predictor", "survival", "0.45", "predictor", "0.45", "target", 0.1792, 0.1857),
        ("survival minus untrained resid", "survival", "0.45", "untrained", "0.45", "resid_trace", 0.2277, 0.2245),
    ]:
        check(f"section 16 contrast {label} (legacy)", _m[(da, a, metric)][0] - _m[(db, b, metric)][0], expect_leg)
        check(f"section 16 contrast {label} (explicit)", _m[(da, a, metric)][1] - _m[(db, b, metric)][1], expect_exp)
    _l0 = _load_art("expB2", "second_seed_l3_h8_gseed2_gpu.json")["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]
    check_int("section 16 L0: ten drift-0 survival seeds", len(_l0), 10)
    check("section 16 L0 mean 0.5393", float(np.mean(_l0)), 0.5393)
    check("section 16 L0 sd 0.0397", float(np.std(_l0, ddof=1)), 0.0397)
    check("section 16 L0 TOST p 0.207", equivalence_test(_l0).p_value, 0.207)

    # ---- section 16 and note 8: the decisions and the comparator convention --
    print("")
    print("== section 16 decisions and the comparator convention (2026-09-30) ==")
    for relpath, needle, label in [
        ("docs/FINDINGS.md", "Twelve of twelve up, mean +0.0212",
         "section 16 states the uniform shift at hidden 8"),
        ("docs/FINDINGS.md", "hidden 7 stays the second in-band capacity",
         "section 16 records the hidden 7 decision"),
        ("docs/FINDINGS.md", "report the clause as open; do not buy it with seeds",
         "section 16 records the L0 decision"),
        ("docs/FINDINGS.md", "Comparator convention (fixed 2026-09-30)",
         "methods note 8 states the comparator convention"),
        ("docs/FINDINGS.md", "the gap is **0.162** rather than 0.140",
         "10.9 compares the GPU re-measure like for like"),
    ]:
        check_true(label, needle in _read(relpath))
    check_true("FINDINGS no longer says 'The gap to 0.752 therefore sits with the instance'",
               "The gap to 0.752 therefore sits with the instance" not in _read("docs/FINDINGS.md"))

    # ---- blind clarification of the skill-match clause (2026-09-30) ----------
    # Stage 1 of the skill-matched baseline overshot the return it was meant to match
    # (+0.0067 against -0.219), which made the stage-2 clause's one-sided vs two-sided
    # reading load-bearing. The reading was fixed asymmetrically while the stage-2 cells
    # directory was still empty. These pins fail if that provenance is edited away.
    print("")
    print("== skill-match clause: blind asymmetric reading (2026-09-30) ==")
    _SKILL_SPEC = "docs/specs/2026-09-29-l3-skill-matched-baseline-design.md"
    for relpath, needle, label in [
        (_SKILL_SPEC, "Amendment (2026-09-30 03:17 UTC): the skill-match clause is asymmetric",
         "skill-matched spec carries the dated amendment"),
        (_SKILL_SPEC, "before any stage 2 cell existed",
         "skill-matched spec records that the reading was fixed blind"),
        (_SKILL_SPEC, "+0.0067", "skill-matched spec carries the stage 1 three-seed mean"),
        (_SKILL_SPEC, "skill-advantaged, not skill-matched",
         "skill-matched spec names the overshoot case"),
        (_SKILL_SPEC, "budget ladder ascends only",
         "skill-matched spec records the ascending-only ladder limitation"),
        ("docs/PREREGISTRATION_L3.md", "BLIND CLARIFICATION OF THE",
         "prereg sec 12 logs the blind clarification"),
        ("docs/PREREGISTRATION_L3.md", "+0.0067",
         "prereg sec 12 carries the stage 1 three-seed mean"),
    ]:
        check_true(label, needle in _read(relpath))

    # ---- FINDINGS methods note 2: engagement margin on committed cells --------
    print("\n== FINDINGS note 2: engagement-margin sweep on committed cells ==")
    import glob as _g2
    engs = list(_load_art("expB2", "bv3_n10_gates.json")["engagement"])
    for run in ("l3_h8_nowm", "l3_h10_gseed1", "l3_h8_wm_cpu"):
        for cp in sorted(_g2.glob(os.path.join(ARTROOT, "reviewer_gaps_runs", run, "cells", "cell_*.json"))):
            with open(cp, encoding="utf-8") as fh:
                engs.append(json.load(fh)["cell"]["eng"])

    def _engaged(e: dict, margin: float) -> bool:
        return (e["trained_return"] >= max(e["random_return"], e["scripted_return"]) + margin
                and e["trained_len"] >= e["random_len"] - 2.0)

    check_int("note 2 committed cells (80)", len(engs), 80)
    for m in (0.05, 0.10, 0.15):
        check_int(f"note 2 cells engaged at margin {m:.2f} (80)", sum(_engaged(e, m) for e in engs), 80)
    for m in (0.20, 0.25, 0.30):
        check_int(f"note 2 cells engaged at margin {m:.2f} (79)", sum(_engaged(e, m) for e in engs), 79)
    check("note 2 tightest committed cell clears by 0.182",
          min(e["trained_return"] - max(e["random_return"], e["scripted_return"]) for e in engs), 0.182)
    # index.html is now GENERATED from index.template.html by scripts/build_index.py,
    # which fills {{...}} placeholders from the artifact-derived site metrics. So instead
    # of pinning bare number strings, regenerate the page and require it to be already up
    # to date: any artifact/headline change must be re-rendered in the same commit or this
    # fails. (Replaces the 2026-07-18 headline string pins with real regeneration.)
    sys.path.insert(0, os.path.dirname(__file__))
    import build_index
    check_true("index.html is regenerable and current (build_index --check)",
               build_index.build(os.path.join(root), check=True) == 0)
    # Revision step 1 (docs/REVISION_2026-10.md): every committed artifact belongs to a run in
    # the results manifest, and the generated manifest pages are current.
    import build_results_manifest
    check_true("results manifest covers every artifact and is current "
               "(build_results_manifest --check)",
               build_results_manifest.main(["--check"]) == 0)
    # Revision step 5: the gate table recomputes L0 and the floor from committed per-seed values.
    import build_gate_table
    check_true("gate table is current (build_gate_table --check)",
               build_gate_table.main(["--check"]) == 0)
    # Revision step 13: margins get intervals of the difference.
    import build_contrast_intervals
    check_true("contrast intervals are current (build_contrast_intervals --check)",
               build_contrast_intervals.main(["--check"]) == 0)
    # Revision step 4: the corrected runs under the frozen decision rules.
    import build_corrected_verdicts
    check_true("corrected verdicts are current (build_corrected_verdicts --check)",
               build_corrected_verdicts.main(["--check"]) == 0)
    # Revision step 4: FINDINGS 17 quotes the corrected runs; every number it states for a
    # complete run is recomputed from artifacts/corrected_verdicts.json (itself checked current
    # against the committed cells above).
    print("\n== FINDINGS 17: corrected-trainer confirmation ==")
    _f17 = _read("docs/FINDINGS.md")
    _s17 = _f17[_f17.index("## 17. Corrected-trainer confirmation"):]
    _v = _load_art("corrected_verdicts.json")
    _c1 = _v["runs"]["C1"]
    if _c1.get("status") == "complete":
        _p, _g = _c1["primary"], _c1["gates"]
        _iv = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
        _sv = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"
        for label, needle in [
            ("C1 survival", f"**{_p['survival']['mean']:.3f} {_iv(_p['survival']['t90'])}**"),
            ("C1 predictor", f"{_p['predictor']['mean']:.3f} {_iv(_p['predictor']['t90'])}"),
            ("C1 untrained", f"{_p['untrained']['mean']:.3f} {_iv(_p['untrained']['t90'])}"),
            ("C1 seeds at bar", f"| {_p['survival']['seeds_at_bar']}/10 |"),
            ("C1 margin vs predictor", f"**{_p['contrast_vs_predictor']['mean']:+.3f} {_sv(_p['contrast_vs_predictor']['t90'])}**"),
            ("C1 margin vs untrained", f"**{_p['contrast_vs_untrained']['mean']:+.3f} {_sv(_p['contrast_vs_untrained']['t90'])}**"),
            ("C1 per-seed survival", ", ".join(f"{x:.3f}" for x in _p["survival"]["per_seed"])),
            ("C1 L0 mean", f"average **{_g['l0']['mean']:.3f}**"),
            ("C1 L0 per seed", ", ".join(f"{x:.3f}" for x in _g["l0"]["per_seed"])),
            ("C1 L0 TOST p", f"p = {_g['l0']['tost_p']:.3f}"),
            ("C1 L0 ROPE share", f"{_g['l0']['rope']['boot_share_in_rope']:.3f} of bootstrap means"),
            ("C1 speed min", f"at least {_g['speed']['min']:.3f}"),
            ("C1 verdict", "MET on the decodability clauses, conditional on the open L0\ngate"
             if _p["verdict"].endswith("l0") else _p["verdict"]),
        ] + [(f"C1 correction {d} {q}", f"{x['corrected']['mean'] if q == 'survival_target' else x['corrected']:.3f} | "
              f"{x['historical']['mean'] if q == 'survival_target' else x['historical']:.3f} | "
              f"{x['difference']['mean']:+.3f} {_sv(x['difference']['t90'])}")
             for d, blk in _c1["correction_effect"].items()
             for q, x in blk.items() if q in ("survival_target", "engagement_return")]:
            check_true(f"FINDINGS 17 quotes {label}", needle in _s17)
        check_true("C1 integrity passes (successor bootstrap, baselines bit-identical)",
                   _c1["integrity"]["pass"])
    _c2 = _v["runs"].get("C2", {})
    if _c2.get("status") == "complete":
        _p2, _g2 = _c2["primary"], _c2["gates"]
        _iv2 = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
        _sv2 = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"
        _s17n2 = " ".join(_s17.split())
        _ce = _c2["correction_effect"]
        for label, needle in [
            ("C2 survival", f"**{_p2['survival']['mean']:.3f} {_iv2(_p2['survival']['t90'])}**"),
            ("C2 predictor", f"{_p2['predictor']['mean']:.3f} {_iv2(_p2['predictor']['t90'])}"),
            ("C2 untrained", f"{_p2['untrained']['mean']:.3f} {_iv2(_p2['untrained']['t90'])}"),
            ("C2 per-seed survival", ", ".join(f"{x:.3f}" for x in _p2["survival"]["per_seed"])),
            ("C2 margin vs predictor", f"{_p2['contrast_vs_predictor']['mean']:+.3f} {_sv2(_p2['contrast_vs_predictor']['t90'])}"),
            ("C2 margin vs untrained", f"{_p2['contrast_vs_untrained']['mean']:+.3f} {_sv2(_p2['contrast_vs_untrained']['t90'])}"),
            ("C2 L0", f"average {_g2['l0']['mean']:.3f} (TOST p = {_g2['l0']['tost_p']:.3f})"),
            ("C2 verdict", f"**Verdict: {_p2['verdict']}**"),
            ("C2 correction 0.45", f"{_ce['0.45']['survival_target']['difference']['mean']:+.3f} "
                                   f"{_sv2(_ce['0.45']['survival_target']['difference']['t90'])}"),
            ("C2 correction 0.00", f"{_ce['0.00']['survival_target']['difference']['mean']:+.3f} "
                                   f"{_sv2(_ce['0.00']['survival_target']['difference']['t90'])}"),
        ]:
            check_true(f"FINDINGS 17.4 quotes {label}", needle in _s17n2)
        check_true("C2 integrity passes", _c2["integrity"]["pass"])
        _ac = _v["auxiliary_comparison"]
        check_true("FINDINGS 17.4 quotes the auxiliary contrast",
                   f"**{_ac['c1_minus_c2']:+.3f}**, paired by seed **{_sv2(_ac['paired']['t90'])}**" in _s17n2)
        check_true("FINDINGS 17.4 counts the seeds favoring the decoder",
                   f"{sum(x > 0 for x in _ac['paired']['diff_per_seed'])} of 10 seeds favor the decoder" in _s17n2)
        check_true("17.4 wording matches the frozen auxiliary rule (holds on decodability, not in full)",
                   (not _ac["holds"]) and _ac["holds_on_decodability_clauses"]
                   and "holds on the decodability clauses, conditional on `C1`'s open gate" in _s17n2)
    _bcp = os.path.join(ARTROOT, "budget_curve.json")
    if os.path.exists(_bcp):
        _bc = _load_art("budget_curve.json")
        _on = {r["updates"]: r for r in _bc["series"]["decoder on"]}
        _off = {r["updates"]: r for r in _bc["series"]["decoder off"]}
        for u in (100, 200, 300, 450):
            f = lambda r, k: f"{r[k]['mean']:.3f} [{r[k]['t90'][0]:.3f}, {r[k]['t90'][1]:.3f}]"
            row = (f"| {f(_on[u], 'return')} | {f(_off[u], 'return')} | "
                   f"{f(_on[u], 'target')} | {f(_off[u], 'target')} |")
            check_true(f"FINDINGS 17.10 budget row {u}", row in _s17)
        for c in _bc["contrast"]:
            row = (f"| {c['updates']} | {c['target']['mean']:+.3f} [{c['target']['t90'][0]:+.3f}, {c['target']['t90'][1]:+.3f}] | "
                   f"{c['return']['mean']:+.3f} [{c['return']['t90'][0]:+.3f}, {c['return']['t90'][1]:+.3f}] |")
            check_true(f"FINDINGS 17.10 contrast row {c['updates']}", row in _s17)
    _xrp = os.path.join(ARTROOT, "cross_replay", "corrected_c1_c2.json")
    if os.path.exists(_xrp):
        _xr = _load_art("cross_replay", "corrected_c1_c2.json")
        _xa = _xr["aggregate"]
        g = lambda k: f"{_xa[k]['mean']:.3f} [{_xa[k]['t90'][0]:.3f}, {_xa[k]['t90'][1]:.3f}]"
        check_true("FINDINGS 17.10 cross-replay row, decoder-on trunk",
                   f"| decoder-on trunk | {g('a_on_a')} | {g('a_on_b')} |" in _s17)
        check_true("FINDINGS 17.10 cross-replay row, decoder-off trunk",
                   f"| decoder-off trunk | {g('b_on_a')} | {g('b_on_b')} |" in _s17)
        for k, c in _xr["contrasts"].items():
            check_true(f"FINDINGS 17.10 quotes {k.split(' ')[0]}",
                       f"{c['mean']:+.3f} [{c['t90'][0]:+.3f}, {c['t90'][1]:+.3f}]" in _s17)
        check_true("the cross-replay artifact is labeled exploratory", "exploratory" in _xr["status"])
    _l0p = os.path.join(ARTROOT, "l0_audit", "corrected_l3_h8_wm.json")
    if os.path.exists(_l0p):
        _l0 = _load_art("l0_audit", "corrected_l3_h8_wm.json")
        _ls = _l0["l0_summary"]
        for r in _ls["by_pair"]:
            check_true(f"FINDINGS 17.5 quotes L0 pair {r['bases'][0]}",
                       f"| {r['mean']:.3f} | {r['tost_p']:.3f} | {r['first_state_mean']:.3f} |" in _s17)
        _ti = _ls["tost_over_independent_pairs"]
        check_true("FINDINGS 17.5 quotes the independent-pair mean and sd",
                   f"average **{_ti['mean']:.3f}** with sd **{_ls['independent_sd_of_pair_means']:.3f}**" in _s17)
        check_true("FINDINGS 17.5 quotes the TOST over pairs", f"(p = {_ti['p']:.3f})" in _s17)
        _ps = _l0["paired_summary"]
        check_true("FINDINGS 17.5 quotes the balanced survival readout",
                   f"**{_ps['survival']['mean']:.3f}**\n[{_ps['survival']['t90'][0]:.3f}, {_ps['survival']['t90'][1]:.3f}]" in _s17
                   or f"**{_ps['survival']['mean']:.3f}** [{_ps['survival']['t90'][0]:.3f}, {_ps['survival']['t90'][1]:.3f}]" in _s17)
        for arm in ("predictor", "untrained"):
            check_true(f"FINDINGS 17.5 quotes the balanced {arm} readout",
                       f"{arm} {_ps[arm]['mean']:.3f} [{_ps[arm]['t90'][0]:.3f}, {_ps[arm]['t90'][1]:.3f}]" in _s17)
    _l1p = os.path.join(ARTROOT, "control_diagnostics", "l1_stream_readout.json")
    if os.path.exists(_l1p):
        _f14b = _read("docs/FINDINGS.md")
        _s147 = _f14b[_f14b.index("### 14.7.1"):_f14b.index("## 15. Matched-handicap")]
        _l1 = _load_art("control_diagnostics", "l1_stream_readout.json")
        _la = _l1["aggregate"]
        check_true("14.7.1: every state pool bit-matches the run's dumps",
                   bool(_l1["all_dumps_bit_match"]))
        check_int("14.7.1: thirty cells scored", len(_l1["cells"]), 30)
        for _k, _lab in (("target", "state probe (h_t)"),
                         ("obs_summary_only", "observation summary features"),
                         ("seq_flat_linear", "flattened sequence, linear"),
                         ("seq_gru", "supervised GRU on the stream")):
            _r = "| " + _lab + " | " + " | ".join(
                f"{_la[f'd=0.02 {_a} {_k}']['mean']:.3f} "
                f"[{_la[f'd=0.02 {_a} {_k}']['t90'][0]:.3f}, "
                f"{_la[f'd=0.02 {_a} {_k}']['t90'][1]:.3f}]"
                for _a in ("survival", "predictor", "untrained")) + " |"
            check_true(f"14.7.1 table row {_k}", _r in _s147)
        _dmax = max(_la["d=0.02 survival obs_summary_only"]["mean"],
                    _la["d=0.02 survival seq_gru"]["mean"])
        check_true("14.7.1: branch S1, the larger stream decoder clears the bar", _dmax >= 0.65)
        check_true("14.7.1: the state does not clear the bar",
                   _la["d=0.02 survival target"]["mean"] < 0.65)
        check_true("14.7.1 quotes the larger stream decoder", f"**{_dmax:.3f}**" in _s147)
    _gnp = os.path.join(ARTROOT, "texture", "T_gn_l3_h8_wm.json")
    if os.path.exists(_gnp):
        _f14 = _read("docs/FINDINGS.md")
        _s14 = _f14[_f14.index("## 14. H2 substrate-grounding"):_f14.index("## 15.")]
        _gn = _load_art("texture", "T_gn_l3_h8_wm.json")
        _sp = _gn["gates"]["speed_positive_control"]
        check_true("14.5.1: the gn run fails exactly the speed positive control",
                   _gn["gates_failed"] == ["speed_positive_control"])
        check_true("14.5.1: the registered rule is not met", not _gn["rule_met"])
        check_true("14.5.1: it routes uninformative", _gn["routing"].startswith("UNINFORMATIVE"))
        check_true("14.5.1 quotes the worst pool, over every arm not just survival",
                   f"**{_sp['min_all_pools']:.4f}**" in _s14)
        check_true("14.5.1 quotes how many pools are short",
                   f"**{_sp['n_below']} of the {_sp['n_pools']}**" in _s14)
        check_true("14.5.1 still records the survival-arm worst",
                   f"{_sp['min_survival']:.5f}" in _s14)
        check_true("14.5.1: the gate is scored over every pool",
                   _sp["n_pools"] == 60 and _sp["min_all_pools"] < _sp["min_survival"])
        check_true("14.5.1: engagement is verified from the cells, not asserted",
                   _gn["gates"]["engagement"]["n_cells"] == 20
                   and _gn["gates"]["engagement"]["n_engaged"] == 20)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from promote_texture_trained import REGISTERED_GATES as _REG
        check_true("14.5.1: every registered gate is scored in the artifact",
                   set(_gn["gates"]) == set(_REG))
        # The matched forward margin, recomputed, so the like-for-like comparison cannot drift.
        _fw = {}
        for _c in _load_art("fold_rescore", "l3_h8_heldout.json")["cells"]:
            if float(_c["drift"]) == 0.45 and "explicit" in _c:
                _fw.setdefault(_c["agent"], []).append(float(_c["explicit"]["target"]))
        _mfwd = float(np.mean(_fw["survival"]) - np.mean(_fw["untrained"]))
        check_true("17.5.2 quotes the matched forward margin",
                   f"that is {_mfwd:+.3f}" in _s17)
        check_true("17.5.2 no longer claims the reverse margin is the larger one",
                   "larger than the forward direction" not in _s17)
        for _arm in ("untrained", "predictor", "survival"):
            _a = _gn["primary"][_arm]
            _iv = f"[{_a['t90'][0]:.3f}, {_a['t90'][1]:.3f}]"
            # The survival mean is bolded and may wrap before its interval.
            check_true(f"14.5.1 quotes the {_arm} reading",
                       any(f"{_m} {_iv}" in _s14 or f"{_m}\n{_iv}" in _s14
                           for _m in (f"{_a['mean']:.3f}", f"**{_a['mean']:.3f}**")))
        check_int("14.5.1: no seed reaches the bar",
                  _gn["primary"]["survival"]["seeds_at_or_above_bar"], 0)
        check_true("14.5.1: every other gate passes",
                   all(g["pass"] for k, g in _gn["gates"].items()
                       if k != "speed_positive_control"))
    _rdp = os.path.join(ARTROOT, "l0_audit", "reverse_direction_l3_h8_heldout.json")
    if os.path.exists(_rdp):
        from scripts.reverse_direction import adjudicate as _radj
        _rd = _load_art("l0_audit", "reverse_direction_l3_h8_heldout.json")
        _rs = _rd["summary"]
        check_true("17.5.2: the agents scored are the drift-0 ones",
                   _rd["agent_drift"] == 0.0 and _rd["readout_drift"] == 0.45)
        check_true("17.5.2: the integrity gate passed", bool(_rd["integrity"]["pass"]))
        check_int("17.5.2: thirty cells checked by the gate", _rd["integrity"]["n_checked"], 30)
        check_true("17.5.2 quotes the gate's worst deviation",
                   f"worst abs dev {_rd['integrity']['worst_abs_dev']:.4f}" in _s17)
        for _arm in ("untrained", "predictor", "survival"):
            _a, _b = _rs["standard"][_arm], _rs["balanced"][_arm]
            _per = [c["target"] for c in sorted(_rd["cells"], key=lambda c: c["seed"])
                    if c["arm"] == _arm]
            check("17.5.2: " + _arm + " mean recomputes from the cells",
                  _a["mean"], float(np.mean(_per)))
            check_true(f"17.5.2 table row {_arm}",
                       f"| {_arm} | {_a['mean']:.3f} [{_a['t90'][0]:.3f}, {_a['t90'][1]:.3f}] | "
                       f"{_b['mean']:.3f} [{_b['t90'][0]:.3f}, {_b['t90'][1]:.3f}] |" in _s17)
        _re = _radj(survival=_rs["standard"]["survival"]["mean"],
                    untrained=_rs["standard"]["untrained"]["mean"],
                    bar=_rd["bar"], margin=_rd["margin"])
        check_true("17.5.2: the verdict recomputes under the frozen rule",
                   _re["verdict"] == _rs["verdict"])
        check_true("17.5.2 quotes the verdict",
                   f"frozen before the run: {_rs['verdict']}" in _s17)
    _s13 = _f17[_f17.index("## 13. Experiment C"):_f17.index("## 14.")]
    _pip = os.path.join(ARTROOT, "expC", "emergence_pilot_per_individual_summary.json")
    if os.path.exists(_pip):
        _pi = _load_art("expC", "emergence_pilot_per_individual_summary.json")
        _g1 = _pi["gates"]["gate1_exploitability"]
        check_true("13.F: gate 1 (exploitability) fails", not _g1["passes_gate1"])
        _g1txt = (f"treatment gap {_g1['treatment_gap_mean']:.5f}, 90%\n"
                  f"CI [{_g1['treatment_ci90'][0]:.5f}, {_g1['treatment_ci90'][1]:.5f}]")
        check_true("13.F quotes the gate-1 payoff and interval", _g1txt in _s13)
        _g5 = _pi["gates"]["gate5_speed_control"]
        check_true("13.F: gate 5 (speed positive control) fails", not _g5["pass"])
        check_true("13.F quotes the worst speed-control reading",
                   f"minimum {_g5['min']:.3f} against the 0.75 bar" in _s13)
        check_true("13.F records the uninformative routing",
                   "UNINFORMATIVE" in _pi["gates"]["routing"] and "UNINFORMATIVE" in _s13)
        _ind = [c[k] for c in _pi["cells"]
                for k in ("indiv_gen0", "indiv_final_treat", "indiv_final_ctrl")]
        check_int("13.F: 48 individuals scored per population",
                  sorted({x["n_individuals"] for x in _ind})[0], 48)
        check_true("13.F: no individual reaches the bar in any population",
                   all(x["share_at_or_above_bar"] == 0.0 for x in _ind))
        def _three(key):
            v = [f"{c[key]['mean']:.3f}" for c in _pi["cells"]]
            return f"{v[0]}, {v[1]} and {v[2]}"
        check_true("13.F quotes the generation-0 per-individual means",
                   f"Means at generation 0 are {_three('indiv_gen0')}" in _s13)
        check_true("13.F quotes the final treatment per-individual means",
                   f"treatment populations {_three('indiv_final_treat')}" in _s13)
        check_true("13.F quotes the final control per-individual means",
                   f"control populations\n{_three('indiv_final_ctrl')}" in _s13)
    _xevp = os.path.join(ARTROOT, "corrected_runs", "corrected_l3_h8_wm", "cells")
    if os.path.isdir(_xevp):
        _xv = {}
        for _fn in sorted(os.listdir(_xevp)):
            with open(os.path.join(_xevp, _fn), encoding="utf-8") as _fh:
                _c = json.load(_fh)["cell"]
            if _c.get("xeval"):
                _xv.setdefault(float(_c["drift"]), []).append(_c["xeval"])
        for _td in sorted(_xv):
            _rows = _xv[_td]
            check_int(f"17.2.1: ten cells trained at drift {_td}", len(_rows), 10)
            _cells = " | ".join(f"{float(np.mean([r[_ed] for r in _rows])):+.3f}"
                                for _ed in sorted(_rows[0]))
            _lab = "authentic (drift 0)" if _td == 0.0 else f"surrogate (drift {_td})"
            check_true(f"17.2.1 quotes the cross-evaluation row for {_lab}",
                       f"| {_lab} | {_cells} |" in _s17)
    _wsp = os.path.join(ARTROOT, "l0_audit", "d045_world_samples_l3_h8_heldout.json")
    if os.path.exists(_wsp):
        from itasorl.l0_audit import world_sample_summary as _wss
        _ws = _load_art("l0_audit", "d045_world_samples_l3_h8_heldout.json")
        _wsum = _ws["summary"]["survival"]
        _wcells = sorted(_ws["cells"], key=lambda c: c["seed"])
        check_int("17.5.1: ten survival agents scored", len(_wcells), 10)
        check_true("17.5.1: the integrity gate passed", bool(_ws["integrity"]["pass"]))
        check_true("17.5.1 quotes the integrity gate's worst deviation",
                   f"worst |dev| {_ws['integrity']['worst_abs_dev']:.4f}" in _s17)
        _wm = [sum(c["rows"][k]["target"] for c in _wcells) / len(_wcells)
               for k in range(len(_ws["bases"]))]
        for k, (b, m) in enumerate(zip(_ws["bases"], _wm)):
            _lab = "registered" if k == 0 else f"independent {k}"
            check_true(f"17.5.1 quotes the drift-0.45 draw {b[0]}",
                       f"| {_lab} | {b[0]} / {b[1]} | {m:.3f} |" in _s17)
        _re = _wss(_wm[1:], _wm[0], bar=_ws["bar"])
        check_true("17.5.1: the verdict recomputes from the cells under the frozen rule",
                   _re["verdict"] == _wsum["verdict"] and _re["registered_rank"] == _wsum["registered_rank"])
        check("17.5.1: registered draw mean", _wsum["registered"], _wm[0])
        check_true("17.5.1 quotes the registered mean and rank",
                   f"registered **{_wsum['registered']:.3f}** ranks **{_wsum['registered_rank']}** of 9" in _s17)
        check_true("17.5.1 quotes the independent range and sd",
                   f"**{_wsum['min']:.3f}** to **{_wsum['max']:.3f}**, sd "
                   f"**{_wsum['between_draw_sd']:.3f}**" in _s17)
        check_true("17.5.1 quotes the t-based interval over draws",
                   f"[{_wsum['t90_over_draws'][0]:.3f}, {_wsum['t90_over_draws'][1]:.3f}]" in _s17)
        check_true("17.5.1 quotes the count at or above the bar",
                   f"**{_wsum['n_at_or_above_bar']} of {_wsum['n_draws']}** independent draws" in _s17)
        check_true("17.5.1 quotes the verdict", f"verdict **{_wsum['verdict']}**" in _s17)
    _pcp = os.path.join(ARTROOT, "policy_controls", "corrected_l3_h8_wm.json")
    if os.path.exists(_pcp):
        from itasorl.stats import paired_contrast as _pc
        _pcd = _load_art("policy_controls", "corrected_l3_h8_wm.json")
        check_true("17.6: every survival retrain was bit-identical",
                   all(c["retrain_identical"] for c in _pcd["cells"]) and len(_pcd["cells"]) == 10)
        _agg = _pcd["aggregate"]
        for arm in ("untrained", "predictor", "predictor_logged", "survival"):
            row = f"| {arm} | " + " | ".join(
                f"{_agg[f'{arm} {pr}']['mean']:.3f} [{_agg[f'{arm} {pr}']['t90'][0]:.3f}, "
                f"{_agg[f'{arm} {pr}']['t90'][1]:.3f}]" for pr in ("own", "scripted", "replay")) + " |"
            check_true(f"FINDINGS 17.6 table row {arm}", row in _s17)
        _cells = sorted(_pcd["cells"], key=lambda c: c["seed"])
        _tv = lambda arm, pr: [c["targets"][arm][pr]["target"] for c in _cells]
        for pr in ("own", "scripted", "replay"):
            cs = [_pc(_tv("survival", pr), _tv(b2, pr)) for b2 in ("predictor", "predictor_logged", "untrained")]
            row = f"| {pr} | " + " | ".join(f"{c['mean']:+.3f} [{c['t90'][0]:+.3f}, {c['t90'][1]:+.3f}]" for c in cs) + " |"
            check_true(f"FINDINGS 17.6 contrast row {pr}", row in _s17)
    _psp = os.path.join(ARTROOT, "persistence", "corrected_l3_h8_wm.json")
    if os.path.exists(_psp):
        _pa = _load_art("persistence", "corrected_l3_h8_wm.json")["aggregate"]
        check_true("17.7: every drift-0 condition reads exactly 0.5",
                   all(v["window_mean"] == 0.5 for k, v in _pa.items() if k.startswith("d=0.00")))
        _w = lambda arm, c: (f"{_pa[f'd=0.45 {arm} {c}']['window_mean']:.3f} "
                             f"[{_pa[f'd=0.45 {arm} {c}']['window_t90'][0]:.3f}, "
                             f"{_pa[f'd=0.45 {arm} {c}']['window_t90'][1]:.3f}]")
        for c in ("replay:prefix", "common_state:prefix", "factorial:hidden", "factorial:physical",
                  "reset_hidden:prefix"):
            for arm in ("untrained", "predictor", "survival"):
                check_true(f"FINDINGS 17.7 quotes {arm} {c}", _w(arm, c) in _s17)
            check_true(f"FINDINGS 17.7 quotes survival late {c}",
                       f"| {_pa[f'd=0.45 survival {c}']['late_mean']:.3f} |" in _s17)
        _rp = _pa["d=0.45 survival replay:prefix"]
        _first = next(i + 1 for i, x in enumerate(_rp["auc_by_t_mean"]) if x < 0.55)
        check_true("17.7: the frozen replay rule is not met (mean or lower bound under 0.65)",
                   _rp["window_mean"] < 0.65 or _rp["window_t90"][0] < 0.65)
        check_true("FINDINGS 17.7 states the first step under 0.55 (the fifth)", _first == 5
                   and "falls below\n0.55 at the fifth" in _s17)
    _cdp = os.path.join(ARTROOT, "control_diagnostics", "corrected_l3_h8_wm.json")
    if os.path.exists(_cdp):
        _cd = _load_art("control_diagnostics", "corrected_l3_h8_wm.json")
        check_true("17.8: regenerated pools bit-match the run's dumps", _cd["all_dumps_bit_match"])
        _ca = _cd["aggregate"]
        _cv = lambda arm, m: (f"{_ca[f'd=0.45 {arm} {m}']['mean']:.3f} [{_ca[f'd=0.45 {arm} {m}']['t90'][0]:.3f}, "
                              f"{_ca[f'd=0.45 {arm} {m}']['t90'][1]:.3f}]")
        for m in ("target", "resid_trace", "resid_trace_act", "resid_obs", "resid_obs_hist",
                  "resid_obs_hist_beh", "resid_joint_mlp", "obs_summary_only", "seq_gru", "seq_flat_linear"):
            row = " | ".join(_cv(arm, m) for arm in ("survival", "predictor", "untrained"))
            check_true(f"FINDINGS 17.8 row {m}", row.replace("[", "[").replace(_cv("survival", m), _cv("survival", m)) in _s17.replace("**", ""))
            if m.startswith("resid_"):
                _r2 = _ca[f"d=0.45 survival {m}"]["nuisance_r2_heldout_mean"]
                check_true(f"FINDINGS 17.8 held-out R2 {m} ({_r2:.2f})", f"| {_r2:.2f} |" in _s17)
                check_true(f"17.8: basis not recoverable from the survival residual ({m})",
                           _ca[f"d=0.45 survival {m}"]["nuisance_from_residual_r2_mean"] <= -0.005)
    for _fam in ("gn", "qd"):
        _tp = os.path.join(ARTROOT, "texture", f"corrected_l3_h8_wm_{_fam}.json")
        if not os.path.exists(_tp):
            continue
        from itasorl.stats import paired_contrast as _pc2
        _ta = _load_art("texture", f"corrected_l3_h8_wm_{_fam}.json")["aggregate"]
        for _q in ("transfer", "fresh", "fresh_paired"):
            _s, _u = _ta[f"survival {_q}"], _ta[f"untrained {_q}"]
            _c = _pc2(_s["per_seed"], _u["per_seed"])
            row = (f"{_s['mean']:.3f} [{_s['t90'][0]:.3f}, {_s['t90'][1]:.3f}] | "
                   f"{_ta[f'predictor {_q}']['mean']:.3f} | {_u['mean']:.3f} | "
                   f"{_c['mean']:+.3f} [{_c['t90'][0]:+.3f}, {_c['t90'][1]:+.3f}] |")
            check_true(f"FINDINGS 17.9 quotes {_fam} {_q}", row in _s17)
        # the frozen Q1 / Q2 rule: mean >= 0.65 and >= 0.05 above untrained
        _s17n = " ".join(_s17.split())
        for _q, _yes, _no in (("transfer", f"For `{_fam}`: the L3 direction reads the comparator",
                               "the original direction does not transfer"),
                              ("fresh", "separates the comparator when a probe is fit to it",
                               "a fresh probe did not meet the registered criterion")):
            _m = _ta[f"survival {_q}"]["mean"]
            _ok = _m >= 0.65 and _m - _ta[f"untrained {_q}"]["mean"] >= 0.05
            check_true(f"17.9 {_fam} {_q}: frozen rule {'passes' if _ok else 'fails'} and the wording says so",
                       (_yes if _ok else _no) in _s17n)
    _prp = os.path.join(ARTROOT, "population_readout", "corrected_l3_h8_wm.json")
    if os.path.exists(_prp):
        _pr = _load_art("population_readout", "corrected_l3_h8_wm.json")
        for _arm, _lab in (("survival", "survival agents"), ("untrained", "untrained agents")):
            _pp = _pr["panels"][_arm]
            _n65 = sum(x >= 0.65 for x in _pp["per_individual"])
            check_true(f"FINDINGS 17.11 quotes the {_arm} panel",
                       f"| {_lab} | {_pp['mean']:.3f} [{_pp['t90'][0]:.3f}, {_pp['t90'][1]:.3f}] | "
                       f"{_n65} of 10 | {_pp['pooled_probe']:.3f} |" in _s17)
        _vi = _pr["value_of_world_information"]
        check_true("FINDINGS 17.11 quotes the value of world information",
                   f"**{_vi['mean']:.3f}** return (t-based 90% CI [{_vi['t90'][0]:.3f}, {_vi['t90'][1]:.3f}]"
                   in _s17)
    # Revision step 10: FINDINGS 10.1.1 quotes the surrogate diagnostics; pin every cell.
    print("\n== FINDINGS 10.1.1: surrogate and detector diagnostics ==")
    _f = _read("docs/FINDINGS.md")
    _sec = _f[_f.index("### 10.1.1"):_f.index("### 10.2 Headline result")]
    _sd = _load_art("surrogate_diagnostics.json")["surrogates"]
    for _name, _v in _sd.items():
        _g = _v["diagnostics"]
        for _label, _val, _fmt in (
                ("one-step train", _g["g_one_step"]["rms_train"], "{:.4f}"),
                ("one-step held out", _g["g_one_step"]["rms_heldout"], "{:.4f}"),
                ("lag-1 autocorrelation", _v["profile"]["lag1_autocorr"], "{:.2f}"),
                ("agent-accessible detector", _v["agent_accessible_detector"]["auroc"], "{:.3f}")):
            check_true(f"10.1.1 {_name} {_label} {_fmt.format(_val)} quoted", _fmt.format(_val) in _sec)
    check_true("10.1.1 linear-fit held-out RMS is rounding-level (< 1e-7)",
               all(v["diagnostics"]["linear_fit"]["rms_heldout"] < 1e-7 for v in _sd.values()))
    # Revision step 15: the manuscript's tables are generated from the checked artifacts.
    # The manuscript and its generated tables stay local (gitignored docs/paper/); render the
    # tables in memory from the committed artifacts instead of comparing committed files.
    import build_paper_tables
    _pt = build_paper_tables.tables()
    check_true("manuscript tables render from the committed artifacts (gate table, contrasts, "
               "corrected verdicts)",
               {"gate_table.tex", "contrast_intervals.tex", "corrected_runs.tex"} <= set(_pt)
               and all("do not edit" in t for t in _pt.values()))
    check_true("local manuscript tables, if present, are current (build_paper_tables --check)",
               build_paper_tables.main(["--check"]) == 0)

    print()
    if failures:
        print(f"RESULT: {len(failures)} of {n_checks} checks FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"RESULT: all {n_checks} checks passed.")
    return 0


def _find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while (pos := haystack.find(needle, start)) != -1:
        out.append(pos)
        start = pos + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
