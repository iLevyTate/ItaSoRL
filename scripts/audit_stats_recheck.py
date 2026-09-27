"""Recompute every published L3/B-v2 number from committed artifacts.

Publication gate: loads the committed JSONs under artifacts/expB2/ and
recomputes, from the per-seed cell values, every quantitative claim quoted in
README.md, docs/FINDINGS.md sections 9-10 and 14 (including the 14.5-14.7 H2
batteries under artifacts/expH2 and artifacts/expL1), docs/PAPER_OUTLINE.md,
and the B-v3 n=10 gate values recorded in docs/PREREGISTRATION_Bv3.md
section 12 (artifacts/expB2/bv3_n10_gates.json). Fails loudly (non-zero exit)
on any mismatch beyond rounding.

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

    # ---- derived-doc resolution guard -----------------------------------
    # The reactive-vs-persistent reading was PROVISIONAL until the section 10.6
    # re-score; it is now RESOLVED (FINDINGS 10.6.1, 2026-07-19): the corrected
    # common-garden control passes the frozen rule on both directions, so the
    # signal is a modest persistent world-identity component. The public-facing
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
        ("index.html", "modest persistent",
         "index.html carries the resolved persistent reading"),
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
    # and the old reactive-only claim must not reappear in index.html.
    idx = _read("index.html")
    for phrase in ("not a persistent stored representation",
                   "not a stored representation"):
        check_true(f"index.html no longer states the reactive-only claim '{phrase}'",
                   len(_find_all(idx, phrase)) == 0)
    # index.html is now GENERATED from index.template.html by scripts/build_index.py,
    # which fills {{...}} placeholders from the artifact-derived site metrics. So instead
    # of pinning bare number strings, regenerate the page and require it to be already up
    # to date: any artifact/headline change must be re-rendered in the same commit or this
    # fails. (Replaces the 2026-07-18 headline string pins with real regeneration.)
    sys.path.insert(0, os.path.dirname(__file__))
    import build_index
    check_true("index.html is regenerable and current (build_index --check)",
               build_index.build(os.path.join(root), check=True) == 0)

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
