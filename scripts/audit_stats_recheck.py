"""Check quoted numbers and watched wording against the committed artifacts.

What it does. For each result it guards, it loads the committed JSONs under artifacts/,
recomputes the summary statistic (mean, t-based or bootstrap interval, TOST, seed count)
from the per-seed values where the artifact stores them, and compares it with the number the
documents quote, which this script mirrors as a constant. Where an artifact stores no
per-seed values (the Experiment A and B summaries, gate-0 oracle AUROCs, configuration
values, recorded integrity results and a few stored gate readings), the stored value itself
is compared with the constant. Those checks are labeled "pinned:" and the run prints how
many there are: 84 pinned of 3231 on 2026-10-17 (PIN_COUNT and CHECK_FLOOR below). The count
is of check() calls; composite check_true gates that also compare a stored reading with a
literal are not counted. A stored L0 gate (mean, TOST
p and verdict) whose artifact keeps the drift-0 per-seed values must equal its recomputation
from them, and every tracked artifact a guarded block reads must be present: a missing one
fails by name instead of dropping the block's checks. It also checks that the generated pages
(index.html, the results manifest, the gate table, the contrast intervals, the manuscript
tables) match a fresh render, and that retired wording has not returned to the pages it
watches. Fails loudly (non-zero exit) on any mismatch beyond rounding.

What it reads from the documents. Numbers in the document text are compared with artifact
values, recomputed where per-seed values exist, for FINDINGS 3.1, 10.1.1, 13.F, 14.5.1,
14.7, 14.7.1 and section 17, the Experiment C lines of the FINDINGS summary, the dated notes
of 2026-10-10 in FINDINGS 3.4, 9, 10.5, 10.9, 12, 13.E, 13.F, 14, 14.6, 16 and 17.11 (where
they quote a committed artifact), section 19 and the dated notes of 2026-10-11 in FINDINGS
10.1, 10.3, 10.5, 10.7, 10.9, 13.F, 14.5, 14.5.1 and 17.9 (every table row rebuilt from the
agent-free artifacts), section 20 (every table row rebuilt from the per-seed values of the
corrected reruns, their historical counterparts, the post-review readouts and the L1 capacity
curve, with the readouts' pre-stated labels derived again) and the 2026-10-11 notes that point
to it, the 2026-10-11 results entry of PREREGISTRATION_L3, the 2026-10-11 entries of
docs/CORRECTIONS.md, PAPER_OUTLINE rows 27, 40, 44, 46 and 54, the FINDINGS 18.2, 18.3 and 18.4
tables and the figures 18.2 and 18.6 quote beside them (from the goal-and-stakes artifacts, the
T-touch cells and each run's mechanism, policy-control and control-diagnostic readouts; the 18.5
table and the dated notes inside section 18 are not checked), the FINDINGS 22.3 sweep tables,
contrasts and ranges (texture/ar_sweep and texture/ar_sweep_c), the FINDINGS 23.3 refit probe
readings, partial-erasure label and correction note (mechanism/erasure), the O2 table of
docs/optional/README.md (its long-history ridge readout, recomputed from per_seed and checked
against the battery's fixed-penalty readings) and the corrected L3 headline sentence of
README.md (2026-10-16). In the rest of README.md, CITATION.cff, the site, the rest of
PAPER_OUTLINE and the rest of FINDINGS it checks selected phrases only, a few of which
contain a number; a number edited outside those phrases is not caught. Numbers
the notes take from gitignored run bundles under fullruns/ are not checked.

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

import glob
import json
import re
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import equivalence_test, mean_ci, rope_test, t_ci90  # noqa: E402

ARTROOT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
ART = os.path.join(ARTROOT, "expB2")
TOL = 5e-4  # docs quote 3 decimals; allow rounding slack
BAR = 0.65
# The exact number of checks a full run records, skipped ones included
# (tests/test_audit_selftest.py). A run that records fewer has lost checks without failing one.
# Change it deliberately when checks are added or removed, from the count the run prints, never
# by hand; the manuscript sentences that quote it are checked against it by
# scripts/build_paper_tables.py --manuscript. The count on 2026-10-17 is 3231.
CHECK_FLOOR = 3231
# How many of those checks are pins (labels start "pinned:"), as printed in "pins: P of N".
PIN_COUNT = 84

failures: list[str] = []
n_checks = 0
n_pins = 0
# Checks skipped because the anonymized supplement leaves their file out (_read_or_skip in
# main), one per skipped file; each such file guards exactly one check. The floor test counts
# them with n_checks, so a run inside the extracted archive meets the same CHECK_FLOOR as a
# checkout (finding p5, 2026-10-16).
n_skipped = 0


def t_ci(vals: list[float], level: float = 0.90) -> tuple[float, float]:
    from scipy.stats import t as student_t
    x = np.asarray(vals, dtype=float)
    n = x.size
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    q = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, n - 1))
    return (m - q * se, m + q * se)


def check(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    # Convention: a check whose actual value is read as stored in an artifact (no per-seed
    # recomputation) carries a label starting "pinned:", so the printed pin count is honest.
    global n_checks, n_pins
    n_checks += 1
    if label.startswith("pinned:"):
        n_pins += 1
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


def _artifact_present(path: str) -> bool:
    """Record a presence check for a tracked artifact before the block that reads it.

    Every artifact these blocks read is tracked and ships in the supplement, so a missing one
    is a failure, never a silent skip of the checks that read it (finding p5, 2026-10-16)."""
    rel = os.path.relpath(path, ARTROOT).replace(os.sep, "/")
    ok = os.path.isdir(path) if not path.endswith(".json") else os.path.isfile(path)
    check_true(f"artifact present: artifacts/{rel}", ok)
    return ok


def load(name: str) -> dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def seed_vals(doc: dict, drift: str, agent: str, metric: str) -> list[float]:
    rows = [r for r in doc["cells"]
            if r["drift"] == drift and r["agent"] == agent and metric in r]
    rows.sort(key=lambda r: r["seed"])
    return [float(r[metric]) for r in rows]


def _seed_mean(block: dict, key: str = "per_seed") -> float:
    """Mean recomputed from the per-seed values stored beside an aggregate, so a quoted
    number is compared with what the seeds give and not with the stored aggregate."""
    return float(np.mean(block[key]))


def _lead(doc: dict, arm: str) -> float:
    """Survival-minus-arm lead at the run's strongest drift, from per-seed pool targets."""
    arms = doc["arms"][doc["dmax"]]
    return _seed_mean(arms["survival"]["pool_target"]) - _seed_mean(arms[arm]["pool_target"])


def _fold_seeds(art: dict, met: str, scheme: str) -> list:
    """Per-seed survival values at drift 0.45 of a fold re-score, under one partition."""
    return [c[scheme][met] for c in art["cells"]
            if c["agent"] == "survival" and float(c["drift"]) == 0.45]


def _same(a: float, b: float, tol: float = 1e-9) -> bool:
    """Both finite and equal within tol. A NaN on either side is a mismatch, never a match."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def l0_gate_reproduces(label: str, doc: dict):
    """The stored L0 gate of a promoted expB2 run, recomputed from its drift-0 survival per-seed
    values with the registered TOST (against 0.5, margin 0.05) and ROPE. The stored mean, TOST p,
    TOST verdict and ROPE verdict must all equal the recomputation, so a shifted per-seed list
    cannot pass behind an unchanged stored gate (finding c25, 2026-10-16). Returns the TOST."""
    ps = doc["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]
    m, e, g = float(np.mean(ps)), equivalence_test(ps, margin=0.05), doc["gates"]
    ok = (_same(m, float(g["l0_tost"]["mean"]))
          and _same(m, float(g.get("l0_survival_mean", m)))
          and _same(float(e.p_value), float(g["l0_tost"]["p_value"]))
          and bool(e.equivalent) is bool(g["l0_tost"]["equivalent"])
          and ("l0_rope" not in g or bool(rope_test(ps).accept) is bool(g["l0_rope"]["accept"])))
    check_true(f"{label}: stored L0 gate (mean, TOST p and verdict, ROPE verdict) recomputes "
               f"from the drift-0 per-seed values", ok)
    return e


def texture_l0_gate_reproduces(label: str, doc: dict):
    """The same for a texture-trained artifact, whose drift-0 cells sit under l0_control."""
    ps = doc["l0_control"]["survival"]["per_seed"]
    m, e, g = float(np.mean(ps)), equivalence_test(ps, margin=0.05), doc["gates"]["l0_tost"]
    check_true(f"{label}: stored L0 gate (mean, TOST p and verdict) recomputes from the drift-0 "
               f"per-seed values",
               _same(m, float(g["mean"])) and _same(float(e.p_value), float(g["p"]))
               and bool(e.equivalent) is bool(g["pass"]))
    return e


def verify_aggregate_consistency(name: str, doc: dict) -> None:
    """The stored `aggregate` block must be exactly reproducible from `cells`.

    A block fails when it has no matching cells, when a matching cell is non-finite, when the
    finite cell count differs from the stored n_seeds (where stored), or when any stored or
    recomputed mean/lo/hi is non-finite."""
    global n_checks
    bad = 0
    for arm, metrics in doc["aggregate"].items():
        d, agent = arm.split(" ", 1)
        drift = d.split("=")[1]
        for metric, stored in metrics.items():
            raw = seed_vals(doc, drift, agent, metric)
            vals = [v for v in raw if np.isfinite(v)]
            if not vals or len(vals) != len(raw):
                bad += 1
                failures.append(f"{name} aggregate {arm}/{metric}: "
                                f"{len(vals)} finite of {len(raw)} cells")
                continue
            mean, lo, hi = mean_ci(vals)
            above = int(sum(v >= doc.get("bar", BAR) for v in vals))
            if (not _same(mean, float(stored["mean"])) or not _same(lo, float(stored["lo"]))
                    or not _same(hi, float(stored["hi"]))
                    or above != stored["n_above_bar"]
                    or ("n_seeds" in stored and len(vals) != stored["n_seeds"])):
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
        # The tails are recomputed from the per-seed values; the late tail and the margin
        # threshold are stored without them, so those two are pins (H40, 2026-10-17).
        surv = float(np.mean(sd["survival"]["cg_tail_per_seed"]))
        untr = float(np.mean(sd["untrained"]["cg_tail_per_seed"]))
        late = float(sd["survival"]["cg_latetail_mean"])
        check(f"cg re-score {tag} survival tail", surv, exp["surv"])
        check(f"cg re-score {tag} untrained floor", untr, exp["untr"])
        check(f"cg re-score {tag} predictor tail",
              float(np.mean(sd["predictor"]["cg_tail_per_seed"])), exp["pred"])
        check(f"pinned: cg re-score {tag} late-tail", late, exp["late"])
        check(f"pinned: cg re-score {tag} margin threshold (untrained + 0.05)",
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
    check("pinned: L1 oracle at delta=0.06 (1.000)",
          l1["L1 discretization, delta=0.06"]["oracle_auroc"], 1.000)
    check("pinned: L1 L0 control (0.523)",
          l1["L0 control, identical world"]["oracle_auroc"], 0.523)
    check_true("L1 leakage audit passes on clean cells",
               l1["L1 discretization, delta=0.06"]["leakage_pass"]
               and l1["L0 control, identical world"]["leakage_pass"])
    check_true("L1 contaminated-reward negative control CAUGHT (leakage_pass False)",
               not l1["L1 + contaminated reward (+0.02 in surrogate)"]["leakage_pass"])
    l2 = {c["label"]: c for c in ea["expA_l2"]["cells"]}
    check("pinned: L2 oracle at drift=0.30 (0.993)",
          l2["L2 rollout drift, drift_sigma=0.30"]["oracle_auroc"], 0.993)
    check("pinned: L2 L0 control (0.440)",
          l2["L2 control, drift_sigma=0  -> identical dynamics"]["oracle_auroc"], 0.440)
    with open(os.path.join(ARTROOT, "expB", "summary.json"), encoding="utf-8") as fh:
        eb = json.load(fh)
    sweep = {r["drift"]: r for r in eb["expB_full"]["drift_sweep"]}
    check("pinned: expB recurrent-state target @0.45 (0.510)", sweep[0.45]["target_mean"], 0.510)
    check("pinned: expB recurrent-state std @0.45 (0.039)", sweep[0.45]["target_std"], 0.039)
    check("pinned: expB recurrent-state target @0.20 (0.509)", sweep[0.2]["target_mean"], 0.509)
    # 2026-10-10: every cell of the FINDINGS 3.1 table that the artifact stores is matched
    # exactly, as printed (three cells had drifted by 0.001). The artifact stores no shuffled
    # standard deviation; those three come from the run log and are not checked here.
    _f31 = open(os.path.join(ARTROOT, "..", "docs", "FINDINGS.md"), encoding="utf-8").read()
    _f31 = _f31[_f31.index("### 3.1 "):_f31.index("### 3.2 ")]
    for _d, _lab in ((0.0, "0.00 (control)"), (0.2, "0.20"), (0.45, "0.45")):
        _r = sweep[_d]
        _row = (f"| {_lab} | {_r['target_mean']:.3f} ± {_r['target_std']:.3f} | "
                f"{_r['shuffled_mean']:.3f} ± ")
        check_true(f"FINDINGS 3.1 row {_lab} quotes the artifact's target and shuffled means",
                   _row in _f31 and f"| {_r['speed_mean']:.3f} ± " in _f31)
    surp = {r["drift"]: r for r in eb["expB_surprise"]["drift_sweep_with_std"]}
    check("pinned: surprise channel @0.45 (0.596)", surp[0.45]["mean"], 0.596)
    check("pinned: surprise channel std @0.45 (0.007)", surp[0.45]["std"], 0.007)
    nl = {r["drift"]: r for r in eb["expB_nonlinear"]["drift_sweep_with_std"]}
    check("pinned: nonlinear probe @0.45 (0.482)", nl[0.45]["mean"], 0.482)
    check("pinned: nonlinear probe std @0.45 (0.031)", nl[0.45]["std"], 0.031)
    for row in eb["expB_kstep_rerun_20260713"]["horizons"]:
        check_true(f"kstep rerun horizon {row['open_horizon']} at chance "
                   f"({row['drift_045_target_mean']:.3f} in 0.48-0.51)",
                   0.48 - TOL <= row["drift_045_target_mean"] <= 0.51 + TOL)

    print("== FINDINGS 7.1: B-v3 n=10 and capacity ceiling (committed) ==")
    with open(os.path.join(ART, "bv3_n10_summary.json"), encoding="utf-8") as fh:
        bv3 = json.load(fh)["pooled_target_drift045"]["survival"]
    check("B-v3 survival n=10 (0.610)", _seed_mean(bv3, "pool_target_per_seed"), 0.610)
    check("B-v3 boot lo (0.585)", mean_ci(bv3["pool_target_per_seed"])[1], 0.585)
    check("B-v3 boot hi (0.634)", mean_ci(bv3["pool_target_per_seed"])[2], 0.634)
    check_true("B-v3 CI entirely below 0.65 bar", bv3["boot90_hi"] < BAR)
    with open(os.path.join(ART, "sysid_ceiling_n10_summary.json"), encoding="utf-8") as fh:
        ceil = json.load(fh)["pooled_target_drift045"]["survival"]
    check("capacity ceiling n=10 (0.596)", _seed_mean(ceil, "pool_target_per_seed"), 0.596)
    check("ceiling boot lo (0.577)", mean_ci(ceil["pool_target_per_seed"])[1], 0.577)
    check("ceiling boot hi (0.616)", mean_ci(ceil["pool_target_per_seed"])[2], 0.616)
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
    check("cross-recipe rule margin over the bar (0.034)", _seed_mean(tr, "survival_per_seed") - BAR, 0.034)
    check("pinned: cross-recipe integrity gate reproduced 0.752",
          xr["integrity_gate"]["drift045_survival_mean_reproduced"], 0.752)
    g0 = xr["gate0"]
    check("pinned: cross-recipe gate-0 rff oracle (0.887)", g0["rff"]["selected_oracle_auroc"], 0.887)
    check_true("cross-recipe gate-0 rff oracle in band",
               0.85 <= g0["rff"]["selected_oracle_auroc"] <= 0.95)
    check_true("cross-recipe cd dropped with empty window (floors > 0.6 "
               "wherever oracle in band)",
               g0["cd"]["dropped"] and all(
                   f > 0.6 for o, f in zip(g0["cd"]["sweep_oracle"],
                                           g0["cd"]["sweep_floor"]) if o >= 0.85))

    # Estimator sanity gate on the promoted cg re-scores (artifacts/expB2/*cg_rescore*.json,
    # written by scripts/promote_cg_rescore.py): the drift-0 L0 floors must sit in the chance
    # band. A floor outside [0.4, 0.6] means a broken estimator and the audit FAILS, so the
    # July-18 bias class cannot silently pass this gate again. The == 0.500 pin above is
    # stricter; this band is the estimator-class guard. Until 2026-10-17 the loop read an
    # "aggregate" key the promoted files do not have and ran no check (H55). A promoted file
    # without floor_drift0 now fails by KeyError instead of becoming a no-op.
    import glob as _glob
    for rp in sorted(_glob.glob(os.path.join(ARTROOT, "expB2", "*cg_rescore*.json"))):
        with open(rp, encoding="utf-8") as fh:
            rs = json.load(fh)
        name = os.path.basename(rp)
        for agent, floor in rs["floor_drift0"].items():
            check_true(f"cg re-score L0 floor in chance band ({name}:{agent})",
                       0.4 <= float(floor) <= 0.6)

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
              _seed_mean(rs["strong_drift"]["survival"], "mp_fixed_per_seed"), quoted, tol=5e-4)

    print("== FINDINGS Exp C: emergence under selection, fixed-code re-run (N=48 G=30 3 seeds) ==")
    # This block verifies the fixed-code re-run (artifact git_commit_at_run a0cb850; FINDINGS
    # 13.D). The invalidated pilot (git_commit 9758202, FINDINGS 13.C) was overwritten in place
    # by bfc14bd and survives only as
    # `git show 3c3e4f4:artifacts/expC/emergence_pilot_summary.json`. Registered verdict:
    # uninformative (13.F).
    print("  [record note] Exp C checks verify the fixed-code re-run (a0cb850, FINDINGS 13.D); "
          "registered verdict uninformative (13.F)")
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
    check("pinned: H2 integrity gate alpha=1 survival mean (0.752)",
          h2["integrity"]["survival_mean_alpha1"], 0.752)
    check_true("H2 integrity pools bit-match saved dumps (determinism check #5)",
               h2["integrity"]["pools_bit_match"] is True)
    # L0 anchor: alpha=0 survival equivalent to chance (ROPE [0.45, 0.55])
    h2rr = rope_test(h2_surv["a0.00"]["per_seed"], rope=(0.45, 0.55))
    check("H2 L0 anchor alpha=0 survival mean (0.506)", h2rr.mean, 0.506)
    check_true("H2 L0 anchor accepts equivalence to chance (ROPE)", h2rr.accept)
    check_true("H2 L0 anchor recompute matches stored accept",
               h2rr.accept == bool(h2["l0_anchor_alpha0"]["accept_equiv"]))
    # untrained floor control: flat near chance at EVERY alpha at hidden 8, so there the
    # collapse is specific to the learned survival signal. At hidden 7 the floor rises with
    # the seam (FINDINGS 14, note of 2026-10-10; artifacts/expH2/summary_h7.json, checked below).
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
            m = float(np.mean(ps))
            if not (math.isfinite(m) and math.isfinite(float(e["mean"])) and abs(m - e["mean"]) <= 1e-4):
                bad += 1
            if int(sum(v >= BAR for v in ps)) != e["n_ge_065"]:
                bad += 1
            if "per_seed_full" in e and any(not abs(a - b) <= 5.1e-5
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
        check(f"pinned: h{hid} gn sigma_v (0.010)", doc["config"]["gn_sigma_v"], 0.01)
        # integrity gate: 60 reloaded agents regenerate pools bit-identically and the
        # drift-0.45 survival mean reproduces the published headline
        integ = doc["integrity"]
        check_true(f"h{hid} integrity gate PASSED in run log", integ.get("gate_passed") is True)
        check(f"pinned: h{hid} integrity reproduces published survival ({exp['pub']:.3f})",
              integ.get("survival_mean_reproduced", float("nan")), exp["pub"])
        check(f"pinned: h{hid} published_target_check recorded ({exp['pub']:.3f})",
              doc["config"]["published_target_check"], exp["pub"])
        check_int(f"h{hid} integrity: 60 reloaded agents ok", integ.get("n_integrity_ok_lines", -1), 60)
        # capacity ladder oracles are intentionally sub-band
        lo = doc["ladder_oracles"]
        check(f"pinned: h{hid} ladder oracle h16 (0.788)", lo["h16"], 0.788)
        check(f"pinned: h{hid} ladder oracle h32 (0.656)", lo["h32"], 0.656)
        check(f"pinned: h{hid} ladder oracle h64 (0.603)", lo["h64"], 0.603)
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
    check("pinned: gate-0 gn selected sigma_v (0.010)", sel["sigma_v"], 0.01)
    check("pinned: gate-0 gn oracle AUROC (0.865)", sel["oracle_auroc"], 0.865)
    check("gate-0 gn untrained floor (0.448)", _seed_mean(sel, "floor_per_seed"), 0.448)
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
    check("pinned: L1 gate-0 chosen delta (0.023)", gd["chosen_delta"], 0.023)
    check("pinned: L1 gate-0 oracle AUROC at delta (0.873)", gd["chosen_auroc"], 0.873)
    check_true("L1 gate-0 delta row leakage clean",
               gd["chosen_row"] is not None and gd["chosen_row"]["leakage_pass"] is True)
    check("pinned: L1 noise comparator sigma_o (0.010)", gn_["chosen_sigma"], 0.01)
    check("pinned: L1 noise comparator oracle AUROC (0.873)", gn_["chosen_row"]["oracle_auroc"], 0.873)
    check_true("L1 noise comparator leakage clean", gn_["chosen_row"]["leakage_pass"] is True)
    check_true("L1 gate-0 sensor sigma 0.01 shared by delta and noise calibrations",
               abs(gd["sensor_sigma"] - 0.01) <= 1e-12
               and abs(gn_["chosen_row"]["sensor_sigma"] - 0.01) <= 1e-12)
    # organism result at the headline grid (n = 10, three arms)
    hd = org["headline_drift"]
    check("pinned: L1 headline drift key is 0.023", float(hd), 0.023)
    l1s = [float(v) for v in org["arms"][hd]["survival"]["pool_target"]]
    l1p = [float(v) for v in org["arms"][hd]["predictor"]["pool_target"]]
    l1u = [float(v) for v in org["arms"][hd]["untrained"]["pool_target"]]
    check_int("L1 n seeds (10)", len(l1s), 10)
    check("L1 survival pooled target (0.533)", float(np.mean(l1s)), 0.533)
    m_, lo_, hi_ = mean_ci(l1s)
    check("L1 survival boot lo (0.509)", lo_, 0.509)
    check("L1 survival boot hi (0.556)", hi_, 0.556)
    # 2026-10-10: the t interval and the TOST against chance the manuscript quotes.
    l1tlo, l1thi = t_ci(l1s)
    check("L1 survival t lo (0.506)", l1tlo, 0.506)
    check("L1 survival t hi (0.561)", l1thi, 0.561)
    l1eq = equivalence_test(l1s, h0=0.5, margin=0.05)
    check("L1 survival TOST p against chance (0.147)", l1eq.p_value, 0.147)
    check_true("L1 survival is not shown equivalent to chance", not l1eq.equivalent)
    with open(os.path.join(ARTROOT, "..", "docs", "FINDINGS.md"), encoding="utf-8") as fh:
        _f147 = " ".join(fh.read().split())
    check_true("FINDINGS 14.7 quotes the L1 t interval and TOST p",
               f"[{l1tlo:.3f}, {l1thi:.3f}]" in _f147
               and f"does not accept equivalence, p = {l1eq.p_value:.3f}" in _f147)
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
    # 2026-10-12: the agent-accessible L1 ceiling. The grid-residual oracle is blind to the
    # static Hyd/Hydmax offset; the episode-mean probe reads it.
    _l1c = _load_art("expL1", "observation_ceiling.json")
    # The artifact stores these five as scalars with no per-seed values, so they are pins.
    check("pinned: L1 observation ceiling, grid-residual oracle (0.782)",
          _l1c["grid_residual_oracle"], 0.782)
    check("pinned: L1 observation ceiling, episode-mean probe (0.977)",
          _l1c["episode_mean_probe"], 0.977)
    check("pinned: L1 observation ceiling, Hyd channel ablated (0.627)",
          _l1c["hyd_ablated_probe"], 0.627)
    check("pinned: L1 observation ceiling, L0 control (0.478)", _l1c["l0_control"], 0.478)
    check("pinned: L1 observation ceiling, same-seed twins (0.995)", _l1c["same_seed_twins"], 0.995)
    check_true("L1 ceilings readable (energy / food anchors > 0.6 in every survival pool)",
               all(v > 0.6 for v in org["arms"][hd]["survival"]["pool_anchor_energy"]
                   + org["arms"][hd]["survival"]["pool_anchor_food"]))
    check_true("L1 run provenance: commit 8a71593, single fingerprint",
               org["git_commit_at_run"] == ["8a71593"] and bool(org["fingerprint"]))
    # A1 graded seam (delta ladder): flat near chance, integrity bit-match
    a1 = abl["a1_graded_seam"]
    check_true("L1 A1 integrity: pools bit-match saved dumps",
               a1["integrity"]["checked"] and a1["integrity"]["pools_bit_match"] is True)
    check("pinned: L1 A1 integrity reproduces headline survival (0.533)",
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
    check("pinned: L1 A2 noise sigma (0.010)", nz["noise_sigma"], 0.01)
    check("L1 A2 noise survival (0.525)", float(np.mean(nz["survival_per_seed"])), 0.525)
    check("L1 A2 noise untrained (0.509)", float(np.mean(nz["untrained_per_seed"])), 0.509)
    check("L1 A2 noise predictor (0.516)", float(np.mean(nz["predictor_per_seed"])), 0.516)
    check_true("L1 A2 frozen rule fails both clauses (recomputed; verdict H2_SUPPORTED)",
               float(np.mean(nz["survival_per_seed"])) < BAR and nz["gn_rule_pass"] is False
               and nz["gn_verdict"] == "H2_SUPPORTED")
    # A2 observation-channel localization at L1
    lm = lob["masks"]
    check("pinned: L1 obs delta (0.023)", lob["config"]["l1_delta"], 0.023)
    check("pinned: L1 obs sensor sigma (0.010)", lob["config"]["sensor_sigma"], 0.01)
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
    check("pinned: B-v3 gate 4 exception reward_sum AUROC (0.393)", fails[0]["value"], 0.393)
    check("pinned: B-v3 gate 4 exception max_abs_dev (0.107)", fails[0]["max_abs_dev"], 0.107)
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
    check("10.4.2 survival target reproduces 0.752", _seed_mean(sv["target"]), 0.752)
    check("10.4.2 survival resid_trace reproduces 10.4.1's 0.723", _seed_mean(sv["resid_trace"]), 0.723)
    check("10.4.2 survival obs_trace_only (0.709)", _seed_mean(sv["obs_trace_only"]), 0.709)
    check("10.4.2 survival resid_obs (0.731)", _seed_mean(sv["resid_obs"]), 0.731)
    lo, hi = t_ci(sv["resid_obs"]["per_seed"])
    check("10.4.2 survival resid_obs t90 lo (0.690)", lo, 0.690)
    check("10.4.2 survival resid_obs t90 hi (0.772)", hi, 0.772)
    check_int("10.4.2 survival resid_obs seeds >= 0.65 (8)", sv["resid_obs"]["n_ge_065"], 8)
    check("10.4.2 survival resid_obs_int (0.758)", _seed_mean(sv["resid_obs_int"]), 0.758)
    check("10.4.2 survival resid_obs_beh (0.670)", _seed_mean(sv["resid_obs_beh"]), 0.670)
    lo, hi = t_ci(sv["resid_obs_beh"]["per_seed"])
    check("10.4.2 survival resid_obs_beh t90 lo (0.638)", lo, 0.638)
    check("10.4.2 survival resid_obs_beh t90 hi (0.702)", hi, 0.702)
    check("10.4.2 predictor resid_obs (0.542)",
          _seed_mean(se["aggregate"]["d=0.45 predictor"]["resid_obs"]), 0.542)
    check("10.4.2 untrained resid_obs (0.534)",
          _seed_mean(se["aggregate"]["d=0.45 untrained"]["resid_obs"]), 0.534)
    check("10.4.2 drift-0 survival resid_obs floor (0.504)",
          _seed_mean(se["aggregate"]["d=0.00 survival"]["resid_obs"]), 0.504)
    check("10.4.2 drift-0 untrained resid_obs floor (0.472)",
          _seed_mean(se["aggregate"]["d=0.00 untrained"]["resid_obs"]), 0.472)
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
        check(f"10.8 {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"10.8 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv = ab["arms"][ab["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.8 survival t90 lo (0.549)", lo, 0.549)
    check("10.8 survival t90 hi (0.654)", hi, 0.654)
    check_int("10.8 survival seeds >= 0.65 (1)", sv["n_ge_065"], 1)
    check("10.8 survival resid_trace (0.646)", _seed_mean(ab["behavior_audit"]["survival"]["resid_trace"]), 0.646)
    l0_gate_reproduces("10.8", ab)
    check_true("10.8 all gates pass", ab["gates"]["l0_tost"]["equivalent"] and ab["gates"]["l0_rope"]["accept"]
               and ab["gates"]["pool_leak_clean_all"] and ab["gates"]["untrained_floor_ok"]
               and all(v["pass"] == v["n"] for v in ab["gates"]["engagement"].values()))
    check_true("10.8 auxiliary-conditional cell: survival below bar, predictor margin not met",
               (not ab["decision"]["pass_bar"]) and (not ab["decision"]["pass_margin_predictor"]))
    print("\n== FINDINGS 10.9: second fingerprint instance (G seed 1, hidden 10) ==")
    si = _load_art("expB2", "second_instance_l3_h10_gseed1.json")
    check_int("10.9 gate 0 selected hidden (10)", int(si["gate0_calibration"]["selected_hidden"]), 10)
    rows = {int(r["hidden"]): r for r in si["gate0_calibration"]["rows"]}
    check("pinned: 10.9 gate 0 hidden 8 oracle (0.928)", rows[8]["oracle_auroc"], 0.928)
    check("10.9 gate 0 hidden 8 floor dirty (0.664)", _seed_mean(rows[8], "floor_per_seed"), 0.664)
    check_true("10.9 gate 0 hidden 8 fails at seed 1", not rows[8]["passes_gate0"])
    check("pinned: 10.9 gate 0 hidden 10 oracle (0.893)", rows[10]["oracle_auroc"], 0.893)
    check("10.9 gate 0 hidden 10 floor (0.484)", _seed_mean(rows[10], "floor_per_seed"), 0.484)
    check_true("10.9 gate 0 hidden 10 passes", bool(rows[10]["passes_gate0"]))
    for arm, ref in (("survival", 0.639), ("predictor", 0.534), ("untrained", 0.514)):
        blk = si["arms"][si["dmax"]][arm]["pool_target"]
        check(f"10.9 {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"10.9 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv = si["arms"][si["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.9 survival t90 lo (0.610)", lo, 0.610)
    check("10.9 survival t90 hi (0.669)", hi, 0.669)
    check_int("10.9 survival seeds >= 0.65 (5)", sv["n_ge_065"], 5)
    rt = si["behavior_audit"]["survival"]["resid_trace"]
    check("10.9 survival resid_trace (0.658)", _seed_mean(rt), 0.658)
    lo, hi = t_ci(rt["per_seed"])
    check("10.9 survival resid_trace t90 lo (0.625)", lo, 0.625)
    check("10.9 survival resid_trace t90 hi (0.691)", hi, 0.691)
    l0_gate_reproduces("10.9", si)
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
    check("pinned: 10.9-gpu gate 0 hidden 10 oracle on CUDA (0.893)", g10["oracle_auroc"], 0.893)
    check("10.9-gpu gate 0 hidden 10 floor on CUDA (0.521)", _seed_mean(g10, "floor_per_seed"), 0.521)
    for arm, ref in (("survival", 0.612), ("predictor", 0.529), ("untrained", 0.521)):
        blk = sg["arms"][sg["dmax"]][arm]["pool_target"]
        check(f"10.9-gpu {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"10.9-gpu {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    svg = sg["arms"][sg["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(svg["per_seed"])
    check("10.9-gpu survival t90 lo (0.582)", lo, 0.582)
    check("10.9-gpu survival t90 hi (0.642)", hi, 0.642)
    check_int("10.9-gpu survival seeds >= 0.65 (3)", svg["n_ge_065"], 3)
    rtg = sg["behavior_audit"]["survival"]["resid_trace"]
    check("10.9-gpu survival resid_trace (0.638)", _seed_mean(rtg), 0.638)
    lo, hi = t_ci(rtg["per_seed"])
    check("10.9-gpu survival resid_trace t90 lo (0.612)", lo, 0.612)
    check("10.9-gpu survival resid_trace t90 hi (0.663)", hi, 0.663)
    check("10.9-gpu lead over predictor (+0.083)", _lead(sg, "predictor"), 0.083)
    check("10.9-gpu lead over untrained (+0.091)", _lead(sg, "untrained"), 0.091)
    check_true("10.9-gpu rule not met: bar missed, both margin clauses pass",
               (not sg["decision"]["pass_bar"]) and sg["decision"]["pass_margin_predictor"]
               and sg["decision"]["pass_margin_untrained"])
    l0_gate_reproduces("10.9-gpu", sg)
    check_true("10.9-gpu L0 gate open as stated (TOST and ROPE both reject at n = 10)",
               (not sg["gates"]["l0_tost"]["equivalent"]) and (not sg["gates"]["l0_rope"]["accept"]))
    check("10.9-gpu L0 survival mean (0.539)", _seed_mean(sg["arms"]["0.0"]["survival"]["pool_target"]), 0.539)
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
    check("h7 10.4.2 survival target reproduces 0.737", _seed_mean(sv7["target"]), 0.737)
    check("h7 10.4.2 survival obs_trace_only (0.697)", _seed_mean(sv7["obs_trace_only"]), 0.697)
    check("h7 10.4.2 survival resid_trace, seven-channel (0.739)", _seed_mean(sv7["resid_trace"]), 0.739)
    check("h7 10.4.2 survival resid_obs (0.684)", _seed_mean(sv7["resid_obs"]), 0.684)
    lo, hi = t_ci(sv7["resid_obs"]["per_seed"])
    check("h7 10.4.2 survival resid_obs t90 lo (0.627)", lo, 0.627)
    check("h7 10.4.2 survival resid_obs t90 hi (0.741)", hi, 0.741)
    check_int("h7 10.4.2 survival resid_obs seeds >= 0.65 (7)", sv7["resid_obs"]["n_ge_065"], 7)
    check("h7 10.4.2 survival resid_obs_int (0.695)", _seed_mean(sv7["resid_obs_int"]), 0.695)
    check("h7 10.4.2 survival resid_obs_beh (0.671)", _seed_mean(sv7["resid_obs_beh"]), 0.671)
    lo, hi = t_ci(sv7["resid_obs_beh"]["per_seed"])
    check("h7 10.4.2 survival resid_obs_beh t90 lo (0.618)", lo, 0.618)
    check("h7 10.4.2 survival resid_obs_beh t90 hi (0.725)", hi, 0.725)
    p7o = _seed_mean(s7["aggregate"]["d=0.45 predictor"]["resid_obs"])
    check("h7 10.4.2 predictor resid_obs (0.589)", p7o, 0.589)
    check("h7 10.4.2 predictor raw target (0.714)", _seed_mean(s7["aggregate"]["d=0.45 predictor"]["target"]), 0.714)
    check("h7 10.4.2 untrained resid_obs (0.549)", _seed_mean(s7["aggregate"]["d=0.45 untrained"]["resid_obs"]), 0.549)
    check("h7 10.4.2 untrained obs_trace_only (0.643)", _seed_mean(s7["aggregate"]["d=0.45 untrained"]["obs_trace_only"]), 0.643)
    check("h7 10.4.2 survival-minus-predictor lead under the sensory control (+0.095)",
          _seed_mean(sv7["resid_obs"]) - p7o, 0.095)
    check("h7 10.4.2 drift-0 survival resid_obs floor (0.504)", _seed_mean(s7["aggregate"]["d=0.00 survival"]["resid_obs"]), 0.504)
    check("h7 10.4.2 drift-0 untrained resid_obs floor (0.472)", _seed_mean(s7["aggregate"]["d=0.00 untrained"]["resid_obs"]), 0.472)
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
            check(f"fold h{h} survival {met} legacy reproduces published ({leg})",
                  float(np.mean(_fold_seeds(fr[h], met, "legacy"))), leg, tol=tol)
            check(f"fold h{h} survival {met} explicit ({exp})",
                  float(np.mean(_fold_seeds(fr[h], met, "explicit"))), exp)
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
    check("fold cg h8 legacy reproduces the forward tail (0.666)", _seed_mean(cg[(8, "legacy")]["d0.45_survival"], "cg_tail_per_seed"), 0.666)
    check("fold cg h7 legacy reproduces the reverse tail (0.684)", _seed_mean(cg[(7, "legacy")]["d0.45_survival"], "cg_tail_per_seed"), 0.684)
    check("fold cg h8 explicit forward tail (0.677)", _seed_mean(cg[(8, "explicit")]["d0.45_survival"], "cg_tail_per_seed"), 0.677)
    check("fold cg h7 explicit reverse tail (0.677)", _seed_mean(cg[(7, "explicit")]["d0.45_survival"], "cg_tail_per_seed"), 0.677)
    for h in (8, 7):
        a = cg[(h, "explicit")]
        check_true(f"fold cg h{h} explicit: both frozen clauses still pass",
                   a["d0.45_survival"]["cg_tail_mean"] >= 0.65
                   and a["d0.45_survival"]["cg_tail_mean"] > a["d0.45_untrained"]["cg_tail_mean"] + 0.05)
    for h, leg, exp in ((8, 0.893, 0.903), (7, 0.855, 0.848)):
        mpl = _fold_art(f"mp_h{h}_legacy.json")["aggregate"]["d0.45_survival"]
        mpe = _fold_art(f"mp_h{h}_explicit.json")["aggregate"]["d0.45_survival"]
        check_true(f"fold mp h{h} legacy reproduces every stored value", bool(mpl["all_reproduce_stored"]))
        check(f"fold mp h{h} legacy survival ({leg})", _seed_mean(mpl, "mp_fixed_per_seed"), leg)
        check(f"fold mp h{h} explicit survival ({exp})", _seed_mean(mpe, "mp_fixed_per_seed"), exp)
    g0 = {sch: {int(r["hidden"]): r for r in _fold_art(f"gate0_h8h7_{sch}.json")["rows"]} for sch in ("legacy", "explicit")}
    check("pinned: fold gate0 legacy h8 oracle reproduces (0.928)", g0["legacy"][8]["oracle_auroc"], 0.928)
    check("fold gate0 legacy h8 floor reproduces (0.482)", _seed_mean(g0["legacy"][8], "floor_per_seed"), 0.482)
    check("pinned: fold gate0 legacy h7 oracle reproduces (0.922)", g0["legacy"][7]["oracle_auroc"], 0.922)
    check("fold gate0 legacy h7 floor reproduces (0.566)", _seed_mean(g0["legacy"][7], "floor_per_seed"), 0.566)
    check_true("fold gate0 legacy: both capacities pass", g0["legacy"][8]["passes_gate0"] and g0["legacy"][7]["passes_gate0"])
    check("fold gate0 explicit h8 floor (0.508)", _seed_mean(g0["explicit"][8], "floor_per_seed"), 0.508)
    check_true("fold gate0 explicit h8 passes", bool(g0["explicit"][8]["passes_gate0"]))
    check("section 16: gate0 explicit h7 floor (0.615)", _seed_mean(g0["explicit"][7], "floor_per_seed"), 0.615)
    check("pinned: section 16: gate0 explicit h7 oracle still in band (0.918)", g0["explicit"][7]["oracle_auroc"], 0.918)
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
    check("pinned: h8-seed2 gate 0 oracle (0.939)", g8["oracle_auroc"], 0.939)
    check("h8-seed2 gate 0 floor (0.540)", _seed_mean(g8, "floor_per_seed"), 0.540)
    check_true("h8-seed2 gate 0 passes with a clean mechanical floor",
               bool(g8["passes_gate0"]) and bool(g8["mech_leak_pass"]) and bool(g8["in_band"]))
    for arm, ref in (("survival", 0.676), ("predictor", 0.627), ("untrained", 0.550)):
        blk = s2["arms"][s2["dmax"]][arm]["pool_target"]
        check(f"h8-seed2 {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"h8-seed2 {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    sv2 = s2["arms"][s2["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv2["per_seed"])
    check("h8-seed2 survival t90 lo (0.636)", lo, 0.636)
    check("h8-seed2 survival t90 hi (0.717)", hi, 0.717)
    check_int("h8-seed2 survival seeds >= 0.65 (6)", sv2["n_ge_065"], 6)
    rt2 = s2["behavior_audit"]["survival"]["resid_trace"]
    check("h8-seed2 survival resid_trace (0.660)", _seed_mean(rt2), 0.660)
    lo, hi = t_ci(rt2["per_seed"])
    check("h8-seed2 survival resid_trace t90 lo (0.631)", lo, 0.631)
    check("h8-seed2 survival resid_trace t90 hi (0.689)", hi, 0.689)
    check("h8-seed2 survival behavior_trace_only (0.745)", _seed_mean(s2["behavior_audit"]["survival"]["behavior_trace_only"]), 0.745)
    ba2 = s2["behavior_audit"]
    check("h8-seed2 predictor behavior_trace_only (0.719)", _seed_mean(ba2["predictor"]["behavior_trace_only"]), 0.719)
    check("h8-seed2 untrained behavior_trace_only (0.683)", _seed_mean(ba2["untrained"]["behavior_trace_only"]), 0.683)
    check("h8-seed2 predictor resid_trace (0.606)", _seed_mean(ba2["predictor"]["resid_trace"]), 0.606)
    check("h8-seed2 untrained resid_trace (0.563)", _seed_mean(ba2["untrained"]["resid_trace"]), 0.563)
    check("h8-seed2 resid_trace lead over predictor (+0.054)", _seed_mean(rt2) - _seed_mean(ba2["predictor"]["resid_trace"]), 0.054)
    check("h8-seed2 resid_trace lead over untrained (+0.098)", _seed_mean(rt2) - _seed_mean(ba2["untrained"]["resid_trace"]), 0.098)
    check_int("h8-seed2 survival resid_trace seeds >= 0.65 (5)", sum(v >= BAR for v in rt2["per_seed"]), 5)
    d2 = s2["decision"]
    check("h8-seed2 lead over predictor (+0.0491)", _lead(s2, "predictor"), 0.0491, tol=0.0001)
    check("h8-seed2 lead over untrained (+0.126)", _lead(s2, "untrained"), 0.126)
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
    l0_gate_reproduces("h8-seed2", s2)
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
        check(f"skill-match {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"skill-match {arm}: stored mean reproduces per-seed mean", float(np.mean(blk["per_seed"])), blk["mean"])
    svk = sk["arms"][sk["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(svk["per_seed"])
    check("skill-match survival t90 lo (0.679)", lo, 0.679)
    check("skill-match survival t90 hi (0.755)", hi, 0.755)
    check_int("skill-match survival seeds >= 0.65 (8)", svk["n_ge_065"], 8)
    rtk = sk["behavior_audit"]["survival"]["resid_trace"]
    check("skill-match survival resid_trace (0.724)", _seed_mean(rtk), 0.724)
    lo, hi = t_ci(rtk["per_seed"])
    check("skill-match survival resid_trace t90 lo (0.690)", lo, 0.690)
    check("skill-match survival resid_trace t90 hi (0.757)", hi, 0.757)
    check_int("skill-match survival resid_trace seeds >= 0.65 (9)", sum(v >= BAR for v in rtk["per_seed"]), 9)
    bak = sk["behavior_audit"]
    check("skill-match predictor resid_trace (0.590)", _seed_mean(bak["predictor"]["resid_trace"]), 0.590)
    check("skill-match untrained resid_trace (0.539)", _seed_mean(bak["untrained"]["resid_trace"]), 0.539)
    check("skill-match survival behavior_trace_only (0.838)", _seed_mean(bak["survival"]["behavior_trace_only"]), 0.838)
    dk = sk["decision"]
    check("skill-match lead over predictor (+0.129)", _lead(sk, "predictor"), 0.129)
    check("skill-match lead over untrained (+0.204)", _lead(sk, "untrained"), 0.204)
    check_true("skill-match probe passes bar, t-CI excludes it, and both margins pass",
               bool(dk["pass_bar"]) and bool(dk["t90_excludes_bar"])
               and bool(dk["pass_margin_predictor"]) and bool(dk["pass_margin_untrained"]))
    # The frozen match clause: an overshoot voids the mediation inference even though
    # every probe clause above passes. Both halves must stay true together.
    sm = sk["skill_match"]
    _sk_r = _seed_mean(sm, "per_seed_returns")
    _sk_se = float(np.std(sm["per_seed_returns"], ddof=1) / np.sqrt(len(sm["per_seed_returns"])))
    check("pinned: skill-match reference return (-0.219)", sm["reference_return"], -0.219)
    check("pinned: skill-match window lo (-0.269)", sm["window"][0], -0.269)
    check("pinned: skill-match window hi (-0.169)", sm["window"][1], -0.169)
    check("skill-match stage-2 return R (-0.1559)", _seed_mean(sm, "per_seed_returns"), -0.1559, tol=0.0001)
    check("skill-match R reproduces the per-seed mean", float(np.mean(sm["per_seed_returns"])), sm["match_return"])
    check_int("skill-match stage-2 seeds (10)", sm["n_seeds"], 10)
    check("skill-match stage-2 return se (0.096)", _sk_se, 0.096)
    check("skill-match overshoot past the window (+0.0131)", _sk_r - sm["window"][1], 0.0131, tol=0.0001)
    check("skill-match overshoot in standard errors (0.14)", (_sk_r - sm["window"][1]) / _sk_se, 0.14, tol=0.005)
    check("skill-match deviation from reference (+0.0631)", _sk_r - sm["reference_return"], 0.0631, tol=0.0001)
    check_true("skill-match R is outside the window and above it",
               (not sm["in_window"]) and sm["match_return"] > sm["window"][1])
    check_true("skill-match probe does not read below the bar", not sm["probe_below_bar"])
    check_true("skill-match frozen rule gives no skill-mediation verdict",
               sm["verdict"].startswith("NO SKILL-MEDIATION VERDICT"))
    check_true("skill-match gates pass except L0 (engagement, leak, floor, deaths)",
               sk["gates"]["pool_leak_clean_all"] and sk["gates"]["untrained_floor_ok"]
               and sk["gates"]["deaths_total"] == 0
               and all(v["pass"] == v["n"] for v in sk["gates"]["engagement"].values()))
    # 2026-10-10: speed_min is the lowest pool, as the registered gate 3 reads it; 0.837 was the
    # lowest arm mean (kept in the artifact as speed_min_of_arm_means).
    check("pinned: skill-match speed positive control, lowest pool (0.790)", sk["gates"]["speed_min"], 0.790)
    check("skill-match L0 survival mean (0.539)", _seed_mean(sk["arms"]["0.0"]["survival"]["pool_target"]), 0.539)
    check("skill-match L0 TOST p (0.285)",
          equivalence_test(sk["arms"]["0.0"]["survival"]["pool_target"]["per_seed"], margin=0.05).p_value, 0.285)
    check("skill-match L0 ROPE P (0.736)",
          rope_test(sk["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]).p_in_rope, 0.736)
    l0_gate_reproduces("skill-match", sk)
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
        check(f"10.8 device control {arm} pooled target ({ref})", _seed_mean(blk), ref)
        check(f"10.8 device control {arm}: stored mean reproduces per-seed mean",
              float(np.mean(blk["per_seed"])), blk["mean"])
    sv = dc["arms"][dc["dmax"]]["survival"]["pool_target"]
    lo, hi = t_ci(sv["per_seed"])
    check("10.8 device control survival t90 lo (0.668)", lo, 0.668)
    check("10.8 device control survival t90 hi (0.791)", hi, 0.791)
    check_int("10.8 device control survival seeds >= 0.65 (8)", sv["n_ge_065"], 8)
    rt = dc["behavior_audit"]["survival"]["resid_trace"]
    check("10.8 device control survival resid_trace (0.710)", _seed_mean(rt), 0.710)
    lo, hi = t_ci(rt["per_seed"])
    check("10.8 device control survival resid_trace t90 lo (0.656)", lo, 0.656)
    check("10.8 device control survival resid_trace t90 hi (0.764)", hi, 0.764)
    l0_gate_reproduces("10.8 device control", dc)
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
    check("nonlinear survival resid_obs_beh_mlp (0.654)", _seed_mean(sv["resid_obs_beh_mlp"]), 0.654)
    lo, hi = t_ci(sv["resid_obs_beh_mlp"]["per_seed"])
    check("nonlinear survival resid_obs_beh_mlp t90 lo (0.621)", lo, 0.621)
    check("nonlinear survival resid_obs_beh_mlp t90 hi (0.687)", hi, 0.687)
    check_int("nonlinear survival resid_obs_beh_mlp seeds >= 0.65 (6)", sv["resid_obs_beh_mlp"]["n_ge_065"], 6)
    check("nonlinear survival resid_obs_mlp (0.653)", _seed_mean(sv["resid_obs_mlp"]), 0.653)
    check("nonlinear untrained resid_obs_beh_mlp (0.509)",
          _seed_mean(nl["aggregate"]["d=0.45 untrained"]["resid_obs_beh_mlp"]), 0.509)
    check("nonlinear predictor resid_obs_beh_mlp (0.542)",
          _seed_mean(nl["aggregate"]["d=0.45 predictor"]["resid_obs_beh_mlp"]), 0.542)
    check("nonlinear run reproduces the linear joint control (0.670)", _seed_mean(sv["resid_obs_beh"]), 0.670)
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
        check(f"pinned: 15 {cell} @ sigma {sig} ({ref})", cells[(cell, sig)]["oracle_auroc"], ref)
    check_true("15 mechanical leakage clean in every cell", all(r["leakage_pass"] for r in mh["rows"]))
    check("pinned: 15 L0 anchor mean over noise seeds (0.475)", mh["l0_anchor_noise_seeds"]["mean"], 0.475)
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

    # The anonymized supplement leaves out the film VO and music spec (SUPPLEMENT_EXCLUDE in
    # scripts/reproduce.py). Only there may a file this guard reads be absent: a checkout has
    # no README_SUPPLEMENT.md, so in CI a missing file still raises.
    _in_supplement = os.path.exists(os.path.join(root, "README_SUPPLEMENT.md"))

    def _read_or_skip(relpath: str) -> str | None:
        global n_skipped
        if _in_supplement and not os.path.exists(os.path.join(root, relpath)):
            n_skipped += 1
            print(f"  [skip] {relpath}: not shipped in the anonymized supplement")
            return None
        return _read(relpath)

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
    check_true("FINDINGS no longer says 'H2 confirmed'",
               "H2 confirmed" not in _read("docs/FINDINGS.md"))
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
    _vo = _read_or_skip("docs/specs/2026-10-01-two-minds-vo-music.md")
    if _vo is not None:
        check_true("the Two Minds VO spec records that its survival-alone line has a shelf life",
                   "The one line with a shelf life" in _vo)

    # ---- derived-doc guard: sensory echo (10.4.2), H2 (14), Exp C (13.D) -------
    # The site carried Experiment C as "next" for two months after 13.D closed it,
    # and never mentioned the sensory-echo or graded-seam controls. Pin them.
    print("\n== derived-doc guard (10.4.2 sensory echo, 14 H2, 13.D Exp C) ==")
    for relpath, needle, label in [
        ("index.html", "0.731", "index.html carries the sensory-echo control"),
        ("index.html", "0.654", "index.html carries the nonlinear joint control"),
        ("index.html", "0.506", "index.html carries the graded-seam collapse"),
        ("index.html", "0.509", "index.html carries the Exp C final AUROC"),
    ]:
        check_true(label, needle in _read(relpath))
    # 2026-10-10: the July demo loops were retired and the readout chart plots per-seed
    # corrected readouts, not a training curve.
    for relpath, banned in [("index.html", "EXP C · next"),
                            ("index.html", "no matter how loud the artifact"),
                            ("index.html", "loop-idle-clouds"), ("index.html", "loop-survival-clouds"),
                            ("index.html", "loop-race"), ("index.html", "loop-brain-pair"),
                            ("viz/player/index.html", "loop-race"),
                            ("viz/player/index.html", "loop-brain-pair"),
                            ("index.html", "training progress"),
                            ("index.html", "survival pressure applied")]:
        check_true(f"{relpath} no longer says '{banned}'", banned not in _read(relpath))
    # 2026-10-10: Experiment C's registered verdict is UNINFORMATIVE (FINDINGS 13.F). Section 7
    # of PREREGISTRATION_C interprets no contrast unless every gate passes and the speed control
    # (gate 5) fails; gate 1 fails only against the 0.005 margin of the unratified section-14
    # draft (13.F, second note of 2026-10-10). "Validated null" is no longer the verdict wording
    # and the pin that required it on index.html is retired. The summary surfaces state the
    # uninformative verdict with a pointer to 13.F and may not use the old phrases. The 13.D
    # heading, the 13.F reconciliation and the frozen PREREGISTRATION_C keep the historical
    # phrase and are not watched. index.html keeps one "validated null", in the sentence that
    # says 13.F supersedes it, so there the phrase is allowed only with "13.F supersedes" right
    # after it.
    _fx = _read("docs/FINDINGS.md")
    _expc_surfaces = {
        "FINDINGS banner, TL;DR and section 1": _fx[:_fx.index("## 2. Experiment A")],
        "FINDINGS section 6": _fx[_fx.index("## 6. Status against"):_fx.index("## 7. Levers")],
        "FINDINGS 17.12": _fx[_fx.index("### 17.12"):],
        "docs/PAPER_OUTLINE.md": _read("docs/PAPER_OUTLINE.md"),
        "README.md": _read("README.md"),
        "index.html": _read("index.html"),
        "docs/LEARNING.md": _read("docs/LEARNING.md"),
    }
    for _where, _txt in _expc_surfaces.items():
        _flat = " ".join(_txt.split())
        check_true(f"{_where} states the Exp C verdict as uninformative with a 13.F pointer",
                   "uninformative" in _flat and "13.F" in _flat)
        for _banned in ("validated null", "open test", "evolutionary null"):
            _hits = _find_all(_flat, _banned)
            if _where == "index.html" and _banned == "validated null":
                _hits = [i for i in _hits if "13.F supersedes" not in _flat[i:i + 120]]
            check_true(f"{_where} no longer says '{_banned}'", not _hits)
        # Any listing of gates 1 and 5 together ("gates 1 and 5 fail", "gates 1, 5"), so a
        # card footer cannot bring back the retired routing in a shorter form.
        check_true(f"{_where} does not say gates 1 and 5 decide the routing (13.F, second note)",
                   not re.search(r"gates?\s*1\s*(?:,|and|&|&amp;)\s*5", _flat.lower()))
    # PAPER_OUTLINE row 40: where the quadratic-drag floor first leaves its tolerance,
    # recomputed from the gate-0 sweep rather than copied (it once said eps 4).
    _q0 = _load_art("texture", "gate0_qd.json")
    _q_first = min(r["eps"] for r in _q0["rows"] if not r["floor_ok"])
    _q_prev = max(r["eps"] for r in _q0["rows"] if r["eps"] < _q_first)
    _q_prev_floor = next(r["floor"] for r in _q0["rows"] if r["eps"] == _q_prev)
    check_true(f"PAPER_OUTLINE row 40: the qd floor leaves tolerance from eps {_q_first:g}",
               f"floor leaves tolerance from eps {_q_first:g} ({_q_prev_floor:.3f} at eps {_q_prev:g} "
               f"is inside the {_q0['floor_tol']:g} tolerance)" in _read("docs/PAPER_OUTLINE.md"))
    _pix = _load_art("expC", "emergence_pilot_per_individual_summary.json")
    _ft = sorted(c["indiv_final_treat"]["mean"] for c in _pix["cells"])
    check_true("TL;DR and PAPER_OUTLINE row 27 quote the per-individual final treatment range",
               all(f"final treatment means {_ft[0]:.3f} to {_ft[-1]:.3f}" in " ".join(_t.split())
                   for _t in (_expc_surfaces["FINDINGS banner, TL;DR and section 1"],
                              _expc_surfaces["docs/PAPER_OUTLINE.md"])))

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
            ("C1 verdict", "NOT MET as registered, because the L0 gate is open; the\ndecodability clauses are met"
             if _p["verdict"].startswith("NOT MET as registered (open gate(s): l0)") else _p["verdict"]),
        ] + [(f"C1 correction {d} {q}", f"{x['corrected']['mean'] if q == 'survival_target' else x['corrected']:.3f} | "
              f"{x['historical']['mean'] if q == 'survival_target' else x['historical']:.3f} | "
              f"{x['difference']['mean']:+.3f} {_sv(x['difference']['t90'])}")
             for d, blk in _c1["correction_effect"].items()
             for q, x in blk.items() if q in ("survival_target", "engagement_return")]:
            check_true(f"FINDINGS 17 quotes {label}", needle in _s17)
        check_true("C1 integrity passes (successor bootstrap, baselines bit-identical)",
                   _c1["integrity"]["pass"])
        # H68: the integrity pass is vacuous when the historical cells are absent, so the
        # "all 20 cells" sentence (main.tex, FINDINGS 17) is pinned to the compared counts.
        check_true("C1 baselines compared in all 20 cells",
                   bool(_c1["integrity"]["bit_identical_to_historical"])
                   and all(v["compared"] == 20
                           for v in _c1["integrity"]["bit_identical_to_historical"].values()))
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
        check_true("C2 baselines compared in all 20 cells",
                   bool(_c2["integrity"]["bit_identical_to_historical"])
                   and all(v["compared"] == 20
                           for v in _c2["integrity"]["bit_identical_to_historical"].values()))
        _ac = _v["auxiliary_comparison"]
        check_true("FINDINGS 17.4 quotes the auxiliary contrast",
                   f"**{_ac['c1_minus_c2']:+.3f}**, paired by seed **{_sv2(_ac['paired']['t90'])}**" in _s17n2)
        check_true("FINDINGS 17.4 counts the seeds favoring the decoder",
                   f"{sum(x > 0 for x in _ac['paired']['diff_per_seed'])} of 10 seeds favor the decoder" in _s17n2)
        check_true("17.4 wording matches the frozen auxiliary rule (not met as registered, "
                   "holds on decodability)",
                   (not _ac["holds"]) and _ac["holds_on_decodability_clauses"]
                   and "The auxiliary rule is therefore NOT MET as registered" in _s17n2
                   and "the lead holds on the decodability clauses" in _s17n2)
    _bcp = os.path.join(ARTROOT, "budget_curve.json")
    if _artifact_present(_bcp):
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
    if _artifact_present(_xrp):
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
    if _artifact_present(_l0p):
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
    if _artifact_present(_l1p):
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
    if _artifact_present(_gnp):
        _f14 = _read("docs/FINDINGS.md")
        _s14 = _f14[_f14.index("## 14. H2 substrate-grounding"):_f14.index("## 15.")]
        _gn = _load_art("texture", "T_gn_l3_h8_wm.json")
        texture_l0_gate_reproduces("14.5.1", _gn)
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
    if _artifact_present(_rdp):
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
        check("pinned: 17.5.2: the artifact carries the registered bar", _rd["bar"], BAR)
        check("pinned: 17.5.2: the artifact carries the registered margin", _rd["margin"], 0.05)
        _re = _radj(survival=_rs["standard"]["survival"]["mean"],
                    untrained=_rs["standard"]["untrained"]["mean"],
                    bar=BAR, margin=0.05)
        check_true("17.5.2: the verdict recomputes under the frozen rule",
                   _re["verdict"] == _rs["verdict"])
        check_true("17.5.2 quotes the verdict",
                   f"frozen before the run: {_rs['verdict']}" in _s17)
    _s13 = _f17[_f17.index("## 13. Experiment C"):_f17.index("## 14.")]
    _pip = os.path.join(ARTROOT, "expC", "emergence_pilot_per_individual_summary.json")
    if _artifact_present(_pip):
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
    if _artifact_present(_xevp):
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
        # 17.2.1 note (2026-10-10): what the surrogate costs each policy, paired by seed: return
        # evaluated authentic minus return evaluated at drift 0.45, per cell, for each training
        # world. The manuscript quotes both gaps; they are recomputed here from the same cells.
        from itasorl.stats import paired_contrast as _pcx
        _s17f = " ".join(_s17.split())
        for _td, _pin, _who, _txt in (
                (0.0, 0.582, "authentic-raised",
                 "is {m} for the authentic-raised policy (t-based 90% CI [{lo}, {hi}])"),
                (0.45, 0.498, "surrogate-raised",
                 "and {m} for the surrogate-raised policy ([{lo}, {hi}])")):
            _rows = _xv.get(_td, [])
            _ek = sorted(_rows[0], key=float) if _rows else ["0.00", "0.45"]
            _g = _pcx([r[_ek[0]] for r in _rows], [r[_ek[-1]] for r in _rows]) if _rows else None
            if _g is None:
                check_true(f"17.2.1 note: cells trained at drift {_td} present", False)
                continue
            check(f"17.2.1 note: {_who} policy, paired return gap authentic minus drift 0.45 "
                  f"({_pin:+.3f})", _g["mean"], _pin)
            check_int(f"17.2.1 note: seeds where the {_who} policy returns less at drift 0.45",
                      sum(x > 0 for x in _g["diff_per_seed"]), 10)
            check_true(f"17.2.1 quotes the {_who} policy's paired return gap and interval",
                       _txt.format(m=f"{_g['mean']:+.3f}", lo=f"{_g['t90'][0]:+.3f}",
                                   hi=f"{_g['t90'][1]:+.3f}") in _s17f)
        check_true("17.2.1 quotes the seed count of the paired return gaps",
                   "each policy returns less at drift 0.45 in 10 of 10 seeds" in _s17f)
    _wsp = os.path.join(ARTROOT, "l0_audit", "d045_world_samples_l3_h8_heldout.json")
    if _artifact_present(_wsp):
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
        check("pinned: 17.5.1: the artifact carries the registered bar", _ws["bar"], BAR)
        check("pinned: 17.5.1: the summary carries the registered bar", _wsum["bar"], BAR)
        _re = _wss(_wm[1:], _wm[0], bar=BAR)
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
    if _artifact_present(_pcp):
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
    if _artifact_present(_psp):
        _pa = _load_art("persistence", "corrected_l3_h8_wm.json")["aggregate"]
        for _pk, _pv in _pa.items():
            _ps = _pv.get("window_per_seed")
            if not _ps:
                continue
            check(f"17.7 {_pk} window_mean recomputed from per-seed",
                  float(np.mean(_ps)), _pv["window_mean"])
            _lo, _hi = t_ci90(_ps)
            check(f"17.7 {_pk} window_t90 lo recomputed", float(_lo), _pv["window_t90"][0])
            check(f"17.7 {_pk} window_t90 hi recomputed", float(_hi), _pv["window_t90"][1])
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
    if _artifact_present(_cdp):
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
        # 2026-10-10: the post hoc paired contrasts the manuscript quotes (state minus each
        # readout, survival arm, drift 0.45), recomputed from the per_seed arrays and checked
        # against the quoted values and against the FINDINGS 17.8 table and PAPER_OUTLINE row 46.
        from itasorl.stats import paired_contrast as _pc178
        _st = _ca["d=0.45 survival target"]["per_seed"]
        _o46 = " ".join(_read("docs/PAPER_OUTLINE.md").split())
        for _m, _lab, _qm, _qlo, _qhi, _qp in (
                ("seq_gru", "supervised GRU on the stream", 0.009, -0.011, 0.029, 0.002),
                ("obs_summary_only", "summary features of the observation trace",
                 0.018, -0.004, 0.040, 0.013),
                ("resid_joint_mlp", "the residual the best-fitting control leaves (64-unit MLP)",
                 0.113, 0.065, 0.161, 0.980)):
            _ot = _ca[f"d=0.45 survival {_m}"]["per_seed"]
            check_int(f"17.8 post hoc state minus {_m}: ten paired seeds", min(len(_st), len(_ot)), 10)
            _pc = _pc178(_st, _ot)
            _eq = equivalence_test(np.asarray(_st, float) - np.asarray(_ot, float), h0=0.0, margin=0.05)
            check(f"17.8 post hoc state minus {_m} mean ({_qm:+.3f})", _pc["mean"], _qm)
            check(f"17.8 post hoc state minus {_m} t lo ({_qlo:+.3f})", _pc["t90"][0], _qlo)
            check(f"17.8 post hoc state minus {_m} t hi ({_qhi:+.3f})", _pc["t90"][1], _qhi)
            check(f"17.8 post hoc state minus {_m} TOST p ({_qp:.3f})", _eq.p_value, _qp)
            _txt = (f"{_pc['mean']:+.3f} [{_pc['t90'][0]:+.3f}, {_pc['t90'][1]:+.3f}]")
            check_true(f"FINDINGS 17.8 post hoc row {_m}",
                       f"| {_lab} | {_txt} | {_eq.p_value:.3f} |" in _s17)
            check_true(f"PAPER_OUTLINE row 46 quotes the post hoc contrast {_m}",
                       f"{_txt} (p {_eq.p_value:.3f})" in _o46)
        # 2026-10-10: the stream decoder's recipe and its final training loss (FINDINGS 17.8,
        # docs/METHODS_ARMS.md), read from both control-diagnostics artifacts.
        for _fn in ("corrected_l3_h8_wm.json", "l1_stream_readout.json"):
            if not _artifact_present(os.path.join(ARTROOT, "control_diagnostics", _fn)):
                continue
            _cells = _load_art("control_diagnostics", _fn)["cells"]
            _ftl = [x for c in _cells for x in c["seq_gru"]["final_train_loss"]]
            check_true(f"17.8: the stream decoder of {_fn} is the recorded recipe (64, GRU 96, 60 epochs)",
                       all((c["seq_gru"]["embed"], c["seq_gru"]["hidden"], c["seq_gru"]["epochs"])
                           == (64, 96, 60) for c in _cells))
            check_int(f"17.8: {_fn} records 150 decoder folds", len(_ftl), 150)
            check_true(f"17.8: the decoder's final training loss is below 0.001 in every fold of {_fn}",
                       max(_ftl) < 1e-3)
            check_true(f"17.8: {_fn} scores 220 episodes per cell (about 176 per training fold)",
                       {c["n"] for c in _cells} == {220})
        _s17f = " ".join(_s17.split())
        check_true("FINDINGS 17.8 quotes the decoder's fit",
                   "below 0.001 in every fold" in _s17f
                   and "roughly 176 training episodes per fold" in _s17f)
    for _fam in ("gn", "qd"):
        _tp = os.path.join(ARTROOT, "texture", f"corrected_l3_h8_wm_{_fam}.json")
        if not _artifact_present(_tp):
            continue
        from itasorl.stats import paired_contrast as _pc2
        _ta = _load_art("texture", f"corrected_l3_h8_wm_{_fam}.json")["aggregate"]
        for _q in ("transfer", "fresh", "fresh_paired"):
            _s, _u = _ta[f"survival {_q}"], _ta[f"untrained {_q}"]
            check(f"17.9 {_fam} {_q} survival mean recomputed",
                  float(np.mean(_s["per_seed"])), _s["mean"])
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
    if _artifact_present(_prp):
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
    # 2026-10-10 record notes: every number the dated notes of that day quote from a committed
    # artifact is recomputed here from per-seed or per-cell values and looked up in the text,
    # whitespace-flattened. Numbers the notes take from gitignored run bundles (the hidden-7
    # graded seam, the gate-0 recalibration, the Experiment C run file) are not read.
    print("\n== 2026-10-10 record notes (FINDINGS 3.4, 9, 10.5, 10.9, 12, 13, 14, 14.6, 16, 17) ==")
    from itasorl.stats import paired_contrast as _pc2
    _fx2 = _read("docs/FINDINGS.md")
    _fl = lambda s: " ".join(s.split())
    _sec = lambda a, b: _fl(_fx2[_fx2.index(a):_fx2.index(b)])
    _ivs = lambda c: f"[{c['t90'][0]:+.3f}, {c['t90'][1]:+.3f}]"
    # FINDINGS 17.5: the drift-zero arms of C1 and the C1 minus C2 drift-zero gap, built from
    # artifacts/corrected_verdicts.json runs.<run>.drift_zero.
    _s175 = _sec("### 17.5 The L0 gate", "### 17.5.1")
    _dz1, _dz2 = _v["runs"]["C1"]["drift_zero"], _v["runs"]["C2"]["drift_zero"]
    check_true("C1 drift-zero survival cells are the L0 gate's cells",
               _dz1["survival"]["per_seed"] == _v["runs"]["C1"]["gates"]["l0"]["per_seed"])
    for _arm in ("untrained", "predictor"):
        _b = _dz1[_arm]
        check("17.5: C1 drift-zero " + _arm + " mean recomputes from its seeds",
              _seed_mean(_b), _b["mean"])
        check_true(f"17.5 quotes the C1 drift-zero {_arm} arm",
                   f"{_arm} arm of `C1` reads {_b['mean']:.3f} [{_b['t90'][0]:.3f}, {_b['t90'][1]:.3f}]"
                   in _s175 or f"{_arm} {_b['mean']:.3f} [{_b['t90'][0]:.3f}, {_b['t90'][1]:.3f}]" in _s175)
    _b2 = _dz2["survival"]
    check_true("17.5 quotes the C2 drift-zero survival agents",
               f"read {_b2['mean']:.3f} [{_b2['t90'][0]:.3f}, {_b2['t90'][1]:.3f}]" in _s175)
    _gap = _pc2(_dz1["survival"]["per_seed"], _b2["per_seed"])
    _sd1 = float(np.std(_dz1["survival"]["per_seed"], ddof=1))
    check_true(f"17.5 quotes the C1 minus C2 drift-zero gap ({_gap['mean']:+.3f})",
               f"read {_gap['mean']:.3f} higher at drift zero (paired by seed {_ivs(_gap)})" in _s175)
    check_true(f"17.5 quotes C1's drift-zero between-seed sd ({_sd1:.3f})",
               f"between-seed sd of {_sd1:.3f} for `C1`" in _s175)
    # FINDINGS 17.2: the worst reward-leak pool and the survival arm's drift-0.45 pools.
    _s172 = _sec("### 17.2 C1", "### 17.2.1")
    _lk = []
    _cdir = os.path.join(ARTROOT, "corrected_runs", "corrected_l3_h8_wm", "cells")
    for _fn in sorted(os.listdir(_cdir)):
        with open(os.path.join(_cdir, _fn), encoding="utf-8") as _fh:
            _c = json.load(_fh)["cell"]
        for _arm, _a in _c["agents"].items():
            _lk.append((float(_c["drift"]), int(_c["seed"]), _arm, float(_a["pool"]["pool_reward_leak"])))
    _w = max(_lk, key=lambda r: abs(r[3] - 0.5))
    check("17.2: worst pooled reward leak equals the verdict artifact's gate value",
          abs(_w[3] - 0.5), _v["runs"]["C1"]["gates"]["reward_leak"]["max_abs_dev"], tol=1e-6)
    check_true(f"17.2 names the worst reward-leak pool ({_w[2]}, drift {_w[0]}, seed {_w[1]}, {_w[3]:.3f})",
               f"the worst reward-leak pool is the {_w[2]} arm's at drift {_w[0]:.2f}, seed {_w[1]}, "
               f"at {_w[3]:.3f}, within {0.1 - abs(_w[3] - 0.5):.3f} of the tolerance" in _s172)
    _sl = [r[3] for r in _lk if r[2] == "survival" and r[0] == 0.45]
    check_true("17.2 quotes the survival arm's drift-0.45 reward-leak pools",
               len(_sl) == 10 and f"pools average {np.mean(_sl):.3f} and range from {min(_sl):.3f} to "
               f"{max(_sl):.3f}" in _s172)
    # FINDINGS 17.6: the contrasts against the untrained arm, and their echo in PAPER_OUTLINE row 44.
    _pcd2 = _load_art("policy_controls", "corrected_l3_h8_wm.json")
    _pcells = sorted(_pcd2["cells"], key=lambda c: c["seed"])
    _tv2 = lambda arm, pr: [c["targets"][arm][pr]["target"] for c in _pcells]
    _fm = lambda c: f"{c['mean']:+.3f} {_ivs(c)}"
    for _pr in ("own", "scripted", "replay"):
        _row = (f"| {_pr} | {_fm(_pc2(_tv2('predictor', _pr), _tv2('untrained', _pr)))} | "
                f"{_fm(_pc2(_tv2('predictor_logged', _pr), _tv2('untrained', _pr)))} |")
        check_true(f"FINDINGS 17.6 contrast against the untrained arm, {_pr}", _row in _s17)
    _rp = {b: _pc2(_tv2("survival", "replay"), _tv2(b, "replay"))
           for b in ("predictor", "predictor_logged", "untrained")}
    _rpl = _pc2(_tv2("predictor_logged", "replay"), _tv2("untrained", "replay"))
    _s176 = _sec("### 17.6 Who drives", "### 17.7")
    check_true("17.6 quotes survival minus predictor on replay",
               f"survival minus predictor {_fm(_rp['predictor'])} on replay" in _s176)
    check_true("17.6 quotes the trained arms' lead over the untrained one on replay",
               f"({_rp['untrained']['mean']:+.3f} for survival and {_rpl['mean']:+.3f} for "
               f"`predictor_logged`)" in _s176)
    check_true("PAPER_OUTLINE row 44 quotes the replay contrasts of 17.6",
               f"survival minus predictor {_fm(_rp['predictor'])}, minus predictor_logged "
               f"{_fm(_rp['predictor_logged'])}, minus untrained {_fm(_rp['untrained'])}; "
               f"predictor_logged minus untrained {_fm(_rpl)}" in _fl(_read("docs/PAPER_OUTLINE.md")))
    # PAPER_OUTLINE row 54 (2026-10-10): the four calibrated fingerprint instances read
    # together, each pooled survival target and its margins recomputed from its artifact.
    _inst = []
    for _fn in ("behavior_audit_l3_h8_traces.json", "behavior_audit_l3_h7_traces.json"):
        _ag = _load_art("expB2", _fn)["aggregate"]
        _inst.append({a: _ag[f"d=0.45 {a}"]["target"]["mean"]
                      for a in ("survival", "predictor", "untrained")})
    for _fn in ("second_seed_l3_h8_gseed2_gpu.json", "second_instance_l3_h10_gseed1.json"):
        _ar = _load_art("expB2", _fn)["arms"]["0.45"]
        _inst.append({a: float(np.mean(_ar[a]["pool_target"]["per_seed"]))
                      for a in ("survival", "predictor", "untrained")})
    _sv = [i["survival"] for i in _inst]
    _mu = [i["survival"] - i["untrained"] for i in _inst]
    _mp = [i["survival"] - i["predictor"] for i in _inst]
    _r54 = _fl(_read("docs/PAPER_OUTLINE.md"))
    check("PAPER_OUTLINE row 54: descriptive mean of the four instances (0.701)",
          float(np.mean(_sv)), 0.701)
    check_int("PAPER_OUTLINE row 54: untrained margin above 0.05 on all four",
              sum(m > 0.05 for m in _mu), 4)
    check_int("PAPER_OUTLINE row 54: predictor margin above 0.05 on two",
              sum(m > 0.05 for m in _mp), 2)
    check_true("PAPER_OUTLINE row 54 quotes the four instances and their margins",
               ("survival pooled targets as published " + ", ".join(
                   f"{v:.3f} ({w})" for v, w in zip(_sv, ("hidden 8, G seed 0", "hidden 7",
                                                     "hidden 8, G seed 2",
                                                     "hidden 10, G seed 1, CPU"))))
               + ";"
               in _r54
               and f"descriptive mean {np.mean(_sv):.3f}" in _r54
               and "margin over the untrained arm " + ", ".join(f"{m:+.3f}" for m in _mu) in _r54
               and "over the predictor " + ", ".join(f"{m:+.3f}" for m in _mp) in _r54)
    # FINDINGS 12: retrain time against cell time, from the policy-controls cells.
    _rs = [c["retrain_seconds"] for c in _pcells]
    _cs = [c["seconds"] for c in _pcells]
    check_true("12 quotes the survival retrain time per seed",
               f"{min(_rs) / 60:.1f} to {max(_rs) / 60:.1f} minutes per seed (`retrain_seconds`, "
               f"{min(_rs):.0f} to {max(_rs):.0f} s)" in _fl(_fx2))
    check_true("12 quotes the whole policy-controls cell time",
               f"took {min(_cs) / 60:.1f} to {max(_cs) / 60:.1f} minutes (`seconds`, "
               f"{min(_cs):.0f} to {max(_cs):.0f} s)" in _fl(_fx2))
    # FINDINGS 13.F second note and 17.11: the per-individual contrast and means, gate 3, and the
    # attribution of each estimand to its run.
    _s13n = _fl(_s13)
    _pi2 = _load_art("expC", "emergence_pilot_per_individual_summary.json")
    _cl = sorted(_pi2["cells"], key=lambda c: c["seed"])
    _ft2 = [c["indiv_final_treat"]["mean"] for c in _cl]
    _fc2 = [c["indiv_final_ctrl"]["mean"] for c in _cl]
    _ic = [a - b for a, b in zip(_ft2, _fc2)]
    _icp = _pc2(_ft2, _fc2)
    check_true("13.F second note quotes the per-individual contrast",
               f"{_ic[0]:+.3f}, {_ic[1]:+.3f} and {_ic[2]:+.3f} (seeds 0, 1, 2), mean {_icp['mean']:+.3f}, "
               f"t-based 90% interval [{_icp['t90'][0]:+.3f}, {_icp['t90'][1]:+.3f}]" in _s13n)
    check_true("13.F second note quotes the final treatment individuals' average",
               f"average {np.mean(_ft2):.3f} (mean of the three population means)" in _s13n)
    check_true("17.11 quotes the final treatment per-individual range",
               f"individuals average {min(_ft2):.3f} to {max(_ft2):.3f} in the final treatment populations"
               in _fl(_s17))
    _g3 = _pi2["gates"]["gate3_l0_tost"]
    check_true("13.F second note: gate 3 reads exactly 0.500 on all 21 panels (by construction)",
               _g3["mean"] == 0.5 and _g3["n"] == 21 and "all 21 panels read exactly 0.500" in _s13n)
    check_true("13.F second note quotes the lowest speed-control panel",
               f"speed control reads {_pi2['gates']['gate5_speed_control']['min']:.3f} to" in _s13n)
    _est = _pi2["estimand"]
    check_true("13.F second note attributes the 06ecc10 estimand",
               _pi2["git_commit_at_run"].startswith("06ecc10")
               and (", ".join(f"{x:+.3f}" for x in _est["contrast"]) + f" (mean {_est['mean_contrast']:+.3f}, "
                    f"t-based [{_est['t_ci90'][0]:+.3f}, {_est['t_ci90'][1]:+.3f}]) with final treatment "
                    f"{_est['mean_final_treat_auroc']:.3f} ("
                    + ", ".join(f"{c['final_treat_auroc']:.3f}" for c in _cl) + ")") in _s13n)
    _pd = _load_art("expC", "emergence_pilot_summary.json")
    check_true("13.F second note attributes the a0cb850 estimand",
               _pd["git_commit_at_run"].startswith("a0cb850")
               and ("per-seed contrast " + ", ".join(f"{x:+.3f}" for x in _pd["estimand"]["contrast"])
                    + "; final treatment " + ", ".join(f"{c['final_treat_auroc']:.3f}" for c in
                                                       sorted(_pd["cells"], key=lambda c: c["seed"])))
               in _s13n)
    # Experiment C compute: both runs' wall times, as 13.D and 13.F quote them.
    _wd, _wi = _pd["wall_seconds"], _pi2["wall_seconds"]
    check("pinned: Exp C fixed-code re-run wall_seconds (34561.7)", _wd, 34561.7, tol=0.05)
    check("pinned: Exp C per-individual rerun wall_seconds (81183.2)", _wi, 81183.2, tol=0.05)
    check_true("13.D quotes the re-run's wall time", f"Wall time was {_wd / 3600:.1f} hours" in _s13n)
    check_true("13.F quotes the rerun's wall time", f"Wall time {_wi:.0f} s ({_wi / 3600:.1f} hours" in _s13n)

    # FINDINGS 13.E and 17.11 notes: the matched assignment minus the better single policy is not
    # a value of information; the best of the two policies per world is, over these two policies.
    # Both recomputed from the cross-evaluation cells.
    def _best_of_two(run_dir: str) -> list[float]:
        out = []
        for _sd in range(10):
            _a = _load_art(run_dir, "cells", f"cell_d0.00_s{_sd}.json")["cell"]["xeval"]
            _s = _load_art(run_dir, "cells", f"cell_d0.45_s{_sd}.json")["cell"]["xeval"]
            _k0, _k1 = sorted(_a, key=float)[0], sorted(_a, key=float)[-1]
            _single = max(0.5 * (_a[_k0] + _a[_k1]), 0.5 * (_s[_k0] + _s[_k1]))
            out.append(0.5 * (max(_a[_k0], _s[_k0]) + max(_a[_k1], _s[_k1])) - _single)
        return out
    _bh = _best_of_two(os.path.join("reviewer_gaps_runs", "l3_h8_wm_cpu"))
    _bc = _best_of_two(os.path.join("corrected_runs", "corrected_l3_h8_wm"))
    check_true("best-of-two values are never negative", min(_bh + _bc) >= 0.0)
    check("Exp C 13.E note: historical best-of-two (+0.013)", float(np.mean(_bh)), 0.013)
    check_true("13.E note quotes the historical best-of-two value and its zero seeds",
               f"On these historical CPU cells it is {np.mean(_bh):+.3f} (zero in "
               f"{sum(x == 0 for x in _bh)} of 10 seeds)" in _s13n)
    _bci = _pc2(_bc, [0.0] * len(_bc))
    _nz = [(i, x) for i, x in enumerate(_bc) if x != 0]
    check("Exp C 17.11 note: corrected best-of-two (+0.003)", float(np.mean(_bc)), 0.003)
    check_true("17.11 note quotes the corrected best-of-two value, interval and zero seeds",
               f"Choosing the better policy in each world gains {_bci['mean']:+.3f} return (t-based 90% CI "
               f"[{_bci['t90'][0]:+.3f}, {_bci['t90'][1]:+.3f}]): in {10 - len(_nz)} of 10 seeds one policy "
               f"returns more in both worlds" in _fl(_s17)
               and all(f"seed {i} gains {x:+.3f}" in _fl(_s17) for i, x in _nz))
    # FINDINGS 3.4: the random-forest probe values.
    _nl = {r["drift"]: r for r in _load_art("expB", "summary.json")["expB_nonlinear"]["drift_sweep_with_std"]}
    check_true("3.4 quotes the random-forest probe as stored",
               f"**{_nl[0.0]['mean']:.3f} ± {_nl[0.0]['std']:.3f}** control and "
               f"**{_nl[0.45]['mean']:.3f} ± {_nl[0.45]['std']:.3f}** drift 0.45" in _fl(_fx2))
    # FINDINGS 9: the speed control's pool range, and the tables' population standard deviations.
    _s9 = _sec("## 9. Experiment B-v2", "## 10. Experiment B, L3")
    _b2r = load("expB2_results.json")
    _sp = [(float(x), d, a) for d, v in _b2r.items() if isinstance(v, dict)
           for a, b in v.items() if isinstance(b, dict) and "pool_speed" in b for x in b["pool_speed"]]
    _am = [float(np.mean(b["pool_speed"])) for v in _b2r.values() if isinstance(v, dict)
           for b in v.values() if isinstance(b, dict) and "pool_speed" in b]
    _lo = min(_sp)
    check_true("9 quotes the B-v2 speed range over pools and arm means",
               f"run from {_lo[0]:.3f} ({_lo[2]}, drift {float(_lo[1]):g}) to {max(_sp)[0]:.3f} and the arm "
               f"means from {min(_am):.3f} to {max(_am):.3f}" in _s9)
    for _name, _art in (("replication", _b2r), ("confirmatory", load("expB2_results_confirmatory_n3.json"))):
        for _arm in ("untrained", "predictor", "survival"):
            _cell = lambda d: (f"{np.mean(_art[d][_arm]['pool_target']):.3f} ± "
                               f"{np.std(_art[d][_arm]['pool_target']):.3f}")
            _bold = "**" if _arm == "survival" else ""
            check_true(f"9 {_name} table row {_arm} uses population standard deviations",
                       f"| {_bold}{_arm}{_bold} | {_cell('0.0')} | {_bold}{_cell('0.45')}{_bold} |" in _s9)
    _u0 = _b2r["0.0"]["untrained"]["pool_target"]
    check_true("9 note compares sample and population standard deviations",
               f"{np.std(_u0, ddof=1):.3f} rather than {np.std(_u0):.3f}" in _s9)
    # FINDINGS 10.5: the hidden-7 reward leak counted over every arm.
    _s105 = _sec("### 10.5 Second in-band", "### 10.6 Held-out")
    _h7r = _load_art("expB2", "heldout_l3_h7_reverse_summary.json")
    _bad = [c for c in _h7r["cells"] if not c["pool_leak_clean"]]
    _sv45 = [c["pool_reward_leak"] for c in _h7r["cells"] if c["agent"] == "survival" and c["drift"] == "0.45"]
    check_true("10.5: survival reward leak at drift 0.45 averages the quoted 0.567, clean 10/10",
               f"{np.mean(_sv45):.3f}" == "0.567" and all(abs(x - 0.5) < 0.1 for x in _sv45))
    check_true("10.5 names the one hidden-7 pool outside the leakage tolerance",
               len(_bad) == 1 and len(_h7r["cells"]) == 60
               and f"one pool of the 60 lies outside the 0.1 tolerance: {_bad[0]['agent']}, drift "
                   f"{_bad[0]['drift']}, seed {_bad[0]['seed']}, at {_bad[0]['pool_reward_leak']:.3f}" in _s105)
    # FINDINGS 14: the hidden-8 alpha-0 TOST and the survival-minus-untrained margins.
    _s14 = _sec("## 14. H2 substrate-grounding", "## 14.5 H2 texture")
    _et = equivalence_test(h2_surv["a0.00"]["per_seed"], 0.5, 0.05)
    check_true(f"14 note quotes the hidden-8 alpha-0 TOST (p = {_et.p_value:.3f}, accepted)",
               _et.equivalent and f"accepts at hidden 8 (p = {_et.p_value:.3f})" in _s14)
    check_true("section 6 quotes the same TOST", f"TOST p = {_et.p_value:.3f}, and the bootstrap ROPE" in
               _sec("## 6. Status against", "## 7. Levers"))
    _m1 = _pc2(h2_surv["a1.00"]["per_seed"], h2_untr["a1.00"]["per_seed"])
    _m0 = _pc2(h2_surv["a0.00"]["per_seed"], h2_untr["a0.00"]["per_seed"])
    check_true("14 note quotes the hidden-8 survival-minus-untrained margins",
               f"at hidden 8 from {_fm(_m1)} to {_fm(_m0)}, where the untrained arm sits below 0.5 "
               f"({np.mean(h2_untr['a0.00']['per_seed']):.3f})" in _s14)
    # FINDINGS 14.6: what the masks do to the untrained arm at hidden 7.
    _s146 = _sec("## 14.6 A2 observation-channel", "## 14.7 A3 L1")
    _o7 = _load_art("expH2", "obs_localization_h7.json")["masks"]
    _om = lambda mk: _pc2(_o7[mk]["survival"]["per_seed_full"], _o7[mk]["untrained"]["per_seed_full"])
    check_true("14.6 note quotes the hidden-7 untrained arm under each single-channel mask",
               f"{_o7['vision']['untrained']['mean']:.3f} under the vision mask "
               f"({_o7['vision']['untrained']['n_ge_065']} of 10 seeds at the bar) and "
               f"{_o7['intero']['untrained']['mean']:.3f} under the interoception mask "
               f"({_o7['intero']['untrained']['n_ge_065']} of 10)" in _s146)
    check_true("14.6 note quotes the hidden-7 paired margins under the masks",
               f"{_fm(_om('vision'))} under the vision mask and {_fm(_om('intero'))} under the "
               f"interoception mask, against {_fm(_om('none'))} unmasked" in _s146)
    check_true("14.6: the all-mask row reads exactly 0.500 for every arm and seed",
               all(x == 0.5 for _h in ("h7", "h8")
                   for _a in _load_art("expH2", f"obs_localization_{_h}.json")["masks"]["all"].values()
                   if isinstance(_a, dict) and "per_seed_full" in _a for x in _a["per_seed_full"]))
    # FINDINGS 14 note: the hidden-7 graded seam, promoted as artifacts/expH2/summary_h7.json.
    _h7c = _load_art("expH2", "summary_h7.json")["cells"]
    _al = sorted(_h7c["untrained"], key=lambda k: float(k[1:]))
    _uc = [f"{np.mean(_h7c['untrained'][k]['per_seed']):.3f}" for k in _al]
    check_true("14 note quotes the hidden-7 untrained curve",
               f"{', '.join(_uc[:-1])} and {_uc[-1]} from alpha 0 to alpha 1" in _s14)
    _u10 = _pc2(_h7c["untrained"]["a1.00"]["per_seed"], _h7c["untrained"]["a0.00"]["per_seed"])
    _nup = sum(a > b for a, b in zip(_h7c["untrained"]["a1.00"]["per_seed"],
                                     _h7c["untrained"]["a0.00"]["per_seed"]))
    check_true("14 note quotes the hidden-7 untrained rise, alpha 1 minus alpha 0",
               f"alpha 1 minus alpha 0, paired by seed, {_fm(_u10)}, {_nup} of 10 seeds up" in _s14)
    check_true("14 note quotes the hidden-7 predictor endpoints",
               f"the predictor arm rises from {np.mean(_h7c['predictor']['a0.00']['per_seed']):.3f} to "
               f"{np.mean(_h7c['predictor']['a1.00']['per_seed']):.3f}" in _s14)
    _h7m = {a: _pc2(_h7c["survival"][a]["per_seed"], _h7c["untrained"][a]["per_seed"])
            for a in ("a1.00", "a0.00")}
    check_true("14 note quotes the hidden-7 survival-minus-untrained margins",
               f"at hidden 7 from {_fm(_h7m['a1.00'])} at alpha 1 to {_fm(_h7m['a0.00'])} at alpha 0" in _s14)
    _et7 = equivalence_test(_h7c["survival"]["a0.00"]["per_seed"], 0.5, 0.05)
    check_true(f"14 note quotes the hidden-7 alpha-0 TOST (p = {_et7.p_value:.3f}, not accepted)",
               (not _et7.equivalent) and f"does not accept at hidden 7 (p = {_et7.p_value:.3f})" in _s14)
    # FINDINGS 14.6 addendum: the hidden-8 margin under the vision mask.
    _o8 = _load_art("expH2", "obs_localization_h8.json")["masks"]
    _om8 = lambda mk: _pc2(_o8[mk]["survival"]["per_seed_full"], _o8[mk]["untrained"]["per_seed_full"])
    check_true("14.6 addendum quotes the hidden-8 paired margin under the vision mask",
               f"survival minus untrained is {_fm(_om8('vision'))} under the vision mask, against "
               f"{_fm(_om8('none'))} unmasked" in _s146)
    # FINDINGS 10.9 note: the speed positive control read in every pool, for every promoted run.
    _s109 = _fx2[_fx2.index("### 10.9 Second fingerprint"):_fx2.index("## 11. Methods notes")]
    _under = []
    for _run, _fn in (("L3-H8-NOWM-CPU", "arch_baseline_l3_h8_nowm"),
                      ("L3-H8-WM-CPU", "device_control_l3_h8_wm_cpu"),
                      ("L3-H8-NOWM-U450", "skill_matched_l3_h8_nowm_u450"),
                      ("L3-H10-GS1-CPU", "second_instance_l3_h10_gseed1"),
                      ("L3-H10-GS1-GPU", "second_instance_l3_h10_gseed1_gpu"),
                      ("L3-H8-GS2-GPU", "second_seed_l3_h8_gseed2_gpu"),
                      ("C1", "corrected_l3_h8_wm"), ("C2", "corrected_l3_h8_nowm")):
        _gg = load(_fn + ".json")["gates"]
        _at = _gg["speed_min_at"]
        _tail = (f"| {_gg['speed_min_of_arm_means']:.3f} | {_gg['speed_min']:.3f} | {_at['arm']}, drift "
                 f"{float(_at['drift']):g}, seed index {_at['seed_index']} | {_gg['speed_n_below']} of "
                 f"{_gg['speed_n_pools']} |")
        _line = next((ln for ln in _s109.split("\n") if ln.startswith(f"| `{_run}` |")), "")
        check_true(f"10.9 note quotes {_run}'s lowest arm mean, lowest pool and where it sits",
                   _line.endswith(_tail))
        if _gg["speed_n_below"]:
            _under.append(_run)
    check_true("10.9 note: only the hidden-10 GPU re-measure has a pool under 0.75",
               _under == ["L3-H10-GS1-GPU"] and "Every run but one still passes." in _s109)
    # FINDINGS 16: rows of the gate table that share one set of drift-zero agents.
    _gt = _load_art("gate_table.json")["rows"]
    _key = lambda r: tuple(round(r["l0"][k], 6) for k in ("n", "mean", "sd", "tost_p"))
    _xh = [r for r in _gt if r["status"] == "historical" and r["partition"].startswith("explicit")]
    _words = {5: "five", 10: "10", 11: "11", 12: "twelve", 22: "22", 26: "26", 27: "27"}
    _nset, _nxh, _nxs = len({_key(r) for r in _gt}), len(_xh), len({_key(r) for r in _xh})
    check_true(f"16 note counts {_nset} drift-zero sets over {len(_gt)} gate-table rows, "
               f"{_nxs} over the {_nxh} explicit-partition historical rows",
               f"the {_words.get(len(_gt))} rows hold {_words.get(_nset)} distinct sets. The "
               f"{_words.get(_nxh)} explicit-partition historical rows hold {_words.get(_nxs)}"
               in _sec("## 16. Fold-split", "## 17. Corrected-trainer"))

    # Revision step 10: FINDINGS 10.1.1 quotes the surrogate diagnostics; pin every cell.
    # 2026-10-10: the per-surrogate `diagnostics` were scored on a seed-77 sample that lies
    # inside every G's training episodes. They stay quoted, relabeled as the calibration
    # sample, and the `disjoint_heldout` block carries the held-out rescoring. Each table is
    # pinned in its own part of the section, and the seed ranges are recomputed.
    print("\n== FINDINGS 10.1.1: surrogate and detector diagnostics ==")
    _f = _read("docs/FINDINGS.md")
    _sec = _f[_f.index("### 10.1.1"):_f.index("### 10.2 Headline result")]
    _cut = _sec.index("**The seed-77 calibration sample")
    _sec_ho, _sec_cal = _sec[:_cut], _sec[_cut:]
    _sdoc = _load_art("surrogate_diagnostics.json")
    _sd = _sdoc["surrogates"]
    for _name, _v in _sd.items():
        _g = _v["diagnostics"]
        for _label, _val, _fmt in (
                ("one-step seed-0 sample", _g["g_one_step"]["rms_train"], "{:.4f}"),
                ("one-step seed-77 calibration sample", _g["g_one_step"]["rms_heldout"], "{:.4f}"),
                ("lag-1 autocorrelation", _v["profile"]["lag1_autocorr"], "{:.2f}"),
                ("agent-accessible detector", _v["agent_accessible_detector"]["auroc"], "{:.3f}")):
            check_true(f"10.1.1 calibration {_name} {_label} {_fmt.format(_val)} quoted",
                       _fmt.format(_val) in _sec_cal)
    check_true("10.1.1 linear-fit RMS on the calibration sample is rounding-level (< 1e-7)",
               all(v["diagnostics"]["linear_fit"]["rms_heldout"] < 1e-7 for v in _sd.values()))
    _dh = _sdoc["disjoint_heldout"]
    _G_TRAIN = {"gmotion_h8_s0": 0, "gmotion_h7_s0": 0, "gmotion_h10_s1": 1, "gmotion_h8_s2": 2}
    _n_tr, _n_ho = 250, 60
    check_true("10.1.1: the seed-77 sample lies inside every G's training episodes",
               all(_s <= 77 and 77 + _n_ho <= _s + _n_tr for _s in _G_TRAIN.values()))
    check_true("10.1.1: the held-out range is disjoint from every G's training episodes",
               all(_dh["heldout_seed"] >= _s + _n_tr or _dh["heldout_seed"] + _n_ho <= _s
                   for _s in _G_TRAIN.values()))
    check_true("10.1.1: each G's train column is scored on its own training seed",
               all(_dh["surrogates"][_k]["diagnostics"]["train_seed"] == _s
                   for _k, _s in _G_TRAIN.items()))
    check_true("10.1.1: the rescoring reproduced the stored detector exactly and the stored "
               "profiles to its tolerance",
               _dh["determinism"]["detector_auroc_exact"]
               and _dh["determinism"]["profile_max_abs_diff"] <= _dh["determinism"]["profile_tol"])
    check_true("10.1.1: the held-out block covers the same surrogates", set(_dh["surrogates"]) == set(_sd))
    for _name, _v in _dh["surrogates"].items():
        _g = _v["diagnostics"]
        _roll = " / ".join(f"{_g['rollout_rms_velocity_gap'][_h]:.3f}" for _h in ("1", "5", "10", "24"))
        for _label, _txt in (
                ("one-step train", f"{_g['g_one_step']['rms_train']:.4f}"),
                ("one-step held out", f"{_g['g_one_step']['rms_heldout']:.4f}"),
                ("rollout gaps", _roll),
                ("lag-1 autocorrelation held out", f"{_v['profile_heldout']['lag1_autocorr']:.2f}"),
                ("agent-accessible detector", f"{_sd[_name]['agent_accessible_detector']['auroc']:.3f}")):
            check_true(f"10.1.1 held out {_name} {_label} {_txt} quoted", _txt in _sec_ho)
    check_true("10.1.1 linear-fit held-out RMS is rounding-level (< 1e-7)",
               all(v["diagnostics"]["linear_fit"]["rms_heldout"] < 1e-7
                   for v in _dh["surrogates"].values()))
    check_true("10.1.1 quotes the headline G's held-out and calibration one-step errors together",
               f"{_dh['surrogates']['gmotion_h8_s0']['diagnostics']['g_one_step']['rms_heldout']:.4f} per step "
               f"held out against {_sd['gmotion_h8_s0']['diagnostics']['g_one_step']['rms_heldout']:.4f}"
               in " ".join(_sec_cal.split()))
    # Revision step 15: the manuscript's tables are generated from the checked artifacts.
    # The manuscript and its generated tables stay local (gitignored docs/paper/); render the
    # tables in memory from the committed artifacts instead of comparing committed files.
    import build_paper_tables
    _pt = build_paper_tables.tables()
    check_true("manuscript tables render from the committed artifacts (gate table, contrasts, "
               "corrected verdicts)",
               {"gate_table.tex", "contrast_intervals.tex", "corrected_runs.tex"} <= set(_pt)
               and all("do not edit" in t for t in _pt.values()))
    check_true("rule ledger matches its artifacts, and local manuscript tables, if present, are "
               "current (build_paper_tables --check)",
               build_paper_tables.main(["--check"]) == 0)

    # ---- section 18: goal and stakes, and the white-jitter run at 96 (2026-10-11) ----
    print("")
    print("== FINDINGS 18: goal and stakes, and the registered-trunk white-jitter run ==")
    _f18 = _read("docs/FINDINGS.md")
    _f18 = " ".join(_f18[_f18.find("## 18. Goal and stakes"):].split())
    _ga18, _e0s = {"C1": _load_art("expB2", "corrected_l3_h8_wm.json")}, {}
    for _run, _surv, _lo, _hi, _l0, _eq in (("T-touch", 0.796, 0.762, 0.830, 0.522, True),
                                             ("S-immortal", 0.744, 0.696, 0.793, 0.537, False)):
        _ga = _ga18[_run] = _load_art("expB2", f"goal_stakes_{_run}.json")
        _ps = _ga["arms"]["0.45"]["survival"]["pool_target"]["per_seed"]
        check(f"18.2 {_run} survival mean", float(np.mean(_ps)), _surv)
        check_true(f"18.2 {_run} stored decision survival equals the per-seed mean",
                   _same(float(_ga["decision"]["survival"]), float(np.mean(_ps))))
        _lo_r, _hi_r = t_ci90(_ps)
        check(f"18.2 {_run} survival t90 lower", _lo_r, _lo)
        check(f"18.2 {_run} survival t90 upper", _hi_r, _hi)
        # 2026-10-16 (finding c25): the L0 mean and TOST are recomputed from the drift-0
        # per-seed values, and the stored gate must equal the recomputation.
        _e0 = _e0s[_run] = l0_gate_reproduces(f"18.2 {_run}", _ga)
        check(f"18.2 {_run} L0 mean at drift 0",
              _seed_mean(_ga["arms"]["0.0"]["survival"]["pool_target"]), _l0)
        check_true(f"18.2 {_run} L0 equivalence is {_eq}", bool(_e0.equivalent) is _eq)
        check_true(f"18.2 quotes {_run} survival {_surv:.3f} [{_lo:.3f}, {_hi:.3f}]",
                   f"{_surv:.3f} [{_lo:.3f}, {_hi:.3f}]" in _f18)
    check_true("18.2 T-touch passes every gate (promoted zone)",
               _load_art("expB2", "goal_stakes_T-touch.json")["decision"]["zone"]
               == "ENCODING INDUCED (all clauses)")
    _dec = _load_art("goal_stakes", "decision.json")
    check_true("18.2 quotes the T-touch primary wording from decision.json",
               _dec["T-touch"]["primary"]["wording"].split("; ")[-1] in _f18)
    check_true("18.2 records the stakes rule as not testable",
               _dec["stakes"]["testable"] is False and "not testable" in _f18)
    # 2026-10-16 (review of finding c27): every 18.2 table row is rendered from its promoted
    # artifact (C1 from 17.2): survival and t90, seeds at the bar, both leads from the per-seed
    # arm means, the recomputed L0 reading and the open gates. The seed-paired differences from
    # C1 and the touch engagement means are recomputed the same way.
    _pt18 = lambda d, a: np.asarray(d["arms"]["0.45"][a]["pool_target"]["per_seed"], float)
    _rows18 = {}
    for _run in ("C1", "S-immortal", "T-touch"):
        _ga = _ga18[_run]
        _sv, _pr, _un = (_pt18(_ga, a) for a in ("survival", "predictor", "untrained"))
        _lp, _lu = float(_sv.mean() - _pr.mean()), float(_sv.mean() - _un.mean())
        _nge = int(np.sum(_sv >= BAR))
        _st = _ga["decision"]
        check_true(f"18.2 {_run}: stored leads and seeds at the bar equal the per-seed recomputation",
                   _same(_lp, float(_st["lead_over_predictor"])) and _same(_lu, float(_st["lead_over_untrained"]))
                   and _nge == int(_ga["arms"]["0.45"]["survival"]["pool_target"]["n_ge_065"]))
        _e0 = l0_gate_reproduces(f"18.2 {_run}", _ga) if _run == "C1" else _e0s[_run]
        _l0m = _seed_mean(_ga["arms"]["0.0"]["survival"]["pool_target"])
        _open = list(_st["gates_open"])
        if _open == ["l0"]:
            _gw = "L0 open"
        elif not _open and _run in _dec and _dec[_run]["primary"]["gates"]["all"] is True \
                and len(_dec[_run]["primary"]["gates"]["per_gate"]) == 6:
            _gw = "all six pass"
        else:
            _gw = "open: " + ", ".join(_open)
        _lo_r, _hi_r = t_ci90(_sv)
        _lab = "`C1` (17.2)" if _run == "C1" else f"`{_run}`"
        _rows18[_run] = (f"| {_lab} | {_sv.mean():.3f} [{_lo_r:.3f}, {_hi_r:.3f}] | {_nge} | {_lp:+.3f} | "
                         f"{_lu:+.3f} | {_l0m:.3f}, {'equivalent' if _e0.equivalent else 'not equivalent'} "
                         f"| {_gw} |")
        check_true(f"FINDINGS 18.2 table row: {_run}", _rows18[_run] in _f18)
    _c1pr, _c1un = _pt18(_ga18["C1"], "predictor"), _pt18(_ga18["C1"], "untrained")
    check_true("18.2 predictor and untrained per-seed values are the same in all three runs (so the "
               "survival seeds pair with C1's)",
               all(list(_pt18(_ga18[r], "predictor")) == list(_c1pr)
                   and list(_pt18(_ga18[r], "untrained")) == list(_c1un) for r in ("S-immortal", "T-touch")))
    check_true("18.2 quotes the shared predictor and untrained means",
               f"The predictor ({_c1pr.mean():.3f}) and untrained ({_c1un.mean():.3f}) arms are the same "
               f"in all three runs" in _f18)
    _dq = {}
    for _run in ("T-touch", "S-immortal"):
        _dd = _pt18(_ga18[_run], "survival") - _pt18(_ga18["C1"], "survival")
        _dt = t_ci90(_dd)
        check_true(f"18.2 {_run} minus C1: the paired interval does not exclude zero",
                   not (_dt[0] > 0 or _dt[1] < 0))
        _dq[_run] = f"{_dd.mean():+.3f} [{_dt[0]:+.3f}, {_dt[1]:+.3f}]"
    check_true("18.2 quotes the seed-paired differences from C1",
               f"The seed-paired difference from `C1` is {_dq['T-touch']} for `T-touch` and "
               f"{_dq['S-immortal']} for `S-immortal`; neither interval excludes zero" in _f18)
    # Touch engagement: per-episode pellet means of the trained arm and both baselines over the
    # ten drift-0.45 cells; the touch gate's ratio over the better baseline is 1.5
    # (itasorl/experiment_b2.py TOUCH_ENGAGE_RATIO).
    _eng = []
    for _p in sorted(glob.glob(os.path.join(ARTROOT, "goal_stakes", "T-touch", "cells", "cell_d0.45_s*.json"))):
        with open(_p, encoding="utf-8") as _fh:
            _eng.append(json.load(_fh)["cell"]["eng"])
    _em = [float(np.mean([e[k] for e in _eng])) for k in ("trained_return", "random_return", "scripted_return")]
    check_true("18.2 T-touch: ten drift-0.45 cells, each engaged, and the mean trained intake is at "
               "least 1.5x the better baseline",
               len(_eng) == 10 and all(e["engaged"] is True for e in _eng) and _em[0] >= 1.5 * max(_em[1:]))
    check_true("18.2 quotes the T-touch engagement means",
               f"{_em[0]:.2f} pellets per episode against {_em[1]:.2f} for the random walker and "
               f"{_em[2]:.2f} for the scripted one, past the registered 1.5x ratio" in _f18)
    # 2026-10-16 (findings c25, c27): the 18.3 readouts are recomputed from the per-seed rows
    # of each run's mechanism.json the way scripts/run_mechanism_readouts.py summarizes them
    # (finite values only; a non-informative seed has no intervention score), checked against
    # the stored summary and decision.json, and the 18.3 table is rendered from them.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import decide_goal_stakes as _dgs

    def _mech(rows, fn):
        v = np.asarray([fn(r) for r in rows], float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "mean": float(v.mean()), "ci90": [float(x) for x in t_ci90(v)]}

    def _same_block(a, b):
        return a["n"] == b["n"] and _same(a["mean"], b["mean"]) and all(
            _same(x, y) for x, y in zip(a["ci90"], b["ci90"]))

    def _iv_block(rows, get):
        """The intervention part of run_mechanism_readouts.summarize's block for one arm."""
        return {"n_informative": sum(bool(get(r)["intervention"]["informative"]) for r in rows),
                "intervention_real_minus_sham": _mech(rows, lambda r: (
                    get(r)["intervention"]["score_real"] - get(r)["intervention"]["score_sham"])
                    if get(r)["intervention"]["informative"] else float("nan"))}

    # Table labels for the decide_goal_stakes wordings; a wording outside these renders as
    # itself, so a changed verdict cannot match the committed table.
    _lab_iv = {"the decoded direction did not move behavior under the tested nudge": "did not move behavior"}
    _lab_sv = {"the decoded direction tracks the agent's own prediction error": "tracks the prediction error",
               "the surrogate is surprising to the agent, but the decoded direction is not explained by "
               "that surprise": "not explained by surprise",
               "prediction error does not separate the worlds for this agent":
               "prediction error does not separate the worlds"}
    _lab_ad = {"foraging in the fake world recovers within a lifetime": "recovers",
               "no within-lifetime recovery was detected": "no recovery detected"}
    _f18t = " ".join(_read("docs/FINDINGS.md").split())
    _mrow = {"int": [], "auc": [], "corr": [], "adapt": []}
    for _run, _mean, _auc, _corr in (("T-touch", -0.001, 0.912, -0.146),
                                     ("S-immortal", -0.029, 0.848, -0.240),
                                     ("C1", 0.017, 0.846, -0.126)):
        _mj = _load_art("goal_stakes", _run, "mechanism.json")
        _rows, _sm = _mj["per_seed"], _mj["summary"]
        check_int(f"18.3 {_run}: ten seeds in mechanism.json", len(_rows), 10)
        _iv = _mech(_rows, lambda r: (r["intervention"]["score_real"] - r["intervention"]["score_sham"])
                    if r["intervention"]["informative"] else float("nan"))
        _au = _mech(_rows, lambda r: r["surprise"]["auroc"])
        _co = _mech(_rows, lambda r: r["surprise"]["corr_mean"])
        check(f"18.3 {_run} intervention real minus sham", _iv["mean"], _mean)
        check(f"18.3 {_run} surprise AUROC", _au["mean"], _auc)
        check(f"18.3 {_run} surprise correlation", _co["mean"], _corr)
        check_true(f"18.3 {_run}: stored mechanism summary equals the per-seed recomputation",
                   _same_block(_iv, _sm["intervention_real_minus_sham"])
                   and _same_block(_au, _sm["surprise_auroc"]) and _same_block(_co, _sm["surprise_corr"])
                   and _iv["n"] == _sm["n_informative"])
        # The three verdicts are derived again with decide_goal_stakes from the recomputed blocks
        # (the intervention one with both floors, recomputed from their per-seed rows); for C1,
        # which decision.json does not carry, the derived verdict is the only source of its labels.
        _fl = {g: _iv_block(_rows, lambda r, g=g: r[f"{g}_arm"]) for g in ("untrained", "predictor")}
        _g1 = _mech(_rows, lambda r: r["adaptation"]["gap_first"])
        _ad = _mech(_rows, lambda r: r["adaptation"]["adaptation"])
        check_true(f"18.3 {_run}: stored floors and adaptation summary equal the per-seed recomputation",
                   all(_fl[g]["n_informative"] == _sm["floors"][g]["n_informative"]
                       and _same_block(_fl[g]["intervention_real_minus_sham"],
                                       _sm["floors"][g]["intervention_real_minus_sham"]) for g in _fl)
                   and _same_block(_g1, _sm["adaptation_gap_first"]) and _same_block(_ad, _sm["adaptation"]))
        _ivv = _dgs.intervention_verdict({**_iv_block(_rows, lambda r: r), "floors": _fl})
        _sv3 = _dgs.surprise_verdict({"surprise_auroc": _au, "surprise_corr": _co})
        _adv = _dgs.adaptation_verdict({"adaptation_gap_first": _g1, "adaptation": _ad})
        if _run in _dec:
            check_true(f"18.3 {_run}: decision.json intervention contrast equals the recomputation",
                       _same_block(_iv, _dec[_run]["intervention"]["contrast"]))
            check_true(f"18.3 {_run}: decision.json intervention, surprise and adaptation verdicts "
                       f"equal the derived ones",
                       _dec[_run]["intervention"]["pass"] is _ivv["pass"]
                       and _dec[_run]["intervention"]["wording"] == _ivv["wording"]
                       and all(_dec[_run]["surprise"][k] == _sv3[k] for k in ("auroc_met", "corr_met", "wording"))
                       and all(_dec[_run]["adaptation"].get(k) == _adv.get(k)
                               for k in ("informative", "pass", "wording")))
        _mrow["int"].append(f"{_iv['mean']:+.3f} [{_iv['ci90'][0]:+.3f}, {_iv['ci90'][1]:+.3f}]: "
                            + _lab_iv.get(_ivv["wording"], _ivv["wording"]))
        _mrow["auc"].append(f"{_au['mean']:.3f}")
        _mrow["corr"].append(f"{_co['mean']:.3f}: " + _lab_sv.get(_sv3["wording"], _sv3["wording"]))
        _mrow["adapt"].append(_lab_ad.get(_adv["wording"], _adv["wording"]))
    for _k, _lab in (("int", "intervention, real minus sham nudge"),
                     ("auc", "surprise, prediction-error AUROC"),
                     ("corr", "surprise, correlation with the direction"),
                     ("adapt", "adaptation within a lifetime")):
        check_true(f"FINDINGS 18.3 table row: {_lab}",
                   f"| {_lab} | " + " | ".join(_mrow[_k]) + " |" in _f18t)
    # 18.4: the policy-control and control-diagnostic tables, means recomputed from per_seed;
    # the C1 column comes from the 17.6 and 17.8 artifacts.
    _p18 = {"C1": _load_art("policy_controls", "corrected_l3_h8_wm.json")}
    _d18 = {"C1": _load_art("control_diagnostics", "corrected_l3_h8_wm.json")}
    for _run in ("S-immortal", "T-touch"):
        _p18[_run] = _load_art("goal_stakes", _run, "policy_controls.json")
        _d18[_run] = _load_art("goal_stakes", _run, "control_diagnostics.json")
        check_true(f"18.4 {_run}: every retrained agent bit-identical (10 of 10)",
                   len(_p18[_run]["cells"]) == 10
                   and all(c["retrain_identical"] for c in _p18[_run]["cells"]))
    _cols = ("C1", "S-immortal", "T-touch")

    def _row18(lab, src, key):
        vals = []
        for _run in _cols:
            blk = src[_run]["aggregate"][key]
            check(f"18.4 {_run} {key}: stored mean equals the per-seed mean",
                  float(np.mean(blk["per_seed"])), float(blk["mean"]), 1e-9)
            vals.append(f"{float(np.mean(blk['per_seed'])):.3f}")
        check_true(f"FINDINGS 18.4 table row: {lab}", f"| {lab} | " + " | ".join(vals) + " |" in _f18t)

    for _lab, _key in (("trained arm, own policy", "survival own"),
                       ("trained arm, one scripted action sequence", "survival scripted"),
                       ("untrained trunk replaying the trained arm's episodes", "untrained replay"),
                       ("predictor trunk replaying them", "predictor replay"),
                       ("predictor trained on the trained arm's own batches, replaying them",
                        "predictor_logged replay")):
        _row18(_lab, _p18, _key)
    for _lab, _key in (("none: the primary probe", "target"),
                       ("behavior trace (ridge)", "resid_trace"),
                       ("observation history plus behavior (ridge)", "resid_obs_hist_beh"),
                       ("best nonlinear control (64-unit MLP)", "resid_joint_mlp"),
                       ("observation stream alone, no state: summary decoder", "obs_summary_only"),
                       ("observation stream alone, no state: sequence GRU", "seq_gru")):
        _row18(_lab, _d18, f"d=0.45 survival {_key}")
    _gn = _load_art("texture", "T_gn_l3_h96_wm.json")
    _gs = _gn["primary"]["survival"]
    check("18.6 white-jitter survival mean", float(np.mean(_gs["per_seed"])), 0.541)
    _glo, _ghi = t_ci90(_gs["per_seed"])
    check("18.6 white-jitter survival t90 lower", float(_glo), 0.523)
    check("18.6 white-jitter survival t90 upper", float(_ghi), 0.559)
    check_true("18.6 white-jitter stored survival mean and t90 equal the per-seed recomputation",
               _same(float(_gs["mean"]), float(np.mean(_gs["per_seed"])))
               and _same(float(_gs["t90"][0]), float(_glo)) and _same(float(_gs["t90"][1]), float(_ghi)))
    _ge0 = texture_l0_gate_reproduces("18.6", _gn)
    check("18.6 white-jitter drift-0 survival", _seed_mean(_gn["l0_control"]["survival"]), 0.559)
    check("18.6 white-jitter L0 TOST p", float(_ge0.p_value), 0.94, tol=0.005)
    check_true("18.6 quotes the white-jitter L0 reading and the survival interval",
               "drift-0 survival 0.559, TOST p = 0.94" in _f18t
               and f"survival arm reads {np.mean(_gs['per_seed']):.3f} [{_glo:.3f}, {_ghi:.3f}]" in _f18t)
    check_true("18.6 white-jitter run routes UNINFORMATIVE on the L0 gate",
               _gn["routing"].startswith("UNINFORMATIVE") and _gn["gates_failed"] == ["l0_tost"])
    check_true("18.6 records the white-jitter run as uninformative", "**uninformative**" in _f18)
    # 2026-10-16 (review of finding c27): the descriptive 18.6 numbers, recomputed from the
    # per-seed arms and the stored gate counts.
    _gl = {a: float(np.mean(_gs["per_seed"])) - float(np.mean(_gn["primary"][a]["per_seed"]))
           for a in ("predictor", "untrained")}
    _gge = int(np.sum(np.asarray(_gs["per_seed"], float) >= BAR))
    _geng, _gsp = _gn["gates"]["engagement"], _gn["gates"]["speed_positive_control"]
    _gne = sum(bool(c["engaged"]) for c in _geng["per_cell"])
    check_true("18.6 white-jitter stored leads, seeds at the bar and engagement count equal the recomputation",
               all(_same(_gl[a], float(_gn["contrasts"][a]["mean"])) for a in _gl)
               and _gge == int(_gs["seeds_at_or_above_bar"]) and _gne == int(_geng["n_engaged"])
               and len(_geng["per_cell"]) == int(_geng["n_cells"]))
    _words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    check_true("18.6 quotes the seeds at the bar, both leads and the engagement and speed counts",
               f"{_words[_gge]} of {_words[len(_gs['per_seed'])]} seeds at the bar, {_gl['predictor']:+.3f} "
               f"over the predictor and {_gl['untrained']:+.3f} over the untrained arm; engagement passes in "
               f"{_gne} of {len(_geng['per_cell'])} cells and the speed positive control in "
               f"{_gsp['n_pools'] - _gsp['n_below']} of {_gsp['n_pools']} pools" in _f18t)
    _c1r_same = 0
    for _p in sorted(glob.glob(os.path.join(ARTROOT, "goal_stakes", "C1-rerun", "cells", "cell_d*_s*.json"))):
        with open(_p, encoding="utf-8") as _fh:
            _r = json.load(_fh)["cell"]["agents"]
        with open(os.path.join(ARTROOT, "corrected_runs", "corrected_l3_h8_wm", "cells",
                               os.path.basename(_p)), encoding="utf-8") as _fh:
            _c = json.load(_fh)["cell"]["agents"]
        _c1r_same += sum(_r[a]["pool"]["target"] == _c[a]["pool"]["target"]
                         for a in ("survival", "predictor", "untrained"))
    check_int("18.1 C1 rerun: arm-cells bit-equal to C1", _c1r_same, 60)

    # ---- FINDINGS 22.3: the correlated-jitter sweeps A, B and C (2026-10-16, finding c27) ----
    # Every table row, both contrasts and the quoted ranges are rebuilt from the per-cell fresh
    # and transfer readings in each point file; the summary's per-seed lists and intervals must
    # equal that rebuild.
    print("\n== FINDINGS 22.3: correlated-jitter sweeps (texture/ar_sweep, texture/ar_sweep_c) ==")
    # Section 22 only: section 23 also says "Reading 2 is met", so a whole-file search would
    # pass with the 22.3 sentence deleted.
    _f22raw = _read("docs/FINDINGS.md")
    _f22 = " ".join(_f22raw[_f22raw.index("\n## 22. "):_f22raw.index("\n## 23. ")].split())
    _iv3 = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"

    def _ar_point(sub, pt):
        cells = sorted(_load_art("texture", sub, pt["file"])["cells"], key=lambda c: c["seed"])
        arm = lambda a, k: np.asarray([c[k] for c in cells if c["arm"] == a], float)
        sv, un = arm("survival", "fresh"), arm("untrained", "fresh")
        m = sv - un
        ok = (len(sv) == len(un) == 10
              and list(sv) == list(pt["survival_fresh"]["per_seed"]) and list(m) == list(pt["M"]["per_seed"])
              and _same(float(sv.mean()), pt["survival_fresh"]["mean"]) and _same(float(m.mean()), pt["M"]["mean"])
              and all(_same(a, b) for a, b in zip(t_ci90(m), pt["M"]["t90"]))
              and _same(float(un.mean()), pt["untrained_fresh_mean"])
              and _same(float(arm("survival", "transfer").mean()), pt["survival_transfer_mean"]))
        check_true(f"22.3 {pt['sweep']} rho {pt['rho']}: summary per-seed values and intervals rebuild "
                   f"from {sub}/{pt['file']}", ok)
        return sv, m, float(arm("survival", "transfer").mean())

    for _sub, _ckey, _pairs in (
            ("ar_sweep", "contrasts", (("A", "A: M(0.99) - M(0)", "+0.029 [-0.009, +0.068]"),
                                       ("B", "B: M(0.99) - M(0)", "-0.006 [-0.036, +0.023]"))),
            ("ar_sweep_c", "contrast", (("C", "C: M(0.99) - M(0)", "+0.009 [-0.027, +0.045]"),))):
        _sum = _load_art("texture", _sub, "summary.json")
        _pts = {}
        for _pt in _sum["points"]:
            _sv, _m, _tr = _ar_point(_sub, _pt)
            _pts[(_pt["sweep"], _pt["rho"])] = (_pt, _sv, _m, _tr)
            if _sub == "ar_sweep":
                _pr = _pt["profile"]
                _lab = "A, B" if (_pt["sweep"], _pt["rho"]) == ("A", 0.9) else _pt["sweep"]
                _row = (f"| {_lab} | {_pt['rho']:g} | {_pr['lag1_autocorr']:.2f} | "
                        f"{_pr['rollout_gap']['24']:.3f} | {_sv.mean():.3f} | {_m.mean():+.3f} "
                        f"{_iv3(t_ci90(_m))} |")
            else:
                _pr = _pt["calibration"]["profile"]
                _row = (f"| {_pt['rho']:g} | {_pt['sigma']:.5f} | {_pr['lag1_autocorr']:.3f} | "
                        f"{_pr['rms_one_step']:.4f} | {_pr['rollout_gap']['24']:.4f} | {_sv.mean():.3f} "
                        f"[{t_ci90(_sv)[0]:.3f}, {t_ci90(_sv)[1]:.3f}] | {_m.mean():+.3f} {_iv3(t_ci90(_m))} |")
            check_true(f"FINDINGS 22.3 table row, sweep {_pt['sweep']} rho {_pt['rho']}", _row in _f22)
        for _sw, _ck, _quote in _pairs:
            _hi_pt, _lo_pt = _pts[(_sw, 0.99)], _pts[(_sw, 0.0)]
            _d = _hi_pt[2] - _lo_pt[2]
            _dt = t_ci90(_d)
            _st = _sum[_ckey][_ck]
            check_true(f"22.3 contrast {_ck}: stored mean and t90 equal the paired recomputation",
                       _st["n"] == len(_d) and _same(float(_d.mean()), _st["mean"])
                       and all(_same(a, b) for a, b in zip(_dt, _st["t90"])))
            check_true(f"FINDINGS 22.3 quotes contrast {_ck}",
                       f"{_d.mean():+.3f} {_iv3(_dt)}" == _quote and _quote in _f22)
            if _sw == "C":
                # The stated wording follows the recomputation, not only the stored reading.
                _in = bool(-0.05 <= _dt[0] and _dt[1] <= 0.05)
                check_true("22.3 sweep C: reading 2 (interval inside [-0.05, +0.05]) as stored and stated",
                           _in is bool(_sum["readings"]["2 within the margin unit"])
                           and ("Reading 2 is met" in _f22) is _in and ("Reading 2 is not met" in _f22) is not _in)
        _svs = [v[1].mean() for v in _pts.values()]
        _trs = [v[3] for v in _pts.values()]
        _rng = f"survival {'stays at' if _sub == 'ar_sweep' else 'fresh stays at'} {min(_svs):.3f} to {max(_svs):.3f}"
        check_true(f"FINDINGS 22.3 quotes the {_sub} survival range", _rng in _f22)
        _trq = (f"transfers below 0.5 at every point ({min(_trs):.3f} to {max(_trs):.3f})" if _sub == "ar_sweep"
                else f"transfers at {min(_trs):.3f} to {max(_trs):.3f}")
        check_true(f"FINDINGS 22.3 quotes the {_sub} transfer range", _trq in _f22)
        check_true(f"22.3 {_sub}: no point reaches the bar and none is kept",
                   max(_svs) < BAR and not any(v[0]["kept"] for v in _pts.values()))

    # ---- FINDINGS 22.3: sweep C follow-up at the exactly matched gap (2026-10-16) ----
    # Each point's margins rebuild from its cells; its sigma gives the target gap exactly in the
    # closed form; the contrast with the stored rho = 0 point rebuilds, and FINDINGS quotes it with
    # reading 3.
    _fu_dir = os.path.join(ARTROOT, "texture", "ar_sweep_c_followup")
    if _artifact_present(os.path.join(_fu_dir, "summary.json")):
        _fu = _load_art("texture", "ar_sweep_c_followup", "summary.json")

        def _gap24(sig, rho, lam=0.925, h=24):
            _t = np.arange(h)
            _w = lam ** (h - 1 - _t)
            return float(sig * np.sqrt(2.0 * _w @ (rho ** np.abs(_t[:, None] - _t[None, :])) @ _w))

        check_true("22.3 follow-up: the target is the shared point's exact gap",
                   _same(_gap24(0.01, 0.9), _fu["gap_target_exact"]))
        _fum = {}
        for _pt in _fu["points"]:
            _cells = sorted(_load_art("texture", "ar_sweep_c_followup", _pt["file"])["cells"],
                            key=lambda c: c["seed"])
            _sv = np.asarray([c["fresh"] for c in _cells if c["arm"] == "survival"], float)
            _un = np.asarray([c["fresh"] for c in _cells if c["arm"] == "untrained"], float)
            _m = _sv - _un
            _fum[_pt["rho"]] = _m
            check_true(f"22.3 follow-up rho {_pt['rho']}: margins rebuild from the cells and the gap is matched",
                       len(_m) == 10 and list(_m) == list(_pt["M"]["per_seed"])
                       and all(_same(a, b) for a, b in zip(t_ci90(_m), _pt["M"]["t90"]))
                       and abs(_gap24(_pt["sigma"], _pt["rho"]) - _fu["gap_target_exact"]) < 1e-12
                       and not _pt["kept"])
            _row = (f"| {_pt['rho']:g} | {_pt['sigma']:.5f} | {_gap24(_pt['sigma'], _pt['rho']):.4f} | "
                    f"{_sv.mean():.3f} | {_un.mean():.3f} | {_m.mean():+.3f} {_iv3(t_ci90(_m))} |")
            check_true(f"FINDINGS 22.3 follow-up table row, rho {_pt['rho']}", _row in _f22)
        _m0 = _ar_point("ar_sweep_c", next(q for q in _load_art("texture", "ar_sweep_c", "summary.json")["points"]
                                          if q["rho"] == 0.0))[1]
        _d = _fum[0.99] - _m0
        _dt = t_ci90(_d)
        _st = _fu["contrast"]["follow-up: M(0.99, matched) - M(0, stored)"]
        check_true("22.3 follow-up contrast: stored mean and t90 equal the paired recomputation",
                   _st["n"] == len(_d) and _same(float(_d.mean()), _st["mean"])
                   and all(_same(a, b) for a, b in zip(_dt, _st["t90"])))
        _r1, _r2 = bool(_dt[0] > 0), bool(-0.05 <= _dt[0] and _dt[1] <= 0.05)
        check_true("22.3 follow-up readings follow the recomputation",
                   _fu["readings"] == {"1 coherence raises at matched gap": _r1, "2 within the margin unit": _r2,
                                       "3 neither": not _r1 and not _r2})
        check_true("FINDINGS 22.3 quotes the follow-up contrast and its reading",
                   f"M(0.99) - M(0) is {_d.mean():+.3f} {_iv3(_dt)}: reading 3 (neither)" in _f22
                   and not _r1 and not _r2)

    # ---- FINDINGS 23.3: the online erasure readout's refit probe (2026-10-16) ----
    # The refit probe's mean and t-interval per arm are rebuilt from rows[].refit_probe_erased
    # with the paired_t90 the run's summarize() used, and the partial-erasure reading from that
    # mean. The superseded split-twin artifact must differ from the rerun only in the refit value
    # and the wall-clock seconds, and the correction note quotes both runs' values.
    print("\n== FINDINGS 23.3: online erasure, refit probe (mechanism/erasure) ==")
    from itasorl.novel_readouts import paired_t90 as _pt90
    _erp = os.path.join(ARTROOT, "mechanism", "erasure", "corrected_l3_h8_wm.json")
    _ers = os.path.join(ARTROOT, "mechanism", "erasure", "superseded_split_twins_2026-10-16",
                        "corrected_l3_h8_wm.json")
    _er_ok = _artifact_present(_erp)
    _ers_ok = _artifact_present(_ers)
    if _er_ok:
        _fx = _read("docs/FINDINGS.md")
        _s233 = " ".join(_fx[_fx.index("### 23.3 "):_fx.index("### 23.4 ")].split())

        def _refit(doc, arm, tag=""):
            vals = [r.get("refit_probe_erased", np.nan) for r in doc["rows"] if r["arm"] == arm]
            rec = _pt90([np.nan if v is None else v for v in vals])
            st = doc["summary"][arm]["refit_probe_erased"]
            check_true(f"23.3 {tag}{arm} refit probe: stored mean and t90 equal the recomputation from rows",
                       rec["n"] == st["n"] and _same(rec["mean"], st["mean"])
                       and all(_same(a, b) for a, b in zip(rec["t90"], st["t90"])))
            return rec

        _er = _load_art("mechanism", "erasure", "corrected_l3_h8_wm.json")
        _rs, _ru = _refit(_er, "survival"), _refit(_er, "untrained")
        check("23.3 survival refit probe mean recomputed (0.747)", _rs["mean"], 0.747)
        check("23.3 survival refit probe t90 lower recomputed (0.702)", _rs["t90"][0], 0.702)
        check("23.3 survival refit probe t90 upper recomputed (0.791)", _rs["t90"][1], 0.791)
        check("23.3 untrained refit probe mean recomputed (0.576)", _ru["mean"], 0.576)
        for _arm, _rec in (("survival", _rs), ("untrained", _ru)):
            _pe = bool(_rec["mean"] >= 0.65)
            check_true(f"23.3 {_arm}: partial erasure (refit probe >= 0.65) is {_pe}, as stored",
                       _pe is bool(_er["summary"][_arm]["readings"]["partial erasure (refit probe >= 0.65)"]))
        _sv = f"{_rs['mean']:.3f} [{_rs['t90'][0]:.3f}, {_rs['t90'][1]:.3f}]"
        check_true("FINDINGS 23.3 table quotes the survival refit probe",
                   f"| refit probe on erased states | {_sv} " in _s233)
        check_true("FINDINGS 23.3 quotes the untrained refit probe",
                   f"refit {_ru['mean']:.3f})" in _s233)
        _pe_s = bool(_rs["mean"] >= 0.65)
        check_true(f"FINDINGS 23.3 {'attaches' if _pe_s else 'does not attach'} the partial-erasure "
                   f"label, as the recomputed survival refit gives",
                   ("partial erasure** label is attached" in _s233) is _pe_s
                   and f"reads the erased states at {_rs['mean']:.3f}" in _s233)
        if _ers_ok:
            _eo = _load_art("mechanism", "erasure", "superseded_split_twins_2026-10-16",
                            "corrected_l3_h8_wm.json")
            _skip = {"refit_probe_erased", "seconds"}
            check_true("23.3 superseded split-twin rows equal the rerun's outside the refit probe "
                       "and wall-clock seconds",
                       len(_eo["rows"]) == len(_er["rows"])
                       and all({k: v for k, v in a.items() if k not in _skip}
                               == {k: v for k, v in b.items() if k not in _skip}
                               for a, b in zip(_eo["rows"], _er["rows"])))
            _os, _ou = _refit(_eo, "survival", "superseded "), _refit(_eo, "untrained", "superseded ")
            check_true("FINDINGS 23.3 correction note quotes both runs' refit values",
                       f"the refit reads {_sv} (was {_os['mean']:.3f} [{_os['t90'][0]:.3f}, "
                       f"{_os['t90'][1]:.3f}]) and the untrained arm {_ru['mean']:.3f} "
                       f"(was {_ou['mean']:.3f})" in _s233)
            _chg = {r["seed"]: (a["refit_probe_erased"], r["refit_probe_erased"])
                    for a, r in zip(_eo["rows"], _er["rows"]) if r["arm"] == "survival"}
            check_true("23.3 correction note: seeds 1 and 5 unchanged, the other eight rise",
                       sorted(s for s, (a, b) in _chg.items() if a == b) == [1, 5]
                       and len(_chg) - 2 == 8
                       and all(b > a for s, (a, b) in _chg.items() if s not in (1, 5))
                       and "Seeds 1 and 5, whose twins already shared a fold, are unchanged; "
                           "the other eight rise." in _s233)

    # ---- O2: long-history ridge with the penalty chosen in fold (2026-10-16) ----
    # The manuscript's Limitations and claims row 46 and FINDINGS 17.8 cite this optional
    # readout, and the supplement ships it, so its numbers are guarded like an artifact's. Each
    # mean and t90 is recomputed from summary.<basis>.<metric>.per_seed, which must equal the
    # per-seed cells; the fixed-penalty rows must reproduce the battery's survival readings in
    # artifacts/control_diagnostics/corrected_l3_h8_wm.json within 0.001; the table in
    # docs/optional/README.md must quote what the seeds give.
    print("\n== O2: long-history ridge, penalty chosen in fold (docs/optional) ==")
    _o2p = os.path.join(root, "docs", "optional", "o2_long_history_ridge.json")
    check_true("file present: docs/optional/o2_long_history_ridge.json", os.path.isfile(_o2p))
    _o2bp = os.path.join(ARTROOT, "control_diagnostics", "corrected_l3_h8_wm.json")
    if os.path.isfile(_o2p) and _artifact_present(_o2bp):
        with open(_o2p, encoding="utf-8") as _fh:
            _o2 = json.load(_fh)
        _o2b = _load_art("control_diagnostics", "corrected_l3_h8_wm.json")["aggregate"]
        _o2r = " ".join(_read("docs/optional/README.md").split())
        check_true("O2 provenance names its script and disclaims adjudication",
                   _o2["provenance"]["generated_by"] == "scripts/run_optional_o2.py"
                   and "not an adjudication" in _o2["provenance"]["status"])
        _o2cells = sorted(_o2["cells"], key=lambda c: c["seed"])
        check_true("O2 has one cell per seed 0 to 9", [c["seed"] for c in _o2cells] == list(range(10)))
        _o2iv = {}
        for _bas, _name, _exp in (
                ("resid_obs_hist", "history + traces + actions",
                 {"auroc": (0.737, 0.690, 0.785), "nuisance_r2_heldout": (0.906, 0.904, 0.908),
                  "fixed_auroc": (0.640, 0.608, 0.671), "fixed_r2": (-0.128, -0.234, -0.021)}),
                ("resid_obs_hist_beh", "the same plus behavior",
                 {"auroc": (0.691, 0.650, 0.732), "nuisance_r2_heldout": (0.909, 0.907, 0.910),
                  "fixed_auroc": (0.606, 0.581, 0.630), "fixed_r2": (-0.177, -0.292, -0.063)})):
            _sm = _o2["summary"][_bas]
            _fromcells = {"auroc": [c[_bas]["auroc"] for c in _o2cells],
                          "nuisance_r2_heldout": [c[_bas]["nuisance_r2_heldout"] for c in _o2cells],
                          "fixed_auroc": [c[_bas]["fixed_alpha"]["auroc"] for c in _o2cells],
                          "fixed_r2": [c[_bas]["fixed_alpha"]["nuisance_r2_heldout"] for c in _o2cells]}
            for _met, (_mu, _lo, _hi) in _exp.items():
                _ps = np.asarray(_sm[_met]["per_seed"], float)
                _tl, _th = t_ci90(_ps)
                _o2iv[(_bas, _met)] = (float(_ps.mean()), float(_tl), float(_th))
                check_true(f"O2 {_bas} {_met}: per_seed equals the cells, and the stored mean and "
                           "t90 equal the recomputation",
                           list(_ps) == _fromcells[_met] and _sm[_met]["n"] == len(_ps) == 10
                           and _same(float(_ps.mean()), _sm[_met]["mean"])
                           and all(_same(a, b) for a, b in zip((_tl, _th), _sm[_met]["t90"])))
                check(f"O2 {_bas} {_met} mean recomputed ({_mu:.3f})", float(_ps.mean()), _mu)
                check(f"O2 {_bas} {_met} t90 lower recomputed ({_lo:.3f})", float(_tl), _lo)
                check(f"O2 {_bas} {_met} t90 upper recomputed ({_hi:.3f})", float(_th), _hi)
            _bat = _o2b[f"d=0.45 survival {_bas}"]
            check(f"O2 {_bas}: the fixed-penalty AUROC reproduces the battery's survival reading "
                  f"({_seed_mean(_bat):.3f}) within 0.001",
                  _o2iv[(_bas, "fixed_auroc")][0], _seed_mean(_bat), tol=1e-3)
            check_true(f"O2 {_bas}: the penalty is chosen from the grid on every fold, never the "
                       "fixed 0.001",
                       all(len(c[_bas]["alphas"]) == c[_bas]["n_folds"] and min(c[_bas]["alphas"]) > 1e-3
                           for c in _o2cells))

            def _cell(met):
                m, lo, hi = _o2iv[(_bas, met)]
                return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

            check_true(f"docs/optional/README.md O2 table row quotes the recomputed {_bas} values",
                       f"| {_name} | {_cell('fixed_auroc')}, {_cell('fixed_r2')} | "
                       f"{_cell('auroc')}, {_cell('nuisance_r2_heldout')} |" in _o2r)

    # ---- README: the corrected L3 headline sentence (2026-10-16, finding c27) ----
    # Rendered from corrected_verdicts.json runs.C1.primary, the source FINDINGS 17 quotes; the
    # survival mean and interval are recomputed from its per-seed list.
    print("\n== README: corrected L3 headline ==")
    _c1p = _load_art("corrected_verdicts.json")["runs"]["C1"]["primary"]
    _c1s = _c1p["survival"]["per_seed"]
    _c1lo, _c1hi = t_ci90(_c1s)
    check("README corrected headline: survival mean recomputed (0.733)", float(np.mean(_c1s)), 0.733)
    check("README corrected headline: survival t90 lower recomputed (0.669)", float(_c1lo), 0.669)
    check("README corrected headline: survival t90 upper recomputed (0.797)", float(_c1hi), 0.797)
    _rm = " ".join(_read("README.md").replace("**", "").split())
    _hl = (f"reads {np.mean(_c1s):.3f} (t-based 90% CI [{_c1lo:.3f}, {_c1hi:.3f}], "
           f"{sum(x >= BAR for x in _c1s)} of {len(_c1s)} seeds at the 0.65 bar), the predictor arm "
           f"{_seed_mean(_c1p['predictor']):.3f} and the untrained arm {_seed_mean(_c1p['untrained']):.3f}, "
           f"with paired margins of {_c1p['contrast_vs_predictor']['mean']:+.3f} and "
           f"{_c1p['contrast_vs_untrained']['mean']:+.3f}")
    check_true("README quotes the corrected L3 headline as corrected_verdicts.json gives it", _hl in _rm)
    check_true("README corrected headline margins recompute from the per-seed lists",
               _same(float(np.mean(np.asarray(_c1s) - np.asarray(_c1p["predictor"]["per_seed"]))),
                     _c1p["contrast_vs_predictor"]["mean"], 1e-5)
               and _same(float(np.mean(np.asarray(_c1s) - np.asarray(_c1p["untrained"]["per_seed"]))),
                         _c1p["contrast_vs_untrained"]["mean"], 1e-5))

    agent_free_checks(_read("docs/FINDINGS.md"), _read("docs/CORRECTIONS.md"))
    repro_check_checks(_read("docs/FINDINGS.md"), _read("docs/CORRECTIONS.md"),
                       _read("docs/REPRODUCE.md"))
    post_review_checks(_read("docs/FINDINGS.md"), _read("docs/PREREGISTRATION_L3.md"))
    _f21 = " ".join(_read("docs/FINDINGS.md").split())
    for _rid, _lo, _hi, _mp, _mu in (
            ("b1", 0.772, 0.829, 0.095, 0.161),
            ("b2", 0.825, 0.860, 0.143, 0.212)):
        _doc = _load_art("balanced_confirmation", f"{_rid}.json")
        _s = np.asarray(_doc["balanced"]["survival"]["per_seed"], float)
        _slo, _shi = t_ci90(_s)
        check(f"21 {_rid} survival t90 lo recomputed", float(_slo), _doc["balanced"]["survival"]["t90"][0])
        check(f"21 {_rid} survival t90 hi recomputed", float(_shi), _doc["balanced"]["survival"]["t90"][1])
        check(f"21 {_rid} survival mean recomputed", float(_s.mean()), _doc["balanced"]["survival"]["mean"])
        check_true(f"21 {_rid} decision is UNINFORMATIVE", _doc["decision"] == "UNINFORMATIVE")
        check_true(f"21 {_rid} untrained floor fails", _doc["gates"]["untrained_floor_ok"] is False)
        _udev = abs(_seed_mean(_doc["balanced"]["untrained"]) - 0.5)
        check(f"21 {_rid} untrained deviation", _udev, 0.114)
        check_true(f"21 {_rid} stored untrained deviation equals the per-seed recomputation",
                   _same(float(_doc["gates"]["untrained_dev"]), _udev))
        check_true(f"FINDINGS 21 quotes {_rid} survival interval",
                   f"[{_lo:.3f}, {_hi:.3f}]" in _f21)
        check_true(f"FINDINGS 21 quotes {_rid} margin lower bounds",
                   f"{_mp:.3f}" in _f21 and f"{_mu:.3f}" in _f21)
    check_true("FINDINGS 21 names the decision", "Both runs are UNINFORMATIVE" in _f21)
    # H01: the pooled readings on the registered pair that the record quotes for B1 and B2
    # (descriptive; decides nothing). Each mean and interval is recomputed from its per-seed list,
    # and each quoted three-decimal value is tied to its mean.
    _bp = _load_art("balanced_pooled", "registered_pair.json")
    for _rid, _run in sorted(_bp["runs"].items()):
        for _d, _arms in sorted(_run["pool_target"].items()):
            for _arm, _cell in sorted(_arms.items()):
                _v = np.asarray(_cell["per_seed"], float)
                _vlo, _vhi = t_ci90(_v)
                check(f"H01 {_rid} d{_d} {_arm} pooled mean recomputed", float(_v.mean()), _cell["mean"])
                check(f"H01 {_rid} d{_d} {_arm} pooled t90 lo recomputed", float(_vlo), _cell["t90"][0])
                check(f"H01 {_rid} d{_d} {_arm} pooled t90 hi recomputed", float(_vhi), _cell["t90"][1])
    for _rid, _d, _arm, _q in (("B1", "0.45", "survival", 0.758), ("B2", "0.45", "survival", 0.835),
                               ("B1", "0.45", "predictor", 0.586), ("B1", "0.45", "untrained", 0.516),
                               ("B1", "0.00", "survival", 0.561), ("B2", "0.00", "survival", 0.536),
                               ("B1", "0.00", "predictor", 0.498), ("B1", "0.00", "untrained", 0.503)):
        check(f"pinned: H01 {_rid} d{_d} {_arm} pooled mean as quoted", round(_bp["runs"][_rid]["pool_target"][_d][_arm]["mean"], 3), _q)
    for _ename, _mean, _lo, _hi in (
            ("sensory_echo_b1.json", 0.761, 0.727, 0.795),
            ("sensory_echo_b2.json", 0.780, 0.757, 0.804)):
        _ed = _load_art("balanced_confirmation", _ename)
        check_true(f"21 {_ename} integrity", _ed["integrity"]["all_match"] is True
                   and _ed["integrity"]["mismatches"] == [])
        check_int(f"21 {_ename} cells", len(_ed["cells"]), 30)
        check_true(f"21 {_ename} recorded targets reproduce",
                   max(c["target_abs_diff"] for c in _ed["cells"]) <= 5e-4
                   and all(c["dropped_pairs"] == 0 and c["identical_first_state"] for c in _ed["cells"]))
        check(f"21 {_ename} untrained resid_obs matches B1",
              float(np.mean(_ed["aggregate"]["d=0.45 untrained"]["resid_obs"]["per_seed"])),
              0.5963223140495867)
        for _arm in ("survival", "predictor", "untrained"):
            for _met in ("target", "obs_trace_only", "resid_trace", "resid_obs",
                         "resid_obs_int", "resid_obs_beh"):
                _blk = _ed["aggregate"][f"d=0.45 {_arm}"][_met]
                _ps = np.asarray(_blk["per_seed"], float)
                check(f"21 {_ename} {_arm} {_met} mean", float(_ps.mean()), float(_blk["mean"]))
                _tlo, _thi = t_ci90(_ps)
                check(f"21 {_ename} {_arm} {_met} t90 lo", float(_tlo), float(_blk["t90"][0]))
                check(f"21 {_ename} {_arm} {_met} t90 hi", float(_thi), float(_blk["t90"][1]))
        _sv = _ed["aggregate"]["d=0.45 survival"]["resid_obs"]
        _un = float(np.mean(_ed["aggregate"]["d=0.45 untrained"]["resid_obs"]["per_seed"]))
        _decb = _ed["decision_balanced"]
        check_true(f"21 {_ename} floor explained",
                   abs(_un - 0.5) <= 0.1 and _decb["floor_explained"] is True)
        check_true(f"21 {_ename} trained clause",
                   float(_sv["mean"]) >= 0.65 and float(_sv["mean"]) > _un + 0.05
                   and float(_sv["t90"][0]) > _un and _decb["pass_bar"] and _decb["pass_margin"]
                   and _decb["t90_lo_above_untrained"])
        check_true(f"FINDINGS 21 quotes {_ename} resid_obs",
                   f"{_mean:.3f} [{_lo:.3f}, {_hi:.3f}]" in _f21)
    check_true("FINDINGS 21 names the sensory-echo decision",
               "sensory-independent under matched worlds" in _f21)

    print()
    if n_skipped:
        print(f"skipped: {n_skipped} check(s) whose file the anonymized supplement leaves out.")
    print(f"pins: {n_pins} of {n_checks} checks compare a value as stored in an artifact with a "
          "constant in this script (labels start 'pinned:'): readings stored without per-seed "
          "values, configuration values, and recorded integrity results.")
    if failures:
        print(f"RESULT: {len(failures)} of {n_checks} checks FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"RESULT: all {n_checks} checks passed.")
    return 0


# ---- The per-value rerun at 4b6e1f3 (CORRECTIONS 2026-10-11) -------------------------------
# The prose says the second rerun reproduced all 157 recorded values of both seed-0 cells with
# matching fingerprints. Recount that from the saved rerun cells and the committed cells, so
# the sentence cannot outlive the files. The comparator is scripts/compare_run_cells.py (G59),
# the committed tool a reader runs on a reproduced run; its entry for each cell must also equal
# the stored report's entry, so the report and the tool cannot drift apart.

def repro_check_checks(findings: str, corrections: str, reproduce: str) -> None:
    print("\n== per-value rerun at 4b6e1f3 (CORRECTIONS 2026-10-11) ==")
    from compare_run_cells import compare_docs

    rep = _af_load("repro_check", "per_value_4b6e1f3_l3_h8_wm_cpu_s0.json")
    for name in ("cell_d0.00_s0.json", "cell_d0.45_s0.json"):
        rerun = _af_load("repro_check", "rerun_cells", name)
        committed = _af_load("reviewer_gaps_runs", "l3_h8_wm_cpu", "cells", name)
        e = compare_docs(rerun, committed)
        check_true(f"{name}: the rerun at 4b6e1f3 has the committed cell's 157 leaves and "
                   "fingerprint, and every value matches",
                   e["n_shared_leaves"] == 157 and not e["only_in_rerun"]
                   and not e["only_in_committed"] and e["n_differ"] == 0
                   and e["fingerprint_match"] and e["fingerprint_rerun"] == committed["fingerprint"]
                   and rep["cells"][name]["n_differ"] == 0 and e == rep["cells"][name])
    check_true("the per-value report records 4b6e1f3 and the pinned stack",
               rep["commit_rerun"] == "4b6e1f3" and rep["stack"]["torch"].startswith("2.14.1")
               and rep["all_shared_values_identical"] is True)
    for where, text in (("FINDINGS 17.1", findings), ("CORRECTIONS", corrections),
                        ("REPRODUCE", reproduce)):
        flat_text = " ".join(text.split())
        check_true(f"{where} states the 157-value rerun and points at its report",
                   "157 recorded values" in flat_text
                   and "per_value_4b6e1f3_l3_h8_wm_cpu_s0.json" in flat_text)


# ---- FINDINGS 19: post-review analyses without agents (2026-10-11) ------------------------
# Each table row of section 19, and each number its dated notes elsewhere quote, is rebuilt
# here from the committed artifacts. Gate-0 points, pooled AUROCs and intervals are recomputed
# from the stored out-of-fold rows; the correlation matrix and its bootstrap from the stored
# per-episode sums; the random-forest and promotion summaries from their per-seed values; the
# selection-intensity readings from the stored per-generation counts.

AF_G0_LABEL = {"mlp_h8_s0": "hidden 8, G seed 0", "mlp_h7_s0": "hidden 7, G seed 0",
               "mlp_h4_s0": "hidden 4, G seed 0", "mlp_h8_s1": "hidden 8, G seed 1",
               "mlp_h7_s1": "hidden 7, G seed 1", "mlp_h9_s1": "hidden 9, G seed 1",
               "mlp_h10_s1": "hidden 10, G seed 1", "mlp_h8_s2": "hidden 8, G seed 2",
               "rff_D80": "`G_rff`, D = 80", "gn_0.01": "`G_gn`, sigma_v = 0.01"}
AF_RF_NAME = {"h8_s0": "h8 s0", "h7_s0": "h7 s0", "h4_s0": "h4 s0", "h10_s1": "h10 s1",
              "h8_s2": "h8 s2", "rff_D80": "rff D80"}
AF_RF_PAIRS = (("h8_s0", "h7_s0"), ("h8_s0", "h8_s2"), ("h8_s0", "h10_s1"),
               ("h10_s1", "h8_s2"), ("h8_s0", "rff_D80"), ("h10_s1", "rff_D80"),
               ("h8_s2", "rff_D80"))


def _af_load(*parts) -> dict:
    with open(os.path.join(ARTROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _af_iv(t, sign: bool = False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"[{f.format(t[0])}, {f.format(t[1])}]"


def af_gate0() -> dict:
    """Section 19.1 rows, recomputed from the stored out-of-fold rows."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gate0_pair_intervals import intervals_from_oof
    from itasorl.stats import auroc
    doc = _af_load("agent_free", "gate0_pair_intervals.json")
    rows_rep, rows_iv, recomputed = [], [], {}
    for inst in doc["instances"]:
        iid, lab = inst["id"], AF_G0_LABEL[inst["id"]]
        sch = inst["schemes"]
        for s, b in sch.items():
            o = b["oof"]
            y, p, f = np.asarray(o["label"]), np.asarray(o["score"]), np.asarray(o["fold"])
            pt = float(np.mean([auroc(y[f == k], p[f == k]) for k in sorted(set(f.tolist()))]))
            check(f"19.1 {iid} {s}: fold-mean oracle AUROC recomputed from the OOF rows",
                  pt, b["oracle_auroc"], 1e-12)
        seen, rec = set(), []
        for r in inst["recorded"]:
            if r["value"] is None or "rounded" in r["note"]:
                continue
            key = (round(r["value"], 6), r["scheme"])
            if key not in seen:
                seen.add(key)
                rec.append(f"{r['value']:.6f} {r['scheme']}")
            check_true(f"19.1 {iid}: recorded {r['value']:.6f} ({r['scheme']}, {r['source']}) "
                       "reproduces exactly", abs(sch[r["scheme"]]["oracle_auroc"] - r["value"]) <= 1e-12)
        rows_rep.append((f"19.1 reproduction row {iid}",
                         f"| {lab} | {'; '.join(rec)} | {sch['legacy']['oracle_auroc']:.6f} | "
                         f"{sch['explicit']['oracle_auroc']:.6f} |"))
        split = inst["recorded"][0]["scheme"]
        iv = intervals_from_oof(sch[split]["oof"])
        for k in ("pair_cluster_ci95", "fold_mean_ci95", "share_draws_in_band"):
            check_true(f"19.1 {iid} {split}: {k} recomputes from the OOF rows",
                       np.allclose(iv[k], sch[split][k], rtol=0, atol=1e-12))
        recomputed[iid] = {"split": split, **iv, "explicit": sch["explicit"]["oracle_auroc"],
                           "legacy": sch["legacy"]["oracle_auroc"]}
        rows_iv.append((f"19.1 interval row {iid}",
                        f"| {lab} | {split} | {iv['point_fold_mean']:.3f} | "
                        f"{iv['pooled_oof_auroc']:.3f} | {_af_iv(iv['pair_cluster_ci95'])} | "
                        f"{_af_iv(iv['fold_mean_ci95'])} | {iv['share_draws_in_band']:.3f} | "
                        f"{iv['share_draws_below_band']:.3f} | {iv['share_draws_above_band']:.3f} |"))
    widths = [r["pair_cluster_ci95"][1] - r["pair_cluster_ci95"][0] for r in recomputed.values()]
    straddle = all(not (0.85 <= r[k][0] and r[k][1] <= 0.95) and r[k][0] <= 0.95 and r[k][1] >= 0.85
                   for r in recomputed.values() for k in ("pair_cluster_ci95", "fold_mean_ci95"))
    check_true("19.1: every 95% interval of either kind straddles a band edge", straddle)
    return {"rows": rows_rep + rows_iv, "recomputed": recomputed,
            "width_lo": min(widths), "width_hi": max(widths)}


def af_residual() -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from residual_field_correlation import summarize
    doc = _af_load("agent_free", "residual_field_correlation.json")
    names = doc["names"]
    s = summarize(doc["episode_stats"], names)
    check_true("19.2: correlation matrix and bootstrap recompute from the stored per-episode sums",
               np.allclose(s["matrices"]["pooled"]["r"], doc["matrices"]["pooled"]["r"], atol=1e-12)
               and np.allclose(s["matrices"]["pooled"]["ci95_lo"], doc["matrices"]["pooled"]["ci95_lo"],
                               atol=1e-12))
    R = np.asarray(s["matrices"]["pooled"]["r"])
    rows = []
    for i, a in enumerate(names[:-1]):
        cells = ["" if j < i else "1" if j == i else f"{R[i, j]:+.3f}" for j in range(len(names))]
        rows.append((f"19.2 matrix row {a}", f"| {AF_RF_NAME[a]} | " + " | ".join(cells) + " |"))
    pairs = {(p["a"], p["b"]): p for p in s["pairs"]}
    for a, b in AF_RF_PAIRS:
        p = pairs[(a, b)]
        rows.append((f"19.2 pair row {a} {b}",
                     f"| {AF_RF_NAME[a]}, {AF_RF_NAME[b]} | {p['r_pooled']:+.3f} | "
                     f"{_af_iv(p['ci95_pooled'], True)} | {p['r_x']:+.3f} | {p['r_y']:+.3f} |"))
    rows.append(("19.2 field sizes", ", ".join(f"{AF_RF_NAME[f['name']]} {f['rms']:.3f}"
                                                for f in doc["fields"])))
    mlp = [R[i, j] for i in range(len(names)) for j in range(i + 1, len(names))
           if "rff" not in names[i] and "rff" not in names[j]]
    rff = [R[i, names.index("rff_D80")] for i in range(len(names)) if names[i] != "rff_D80"]
    lo_hi = (f"{min(mlp):+.2f} and {max(mlp):+.2f}", f"{min(rff):+.2f} and {max(rff):+.2f}")
    rows.append(("19.2 MLP range", f"between {lo_hi[0]}"))
    rows.append(("19.2 rff range", f"between {lo_hi[1]}"))
    h87, h8r = pairs[("h8_s0", "h7_s0")], pairs[("h8_s0", "rff_D80")]
    return {"rows": rows, "h87": h87, "h8r": h8r}


def af_rf() -> dict:
    doc = _af_load("agent_free", "rf_probe_l3_n10_historical.json")
    check_true("19.3: random-forest artifact is complete and labeled historical",
               doc["complete"] and doc["label"].startswith("HISTORICAL"))
    check_true("19.3: linear probe reproduces the run record and the fold re-score exactly",
               doc["reproduction"]["vs_run_record"]["max_abs_diff_linear_target"] == 0.0
               and doc["reproduction"]["vs_run_record"]["max_abs_diff_linear_shuffled_first"] == 0.0
               and doc["reproduction"]["vs_fold_rescore"]["max_abs_diff_legacy"] == 0.0
               and doc["reproduction"]["vs_fold_rescore"]["max_abs_diff_explicit"] == 0.0)
    cells = doc["cells"]

    def agg(drift, arm, split, metric):
        v = [c[split][metric] for c in sorted(cells, key=lambda c: c["seed"])
             if c["drift"] == drift and c["agent"] == arm]
        st = doc["aggregate"][f"d={drift} {arm}"][split][metric]
        check(f"19.3 d={drift} {arm} {split} {metric} mean from cells",
              float(np.mean(v)), st["mean"], 1e-12)
        return float(np.mean(v)), t_ci(v), int(sum(x >= BAR for x in v))

    def cell(m):
        return f"{m[0]:.3f} {_af_iv(m[1])}, {m[2]}/10"

    rows = []
    for drift, dshow in (("0.45", "0.45"), ("0.00", "0")):
        for arm in ("survival", "predictor", "untrained"):
            lin, rf = agg(drift, arm, "legacy", "linear"), agg(drift, arm, "legacy", "rf")
            s1 = agg(drift, arm, "legacy", "linear_shuffled_first")[0]
            s8 = agg(drift, arm, "legacy", "linear_shuffled_mean8")[0]
            rfs = agg(drift, arm, "legacy", "rf_shuffled_first")
            rows.append((f"19.3 legacy row d={drift} {arm}",
                         f"| {dshow} | {arm} | {cell(lin)} | {s1:.3f} / {s8:.3f} | {cell(rf)} | "
                         f"{rfs[0]:.3f} {_af_iv(rfs[1])} |"))
            le, re_ = agg(drift, arm, "explicit", "linear"), agg(drift, arm, "explicit", "rf")
            rse = agg(drift, arm, "explicit", "rf_shuffled_first")
            rows.append((f"19.3 explicit row d={drift} {arm}",
                         f"| {dshow} | {arm} | {cell(le)} | {cell(re_)} | {rse[0]:.3f} {_af_iv(rse[1])} |"))
    att = doc["reproduction_attempts"]
    rec, sm = att["recorded"], att["summary"]
    u45, u0 = sm["d=0.45 untrained"], sm["d=0.00 untrained"]
    for lab, r, a, b in (("drift 0.45, target", rec["untrained d=0.45"], u45["rf_random_state_0"],
                          u45["rf_random_state_seed"]),
                         ("drift 0.45, shuffled", rec["untrained d=0.45 shuffled"],
                          u45["rf_random_state_0_shuffled"], u45["rf_random_state_seed_shuffled"]),
                         ("drift 0, target", rec["untrained d=0.00"], u0["rf_random_state_0"],
                          u0["rf_random_state_seed"])):
        rows.append((f"19.3 attempt row {lab}", f"| {lab} | {r:.3f} | {a:.3f} | {b:.3f} |"))
    ue = agg("0.45", "untrained", "explicit", "rf")
    return {"rows": rows, "untrained_explicit": f"{ue[0]:.3f} {_af_iv(ue[1])}",
            "closest": (f"{u45['rf_random_state_0']:.3f}", f"{u45['rf_random_state_seed']:.3f}")}


def af_promotions() -> dict:
    rows = {}
    for key, fn in (("tqd", ("texture", "T_qd_l3_h8_wm.json")),
                    ("h4", ("expB2", "failed_gates_l3_h4_traces.json"))):
        doc = _af_load(*fn)
        out = []
        for drift, dshow in (("0.45", "0.45"), ("0.00", "0")):
            for arm in ("survival", "predictor", "untrained"):
                e = doc["pooled_target"][drift][arm]
                v = e["per_seed"]
                m, t, b = float(np.mean(v)), t_ci(v), mean_ci(v, level=0.90)
                check(f"19.4 {key} d={drift} {arm} mean from per-seed", m, e["mean"], 1e-12)
                check_true(f"19.4 {key} d={drift} {arm} intervals from per-seed",
                           np.allclose(t, e["t90"], atol=1e-9) and np.allclose(b[1:], e["seed_boot90"], atol=1e-12))
                out.append((f"19.4 {key} row d={drift} {arm}",
                            f"| {dshow} | {arm} | {m:.3f} | {_af_iv(t)} | {_af_iv(b[1:])} | "
                            f"{sum(x >= BAR for x in v)}/10 |"))
        pools = doc["speed_pools"]
        n_below = sum(p["speed"] < 0.75 for p in pools)
        check_int(f"19.4 {key}: speed pools below 0.75 recomputed", n_below,
                  doc["gates"]["speed_positive_control"]["n_below"])
        rows[key] = {"rows": out, "doc": doc, "n_below": n_below,
                     "worst": min(p["speed"] for p in pools)}
    return rows


def af_selection() -> dict:
    doc = _af_load("expC", "selection_intensity.json")
    n = doc["n_individuals"]
    rows = []
    for b in doc["per_seed"]:
        for arm in ("treatment", "control"):
            e = b[arm]
            q = np.asarray(e["n_qualifiers"])
            fr = q / n
            first = int(np.flatnonzero(fr >= 0.9)[0])
            stays = next((g for g in range(len(fr)) if np.all(fr[g:] >= 0.9)), None)
            below = [g for g in range(first + 1, len(fr)) if fr[g] < 0.9]
            late = float(fr[5:].mean())
            f = np.asarray(e["mean_fitness"])
            pk = int(np.argmax(f))
            check_true(f"19.5 seed {b['seed']} {arm}: first/stays/late share recomputed",
                       first == e["first_gen_at_least_90pct"]
                       and stays == e["first_gen_staying_at_least_90pct"]
                       and abs(late - e["mean_share_gens_5_to_29"]) < 1e-12)
            rows.append((f"19.5 counts row seed {b['seed']} {arm}",
                         f"| {b['seed']} | {arm} | {e['threshold_gen0_median']:.3f} | "
                         + " | ".join(str(int(x)) for x in q[:8]) + " |"))
            rows.append((f"19.5 summary row seed {b['seed']} {arm}",
                         f"| {b['seed']} | {arm} | {first} | "
                         f"{stays if stays is not None else 'never'} | "
                         f"{', '.join(str(g) for g in below) if below else 'none'} | {late:.3f} | "
                         f"{f[0]:+.3f}, {f[pk]:+.3f} ({pk}), {f[-1]:+.3f} |"))
    check_true("19.5: main milestone run and per-individual rerun share every series",
               all(doc["main_run_series_identical"].values()))
    return {"rows": rows, "doc": doc}


def af_tables() -> dict:
    """Every row and phrase section 19 and its dated notes quote, keyed by subsection."""
    return {"g0": af_gate0(), "rfc": af_residual(), "rf": af_rf(), "promo": af_promotions(),
            "sel": af_selection()}


def agent_free_checks(findings: str, corrections: str) -> None:
    print("\n== FINDINGS 19: post-review analyses without agents (2026-10-11) ==")
    T = af_tables()
    s19 = findings[findings.index("## 19. Post-review analyses without agents"):]
    sub = {k: s19[s19.index(h):] for k, h in (("19.1", "### 19.1"), ("19.2", "### 19.2"),
                                               ("19.3", "### 19.3"), ("19.4", "### 19.4"),
                                               ("19.5", "### 19.5"))}
    for k, nxt in (("19.1", "### 19.2"), ("19.2", "### 19.3"), ("19.3", "### 19.4"),
                   ("19.4", "### 19.5")):
        sub[k] = sub[k][:sub[k].index(nxt)]
    flat = {k: " ".join(v.split()) for k, v in sub.items()}
    g0 = T["g0"]
    for label, needle in g0["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["19.1"])
    check_true("19.1 quotes the 95% pair-interval widths",
               f"{g0['width_lo']:.2f} to {g0['width_hi']:.2f} wide" in flat["19.1"])
    for label, needle in T["rfc"]["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in flat["19.2"] or needle in sub["19.2"])
    for label, needle in T["rf"]["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["19.3"])
    for key in ("tqd", "h4"):
        for label, needle in T["promo"][key]["rows"]:
            check_true(f"FINDINGS {label} quoted", needle in sub["19.4"])
    tq = T["promo"]["tqd"]
    check_true("19.4 quotes T-qd's short pools and worst pool",
               f"{tq['n_below']} of 60 pools" in flat["19.4"] and f"{tq['worst']:.4f}" in flat["19.4"])
    h4 = T["promo"]["h4"]["doc"]["gates"]
    check_true("19.4 quotes the hidden-4 leak and engagement counts",
               f"clean in {h4['reward_leakage']['seeds_clean_survival']}/10 survival seeds" in flat["19.4"]
               and f"{h4['engagement']['engaged_by_drift']['0.45']['engaged']}/10 at drift 0.45" in flat["19.4"])
    for label, needle in T["sel"]["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["19.5"])
    # The dated notes elsewhere in FINDINGS (2026-10-11) quote numbers from the same artifacts.
    r = g0["recomputed"]
    notes = [n for n in re.findall(r"\*\(Note 2026-10-11:.*?\)\*", findings, flags=re.S)]
    flatn = [" ".join(n.split()) for n in notes]
    check_true("FINDINGS carries the 2026-10-11 dated notes (at least eight)", len(notes) >= 8)

    _h4d = T["promo"]["h4"]["doc"]
    _h4g = _h4d["gates"]
    _pp = {(p["a"], p["b"]): p for p in _af_load("agent_free", "residual_field_correlation.json")["pairs"]}

    def in_note(needle: str) -> bool:
        return any(needle in n for n in flatn)

    for label, needle in [
        ("gate-0 note: typical pair-interval width",
         f"{g0['width_lo']:.2f} to {g0['width_hi']:.2f} wide"),
        ("gate-0 note: hidden 7 seed 1 share above the band",
         f"{r['mlp_h7_s1']['share_draws_above_band']:.3f} of its pair-bootstrap draws above 0.95"),
        ("gate-0 note: G_gn share below the band",
         f"{r['gn_0.01']['share_draws_below_band']:.3f} of its draws below 0.85"),
        ("gate-0 note: G_gn under the explicit split",
         f"{r['gn_0.01']['explicit']:.3f} under the explicit split"),
        ("gate-0 note: hidden 8 seed 0 pair interval",
         f"0.928, pair 95% {_af_iv(r['mlp_h8_s0']['pair_cluster_ci95'])}"),
        ("10.3 note: closest forest values",
         f"{T['rf']['closest'][0]} and {T['rf']['closest'][1]}"),
        ("10.3 note: explicit untrained forest", f"{T['rf']['untrained_explicit']}"),
        ("10.5 note: h8/h7 correlation",
         f"{T['rfc']['h87']['r_pooled']:+.3f} {_af_iv(T['rfc']['h87']['ci95_pooled'], True)}"),
        ("10.5 note: rff share of variance", f"{T['rfc']['h8r']['r_pooled'] ** 2 * 100:.0f}%"),
        ("17.9 / 14.5.1 note: T-qd artifact", "artifacts/texture/T_qd_l3_h8_wm.json"),
        ("10.5 note: hidden-4 artifact", "artifacts/expB2/failed_gates_l3_h4_traces.json"),
        ("13 note: selection-intensity artifact", "artifacts/expC/selection_intensity.json"),
        ("10.5 note: hidden 7 seed 0 pair interval",
         f"hidden 7 reads 0.922 with a pair 95% interval of {_af_iv(r['mlp_h7_s0']['pair_cluster_ci95'])}"),
        ("10.5 note: hidden 4 on world P", f"0.932 {_af_iv(r['mlp_h4_s0']['pair_cluster_ci95'])}"),
        ("10.5 note: hidden-4 gates",
         f"untrained floor {_h4d['pooled_target']['0.45']['untrained']['mean']:.3f}, pooled reward "
         f"leak clean in {_h4g['reward_leakage']['seeds_clean_survival']}/10 survival seeds, "
         f"engagement {_h4g['engagement']['engaged_by_drift']['0.45']['engaged']}/10 at drift 0.45"),
        ("10.5 scope note: MLP maps at other G seeds",
         f"correlate {_pp[('h8_s0', 'h10_s1')]['r_pooled']:+.2f} and "
         f"{_pp[('h8_s0', 'h8_s2')]['r_pooled']:+.2f} with it"),
        ("10.3 note: both forest conventions",
         f"reads {T['rf']['closest'][0]} with random_state 0 and {T['rf']['closest'][1]} with "
         "random_state set to the seed"),
        ("10.7 note: G_rff interval, share below, explicit",
         f"Its pair 95% interval is {_af_iv(r['rff_D80']['pair_cluster_ci95'])}, "
         f"{r['rff_D80']['share_draws_below_band']:.3f} of its pair-bootstrap draws fall below 0.85, "
         f"and it reads {r['rff_D80']['explicit']:.3f} on the explicit partition"),
        ("10.9 note: hidden 7 seed 1 interval",
         f"0.957 has a pair 95% interval of {_af_iv(r['mlp_h7_s1']['pair_cluster_ci95'])}"),
        ("10.9 note: hidden 9 share above",
         f"against {r['mlp_h9_s1']['share_draws_above_band']:.3f} for hidden 9"),
        ("10.9 note: hidden 10 seed 1",
         f"0.893 {_af_iv(r['mlp_h10_s1']['pair_cluster_ci95'])}, with "
         f"{r['mlp_h10_s1']['share_draws_below_band']:.3f} of its draws below 0.85"),
        ("10.9 note: hidden 8 seed 2", f"0.939 {_af_iv(r['mlp_h8_s2']['pair_cluster_ci95'])}"),
        ("14.5 note: G_gn pooled AUROC and interval",
         f"its pooled out-of-fold AUROC is {r['gn_0.01']['pooled_oof_auroc']:.3f}, its pair 95% "
         f"interval {_af_iv(r['gn_0.01']['pair_cluster_ci95'])}"),
        ("17.9 note: T-qd t interval",
         f"the t-based interval {_af_iv(T['promo']['tqd']['doc']['pooled_target']['0.45']['survival']['t90'])} included"),
    ]:
        check_true(f"FINDINGS dated note quotes {label}: {needle}", in_note(needle))
    check_true("FINDINGS 17.9 addendum no longer says T-qd 'is not promoted to `artifacts/`' "
               "without its dated correction",
               "is not promoted to `artifacts/`" not in findings
               or in_note("now promoted, labeled off-design"))
    for needle in ("the random-forest floor of FINDINGS 10.3",
                   "the shared training set"):
        check_true(f"CORRECTIONS carries the 2026-10-11 entry on {needle}",
                   "## 2026-10-11" in corrections and needle in corrections)


# ---- FINDINGS 20: corrected-trainer reruns and post-review readouts (2026-10-11) ----------
# The rerun integrity counts are recounted from the committed cells. Every table row of
# section 20 is rebuilt from per-seed values (the reruns' expB2_results and behavior audits,
# the historical artifacts beside them, the readouts' per_seed blocks, the L1 curve's cells) and
# looked up in the text. Each summary the readout runners wrote is checked against the
# recomputation, and each pre-stated label is derived again from its interval.

S20_RUNS = {"C2-rerun": ("c2-rerun", "aff3863123e3e3ee"),
            "H7-corrected": ("h7-corrected", "ce2a6fe1ba30cae7"),
            "H10GS1-corrected": ("h10gs1-corrected", "03346353ac775154"),
            "H8GS2-corrected": ("h8gs2-corrected", "2823452491c11037")}
S20_ARMS = ("survival", "predictor", "untrained")
S20_SEEDS = [str(i) for i in range(10)]


def _s20_flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _s20_flat(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _s20_flat(v, f"{p}[{i}]")
    else:
        yield p, o


def _s20_same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _s20_cells(run_dir: str, pattern: str = "cell_d*_s*.json") -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ARTROOT, *run_dir.split("/"), "cells", pattern))):
        with open(p, encoding="utf-8") as fh:
            out[os.path.basename(p)] = json.load(fh)
    return out


def _s20_compare(run_dir: str, ref_dir: str, pattern: str) -> dict:
    a, b = _s20_cells(run_dir, pattern), _s20_cells(ref_dir, pattern)
    shared = only_run = only_ref = 0
    differ: list[str] = []
    for name, cell in a.items():
        fa, fb = dict(_s20_flat(cell["cell"])), dict(_s20_flat(b[name]["cell"]))
        keys = fa.keys() & fb.keys()
        shared += len(keys)
        only_run += len(fa.keys() - fb.keys())
        only_ref += len(fb.keys() - fa.keys())
        differ += [k for k in keys if not _s20_same(fa[k], fb[k])]
    return {"n": len(a), "shared": shared, "differ": differ, "only_run": only_run, "only_ref": only_ref}


def _s20_label(t) -> str:
    return "ABOVE" if t[0] > 0 else "BELOW" if t[1] < 0 else "INCLUDES"


def _s20_ms(v) -> tuple:
    return float(np.mean(v)), tuple(float(x) for x in t_ci90(v))


def _s20_cell(v, sign: bool = False) -> str:
    m, t = _s20_ms(v)
    return (f"{m:+.3f} " if sign else f"{m:.3f} ") + _af_iv(t, sign)


def _s20_agree(label: str, mine: tuple, block: dict) -> None:
    check_true(f"{label}: runner summary equals the recomputation",
               abs(mine[0] - block["mean"]) < 1e-12 and np.allclose(mine[1], block["t90"], rtol=0, atol=1e-12))


def s20_integrity() -> dict:
    """20.1: cells, bootstrap and hash of every rerun; C2-rerun against C2; the drift-0 cells
    against C1; H10GS1-corrected against its historical CPU run; the rebuilt dumps."""
    for name, (d, fp) in S20_RUNS.items():
        cells = _s20_cells(f"corrected_runs/{d}")
        check_true(f"20.1 {name}: 20 cells, each with the successor bootstrap and hash {fp}",
                   len(cells) == 20 and all(c["cell"].get("gae_bootstrap") == "successor"
                                            and c["fingerprint"] == fp for c in cells.values()))
    c2 = _s20_compare("corrected_runs/c2-rerun", "corrected_runs/corrected_l3_h8_nowm", "cell_d*_s*.json")
    check_true("20.1 C2-rerun: every shared value equals the committed C2 cell",
               c2["n"] == 20 and not c2["differ"] and c2["only_ref"] == 0)
    d0 = {}
    for name in ("H7-corrected", "H10GS1-corrected", "H8GS2-corrected"):
        r = _s20_compare(f"corrected_runs/{S20_RUNS[name][0]}", "corrected_runs/corrected_l3_h8_wm",
                         "cell_d0.00_s*.json")
        check_true(f"20.1 {name}: the drift-0 cells equal C1's except one value per cell, the "
                   "cross-evaluation return at drift 0.45",
                   r["n"] == 10 and r["only_ref"] == 0 and sorted(set(r["differ"])) == ["/xeval/0.45"]
                   and len(r["differ"]) == 10)
        d0[name] = r["shared"]
    check_true("20.1 the three instance runs share as many drift-0 values with C1",
               len(set(d0.values())) == 1)
    h10 = _s20_cells("corrected_runs/h10gs1-corrected")
    hist = _s20_cells("reviewer_gaps_runs/l3_h10_gseed1")
    eq = {a: sum(h10[k]["cell"]["agents"][a]["pool"]["target"] == hist[k]["cell"]["agents"][a]["pool"]["target"]
                 for k in h10) for a in S20_ARMS}
    check_true("20.1 H10GS1-corrected: predictor and untrained targets bit-equal to L3-H10-GS1-CPU in "
               "20 of 20 cells, survival in none",
               eq["predictor"] == 20 and eq["untrained"] == 20 and eq["survival"] == 0)
    for name in ("H7-corrected", "H10GS1-corrected", "H8GS2-corrected"):
        rb = _af_load("corrected_runs", S20_RUNS[name][0], "state_dump_rebuild.json")
        check_true(f"20.1 {name}: the rebuilt state dumps reproduce 60 of 60 pools",
                   rb["n"] == 60 and rb["reproduced"] == 60 and rb["all_reproduced"] is True
                   and all(c["target"] == c["recorded_target"] for c in rb["cells"]))
    return {"c2": c2, "d0_shared": next(iter(d0.values()))}


def _s20_run_row(label: str, trainer: str, res: dict, audit: dict, gate: tuple) -> tuple:
    s, p, u = (res["0.45"][a]["pool_target"] for a in S20_ARMS)
    rt = seed_vals(audit, "0.45", "survival", "resid_trace")
    sp = np.asarray(s) - np.asarray(p)
    su = np.asarray(s) - np.asarray(u)
    oracle, floor, ok = gate
    n65 = int(sum(x >= BAR for x in s))
    row = (f"| {label} | {trainer} | {_s20_cell(s)} | {n65}/10 | {np.mean(p):.3f} | {np.mean(u):.3f} | "
           f"{_s20_cell(sp, True)} | {_s20_cell(rt)} | {oracle:.3f}, {floor:.3f}{'' if ok else ' (fails)'} |")
    return row, {"s": _s20_ms(s), "p": float(np.mean(p)), "u": float(np.mean(u)), "sp": _s20_ms(sp),
                 "su": _s20_ms(su), "rt": _s20_ms(rt), "n65": n65, "oracle": oracle, "floor": floor,
                 "gate0": ok}


def _s20_gate_cpu(d: str) -> tuple:
    rows = _af_load("corrected_runs", d, "gate0_cpu.json")["rows"]
    g = rows[0]
    ok = bool(g["in_band"] and g["mech_leak_pass"] and abs(g["floor"] - 0.5) < 0.1)
    check_true(f"20.2 {d}: one gate-0 row, its pass flag agrees with band, leakage and floor",
               len(rows) == 1 and ok == bool(g["passes_gate0"]))
    return g["oracle_auroc"], g["floor"], ok


def s20_instances() -> dict:
    """20.2: each corrected instance run beside its historical row, from per-seed values."""
    fr = _af_load("fold_rescore", "l3_h7_traces.json")

    def fr_vals(arm, met):
        cells = sorted([c for c in fr["cells"] if c["agent"] == arm and float(c["drift"]) == 0.45],
                       key=lambda c: int(c["seed"]))
        return [c["explicit"][met] for c in cells]

    hist7 = {"0.45": {a: {"pool_target": fr_vals(a, "target")} for a in S20_ARMS}}
    aud7 = {"cells": [{"drift": "0.45", "agent": "survival", "seed": i, "resid_trace": v}
                      for i, v in enumerate(fr_vals("survival", "resid_trace"))]}
    g7 = next(r for r in _af_load("fold_rescore", "gate0_h8h7_explicit.json")["rows"] if r["hidden"] == 7)
    g10 = next(r for r in _af_load("reviewer_gaps_runs", "l3_gate0_seed1", "calibration.json")["rows"]
               if r["hidden"] == 10)
    g82 = _af_load("reviewer_gaps_runs", "l3_gate0_seed2_gpu", "calibration.json")["rows"][0]

    def corr(d):
        return (_af_load("corrected_runs", d, "expB2_results.json"),
                _af_load("corrected_runs", d, "behavior_audit.json"), _s20_gate_cpu(d))

    def hist(d):
        return (_af_load("reviewer_gaps_runs", d, "expB2_results.json"),
                _af_load("reviewer_gaps_runs", d, "behavior_audit.json"))

    spec = [
        ("h7_hist", "hidden 7, `L3-H7-N10` (10.5)", "historical, GPU, explicit re-score",
         (hist7, aud7, (g7["oracle_auroc"], g7["floor"], bool(g7["passes_gate0"])))),
        ("h7", "`H7-corrected`", "corrected, CPU", corr("h7-corrected")),
        ("h10_hist", "hidden 10, G seed 1, `L3-H10-GS1-CPU` (10.9)", "historical, CPU",
         (*hist("l3_h10_gseed1"), (g10["oracle_auroc"], g10["floor"], bool(g10["passes_gate0"])))),
        ("h10", "`H10GS1-corrected`", "corrected, CPU", corr("h10gs1-corrected")),
        ("h8s2_hist", "hidden 8, G seed 2, `L3-H8-GS2-GPU` (10.9)", "historical, GPU",
         (*hist("l3_h8_gseed2_gpu"), (g82["oracle_auroc"], g82["floor"], bool(g82["passes_gate0"])))),
        ("h8s2", "`H8GS2-corrected`", "corrected, CPU", corr("h8gs2-corrected")),
    ]
    rows, nums = [], {}
    for key, label, trainer, (res, aud, gate) in spec:
        row, n = _s20_run_row(label, trainer, res, aud, gate)
        rows.append((f"20.2 row {key}", row))
        nums[key] = n
    for k in ("h7", "h10", "h8s2"):
        check_true(f"20.2 {k}: the CPU gate-0 oracle equals the recorded oracle to the bit",
                   nums[k]["oracle"] == nums[f"{k}_hist"]["oracle"])
    prom = {k: _af_load("expB2", f"corrected_{n}.json") for k, n in
            (("h7", "H7-corrected"), ("h10", "H10GS1-corrected"), ("h8s2", "H8GS2-corrected"),
             ("c2", "C2-rerun"))}
    for k in ("h7", "h10", "h8s2"):
        check(f"20.2 {k}: promoted survival mean equals the recomputation",
              prom[k]["decision"]["survival"], nums[k]["s"][0], 1e-12)
        check(f"20.2 {k}: drift-0 survival target (C1's L0 reading)",
              _seed_mean(prom[k]["arms"]["0.0"]["survival"]["pool_target"]), 0.559)
        check_true(f"20.2 {k}: the L0 gate is open",
                   l0_gate_reproduces(f"20.2 {k}", prom[k]).equivalent is False)
    # 2026-10-16: the remaining promoted expB2 runs whose stored L0 gate the checks above do not
    # read (C1 is recomputed in section 18.2), so the claim covers every stored L0 gate.
    for _n in ("C2-rerun", "l3_h8_nowm", "L1-corrected"):
        l0_gate_reproduces(f"expB2 corrected_{_n}", _af_load("expB2", f"corrected_{_n}.json"))
    g8p = prom["h8s2"]["gates"]
    check_true("20.2 H7-corrected: the open gates are L0 and reward leak",
               prom["h7"]["decision"]["gates_open"] == ["l0", "reward_leak"])
    check_true("20.2 H10GS1-corrected: the only open gate is L0", prom["h10"]["decision"]["gates_open"] == ["l0"])
    check_true("20.2 H8GS2-corrected: the open gates are L0 and speed, one pool under 0.75 "
               "(untrained, drift 0.45, seed index 0)",
               prom["h8s2"]["decision"]["gates_open"] == ["l0", "speed"] and g8p["speed_n_below"] == 1
               and g8p["speed_min_at"] == {"drift": "0.45", "arm": "untrained", "seed_index": 0})
    res7 = _af_load("corrected_runs", "h7-corrected", "expB2_results.json")
    unclean = sorted((d, a, i, round(res7[d][a]["pool_reward_leak"][i], 3)) for d in res7 for a in S20_ARMS
                     for i, f in enumerate(res7[d][a]["pool_leak_clean"]) if not f)
    check_true("20.2 H7-corrected: the two unclean reward-leak pools are drift 0.45, seed 8, predictor "
               "0.629 and survival 0.600",
               unclean == [("0.45", "predictor", 8, 0.629), ("0.45", "survival", 8, 0.6)])
    c2_arms = _af_load("expB2", "corrected_l3_h8_nowm.json")["arms"]
    check_true("20.2 C2-rerun: the promoted per-seed pooled targets equal C2's in every drift and arm",
               all(prom["c2"]["arms"][d][a]["pool_target"]["per_seed"] == c2_arms[d][a]["pool_target"]["per_seed"]
                   for d in c2_arms for a in S20_ARMS))
    s_new = _af_load("corrected_runs", "h10gs1-corrected", "expB2_results.json")["0.45"]["survival"]["pool_target"]
    s_old = _af_load("reviewer_gaps_runs", "l3_h10_gseed1", "expB2_results.json")["0.45"]["survival"]["pool_target"]
    seeds7 = _af_load("corrected_runs", "h7-corrected", "gate0_cpu.json")["rows"][0]["floor_per_seed"]
    return {"rows": rows, "nums": nums, "speed_min": g8p["speed_min"],
            "h10_speed_min": prom["h10"]["gates"]["speed_min"],
            "corr_h10": _s20_ms(np.asarray(s_new) - np.asarray(s_old)), "h7_floor_seeds": seeds7}


def _s20_r210(ps: dict, sm: dict, tag: str) -> dict:
    cls = ("linear", "mlp", "rf")
    out = {}
    for dr in ("0.45", "0.00"):
        v = {a: {c: [ps[s]["cells"][dr][a]["classes"][c]["target"] for s in S20_SEEDS] for c in cls}
             for a in S20_ARMS}
        shm = [float(np.mean([ps[s]["cells"][dr][a]["classes"][c]["shuffled"] for s in S20_SEEDS]))
               for a in S20_ARMS for c in cls]
        rows = []
        for a in S20_ARMS:
            for c in cls:
                _s20_agree(f"{tag} r210 {dr} {a} {c}", _s20_ms(v[a][c]), sm[dr]["arms"][a][c]["target"])
            diffs = []
            for c in ("mlp", "rf"):
                m, t = _s20_ms(np.asarray(v[a][c]) - np.asarray(v[a]["linear"]))
                blk = sm[dr]["class_minus_linear"][a][c]
                _s20_agree(f"{tag} r210 {dr} {a} {c} minus linear", (m, t), blk)
                check_true(f"{tag} r210 {dr} {a} {c}: label {_s20_label(t)} derived again",
                           _s20_label(t) == blk["reading"])
                diffs.append(f"{m:+.3f} {_af_iv(t, True)} {_s20_label(t)}")
            rows.append((f"{tag} r210 drift {dr} row {a}",
                         f"| {a} | {_s20_cell(v[a]['linear'])} | {_s20_cell(v[a]['mlp'])} | "
                         f"{_s20_cell(v[a]['rf'])} | {diffs[0]} | {diffs[1]} |"))
        margins = []
        for b in ("predictor", "untrained"):
            cells = []
            for c in cls:
                dv = np.asarray(v["survival"][c]) - np.asarray(v[b][c])
                _s20_agree(f"{tag} r210 {dr} survival minus {b} {c}", _s20_ms(dv),
                           sm[dr]["margins"][f"survival_minus_{b}"][c])
                cells.append(_s20_cell(dv, True))
            margins.append((f"{tag} r210 drift {dr} margin {b}",
                            f"| survival minus {b} | " + " | ".join(cells) + " |"))
        out[dr] = {"rows": rows, "margins": margins, "v": v, "sh_range": (min(shm), max(shm))}
    return out


def _s20_r35(ps: dict, sm: dict, tag: str) -> dict:
    norms = ("own", "survival_norm", "refit", "refit_infold")
    v = {a: {k: [ps[s]["arms"][a][k] for s in S20_SEEDS] for k in norms} for a in S20_ARMS}
    rows = []
    for a in S20_ARMS:
        for k in norms:
            _s20_agree(f"{tag} r35 {a} {k}", _s20_ms(v[a][k]), sm["arms"][a][k])
        rows.append((f"{tag} r35 row {a}", f"| {a} | " + " | ".join(_s20_cell(v[a][k]) for k in norms) + " |"))
    margins, labels = [], {}
    for b in ("untrained", "predictor"):
        ms = {}
        for k in norms:
            ref = v["survival"]["own"] if k == "survival_norm" else v["survival"][k]
            ms[k] = _s20_ms(np.asarray(ref) - np.asarray(v[b][k]))
            _s20_agree(f"{tag} r35 survival minus {b} {k}", ms[k], sm["margins"][f"survival_minus_{b}"][k])
        cells = [f"{ms[k][0]:+.3f} {_af_iv(ms[k][1], True)}"
                 + ("" if k == "own" else f" ({ms[k][0] / ms['own'][0]:.2f})") for k in norms]
        margins.append((f"{tag} r35 margin {b}", f"| {b} | " + " | ".join(cells) + " |"))
        # The refit entry is (c) unless (c) and (d) carry different labels, then (d) governs.
        refit = "refit" if _s20_label(ms["refit"][1]) == _s20_label(ms["refit_infold"][1]) else "refit_infold"
        pair = ("survival_norm", refit)
        if all(_s20_label(ms[k][1]) == "ABOVE" and ms[k][0] / ms["own"][0] >= 0.5 for k in pair):
            lab = "TRUNK"
        elif all(_s20_label(ms[k][1]) in ("INCLUDES", "BELOW") for k in pair):
            lab = "NORMALIZER"
        else:
            lab = "SPLIT"
        check_true(f"{tag} r35 {b}: frozen label {lab} derived again", lab == sm["reading"][b])
        labels[b] = lab
    return {"rows": rows, "margins": margins, "labels": labels}


S20_WALKS = {"survival_own": "survival agent's own", "predictor_own": "predictor's own",
             "untrained_own": "untrained agent's own", "scripted": "scripted"}


def _s20_r206(ps: dict, sm: dict, recorded: list, tag: str) -> dict:
    w = {k: {m: [ps[s]["walks"][k][m] for s in S20_SEEDS] for m in ("sum_mean", "speed_mean", "acc_mean", "n_eps")}
         for k in S20_WALKS}
    check_true(f"{tag} r206: every walk kept 110 of 110 episodes in every seed",
               all(x == 110 for k in S20_WALKS for x in w[k]["n_eps"]))
    check_true(f"{tag} r206: the scripted walk's summed deviation is the same in every seed",
               float(np.std(w["scripted"]["sum_mean"])) < 1e-12)
    rows, above = [], []
    for k, name in S20_WALKS.items():
        _s20_agree(f"{tag} r206 {k}", _s20_ms(w[k]["sum_mean"]), sm["walks"][k]["sum_mean"])
        if k == "survival_own":
            minus = ""
        else:
            m, t = _s20_ms(np.asarray(w["survival_own"]["sum_mean"]) - np.asarray(w[k]["sum_mean"]))
            _s20_agree(f"{tag} r206 survival minus {k}", (m, t), sm["survival_minus"][k])
            above.append(t[0] > 0)
            minus = f"{m:+.3f} {_af_iv(t, True)}"
        dev = (f"{np.mean(w[k]['sum_mean']):.3f} (the same in every seed)" if k == "scripted"
               else _s20_cell(w[k]["sum_mean"]))
        rows.append((f"{tag} r206 row {k}", f"| {name} | {dev} | {np.mean(w[k]['speed_mean']):.3f} | "
                     f"{np.mean(w[k]['acc_mean']):.3f} | {minus} |"))
    lab = "HIGHER" if all(above) else "NOT HIGHER" if not any(above) else "MIXED"
    check_true(f"{tag} r206: frozen label {lab} derived again", sm["reading"] == lab)
    r = float(np.corrcoef(w["survival_own"]["sum_mean"], recorded)[0, 1])
    check(f"{tag} r206: correlation with the recorded survival target", r,
          sm["corr_with_recorded_target"]["pearson_r"], 1e-9)
    return {"rows": rows, "r": r, "label": lab}


def _s20_r33(ps: dict, sm: dict, tag: str) -> dict:
    rows = []
    for ad, adlab in ((0.45, "drift 0.45 (forward)"), (0.0, "drift 0 (reverse)")):
        key = f"d{ad:.2f}"
        for base, blab in (("registered", "registered, 800000"), ("fresh", "fresh, 2200000")):
            v = {a: [next(c["target"] for c in ps[s]["cells"] if c["agent_drift"] == ad and c["arm"] == a
                          and c["base"] == base) for s in S20_SEEDS] for a in S20_ARMS}
            for a in S20_ARMS:
                _s20_agree(f"{tag} r33 {key} {base} {a}", _s20_ms(v[a]), sm[f"{key}/{base}"]["arms"][a])
            sp = np.asarray(v["survival"]) - np.asarray(v["predictor"])
            su = np.asarray(v["survival"]) - np.asarray(v["untrained"])
            _s20_agree(f"{tag} r33 {key} {base} survival minus predictor", _s20_ms(sp),
                       sm[f"{key}/{base}"]["survival_minus_predictor"])
            _s20_agree(f"{tag} r33 {key} {base} survival minus untrained", _s20_ms(su),
                       sm[f"{key}/{base}"]["survival_minus_untrained"])
            rows.append((f"{tag} r33 row {key} {base}",
                         f"| {adlab} | {blab} | " + " | ".join(_s20_cell(v[a]) for a in S20_ARMS)
                         + f" | {_s20_cell(sp, True)} | {_s20_cell(su, True)} |"))
    check_int(f"{tag} r33: pairs dropped over all 120 balanced readouts",
              sum(c["dropped_pairs"] for s in S20_SEEDS for c in ps[s]["cells"]), 0)
    return {"rows": rows}


S20_STREAMS = {"survival_own": "survival agent's own walk", "scripted": "scripted walk"}


def _s20_r31(ps: dict, sm: dict, tag: str) -> dict:
    widths = ("1x", "2x", "4x", "8x")
    rows, contrasts, losses, conv_losses = [], [], [], []
    for st, name in S20_STREAMS.items():
        state = [ps[s][st]["state"] for s in S20_SEEDS]
        dec = {k: [ps[s][st]["decoders"]["gru"][k]["auroc"] for s in S20_SEEDS] for k in widths}
        dec["conv"] = [ps[s][st]["decoders"]["conv"]["auroc"] for s in S20_SEEDS]
        losses += [x for s in S20_SEEDS for k in widths for x in ps[s][st]["decoders"]["gru"][k]["final_train_loss"]]
        conv_losses += [x for s in S20_SEEDS for x in ps[s][st]["decoders"]["conv"]["final_train_loss"]]
        _s20_agree(f"{tag} r31 {st} state", _s20_ms(state), sm[st]["state"])
        for k in dec:
            _s20_agree(f"{tag} r31 {st} {k}", _s20_ms(dec[k]), sm[st]["decoders"][k])
        rows.append((f"{tag} r31 row {st}", f"| {name} | {_s20_cell(state)} | "
                     + " | ".join(_s20_cell(dec[k]) for k in (*widths, "conv")) + " |"))
        cap, cells = False, []
        for k in ("2x", "4x", "8x"):
            m, t = _s20_ms(np.asarray(dec[k]) - np.asarray(dec["1x"]))
            cap = cap or (t[0] > 0 and m >= 0.05)
            cells.append(f"{m:+.3f} {_af_iv(t, True)}")
        best = max(dec, key=lambda k: float(np.mean(dec[k])))
        m, t = _s20_ms(np.asarray(state) - np.asarray(dec[best]))
        _s20_agree(f"{tag} r31 {st} state minus best", (m, t), sm[st]["state_minus_best"])
        slab = ("EQUIVALENT" if t[0] > -0.05 and t[1] < 0.05 else "DECODER ABOVE" if t[1] < 0
                else "STATE ABOVE" if t[0] > 0 else "UNRESOLVED")
        clab = "CAPACITY-LIMITED" if cap else "NOT CAPACITY-LIMITED"
        check_true(f"{tag} r31 {st}: labels {clab} and {slab} derived again, best decoder {best}",
                   clab == sm[st]["capacity_reading"] and slab == sm[st]["state_reading"]
                   and best == sm[st]["best_decoder"])
        contrasts.append((f"{tag} r31 contrast row {st}", f"| {name} | " + " | ".join(cells)
                          + f" | {m:+.3f} {_af_iv(t, True)} ({best}) | {clab}; {slab} |"))
    check_true(f"{tag} r31: every GRU's final training loss is below 0.001 in every fold", max(losses) < 0.001)
    return {"rows": rows, "contrasts": contrasts, "conv_max": max(conv_losses),
            "conv_n_ge": int(sum(x >= 0.001 for x in conv_losses))}


def s20_readout(fname: str, tag: str) -> dict:
    """One post-review readout output, recomputed from its per_seed blocks: integrity, then the
    rows of R210, R35, R206, R33 and R31 in the table formats of 20.4 to 20.8."""
    doc = _af_load("post_review", fname)
    ps, sm = doc["per_seed"], doc["summary"]
    check_true(f"{tag} readout complete: five readouts on ten seeds, no errors, the C1 stack, explicit-v1",
               doc["complete"] and doc["complete_all_readouts"] and not doc["errors"]
               and all(sorted(ps[r], key=int) == S20_SEEDS for r in ("r210", "r35", "r206", "r33", "r31"))
               and doc["stack"]["torch"] == "2.14.1+cpu" and doc["stack"]["numpy"] == "2.4.6"
               and doc["stack"]["sklearn"] == "1.9.1" and doc["fold_version"] == "explicit-v1"
               and not any(doc["integrity"]["unreproduced"].values()))
    n = {"cells": 0, "rep": 0, "shuf": 0, "perm": 0, "hit": 0}
    for s in S20_SEEDS:
        for arms in ps["r210"][s]["cells"].values():
            for c in arms.values():
                n["cells"] += 1
                n["rep"] += bool(c["reproduced"] and c["target"] == c["recorded_target"])
                n["shuf"] += bool(c["shuffled_reproduced"] and c["shuffled"] == c["recorded_shuffled"])
                n["perm"] += bool(c["perm_check"])
                n["hit"] += c["classes"]["mlp"]["hit_max_iter"]
    check_true(f"{tag} r210: all 60 cells reproduce the recorded target and shuffled control as floats "
               "and pass the permutation check", n["cells"] == 60 and n["rep"] == n["shuf"] == n["perm"] == 60)
    check_int(f"{tag} r210: MLP folds that stopped at the iteration cap", n["hit"], 0)
    br = doc.get("balanced_reference_check")
    if br is not None:
        check_true(f"{tag} r33 continuity entry: 30 of 30 cells bit-equal to the l0 audit",
                   br["n_checked"] == 30 and br["n_bit_equal"] == 30 and br["max_abs_dev"] == 0.0
                   and br["reference"] == "artifacts/l0_audit/corrected_l3_h8_wm.json")
    recorded = [ps["r210"][s]["cells"]["0.45"]["survival"]["recorded_target"] for s in S20_SEEDS]
    return {"doc": doc, "n210": n, "recorded": recorded,
            "r210": _s20_r210(ps["r210"], sm["r210"], tag), "r35": _s20_r35(ps["r35"], sm["r35"], tag),
            "r206": _s20_r206(ps["r206"], sm["r206"], recorded, tag), "r33": _s20_r33(ps["r33"], sm["r33"], tag),
            "r31": _s20_r31(ps["r31"], sm["r31"], tag),
            "own_state": _s20_ms([ps["r31"][s]["survival_own"]["state"] for s in S20_SEEDS])}


def s20_contrast() -> dict:
    """20.12: the seed-paired C1 minus C2 contrast of the balanced survival target, recomputed from
    the r33 per-seed blocks of the two readouts and compared with the committed contrast output."""
    A, B = _af_load("post_review", "C1-rerun.json"), _af_load("post_review", "C2-rerun.json")
    doc = _af_load("post_review", "c1_minus_c2_balanced.json")
    check_true("20.12 contrast: the two readouts share seed bases and settings, and the output names them",
               A["seed_bases"] == B["seed_bases"] and A["settings"] == B["settings"]
               and doc["a_path"] == "artifacts/post_review/C1-rerun.json"
               and doc["b_path"] == "artifacts/post_review/C2-rerun.json" and doc["primary"] == "d0.45/fresh")

    def vals(D, ad, base):
        return [next(c["target"] for c in D["per_seed"]["r33"][s]["cells"] if c["agent_drift"] == ad
                     and c["arm"] == "survival" and c["base"] == base) for s in S20_SEEDS]

    rows, out = [], {}
    for ad, adlab in ((0.45, "drift 0.45 (forward)"), (0.0, "drift 0 (reverse)")):
        for base, blab in (("registered", "registered, 800000"), ("fresh", "fresh, 2200000")):
            a, b = vals(A, ad, base), vals(B, ad, base)
            m, t = _s20_ms(np.asarray(a) - np.asarray(b))
            lab = ("HOLDS" if t[0] > 0.05 else "DIRECTION ONLY" if t[0] > 0 else "REVERSED" if t[1] < 0
                   else "NOT RESOLVED")
            key = f"d{ad:.2f}/{base}"
            c = doc["contrasts"][key]
            check_true(f"20.12 contrast {key}: mean, interval and label recompute ({lab})",
                       abs(c["mean"] - m) < 1e-12 and np.allclose(c["t90"], t, rtol=0, atol=1e-12)
                       and c["reading"] == lab)
            out[key] = (m, t, lab)
            rows.append((f"20.12 contrast row {key}",
                         f"| {adlab} | {blab} | {np.mean(a):.3f} | {np.mean(b):.3f} | "
                         f"{m:+.3f} {_af_iv(t, True)} | {lab} |"))
    return {"rows": rows, "c": out}


def s20_l1() -> dict:
    """20.9, recomputed from the 150 cells of the L1 capacity curve."""
    doc = _af_load("post_review", "l1_capacity_curve.json")
    ig, rc = doc["integrity"], doc["record_check"]
    check_true("20.9 L1 curve complete and not void; integrity gate passed on 30 cells within 0.01",
               doc["complete"] and not doc["void"] and not doc["errors"] and ig["pass"] is True
               and ig["n_checked"] == 30 and ig["worst_abs_dev"] <= 0.01 and len(doc["cells"]) == 150)
    check_int("20.9 L1 integrity: state targets bit-equal to the record", ig["n_bit_equal"], 15)
    check_true("20.9 L1 record check: the survival 1x mean reproduces the recorded 0.590 within 0.01",
               rc["survival_1x_mean_reproduced"] is True
               and abs(rc["arms"]["survival"]["recorded_mean"] - 0.590) < 5e-4
               and abs(rc["arms"]["survival"]["rescored_mean"] - rc["arms"]["survival"]["recorded_mean"]) < 0.01)
    ks = ("1x", "2x", "4x", "8x", "conv")
    rows, contrasts, labels, conv = [], [], {}, {}
    for a in S20_ARMS:
        cell = {(c["seed"], c["width"]): c for c in doc["cells"] if c["arm"] == a}
        dec = {k: [cell[(s, k)]["decoder"]["auroc"] for s in S20_SEEDS] for k in ks}
        state = [cell[(s, "1x")]["target"] for s in S20_SEEDS]
        check_true(f"20.9 L1 {a}: every task of a cell regenerated the same state target",
                   all(cell[(s, k)]["target"] == state[i] for i, s in enumerate(S20_SEEDS) for k in ks))
        blk = doc["summary"][a]
        _s20_agree(f"20.9 L1 {a} state", _s20_ms(state), blk["state"])
        for k in ks:
            _s20_agree(f"20.9 L1 {a} {k}", _s20_ms(dec[k]), blk["decoders"][k])
        cells, avail = [], False
        for k in ("2x", "4x", "8x", "conv"):
            m, t = _s20_ms(np.asarray(dec[k]) - np.asarray(dec["1x"]))
            if k != "conv":
                avail = avail or (m >= 0.05 and t[0] > 0)
            cells.append(f"{m:+.3f} {_af_iv(t, True)}")
        lab = "SEAM AVAILABLE AT WIDTH" if avail else "READER-RELATIVE"
        check_true(f"20.9 L1 {a}: label {lab} derived again", lab == blk["reading"])
        labels[a], conv[a] = lab, _s20_cell(dec["conv"])
        rows.append((f"20.9 L1 row {a}", f"| {a} | {_s20_cell(state)} | "
                     + " | ".join(_s20_cell(dec[k]) for k in ks) + f" | {lab} |"))
        contrasts.append((f"20.9 L1 contrast row {a}", f"| {a} | " + " | ".join(cells) + " |"))
    check_true("20.9 L1: every decoder's final training loss is below 0.001 in every fold",
               max(max(c["decoder"]["final_train_loss"]) for c in doc["cells"]) < 0.001)
    return {"rows": rows, "contrasts": contrasts, "labels": labels, "conv": conv}


def s20_r205() -> dict:
    """20.15: the mean-h / h_T decomposition of the registered probe on the C1 rerun's agents,
    recomputed from the per_seed block; integrity against the recorded targets and against the
    r210 output, and the pre-stated MEAN CARRIES / FINAL CARRIES / BOTH label derived again."""
    doc = _af_load("post_review", "C1-rerun_r205.json")
    ps, sm = doc["per_seed"]["r205"], doc["summary"]["r205"]
    check_true("20.15 r205 output complete on ten seeds, the C1 stack, explicit-v1, no errors",
               doc["complete"] and not doc["errors"] and doc["requested"]["readouts"] == ["r205"]
               and sorted(ps, key=int) == S20_SEEDS and doc["stack"]["torch"] == "2.14.1+cpu"
               and doc["fold_version"] == "explicit-v1" and doc["integrity"]["checked_by"] == "r205"
               and not any(doc["integrity"]["unreproduced"].values()))
    r210 = _af_load("post_review", "C1-rerun.json")["per_seed"]["r210"]
    n = {"cells": 0, "ok": 0, "same_as_r210": 0}
    for s in S20_SEEDS:
        for dr, arms in ps[s]["cells"].items():
            for arm, c in arms.items():
                n["cells"] += 1
                n["ok"] += bool(c["reproduced"] and c["shuffled_reproduced"] and c["perm_check"]
                                and c["target"] == c["recorded_target"]
                                and c["sets"]["concat"]["target"] == c["recorded_target"])
                n["same_as_r210"] += c["recorded_target"] == r210[s]["cells"][dr][arm]["recorded_target"]
    check_true("20.15 r205: in all 60 cells the concatenation equals the recorded pooled target as a float, "
               "the shuffled control reproduces and the permutation check passes",
               n["cells"] == 60 and n["ok"] == 60)
    check_int("20.15 r205: recorded targets equal to r210's", n["same_as_r210"], 60)
    sets = ("mean", "final", "concat")
    out = {"n": n}
    for dr in ("0.45", "0.00"):
        rows, sh = [], []
        for a in S20_ARMS:
            v = {k: [ps[s]["cells"][dr][a]["sets"][k]["target"] for s in S20_SEEDS] for k in sets}
            sh += [float(np.mean([ps[s]["cells"][dr][a]["sets"][k]["shuffled"] for s in S20_SEEDS])) for k in sets]
            blk = sm[dr]["arms"][a]
            cells = []
            for k in sets:
                _s20_agree(f"20.15 r205 {dr} {a} {k}", _s20_ms(v[k]), blk[k]["target"])
                n65 = int(sum(x >= BAR for x in v[k]))
                check_int(f"20.15 r205 {dr} {a} {k}: seeds at or above 0.65", n65, blk[k]["target"]["n_at_or_above_bar"])
                cells.append(f"{_s20_cell(v[k])}, {n65}/10")
            d = {}
            for key, x, y in (("mean_minus_final", "mean", "final"), ("concat_minus_mean", "concat", "mean"),
                              ("concat_minus_final", "concat", "final")):
                d[key] = _s20_ms(np.asarray(v[x]) - np.asarray(v[y]))
                _s20_agree(f"20.15 r205 {dr} {a} {key}", d[key], sm[dr]["differences"][a][key])
            cm, cf = d["concat_minus_mean"][0], d["concat_minus_final"][0]
            lab = ("MEAN CARRIES" if cm <= 0.02 and cf >= 0.05 else "FINAL CARRIES" if cf <= 0.02 and cm >= 0.05
                   else "BOTH")
            check_true(f"20.15 r205 {dr} {a}: label {lab} derived again", lab == sm[dr]["reading"][a])
            rows.append((f"20.15 r205 drift {dr} row {a}",
                         f"| {a} | " + " | ".join(cells) + " | "
                         + " | ".join(f"{d[k][0]:+.3f} {_af_iv(d[k][1], True)}"
                                      for k in ("mean_minus_final", "concat_minus_mean", "concat_minus_final"))
                         + f" | {lab} |"))
            out[(dr, a)] = {"v": v, "d": d, "label": lab}
        out[dr] = {"rows": rows, "sh_range": (min(sh), max(sh))}
    return out


def post_review_checks(findings: str, prereg: str) -> None:
    print("\n== FINDINGS 20: corrected-trainer reruns and post-review readouts (2026-10-11) ==")
    s20 = findings[findings.index("## 20. Corrected-trainer reruns"):]
    heads = [f"### 20.{i} " for i in range(1, 16)]
    sub = {}
    for i, h in enumerate(heads):
        end = s20.index(heads[i + 1]) if i + 1 < len(heads) else len(s20)
        sub[f"20.{i + 1}"] = s20[s20.index(h):end]
    flat = {k: " ".join(v.split()) for k, v in sub.items()}
    ig = s20_integrity()
    c2 = ig["c2"]
    check_true("20.1 quotes the C2-rerun value counts",
               f"{c2['shared']} values" in flat["20.1"] and f"{c2['only_run']} further leaves" in flat["20.1"])
    check_true("20.1 quotes the drift-0 value counts",
               f"share {ig['d0_shared']} values" in flat["20.1"]
               and f"{ig['d0_shared'] - 10} are bit-equal" in flat["20.1"])
    for name, (_, fp) in S20_RUNS.items():
        check_true(f"20.1 names the {name} hash", f"`{fp}`" in flat["20.1"])
    inst = s20_instances()
    for label, needle in inst["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["20.2"])
    check_true("20.2 quotes the short speed pool", f"{inst['speed_min']:.3f}" in flat["20.2"])
    n = inst["nums"]
    for k in ("h7", "h10", "h8s2"):
        su = n[k]["su"]
        check_true(f"20.2 {k}: quotes the survival-minus-untrained margin",
                   f"{su[0]:+.3f} {_af_iv(su[1], True)}" in flat["20.2"])
    check_true("20.2 quotes the hidden 7 lower bound and floor seeds",
               f"{n['h7']['s'][1][0]:.4f}" in flat["20.2"]
               and ", ".join(f"{x:.3f}" for x in inst["h7_floor_seeds"]) in flat["20.2"])
    ce = inst["corr_h10"]
    check_true("20.2 quotes the hidden 10 correction effect, paired against the historical CPU run",
               f"{ce[0]:+.3f} {_af_iv(ce[1], True)}" in flat["20.2"]
               and f"{inst['h10_speed_min']:.3f}" in flat["20.2"])
    T = s20_readout("C1-rerun.json", "C1")
    check_true("20.3 quotes the integrity counts and the fresh-world survival reading",
               f"all {T['n210']['cells']} (drift, seed, arm) cells" in flat["20.3"] and "30 of 30" in flat["20.3"]
               and f"{T['own_state'][0]:.3f}" in flat["20.3"])
    r210 = T["r210"]
    for k, rows in (("20.4", r210["0.45"]["rows"] + r210["0.45"]["margins"] + r210["0.00"]["rows"]),
                    ("20.5", T["r35"]["rows"] + T["r35"]["margins"]), ("20.6", T["r206"]["rows"]),
                    ("20.7", T["r33"]["rows"]), ("20.8", T["r31"]["rows"] + T["r31"]["contrasts"])):
        for label, needle in rows:
            check_true(f"FINDINGS {label} quoted", needle in sub[k])
    lo, hi = r210["0.45"]["sh_range"]
    check_true("20.4 quotes the range of the drift-0.45 shuffled controls", f"{lo:.3f} to {hi:.3f}" in flat["20.4"])
    check_true("20.5 quotes the frozen labels",
               f"untrained arm: **{T['r35']['labels']['untrained']}**" in flat["20.5"]
               and f"predictor arm: {T['r35']['labels']['predictor']}" in flat["20.5"])
    check_true("20.6 quotes the label and the correlation",
               "**HIGHER**" in flat["20.6"] and f"r = {T['r206']['r']:.3f}" in flat["20.6"])
    L1 = s20_l1()
    for label, needle in L1["rows"] + L1["contrasts"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["20.9"])
    # 20.10 and 20.11: the same readouts on the C2-rerun and H10GS1-corrected agents
    R = {}
    for k, fname, tag in (("20.10", "C2-rerun.json", "C2-rerun"), ("20.11", "H10GS1-corrected.json", "H10GS1"),
                          ("20.13", "H7-corrected.json", "H7"), ("20.14", "H8GS2-corrected.json", "H8GS2")):
        X = s20_readout(fname, tag)
        R[tag] = X
        # The instance runs' drift-0 agents are C1's, so their drift-0 table is 20.4's and is not repeated.
        d0rows = X["r210"]["0.00"]["rows"] if tag == "C2-rerun" else []
        rows = (X["r210"]["0.45"]["rows"] + X["r210"]["0.45"]["margins"] + d0rows
                + X["r35"]["rows"] + X["r35"]["margins"] + X["r206"]["rows"] + X["r33"]["rows"]
                + X["r31"]["rows"] + X["r31"]["contrasts"])
        for label, needle in rows:
            check_true(f"FINDINGS {k} {label} quoted", needle in sub[k])
        check_true(f"{k} quotes the R35 labels, the R206 label and its correlation",
                   f"untrained arm: **{X['r35']['labels']['untrained']}**" in flat[k]
                   and f"predictor arm: {X['r35']['labels']['predictor']}" in flat[k]
                   and f"**{X['r206']['label']}**" in flat[k] and f"r = {X['r206']['r']:.3f}" in flat[k])
    for tag in ("H10GS1", "H7", "H8GS2"):
        check_true(f"{tag} readout: the drift-0 R210 readings equal C1's value for value",
                   R[tag]["r210"]["0.00"]["v"] == T["r210"]["0.00"]["v"])
    check_true("20.8 every decoder's final training loss is below 0.001 in every fold, conv included",
               T["r31"]["conv_max"] < 0.001)
    for k, tag in (("20.13", "H7"), ("20.14", "H8GS2")):
        X = R[tag]["r31"]
        check_true(f"{k} quotes how many conv folds reach a training loss of 0.001 and the largest",
                   f"in {X['conv_n_ge']} of its 100 folds (largest {X['conv_max']:.4f})" in flat[k])
    # The closing paragraph of 20.14 compares the labels across the five reruns.
    _all = {"C1": T, **R}
    check_true("20.14 across-run R35 labels: TRUNK on C1 and the three instances, SPLIT on C2",
               all(_all[t]["r35"]["labels"]["untrained"] == "TRUNK" for t in ("C1", "H10GS1", "H7", "H8GS2"))
               and _all["C2-rerun"]["r35"]["labels"]["untrained"] == "SPLIT")
    check_true("20.14 across-run R206 labels: HIGHER on C1 and H10GS1, MIXED on C2 and H8GS2, NOT HIGHER on H7",
               [_all[t]["r206"]["label"] for t in ("C1", "H10GS1", "C2-rerun", "H8GS2", "H7")]
               == ["HIGHER", "HIGHER", "MIXED", "MIXED", "NOT HIGHER"])
    check_true("20.14 across-run R31: no run is CAPACITY-LIMITED on the survival agent's own stream",
               all("| survival agent's own walk |" in r and "NOT CAPACITY-LIMITED" in r
                   for t in _all for lab, r in _all[t]["r31"]["contrasts"] if "survival_own" in lab))
    c2cells = _s20_cells("corrected_runs/corrected_l3_h8_nowm")
    check_true("20.10 C2-rerun readout: the recorded drift-0.45 survival targets are C2's committed targets",
               R["C2-rerun"]["recorded"] == [c2cells[f"cell_d0.45_s{s}.json"]["cell"]["agents"]["survival"]["pool"]["target"]
                                              for s in S20_SEEDS])
    # PAPER_OUTLINE 4.5 and row 44, and the REVISION claim table: dated notes that point here.
    def _doc(rel):
        with open(os.path.join(ARTROOT, "..", *rel.split("/")), encoding="utf-8") as fh:
            return " ".join(fh.read().split())
    _po, _rv = _doc("docs/PAPER_OUTLINE.md"), _doc("docs/REVISION_2026-10.md")
    _m35 = T["doc"]["summary"]["r35"]["margins"]["survival_minus_untrained"]
    _lab = (f"{T['r35']['labels']['untrained']} on the `C1` agents",
            f"{R['C2-rerun']['r35']['labels']['untrained']} on the `C2` agents")
    _row44 = next((ln for ln in _po.split("| 44 |")[1:2]), "")
    check_true("PAPER_OUTLINE 4.5 note quotes the R35 margins and labels of C1 and C2",
               all(x in _po for x in (f"{_m35['survival_norm']['mean']:+.3f} on the survival agent's normalizer",
                                      f"{_m35['refit']['mean']:+.3f} on one refit",
                                      f"{_m35['own']['mean']:+.3f} on each trunk's own", *_lab)))
    check_true("PAPER_OUTLINE row 44 carries the dated note with both labels",
               "Note 2026-10-11" in _row44.split("| FINDINGS 17.6 |")[0]
               and all(x in _row44 for x in (f"{T['r35']['labels']['untrained']} on the `C1` agents, "
                                             f"{R['C2-rerun']['r35']['labels']['untrained']} on the `C2` agents",)))
    _rvn = _rv[_rv.index("*(Note 2026-10-11, claims 21, 22 and 31"):]
    check_true("REVISION note gives each rerun's outcome and the inherited L0 gate",
               all(x in _rvn for x in (f"({n['h7']['floor']:.3f})",
                                       f"{n['h10']['s'][0]:.3f} {_af_iv(n['h10']['s'][1])}",
                                       f"{inst['speed_min']:.3f}", "off-design", "uninformative",
                                       "open L0 gate (0.559")))
    CT = s20_contrast()
    for label, needle in CT["rows"]:
        check_true(f"FINDINGS {label} quoted", needle in sub["20.12"])
    check_true("section 20 states that every run and readout is complete, with no pending placeholder",
               "every run and readout the two specs name is complete" in " ".join(s20.split())
               and "pending" not in s20)
    notes = [" ".join(x.split()) for x in re.findall(r"\*\(Note 2026-10-11.*?\)\*", findings, flags=re.S)]

    def in_note(needle: str) -> bool:
        return any(needle in x for x in notes)

    for label, needle in [
        ("TL;DR: the reruns", "rerun with the corrected trainer (section 20)"),
        ("10.5: H7-corrected gate-0 floor", f"floor {n['h7']['floor']:.3f}"),
        ("10.9: H10GS1-corrected survival", f"{n['h10']['s'][0]:.3f} {_af_iv(n['h10']['s'][1])}"),
        ("10.9: H8GS2-corrected lead", f"{n['h8s2']['sp'][0]:+.3f} over the predictor"),
        ("methods note 4", "(section 20)"),
        ("10.3: corrected forest on the untrained arm", _s20_cell(r210["0.45"]["v"]["untrained"]["rf"])),
        ("14.7.1: READER-RELATIVE", "READER-RELATIVE"),
        ("14.7.1: the conv reader on the survival streams", L1["conv"]["survival"]),
        ("17.5: C2 agents kept", "`C2-rerun`"),
        ("17.4 and 17.5: the draw-free contrast, fresh base",
         f"{CT['c']['d0.45/fresh'][0]:+.3f} {_af_iv(CT['c']['d0.45/fresh'][1], True)}"),
        ("17.4: the draw-free contrast, registered base",
         f"{CT['c']['d0.45/registered'][0]:+.3f} {_af_iv(CT['c']['d0.45/registered'][1], True)}"),
        ("17.6: R35", f"**{T['r35']['labels']['untrained']}**"),
        ("17.6: R206", "**HIGHER**"),
        ("17.8: R31", "NOT CAPACITY-LIMITED"),
    ]:
        check_true(f"FINDINGS dated note (2026-10-11) quotes {label}: {needle}", in_note(needle))
    # 20.15: R205, the mean-h / h_T decomposition, and the 2026-10-12 notes that point to it.
    D = s20_r205()
    for dr in ("0.45", "0.00"):
        for label, needle in D[dr]["rows"]:
            check_true(f"FINDINGS {label} quoted", needle in sub["20.15"])
    lo, hi = min(D["0.45"]["sh_range"][0], D["0.00"]["sh_range"][0]), max(D["0.45"]["sh_range"][1], D["0.00"]["sh_range"][1])
    check_true("20.15 quotes the range of the shuffled controls", f"{lo:.3f} to {hi:.3f}" in flat["20.15"])
    labs = {D[(dr, a)]["label"] for dr in ("0.45", "0.00") for a in S20_ARMS}
    check_true("20.15 says the label is BOTH for every arm at both drifts",
               labs == {"BOTH"} and "The label is BOTH for every arm at both drifts" in flat["20.15"])
    sv, sv0 = D[("0.45", "survival")], D[("0.00", "survival")]
    _cm, _cf, _mf = sv["d"]["concat_minus_mean"], sv["d"]["concat_minus_final"], sv["d"]["mean_minus_final"]
    for needle in (f"short of the concatenation's {np.mean(sv['v']['concat']):.3f} by {_cm[0]:+.3f} {_af_iv(_cm[1], True)}",
                   f"The final state alone reads {np.mean(sv['v']['final']):.3f}, within {_cf[0]:.3f} of the "
                   f"concatenation ({_cf[0]:+.3f} {_af_iv(_cf[1], True)})",
                   f"mean h minus h_T is {_mf[0]:+.3f} {_af_iv(_mf[1], True)}",
                   f"it falls {_cm[0]:.3f} short",
                   f"({sv0['d']['mean_minus_final'][0]:+.3f} {_af_iv(sv0['d']['mean_minus_final'][1], True)})"):
        check_true(f"20.15 reading quotes: {needle}", needle in flat["20.15"])
    check_true("20.15: the survival arm's mean h alone falls short of the concatenation (interval above zero) "
               "and h_T alone reads above mean h on the seed means", _cm[1][0] > 0 and _mf[0] < 0)
    _n12 = [" ".join(x.split()) for x in re.findall(r"\*\(Note 2026-10-12.*?\)\*", findings, flags=re.S)]
    _r205 = (f"{np.mean(sv['v']['mean']):.3f} from mean h alone, {np.mean(sv['v']['final']):.3f} from h_T alone "
             f"and {np.mean(sv['v']['concat']):.3f} from the concatenation")
    check_true("17.7 note (2026-10-12) quotes the R205 decomposition and its label",
               any(_r205 in x and "label is BOTH" in x and f"{_mf[0]:+.3f} {_af_iv(_mf[1], True)}" in x for x in _n12))
    check_true("20.4 carries a 2026-10-12 pointer to 20.15",
               "*(Note 2026-10-12: R205 splits the registered linear feature" in sub["20.4"])
    with open(os.path.join(ARTROOT, "..", "docs", "PAPER_OUTLINE.md"), encoding="utf-8") as fh:
        _po12 = " ".join(fh.read().split())
    check_true("PAPER_OUTLINE 4.3 note (2026-10-12) quotes the R205 halves and label",
               f"Mean h alone reads {np.mean(sv['v']['mean']):.3f} and h_T alone {np.mean(sv['v']['final']):.3f}" in _po12
               and "the pre-stated label is BOTH" in _po12)
    head_a = "CORRECTED-TRAINER RERUNS OF THE GENERALITY RUNS ANNOUNCED"
    head_r = "CORRECTED-TRAINER RERUNS OF THE GENERALITY RUNS: RESULTS"
    check_true("PREREGISTRATION_L3 carries the results entry after the announcement",
               head_r in prereg and prereg.index(head_a) < prereg.index(head_r) < prereg.index("## 13. How to run"))
    entry = " ".join(prereg[prereg.index(head_r):prereg.index("## 13. How to run")].split())
    for needle in (f"{n['h7']['floor']:.3f}", f"{n['h10']['s'][0]:.3f} {_af_iv(n['h10']['s'][1])}",
                   f"{n['h8s2']['s'][0]:.3f} {_af_iv(n['h8s2']['s'][1])}", f"{inst['speed_min']:.3f}",
                   "off-design", "uninformative", f"{c2['shared']} values"):
        check_true(f"PREREGISTRATION_L3 results entry quotes {needle}", needle in entry)


def _find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while (pos := haystack.find(needle, start)) != -1:
        out.append(pos)
        start = pos + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
