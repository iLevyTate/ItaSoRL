"""The statistics layer must fail closed, not open.

A statistics-code audit (2026-10-09) found three paths that are harmless on the inputs this
project actually uses and wrong on degenerate ones: a Student-t interval that silently becomes
a normal interval if scipy is absent, two equivalence checks that ACCEPT equivalence from a
single observation, and a registered margin applied with >= in one place and > in another. None
moves a published number. All three would mislead silently if they ever fired, which is the
property a decision layer must not have.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from itasorl.stats import equivalence_test, rope_test, t_ci90


# --------------------------------------------------------------- the t interval is a t interval

def test_the_t_interval_refuses_to_quietly_become_a_normal_interval(monkeypatch):
    """scipy is a declared dependency. Without it the old code substituted 1.645 for the t
    quantile (2.262 at n = 10), narrowing every interval by a quarter while the paper calls
    them t-based. Stopping is better than relabelling."""
    import itasorl.stats as st

    v = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.60, 0.62, 0.58]
    lo, hi = t_ci90(v)                      # with scipy present, the real thing
    import scipy.stats as sps
    crit = sps.t.ppf(0.95, len(v) - 1)
    se = float(np.std(v, ddof=1)) / np.sqrt(len(v))
    assert lo == pytest.approx(np.mean(v) - crit * se)
    assert hi == pytest.approx(np.mean(v) + crit * se)

    monkeypatch.setattr(st, "_HAVE_SCIPY", False)
    with pytest.raises(RuntimeError, match="scipy"):
        st.t_ci90(v)


# --------------------------------------------------------------- equivalence cannot come free

def test_one_observation_cannot_establish_equivalence_to_chance():
    """The L0 gate is an equivalence test. With a single value there is no variance, so the
    old code checked only whether the point sat inside the band and returned equivalent=True."""
    r = equivalence_test([0.50])
    assert not r.equivalent, "a single observation must never accept equivalence"
    assert np.isnan(r.p_value)
    assert equivalence_test([]).equivalent is False


def test_one_observation_cannot_accept_the_rope_either():
    r = rope_test([0.50])
    assert not r.accept
    assert rope_test([]).accept is False


def test_equivalence_still_accepts_and_rejects_correctly_at_n_ten():
    tight = [0.50, 0.501, 0.499, 0.5005, 0.4995, 0.50, 0.5002, 0.4998, 0.5001, 0.4999]
    assert equivalence_test(tight).equivalent
    off = [0.60, 0.61, 0.59, 0.60, 0.62, 0.58, 0.60, 0.61, 0.59, 0.60]
    assert not equivalence_test(off).equivalent
    # The binding p is the larger of the two one-sided legs, never the smaller.
    r = equivalence_test(tight)
    assert r.p_value == pytest.approx(max(r.p_lower, r.p_upper))


# --------------------------------------------------------------- the margin is "at least"

def _bcv():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import build_corrected_verdicts as bcv
    return bcv


def test_the_registered_margin_is_at_least_everywhere_it_is_applied():
    """Section 6 of the preregistration reads "exceeds both baselines by at least 0.05". One
    instance on record missed a margin by a thousandth, so the boundary is not hypothetical.
    The behavioral tests below protect the rule; this grep only catches the obvious spelling."""
    import inspect

    src = inspect.getsource(_bcv())
    # No comparison against the margin may use a strict inequality.
    assert "> MARGIN" not in src.replace(">= MARGIN", ""), \
        "the registered margin is 'at least', so every comparison must be >="


# A lead equal to the margin must pass and one ulp below it must fail. No three-decimal pair
# of AUROCs differs by exactly 0.05 in float64 (0.6875 - 0.6375 is 0.050000000000000044), so
# the tests compute the lead the way the code does and move the margin onto it, bit for bit.
_SURV = [0.6875, 0.7012, 0.6633, 0.7240, 0.6801, 0.6950, 0.6712, 0.7105, 0.6890, 0.6982]
_PRED = [0.6375, 0.6211, 0.6490, 0.6302, 0.6188, 0.6455, 0.6270, 0.6333, 0.6401, 0.6219]
_UNTR = [0.5012, 0.4870, 0.5133, 0.4951, 0.5066, 0.4902, 0.5098, 0.4987, 0.5041, 0.4929]


def _cells(surv, pred, untr):
    bcv = _bcv()
    return {(0.45, s): {"agents": {"survival": {"pool": {"target": surv[s]}},
                                   "predictor": {"pool": {"target": pred[s]}},
                                   "untrained": {"pool": {"target": untr[s]}}}}
            for s in bcv.SEEDS}


def test_primary_margin_passes_at_exactly_the_margin_and_fails_one_ulp_above(monkeypatch):
    bcv = _bcv()
    cells = _cells(_SURV, _PRED, _UNTR)
    s_mean = bcv._summ(_SURV)["mean"]
    for clause, base in (("margin_predictor", _PRED), ("margin_untrained", _UNTR)):
        lead = s_mean - float(np.mean(base))          # exactly as primary() computes it
        monkeypatch.setattr(bcv, "MARGIN", lead)
        assert bcv.primary(cells, {"failed": []})["clauses"][clause] is True, clause
        monkeypatch.setattr(bcv, "MARGIN", float(np.nextafter(lead, np.inf)))
        assert bcv.primary(cells, {"failed": []})["clauses"][clause] is False, clause


def test_paired_contrast_margin_passes_at_exactly_the_margin_and_fails_one_ulp_above():
    from itasorl.stats import paired_contrast

    lead = float((np.asarray(_SURV) - np.asarray(_PRED)).mean())   # as paired_contrast does
    assert paired_contrast(_SURV, _PRED, margin=lead)["mean_ge_margin"] is True
    assert paired_contrast(_SURV, _PRED, margin=float(np.nextafter(lead, np.inf)))[
        "mean_ge_margin"] is False
