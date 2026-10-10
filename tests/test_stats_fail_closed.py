"""The statistics layer must fail closed, not open.

A statistics-code audit (2026-10-09) found three paths that are harmless on the inputs this
project actually uses and wrong on degenerate ones: a Student-t interval that silently becomes
a normal interval if scipy is absent, two equivalence checks that ACCEPT equivalence from a
single observation, and a registered margin applied with >= in one place and > in another. None
moves a published number. All three would mislead silently if they ever fired, which is the
property a decision layer must not have.
"""

from __future__ import annotations

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

def test_the_registered_margin_is_at_least_everywhere_it_is_applied():
    """Section 6 of the preregistration reads "exceeds both baselines by at least 0.05". One
    instance on record missed a margin by a thousandth, so the boundary is not hypothetical."""
    import inspect

    import scripts  # noqa: F401
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import build_corrected_verdicts as bcv

    src = inspect.getsource(bcv)
    # No comparison against the margin may use a strict inequality.
    assert "> MARGIN" not in src.replace(">= MARGIN", ""), \
        "the registered margin is 'at least', so every comparison must be >="
