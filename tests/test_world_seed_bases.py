"""Closing item G31 and H25 (runbook 1.5): the world-seed bases registered for the closing
readouts are fresh.

The B2 walk-speed pools, the B6 development transfer pools and the G03 fresh read base
(itasorl.post_review.REGISTERED_LATER) must miss every base a registered readout has drawn
(BASES_IN_USE plus the post-review NEW_BASES, including the H25 entry for the predictor
schedule of seeds 14 to 19), and must miss each other, by seed footprint over all three RNG
streams (post_review.seed_footprint)."""

from __future__ import annotations

import itertools

import pytest

from itasorl import post_review as pr

LATER = {name: (base, n) for name, base, n in pr.REGISTERED_LATER}


def test_registered_later_names_the_closing_bases():
    assert LATER == {"B2-auth": (2_800_000, pr.POOL_N), "B2-surr": (2_850_000, pr.POOL_N),
                     "B6-auth": (2_700_000, pr.POOL_N), "B6-surr": (2_750_000, pr.POOL_N),
                     "G03-fresh": (2_250_000, pr.POOL_N),
                     # prospective balanced confirmation, spec 2026-10-20
                     "PB-primary": (3_000_000, pr.POOL_N), "PB-draw2": (3_100_000, pr.POOL_N)}
    for name in LATER:
        assert pr.registered_later(name) == (name, *LATER[name])
    with pytest.raises(KeyError):
        pr.registered_later("B9-auth")


def test_all_bases_in_use_is_the_record_and_carries_the_h25_entry():
    assert pr.all_bases_in_use() == pr.BASES_IN_USE + pr.NEW_BASES
    h25 = [e for e in pr.BASES_IN_USE if e[1:] == (326_000, 49_800)]
    assert len(h25) == 1 and "seeds 14 to 19" in h25[0][0]
    # 200000 + 9000 x 14 to 200000 + 9000 x 19 + 16 x 300, past the first entry's span
    first = pr.BASES_IN_USE[0]
    assert first[1] + first[2] < 200_000 + 9_000 * 19 + 16 * 300 == 326_000 + 49_800
    # REGISTERED_LATER is kept apart from the record of drawn bases
    names = {e[0] for e in pr.all_bases_in_use()}
    assert not names & set(LATER)


@pytest.mark.parametrize("name", sorted(LATER))
def test_each_closing_base_misses_every_base_in_use(name):
    assert pr.base_collisions(new=(pr.registered_later(name),), in_use=pr.all_bases_in_use()) == []


def test_the_closing_bases_miss_the_h25_entry_by_footprint():
    h25 = next(e for e in pr.BASES_IN_USE if e[1:] == (326_000, 49_800))
    for name, base, n in pr.REGISTERED_LATER:
        assert not pr.footprints_overlap(base, n, h25[1], h25[2]), name


def test_the_closing_bases_are_pairwise_disjoint_by_footprint():
    for (a, ba, na), (b, bb, nb) in itertools.combinations(pr.REGISTERED_LATER, 2):
        assert not pr.footprints_overlap(ba, na, bb, nb), (a, b)
    assert pr.base_collisions(new=pr.REGISTERED_LATER, in_use=pr.all_bases_in_use()) == []


def test_the_check_sees_a_collision():
    # 2_200_000 is r33's balanced base; the G03 base was moved off it (R3-3)
    assert pr.base_collisions(new=(("x", 2_200_000, pr.POOL_N),), in_use=pr.all_bases_in_use()) == [
        ("x", "post-review balanced pairs")]
    # a pool that runs into a neighbour's footprint through a second RNG stream collides too
    b2a = pr.registered_later("B2-auth")
    assert pr.base_collisions(new=(("y", b2a[1] + 17_001, 5),), in_use=(b2a,)) == [("y", "B2-auth")]
