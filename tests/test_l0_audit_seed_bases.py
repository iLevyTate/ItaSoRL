"""The L0 audit's independent world samples, against every other seed base the code uses.

itasorl/l0_audit.py documents which evaluation bases coincide with audit pools. This pins
that record: the audit pairs must stay disjoint from each other, from the registered pair
and from training, and the only evaluation bases that land on an audit pool are the two the
comment names. A new runner base that collides with an audit pool fails here until the
comment (and this list) say so.

The constants are read from the runner sources with ast, so the test imports no runner.
"""

from __future__ import annotations

import ast
import os

from itasorl.l0_audit import AUDIT_BASES, STANDARD_BASES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOL_N = 110                 # the largest pool any of these readouts draws
TRAINING_MAX = 285_800       # predictor seeds 200000 + 9000*9 + 16*299 + 15 < this


def _literal(node):
    """Plain literal of a constant, tuple or dict node (the only shapes these constants use)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    raise AssertionError(f"unexpected node {ast.dump(node)}")


def _constant(script: str, name: str):
    with open(os.path.join(ROOT, "scripts", script), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return _literal(node.value)
    raise AssertionError(f"{name} not found in scripts/{script}")


def _evaluation_bases() -> dict[str, int]:
    out = {}
    for tag, (a, s) in _constant("run_l3_h2_ablations.py", "SEED_BASES").items():
        out[f"h2 {tag} authentic"], out[f"h2 {tag} surrogate"] = a, s
    for fam, (a, s) in _constant("run_texture_fresh_probe.py", "BASES").items():
        out[f"texture {fam} authentic"], out[f"texture {fam} surrogate"] = a, s
    out["mechanism world"] = _constant("run_mechanism_readouts.py", "WORLD_BASE")
    return out


def _overlaps(b1: int, b2: int, n: int = POOL_N) -> bool:
    return b1 < b2 + n and b2 < b1 + n


def test_audit_pairs_are_disjoint_from_each_other_the_registered_pair_and_training():
    pools = [b for pair in AUDIT_BASES for b in pair]
    assert len(pools) == 16
    for i, a in enumerate(pools):
        assert a > TRAINING_MAX
        for b in pools[i + 1:]:
            assert not _overlaps(a, b), (a, b)
        for b in STANDARD_BASES:
            assert not _overlaps(a, b), (a, b)


def test_only_the_documented_evaluation_bases_land_on_an_audit_pool():
    audit = {b: (k + 1, side) for k, pair in enumerate(AUDIT_BASES)
             for side, b in zip(("authentic", "surrogate"), pair)}
    hits = {(name, audit[a]) for name, base in _evaluation_bases().items()
            for a in audit if _overlaps(base, a)}
    assert hits == {("h2 h32 authentic", (1, "authentic")),
                    ("mechanism world", (5, "authentic"))}


def test_the_comment_names_the_overlaps_it_documents():
    with open(os.path.join(ROOT, "itasorl", "l0_audit.py"), encoding="utf-8") as fh:
        src = fh.read()
    for phrase in ("audit draw 1's authentic pool", "audit draw 5's authentic base",
                   "980_000 to 1_030_109", "1_900_000 / 1_950_000"):
        assert phrase in src
