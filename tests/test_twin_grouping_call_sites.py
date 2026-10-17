"""A structural guard against the twin-grouping defect (A6d), which the review found three
times: a cross-validated readout on paired or twin rollouts that does not pass `groups=` puts
one world seed's episodes in different folds, and the probe reads the training twin's label off
its test twin.

Every probe function under itasorl/ whose `groups` parameter defaults to None (one group per
row) is found by parsing the source, and every call to one of them under itasorl/ and scripts/
that passes no `groups=` (or passes a value that is None on the default path) must be in
ALLOWED with a reason. Each reason names a needle, a piece of the calling file's source that
shows why the default is right there, and the test checks the needle is present, so a reason
cannot outlive the code it describes. The three pooled-row sites are marked "disclosed, not
fixed": their default groups pooled rows by row index although several individuals share one
world seed (docs/CORRECTIONS.md; the opt-in world-seed grouping is the closing hook A7a). A new
call site fails this test until it passes `groups=` or is added here with a checked reason."""

from __future__ import annotations

import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCAN = ("itasorl", "scripts")

INDEPENDENT = ("independent pools: the authentic and surrogate episodes are drawn at different "
               "world-seed bases, so no world seed repeats across rows and one group per row is "
               "the right partition")
PAIRS = ("one agent's matched pairs: cg_probe's default puts the two members of a pair, which "
         "share one world seed, in one group, and each pair has its own seed")
DISCLOSED = ("disclosed, not fixed: tails of several individuals run on the same pair seeds are "
             "pooled, and the default groups by pooled row, so one world seed's pairs can sit in "
             "different folds (CORRECTIONS 2026-10-16; world-seed grouping is opt-in)")

# (file, enclosing function, callee, first argument as written): (calls, reason, needle)
ALLOWED = {
    ("itasorl/behavior_audit.py", "audit_cell", "behavior_only_auroc", "B"):
        (2, INDEPENDENT + " (a pooled-readout dump: Ha at 800000, Hs at 850000)", 'npz["Hs"]'),
    ("itasorl/behavior_audit.py", "audit_cell", "residual_probe_auroc", "X"):
        (2, INDEPENDENT + " (a pooled-readout dump)", 'npz["Hs"]'),
    ("itasorl/behavior_audit.py", "audit_cell", "trace_residual_probe_auroc", "H"):
        (2, INDEPENDENT + " (a pooled-readout dump)", 'npz["Hs"]'),
    ("itasorl/experiment_b2.py", "pooled_readout", "_auroc_with_ci", "X"):
        (1, INDEPENDENT, "The two pools are drawn from DIFFERENT world samples"),
    ("itasorl/experiment_b2.py", "pooled_readout", "_auroc_with_ci", "Xv"):
        (1, INDEPENDENT, "The two pools are drawn from DIFFERENT world samples"),
    ("itasorl/experiment_b2.py", "pooled_readout", "_auroc_with_ci", "Xf"):
        (1, INDEPENDENT, "The two pools are drawn from DIFFERENT world samples"),
    ("itasorl/experiment_b2.py", "pooled_readout", "leakage_audit_b2", "auth_eps"):
        (1, INDEPENDENT + " (the same pooled episodes)", "seed_base_surr"),
    ("itasorl/experiment_c.py", "individual_probe_panel", "cg_probe", "a"):
        (1, PAIRS, "a, s = common_garden_rollout(agent"),
    ("itasorl/experiment_c.py", "individual_probe_panel", "cg_probe", "auth_all"):
        (1, DISCLOSED, "auth_all.extend(a)"),
    ("itasorl/experiment_c.py", "common_garden_panel", "cg_probe", "auth"):
        (1, DISCLOSED, "if world_seed_groups else None"),
    ("itasorl/experiment_c.py", "common_garden_panel", "cg_probe", "a0"):
        (1, DISCLOSED, "if world_seed_groups else None"),
    ("scripts/validate_population_readout.py", "pooled_probe_block", "cg_probe", "auth"):
        (1, DISCLOSED, '"pooled_probe": float(cg_probe(auth, surr)'),
    ("scripts/reanalyze_cg_states.py", "rescore_cell", "cg_probe", "list(auth)"):
        (1, PAIRS + " (the stored tails of one agent)", 'npz["surr"]'),
    ("scripts/run_expB2.py", "evaluate_agent", "cg_probe", "at"):
        (1, PAIRS, "at, st = common_garden_rollout(agent"),
    ("scripts/rescore_fold_split.py", "score_cell", "trace_residual_probe_auroc", "H"):
        (1, INDEPENDENT + " (a pooled-readout dump)", 'npz["Hs"]'),
    ("scripts/run_control_diagnostics.py", "run_one", "trace_residual_probe_auroc", "H"):
        (1, INDEPENDENT, '("s", d, 850_000)'),
    ("scripts/run_control_diagnostics.py", "run_one", "sensory_residual_probe_auroc", "H"):
        (1, INDEPENDENT, '("s", d, 850_000)'),
    ("scripts/run_control_diagnostics.py", "run_one", "residual_probe_with_diagnostics", "H"):
        (6, INDEPENDENT, '("s", d, 850_000)'),
    ("scripts/run_control_diagnostics.py", "run_one", "sequence_gru_auroc", "seq"):
        (1, INDEPENDENT, '("s", d, 850_000)'),
    ("scripts/run_control_diagnostics.py", "run_one", "flat_sequence_linear_auroc", "seq"):
        (1, INDEPENDENT, '("s", d, 850_000)'),
    ("scripts/run_cross_replay.py", "run_seed", "readout_from_states", "replay_states(ag, nm, oa, aa)"):
        (1, INDEPENDENT, '"cpu", 850_000, rs)'),
    ("scripts/run_development_readout.py", "battery", "residual_probe_with_diagnostics", "H"):
        (3, INDEPENDENT + " (AUTH_BASE 800000, SURR_BASE 850000)", '("s", DRIFT, SURR_BASE)'),
    ("scripts/run_development_readout.py", "run_task", "readout_from_states", "replay_states(ag, nm, Oa, Aa)"):
        (1, INDEPENDENT + " (AUTH_BASE 800000, SURR_BASE 850000)", '"cpu", SURR_BASE, rs)'),
    ("scripts/run_l1_capacity_curve.py", "decode", "sequence_gru_auroc", "seq"):
        (1, INDEPENDENT + " (the pools of run_cell)", '("s", d, 850_000)'),
    ("scripts/run_l1_capacity_curve.py", "decode", "conv_sequence_auroc", "seq"):
        (1, INDEPENDENT + " (the pools of run_cell)", '("s", d, 850_000)'),
    ("scripts/run_mechanism_readouts.py", "run_seed", "sequence_gru_auroc", "seq"):
        (1, INDEPENDENT + " (the scripted streams)", "seed_base=850_000"),
    ("scripts/run_optional_o2.py", "run_seed", "long_history_ridge", "H"):
        (1, INDEPENDENT, '("s", 0.45, 850_000)'),
    ("scripts/run_policy_controlled_readouts.py", "run_seed", "readout_from_states", "Ha"):
        (1, INDEPENDENT, '"cpu", 850_000, rs)'),
    ("scripts/run_policy_controlled_readouts.py", "run_seed", "readout_from_states", "Sa"):
        (1, INDEPENDENT, '"cpu", 850_000, rs)'),
    ("scripts/run_policy_controlled_readouts.py", "run_seed", "readout_from_states",
     "replay_states(ag, nm, obs_a, act_a)"):
        (1, INDEPENDENT, '"cpu", 850_000, rs)'),
    ("scripts/run_post_review_readouts.py", "task_r35", "readout_from_states", "H[:na]"):
        (1, INDEPENDENT + " (FRESH_AUTH_BASE and FRESH_SURR_BASE)", "pr.FRESH_SURR_BASE"),
    ("scripts/run_post_review_readouts.py", "task_r35", "readout_from_states", "H_refit[:na]"):
        (1, INDEPENDENT + " (FRESH_AUTH_BASE and FRESH_SURR_BASE)", "pr.FRESH_SURR_BASE"),
    ("scripts/run_post_review_readouts.py", "task_r35", "infold_normalizer_auroc", "obs"):
        (1, INDEPENDENT + " (FRESH_AUTH_BASE and FRESH_SURR_BASE)", "pr.FRESH_SURR_BASE"),
    ("scripts/run_surprise_readout.py", "run_seed", "surprise_auroc", "Ea"):
        (1, INDEPENDENT + " (AUTH_BASE and SURR_BASE streams; the twin-paired surprise readout "
            "is run_mechanism_readouts, which passes groups)", '"cpu", SURR_BASE, rs)'),
    ("scripts/run_surprise_readout.py", "run_seed", "surprise_auroc", "mr.replay_prediction_error(surv, snorm, O1, A1)"):
        (1, INDEPENDENT + " (the L0 analogue: two authentic pools at AUTH_BASE and SURR_BASE)",
         'O2, A2, _ = record_trajectories(surv, snorm, P, 0.0, n, T, "cpu", SURR_BASE, rs)'),
    ("scripts/run_surprise_readout.py", "run_seed", "readout_from_states", "Ha"):
        (1, INDEPENDENT + " (AUTH_BASE and SURR_BASE streams)", '"cpu", SURR_BASE, rs)'),
    ("scripts/run_texture_fresh_probe.py", "run_one", "readout_from_states", "Ha_c"):
        (1, INDEPENDENT + " (the family's own two bases)", "ba, bs = BASES[task[\"family\"]]"),
}


def _files():
    for d in SCAN:
        for dp, _, fs in os.walk(os.path.join(ROOT, d)):
            for f in sorted(fs):
                if f.endswith(".py"):
                    yield os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/")


def _parse(rel: str) -> ast.Module:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def optional_group_functions(paths=None) -> dict[str, int | None]:
    """Name -> positional index of `groups` (None if keyword-only), for every function in
    itasorl/ whose `groups` parameter defaults to None."""
    out = {}
    for rel in paths or [p for p in _files() if p.startswith("itasorl/")]:
        for n in ast.walk(_parse(rel)):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = n.args
            pos = [x.arg for x in a.posonlyargs + a.args]
            if "groups" in pos:
                i = pos.index("groups")
                k = i - (len(pos) - len(a.defaults))
                if k >= 0 and isinstance(a.defaults[k], ast.Constant) and a.defaults[k].value is None:
                    out[n.name] = i
            kwo = [x.arg for x in a.kwonlyargs]
            if "groups" in kwo:
                d = a.kw_defaults[kwo.index("groups")]
                if isinstance(d, ast.Constant) and d.value is None:
                    out[n.name] = None
    return out


def _none_on_default(value: ast.AST, fn: ast.AST | None) -> bool:
    """True for None, `x if cond else None`, or a name the function binds to either."""
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if isinstance(value, ast.IfExp):
        return any(isinstance(b, ast.Constant) and b.value is None for b in (value.body, value.orelse))
    if isinstance(value, ast.Name) and fn is not None:
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets):
                if _none_on_default(n.value, None):
                    return True
    return False


def ungrouped_calls(paths=None, funcs=None) -> list[tuple[str, str, str, str]]:
    funcs = funcs if funcs is not None else optional_group_functions()
    out = []
    for rel in paths or list(_files()):
        tree = _parse(rel) if isinstance(rel, str) else rel
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name not in funcs:
                continue
            fn = n
            while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents[fn]
            fn = fn if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            kw = {k.arg: k.value for k in n.keywords}
            if None in kw:                       # **kwargs: cannot tell, treat as passed
                continue
            if "groups" in kw:
                if not _none_on_default(kw["groups"], fn):
                    continue
            elif funcs[name] is not None and len(n.args) > funcs[name]:
                continue
            first = ast.unparse(n.args[0]) if n.args else ""
            out.append((rel, fn.name if fn else "<module>", name, first))
    return out


def test_every_ungrouped_call_is_allowed_with_its_count():
    found: dict = {}
    for key in ungrouped_calls():
        found[key] = found.get(key, 0) + 1
    unknown = {k: v for k, v in found.items() if k not in ALLOWED}
    assert not unknown, ("CV call(s) on rollouts without groups=; pass groups (one id per world "
                         "seed or twin pair) or add a checked reason here", unknown)
    wrong = {k: (found.get(k, 0), ALLOWED[k][0]) for k in ALLOWED if found.get(k, 0) != ALLOWED[k][0]}
    assert not wrong, ("allowlist entries whose call count changed or that no longer exist", wrong)


def test_every_reason_is_checked_against_the_source():
    for (rel, fn, callee, _), (_, reason, needle) in ALLOWED.items():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            src = fh.read()
        assert f"def {fn}(" in src, (rel, fn)
        assert needle in src, (rel, fn, callee, needle)
        assert reason.startswith(("independent pools", "one agent's matched pairs", "disclosed, not fixed"))


def test_the_pooled_row_sites_are_the_disclosed_ones():
    disclosed = sorted({(k[0], k[1]) for k, v in ALLOWED.items() if v[1].startswith("disclosed, not fixed")})
    assert disclosed == [("itasorl/experiment_c.py", "common_garden_panel"),
                         ("itasorl/experiment_c.py", "individual_probe_panel"),
                         ("scripts/validate_population_readout.py", "pooled_probe_block")]


def test_the_scanner_sees_a_missing_and_a_passed_groups(tmp_path):
    src = ("def probe(X, y, groups=None):\n    pass\n"
           "def caller(X, y, g, flag):\n"
           "    probe(X, y)\n"
           "    probe(X, y, g)\n"
           "    probe(X, y, groups=g)\n"
           "    probe(X, y, groups=None)\n"
           "    h = g if flag else None\n"
           "    probe(X, y, groups=h)\n")
    tree = ast.parse(src)
    funcs = {"probe": 2}
    got = ungrouped_calls([tree], funcs)
    assert [k[3] for k in got] == ["X", "X", "X"] and len(got) == 3
    assert "readout_from_states" in optional_group_functions() and "cg_probe" in optional_group_functions()
