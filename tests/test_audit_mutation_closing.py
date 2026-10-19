"""Each closing-pass audit block can fail (runbook Stage 6.2, R3-9).

One test per block of scripts/audit_stats_recheck.py that binds a Stage 6 result: B1 to B8, C1,
G01, G02, G03, G04, G07, G47, G57, H31, H53, H01, the H76 steepness checks and the X03, X07, X08 and
X09 blocks. Each test first runs its block on the committed artifacts and records the files the block
opens (the block must pass there), copies exactly those files into a temporary artifacts root, moves
one per-seed value in the tmp copy (for the agent-free X08 and X09 blocks, one value of a committed
input), points the audit at the tmp root and asserts that the block records a failure, and that every
failure is inside that block. Nothing here reads fullruns/.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_stats_recheck as asr  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_counters():
    """The audit keeps its tally in module globals; give every test a clean slate."""
    before = (asr.n_checks, asr.n_skipped, asr.n_pins, list(asr.failures))
    asr.n_checks, asr.n_skipped, asr.n_pins, asr.failures[:] = 0, 0, 0, []
    yield
    asr.n_checks, asr.n_skipped, asr.n_pins, asr.failures[:] = before[0], before[1], before[2], before[3]


def _first(rows, **match):
    return next(r for r in rows if all(r.get(k) == v for k, v in match.items()))


def _bump(container, key, by: float = 0.05) -> None:
    container[key] = float(container[key]) + by


# item tag -> (block, file to move inside the tmp root, mutation of the parsed JSON)
CASES = {
    "B1": (asr.closing_b1, "expL1/l1_coarse_grid_corrected.json",
           lambda d: _bump(d["untrained_d0.0600_per_seed"], 0)),
    "B2": (asr.closing_b2, "walk_speed/corrected_l3_h8_wm.json",
           lambda d: _bump(_first(d["rows"], seed=0)["arms"]["survival"]["own"], "target")),
    "B3": (asr.closing_b3, "l0_audit/reverse_direction_corrected_l3_h8_wm.json",
           lambda d: _bump(_first(d["cells"], arm="survival", seed=0), "target")),
    "B4": (asr.closing_b4, "population_readout/corrected_l3_h8_wm_world_grouped.json",
           lambda d: _bump(d["panels"]["survival"]["per_individual"], 0)),
    "B5": (asr.closing_b5, "goal_stakes_grouped_surprise/C1.json",
           lambda d: _bump(_first(d["per_seed"], seed=0)["surprise"], "auroc")),
    "B6": (asr.closing_b6, "development/heldout_transfer.json",
           lambda d: _bump(d["runs"]["wm"]["summary"]["450"]["heldout_transfer_from_300"]["per_seed"], 0)),
    "B7": (asr.closing_b7, "expL1/l1_h2_noise_per_episode.json",
           lambda d: _bump(d["survival_per_seed"], 0)),
    "B8": (asr.closing_b8, "texture/ar_sweep_c_realization2/C_rho0.99.json",
           lambda d: _bump(d["aggregate"]["survival fresh"]["per_seed"], 0)),
    "C1": (asr.closing_c1, "expB_aligned_2026-10-18/expB_full.json",
           lambda d: _bump(_first(d["drift_sweep"], drift=0.45)["per_seed"]["target"], 0)),
    "G01": (asr.closing_g01, "probe_sensitivity/corrected_l3_h8_wm_regenerated.json",
            lambda d: _bump(_first(d["cells"], drift="0.00", arm="survival", seed=0)["grid"]["0.001"], "target")),
    "G02": (asr.closing_g02, "l0_audit/d045_world_samples_corrected_l3_h8_wm.json",
            lambda d: _bump(_first(d["cells"], seed=0)["rows"][1], "target")),
    "G03": (asr.closing_g03, "balanced_confirmation/sensory_echo_b1_fresh.json",
            lambda d: _bump(d["d=0.45 survival"]["resid_obs"]["per_seed"], 0)),
    "G04": (asr.closing_g04, "step_resolved/l3_h8_heldout.json",
            lambda d: _bump(_first(d["cells"], drift="0.45", arm="survival", seed=0)["steps"]["1"], "target")),
    "G07": (asr.closing_g07, "world_features/own_episode_posthoc/corrected_l3_h8_wm.json",
            lambda d: _bump(_first(d["rows"], seed=0)["arms"]["survival_d0.45"]["I1_nearest_pellet"], "r2_resid")),
    # Agent-free and stored without per-seed values: the oracle reading at the registered grid.
    "G47": (asr.closing_g47, "fold_rescore/gate0_l1_explicit.json",
            lambda d: _bump(_first(d["deltas_sweep"], delta=0.023), "oracle_auroc")),
    "G57": (asr.closing_g57, "repro_check_c1_regen/c1_regen_local_stack.json",
            lambda d: _bump(d["cells"][0], "target")),
    "H31": (asr.closing_h31, "agent_free/qd_eps6_oracle.json",
            lambda d: _bump(_first(d["rows"], eps=6.0)["floor_per_seed"], 0)),
    "H53": (asr.closing_h53, "expL1/l1_nonlinear_corrected.json",
            lambda d: _bump(_first(d["cells"], drift="0.0230", arm="survival", seed=0)["classes"]["rf"], "target")),
    "X03": (asr.closing_x03, "head_overlap/corrected_l3_h8_wm.json",
            lambda d: _bump(_first(d["rows"], arm="survival", seed=0)["overlap"]["actor"], "ratio_u")),
    "X07": (asr.closing_x07, "pellet_dose/corrected_l3_h8_wm.json",
            lambda d: _bump(_first(d["rows"], kind="arm", arm="survival", n_pellets=2, seed=0), "target")),
    # X08 and X09 recompute from their committed inputs, so the moved value is an input's.
    "X08": (asr.closing_x08, "budget_curve.json",
            lambda d: _bump(_first(d["series"]["decoder on"], updates=100)["target"]["per_seed"], 0)),
    "X09": (asr.closing_x09, "corrected_verdicts.json",
            lambda d: _bump(d["runs"]["C1"]["drift_zero"]["survival"]["per_seed"], 0)),
}


def _opened_by(block, monkeypatch) -> list[str]:
    """Run the block on the committed artifacts; return the artifact files it opened."""
    art = os.path.normcase(os.path.abspath(asr.ARTROOT)) + os.sep
    seen: set[str] = set()
    real_open = builtins.open

    def recording_open(file, *a, **k):
        if isinstance(file, (str, os.PathLike)):
            p = os.path.normcase(os.path.abspath(os.fspath(file)))
            if p.startswith(art):
                seen.add(p[len(art):])
        return real_open(file, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", recording_open)
        block()
    return sorted(seen)


@pytest.mark.parametrize("tag", sorted(CASES))
def test_closing_block_fails_when_one_value_moves(tag, tmp_path, monkeypatch, capsys):
    block, rel, mutate = CASES[tag]
    opened = _opened_by(block, monkeypatch)
    assert asr.failures == [], (tag, asr.failures)
    assert asr.n_checks > 0
    target = os.path.normcase(os.path.normpath(rel))
    assert target in opened, (tag, opened)
    root = tmp_path / "artifacts"
    for r in opened:
        dst = root / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(asr.ARTROOT, r), dst)
    with open(root / target, encoding="utf-8") as fh:
        doc = json.load(fh)
    mutate(doc)
    with open(root / target, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    asr.n_checks, asr.failures[:] = 0, []
    monkeypatch.setattr(asr, "ARTROOT", str(root))
    block()
    capsys.readouterr()
    assert asr.failures, f"{tag}: a moved value in {rel} must make the block fail"
    assert all(f"6.2 {tag}" in f for f in asr.failures), asr.failures


def _h76_note(sw: dict) -> str:
    """A section-13 note carrying exactly the quotes the five H76 checks look for."""
    pc = next(c for c in sw["cells"] if c["name"] == "r0.25_h30")
    return (f"best raw gap {sw['best_by_gap']['gap']:.5f}, {sw['best_by_gap']['ratio_to_baseline']:.1f} times "
            f"the baseline; best normalized {sw['best_by_norm_gap']['norm_gap']:.4f}; pilot cell "
            f"{pc['norm_gap']:.4f}; artifacts/expC/gate1_steepness_sweep.json, seeds 7100-7105")


def test_h76_steepness_checks_fail_when_the_best_normalized_gap_moves():
    with open(os.path.join(asr.ARTROOT, "expC", "gate1_steepness_sweep.json"), encoding="utf-8") as fh:
        sw = json.load(fh)
    note = _h76_note(sw)
    asr._h76_steepness(sw, note)
    assert asr.failures == [] and asr.n_checks == 5, asr.failures
    sw["best_by_norm_gap"]["norm_gap"] = float(sw["best_by_norm_gap"]["norm_gap"]) + 0.05
    asr._h76_steepness(sw, note)
    assert asr.failures == ["13 sweep note quotes the best normalized cell"], asr.failures


def test_full_audit_fails_inside_the_h01_block_when_one_pooled_value_moves(monkeypatch, capsys):
    """H01's block (62ecfba) lives inside main, so this runs the whole audit once (about half a minute)."""
    target = os.path.normcase(os.path.abspath(os.path.join(asr.ARTROOT, "balanced_pooled", "registered_pair.json")))
    real_load = json.load
    seen = []

    def mutating_load(fh, *a, **k):
        doc = real_load(fh, *a, **k)
        caller = os.path.basename(sys._getframe(1).f_code.co_filename)
        if (os.path.normcase(os.path.abspath(getattr(fh, "name", ""))) == target
                and caller == "audit_stats_recheck.py"):
            seen.append(1)
            cell = doc["runs"]["B1"]["pool_target"]["0.45"]["survival"]
            cell["per_seed"][0] = float(cell["per_seed"][0]) + 0.05
        return doc

    monkeypatch.setattr(json, "load", mutating_load)
    assert asr.main() == 1
    capsys.readouterr()
    assert seen, "the audit did not read the registered-pair artifact"
    assert "H01 B1 d0.45 survival pooled mean recomputed" in asr.failures
    assert all(f.startswith(("H01 ", "pinned: H01 ")) for f in asr.failures), asr.failures
