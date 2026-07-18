"""Faithfulness pins for scripts/reanalyze_cg_states.py (the offline cg re-score).

The script's whole value is that its numbers EQUAL what the fixed estimator would
have produced in-run, so the load-bearing contracts are:
  - rescore_cell(path, seed) == cg_probe(auth, surr, late_k=8, seed=seed) on the
    same arrays - same default late_k, same per-cell probe seed, same features
    (the seed is parsed from the states_d<drift>_s<seed>_<agent>_cg.npz filename);
  - the discovery path only picks up *_cg.npz dumps - the sibling
    *_h7transfer.npz / pooled-readout .npz files written into the same states/
    dir by run_expB2.py must never be scored (their keys differ; scoring one
    would crash or, worse, emit a bogus cell).
All synthetic, CPU-only, seconds."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

reanalyze = importlib.import_module("reanalyze_cg_states")

from itasorl.experiment_b2 import cg_probe  # noqa: E402


def _synthetic_tails(seed: int = 42, n_pairs: int = 17, tail_steps: int = 12, hidden: int = 6):
    """Tails with a DECAYING surrogate offset, so the late-window AUROC genuinely
    depends on the window size - any late_k drift then changes the numbers. n_pairs
    is deliberately not a multiple of 5 (the regime the pair-grouping fix targets)."""
    rng = np.random.default_rng(seed)
    auth = rng.normal(0.0, 1.0, (n_pairs, tail_steps, hidden)).astype(np.float32)
    surr = rng.normal(0.0, 1.0, (n_pairs, tail_steps, hidden)).astype(np.float32)
    surr[:, :, 0] += np.linspace(2.0, 0.2, tail_steps)[None, :]
    return auth, surr


def _write_dump(states_dir: Path, auth: np.ndarray, surr: np.ndarray,
                name: str = "states_d0.45_s3_survival_cg.npz") -> Path:
    """Mirror run_expB2.evaluate_agent's cg dump: keys auth/surr, (n, T, hidden)."""
    states_dir.mkdir(parents=True, exist_ok=True)
    path = states_dir / name
    np.savez_compressed(path, auth=auth, surr=surr)
    return path


def test_rescore_cell_matches_direct_cg_probe(tmp_path):
    """rescore_cell must equal a direct cg_probe call with the run convention:
    default late_k=8 and the per-cell seed. Exact float equality - both paths run
    the identical deterministic estimator on the identical arrays, so ANY drift in
    late_k, feature construction, or seed threading shows up as a mismatch (the
    synthetic data makes cg_latetail_target and the bootstrap CI seed-/window-
    sensitive by construction)."""
    auth, surr = _synthetic_tails()
    path = _write_dump(tmp_path, auth, surr)
    got = reanalyze.rescore_cell(str(path), 3)
    want = cg_probe(list(auth), list(surr), late_k=8, seed=3)
    assert got == want
    # the equality above has teeth: a different window or probe seed changes the dict
    assert cg_probe(list(auth), list(surr), late_k=4, seed=3) != want
    assert cg_probe(list(auth), list(surr), late_k=8, seed=0) != want


def test_filename_convention_parses_drift_seed_agent():
    """The per-cell probe seed comes from the filename; the regex must keep parsing
    the exact dump names run_expB2.evaluate_agent writes (non-greedy agent token so
    'survival_cg' yields agent='survival')."""
    m = reanalyze.FNAME.search("states_d0.45_s3_survival_cg.npz")
    assert m is not None
    assert m["drift"] == "0.45" and int(m["seed"]) == 3 and m["agent"] == "survival"
    assert reanalyze.FNAME.search("states_d0.45_s3_survival_h7transfer.npz") is None
    assert reanalyze.FNAME.search("states_d0.45_s3_survival.npz") is None


def test_main_scores_only_cg_dumps_and_matches_probe(tmp_path, monkeypatch):
    """End-to-end through main(): a states/ dir holding one real cg dump plus the
    sibling decoys run_expB2 writes alongside it (states_*_h7transfer.npz and the
    pooled states_*.npz, whose keys would crash rescore_cell) must yield EXACTLY one
    scored cell, parsed from the filename and numerically equal to the direct
    cg_probe call."""
    auth, surr = _synthetic_tails()
    _write_dump(tmp_path, auth, surr)
    np.savez_compressed(tmp_path / "states_d0.45_s3_survival_h7transfer.npz",
                        Ha2=auth, H7=surr)
    np.savez_compressed(tmp_path / "states_d0.45_s3_survival.npz", Ha=auth, Hs=surr)

    out = tmp_path / "rescore.json"
    monkeypatch.setattr(sys, "argv",
                        ["reanalyze_cg_states.py", str(tmp_path), "--json", str(out)])
    reanalyze.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    cells = payload["cells"]
    assert len(cells) == 1, f"decoy npz files leaked into the re-score: {cells}"
    cell = cells[0]
    assert (cell["drift"], cell["seed"], cell["agent"]) == ("0.45", 3, "survival")
    want = cg_probe(list(auth), list(surr), late_k=8, seed=3)
    for k, v in want.items():
        assert cell[k] == v, f"re-scored {k}={cell[k]} != direct cg_probe {v}"
