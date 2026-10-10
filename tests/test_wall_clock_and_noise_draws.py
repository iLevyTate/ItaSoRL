"""Two provenance records added 2026-10-10 for numbers the compute and matched-handicap
passages quote: the GPU runs' wall clocks (artifacts/wall_clock_gpu_runs.json) and the
detector-noise draws of the in-configuration oracle (artifacts/expA/l2_inconfig_oracle.json).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import record_wall_clock as rwc  # noqa: E402

WALL = ROOT / "artifacts" / "wall_clock_gpu_runs.json"
ORACLE = ROOT / "artifacts" / "expA" / "l2_inconfig_oracle.json"


def _runs():
    return {r["run_dir"]: r for r in json.loads(WALL.read_text(encoding="utf-8"))["runs"]}


def _utc(stamp: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return t.astimezone(dt.timezone.utc)


def test_chain_events_drop_machine_paths_and_ram_readings(tmp_path):
    log = tmp_path / "chain.log"
    profile = "C:/" + "Us" + "ers/someone/repo"       # built from pieces: no literal profile path
    log.write_text(f"[2026-09-28T21:14:16Z] STEP B organism run -> {profile}/fullruns/r1\n"
                   "[2026-09-28T21:14:17Z] unrelated line\n"
                   "[2026-09-29T02:58:26Z] STEP B exit 0; free RAM GB 4.5\n", encoding="utf-8")
    ev = rwc.chain_events(str(log), r"STEP B")
    assert ev == [{"utc": "2026-09-28T21:14:16Z", "event": "STEP B organism run -> fullruns/r1"},
                  {"utc": "2026-09-29T02:58:26Z", "event": "STEP B exit 0"}]


def test_the_wall_clock_record_names_its_method_and_covers_the_gpu_runs():
    doc = json.loads(WALL.read_text(encoding="utf-8"))
    assert doc["generated_by"] == "scripts/record_wall_clock.py" and "NTFS" in doc["method"]
    runs = _runs()
    assert {f"fullruns/{r}" for r in rwc.RUNS} == set(runs)
    for r in runs.values():
        assert r["n_cells"] == 20 and r["wall_hours"] > 0
        assert _utc(r["start"]) < _utc(r["first_cell_written"]) <= _utc(r["last_cell_written"])


def test_chain_logs_corroborate_the_file_times_within_a_minute():
    for run, r in _runs().items():
        ch = r.get("chain_log")
        if not ch:
            continue
        start = _utc(ch["events"][0]["utc"])
        end = _utc([e for e in ch["events"] if e["event"].endswith("exit 0")][-1]["utc"])
        assert abs((start - _utc(r["start"])).total_seconds()) <= 60, run
        assert abs((end - _utc(r["end"])).total_seconds()) <= 60, run


def test_the_wall_times_the_compute_paragraph_can_quote():
    r = _runs()
    assert [r[f"fullruns/{k}"]["wall_hours"] for k in ("l3_n10", "l3_h8_traces", "l3_h8_heldout")] \
        == [5.83, 7.91, 6.89]
    gpu = r["fullruns/l3_h10_gseed1_gpu"]
    assert gpu["wall_hours"] == 5.74 and gpu["chain_log"]["resumed"] is True
    assert any("exit 127" in e["event"] for e in gpu["chain_log"]["events"])
    assert r["fullruns/l3_h8_gseed2_gpu"]["wall_hours"] == 6.53


def test_every_noise_draw_block_recomputes_and_its_seed_zero_is_the_tabulated_row():
    d = json.loads(ORACLE.read_text(encoding="utf-8"))
    assert d["generated_by"] == "scripts/run_expA_l2_inconfig.py"
    assert d["fold_scheme"] == "legacy"
    rows = {(r["cell"], r["sigma_meas"]): r["oracle_auroc"] for r in d["rows"]}
    draws = d["noise_seed_draws"]
    assert set(draws) == {"l0", "ar1", "regime", "l3_h8_seed0", "l3_h7_seed0"}
    for cell, b in draws.items():
        v = b["oracle_auroc"]
        assert b["seeds"] == list(range(8)) and b["sigma_meas"] == d["sigma_frozen"]
        assert abs(np.mean(v) - b["mean"]) < 1e-12 and abs(np.std(v, ddof=1) - b["sd"]) < 1e-12
        assert v[0] == rows[(cell, d["sigma_frozen"])], cell


def test_the_l0_anchor_block_the_record_quotes():
    d = json.loads(ORACLE.read_text(encoding="utf-8"))
    a, b = d["l0_anchor_noise_seeds"], d["noise_seed_draws"]["l0"]
    assert a == {"seeds": b["seeds"], "oracle_auroc": b["oracle_auroc"], "mean": b["mean"]}
    assert round(a["mean"], 3) == 0.475
    assert [round(x, 3) for x in (min(a["oracle_auroc"]), max(a["oracle_auroc"]))] == [0.394, 0.581]


def test_the_start_time_is_labeled_by_what_the_platform_records():
    """st_ctime is creation time on Windows but the last inode change on Linux, so a record
    built there must not call it a creation time."""
    from types import SimpleNamespace
    assert rwc._creation(SimpleNamespace(st_birthtime=3.0, st_ctime=9.0), "posix") == (3.0, "creation time")
    assert rwc._creation(SimpleNamespace(st_ctime=9.0), "nt") == (9.0, "creation time")
    t, kind = rwc._creation(SimpleNamespace(st_ctime=9.0), "posix")
    assert t == 9.0 and kind.startswith("inode change time") and "creation" in kind
    t, kind = rwc._birth(__file__)
    assert t > 0 and isinstance(kind, str)
