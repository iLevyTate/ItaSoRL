"""Wall-clock record of the GPU organism runs, read from the local run bundles (read only).

No GPU run recorded its own duration, so the compute statement had no committed source. This
writes the best record the gitignored bundles still hold:

* start: the creation time of the run's run.log (the runner opens it before the first cell),
  or of the run directory when there is no run.log (fullruns/l3_n10);
* end: the modification time of expB2_results.json, which the runner rewrites after every cell
  and last after the final one;
* the first and last cell file times and the worker count from the run.log header;
* for the three runs a chain script launched, the chain log's own UTC timestamps for the
  organism step (start, crashes, resumes, exit), which also cross-check the file times.

File times are filesystem metadata on the machine that ran the runs (NTFS creation and
modification times); a copy of a bundle resets them, so this record is written once, from the
original bundles. Wall times include whatever else the machine did; workers and resumes are
listed so a figure is not read as a clean single-run cost.

Usage:
    python scripts/record_wall_clock.py --runs-root fullruns --out artifacts/wall_clock_gpu_runs.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

RUNS = ["l3_n10", "l3_h8_traces", "l3_h7_traces", "l3_h4_traces", "l3_h8_heldout",
        "l3_h7_heldout", "l3_h10_gseed1_gpu", "l3_h8_gseed2_gpu", "l3_h8_nowm_skill_u450"]
# Chain logs that launched a run: (log file, pattern of the organism-step lines).
CHAINS = {
    "l3_h10_gseed1_gpu": ("second_instance_gpu_chain.log", r"STEP B|RESUME step B|cells already done"),
    "l3_h8_gseed2_gpu": ("gpu_queue_chain.log", r"Q3 organism"),
    "l3_h8_nowm_skill_u450": ("q4_skill_matched_chain.log", r"stage2"),
}
METHOD = ("start = creation time of run.log (or of the run directory when there is no "
          "run.log); end = modification time of expB2_results.json; cell times are modification "
          "times of cells/cell_*.json; chain-log events are the launching chain's own UTC "
          "stamps. File times are NTFS metadata of the original bundles on the machine that "
          "ran them. Wall times include any other load on that machine.")
_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\]\s*(.*)$")
_ABS_RUNS = re.compile(r"[A-Za-z]:[\\/][^\s;]*?fullruns[\\/]")
_FREE_RAM = re.compile(r";?\s*free RAM GB [\d.]+")


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def _creation(st, os_name: str) -> tuple[float, str]:
    """(time, what it is) from a stat result. A file's creation time where the platform records
    one: st_birthtime (macOS, BSD, and Windows on Python 3.12+), or st_ctime on Windows, which is
    creation time there. Elsewhere st_ctime is the last inode change, not creation, and is
    labeled so."""
    if getattr(st, "st_birthtime", None) is not None:
        return st.st_birthtime, "creation time"
    if os_name == "nt":
        return st.st_ctime, "creation time"
    return st.st_ctime, "inode change time (ctime; this platform records no creation time)"


def _birth(path: str) -> tuple[float, str]:
    return _creation(os.stat(path), os.name)


def _workers(run_log: str) -> int | None:
    try:
        with open(run_log, encoding="utf-8", errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return None
    m = re.search(r"workers=(\d+)", head)
    return int(m.group(1)) if m else None


def chain_events(log_path: str, pattern: str) -> list[dict]:
    """[{utc, event}] for the matching lines of a chain log, machine paths made repo-relative
    and the free-RAM readings dropped."""
    out = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _STAMP.match(line.strip())
            if m and re.search(pattern, m.group(2)):
                text = _FREE_RAM.sub("", _ABS_RUNS.sub("fullruns/", m.group(2))).replace("\\", "/")
                out.append({"utc": m.group(1), "event": text.strip()})
    return out


def _hours(a: str, b: str) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (dt.datetime.strptime(b, fmt) - dt.datetime.strptime(a, fmt)).total_seconds() / 3600


def record(runs_root: str, run: str) -> dict:
    d = os.path.join(runs_root, run)
    run_log = os.path.join(d, "run.log")
    start_src = run_log if os.path.exists(run_log) else d
    start, kind = _birth(start_src)
    end = os.path.getmtime(os.path.join(d, "expB2_results.json"))
    cells = sorted(os.path.getmtime(p) for p in glob.glob(os.path.join(d, "cells", "cell_*.json")))
    row = {"run_dir": f"fullruns/{run}", "start": _iso(start),
           "start_source": (f"{kind} of run.log" if start_src == run_log
                            else f"{kind} of the run directory (no run.log)"),
           "end": _iso(end), "end_source": "modification time of expB2_results.json",
           "wall_hours": round((end - start) / 3600, 2), "n_cells": len(cells),
           "first_cell_written": _iso(cells[0]) if cells else None,
           "last_cell_written": _iso(cells[-1]) if cells else None,
           "workers_at_launch": _workers(run_log)}
    if run in CHAINS:
        log, pat = CHAINS[run]
        lp = os.path.join(runs_root, log)
        if os.path.exists(lp):
            ev = chain_events(lp, pat)
            row["chain_log"] = {"file": f"fullruns/{log}", "events": ev}
            done = [e for e in ev if re.search(r"exit 0\b", e["event"])]
            if ev and done:
                row["chain_log"]["wall_hours_first_start_to_exit_0"] = round(
                    _hours(ev[0]["utc"], done[-1]["utc"]), 2)
            row["chain_log"]["resumed"] = any(re.search(r"RESUME|RELAUNCH", e["event"]) for e in ev)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default="fullruns")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [record(a.runs_root, r) for r in RUNS if os.path.isdir(os.path.join(a.runs_root, r))]
    out = {"generated_by": "scripts/record_wall_clock.py",
           "method": METHOD,
           "runs": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        ch = r.get("chain_log", {})
        print(f"{r['run_dir']:34s} {r['wall_hours']:6.2f} h  workers {r['workers_at_launch']}"
              + (f"  chain {ch.get('wall_hours_first_start_to_exit_0')} h"
                 f"{' (resumed)' if ch.get('resumed') else ''}" if ch else ""))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
