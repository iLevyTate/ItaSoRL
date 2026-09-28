"""Append a dated deviation-log entry to PREREGISTRATION_L3.md section 12
(inserted before the '## 13.' heading) recording the second-instance gate-0
selection, BEFORE the organism run launches. Mechanical application of the
frozen fallback rule; the chain calls it, no human edits the numbers.

Usage: python scripts/reviewer_gaps/prereg_append.py <prereg.md> <calibration.json> <g_seed> <selected_hidden_or_NONE>
"""

from __future__ import annotations

import datetime
import json
import sys


def entry_text(rows: list[dict], g_seed: str, selected: str, today: str) -> str:
    lines = []
    for r in rows:
        lines.append(f"hidden={r['hidden']}: oracle {r['oracle_auroc']:.3f} "
                     f"(in band {r['in_band']}), mech leak {'pass' if r['mech_leak_pass'] else 'FAIL'}, "
                     f"floor {r['floor']:.3f} ({'ok' if r['floor_ok'] else 'dirty'}) -> "
                     f"{'PASS' if r['passes_gate0'] else 'fail'}")
    if selected == "NONE":
        verdict = (f"No candidate at G seed {g_seed} passed gate 0; per the frozen fallback the "
                   f"next seed is tried before any organism run.")
    else:
        verdict = (f"Selected hidden={selected} at G seed {g_seed} (first passing candidate in the "
                   f"frozen order 8, 7, 9, 10). The organism run launches with this instance; its "
                   f"result is recorded in a later entry.")
    return (f"- **{today} - SECOND FINGERPRINT INSTANCE, GATE 0 (G seed {g_seed}; spec "
            f"`docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md`; recorded by the "
            f"run chain mechanically BEFORE launch).** Calibration on world P at the frozen "
            f"sigma=0.02: " + "; ".join(lines) + f". {verdict}\n\n")


def main() -> int:
    prereg, calib, g_seed, selected = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    with open(calib, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    entry = entry_text(rows, g_seed, selected, datetime.date.today().isoformat())
    with open(prereg, encoding="utf-8") as fh:
        s = fh.read()
    marker = "## 13. How to run"
    i = s.index(marker)
    with open(prereg, "w", encoding="utf-8") as fh:
        fh.write(s[:i] + entry + s[i:])
    print("appended:", entry[:160], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
