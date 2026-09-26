"""Apply the frozen fallback order from
docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md to a gate-0
calibration JSON: print the FIRST hidden width, in the given order, whose row
passes gate 0 (oracle in band, mechanical leakage clean, floor within
tolerance), or NONE.

Usage: python scripts/reviewer_gaps/select_instance.py calibration.json 8 7 9 10
"""

from __future__ import annotations

import json
import sys


def select(rows: list[dict], order: list[int]) -> int | None:
    by_h = {int(r["hidden"]): r for r in rows if "hidden" in r}
    for h in order:
        r = by_h.get(h)
        if r and r.get("passes_gate0"):
            return h
    return None


def main() -> int:
    path, order = sys.argv[1], [int(x) for x in sys.argv[2:]]
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    h = select(rows, order)
    print("NONE" if h is None else h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
