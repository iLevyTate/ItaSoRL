"""Compare a reproduced run or readout with the record, value by value.

Every JSON file under the rerun location is matched by its relative path with the file of the
same name under the committed location. A cell file (one with a top-level "cell" block) is
compared on that block and on its recorded fingerprint; any other JSON (a readout summary, a
promoted artifact) is compared whole. Leaves are matched by their path in the document. Numbers
agree when they differ by at most --tolerance (default 0, which means bit-exact), NaN agrees
with NaN, and booleans, strings and nulls must be equal.

The report has the schema of artifacts/repro_check/per_value_4b6e1f3_l3_h8_wm_cpu_s0.json
(commit_rerun, committed_dir, stack, and per file fingerprint_rerun, fingerprint_committed,
fingerprint_match, n_shared_leaves, shared_kinds, only_in_rerun, only_in_committed, n_differ,
differences, then all_shared_values_identical), with the tolerance, the largest absolute
difference and a pass flag added. A rerun file with no committed counterpart is listed and
fails; committed files the rerun did not produce are not compared. The comparator used to live
inside scripts/audit_stats_recheck.py, which now imports it from here (G59, 2026-10-17).

Usage:
    python scripts/compare_run_cells.py --rerun DIR_OR_FILE --committed DIR_OR_FILE
        [--tolerance ABS] [--out report.json]

Exit status 0 only when every compared value agrees within the tolerance, every fingerprint
matches and no leaf or file is missing on either side.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterator


def flatten(o, p: str = "") -> Iterator[tuple[str, object]]:
    """Every leaf of a JSON document as (path, value); dict keys join with "/", list items
    with "[i]"."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, o


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def same_value(a, b, tol: float = 0.0) -> bool:
    """True when two leaves agree: NaN with NaN, numbers within `tol` (0 means equal), and
    every other kind by equality. A bool never matches a number."""
    if _is_nan(a) or _is_nan(b):
        return _is_nan(a) and _is_nan(b)
    if _is_number(a) and _is_number(b):
        return a == b or (tol > 0 and abs(a - b) <= tol)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _kind(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if _is_number(v):
        return "number"
    if v is None:
        return "null"
    return "string"


def compare_docs(rerun: dict, committed: dict, tol: float = 0.0) -> dict:
    """One file's entry of the report. Cell files compare their "cell" block and fingerprint."""
    is_cell = isinstance(rerun, dict) and isinstance(committed, dict) \
        and "cell" in rerun and "cell" in committed
    fr = dict(flatten(rerun["cell"] if is_cell else rerun))
    fc = dict(flatten(committed["cell"] if is_cell else committed))
    shared = sorted(fr.keys() & fc.keys())
    kinds: dict[str, int] = {}
    for k in shared:
        kinds[_kind(fc[k])] = kinds.get(_kind(fc[k]), 0) + 1
        if _is_nan(fc[k]):
            kinds["nan"] = kinds.get("nan", 0) + 1
    differences, n_beyond, worst = [], 0, 0.0
    for k in shared:
        a, b = fr[k], fc[k]
        if same_value(a, b):
            continue
        d = abs(a - b) if _is_number(a) and _is_number(b) and not (_is_nan(a) or _is_nan(b)) \
            else None
        within = d is not None and d <= tol
        n_beyond += 0 if within else 1
        if d is not None and math.isfinite(d):
            worst = max(worst, d)
        differences.append({"path": k, "rerun": a, "committed": b, "abs_diff": d,
                            "within_tolerance": within})
    fp_r = rerun.get("fingerprint") if is_cell else None
    fp_c = committed.get("fingerprint") if is_cell else None
    entry = {"fingerprint_rerun": fp_r, "fingerprint_committed": fp_c,
             "fingerprint_match": fp_r == fp_c,
             "n_shared_leaves": len(shared), "shared_kinds": kinds,
             "only_in_rerun": sorted(fr.keys() - fc.keys()),
             "only_in_committed": sorted(fc.keys() - fr.keys()),
             "n_differ": len(differences), "differences": differences}
    if tol > 0:
        entry["n_beyond_tolerance"] = n_beyond
        entry["max_abs_diff"] = worst
    return entry


def entry_passes(entry: dict, tol: float = 0.0) -> bool:
    beyond = entry.get("n_beyond_tolerance", entry["n_differ"]) if tol > 0 else entry["n_differ"]
    return (entry["fingerprint_match"] and not entry["only_in_rerun"]
            and not entry["only_in_committed"] and beyond == 0)


def _json_files(root: str) -> list[str]:
    """Relative paths of the JSON files under a directory (or the one file named)."""
    if os.path.isfile(root):
        return [os.path.basename(root)]
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(out)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare(rerun: str, committed: str, tol: float = 0.0) -> dict:
    """The report for every JSON file under `rerun` against its namesake under `committed`."""
    single = os.path.isfile(rerun)
    cells, missing, commits = {}, [], set()
    for rel in _json_files(rerun):
        rp = rerun if single else os.path.join(rerun, rel)
        cp = committed if single else os.path.join(committed, rel)
        if not os.path.isfile(cp):
            missing.append(rel)
            continue
        doc_r = _load(rp)
        if isinstance(doc_r, dict) and doc_r.get("git_commit"):
            commits.add(str(doc_r["git_commit"]))
        cells[rel] = compare_docs(doc_r, _load(cp), tol)
    identical = not missing and bool(cells) and all(
        e["n_differ"] == 0 and not e["only_in_rerun"] and not e["only_in_committed"]
        and e["fingerprint_match"] for e in cells.values())
    return {"commit_rerun": commits.pop() if len(commits) == 1 else sorted(commits) or None,
            "rerun_dir": rerun.replace(os.sep, "/"),
            "committed_dir": committed.replace(os.sep, "/"),
            "tolerance": tol,
            "cells": cells,
            "files_only_in_rerun": missing,
            "all_shared_values_identical": identical,
            "max_abs_diff": max((e.get("max_abs_diff", 0.0) for e in cells.values()), default=0.0),
            "pass": not missing and bool(cells) and all(entry_passes(e, tol) for e in cells.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rerun", required=True, help="reproduced cells or readouts (dir or file)")
    ap.add_argument("--committed", required=True, help="the recorded files (dir or file)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="largest absolute difference a number may show (default 0: bit-exact)")
    ap.add_argument("--out", help="write the report here (JSON)")
    a = ap.parse_args(argv)
    if a.tolerance < 0 or not math.isfinite(a.tolerance):
        ap.error("--tolerance must be a finite number >= 0")
    for p in (a.rerun, a.committed):
        if not os.path.exists(p):
            ap.error(f"not found: {p}")
    rep = compare(a.rerun, a.committed, a.tolerance)
    import _bootstrap  # noqa: F401
    from agent_free_common import stack_record
    rep["stack"] = stack_record()
    rep["generated_by"] = "scripts/compare_run_cells.py"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, indent=1)
            fh.write("\n")
    for rel, e in rep["cells"].items():
        print(f"  [{'ok  ' if entry_passes(e, a.tolerance) else 'FAIL'}] {rel}: "
              f"{e['n_shared_leaves']} shared values, {e['n_differ']} differ"
              + (f", max |diff| {e['max_abs_diff']:.3g}" if a.tolerance > 0 else "")
              + ("" if e["fingerprint_match"] else ", fingerprint differs"))
    for rel in rep["files_only_in_rerun"]:
        print(f"  [FAIL] {rel}: no committed file of that name")
    if not rep["cells"] and not rep["files_only_in_rerun"]:
        print("  [FAIL] no JSON file to compare")
    print("RESULT:", "match" if rep["pass"] else "MISMATCH",
          f"(tolerance {a.tolerance:g})")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
