"""Read an independent rerun of PB1 or PB2 under the frozen rule, beside the committed record.

The prospective balanced confirmation (spec
docs/specs/2026-10-20-prospective-balanced-confirmation-design.md, FINDINGS 24) is recorded in
artifacts/prospective_balanced/. A rerun of it, for example from the Colab notebook's Extra
checks, trains into its own folder with the registered command and runs the spec's matched-gates
and primary sensory-echo readouts there. This script applies the promotion code of
scripts/promote_prospective_balanced.py (its `promote`, unchanged) to that folder and sets the
result beside the committed decision.

It is a replication reading. It never calls the promoter's main(), never writes under
artifacts/, and refuses the recorded run folders. Whatever a rerun reads, the recorded decision
stays as FINDINGS 24 states it. A rerun on other hardware or libraries is not expected to match
the recorded numbers bit for bit, and the recorded floor clause held by 0.002, so a rerun can
read UNINFORMATIVE on the floor alone without contradicting the record. The second-draw reading
(S4) is descriptive in the spec and is not part of a rerun here.

    python scripts/read_pb_replication.py --run PB1 --dir fullruns/pb1_colab_replication
    python scripts/read_pb_replication.py --run PB2 --dir fullruns/pb2_colab_replication \\
        --out fullruns/pb2_colab_replication/replication_reading.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_prospective_balanced as ppb  # noqa: E402

ROOT = ppb.ROOT
COMMITTED = {r: os.path.join(ROOT, "artifacts", "prospective_balanced", f"{r.lower()}.json")
             for r in ppb.RUNS}
LABEL = ("independent replication, read with the frozen 2026-10-20 promotion code; the recorded "
         "decision (FINDINGS 24) is unchanged by this reading")


def _inside(path: str, parent: str) -> bool:
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


def _count_cells(run_dir: str) -> int:
    d = os.path.join(run_dir, "cells")
    return len([n for n in os.listdir(d) if n.endswith(".json")]) if os.path.isdir(d) else 0


def _trained(doc: dict) -> dict:
    tc = doc.get("trained_clause") or {}
    return {"resid_obs": tc.get("trained_resid_obs"), "t90": tc.get("trained_t90")}


def _floor(doc: dict) -> dict:
    fc = doc.get("floor_clause") or {}
    return {"resid_obs": fc.get("untrained_resid_obs"), "dev": fc.get("untrained_dev"),
            "holds": fc.get("holds")}


def _num(v, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _interval(t90) -> str:
    return f"[{_num(t90[0])}, {_num(t90[1])}]" if isinstance(t90, (list, tuple)) and len(t90) == 2 else ""


def _stack(stack) -> str:
    if not isinstance(stack, dict):
        return "not recorded"
    return (f"python {stack.get('python', '?')}, numpy {stack.get('numpy', '?')}, "
            f"torch {stack.get('torch', '?')} on {stack.get('device', '?')}")


def read(run_id: str, run_dir: str) -> dict:
    """The promotion of one rerun folder, with the committed values beside it."""
    run_dir = os.path.abspath(run_dir)
    committed = ppb._load(COMMITTED[run_id])
    out = {"label": LABEL, "spec": ppb.SPEC, "run": run_id,
           "dir": os.path.relpath(run_dir, ROOT) if _inside(run_dir, ROOT) else run_dir,
           "cells_found": _count_cells(run_dir), "cells_expected": len(ppb.DRIFTS) * len(ppb.SEEDS),
           "committed": {"decision": committed["decision"], "fingerprint": committed["fingerprint"],
                         "trained": _trained(committed), "floor": _floor(committed),
                         "strict": (committed.get("S1_strict") or {}).get("label"),
                         "stack": committed.get("stack")}}
    original = ppb.RUNS[run_id]["dir"]
    ppb.RUNS[run_id]["dir"] = run_dir      # promote() joins ROOT with it; an absolute path wins
    try:
        doc = ppb.promote(run_id)
    except FileNotFoundError as exc:
        doc = {"decision": "INCOMPLETE",
               "reason": f"{os.path.basename(exc.filename or str(exc))} is missing; finish the "
                         "matched-gates and sensory-echo steps, then read again"}
    except (ValueError, KeyError) as exc:
        hint = (" If the repository moved between sessions, the cells record two commits and the "
                "promotion code refuses that mix." if "more than one git_commit" in str(exc) else "")
        doc = {"decision": "INCOMPLETE", "reason": f"{type(exc).__name__}: {exc}{hint}"}
    finally:
        ppb.RUNS[run_id]["dir"] = original
    fp = doc.get("fingerprint")
    out.update(decision=doc["decision"], reason=doc.get("reason"), fingerprint=fp,
               same_configuration=(None if fp is None else fp == committed["fingerprint"]),
               trained=_trained(doc), floor=_floor(doc),
               strict=(doc.get("S1_strict") or {}).get("label"), stack=doc.get("stack"),
               gates={k: v for k, v in (doc.get("gates") or {}).items()
                      if k.startswith("gate") and k.endswith("_ok")})
    out["lines"] = lines(out)
    return out


def lines(r: dict) -> list[str]:
    c = r["committed"]
    head = [f"{r['run']} replication ({r['dir']}): {r['cells_found']} of {r['cells_expected']} cells"]
    if r["decision"] in ("NOT RUN", "INCOMPLETE"):
        return head + [f"  not readable yet: {r['reason']}",
                       f"  committed decision: {c['decision']} (FINDINGS 24)"]
    if r["same_configuration"]:
        head.append(f"  configuration: the recorded one (fingerprint {r['fingerprint']})")
    else:
        head.append(f"  configuration DIFFERS from the recorded run (fingerprint {r['fingerprint']}, "
                    f"recorded {c['fingerprint']}), so this is not a rerun of {r['run']}")
    t, ct, f, cf = r["trained"], c["trained"], r["floor"], c["floor"]
    return head + [
        f"  decision under the frozen rule: {r['decision']} ({r['reason']})",
        f"  committed decision: {c['decision']} (FINDINGS 24; this reading does not change it)",
        f"  trained resid_obs: {_num(t['resid_obs'])} {_interval(t['t90'])}"
        f"   committed {_num(ct['resid_obs'])} {_interval(ct['t90'])}",
        f"  untrained floor: {_num(f['resid_obs'])}, {_num(f['dev'])} from 0.5 against a limit of 0.1"
        f"   committed {_num(cf['resid_obs'])}, {_num(cf['dev'])} from 0.5",
        f"  strict interval reading: {r['strict']}   committed {c['strict']}",
        f"  stack: {_stack(r['stack'])}",
        f"  recorded stack: {_stack(c['stack'])}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=sorted(ppb.RUNS), required=True)
    ap.add_argument("--dir", required=True, help="the rerun's folder (cells/, matched_gates.json, sensory_echo/)")
    ap.add_argument("--out", help="also write the reading as JSON here (never under artifacts/)")
    a = ap.parse_args(argv)
    if any(_inside(a.dir, os.path.join(ROOT, r["dir"])) for r in ppb.RUNS.values()):
        ap.error(f"{a.dir} is a recorded run folder; scripts/promote_prospective_balanced.py reads it")
    if a.out and _inside(a.out, os.path.join(ROOT, "artifacts")):
        ap.error("a replication reading is never written under artifacts/")
    r = read(a.run, a.dir)
    print("\n".join(r["lines"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
            fh.write("\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
