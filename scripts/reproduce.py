"""One entry point for reproducing the results (revision step 15; docs/REPRODUCE.md).

  tables       No training. Rebuild the results manifest, the gate table, and the contrast
               intervals from committed per-seed artifacts, then run the scalar audit. With
               --dumps DIR (state dumps from a run or the supplement), also recompute every
               per-seed readout from the raw recurrent states under both fold partitions
               (scripts/rescore_fold_split.py) and the behavior control
               (scripts/audit_behavior_mediation.py). Minutes on a laptop.
  retrain RUN  Print, or with --execute run, the exact commands that retrain a run listed in
               docs/RESULTS_MANIFEST.md. Hours of CPU per run (estimates in docs/REPRODUCE.md).
  supplement   Build an anonymized submission archive under dist/: code, tests, committed
               artifacts, the method documents, and (when present) the corrected runs' cells,
               saved agents and state dumps, with author names, handles, e-mail addresses and
               local paths scrubbed from text files and a SHA-256 list of every file.

Usage:
    python scripts/reproduce.py tables [--dumps fullruns/corrected_l3_h8_wm/states]
    python scripts/reproduce.py retrain C1 [--execute]
    python scripts/reproduce.py supplement [--out dist/itasorl_supplement.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable

L3 = "python scripts/run_expB2.py --drift-mode l3 --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --device cpu"
RETRAIN = {
    "C1": ["bash scripts/revision/run_corrected_confirmation.sh . 4   # runs C1 then C2"],
    "C2": ["bash scripts/revision/run_corrected_confirmation.sh . 4   # runs C1 then C2"],
    "L3-H8-WM-CPU": [f"{L3} --l3-hidden 8 --gae-bootstrap pre_transition --workers 3 "
                     "--save-agents --dump-states fullruns/l3_h8_wm_cpu/states --out-dir fullruns/l3_h8_wm_cpu"],
    "L3-H8-NOWM-CPU": [f"{L3} --l3-hidden 8 --no-world-model --gae-bootstrap pre_transition --workers 3 "
                       "--save-agents --dump-states fullruns/l3_h8_nowm/states --out-dir fullruns/l3_h8_nowm"],
    "L3-H10-GS1-CPU": [f"{L3} --l3-hidden 10 --l3-seed 1 --gae-bootstrap pre_transition --workers 3 "
                       "--save-agents --dump-states fullruns/l3_h10_gseed1/states --out-dir fullruns/l3_h10_gseed1"],
    "L3-H8-N10": [f"{L3.replace('--device cpu', '--device cuda')} --l3-hidden 8 --gae-bootstrap "
                  "pre_transition --dump-states fullruns/l3_h8_traces/states --out-dir fullruns/l3_h8_traces"
                  "   # published on GPU; ITASORL_FOLDS=legacy on the publishing stack reproduces 0.752"],
}
READOUTS = [
    "python scripts/run_l0_audit.py --agents-dir {run}/agents --out artifacts/l0_audit/{name}.json",
    "python scripts/run_policy_controlled_readouts.py --run-dir {run} --out artifacts/policy_controls/{name}.json --workers 4",
    "python scripts/run_persistence_readout.py --agents-dir {run}/agents --out artifacts/persistence/{name}.json --workers 4",
    "python scripts/run_control_diagnostics.py --run-dir {run} --out artifacts/control_diagnostics/{name}.json --workers 4",
    "python scripts/run_texture_fresh_probe.py --agents-dir {run}/agents --family gn --param 0.01 --out artifacts/texture/{name}_gn.json --workers 4",
    "python scripts/validate_population_readout.py --run-dir {run} --out artifacts/population_readout/{name}.json",
]


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def tables(dumps: list[str]) -> int:
    rc = 0
    for script in ("build_results_manifest.py", "build_gate_table.py", "build_contrast_intervals.py"):
        rc |= _run([PY, os.path.join("scripts", script)])
    rc |= _run([PY, os.path.join("scripts", "audit_stats_recheck.py")])
    for d in dumps:
        rc |= _run([PY, os.path.join("scripts", "rescore_fold_split.py"), d])
        rc |= _run([PY, os.path.join("scripts", "audit_behavior_mediation.py"), d])
    return rc


def retrain(run: str, execute: bool) -> int:
    cmds = RETRAIN.get(run)
    if cmds is None:
        print(f"no retraining recipe recorded for {run!r}; known: {sorted(RETRAIN)}")
        print("(every run's configuration is in docs/RESULTS_MANIFEST.md)")
        return 1
    run_dir = {"C1": "fullruns/corrected_l3_h8_wm", "C2": "fullruns/corrected_l3_h8_nowm"}.get(run)
    for c in cmds:
        print(c)
    if run_dir:
        print("\n# readout-only follow-ups on the saved agents (revision steps 5 to 12):")
        for r in READOUTS:
            print(r.format(run=run_dir, name=os.path.basename(run_dir)))
    if execute:
        rc = 0
        for c in cmds:
            rc |= subprocess.call(c.split("   #")[0], shell=True, cwd=ROOT)
        return rc
    return 0


# Text scrubbing for the anonymized supplement. Identity terms are read at build time from
# CITATION.cff (excluded from the archive) and the git remote, so this file names no one.
GENERIC_SCRUB = [
    (re.compile(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s\"']+", re.I), "<local-path>"),
    (re.compile(r"/home/[^/\s\"']+"), "<local-path>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"orcid\.org/[\d-]+X?", re.I), "orcid.org/<orcid>"),
]


def identity_terms() -> list[str]:
    terms = []
    cff = os.path.join(ROOT, "CITATION.cff")
    if os.path.exists(cff):
        for line in open(cff, encoding="utf-8"):
            m = re.match(r"\s*-?\s*(family-names|given-names|alias|name):\s*\"?([^\"]+)\"?", line)
            if m and m.group(1) != "name":
                terms.append(m.group(2).strip())
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip()
        m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
        if m:
            terms.append(m.group(1))
    except Exception:
        pass
    return sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True)


def scrub_rules() -> list:
    """Generic rules plus one rule per identity term. A term matches only as a whole word
    and with its own capitalization (a case-insensitive "Tate" would also hit "state");
    a mixed-case handle (e.g. a GitHub owner) also matches in lower case, as in URLs."""
    rules = list(GENERIC_SCRUB)
    for t in identity_terms():
        pats = [t] + ([t.lower()] if t != t.lower() and t != t.capitalize() else [])
        for v in pats:
            rules.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![a-z0-9])"), "<author>"))
    return rules


TEXT_EXT = {".py", ".md", ".json", ".txt", ".sh", ".toml", ".ini", ".cfg", ".log", ".yml", ".yaml"}
INCLUDE_DIRS = ["itasorl", "scripts", "tests", "artifacts"]
INCLUDE_FILES = ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "pytest.ini",
                 "ruff.toml", "LICENSE",
                 "docs/FINDINGS.md", "docs/CORRECTIONS.md", "docs/REVISION_2026-10.md",
                 "docs/RESULTS_MANIFEST.md", "docs/GATE_TABLE.md", "docs/CONTRAST_INTERVALS.md",
                 "docs/METHODS_ARMS.md", "docs/REPRODUCE.md", "docs/PAPER_OUTLINE.md",
                 "docs/PREREGISTRATION.md", "docs/PREREGISTRATION_Bv3.md",
                 "docs/PREREGISTRATION_L3.md", "docs/PREREGISTRATION_C.md",
                 "docs/ITASORL_world_spec.md", "docs/ITASORL.md", "docs/LEARNING.md",
                 "docs/STATUS_2026-09-27.md", "docs/AUDIT_2026-07.md",
                 # read by the scalar audit's wording guards and the site check:
                 "README.md", "CITATION.cff", "index.html", "index.template.html",
                 "viz/player/brain/brain.js"]
RUN_DIRS = ["fullruns/corrected_l3_h8_wm", "fullruns/corrected_l3_h8_nowm"]


def scrub(text: str, rules=None) -> str:
    for pat, rep in rules if rules is not None else scrub_rules():
        text = pat.sub(rep, text)
    return text


def _files() -> list[str]:
    out = []
    for d in INCLUDE_DIRS + ["docs/specs"] + [r for r in RUN_DIRS if os.path.isdir(os.path.join(ROOT, r))]:
        for base, dirs, files in os.walk(os.path.join(ROOT, d)):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in files:
                out.append(os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/"))
    out += [f for f in INCLUDE_FILES if os.path.exists(os.path.join(ROOT, f))]
    return sorted(set(out))


def supplement(out: str) -> int:
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)) or ".", exist_ok=True)
    sums = []
    rules = scrub_rules()
    with zipfile.ZipFile(os.path.join(ROOT, out), "w", zipfile.ZIP_DEFLATED) as z:
        for rel in _files():
            with open(os.path.join(ROOT, rel), "rb") as fh:
                data = fh.read()
            if os.path.splitext(rel)[1] in TEXT_EXT:
                data = scrub(data.decode("utf-8", errors="replace"), rules).encode("utf-8")
            z.writestr(rel, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
        readme = ("Anonymized supplement. Start with docs/REPRODUCE.md.\n"
                  "python scripts/reproduce.py tables   regenerates the tables and gates without training.\n"
                  "python scripts/reproduce.py retrain C1   prints the retraining commands.\n")
        z.writestr("README_SUPPLEMENT.md", readme)
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    leaks = []
    with zipfile.ZipFile(os.path.join(ROOT, out)) as z:
        for n in z.namelist():
            if os.path.splitext(n)[1] in TEXT_EXT:
                t = z.read(n).decode("utf-8", errors="replace")
                if any(p.search(t) for p, _ in rules):
                    leaks.append(n)
    print(f"wrote {out}: {len(sums)} files" + (f"; POSSIBLE IDENTITY LEAKS in {leaks}" if leaks else "; no identity strings found"))
    return 1 if leaks else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--dumps", nargs="*", default=[])
    r = sub.add_parser("retrain")
    r.add_argument("run")
    r.add_argument("--execute", action="store_true")
    s = sub.add_parser("supplement")
    s.add_argument("--out", default="dist/itasorl_supplement.zip")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        return tables(a.dumps)
    if a.cmd == "retrain":
        return retrain(a.run, a.execute)
    return supplement(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
