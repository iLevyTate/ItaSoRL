"""Manuscript tables generated from the checked results, and a manuscript check (revision step 15).

Writes LaTeX tables from the committed, audited artifacts (the gate table, the contrast
intervals, and the corrected-run verdicts), so the manuscript can \\input them instead of
retyping numbers. The tables are manuscript material and stay LOCAL: they are written to
docs/paper_tables/, which .gitignore excludes, beside (never inside) the gitignored manuscript
directory docs/paper/, so generating them cannot overwrite the author's own files. The scalar
audit renders them in memory from the committed artifacts, so CI needs no paper files.

With --manuscript DIR (the LaTeX source, also local and outside git) it checks that at least
one generated table is \\input and every such \\input refers to a current generated file,
that every headline number of artifacts/corrected_verdicts.json is quoted somewhere, and it
flags wording the 2026-10 revision retired (the claim table in docs/REVISION_2026-10.md;
matched after LaTeX markup is stripped) and any pre-correction headline that appears without a
historical label in its section. A sentence that withdraws a retired reading may quote it, and
a historical label anywhere in the enclosing sectioning block covers the numbers inside it. It
reports; the author decides each flagged line.

Usage:
    python scripts/build_paper_tables.py                     # write docs/paper_tables/*.tex (local)
    python scripts/build_paper_tables.py --check             # exit 1 if a local table is stale
    python scripts/build_paper_tables.py --manuscript docs/paper
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "docs", "paper_tables")   # gitignored; beside, not inside, docs/paper/

# Retired wording (docs/REVISION_2026-10.md, step 14). A hit is a line to rewrite, not an error
# by itself: a sentence quoting the old wording to retire it is fine.
RETIRED = [
    (r"behaviou?r[- ]independent", "name the controlled features and the residualization model"),
    (r"\bnot encoded\b", "did not meet the registered encoding criterion"),
    (r"neither objective alone(?![^.]*(budget|updates))", "state the tested budget and protocol"),
    (r"persistent (internal|stored)", "prefix condition remains decodable after restoring authentic dynamics"),
    (r"learned[- ]texture (mechanism|specific)", "only after the comparator experiments"),
    (r"heritable detector", "restrict to the pooled population readout and budget"),
    (r"every decision (was|is) pre-?registered", "separate prospective protocols, amendments, exploration, corrections"),
    (r"every number (recomputes|is recomputed)(?![^.]*\b(document|inventory|findings)\b)",
     "say which artifacts and documents are checked"),
    (r"(live|lives|living) in both worlds", "each cell trains in one condition and is evaluated in two"),
    (r"never (enters|in) the observation", "no explicit world label is supplied"),
    (r"authentic manifold", "reachability is not established"),
    (r"\bHDI\b", "percentile bootstrap interval"),
    (r"posterior probability", "share of bootstrap means inside the ROPE"),
    (r"at the agent'?s resolution", "describe the common detector-side handicap"),
    (r"early[- ]stop", "the surrogate trains for a fixed 300 epochs"),
    (r"sees raycasts, not velocity", "interoception carries velocity"),
    (r"survival[- ]specific", "at matched input the predictor reads equally (FINDINGS 17.6); say training regimes"),
    (r"(encoded|uniquely) by the survival objective", "the signal rides on the foraging trajectories"),
    (r"texture[- ]specific", "coherent hand-written drag is read too (FINDINGS 17.9)"),
    # Added by the 2026-10-07 revision audit (item 2):
    (r"stored component", "the common garden does not separate memory from the prefix's footprint"),
    (r"\bpersists\b", "retention under identical input is not shown (FINDINGS 17.7)"),
    (r"modestly persistent", "remains decodable after restoring authentic dynamics"),
    (r"loads on the learned texture", "temporally coherent, state-dependent deviation; nothing specific to learned dynamics"),
    (r"which objective does the encoding", "at matched input it is not a difference of objective"),
    (r"property of the two objectives", "a difference between training regimes"),
    (r"not incidentally encoded", "did not meet the registered encoding criterion"),
    (r"did not encode\b", "did not meet the registered encoding criterion"),
    (r"encoded neither", "did not meet the registered encoding criterion"),
    (r"every number reported here is recomputed", "say which artifacts and documents are checked"),
    (r"recomputes every number(?![^.]*\b(document|inventory|findings)\b)",
     "say which artifacts and documents are checked"),
]

# Headline numbers of the pre-correction trainer that may appear only with a historical label.
# The label counts when it appears anywhere in the enclosing sectioning block (a paragraph
# headed "The historical headline" labels every number inside it), not only on the same line.
HISTORICAL_HEADLINES = ("0.752",)       # L3-H8-N10, the published GPU headline
HISTORICAL_LABEL = re.compile(r"historical|pre-?correction|labeled", re.I)
SECTIONING = re.compile(r"^\\(?:sub)*(?:section|paragraph)\*?\{")

# A sentence that withdraws a retired reading may quote it (the RETIRED note above). These
# markers, appearing in the same sentence before the phrase, mean the line is already correct.
WITHDRAWN = re.compile(r"does not support|do not support|is withdrawn|are withdrawn|no longer|"
                       r"we do not claim|is not claimed|retired in favor|withdrawn in favor", re.I)


def _block_starts(lines: list[str]) -> list[int]:
    """For each line, the index where its sectioning block begins."""
    out, start = [], 0
    for i, line in enumerate(lines):
        if SECTIONING.match(line):
            start = i
        out.append(start)
    return out


def _withdrawn_before(text: str, at: int) -> bool:
    """True when the sentence containing position `at` withdraws the phrase before quoting it."""
    before = text[:at]
    cut = max(before.rfind(". "), before.rfind("! "), before.rfind("? "))
    return bool(WITHDRAWN.search(before[cut + 1:] if cut >= 0 else before))


def strip_tex(s: str) -> str:
    """Drop LaTeX commands but keep their argument text, so 'survival-\\emph{specific}' reads
    'survival-specific' for the wording patterns."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def headline_numbers() -> list[tuple[str, float]]:
    """The numbers the manuscript must quote, read from artifacts/corrected_verdicts.json."""
    with open(os.path.join(ART, "corrected_verdicts.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    out = []
    for run in ("C1", "C2"):
        r = v["runs"].get(run, {})
        if r.get("status") != "complete":
            continue
        s = r["primary"]["survival"]
        out += [(f"{run} survival mean", s["mean"]), (f"{run} survival t90 lower", s["t90"][0]),
                (f"{run} survival t90 upper", s["t90"][1])]
        if run == "C1":
            out += [("C1 predictor mean", r["primary"]["predictor"]["mean"]),
                    ("C1 untrained mean", r["primary"]["untrained"]["mean"]),
                    ("C1 L0 mean", r["gates"]["l0"]["mean"]),
                    ("C1 L0 TOST p", r["gates"]["l0"]["tost_p"])]
    ac = v.get("auxiliary_comparison")
    if ac:
        c = ac["paired"]
        out += [("C1 minus C2 paired mean", c["mean"]), ("C1 minus C2 t90 lower", c["t90"][0]),
                ("C1 minus C2 t90 upper", c["t90"][1])]
    return out


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def gate_table() -> str:
    with open(os.path.join(ART, "gate_table.json"), encoding="utf-8") as fh:
        t = json.load(fh)
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/gate_table.json; do not edit",
         r"\begin{tabular}{llllrl}", r"\toprule",
         r"Run & Status & Partition & L0 mean & TOST $p$ & L0 \\", r"\midrule"]
    for r in t["rows"]:
        st = "pass" if r["l0"]["status"] == "pass" else "inconclusive"
        L.append(f"{_tex_escape(r['run'])} & {r['status']} & {_tex_escape(r['partition'].split(' ')[0])} & "
                 f"{r['l0']['mean']:.3f} & {r['l0'].get('tost_p', float('nan')):.3f} & {st} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def contrast_table() -> str:
    with open(os.path.join(ART, "contrast_intervals.json"), encoding="utf-8") as fh:
        t = json.load(fh)
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/contrast_intervals.json; do not edit",
         r"\begin{tabular}{lllll}", r"\toprule",
         r"Run & Partition & Survival (t 90\%) & $-$ predictor (t 90\%) & $-$ untrained (t 90\%) \\",
         r"\midrule"]
    for r in t["rows"]:
        if r["metric"] != "target":
            continue
        vp, vu = r["vs_predictor"], r["vs_untrained"]
        L.append(f"{_tex_escape(r['run'])} & {_tex_escape(r['partition'].split(' ')[0])} & "
                 f"{r['survival_mean']:.3f} [{r['survival_t90'][0]:.3f}, {r['survival_t90'][1]:.3f}] & "
                 f"{vp['mean']:+.3f} [{vp['t90'][0]:+.3f}, {vp['t90'][1]:+.3f}] & "
                 f"{vu['mean']:+.3f} [{vu['t90'][0]:+.3f}, {vu['t90'][1]:+.3f}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def corrected_table() -> str | None:
    """The corrected runs under the frozen decision rules (artifacts/corrected_verdicts.json,
    scripts/build_corrected_verdicts.py), so the manuscript quotes the gate-aware verdict."""
    p = os.path.join(ART, "corrected_verdicts.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        v = json.load(fh)
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/corrected_verdicts.json; do not edit",
         r"\begin{tabular}{llllll}", r"\toprule",
         r"Run & Survival (t 90\%) & Predictor & Untrained & L0 (TOST $p$) & Verdict \\", r"\midrule"]
    for name, r in v["runs"].items():
        if r.get("status") != "complete":
            L.append(f"{name} & \\multicolumn{{5}}{{l}}{{{_tex_escape(r.get('status', 'absent'))}}} \\\\")
            continue
        p_, g = r["primary"], r["gates"]
        s = p_["survival"]
        L.append(f"{name} & {s['mean']:.3f} [{s['t90'][0]:.3f}, {s['t90'][1]:.3f}] & "
                 f"{p_['predictor']['mean']:.3f} & {p_['untrained']['mean']:.3f} & "
                 f"{g['l0']['mean']:.3f} ({g['l0']['tost_p']:.3f}) & {_tex_escape(p_['verdict'])} \\\\")
    ac = v.get("auxiliary_comparison")
    if ac:
        c = ac["paired"]
        L.append(r"\midrule")
        L.append(f"C1 $-$ C2 & \\multicolumn{{5}}{{l}}{{{c['mean']:+.3f} [{c['t90'][0]:+.3f}, {c['t90'][1]:+.3f}]; "
                 f"auxiliary reading: {'holds' if ac['holds'] else 'holds on the decodability clauses, conditional on the C1 gates' if ac.get('holds_on_decodability_clauses') else 'does not hold'}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def tables() -> dict[str, str]:
    out = {"gate_table.tex": gate_table(), "contrast_intervals.tex": contrast_table()}
    c = corrected_table()
    if c is not None:
        out["corrected_runs.tex"] = c
    return out


def check_manuscript(d: str) -> int:
    """Three guards, each a reviewable item: the manuscript must \\input at least one generated
    table (never retype them), must quote every headline number of artifacts/corrected_verdicts.json
    somewhere, and must not use retired wording (matched after LaTeX markup is stripped) or quote
    a pre-correction headline without a historical label."""
    texs = sorted(glob.glob(os.path.join(d, "**", "*.tex"), recursive=True))
    current = tables()
    issues = 0
    inputs = 0
    full = []
    for p in texs:
        rel = os.path.relpath(p, ROOT)
        text = open(p, encoding="utf-8", errors="replace").read()
        full.append(text)
        raw_lines = text.splitlines()
        blocks = _block_starts(raw_lines)
        for m in re.finditer(r"\\input\{[^}]*?(?:paper_tables|tables)/([^}]+?)(\.tex)?\}", text):
            name = m.group(1) + ".tex"
            if name not in current:
                print(f"{rel}: \\input of unknown table {name}")
                issues += 1
            else:
                inputs += 1
        for i, raw in enumerate(raw_lines, 1):
            line = strip_tex(raw)
            for pat, fix in RETIRED:
                m = re.search(pat, line, re.I)
                if m and not _withdrawn_before(line, m.start()):
                    print(f"{rel}:{i}: retired wording /{pat}/ -> {fix}")
                    issues += 1
            if any(h in line for h in HISTORICAL_HEADLINES):
                block = "\n".join(raw_lines[blocks[i - 1]:i])
                for h in HISTORICAL_HEADLINES:
                    if h in line and not HISTORICAL_LABEL.search(block):
                        print(f"{rel}:{i}: historical headline {h} without a historical or "
                              "pre-correction label in its section")
                        issues += 1
    if texs and not inputs:
        print(f"{d}: no generated table is \\input (docs/paper_tables/{{{', '.join(sorted(current))}}})")
        issues += 1
    if texs:
        joined = "\n".join(full)
        for label, value in headline_numbers():
            s = f"{abs(value):.3f}"
            if s not in joined:
                print(f"{d}: headline number {s} ({label}, artifacts/corrected_verdicts.json) "
                      "not found in the manuscript")
                issues += 1
    print(f"manuscript check: {len(texs)} .tex files, {issues} item(s) to review")
    return 1 if issues else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args(argv)
    if a.manuscript:
        return check_manuscript(a.manuscript)
    want = tables()
    if a.check:
        if not os.path.isdir(OUT):
            print(f"paper tables: no local {os.path.relpath(OUT, ROOT)}/ (manuscript material stays "
                  "local); nothing to compare")
            return 0
        stale = [n for n, w in want.items() if not os.path.exists(os.path.join(OUT, n))
                 or open(os.path.join(OUT, n), encoding="utf-8").read() != w]
        for n in stale:
            print("PAPER TABLE stale:", n)
        print("paper tables: " + ("OK" if not stale else "stale; run python scripts/build_paper_tables.py"))
        return 1 if stale else 0
    os.makedirs(OUT, exist_ok=True)
    for n, w in want.items():
        with open(os.path.join(OUT, n), "w", encoding="utf-8") as fh:
            fh.write(w)
    print(f"wrote {len(want)} table(s) to {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
