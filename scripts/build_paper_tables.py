"""Manuscript tables generated from the checked results, the rule ledger, and a manuscript
check (revision step 15).

Writes LaTeX tables from the committed, audited artifacts, so the manuscript can \\input them
instead of retyping numbers:

  corrected_runs.tex      C1 and C2 under the frozen rules: every arm at drift 0.45, and every
                          arm at drift zero on the registered evaluation-world pair
  contrast_intervals.tex  the survival target and its seed-paired margins for every n = 10 run
  gate_table.tex          the authentic-versus-authentic (L0) gate for every run and partition
  run_legend.tex          what each run identifier in the two tables above names
  rule_ledger.tex         every frozen decision rule adjudicated at the 0.65 bar or the 0.05 margin

The tables are manuscript material and stay LOCAL: they are written to docs/paper_tables/,
which .gitignore excludes, beside (never inside) the gitignored manuscript directory
docs/paper/, so generating them cannot overwrite the author's own files. The scalar audit
renders them in memory from the committed artifacts, so CI needs no paper files.

The rule ledger is committed. docs/rule_ledger.json holds one row per frozen rule and run,
with the outcome as recorded and, for every clause, the artifact value it reads; this script
recomputes each value, compares it with the recorded reading and the recorded outcome, and
renders docs/RULE_LEDGER.md from it. --check runs that comparison and fails on any
difference, with or without a local manuscript.

With --manuscript DIR (the LaTeX source, also local and outside git) it checks that at least
one generated table is \\input and every such \\input refers to a current generated file. It
checks the headline numbers of artifacts/corrected_verdicts.json by location as well as by
presence: every interval the artifact records for a headline (C1 and C2 survival, the C1
baselines and margins, the drift-zero baselines, the C1 minus C2 contrast) must be quoted at
least once right after its own mean, and wherever the manuscript quotes one of those intervals
the number right before it must be that mean; a per-seed list that differs from the
artifact's in one or two places is flagged; and every three-decimal number in the paragraph
headed "The corrected primary run" must be a value of the C1 verdict entry. It flags wording
the 2026-10 revision retired (the claim table in docs/REVISION_2026-10.md; matched after LaTeX
markup is stripped) and any pre-correction headline that appears without a historical label in
its section. A sentence that withdraws a retired reading may quote it, and a historical label
anywhere in the enclosing sectioning block covers the numbers inside it. It reports; the
author decides each flagged line.

Usage:
    python scripts/build_paper_tables.py                     # write the local tables and docs/RULE_LEDGER.md
    python scripts/build_paper_tables.py --check             # exit 1 if the ledger or a local table is stale
    python scripts/build_paper_tables.py --manuscript docs/paper
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import operator
import os
import re
import sys

import numpy as np

from itasorl.stats import t_ci90

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "docs", "paper_tables")   # gitignored; beside, not inside, docs/paper/
# The ledger is curated and committed beside the documents it summarizes. It sits outside
# artifacts/ because every file there must belong to a run in the results manifest.
LEDGER = os.path.join(ROOT, "docs", "rule_ledger.json")
LEDGER_MD = os.path.join(ROOT, "docs", "RULE_LEDGER.md")

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
# Interval macros, dropped before the historical-headline scan: a bound is not a claim.
_CI_MACRO = re.compile(r"\\ci\{[^}]*\}\{[^}]*\}")
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


def _verdicts() -> dict:
    with open(os.path.join(ART, "corrected_verdicts.json"), encoding="utf-8") as fh:
        return json.load(fh)


def headline_numbers() -> list[tuple[str, float]]:
    """The numbers the manuscript must quote, read from artifacts/corrected_verdicts.json."""
    v = _verdicts()
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


def anchored_intervals(v: dict | None = None) -> list[dict]:
    """Each headline interval with the mean it belongs to. `required` ones must be quoted at
    least once as MEAN followed by its interval; every quotation of any of them must follow
    its own mean."""
    v = v if v is not None else _verdicts()
    out = []

    def add(label, mean, t90, required):
        out.append({"label": label, "mean": mean, "lo": t90[0], "hi": t90[1], "required": required})

    for run in ("C1", "C2"):
        r = v["runs"].get(run, {})
        if r.get("status") != "complete":
            continue
        p, c1 = r["primary"], run == "C1"
        add(f"{run} survival", p["survival"]["mean"], p["survival"]["t90"], True)
        add(f"{run} predictor", p["predictor"]["mean"], p["predictor"]["t90"], c1)
        add(f"{run} untrained", p["untrained"]["mean"], p["untrained"]["t90"], c1)
        for k, name in (("contrast_vs_predictor", "margin over the predictor"),
                        ("contrast_vs_untrained", "margin over the untrained arm")):
            add(f"{run} {name}", p[k]["mean"], p[k]["t90"], c1)
        dz = r.get("drift_zero") or {}
        for arm in ("predictor", "untrained", "survival"):
            if arm in dz:
                # The drift-zero baselines of the registered pair are a reported headline;
                # the drift-zero survival interval is quoted on its own, so it is not required.
                add(f"{run} drift-zero {arm}", dz[arm]["mean"], dz[arm]["t90"], c1 and arm != "survival")
    ac = v.get("auxiliary_comparison")
    if ac:
        add("C1 minus C2", ac["paired"]["mean"], ac["paired"]["t90"], True)
    return out


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ------------------------------------------------------------------ display conventions
# Registered verdict words are set in small caps, one convention for every generated table.
VERDICT_WORDS = ("NOT MET", "MET", "INCONCLUSIVE", "UNINFORMATIVE")
# Gate keys as the artifacts store them, and as a reader sees them.
GATE_LABELS = {"l0": "L0", "engagement": "engagement", "speed": "speed control",
               "reward_leak": "reward leakage", "untrained_floor": "untrained floor",
               "deaths": "deaths"}


def _sc(verdict: str) -> str:
    """A registered verdict word in small caps; anything else is escaped and passed through."""
    v = verdict.strip()
    return rf"\textsc{{{v.lower()}}}" if v in VERDICT_WORDS else _tex_escape(v)


def _gates(failed) -> str:
    return ", ".join(GATE_LABELS.get(g, g) for g in sorted(failed))


def _iv(t90, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"[{f.format(t90[0])}, {f.format(t90[1])}]"


# Plain names for the run identifiers the gate and contrast tables use (the manifest IDs).
# A run that appears in either artifact without an entry here fails tests/test_paper_tables.py.
RUN_LABELS = {
    "BV2-L2-AR1": "B-v2: L2 rollout drift (AR(1) drag), survival-coupled, $n = 3$",
    "BV3-REGIME-N10": "B-v3: L2 per-episode drag regime, survival-coupled",
    "L3-H8-N10": "L3, surrogate hidden 8: the historical headline run (GPU)",
    "L3-H7-N10": "L3, surrogate hidden 7: the second capacity (GPU)",
    "L3-H4": "L3, surrogate hidden 4: gate 0 failed, uninformative",
    "L3-H8-HELDOUT": "L3 hidden 8 held-out run: same-recipe transfer and common garden; its "
                     "saved agents serve the readout-only analyses",
    "L3-H7-REVERSE": "L3 reverse transfer: trained at hidden 7, hidden 8 held out",
    "L1-ORGANISM": "L1 observation quantization at the in-band grid",
    "L3-H8-NOWM-CPU": "Architecture baseline: no next-observation auxiliary (CPU)",
    "L3-H8-WM-CPU": "Device control: the hidden-8 protocol with the auxiliary (CPU)",
    "L3-H10-GS1-CPU": "Second surrogate instance (seed 1, hidden 10), CPU run",
    "L3-H10-GS1-GPU": "Second surrogate instance (seed 1, hidden 10), GPU re-measure",
    "L3-H8-GS2-GPU": "Hidden 8 at a new surrogate seed (seed 2), GPU",
    "L3-H8-NOWM-U450": "No-auxiliary arm at 450 updates (the skill-matched check)",
    "CORRECTED-L3_H8_NOWM": "C2: corrected trainer, no auxiliary",
    "CORRECTED-L3_H8_WM": "C1: corrected trainer, with the auxiliary (the primary run)",
}


def _load(name: str) -> dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def _l0_sets(rows: list[dict]) -> tuple[list[str], int]:
    """A letter per row. Rows share a letter when their drift-zero statistics agree to six
    decimals (n, mean, standard deviation, TOST p and ROPE interval), which identical per-seed
    values produce: one set of drift-zero agents and pools read under several run names."""
    keys, out = {}, []
    for r in rows:
        l0 = r["l0"]
        k = (l0.get("n"), *(round(float(l0.get(f, float("nan"))), 6) for f in ("mean", "sd", "tost_p")),
             *(round(float(x), 6) for x in l0.get("rope_interval", [])))
        if k not in keys:
            n = len(keys)
            keys[k] = chr(ord("a") + n) if n < 26 else f"z{n}"
        out.append(keys[k])
    return out, len(keys)


def gate_table() -> str:
    t = _load("gate_table.json")
    sets, n_sets = _l0_sets(t["rows"])
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/gate_table.json; do not edit",
         r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Run & Status & Partition & L0 mean & TOST $p$ & L0 & Set \\", r"\midrule"]
    for r, s in zip(t["rows"], sets):
        st = "pass" if r["l0"]["status"] == "pass" else "inconclusive"
        L.append(f"{_tex_escape(r['run'])} & {r['status']} & {_tex_escape(r['partition'].split(' ')[0])} & "
                 f"{r['l0']['mean']:.3f} & {r['l0'].get('tost_p', float('nan')):.3f} & {st} & {s} \\\\")
    L += [r"\midrule",
          rf"\multicolumn{{7}}{{p{{0.9\textwidth}}}}{{Set: rows with the same letter agree on every "
          rf"drift-zero statistic to six decimals, the signature of one set of drift-zero agents and "
          rf"pools read under several run names, so they are not independent tests. The "
          rf"{len(t['rows'])} rows read {n_sets} distinct sets. The run legend names each "
          r"identifier.} \\",
          r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def _contrast_key(r: dict) -> tuple:
    vp, vu = r["vs_predictor"], r["vs_untrained"]
    return (r["partition"].split(" ")[0], round(r["survival_mean"], 6),
            tuple(round(x, 6) for x in vp.get("diff_per_seed", [])),
            tuple(round(x, 6) for x in vu.get("diff_per_seed", [])))


def contrast_table() -> str:
    t = _load("contrast_intervals.json")
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/contrast_intervals.json; do not edit",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Run & Partition & Survival (t 90\%) & $-$ predictor (t 90\%) & $-$ untrained (t 90\%) \\",
         r"\midrule"]
    first, dup = {}, False
    for r in t["rows"]:
        if r["metric"] != "target":
            continue
        k = _contrast_key(r)
        name = _tex_escape(r["run"])
        if k in first and first[k] != r["run"]:
            name += f" (= {_tex_escape(first[k])})"
            dup = True
        first.setdefault(k, r["run"])
        vp, vu = r["vs_predictor"], r["vs_untrained"]
        L.append(f"{name} & {_tex_escape(r['partition'].split(' ')[0])} & "
                 f"{r['survival_mean']:.3f} {_iv(r['survival_t90'])} & "
                 f"{vp['mean']:+.3f} {_iv(vp['t90'], True)} & {vu['mean']:+.3f} {_iv(vu['t90'], True)} \\\\")
    if dup:
        L += [r"\midrule",
              r"\multicolumn{5}{p{0.9\textwidth}}{(= X): per-seed values identical to run X under the "
              r"same partition, so the row reads the same agents and pools under a second run name and "
              r"is not independent evidence.} \\"]
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def run_legend_table() -> str:
    """The run identifiers of the gate and contrast tables, in order of first appearance."""
    order = []
    for name in ("gate_table.json", "contrast_intervals.json"):
        for r in _load(name)["rows"]:
            if r["run"] not in order:
                order.append(r["run"])
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/gate_table.json and "
         r"artifacts/contrast_intervals.json; do not edit",
         r"\begin{tabular}{lp{0.62\textwidth}}", r"\toprule", r"Run identifier & What it is \\",
         r"\midrule"]
    for run in order:
        L.append(f"{_tex_escape(run)} & {RUN_LABELS.get(run, 'no label recorded')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def corrected_table() -> str | None:
    """The corrected runs under the frozen decision rules (artifacts/corrected_verdicts.json,
    scripts/build_corrected_verdicts.py), so the manuscript quotes the gate-aware verdict. One
    row per run and drift: drift 0.45 carries the decision, and drift zero carries the L0 gate
    with every arm's reading on the registered evaluation-world pair beside it."""
    p = os.path.join(ART, "corrected_verdicts.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        v = json.load(fh)
    L = [r"% generated by scripts/build_paper_tables.py from artifacts/corrected_verdicts.json; do not edit",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Run & Drift & Survival (t 90\%) & Predictor (t 90\%) & Untrained (t 90\%) & "
         r"Registered rule & Decodability clauses (joint) \\", r"\midrule"]
    for name, r in v["runs"].items():
        if r.get("status") != "complete":
            L.append(f"{name} & \\multicolumn{{6}}{{l}}{{{_tex_escape(r.get('status', 'absent'))}}} \\\\")
            continue
        p_, g = r["primary"], r["gates"]
        # The artifact's verdict string for C1 begins "MET on the decodability clauses", which
        # reads as a pass at a glance. Render the two facts in their own columns instead, from
        # the artifact's own `met`, `decodability` and `gates.failed` fields.
        failed = g.get("failed") or []
        rule = _sc("MET" if p_["met"] else "NOT MET")
        if failed:
            rule += f" ({_gates(failed)} open)"
        L.append(f"{name} & 0.45 & {p_['survival']['mean']:.3f} {_iv(p_['survival']['t90'])} & "
                 f"{p_['predictor']['mean']:.3f} {_iv(p_['predictor']['t90'])} & "
                 f"{p_['untrained']['mean']:.3f} {_iv(p_['untrained']['t90'])} & {rule} & "
                 f"{_sc(p_['decodability'])} \\\\")
        dz = r.get("drift_zero")
        if dz:
            l0 = g["l0"]
            gate = (f"L0 {'passes' if l0['pass'] else 'open'} (TOST $p = {l0['tost_p']:.3f}$)")
            L.append(f"{name} & 0 & {dz['survival']['mean']:.3f} {_iv(dz['survival']['t90'])} & "
                     f"{dz['predictor']['mean']:.3f} {_iv(dz['predictor']['t90'])} & "
                     f"{dz['untrained']['mean']:.3f} {_iv(dz['untrained']['t90'])} & {gate} & -- \\\\")
    ac = v.get("auxiliary_comparison")
    if ac:
        c = ac["paired"]
        reading = ("holds" if ac["holds"] else
                   "holds on the decodability clauses, conditional on the C1 gates"
                   if ac.get("holds_on_decodability_clauses") else "does not hold")
        L.append(r"\midrule")
        L.append(f"C1 $-$ C2 & 0.45 & \\multicolumn{{5}}{{l}}{{{c['mean']:+.3f} {_iv(c['t90'], True)}; "
                 f"auxiliary reading: {reading}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ the rule ledger
OUTCOME_CLASSES = ("threshold met", "threshold not met", "uninformative", "no verdict")
_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}
BOUNDARY = 0.01      # a clause this close to its threshold is listed as a boundary case


def load_ledger() -> dict | None:
    if not os.path.exists(LEDGER):
        return None
    with open(LEDGER, encoding="utf-8") as fh:
        return json.load(fh)


def _walk(doc, path: str):
    x = doc
    for part in path.split("/"):
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def ledger_value(spec: dict, cache: dict):
    """The number (or flag) a ledger clause reads, recomputed from the committed artifact."""
    if "minus" in spec:
        a, b = (ledger_value(x, cache) for x in spec["minus"])
        return a - b
    if "max" in spec:
        return max(ledger_value(x, cache) for x in spec["max"])
    art = spec["artifact"]
    if art not in cache:
        with open(os.path.join(ART, art), encoding="utf-8") as fh:
            cache[art] = json.load(fh)
    x = _walk(cache[art], spec["path"])
    if "where" in spec:
        x = [c[spec["key"]] for c in x if all(_same(c.get(k), w) for k, w in spec["where"].items())]
    stat = spec.get("stat")
    if stat is None:
        return x if isinstance(x, bool) else float(x)
    a = np.asarray(x, float)
    if a.size == 0:
        raise ValueError(f"{art}:{spec['path']} selects no values")
    if stat == "mean":
        return float(a.mean())
    if stat == "t90_lower":
        return float(t_ci90(a)[0])
    if stat == "min":
        return float(a.min())
    raise ValueError(f"unknown stat {stat!r}")


def _reading_ok(value: float, reading: str) -> bool:
    nd = len(reading.split(".")[1]) if "." in reading else 0
    return abs(value - float(reading)) <= 0.5 * 10 ** -nd + 1e-9


def evaluate_ledger(ledger: dict) -> tuple[list[dict], list[str]]:
    """Every ledger row with each clause recomputed, and a list of problems: a frozen file that
    does not exist or does not state the recorded freeze date, a value that does not match its
    recorded reading, or a recorded outcome that the recomputed clauses contradict."""
    cache: dict = {}
    rows, problems = [], []
    for row in ledger["rows"]:
        rid = row["id"]
        src = os.path.join(ROOT, row["source"])
        if not os.path.exists(src):
            problems.append(f"{rid}: source {row['source']} does not exist")
        elif row["frozen"] not in open(src, encoding="utf-8").read():
            problems.append(f"{rid}: frozen date {row['frozen']} does not appear in {row['source']}")
        if row.get("outcome_class") not in OUTCOME_CLASSES:
            problems.append(f"{rid}: outcome_class {row.get('outcome_class')!r} is not one of {OUTCOME_CLASSES}")
        out = []
        for c in row["clauses"]:
            try:
                val = ledger_value(c["value"], cache)
            except (OSError, KeyError, IndexError, ValueError, TypeError) as e:
                problems.append(f"{rid}: clause {c['clause']!r} cannot be read ({e!r})")
                continue
            if c["op"] == "is_true":
                out.append({**c, "computed": bool(val), "passes": bool(val), "distance": None})
                continue
            passes = bool(_OPS[c["op"]](val, c["threshold"]))
            dist = val - c["threshold"] if c["op"] in (">=", ">") else c["threshold"] - val
            out.append({**c, "computed": val, "passes": passes, "distance": dist})
            if c.get("reading") is not None and not _reading_ok(val, c["reading"]):
                problems.append(f"{rid}: clause {c['clause']!r} ({c['basis']}) reads {val:.4f} in the "
                                f"artifact, recorded as {c['reading']}")
        for role, key in (("rule", "clauses_hold"), ("gate", "gates_hold")):
            mine = [c for c in out if c["role"] == role]
            want = row.get(key)
            if mine and want is None and role == "gate":
                problems.append(f"{rid}: has gate clauses but records no gates_hold")
            if mine and want is not None and all(c["passes"] for c in mine) != want:
                problems.append(f"{rid}: recorded {key} {want} but the recomputed {role} clauses give "
                                f"{all(c['passes'] for c in mine)}")
        cls = row.get("outcome_class")
        if cls == "threshold met" and (row.get("clauses_hold") is not True or row.get("gates_hold") is False):
            problems.append(f"{rid}: outcome 'threshold met' with a failing clause or gate")
        if cls == "threshold not met" and row.get("clauses_hold") is True and row.get("gates_hold") is not False:
            problems.append(f"{rid}: outcome 'threshold not met' while every clause and gate holds")
        rows.append({**row, "clauses": out})
    return rows, problems


def boundary_cases(rows: list[dict]) -> list[tuple[dict, dict, str]]:
    """Clauses within BOUNDARY of their threshold, each with what its side of the threshold
    does: rule and gate clauses first, then values reported beside the rule's own basis."""
    out = []
    for r in rows:
        for c in r["clauses"]:
            if c["distance"] is None or abs(c["distance"]) > BOUNDARY:
                continue
            if c["role"] == "reported":
                why = "reported beside the rule's own basis; it decides nothing"
            elif c["passes"]:
                peers = [x for x in r["clauses"] if x["role"] == c["role"]]
                why = ("it passes while another clause fails" if not all(x["passes"] for x in peers)
                       else "the rule's clauses hold, this one exactly at the bound" if c["distance"] == 0
                       else "the rule's clauses hold by this margin")
            else:
                others = [x for x in r["clauses"] if x["role"] == c["role"] and x is not c
                          and not x["passes"]]
                why = ("the only clause short of its threshold" if not others
                       else "other clauses fail as well")
            if c.get("note"):
                why += f"; {c['note']}"
            out.append((r, c, why))
    return sorted(out, key=lambda t: (t[1]["role"] == "reported", abs(t[1]["distance"])))


def _dist(d: float) -> str:
    """A signed distance, with a fourth decimal where three would round it to zero or hide
    which side of the threshold it sits on."""
    if d == 0:
        return "0, at the bound"
    return f"{d:+.4f}" if abs(d) < 0.005 else f"{d:+.3f}"


def _val(c: dict) -> str:
    v = c["computed"]
    if c["op"] == "is_true":
        return "yes" if v else "no"
    if c["basis"].startswith("rank"):
        return f"{v:g}"
    r = c.get("reading")
    nd = len(r.split(".")[1]) if r and "." in r else 3
    return f"{v:.{max(nd, 3)}f}"


def _clause_text(c: dict, tex: bool = False) -> str:
    if c["op"] == "is_true":
        body = f"{c['clause']}: {_val(c)}"
    else:
        op = {">=": r"$\geq$", ">": "$>$", "<=": r"$\leq$", "<": "$<$"}[c["op"]] if tex else c["op"]
        body = f"{c['clause']} ({c['basis']}) {_val(c)} {op} {c['threshold']:g}, {_dist(c['distance'])}"
    tag = {"rule": "", "gate": "gate: ", "reported": "reported, not in the rule: "}[c["role"]]
    return tag + body


def _tally(rows: list[dict]) -> dict:
    return {k: sum(r["outcome_class"] == k for r in rows) for k in OUTCOME_CLASSES}


def _tally_sentence(rows: list[dict]) -> str:
    t = _tally(rows)

    def n(k, one, many):
        return f"{t[k]} {one if t[k] == 1 else many}"
    return (f"{len(rows)} rule-and-run rows: the threshold condition holds on {t['threshold met']} "
            f"and fails on {t['threshold not met']}; {n('uninformative', 'is', 'are')} uninformative "
            f"and {n('no verdict', 'gives', 'give')} no verdict.")


_SHORT_BASIS = {"mean": "mean", "t90 lower bound": "LB", "minimum over eight draws": "min",
                "rank of nine, 1 = largest": "rank"}
_SHORT_CLAUSE = {"margin over predictor": "vs predictor", "margin over untrained": "vs untrained",
                 "behavior-controlled bar": "controlled bar", "L0 gate": "L0 gate",
                 "registered draw at or below the median": "rank",
                 "every independent draw at the bar": "every draw",
                 "lead over the no-auxiliary run": "vs no auxiliary",
                 "larger stream decoder at the bar": "stream decoder",
                 "skill match: return at most reference + 0.05": "skill match",
                 "speed positive control in every pool": "speed control",
                 "contrast at the margin": "contrast", "contrast interval excludes zero": "contrast LB",
                 "final treatment AUROC at the bar": "final AUROC",
                 "t90 lower bound minus the untrained mean": "LB vs untrained",
                 "C1 meets the primary rule": "C1 primary rule", "gate 1, exploitability": "gate 1",
                 "gate 5, speed control": "gate 5"}


def _bar_basis(r: dict) -> str:
    """How the rule reads the bar: the mean, the t90 lower bound (LB), a minimum or a rank."""
    bases = []
    for c in r["clauses"]:
        b = _SHORT_BASIS.get(c["basis"])
        if c["role"] == "rule" and c["threshold"] in (0.65, 5) and b and b not in bases:
            bases.append(b)
    return ", ".join(bases) or "--"


def _deciding(r: dict) -> str:
    """The clause that decides the row, as short text: a failing gate first, then the failing
    rule clause nearest its threshold; when everything holds, the clause nearest its threshold."""
    def short(c):
        return _SHORT_CLAUSE.get(c["clause"], c["clause"])

    for role in ("gate", "rule"):
        failing = [c for c in r["clauses"] if c["role"] == role and not c["passes"]]
        if not failing:
            continue
        flags = [c for c in failing if c["distance"] is None or c["basis"].startswith("TOST")]
        if flags:
            names = [short(c) for c in flags]
            return f"{' and '.join(names)} {'fails' if len(names) == 1 else 'fail'}"
        c = min(failing, key=lambda c: abs(c["distance"]))
        return f"{short(c)} {_dist(c['distance'])}"
    cs = [c for c in r["clauses"] if c["role"] in ("rule", "gate") and c["distance"] is not None]
    if not cs:
        return "--"
    c = min(cs, key=lambda c: abs(c["distance"]))
    return f"{short(c)} {_dist(c['distance'])}".replace(", at the bound", " (at the bound)")


def rule_ledger_md(rows: list[dict], ledger: dict) -> str:
    L = ["# Rule ledger", "",
         "*Generated by `scripts/build_paper_tables.py` from `docs/rule_ledger.json`. Do not edit "
         "by hand. Every value below is recomputed from the committed artifact the ledger names, "
         "and `python scripts/build_paper_tables.py --check` fails if a value, a recorded "
         "outcome or this page disagrees with it.*", "",
         ledger["about"], "",
         _tally_sentence(rows) + " \"Holds\" means the clauses at the bar and the margin are "
         "satisfied as frozen; for a knockout or a control whose positive rule is the reading it "
         "looked for, the outcome column says what that means.", "",
         "Each clause is listed with its basis (the mean, the t-based 90% lower bound, a minimum "
         "or a rank), its recomputed value, the threshold as frozen, and the signed distance to "
         "that threshold (positive is on the passing side). Clauses marked \"reported, not in the "
         "rule\" are values the record quotes beside the rule's own basis, such as an interval "
         "beside a rule adjudicated on the mean; they decide nothing.", ""]
    hits = boundary_cases(rows)
    L += [f"## Boundary cases (within {BOUNDARY:g} of a threshold)", ""]
    for title, pick in (("On the rule's own basis:", lambda c: c["role"] != "reported"),
                        ("Reported values beside the rule's own basis:", lambda c: c["role"] == "reported")):
        sel = [(r, c, why) for r, c, why in hits if pick(c)]
        if sel:
            L += [title, ""]
            L += [f"- {r['rule']} (`{r['run']}`): "
                  f"{_clause_text(c).split('in the rule: ', 1)[-1].replace(', 0, at the bound', ', 0')}; {why}."
                  for r, c, why in sel]
            L.append("")
    if not hits:
        L += ["None.", ""]
    L += ["## Rules", "",
          "| # | Rule | Frozen in | Run | Clauses | Outcome as recorded | Recorded in |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        src = f"`{r['source']}` ({r['source_part']}), {r['frozen']}"
        clauses = "<br>".join(_clause_text(c) for c in r["clauses"])
        L.append(f"| {i} | {r['rule']} | {src} | `{r['run']}` ({r['trainer']}, n = {r['n']}) | {clauses} | "
                 f"{r['outcome']} ({r['outcome_class']}) | {r['recorded_in']} |")
    notes = [(i, r) for i, r in enumerate(rows, 1) if r.get("freeze_note") or r.get("note")]
    if notes:
        L += ["", "## Notes", ""]
        for i, r in notes:
            parts = [x for x in (r.get("freeze_note"), r.get("note")) if x]
            L.append(f"- Row {i}, {r['rule']}: " + " ".join(parts))
    return "\n".join(L) + "\n"


def rule_ledger_table() -> str | None:
    ledger = load_ledger()
    if ledger is None:
        return None
    rows, _ = evaluate_ledger(ledger)
    L = [r"% generated by scripts/build_paper_tables.py from docs/rule_ledger.json and the artifacts "
         r"it names; do not edit",
         r"\begin{tabular}{p{0.24\textwidth}p{0.085\textwidth}p{0.16\textwidth}p{0.07\textwidth}"
         r"p{0.11\textwidth}p{0.16\textwidth}}",
         r"\toprule", r"Rule & Frozen & Run & Bar on & Outcome & Deciding clause \\", r"\midrule"]
    for r in rows:
        L.append(f"{_tex_escape(r.get('short', r['rule']))} & {r['frozen']} & {_tex_escape(r['run'])} & "
                 f"{_bar_basis(r)} & {r['outcome_class'].replace('threshold ', '')} & "
                 f"{_tex_escape(_deciding(r))} \\\\")
    L += [r"\midrule",
          rf"\multicolumn{{6}}{{p{{0.9\textwidth}}}}{{{_tex_escape(_tally_sentence(rows))} Outcome "
          r"says whether the threshold condition was met as frozen. Bar on: the bar is read on the "
          r"mean, the t-based 90\% lower bound (LB), a minimum over draws, or a rank. Deciding "
          r"clause: a failing gate, else the failing clause nearest its threshold, else the clause "
          r"nearest its threshold; distances are signed, positive on the passing side.} \\",
          r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def tables() -> dict[str, str]:
    out = {"gate_table.tex": gate_table(), "contrast_intervals.tex": contrast_table(),
           "run_legend.tex": run_legend_table()}
    c = corrected_table()
    if c is not None:
        out["corrected_runs.tex"] = c
    rl = rule_ledger_table()
    if rl is not None:
        out["rule_ledger.tex"] = rl
    return out


# ------------------------------------------------------------------ the manuscript check
_NUM = re.compile(r"(?<![\d.])([+\-\u2212]?)(\d\.\d{3})(?!\d)")
_IV = re.compile(r"\\ci\{\s*\$?([+\-\u2212]?\d\.\d{3})\$?\s*\}\{\s*\$?([+\-\u2212]?\d\.\d{3})\$?\s*\}"
                 r"|\[\s*\$?([+\-\u2212]?\d\.\d{3})\$?\s*,\s*\$?([+\-\u2212]?\d\.\d{3})\$?\s*\]")
_WINDOW = 60            # characters before an interval in which its mean must sit
PRIMARY_PARAGRAPH = re.compile(r"\\paragraph\{The corrected primary run")


def _f3(x: float) -> str:
    return f"{abs(x):.3f}"


def _intervals(line: str):
    for m in _IV.finditer(line):
        lo, hi = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        yield m, lo.replace("\u2212", "-"), hi.replace("\u2212", "-")


def _number_before(line: str, start: int) -> str | None:
    """The last free-standing three-decimal number in the window before `start`; numbers
    inside another interval do not count."""
    masked = _IV.sub(lambda m: " " * len(m.group(0)), line[:start])
    window = masked[max(0, start - _WINDOW):]
    nums = [m.group(2) for m in _NUM.finditer(window)]
    return nums[-1] if nums else None


def _floats(x, out: set) -> set:
    if isinstance(x, bool):
        return out
    if isinstance(x, (int, float)):
        out.add(_f3(float(x)))
    elif isinstance(x, dict):
        for v in x.values():
            _floats(v, out)
    elif isinstance(x, list):
        for v in x:
            _floats(v, out)
    return out


def _seed_lists(v: dict) -> list[tuple[str, list[str]]]:
    out = []
    for run in ("C1", "C2"):
        r = v["runs"].get(run, {})
        if r.get("status") != "complete":
            continue
        out.append((f"{run} per-seed survival", [_f3(x) for x in r["primary"]["survival"]["per_seed"]]))
        out.append((f"{run} per-seed drift-zero survival", [_f3(x) for x in r["gates"]["l0"]["per_seed"]]))
    return out


def _location_issues(rel: str, lines: list[str], v: dict, found: set) -> list[str]:
    """Per-line checks for one .tex file. `found` collects the required anchors seen."""
    issues = []
    anchors = anchored_intervals(v)
    by_iv = {}
    for a in anchors:
        by_iv.setdefault((_f3(a["lo"]) if a["lo"] >= 0 else f"-{_f3(a['lo'])}",
                          _f3(a["hi"]) if a["hi"] >= 0 else f"-{_f3(a['hi'])}"), []).append(a)
    seeds = _seed_lists(v)
    for i, raw in enumerate(lines, 1):
        for m, lo, hi in _intervals(raw):
            key = (lo.lstrip("+"), hi.lstrip("+"))
            here = by_iv.get(key)
            if not here:
                continue
            before = _number_before(raw, m.start())
            if before is None:
                continue
            match = [a for a in here if before == _f3(a["mean"])]
            if match:
                found.update(a["label"] for a in match)
            else:
                a = here[0]
                issues.append(f"{rel}:{i}: interval [{lo}, {hi}] is the artifact's {a['label']} "
                              f"interval, whose mean is {_f3(a['mean'])}, but the number before it "
                              f"reads {before}")
        # A per-seed list: five or more three-decimal numbers separated by commas.
        for m in re.finditer(r"\d\.\d{3}(?:\s*,\s*\d\.\d{3}){4,}", raw):
            got = re.findall(r"\d\.\d{3}", m.group(0))
            for label, want in seeds:
                if len(got) == len(want):
                    diff = sum(a != b for a, b in zip(got, want))
                    if 1 <= diff <= 2:
                        issues.append(f"{rel}:{i}: a per-seed list differs from the {label} in {diff} "
                                      f"place(s) (artifact: {', '.join(want)})")
    return issues


def _primary_paragraph_issues(rel: str, text: str, v: dict) -> list[str]:
    m = PRIMARY_PARAGRAPH.search(text)
    c1 = v["runs"].get("C1", {})
    if not m or c1.get("status") != "complete":
        return []
    rest = text[m.end():]
    stop = re.search(r"\n\s*\n|\\(?:sub)*section\*?\{|\\paragraph\{|\\begin\{(?:table|figure)", rest)
    para = rest[: stop.start() if stop else len(rest)]
    allowed = _floats(c1, set()) | _floats(v.get("auxiliary_comparison", {}), set())
    line0 = text[: m.start()].count("\n") + 1
    issues = []
    for n in _NUM.finditer(para):
        if n.group(2) not in allowed:
            ln = line0 + para[: n.start()].count("\n")
            issues.append(f"{rel}:{ln}: {n.group(1)}{n.group(2)} in the corrected primary-run paragraph "
                          "is not a value of the C1 entry of artifacts/corrected_verdicts.json")
    return issues


def check_manuscript(d: str) -> int:
    """Guards, each a reviewable item: the manuscript must \\input at least one generated
    table (never retype them); must quote every headline number of
    artifacts/corrected_verdicts.json, the required intervals right after their means, and no
    headline interval after a different number; must not alter a per-seed list or a number of
    the corrected primary-run paragraph; and must not use retired wording (matched after LaTeX
    markup is stripped) or quote a pre-correction headline without a historical label."""
    texs = sorted(glob.glob(os.path.join(d, "**", "*.tex"), recursive=True))
    current = tables()
    v = _verdicts()
    issues = 0
    inputs = 0
    full = []
    found: set = set()
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
            # The title names the paper's contrast; it is not a claim about a run, so the prose
            # guard does not read it.
            is_title = raw.lstrip().startswith("\\title{")
            for pat, fix in RETIRED:
                if is_title:
                    break
                m = re.search(pat, line, re.I)
                if m and not _withdrawn_before(line, m.start()):
                    print(f"{rel}:{i}: retired wording /{pat}/ -> {fix}")
                    issues += 1
            # A confidence bound that happens to equal a historical headline is not a claim
            # about that headline. Interval macros are dropped before this scan; the drift-0.45
            # draw interval \ci{0.752}{0.808} is the case that forced this.
            prose = _CI_MACRO.sub(" ", raw)
            prose = strip_tex(prose)
            if any(h in prose for h in HISTORICAL_HEADLINES):
                block = "\n".join(raw_lines[blocks[i - 1]:i])
                for h in HISTORICAL_HEADLINES:
                    if h in prose and not HISTORICAL_LABEL.search(block):
                        print(f"{rel}:{i}: historical headline {h} without a historical or "
                              "pre-correction label in its section")
                        issues += 1
        for msg in _location_issues(rel, raw_lines, v, found) + _primary_paragraph_issues(rel, text, v):
            print(msg)
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
        for a in anchored_intervals(v):
            if a["required"] and a["label"] not in found:
                print(f"{d}: {a['label']} {_f3(a['mean'])} {_iv((a['lo'], a['hi']), a['lo'] < 0)} is not "
                      "quoted as its mean followed by its interval (artifacts/corrected_verdicts.json)")
                issues += 1
    print(f"manuscript check: {len(texs)} .tex files, {issues} item(s) to review")
    return 1 if issues else 0


def check_ledger() -> list[str]:
    """Problems with the committed rule ledger: a value or outcome its artifacts contradict, or
    a docs/RULE_LEDGER.md that is not the rendering of the current ledger."""
    ledger = load_ledger()
    if ledger is None:
        return []
    rows, problems = evaluate_ledger(ledger)
    want = rule_ledger_md(rows, ledger)
    have = open(LEDGER_MD, encoding="utf-8").read() if os.path.exists(LEDGER_MD) else None
    if have != want:
        problems.append(f"{os.path.relpath(LEDGER_MD, ROOT)} is stale; run python scripts/build_paper_tables.py")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args(argv)
    if a.manuscript:
        return check_manuscript(a.manuscript)
    ledger = load_ledger()
    if a.check:
        bad = check_ledger()
        for msg in bad:
            print("RULE LEDGER:", msg)
        if ledger is None:
            print("rule ledger: docs/rule_ledger.json absent; nothing to compare")
        else:
            print("rule ledger: " + ("OK" if not bad else f"{len(bad)} problem(s)"))
        if not os.path.isdir(OUT):
            print(f"paper tables: no local {os.path.relpath(OUT, ROOT)}/ (manuscript material stays "
                  "local); nothing to compare")
            return 1 if bad else 0
        want = tables()
        stale = [n for n, w in want.items() if not os.path.exists(os.path.join(OUT, n))
                 or open(os.path.join(OUT, n), encoding="utf-8").read() != w]
        for n in stale:
            print("PAPER TABLE stale:", n)
        print("paper tables: " + ("OK" if not stale else "stale; run python scripts/build_paper_tables.py"))
        return 1 if stale or bad else 0
    rc = 0
    if ledger is not None:
        rows, problems = evaluate_ledger(ledger)
        for msg in problems:
            print("RULE LEDGER:", msg)
        rc = 1 if problems else 0
        md = rule_ledger_md(rows, ledger)
        current = open(LEDGER_MD, encoding="utf-8").read() if os.path.exists(LEDGER_MD) else None
        # Compare with line endings normalized, so a checkout with CRLF endings is not rewritten
        # (and shown as modified) when the rendering has not changed.
        if current is None or current.replace("\r\n", "\n") != md:
            with open(LEDGER_MD, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(md)
            print(f"wrote {os.path.relpath(LEDGER_MD, ROOT)}" + (f" ({len(problems)} problem(s))" if problems else ""))
        else:
            print(f"{os.path.relpath(LEDGER_MD, ROOT)} unchanged")
    want = tables()
    os.makedirs(OUT, exist_ok=True)
    for n, w in want.items():
        with open(os.path.join(OUT, n), "w", encoding="utf-8") as fh:
            fh.write(w)
    print(f"wrote {len(want)} table(s) to {os.path.relpath(OUT, ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
