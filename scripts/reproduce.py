"""One entry point for reproducing the results (revision step 15; docs/REPRODUCE.md).

  tables       No training. Rebuild the results manifest, the gate table, and the contrast
               intervals from committed per-seed artifacts, then run the scalar audit. With
               --dumps DIR (state dumps written by a run, fullruns/<run>/states), also
               recompute every per-seed readout from the raw recurrent states under both fold
               partitions (scripts/rescore_fold_split.py) and the behavior control
               (scripts/audit_behavior_mediation.py), into dist/fold_rescore_<run>.json and
               dist/behavior_audit_<run>.json. Minutes on a laptop.
  retrain RUN  Print, or with --execute run, the exact commands that retrain a run listed in
               docs/RESULTS_MANIFEST.md. Hours of CPU per run (estimates in docs/REPRODUCE.md).
  supplement   Build an anonymized submission archive under dist/: code, tests, committed
               artifacts, the method documents, and each run bundle in RUN_DIRS that exists
               under --runs-root (default fullruns/). Author names, handles, e-mail
               addresses, repository URLs, externally hosted images and local paths are
               scrubbed from text files. The package and project name is replaced by a
               neutral one in text and in file names (--anon-name, default anonpkg;
               --keep-name turns this off). Every shipped file, binaries included, is checked
               for identity strings, and a SHA-256 list of every file is written.
               README_SUPPLEMENT.md and the console both say which run bundles were
               included and which were absent.

Usage:
    python scripts/reproduce.py tables [--dumps fullruns/corrected_l3_h8_wm/states]
    python scripts/reproduce.py retrain C1 [--execute]
    python scripts/reproduce.py supplement [--out dist/supplement.zip] [--runs-root DIR]
                                           [--anon-name NAME | --keep-name]
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
    "python scripts/run_texture_fresh_probe.py --agents-dir {run}/agents --family qd --param 6.0 --out artifacts/texture/{name}_qd.json --workers 4",
    "python scripts/validate_population_readout.py --run-dir {run} --out artifacts/population_readout/{name}.json",
]
# After both corrected runs: frozen-rule verdicts, the budget curve, and the exploratory
# cross-run replay (FINDINGS 17.4, 17.10).
AFTER_BOTH = [
    "python scripts/build_corrected_verdicts.py",
    'python scripts/build_budget_curve.py --run "decoder on=artifacts/corrected_runs/corrected_l3_h8_wm" '
    '--run "decoder off=artifacts/corrected_runs/corrected_l3_h8_nowm" --out artifacts/budget_curve.json '
    "--figure docs/figures/budget_curve.png",
    "python scripts/run_cross_replay.py --a fullruns/corrected_l3_h8_wm --b fullruns/corrected_l3_h8_nowm "
    "--out artifacts/cross_replay/corrected_c1_c2.json   # exploratory, post hoc",
]


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


BUILDERS = ("build_results_manifest.py", "build_gate_table.py", "build_contrast_intervals.py",
            "build_corrected_verdicts.py", "build_paper_tables.py")


def tables(dumps: list[str]) -> int:
    # Say first whether the committed pages already match what the builders produce here, so a
    # rewrite that changes a tracked file is visible as such rather than silently dirtying it.
    stale = 0
    for script in BUILDERS:
        stale |= _run([PY, os.path.join("scripts", script), "--check"])
    print("committed tables and pages: " + ("current" if not stale else "stale here; rewriting"),
          flush=True)
    rc = 0
    for script in BUILDERS:
        rc |= _run([PY, os.path.join("scripts", script)])
    rc |= _run([PY, os.path.join("scripts", "audit_stats_recheck.py")])
    if dumps:
        os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)   # dist/ is gitignored
    for d in dumps:
        # rescore_fold_split.py declares --json required, so the path exits 2 without it.
        # Both outputs are named after the run, so two runs never write the same file.
        tag = dump_tag(d)
        out_json = os.path.join("dist", f"fold_rescore_{tag}.json")
        rc |= _run([PY, os.path.join("scripts", "rescore_fold_split.py"), d,
                    "--json", out_json])
        rc |= _run([PY, os.path.join("scripts", "audit_behavior_mediation.py"), d,
                    "--json", os.path.join("dist", f"behavior_audit_{tag}.json")])
    return rc


def dump_tag(d: str) -> str:
    """Output tag for a dump directory. The documented layout is fullruns/<run>/states, whose
    leaf is always "states", so the tag is the run directory's name in that case. The path is
    normalised first, so a trailing separator gives the same tag."""
    p = os.path.normpath(d)
    leaf = os.path.basename(p)
    if leaf == "states":
        leaf = os.path.basename(os.path.dirname(p)) or leaf
    return leaf or "states"


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
        print("\n# once both C1 and C2 exist:")
        for r in AFTER_BOTH:
            print(r)
    if execute:
        rc = 0
        for c in cmds:
            rc |= subprocess.call(c.split("   #")[0], shell=True, cwd=ROOT)
        return rc
    return 0


# Text scrubbing for the anonymized supplement. Identity terms are read at build time from
# CITATION.cff and the git remote, so this file names no one. CITATION.cff itself is packaged
# (the scalar audit's wording guards read it) and scrubbed like every other text file.
# The patterns are assembled from pieces so that none of them matches its own source: this
# file is packaged and scrubbed too, and a rule that matched itself would ship as a placeholder.
GENERIC_SCRUB = [
    # One or more separators: a Windows path inside a JSON string carries doubled backslashes.
    (re.compile(r"[A-Za-z]:[/\\]+" + "Users" + r"[/\\]+[^/\\\s\"']+", re.I), "<local-path>"),
    (re.compile("/ho" + "me/" + r"[^/\s\"']+"), "<local-path>"),
    # The final label must be alphabetic. Without that, the project's own condition notation
    # ("train@0.45", "eval@0.45") and pinned package versions ("react@18.3.1") read as
    # addresses, and the anonymized archive shipped the frozen preregistration with its
    # conditions replaced by the placeholder.
    (re.compile(r"[\w.+-]+" + "@" + r"[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"), "<email>"),
    (re.compile("orc" + r"id\.org/[\d-]+X?", re.I), "orcid.org/<orcid>"),
]
_EMAIL_RULE = GENERIC_SCRUB[2]

# Images served from GitHub-owned hosts (an attachment upload, raw user content, or a Pages
# site). The URL itself is opaque, but the image can show the account behind it: the
# README's social-preview image renders the owner's Pages address. Local images
# (docs/figures/..., relative paths) are left alone.
_GH = "git" + "hub"
_HOSTED = (r"https?://(?:" + _GH + r"\.com/user-" + r"attachments/|[\w.-]*" + _GH
           + r"usercontent\.com/|[\w-]+\." + _GH + r"\.io/)")
IMAGE_PLACEHOLDER = "<!-- image removed for anonymity -->"
IMAGE_SCRUB = [
    (re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']?" + _HOSTED + r"[^>]*>", re.I), IMAGE_PLACEHOLDER),
    (re.compile(r"!\[[^\]]*\]\(\s*<?" + _HOSTED + r"[^)]*\)", re.I), IMAGE_PLACEHOLDER),
]
REPO_PLACEHOLDER = "<anonymous-repository>"
DEFAULT_ANON_NAME = "anonpkg"


def _remote() -> tuple[str, str] | None:
    """(owner, repository) parsed from the git remote, or None outside a checkout."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return None
    m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    return (m.group(1), m.group(2)) if m else None


def identity_terms() -> list[str]:
    terms = []
    cff = os.path.join(ROOT, "CITATION.cff")
    if os.path.exists(cff):
        for line in open(cff, encoding="utf-8"):
            m = re.match(r"\s*-?\s*(family-names|given-names|alias|name):\s*\"?([^\"]+)\"?", line)
            if m and m.group(1) != "name":
                terms.append(m.group(2).strip())
    # Only the owner. The repository name is not a term: its lower-case form is the package
    # name, and a term rule would rewrite every import. url_rules() covers it inside URLs,
    # and the package rename covers it everywhere else.
    remote = _remote()
    if remote:
        terms.append(remote[0])
    # A scrubbed CITATION.cff (an extracted archive rebuilding itself) yields the placeholders
    # themselves; they are not identity terms, or every scrubbed file would report as a leak.
    terms = [t for t in terms if not (t.startswith("<") and t.endswith(">"))]
    return sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True)


def url_rules(remote: tuple[str, str] | None = None) -> list:
    """Repository and site URLs built from the remote's owner and repository name, matched
    without regard to case (GitHub ignores it, and the Pages host is lower case). They run
    before the owner rule: once the owner reads <author>, these patterns no longer match, and
    the repository name would survive inside the rewritten URL."""
    remote = remote if remote is not None else _remote()
    if not remote:
        return []
    owner, repo = (re.escape(x) for x in remote)
    end = r"(?![\w-])"
    pats = [
        r"(?:https?://|git@)?(?:www\.)?" + _GH + r"\.com[/:]" + owner + "/" + repo + r"(?:\.git)?" + end,
        r"(?:https?://)?" + owner + r"\." + _GH + r"\.io/" + repo + end,
        r"(?:https?://)?colab\.research\.google\.com/" + _GH + "/" + owner + "/" + repo + end,
        r"(?:https?://)?raw\." + _GH + r"usercontent\.com/" + owner + "/" + repo + end,
    ]
    return [(re.compile(p, re.I), REPO_PLACEHOLDER) for p in pats]


def term_rules(terms: list[str]) -> list:
    """One rule per identity term. A term matches only as a whole word and with its own
    capitalization (a case-insensitive "Vale" would also hit "valence"); a mixed-case handle
    (e.g. a GitHub owner) also matches in lower case, as in URLs."""
    rules = []
    for t in terms:
        pats = [t] + ([t.lower()] if t != t.lower() and t != t.capitalize() else [])
        for v in pats:
            rules.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![a-z0-9])"), "<author>"))
    return rules


def scrub_rules() -> list:
    """Hosted images first, then repository URLs, then the generic rules, then one rule per
    identity term (term_rules). The order matters: see url_rules."""
    return IMAGE_SCRUB + url_rules() + list(GENERIC_SCRUB) + term_rules(identity_terms())


def project_name() -> str | None:
    """The package's own name, read at build time from pyproject.toml. This file does not
    spell it: it is renamed in the archive, and a literal kept here would survive as a clue."""
    try:
        text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"(?ms)^\[project\]\s*$.*?^name\s*=\s*\"([^\"]+)\"", text)
    return m.group(1) if m else None


def renamer(old: str | None, new: str | None):
    """A function that replaces every occurrence of `old`, in any capitalization, with `new`:
    lower case stays lower, upper case stays upper, and any mixed case becomes `new`
    capitalized. Substrings count, so module paths, environment variables and file names
    (old.stats, OLD_FOLDS, docs/OLD.md) all follow. Identity when either name is missing."""
    if not old or not new or old.lower() == new.lower():
        return lambda s: s
    pat = re.compile(re.escape(old), re.I)

    def rep(m):
        s = m.group(0)
        if s == s.lower():
            return new.lower()
        if s == s.upper():
            return new.upper()
        return new.capitalize()

    return lambda s: pat.sub(rep, s)


def check_anon_name(name: str, old: str | None) -> None:
    """Refuse a replacement name that is not an importable package name or that would leave
    the original (or an identity term) readable. A name equal to the current one is allowed:
    an extracted archive rebuilding itself already carries the neutral name."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SystemExit(f"--anon-name must be a lower-case Python identifier, got {name!r}")
    low = old.lower() if old else ""
    if low and low != name and (low in name or name in low):
        raise SystemExit("--anon-name must not contain, or be contained in, the original name")
    if any(t.lower() in name for t in identity_terms()):
        raise SystemExit("--anon-name must not contain an identity term")


TEXT_EXT = {".py", ".md", ".json", ".txt", ".sh", ".toml", ".ini", ".cfg", ".log", ".yml", ".yaml",
            ".cff", ".html", ".js", ".css", ".err", ".tex", ".bib", ".csv", ".ipynb"}


def is_text(name: str) -> bool:
    """Scrubbed and scanned as text: a known text extension, or no extension at all (LICENSE,
    SHA256SUMS)."""
    base = os.path.basename(name)
    ext = os.path.splitext(base)[1]
    return ext.lower() in TEXT_EXT or (ext == "" and "." not in base)


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
                 "docs/rule_ledger.json", "docs/RULE_LEDGER.md",
                 # read by the scalar audit's wording guards and the site check:
                 "README.md", "CITATION.cff", "index.html", "index.template.html",
                 "viz/player/brain/brain.js",
                 # imported by tests/test_viz_collect.py and tests/test_colab_notebook.py:
                 "viz/collect.py", "notebooks/colab_gpu.ipynb"]
# Run bundles the archive carries when they exist under --runs-root (default fullruns/). They
# are gitignored by design. The corrected runs' directories (cells, saved agents, state dumps)
# exist only where those runs were trained; their cell records ship either way under
# artifacts/corrected_runs/. The saved agents of the historical hidden-8 held-out run and of
# the hidden-8 seed-2 GPU run (60 files each) back the direction-of-rearing readout, the
# drift-0.45 world-sample check and the committed artifacts read from them. A bundle that is
# absent is reported on the console and in README_SUPPLEMENT.md, never skipped silently.
RUN_DIRS = ["fullruns/corrected_l3_h8_wm", "fullruns/corrected_l3_h8_nowm",
            "fullruns/l3_h8_heldout/agents", "fullruns/l3_h8_gseed2_gpu/agents"]


def scrub(text: str, rules=None) -> str:
    for pat, rep in rules if rules is not None else scrub_rules():
        text = pat.sub(rep, text)
    return text


def _git_files() -> list[str] | None:
    """Every path git would commit: tracked plus untracked-but-not-ignored. None outside a
    checkout (an extracted supplement rebuilding itself)."""
    try:
        res = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [n.decode("utf-8", errors="replace") for n in res.stdout.split(b"\0") if n]


def _walk(d: str) -> list[str]:
    out = []
    for base, dirs, files in os.walk(os.path.join(ROOT, d)):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            out.append(os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/"))
    return out


def _run_dir_source(r: str, runs_root: str) -> str:
    """Where archive path fullruns/<sub> is read from: <runs_root>/<sub>."""
    head, _, sub = r.partition("/")
    if head != "fullruns" or not sub:
        raise ValueError(f"RUN_DIRS entries live under fullruns/, got {r!r}")
    return os.path.join(runs_root, *sub.split("/"))


def _sources(runs_root: str | None = None):
    """(archive path, source path) for every file the archive carries, sorted, and the status
    of each RUN_DIRS entry as (entry, file count), the count None when the entry is absent.
    The repository part follows git, not the filesystem, so gitignored scratch output under
    the included directories (artifacts/clip_audit/, about 400 MB) never ships."""
    runs_root = runs_root or os.path.join(ROOT, "fullruns")
    roots = INCLUDE_DIRS + ["docs/specs", "docs/figures"]
    listed = _git_files()
    if listed is None:
        repo = [f for d in roots for f in _walk(d)]
    else:
        repo = [f for f in listed if "__pycache__" not in f
                and any(f == d or f.startswith(d + "/") for d in roots)]
    repo += [f for f in INCLUDE_FILES if os.path.exists(os.path.join(ROOT, f))]
    pairs = {f: os.path.join(ROOT, f) for f in repo if os.path.isfile(os.path.join(ROOT, f))}
    status = []
    for r in RUN_DIRS:
        src = _run_dir_source(r, runs_root)
        if not os.path.isdir(src):
            status.append((r, None))
            continue
        n = 0
        for base, dirs, files in os.walk(src):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in files:
                full = os.path.join(base, f)
                pairs[r + "/" + os.path.relpath(full, src).replace(os.sep, "/")] = full
                n += 1
        status.append((r, n))
    return sorted(pairs.items()), status


def _files(runs_root: str | None = None) -> list[str]:
    """Archive paths of every file the supplement carries (see _sources)."""
    return [f for f, _ in _sources(runs_root)[0]]


def _readme(status, anon_name: str | None) -> str:
    lines = ["Anonymized supplement. Start with docs/REPRODUCE.md.",
             "python scripts/reproduce.py tables   regenerates the tables and gates without training.",
             "python scripts/reproduce.py retrain C1   prints the retraining commands.",
             ""]
    if anon_name:
        lines += [f"For review, the project and package name is replaced by the neutral name "
                  f"\"{anon_name}\" throughout the code, the documents and the file names. "
                  f"Repository URLs read {REPO_PLACEHOLDER}, and externally hosted images are "
                  "removed.", ""]
    lines.append("Run bundles (gitignored outputs of training runs):")
    for r, n in status:
        lines.append(f"  included      {r} ({n} files)" if n is not None else
                     f"  not included  {r} (absent where this archive was built)")
    lines += ["",
              "Saved agents and state dumps of a run that is not listed as included are not in "
              "this archive. The corrected runs' cell records are under artifacts/corrected_runs/ "
              "either way, so every corrected table rebuilds without training."]
    return "\n".join(lines) + "\n"


def _leak_patterns(old_name: str | None) -> list:
    """Leak-only checks beyond the scrub rules: any GitHub Pages host or hosted-image URL that
    survived, and, when the package was renamed, any spelling of the original name."""
    pats = [re.compile(r"[\w-]+\." + _GH + r"\.io\b", re.I), re.compile(_HOSTED, re.I)]
    if old_name:
        pats.append(re.compile(re.escape(old_name), re.I))
    return pats


def _hits(text: str, pats) -> bool:
    return any(p.search(text) for p in pats)


def _binary_hits(data: bytes, pats, depth: int = 0) -> bool:
    """Scan raw bytes, and the decompressed members of a zip container (torch .pt bundles and
    .npz arrays are zips; .npz members are deflated, so the raw bytes alone hide them)."""
    if _hits(data.decode("latin-1"), pats):
        return True
    if depth < 2 and data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                for n in inner.namelist():
                    if _hits(n, pats) or _binary_hits(inner.read(n), pats, depth + 1):
                        return True
        except zipfile.BadZipFile:
            pass
    return False


def leak_scan(path: str, rules, old_name: str | None) -> list[str]:
    """Members of the archive at `path` that still carry an identity string. Text members are
    scanned with every scrub rule; binary members (agent bundles, arrays, images) with every
    rule except the e-mail shape, which arbitrary bytes match by chance about once per few
    tens of MB. Member names are scanned too."""
    text_pats = [p for p, _ in rules] + _leak_patterns(old_name)
    bin_pats = [p for p, r in rules if (p, r) != _EMAIL_RULE] + _leak_patterns(old_name)
    leaks = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            data = z.read(n)
            if _hits(n, bin_pats):
                leaks.append(n)
            elif is_text(n):
                if _hits(data.decode("utf-8", errors="replace"), text_pats):
                    leaks.append(n)
            elif _binary_hits(data, bin_pats):
                leaks.append(n)
    return leaks


def supplement(out: str, runs_root: str | None = None,
               anon_name: str | None = DEFAULT_ANON_NAME) -> int:
    old_name = project_name() if anon_name else None
    if anon_name:
        check_anon_name(anon_name, old_name)
    if old_name and old_name.lower() == anon_name.lower():
        old_name = None          # already neutral (a rebuild from an extracted archive)
    rename = renamer(old_name, anon_name)
    rules = scrub_rules()
    pairs, status = _sources(runs_root)
    print(f"run bundles under {runs_root or os.path.join(ROOT, 'fullruns')}:")
    for r, n in status:
        if n is None:
            print(f"  WARNING: {r} is absent; the archive will NOT carry it", flush=True)
        else:
            print(f"  included {r} ({n} files)")
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)) or ".", exist_ok=True)
    sums = []
    with zipfile.ZipFile(os.path.join(ROOT, out), "w", zipfile.ZIP_DEFLATED) as z:
        for rel, src in pairs:
            with open(src, "rb") as fh:
                data = fh.read()
            if is_text(rel):
                data = rename(scrub(data.decode("utf-8", errors="replace"), rules)).encode("utf-8")
            arc = rename(rel)
            z.writestr(arc, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")
        z.writestr("README_SUPPLEMENT.md", _readme(status, anon_name))
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    leaks = leak_scan(os.path.join(ROOT, out), rules, old_name)
    print(f"wrote {out}: {len(sums)} files" + (f"; POSSIBLE IDENTITY LEAKS in {leaks}" if leaks
          else "; no identity strings found in any file, binaries included"))
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
    s.add_argument("--out", default="dist/supplement.zip")
    s.add_argument("--runs-root", default=None,
                   help="directory that holds the run bundles named in RUN_DIRS (default: "
                        "fullruns/ in this checkout); only read")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--anon-name", default=DEFAULT_ANON_NAME,
                   help="neutral name that replaces the package and project name everywhere "
                        "in the archive (default %(default)s)")
    g.add_argument("--keep-name", action="store_true",
                   help="ship the package under its own name (the archive is then not anonymous)")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        return tables(a.dumps)
    if a.cmd == "retrain":
        return retrain(a.run, a.execute)
    return supplement(a.out, runs_root=a.runs_root, anon_name=None if a.keep_name else a.anon_name)


if __name__ == "__main__":
    raise SystemExit(main())
