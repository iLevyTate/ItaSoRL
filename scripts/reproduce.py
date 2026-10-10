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
               under --runs-root (default fullruns/). Author names, handles, ORCID iDs,
               e-mail addresses, repository URLs, externally hosted images and local paths
               are scrubbed from text files. The package and project name is replaced by a
               neutral one in text and in file names (--anon-name, default anonpkg;
               --keep-name turns this off), and so are the titles of the publicly released
               films, read at build time from the served film files. Outreach and
               film-production specs and run-script figures are left out (SUPPLEMENT_EXCLUDE),
               and Markdown links into material the archive does not carry become plain text.
               Every shipped file, binaries included, is checked for identity strings.
               README_SUPPLEMENT.md says what the archive holds, how to install and run it,
               and which run bundles were included; PROVENANCE.md lists the first and last
               commit of every preregistration and spec, from git at build time; SHA256SUMS
               lists every other file. --require-runs fails the build when a RUN_DIRS bundle
               is absent.

Usage:
    python scripts/reproduce.py tables [--dumps fullruns/corrected_l3_h8_wm/states]
    python scripts/reproduce.py retrain C1 [--execute]
    python scripts/reproduce.py supplement [--out dist/supplement.zip] [--runs-root DIR]
                                           [--anon-name NAME | --keep-name] [--require-runs]
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import io
import os
import posixpath
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
_PROFILE = "Use" + "rs"
GENERIC_SCRUB = [
    # One or more separators: a Windows path inside a JSON string carries doubled backslashes.
    (re.compile(r"[A-Za-z]:[/\\]+" + _PROFILE + r"[/\\]+[^/\\\s\"']+", re.I), "<local-path>"),
    # The same profile written the POSIX way: Git Bash (a drive letter as /c), WSL (/mnt/c) and
    # macOS (no prefix). Case-sensitive, so a /users/ segment of a web URL is left alone.
    (re.compile(r"(?:/mnt/[a-z]|/[a-z])?/" + _PROFILE + r"/[^/\s\"']+"), "<local-path>"),
    # A profile path flattened into one name, as tools that key a folder by its path write it
    # (C, two dashes, the profile folder, the account, then the rest of the path).
    (re.compile(r"\b[A-Za-z]--" + _PROFILE + r"-[^-/\\\s\"']+"), "<local-path>"),
    (re.compile("/ho" + "me/" + r"[^/\s\"']+"), "<local-path>"),
    # The final label must be alphabetic. Without that, the project's own condition notation
    # ("train@0.45", "eval@0.45") and pinned package versions ("react@18.3.1") read as
    # addresses, and the anonymized archive shipped the frozen preregistration with its
    # conditions replaced by the placeholder.
    (re.compile(r"[\w.+-]+" + "@" + r"[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"), "<email>"),
    (re.compile("orc" + r"id\.org/[\d-]+X?", re.I), "orcid.org/<orcid>"),
]
_EMAIL_RULE = GENERIC_SCRUB[4]

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


_GH_REMOTE = re.compile(r"^(?:https?://(?:[^@/]+@)?(?:www\.)?" + _GH + r"\.com/|(?:ssh://)?git@"
                        + _GH + r"\.com[:/])([^/:\s]+)/([^/\s]+?)(?:\.git)?/?$", re.I)


def parse_github(url: str) -> tuple[str, str] | None:
    """(owner, repository) from a GitHub https or ssh URL; None for anything else (a local
    path, a mirror on another host)."""
    m = _GH_REMOTE.match((url or "").strip().strip('"'))
    return (m.group(1), m.group(2)) if m else None


def _remote() -> tuple[str, str] | None:
    """(owner, repository) of the GitHub repository: from the git remote when it is a GitHub
    URL, else from CITATION.cff's repository-code. A clone of a local checkout or a mirror has
    an origin that is not GitHub, and parsing that would turn path segments into identity
    terms. None when neither names a GitHub repository (an extracted archive)."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip()
    except Exception:
        url = ""
    found = parse_github(url)
    if found:
        return found
    cff = os.path.join(ROOT, "CITATION.cff")
    if os.path.exists(cff):
        for line in open(cff, encoding="utf-8"):
            m = re.match(r"\s*repository-code:\s*(\S+)", line)
            if m:
                return parse_github(m.group(1))
    return None


def identity_terms() -> list[str]:
    terms = []
    cff = os.path.join(ROOT, "CITATION.cff")
    if os.path.exists(cff):
        for line in open(cff, encoding="utf-8"):
            m = re.match(r"\s*-?\s*(family-names|given-names|alias|name):\s*\"?([^\"]+)\"?", line)
            if m and m.group(1) != "name":
                terms.append(m.group(2).strip())
            # The ORCID iD on its own, so a bare iD is scrubbed like a name (the generic rule
            # covers it only behind the orcid.org host).
            m = re.match(r"\s*-?\s*orcid:\s*\"?(?:\S*/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", line)
            if m:
                terms.append(m.group(1))
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


_FILM_FILE = re.compile(r"^([a-z0-9]+(?:-[a-z0-9]+)+?)-(?:web|poster|vertical|textless)"
                        r"\.(?:mp4|webm|mov|jpg|jpeg|png|gif)$")
_NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def film_slugs() -> list[str]:
    """Slugs of the publicly released films, read at build time from the names of the served
    film files (assets/film/<slug>-web.mp4, <slug>-poster.jpg), so this file names no film.
    Empty outside a checkout that carries them (an extracted archive)."""
    d = os.path.join(ROOT, "assets", "film")
    try:
        names = os.listdir(d)
    except OSError:
        return []
    return sorted({m.group(1) for n in names for m in [_FILM_FILE.match(n)] if m})


def film_rules(slugs: list[str] | None = None) -> list:
    """One rule per released film: its title in any capitalization, with spaces, hyphens or
    underscores between the words (title, file slug, Python name), becomes a neutral name
    ("film one", "film two", ... in slug order) with the same capitalization and separator.
    The words in lower case with plain spaces between them are ordinary prose, not the title,
    and are left alone. A film title and a release date are a direct search key to the public
    posts."""
    rules = []
    for i, slug in enumerate(slugs if slugs is not None else film_slugs()):
        words = [re.escape(w) for w in slug.split("-")]
        prose = r"(?!(?-i:" + r"\s+".join(words) + r")(?![A-Za-z0-9]))"
        pat = re.compile(r"(?<![A-Za-z0-9])" + prose + r"([\s_-]+)".join(words)
                         + r"(?![A-Za-z0-9])", re.I)
        neutral = ["film", _NUMBER_WORDS[i] if i < len(_NUMBER_WORDS) else str(i + 1)]

        def sub(m, neutral=neutral):
            s = m.group(0)
            sep = next((g for g in m.groups() if g), " ")
            sep = " " if sep.strip() == "" else sep[0]
            if s == s.lower():
                ws = neutral
            elif s == s.upper():
                ws = [w.upper() for w in neutral]
            else:
                ws = [w.capitalize() for w in neutral]
            return sep.join(ws)

        rules.append((pat, sub))
    # A release date written after "released" goes with the title it would identify.
    if rules:
        months = ("January|February|March|April|May|June|July|August|September|October|"
                  "November|December")
        rules.append((re.compile(r"\b(released(?:\s+on)?)\s+(?:\d{1,2}\s+(?:" + months
                                 + r")\s+\d{4}|\d{4}-\d\d-\d\d)"), r"\1 <date>"))
    return rules


def scrub_rules() -> list:
    """Hosted images first, then repository URLs, then the generic rules, then one rule per
    identity term (term_rules), then the film titles (film_rules). The order matters: see
    url_rules."""
    return (IMAGE_SCRUB + url_rules() + list(GENERIC_SCRUB) + term_rules(identity_terms())
            + film_rules())


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


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".pdf",
             ".eps", ".ico"}
# Every image the archive may carry, by archive path and the SHA-256 of the bytes a person
# looked at. Text drawn into an image (a figure title) is pixels: neither the scrub nor the
# byte scan can see it, and two run-script figures once shipped the project name that way.
# So an image ships only when its exact bytes were viewed and found free of names, handles and
# URLs. A new or redrawn image fails the leak check until someone views it and adds its hash
# here. Reviewed 2026-10-10: the eight images below, all redrawn title-free by
# scripts/plot_record_figures.py or scripts/build_budget_curve.py; none prints a name.
REVIEWED_IMAGES = {
    "artifacts/expB2/expB2_survival.png":
        "bcd0d1983e553c46ba83542c1e08eec8931686623348ff90ac6490f626ea0eec",
    "artifacts/expB2/expB2_survival_confirmatory_n3.png":
        "7f16672afc4beaead5fbbb3d3858a477c25b1b8181895c5d88ba3a38d9de7611",
    "docs/figures/budget_curve.png":
        "8c94728392c65a68597297e0f845c4ca8f51611f70b381c97ed0c1260752f29a",
    "docs/figures/expA_ceiling.png":
        "7b3a15fd6c5c42659bff105a5e7ed35c747ef9c28a12a556261e6a595ec7c779",
    "docs/figures/expA_L2_ceiling.png":
        "0fdaa99f143e82daaf8e4090891b021f7a53a0a67b2dfa70fa670fc5c4cb5668",
    "docs/figures/expB_channels.png":
        "f29e6146988e1a6e350f026e5aa22d45ecbc830513f89e55ec1afe08ec77fd55",
    "docs/figures/expB_incidental.png":
        "2082b8ace55c25ab1a905770e5163c2c72b715c80e17f6f491b96ea39be7eb59",
    "docs/figures/expB_kstep.png":
        "e05120842930f4790a2f2bd7b152185abab099bf907b2816e653e3c601a9fa0c",
}


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXT


def image_reviewed(data: bytes, reviewed=None) -> bool:
    """True when these exact image bytes are on the reviewed list (any path: the rename can
    move a file, and the same bytes show the same pixels)."""
    allowed = set((REVIEWED_IMAGES if reviewed is None else reviewed).values())
    return hashlib.sha256(data).hexdigest() in allowed


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
                 "viz/player/brain/brain.js", "viz/player/index.html",
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
# What the README says about a RUN_DIRS bundle that is absent, beyond "not included".
RUN_NOTES = {
    "fullruns/corrected_l3_h8_wm": "C1",
    "fullruns/corrected_l3_h8_nowm": "C2",
}

# Left out of the archive even though they sit under a packaged root, as (pattern, reason).
# Reproduction needs none of them. Patterns are fnmatch patterns on the repository path.
SUPPLEMENT_EXCLUDE = [
    # Outreach and film-production specs: audience, voice and music briefs for the public films.
    ("docs/specs/*outreach-video*", "outreach and film-production specs"),
    # Figures a run script drew inside a run bundle. Text drawn into an image is pixels that no
    # scrub can see, and no one has viewed these for names (REVIEWED_IMAGES); the numbers they
    # plot are in the JSON files beside them.
    ("artifacts/goal_stakes/*", "run-script figures not viewed for names"),
    ("fullruns/*", "run-script figures not viewed for names"),
]
_EXCLUDE_IMAGES_ONLY = {"run-script figures not viewed for names"}


def excluded(rel: str) -> str | None:
    """The reason `rel` is left out of the archive, or None when it ships."""
    for pat, why in SUPPLEMENT_EXCLUDE:
        if fnmatch.fnmatchcase(rel, pat) and (why not in _EXCLUDE_IMAGES_ONLY or is_image(rel)):
            return why
    return None


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


def _sources(runs_root: str | None = None, dropped: list | None = None):
    """(archive path, source path) for every file the archive carries, sorted, and the status
    of each RUN_DIRS entry as (entry, file count), the count None when the entry is absent.
    The repository part follows git, not the filesystem, so gitignored scratch output under
    the included directories (artifacts/clip_audit/, about 400 MB) never ships. A file that
    SUPPLEMENT_EXCLUDE names is left out and, when `dropped` is a list, recorded there as
    (path, reason)."""
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
    gone = []
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
    for rel in list(pairs):
        why = excluded(rel)
        if why:
            del pairs[rel]
            gone.append((rel, why))
    if dropped is not None:
        dropped.extend(sorted(gone))
    # A bundle's count is what ships from it, after anything left out.
    status = [(r, n if n is None else sum(1 for k in pairs if k.startswith(r + "/")))
              for r, n in status]
    return sorted(pairs.items()), status


def _files(runs_root: str | None = None) -> list[str]:
    """Archive paths of every file the supplement carries (see _sources)."""
    return [f for f, _ in _sources(runs_root)[0]]


README_NAME = "README_SUPPLEMENT.md"
PROVENANCE_NAME = "PROVENANCE.md"


def _readme(status, anon_name: str | None, *, dropped=(), films: int = 0,
            provenance: str | None = None) -> str:
    """README_SUPPLEMENT.md: what the archive holds and leaves out, how to install and run it,
    and the state of every run bundle. `provenance` is None when PROVENANCE.md was written,
    else the reason it was not."""
    L = ["# Anonymized supplement", "",
         "This archive holds the code, the tests, every committed artifact and the method and "
         "correction documents of the project, anonymized for review. The project's public "
         "site, its films and illustrated walkthroughs, the Colab links and the git history "
         "are not part of it, so links to them in the documents read as plain text. Start with "
         "section 1 of docs/REPRODUCE.md and skip its clone step: the folder this file sits in "
         "is the repository root.", "",
         "## Install and run", "",
         "- Python 3.10 to 3.13. Install the dependencies with "
         "`pip install -r requirements-dev.txt`.",
         "- `python -m pytest -q` runs the test suite, about 20 minutes on a laptop CPU.",
         "- `python scripts/audit_stats_recheck.py` runs the scalar audit, a few seconds.",
         "- `python scripts/reproduce.py tables` checks and rebuilds the tables and gates from "
         "the committed per-seed artifacts without training, then runs the audit.",
         "- `python scripts/reproduce.py retrain C1` prints the commands that retrain a run "
         "(hours of CPU; costs in docs/REPRODUCE.md section 2).", "",
         "## What is renamed or left out", ""]
    if anon_name:
        L.append(f"- The project and package name is replaced by the neutral name `{anon_name}` "
                 "in the code, the documents and the file names.")
    L.append(f"- Repository URLs read `{REPO_PLACEHOLDER}`. Author names, ORCID iDs, e-mail "
             "addresses and local paths read `<author>`, `<email>` and `<local-path>`, and "
             "externally hosted images are removed.")
    if films:
        names = [f"Film {w.capitalize()}" for w in _NUMBER_WORDS[:films]]
        names = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        L.append(f"- The titles of the project's public films read {names}, and their release "
                 "dates read `<date>`.")
    reasons = sorted({why for _, why in dropped})
    if reasons:
        L.append(f"- Left out ({len(dropped)} files): " + "; ".join(reasons) + ". Reproduction "
                 "needs none of them.")
    L.append("- Figures that run scripts draw inside run bundles (under artifacts/goal_stakes/ "
             "and fullruns/) are left out whenever present: no one has viewed them for names, "
             "and the numbers they plot are in the JSON files beside them.")
    L += ["", "## Run bundles", "",
          "Run bundles are the outputs of training runs (cells, saved agents, state dumps). They "
          "are not kept in version control, so the archive carries those present where it was "
          "built.", ""]
    absent = []
    for r, n in status:
        if n is not None:
            L.append(f"- included: `{r}` ({n} files)")
        else:
            L.append(f"- not included: `{r}`")
            absent.append(r)
    if not status:
        L.append("- none: this archive was built without run bundles")
    gone = [RUN_NOTES[r] for r in absent if r in RUN_NOTES]
    L.append("")
    if gone:
        both = " and ".join(gone)
        L += [f"The saved agents and state dumps of the corrected run{'s' if len(gone) > 1 else ''} "
              f"{both} are not in this archive. {both} {'were' if len(gone) > 1 else 'was'} "
              "trained in a 4-vCPU cloud container, and the bundles were not available where this "
              "archive was built. Their per-seed cell records are under "
              "artifacts/corrected_runs/, so every corrected table rebuilds without training. "
              "`python scripts/reproduce.py retrain C1 --execute` retrains C1 and then C2 (about "
              "3 h 20 min and 4 h 25 min on 4 CPU workers, docs/REPRODUCE.md section 2), and "
              "prints the readout-only follow-ups that need the saved agents.", ""]
    L.append("A run whose bundle is not listed as included has no saved agents or state dumps "
             "here; its committed per-seed records are under artifacts/.")
    L += ["", "## Files", ""]
    if provenance is None:
        L.append(f"- `{PROVENANCE_NAME}` lists the first and last commit of every "
                 "preregistration and spec in the archive, read from git when the archive was "
                 "built, so the freeze timing of each can be checked.")
    else:
        L.append(f"- There is no `{PROVENANCE_NAME}`: {provenance}.")
    L.append(f"- `SHA256SUMS` lists the SHA-256 of every file in the archive except itself, "
             f"`{README_NAME}` included.")
    return "\n".join(L) + "\n"


_MD_LINK = re.compile(r"(?<!!)\[((?:!\[[^\]]*\]\([^)]*\)|[^\[\]])*)\]\(\s*([^)\s]+)"
                      r"(?:\s+\"[^\"]*\")?\s*\)")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def delink(text: str, md_path: str, shipped: set) -> str:
    """A Markdown file with every link whose target the archive cannot resolve turned into its
    link text: a target that is the repository placeholder, or a relative path to a file or
    folder the archive does not carry. A badge (a link whose text is an image) is dropped.
    Web URLs and in-page anchors are kept."""
    folders = {posixpath.dirname(f) for f in shipped}
    folders |= {d for f in folders for d in _parents(f)}

    def dead(target: str) -> bool:
        if target.startswith(REPO_PLACEHOLDER):
            return True
        if _SCHEME.match(target) or target.startswith(("#", "/")):
            return False
        path = target.split("#")[0].split("?")[0]
        if not path:
            return False
        full = posixpath.normpath(posixpath.join(posixpath.dirname(md_path), path))
        return full.startswith("..") or (full not in shipped and full.rstrip("/") not in folders)

    def sub(m):
        label, target = m.group(1), m.group(2)
        if not dead(target):
            return m.group(0)
        return "" if label.lstrip().startswith("![") else label

    return _MD_LINK.sub(sub, text)


def _parents(d: str):
    while d:
        yield d
        d = posixpath.dirname(d)


def _git(args: list[str]) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


def provenance_rows(rels: list[str]):
    """([(path, first commit, first hash, last commit, last hash)], None), or (None, reason)
    when the history is not there to read: outside a git checkout, or in a shallow clone,
    whose oldest commit is the clone boundary and not when a file entered the history.
    Committer dates in UTC, no names or e-mail addresses."""
    if (_git(["rev-parse", "--is-inside-work-tree"]) or "").strip() != "true":
        return None, "the archive was not built from a git checkout"
    if (_git(["rev-parse", "--is-shallow-repository"]) or "").strip() == "true":
        return None, "the archive was built from a shallow clone, whose history is incomplete"

    def utc(ts: str) -> str:
        return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M")

    rows = []
    for rel in rels:
        log = [ln.split() for ln in (_git(["log", "--follow", "--format=%h %ct", "--", rel])
                                     or "").splitlines() if ln.strip()]
        if log:
            rows.append((rel, utc(log[-1][1]), log[-1][0], utc(log[0][1]), log[0][0]))
        else:
            rows.append((rel, "not committed", "", "not committed", ""))
    return rows, None


def provenance_md(rows, rename) -> str:
    out = ["# Provenance of the preregistrations and specs", "",
           "Read from the git history when this archive was built: the first and the last "
           "commit that touched each file, with committer dates in UTC and short commit hashes. "
           "Names and e-mail addresses are not included. The first commit is when the file "
           "entered the history. A later last commit is a later edit, and the file's own dated "
           "entries say what changed.", "",
           "| File | First commit (UTC) | Hash | Last commit (UTC) | Hash |",
           "|---|---|---|---|---|"]
    for rel, first, fh, last, lh in rows:
        out.append(f"| `{rename(rel)}` | {first} | {fh} | {last} | {lh} |")
    return "\n".join(out) + "\n"


def _is_frozen_doc(rel: str) -> bool:
    return (fnmatch.fnmatchcase(rel, "docs/PREREGISTRATION*.md")
            or fnmatch.fnmatchcase(rel, "docs/specs/*.md"))


# Account names too generic to be a leak check (cloud, CI and container defaults).
_GENERIC_ACCOUNTS = {"user", "runner", "root", "admin", "administrator", "ubuntu", "vscode",
                     "jovyan", "codespace", "codespaces", "content", "home", "default", "public",
                     "guest", "build", "docker", "app"}


def local_account() -> str | None:
    """The build machine's account name, as a leak-only check: the scrub rules replace local
    paths, and this catches a form they miss. None when it is too generic to check."""
    name = os.path.basename(os.path.normpath(os.path.expanduser("~")))
    return name if len(name) >= 4 and name.lower() not in _GENERIC_ACCOUNTS else None


def paper_title() -> str | None:
    """The manuscript's main title (before any colon), read at build time from the local,
    untracked LaTeX source when it exists, so no tracked file spells it."""
    path = os.path.join(ROOT, "docs", "paper", "main.tex")
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = re.search(r"\\title\{([^}:%\\]+)", text)
    title = " ".join(m.group(1).split()) if m else ""
    return title if len(title) >= 8 else None


def _leak_patterns(old_name: str | None) -> list:
    """Leak-only checks beyond the scrub rules: any GitHub Pages host or hosted-image URL that
    survived, and, when the package was renamed, any spelling of the original name; the build
    machine's account name; and the manuscript's title when its local source is present."""
    pats = [re.compile(r"[\w-]+\." + _GH + r"\.io\b", re.I), re.compile(_HOSTED, re.I)]
    if old_name:
        pats.append(re.compile(re.escape(old_name), re.I))
    acct = local_account()
    if acct:
        pats.append(re.compile(r"(?<![A-Za-z0-9])" + re.escape(acct) + r"(?![A-Za-z0-9])", re.I))
    title = paper_title()
    if title:
        pats.append(re.compile(r"\s+".join(re.escape(w) for w in title.split()), re.I))
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


def leak_scan(path: str, rules, old_name: str | None, reviewed=None) -> list[str]:
    """Members of the archive at `path` that still carry an identity string, or may. Text
    members are scanned with every scrub rule; binary members (agent bundles, arrays, images)
    with every rule except the e-mail shape, which arbitrary bytes match by chance about once
    per few tens of MB. Member names are scanned too. An image member is a leak unless its
    bytes are on the reviewed list (REVIEWED_IMAGES, or `reviewed`), because text drawn into
    an image is invisible to every pattern."""
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
            elif is_image(n) and not image_reviewed(data, reviewed):
                leaks.append(n)
            elif _binary_hits(data, bin_pats):
                leaks.append(n)
    return leaks


def supplement(out: str, runs_root: str | None = None,
               anon_name: str | None = DEFAULT_ANON_NAME, *, require_runs: bool = False) -> int:
    old_name = project_name() if anon_name else None
    if anon_name:
        check_anon_name(anon_name, old_name)
    if old_name and old_name.lower() == anon_name.lower():
        old_name = None          # already neutral (a rebuild from an extracted archive)
    rename_pkg = renamer(old_name, anon_name)
    films = film_rules()
    rules = scrub_rules()

    def rename(path_or_text: str) -> str:
        """Package name and film titles, as in text, so a path a document quotes still
        resolves in the archive."""
        return rename_pkg(scrub(path_or_text, films))

    dropped: list = []
    pairs, status = _sources(runs_root, dropped)
    print(f"run bundles under {runs_root or os.path.join(ROOT, 'fullruns')}:")
    for r, n in status:
        if n is None:
            print(f"  WARNING: {r} is absent; the archive will NOT carry it", flush=True)
        else:
            print(f"  included {r} ({n} files)")
    missing = [r for r, n in status if n is None]
    if require_runs and missing:
        print(f"--require-runs: {len(missing)} run bundle(s) absent; no archive written")
        return 1
    for rel, why in dropped:
        print(f"  left out {rel} ({why})")
    rows, no_prov = provenance_rows([rel for rel, _ in pairs if _is_frozen_doc(rel)])
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)) or ".", exist_ok=True)
    shipped = {rename(rel) for rel, _ in pairs}
    sums = []
    with zipfile.ZipFile(os.path.join(ROOT, out), "w", zipfile.ZIP_DEFLATED) as z:
        def put(arc: str, data: bytes) -> None:
            z.writestr(arc, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")

        for rel, src in pairs:
            with open(src, "rb") as fh:
                data = fh.read()
            arc = rename(rel)
            if is_text(rel):
                text = rename_pkg(scrub(data.decode("utf-8", errors="replace"), rules))
                if arc.endswith(".md"):
                    text = delink(text, arc, shipped)
                data = text.encode("utf-8")
            put(arc, data)
        if rows is not None:
            put(PROVENANCE_NAME, provenance_md(rows, rename).encode("utf-8"))
        put(README_NAME, _readme(status, anon_name, dropped=dropped,
                                 films=len(film_slugs()),
                                 provenance=no_prov).encode("utf-8"))
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    leaks = leak_scan(os.path.join(ROOT, out), rules, old_name)
    unviewed = [n for n in leaks if is_image(n)]
    if unviewed:
        print(f"  images not on the reviewed list (view each, then add its SHA-256 to "
              f"REVIEWED_IMAGES in scripts/reproduce.py): {unviewed}")
    print(f"wrote {out}: {len(sums) + 1} files ({len(sums)} listed in SHA256SUMS)"
          + (f"; POSSIBLE IDENTITY LEAKS in {leaks}" if leaks
             else "; no identity strings found in any file, binaries included, and every image "
                  "is on the reviewed list"))
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
    s.add_argument("--require-runs", action="store_true",
                   help="fail, writing nothing, when a run bundle named in RUN_DIRS is absent "
                        "(for the submission build)")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        return tables(a.dumps)
    if a.cmd == "retrain":
        return retrain(a.run, a.execute)
    return supplement(a.out, runs_root=a.runs_root, anon_name=None if a.keep_name else a.anon_name,
                      require_runs=a.require_runs)


if __name__ == "__main__":
    raise SystemExit(main())
