"""One entry point for reproducing the results (revision step 15; docs/REPRODUCE.md).

  tables       No training. Rebuild the results manifest, the gate table, and the contrast
               intervals from committed per-seed artifacts, then run the scalar audit. The
               builders first compare the committed pages with their own output; a stale page
               is rewritten, and the command then exits 3 even when the audit passes on the
               rewritten copy. With --check nothing is rewritten: the audit reads the pages as
               shipped and fails on a stale page as well, and a stale page exits 3. With
               --dumps DIR (state dumps written by a run, fullruns/<run>/states), also
               recompute every per-seed readout from the raw recurrent states under both fold
               partitions (scripts/rescore_fold_split.py) and the behavior control
               (scripts/audit_behavior_mediation.py), into dist/fold_rescore_<run>.json and
               dist/behavior_audit_<run>.json. Minutes on a laptop.
  retrain RUN  Print, or with --execute run, the exact commands that retrain a run listed in
               docs/RESULTS_MANIFEST.md. Hours of CPU per run (estimates in docs/REPRODUCE.md).
               Every output, the printed follow-up readouts included, goes under --dest
               (default dist/retrain_<UTC date>), never under artifacts/ or docs/figures/. A
               recipe that runs a bash script needs Git Bash on Windows (git_bash()).
  supplement   Build an anonymized submission archive under dist/: code, tests, committed
               artifacts, the method documents, and the run bundles of the chosen profile that
               exist under --runs-root (default fullruns/). --profile full (the default) takes
               every bundle in RUN_DIRS; --profile review takes only REVIEW_RUN_DIRS, the
               held-out and seed-2 saved agents and the saved agents and cells of the C1 and
               C2 reruns, and fails, writing nothing, when the archive exceeds REVIEW_LIMIT
               (100 MB, 100,000,000 bytes). Author names, handles, ORCID
               iDs, e-mail addresses, repository URLs, externally hosted images and local paths
               are scrubbed from text files. Commit IDs, in either case, and SHA-256 digests
               become keyed pseudonyms of the same length in text and in file names
               (CommitPseudonyms); DIGEST_PSEUDONYMS.json lets a check map a digest it
               computes to the recorded pseudonym (archive_digest). Branch names become
               anon-branch-N in text and in file names (branch_rules), and the identity
               check fails on any branch name that survives. The package and project
               name is replaced by a neutral one in text and in file names (--anon-name,
               default anonpkg; --keep-name turns this off), and so are the titles of the
               publicly released films, read at build time from the served film files.
               Outreach and film-production specs and run-script figures are left out
               (SUPPLEMENT_EXCLUDE), and Markdown links into material the archive does not
               carry become plain text. Every shipped file, binaries included, is checked for
               identity strings (the manuscript's title among them; --paper names its source,
               and a review build refuses to start without one), for any commit ID of this
               repository and for any real digest a tracked file holds; a pseudonym that
               changed a number fails the build too. A failed build leaves no archive and no
               map, not even those of an earlier build at the same path, and a build from a
               shallow, partial or single-branch clone, or one that cannot list the remote's
               branches, refuses to start.
               README_SUPPLEMENT.md says what the archive holds, how to install and run it,
               and which run bundles were included; PROVENANCE.md lists every commit to
               each preregistration and spec (UTC date and time, pseudonymized short hash, a
               count per file), from git at build time; PREREG_HISTORY/ holds the full text
               history of each one as a patch (prereg_history), scrubbed and pseudonymized like
               every other text file; SHA256SUMS lists every other file. PROVENANCE.md and
               PREREG_HISTORY/ are not written from a shallow clone.
               --require-runs fails the build when a bundle of the profile is absent.

Usage:
    python scripts/reproduce.py tables [--check] [--dumps fullruns/corrected_l3_h8_wm/states]
    python scripts/reproduce.py retrain C1 [--execute] [--dest DIR]
    python scripts/reproduce.py supplement [--profile full|review] [--out FILE] [--runs-root DIR]
                                           [--anon-name NAME | --keep-name] [--require-runs]
                                           [--paper PATH]
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import fnmatch
import hashlib
import hmac
import io
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable

L3 = "python scripts/run_expB2.py --drift-mode l3 --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --device cpu"
# Retraining recipes. `{dest}` is the retrain folder (default dist/retrain_<UTC date>, see
# retrain_dest()), so a retrain never writes into artifacts/, docs/figures/ or the run bundles
# under fullruns/ that the supplement ships. A line that starts with "#" is a note: printed,
# never run.
RETRAIN = {
    "C1": ["bash scripts/revision/run_corrected_confirmation.sh {dest} 4 wm"],
    "C2": ["bash scripts/revision/run_corrected_confirmation.sh {dest} 4 nowm"],
    "L3-H8-WM-CPU": [f"{L3} --l3-hidden 8 --gae-bootstrap pre_transition --workers 3 "
                     "--save-agents --dump-states fullruns/l3_h8_wm_cpu/states --out-dir fullruns/l3_h8_wm_cpu"],
    "L3-H8-NOWM-CPU": [f"{L3} --l3-hidden 8 --no-world-model --gae-bootstrap pre_transition --workers 3 "
                       "--save-agents --dump-states fullruns/l3_h8_nowm/states --out-dir fullruns/l3_h8_nowm"],
    "L3-H10-GS1-CPU": [f"{L3} --l3-hidden 10 --l3-seed 1 --gae-bootstrap pre_transition --workers 3 "
                       "--save-agents --dump-states fullruns/l3_h10_gseed1/states --out-dir fullruns/l3_h10_gseed1"],
    "L3-H8-N10": [f"{L3.replace('--device cpu', '--device cuda')} --l3-hidden 8 --gae-bootstrap "
                  "pre_transition --dump-states fullruns/l3_h8_traces/states --out-dir fullruns/l3_h8_traces"
                  "   # published on GPU; ITASORL_FOLDS=legacy on the publishing stack reproduces 0.752"],
    # docs/specs/2026-10-15-corrected-l1-organism-rerun-design.md, "Run"; only the output
    # paths differ.
    "L1-CORRECTED": ["python scripts/run_expB2.py --drift-mode l1 --l1-delta 0.023 --sensor-sigma 0.01 "
                     "--drifts 0.0 0.023 --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 "
                     "--gae-bootstrap successor --workers 3 --device cpu --save-agents --resume "
                     "--dump-states {dest}/fullruns/corrected_l1_heldout/states "
                     "--out-dir {dest}/fullruns/corrected_l1_heldout"],
    # docs/specs/2026-10-12-balanced-confirmation-design.md, "Commands"; only the output paths
    # differ.
    "B1-BALANCED": ["# Both runs are one worker on CUDA, so G and the agents train on the same device "
                    "(the spec's run section). ITASORL_FOLDS=explicit, the default.",
                    "python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 10 11 12 "
                    "13 14 15 16 17 18 19 --gae-bootstrap successor --adv-norm historical "
                    "--balanced-seed-base 910000 --save-agents "
                    "--dump-states {dest}/fullruns/balanced_confirmation_wm/states "
                    "--out-dir {dest}/fullruns/balanced_confirmation_wm --device cuda --workers 1"],
    "B2-BALANCED": ["# Both runs are one worker on CUDA, so G and the agents train on the same device "
                    "(the spec's run section). ITASORL_FOLDS=explicit, the default.",
                    "python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 10 11 12 "
                    "13 14 15 16 17 18 19 --objective touch --mortal off --gae-bootstrap successor "
                    "--adv-norm historical --balanced-seed-base 910000 --save-agents "
                    "--dump-states {dest}/fullruns/balanced_confirmation_touch/states "
                    "--out-dir {dest}/fullruns/balanced_confirmation_touch --device cuda --workers 1"],
    "PB1": ["# The prospective confirmation of the matched-world sensory control (spec 2026-10-20): "
            "fresh seeds 20 to 29, matched worlds at base 3000000, one worker on CUDA. "
            "ITASORL_FOLDS=explicit, the default.",
            "python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 20 21 22 "
            "23 24 25 26 27 28 29 --gae-bootstrap successor --adv-norm historical "
            "--balanced-seed-base 3000000 --save-agents --device cuda --workers 1 "
            "--dump-states {dest}/fullruns/prospective_balanced_wm/states "
            "--out-dir {dest}/fullruns/prospective_balanced_wm"],
    "PB2": ["# As PB1, with the pellet objective and death off (the spec's second command).",
            "python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 20 21 22 "
            "23 24 25 26 27 28 29 --gae-bootstrap successor --adv-norm historical "
            "--balanced-seed-base 3000000 --save-agents --device cuda --workers 1 "
            "--objective touch --mortal off "
            "--dump-states {dest}/fullruns/prospective_balanced_touch/states "
            "--out-dir {dest}/fullruns/prospective_balanced_touch"],
}
# Run-specific flags of the follow-ups that retrain the survival arm: C2 has no decoder.
RUN_FLAGS = {"C1": "", "C2": " --no-world-model"}
# Readout-only follow-ups on a retrained run's saved agents. Every output goes under
# {dest}/readouts/, with the relative name the committed file has under artifacts/, so a
# retrain never overwrites a committed artifact and compare_run_cells.py can set the two
# side by side.
READOUTS = [
    "python scripts/run_l0_audit.py --agents-dir {run}/agents --out {dest}/readouts/l0_audit/{name}.json",
    "python scripts/run_policy_controlled_readouts.py --run-dir {run}{flags} --out {dest}/readouts/policy_controls/{name}.json --workers 4",
    "python scripts/run_persistence_readout.py --agents-dir {run}/agents --out {dest}/readouts/persistence/{name}.json --workers 4",
    "python scripts/run_control_diagnostics.py --run-dir {run} --out {dest}/readouts/control_diagnostics/{name}.json --workers 4",
    "python scripts/run_texture_fresh_probe.py --agents-dir {run}/agents --family gn --param 0.01 --out {dest}/readouts/texture/{name}_gn.json --workers 4",
    "python scripts/run_texture_fresh_probe.py --agents-dir {run}/agents --family qd --param 6.0 --out {dest}/readouts/texture/{name}_qd.json --workers 4",
    "python scripts/validate_population_readout.py --run-dir {run} --out {dest}/readouts/population_readout/{name}.json",
    "python scripts/compare_run_cells.py --rerun {dest}/corrected_runs/{name}/cells "
    "--committed artifacts/corrected_runs/{name}/cells --out {dest}/readouts/compare_cells_{name}.json"
    "   # value by value against the committed cells",
]
# After both corrected runs: the budget curve and the exploratory cross-run replay (FINDINGS
# 17.4, 17.10). The frozen-rule verdicts read the committed cells only, so the first line checks
# that they still match them and rewrites nothing.
AFTER_BOTH = [
    "python scripts/build_corrected_verdicts.py --check",
    'python scripts/build_budget_curve.py --run "decoder on={dest}/corrected_runs/corrected_l3_h8_wm" '
    '--run "decoder off={dest}/corrected_runs/corrected_l3_h8_nowm" --out {dest}/readouts/budget_curve.json '
    "--figure {dest}/readouts/budget_curve.png",
    "python scripts/run_cross_replay.py --a {dest}/fullruns/corrected_l3_h8_wm "
    "--b {dest}/fullruns/corrected_l3_h8_nowm "
    "--out {dest}/readouts/cross_replay/corrected_c1_c2.json   # exploratory, post hoc",
]
CORRECTED_RUN_DIRS = {"C1": "corrected_l3_h8_wm", "C2": "corrected_l3_h8_nowm"}


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


BUILDERS = ("build_results_manifest.py", "build_gate_table.py", "build_contrast_intervals.py",
            "build_corrected_verdicts.py", "build_paper_tables.py")


STALE_EXIT = 3     # tables(): the shipped pages were stale here; they were rewritten


def tables(dumps: list[str], check_only: bool = False) -> int:
    # Say first whether the committed pages already match what the builders produce here, so a
    # rewrite that changes a tracked file is visible as such rather than silently dirtying it.
    stale = 0
    for script in BUILDERS:
        stale |= _run([PY, os.path.join("scripts", script), "--check"])
    print("committed tables and pages: " + ("current" if not stale else
                                            "STALE (not rewritten)" if check_only else
                                            "stale here; rewriting"), flush=True)
    rc = 0
    # With --check nothing is rewritten, so the audit reads the pages as shipped.
    for script in () if check_only else BUILDERS:
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
    # Without --check a stale page is rewritten and the audit passes on the rewritten copy. The
    # exit code still reports it (2026-10-16 review), so 0 means the shipped pages were current.
    # With --check the audit runs every builder's --check itself, so a stale page fails it too;
    # that failure is the same staleness, and the exit code names it.
    if stale and (check_only or not rc):
        print(f"exit {STALE_EXIT}: the committed tables and pages were stale"
              + (" (the audit output above names any other failing check)" if check_only
                 else " before the rewrite"), flush=True)
        rc = STALE_EXIT
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


def retrain_dest() -> str:
    """The default retrain folder, dist/retrain_<UTC date> (dist/ is gitignored)."""
    return "dist/retrain_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")


def _fill(template: str, **kw) -> str:
    return template.format(**kw)


def retrain_lines(run: str, dest: str) -> tuple[list[str], list[str]]:
    """(recipe, follow-ups) for `run` with every `{dest}` filled in: the commands that retrain
    it, then, for C1 and C2, the readout-only follow-ups on its saved agents and the steps
    that need both runs. KeyError for a run with no recipe."""
    recipe = [_fill(c, dest=dest) for c in RETRAIN[run]]
    follow: list[str] = []
    name = CORRECTED_RUN_DIRS.get(run)
    if name:
        run_dir = f"{dest}/fullruns/{name}"
        follow.append("# readout-only follow-ups on the saved agents (revision steps 5 to 12):")
        follow += [_fill(r, run=run_dir, name=name, dest=dest, flags=RUN_FLAGS[run])
                   for r in READOUTS]
        follow.append("# once both C1 and C2 exist:")
        follow += [_fill(r, dest=dest) for r in AFTER_BOTH]
    return recipe, follow


def git_bash() -> str | None:
    """The bash that runs the retrain scripts, or None when there is none to trust. On Windows
    the `bash` on PATH is often the WSL launcher (System32) or the Store alias (WindowsApps),
    which run another system with another python, so both are rejected and Git for Windows'
    bash under %ProgramFiles% is tried instead."""
    found = shutil.which("bash")
    if os.name != "nt":
        return found

    def trusted(p: str) -> bool:
        parts = os.path.normcase(os.path.abspath(p)).split(os.sep)
        return "system32" not in parts and "windowsapps" not in parts

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cands = [found, os.path.join(program_files, "Git", "bin", "bash.exe")]
    return next((c for c in cands if c and os.path.isfile(c) and trusted(c)), None)


def bash_preflight(bash: str) -> str | None:
    """Why the python that `bash` finds cannot run a retrain (it cannot import torch), or
    None when it can."""
    try:
        res = subprocess.run([bash, "-c", "python -c 'import torch'"], cwd=ROOT,
                             capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{bash} could not run python ({e})"
    if res.returncode == 0:
        return None
    last = (res.stderr or res.stdout).strip().splitlines()
    return f"`python -c 'import torch'` inside {bash} exited {res.returncode}" + (
        f": {last[-1]}" if last else "")


def _dest_refusal(dest: str) -> str | None:
    """Why `dest` cannot take a retrain, or None: the repository root, or a folder inside
    artifacts/ or docs/, whose files are the committed record."""
    full = os.path.normcase(os.path.abspath(os.path.join(ROOT, dest)))
    root = os.path.normcase(os.path.abspath(ROOT))
    if full == root:
        return "it is the repository root"
    for d in ("artifacts", "docs"):
        sub = os.path.join(root, d)
        if full == sub or full.startswith(sub + os.sep):
            return f"it is inside {d}/, which holds the committed record"
    return None


def retrain(run: str, execute: bool, dest: str | None = None) -> int:
    if run not in RETRAIN:
        print(f"no retraining recipe recorded for {run!r}; known: {sorted(RETRAIN)}")
        print("(every run's configuration is in docs/RESULTS_MANIFEST.md)")
        return 1
    dest = dest or retrain_dest()
    recipe, follow = retrain_lines(run, dest)
    for c in recipe:
        print(c)
    if follow:
        print()
        for r in follow:
            print(("\n" if r.startswith("# once") else "") + r)
    if not execute:
        return 0
    why = _dest_refusal(dest)
    if why:
        print(f"not run: the retrain folder {dest!r} is refused because {why}")
        return 1
    steps = [c for c in RETRAIN[run] if not c.lstrip().startswith("#")]
    bash = None
    if any(c.startswith("bash ") for c in steps):
        bash = git_bash()
        if bash is None:
            print("not run: no usable bash. On Windows install Git for Windows (its bash.exe under "
                  "%ProgramFiles%\\Git\\bin); the WSL and Store launchers are not used.")
            return 1
        why = bash_preflight(bash)
        if why:
            print(f"not run: {why}. Activate the environment that has the requirements first.")
            return 1
    for c in steps:
        # Split the template before filling it in, so a destination with spaces or backslashes
        # stays one argument; an argv list, never a shell string.
        argv = [_fill(tok, dest=dest) for tok in shlex.split(c.split("   #")[0])]
        argv[0] = {"bash": bash, "python": PY}.get(argv[0]) or argv[0]
        print("+", " ".join(argv), flush=True)
        rc = subprocess.call(argv, cwd=ROOT)
        if rc:
            print(f"exit {rc}; stopped")
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
    In a git checkout only the committed files count, so a local, unpublished render in the
    same folder adds no name. Empty where neither exists (an extracted archive)."""
    listed = _git_files()
    if listed is not None:
        names = [posixpath.basename(f) for f in listed if f.startswith("assets/film/")]
    else:
        try:
            names = os.listdir(os.path.join(ROOT, "assets", "film"))
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


# Branch names. A pull request keeps the name of the branch it came from, and GitHub's search
# finds a pull request by that name across every public repository: on 2026-10-16 one search on
# any of four branch names the review archive carried returned only this project's repository.
# Every branch this checkout knows (local branches, remote-tracking branches and, at build time,
# each remote's branches as `git ls-remote` lists them) becomes `anon-branch-N` in text and in
# file names, and so does any agent-session branch name (the session prefix and a slash), which
# catches a branch deleted from the remote since. A pull request outlives its branch, so the
# names of deleted branches count too: every name a merge commit's subject carries ("Merge pull
# request #N from <owner>/<name>", "Merge branch '<name>' into <name>"), read offline from the
# whole history, and at build time every pull request's head as `gh pr list` lists it, which
# adds the squash-merged ones no merge commit names (third review, 2026-10-16: four deleted
# branch names had survived in the review archive, each found by one search). Names every
# repository has (main, gh-pages) name nothing and stay. The leak scan looks for every name
# again in any case, so a spelling the replacement missed fails the build.
GENERIC_BRANCHES = {"main", "master", "gh-pages", "HEAD"}
BRANCH_PSEUDONYM = "anon-branch-{}"
# Assembled from pieces, as above: this file ships, and the pattern must not match itself.
_SESSION_BRANCH = "cla" + "ude/" + r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"


def _branch_listing(text: str | None) -> set[str]:
    """Branch names from `git for-each-ref` lines (refs/heads/..., refs/remotes/<remote>/...)
    or `git ls-remote --heads` lines (<id> refs/heads/...)."""
    out = set()
    for ln in (text or "").splitlines():
        ref = ln.split()[-1] if ln.split() else ""
        if ref.startswith("refs/heads/"):
            out.add(ref[len("refs/heads/"):])
        elif ref.startswith("refs/remotes/") and "/" in ref[len("refs/remotes/"):]:
            out.add(ref[len("refs/remotes/"):].split("/", 1)[1])
    return out


_MERGE_PR = re.compile(r"^Merge pull request #\d+ from [^/\s]+/(\S+)")
_MERGE_BRANCH = re.compile(r"^Merge (remote-tracking )?branch(?:es)? ((?:'[^']+'(?:,? and |, )?)+)"
                           r"(?: of \S+)?(?: into (\S+))?")


def _merge_listing(text: str | None) -> set[str]:
    """Branch names from commit subjects, one per line: the head of "Merge pull request #N from
    <owner>/<name>", and both sides of "Merge [remote-tracking] branch '<name>' into <name>"
    (a remote-tracking name loses its remote, the first part)."""
    out = set()
    for ln in (text or "").splitlines():
        m = _MERGE_PR.match(ln)
        if m:
            out.add(m.group(1))
            continue
        m = _MERGE_BRANCH.match(ln)
        if m:
            for name in re.findall(r"'([^']+)'", m.group(2)):
                out.add(name.split("/", 1)[1] if m.group(1) and "/" in name else name)
            if m.group(3):
                out.add(m.group(3))
    return out


def _pr_heads() -> str | None:
    """`gh pr list --state all` head branch names, one per line, or None when gh is missing,
    not signed in, or cannot reach the host in 60 s."""
    try:
        res = subprocess.run(["gh", "pr", "list", "--state", "all", "--limit", "10000",
                              "--json", "headRefName", "--jq", ".[].headRefName"], cwd=ROOT,
                             capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return res.stdout


def branch_names(remote: bool = False) -> list[str]:
    """Every branch name this checkout knows, GENERIC_BRANCHES aside, sorted: local and
    remote-tracking branches, and every name a merge commit's subject carries in the whole
    history (_merge_listing; a deleted branch keeps its pull request). With `remote`, also each
    remote's branches as `git ls-remote` lists them (a branch pushed from elsewhere and never
    fetched) and every pull request's head as `gh pr list` lists it (a squash merge leaves no
    merge subject); a warning says when gh cannot list them. Empty outside a checkout."""
    names = _branch_listing(_git(["for-each-ref", "--format=%(refname)", "refs/heads",
                                  "refs/remotes"]))
    names |= _merge_listing(_git(["log", "--all", "--format=%s"]))
    if remote:
        for r in (_git(["remote"]) or "").split():
            names |= _branch_listing(_ls_remote(r))
        heads = _pr_heads()
        if heads is None:
            print("  WARNING: `gh pr list` could not list the pull requests, so the head names "
                  "of squash-merged pull requests whose branch is deleted are neither replaced "
                  "nor looked for", flush=True)
        names |= set((heads or "").split())
    return sorted(n for n in names if n and n not in GENERIC_BRANCHES)


class BranchPseudonyms:
    """A scrub rule's replacement: each known branch name becomes BRANCH_PSEUDONYM with its
    index in the sorted list, and an agent-session name the list lacks takes the next free
    index on first sight. `used` records pseudonym -> name."""

    def __init__(self, names):
        self.index = {n: i + 1 for i, n in enumerate(sorted(set(names)))}
        self.used: dict[str, str] = {}

    def __call__(self, m) -> str:
        name = m.group(0)
        if name not in self.index:
            self.index[name] = len(self.index) + 1
        p = BRANCH_PSEUDONYM.format(self.index[name])
        self.used[p] = name
        return p


def _branch_pattern(names, flags: int = 0):
    alts = [re.escape(n) for n in sorted(set(names), key=len, reverse=True)]
    alts.append(r"(?<!\.)" + _SESSION_BRANCH)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alts) + r")(?![A-Za-z0-9])", flags)


def branch_rules(names=None) -> list:
    """One rule that replaces every branch name in `names` (default branch_names()) and every
    agent-session branch name by its pseudonym (BranchPseudonyms). A name matches as a whole
    token: a letter or digit on either side makes it part of something else."""
    names = branch_names() if names is None else names
    return [(_branch_pattern(names), BranchPseudonyms(names))]


def scrub_rules(branches=None) -> list:
    """Hosted images first, then repository URLs, then the generic rules, then one rule per
    identity term (term_rules), then the film titles (film_rules), then the branch names
    (branch_rules; `branches` as there). The order matters: see url_rules."""
    return (IMAGE_SCRUB + url_rules() + list(GENERIC_SCRUB) + term_rules(identity_terms())
            + film_rules() + branch_rules(branches))


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


TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".sh", ".toml", ".ini", ".cfg", ".log", ".yml",
            ".yaml", ".cff", ".html", ".js", ".css", ".err", ".tex", ".bib", ".csv", ".ipynb",
            ".patch"}


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
    # Inside an extracted archive the listed digests are pseudonyms (archive_digest).
    return archive_digest(hashlib.sha256(data).hexdigest()) in allowed


INCLUDE_DIRS = ["itasorl", "scripts", "tests", "artifacts"]
# Document folders shipped whole (tracked files only). docs/optional holds the optional
# readouts; the manuscript cites O2 in its Limitations and claims row 46, so it ships
# (2026-10-16 review), its provenance commits pseudonymized like every other commit field.
DOC_DIRS = ["docs/specs", "docs/figures", "docs/optional"]
INCLUDE_FILES = ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "pytest.ini",
                 "ruff.toml", "LICENSE",
                 "constraints-cpu-corrected.txt", "constraints-workstation.txt",
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
# artifacts/corrected_runs/. The saved agents of the historical hidden-8 held-out run (60 files)
# back the direction-of-rearing readout and the drift-0.45 world-sample check
# (artifacts/l0_audit/reverse_direction_l3_h8_heldout.json, d045_world_samples_l3_h8_heldout.json).
# The hidden-8 seed-2 GPU run's agents (60 files; the 30 drift-0 files equal the held-out run's)
# back no committed readout and have no integrity reference of their own; both readout scripts
# gate against the held-out run's targets. A bundle that is absent is reported on the console
# and in README_SUPPLEMENT.md, never skipped silently.
RUN_DIRS = ["fullruns/corrected_l3_h8_wm", "fullruns/corrected_l3_h8_nowm",
            "fullruns/l3_h8_heldout/agents", "fullruns/l3_h8_gseed2_gpu/agents",
            "fullruns/balanced_confirmation_wm", "fullruns/balanced_confirmation_touch",
            "fullruns/corrected_l1_heldout",
            "fullruns/prospective_balanced_wm", "fullruns/prospective_balanced_touch"]
# What the README says about a RUN_DIRS bundle that is absent, beyond "not included".
RUN_NOTES = {
    "fullruns/corrected_l3_h8_wm": "C1",
    "fullruns/corrected_l3_h8_nowm": "C2",
    "fullruns/balanced_confirmation_wm": (
        "B1 is absent. The matched-world survival confirmation and its sensory "
        "control cannot be replayed from this archive. The committed summaries are "
        "in artifacts/balanced_confirmation/."
    ),
    "fullruns/balanced_confirmation_touch": (
        "B2 is absent. The matched-world pellet confirmation and its sensory "
        "control cannot be replayed from this archive. The committed summaries are "
        "in artifacts/balanced_confirmation/."
    ),
    "fullruns/prospective_balanced_wm": (
        "PB1 is absent. The prospective survival confirmation, its matched gates and its "
        "sensory-echo readouts cannot be replayed from this archive. The committed summaries are "
        "in artifacts/prospective_balanced/."
    ),
    "fullruns/prospective_balanced_touch": (
        "PB2 is absent. The prospective pellet confirmation, its matched gates and its "
        "sensory-echo readouts cannot be replayed from this archive. The committed summaries are "
        "in artifacts/prospective_balanced/."
    ),
    "fullruns/corrected_l1_heldout": (
        "L1-CORRECTED is absent. Its stream and surprise readouts cannot be replayed from "
        "this archive. The committed cells are in artifacts/corrected_runs/l1-corrected/; "
        "the retraining command is in "
        "docs/specs/2026-10-15-corrected-l1-organism-rerun-design.md."
    ),
}

# What the README says about a RUN_DIRS bundle that is included: where its files came from.
# The C1 and C2 originals did not keep their agents; both were rerun on the same stack to keep
# them, and each rerun reproduces its original's committed cells.
RUN_INCLUDED_NOTES = {
    "fullruns/corrected_l3_h8_wm": (
        "`fullruns/corrected_l3_h8_wm` holds the cells and saved agents of a rerun of C1 on the "
        "same stack, made to keep the agents. Its pooled targets equal the committed C1 cells in "
        "60 of 60 arm-cells (docs/FINDINGS.md 18.1). The agents include the survival arm's "
        "budget snapshots. State dumps are not shipped: `ITASORL_FOLDS=explicit python "
        "scripts/regenerate_state_dumps.py --run-dir fullruns/corrected_l3_h8_wm --out-dir <dir> "
        "--report <json>` rebuilds them from the agents and checks every pool against its "
        "recorded target."),
    "fullruns/corrected_l3_h8_nowm": (
        "`fullruns/corrected_l3_h8_nowm` holds the cells and saved agents of a rerun of C2 on the "
        "same stack, made to keep the agents. All 5000 values its cells share with the committed "
        "C2 cells are bit-equal (docs/FINDINGS.md 20.1). State dumps rebuild from the agents as "
        "for C1."),
    "fullruns/l3_h8_heldout/agents": (
        "`fullruns/l3_h8_heldout/agents` holds the saved agents of the historical hidden-8 "
        "held-out run (pre-correction trainer). The direction-of-rearing and drift-0.45 "
        "world-sample readouts replay from them (docs/REPRODUCE.md, \"Replay from the "
        "archive\")."),
    "fullruns/l3_h8_gseed2_gpu/agents": (
        "`fullruns/l3_h8_gseed2_gpu/agents` holds the saved agents of the historical hidden-8, "
        "G seed 2 run on GPU (pre-correction trainer). They back no committed readout and have "
        "no integrity reference of their own; their 30 drift-0 files equal the held-out run's."),
    "fullruns/balanced_confirmation_wm": (
        "`fullruns/balanced_confirmation_wm` holds the B1 cells, saved agents, frozen "
        "normalizers and matched-world sensory-control output. Seed base 910000. "
        "Replay the matched gates with `python scripts/audit_balanced_matched_gates.py "
        "--run fullruns/balanced_confirmation_wm --objective survival --mortal on`. "
        "State dumps rebuild from the agents."),
    "fullruns/balanced_confirmation_touch": (
        "`fullruns/balanced_confirmation_touch` holds the B2 cells, saved agents, frozen "
        "normalizers and matched-world sensory-control output. Seed base 910000, pellet "
        "objective, death off. Replay the matched gates with "
        "`python scripts/audit_balanced_matched_gates.py --run "
        "fullruns/balanced_confirmation_touch --objective touch --mortal off`."),
    "fullruns/corrected_l1_heldout": (
        "`fullruns/corrected_l1_heldout` holds the L1-CORRECTED run folder: 20 cells, 60 saved "
        "agents and their 60 state dumps. Its cells and summaries are byte-identical to the "
        "committed artifacts/corrected_runs/l1-corrected/. The stream readout "
        "(artifacts/control_diagnostics/l1_stream_readout_corrected.json) and the surprise "
        "readout (artifacts/surprise/corrected_l1_heldout.json) replay from these agents and "
        "state dumps."),
    # The review build ships the C1 and C2 reruns' agents and cells as four entries (no state
    # dumps). Each note says where the folder came from and how its dumps rebuild.
    "fullruns/corrected_l3_h8_wm/agents": (
        "`fullruns/corrected_l3_h8_wm/agents` holds the saved agents of the rerun of C1, the "
        "corrected primary run, made on the same stack to keep the agents; they include the "
        "survival arm's budget snapshots. With the cells beside them, `ITASORL_FOLDS=explicit "
        "python scripts/regenerate_state_dumps.py --run-dir fullruns/corrected_l3_h8_wm "
        "--out-dir <dir> --report <json>` rebuilds the state dumps and checks every pool against "
        "the target recorded in its cell."),
    "fullruns/corrected_l3_h8_nowm/agents": (
        "`fullruns/corrected_l3_h8_nowm/agents` holds the saved agents of the rerun of C2, the "
        "corrected run without the auxiliary, made on the same stack to keep the agents. With "
        "the cells beside them, `ITASORL_FOLDS=explicit python scripts/regenerate_state_dumps.py "
        "--run-dir fullruns/corrected_l3_h8_nowm --out-dir <dir> --report <json>` rebuilds its "
        "state dumps as for C1."),
    "fullruns/corrected_l3_h8_wm/cells": (
        "`fullruns/corrected_l3_h8_wm/cells` holds the cell records of the C1 rerun that wrote "
        "these agents. They are not byte-identical to the committed C1 cells under "
        "artifacts/corrected_runs/, and scripts/regenerate_state_dumps.py reads its recorded "
        "targets from them, so they ship beside the agents."),
    "fullruns/corrected_l3_h8_nowm/cells": (
        "`fullruns/corrected_l3_h8_nowm/cells` holds the cell records of the C2 rerun that wrote "
        "these agents, which scripts/regenerate_state_dumps.py reads beside them."),
}

# The review build (--profile review). TMLR accepts at most 100 MB of supplementary material,
# and the full build was 501 MB on 2026-10-16, nearly all of it run bundles. The review build
# carries the saved agents of the held-out run, from which the direction-of-rearing and
# world-sample readouts replay, those of the seed-2 run, which back no committed readout, and
# the saved agents and cells of the C1 and C2 reruns
# (2026-10-17), from which a reviewer rebuilds their state dumps with
# scripts/regenerate_state_dumps.py, which opens <run>/cells/ beside <run>/agents/. The rerun
# cells differ byte for byte from the committed artifacts/corrected_runs cells, so the agents
# need their own. The limit is read in decimal units, 100,000,000 bytes, the smaller reading of
# "100MB" and the one the paper states.
REVIEW_RUN_DIRS = ["fullruns/l3_h8_heldout/agents", "fullruns/l3_h8_gseed2_gpu/agents",
                   "fullruns/corrected_l3_h8_wm/agents", "fullruns/corrected_l3_h8_nowm/agents",
                   "fullruns/corrected_l3_h8_wm/cells", "fullruns/corrected_l3_h8_nowm/cells"]
REVIEW_LIMIT = 100_000_000
PROFILES = ("full", "review")
# What the review README says about each full-build bundle it leaves out. The C1 and C2 bundles
# ship their agents and cells in the review build too; only their other files stay out.
REVIEW_LEFT_OUT = {
    "fullruns/corrected_l3_h8_wm": "C1, the corrected primary run: every file outside its agents/ "
                                   "and cells/ folders, which this build carries",
    "fullruns/corrected_l3_h8_nowm": "C2, the corrected run without the auxiliary: every file "
                                     "outside its agents/ and cells/ folders, which this build "
                                     "carries",
    "fullruns/balanced_confirmation_wm": "B1, the matched-world survival confirmation (cells, saved "
                                         "agents, state dumps)",
    "fullruns/balanced_confirmation_touch": "B2, the matched-world pellet confirmation (cells, saved "
                                            "agents, state dumps)",
    "fullruns/corrected_l1_heldout": "L1-CORRECTED (cells, saved agents, state dumps)",
    "fullruns/prospective_balanced_wm": "PB1, the prospective matched-world survival confirmation "
                                        "(cells, saved agents, state dumps, matched gates, "
                                        "sensory-echo readouts)",
    "fullruns/prospective_balanced_touch": "PB2, the prospective matched-world pellet confirmation "
                                           "(cells, saved agents, state dumps, matched gates, "
                                           "sensory-echo readouts)",
}


def profile_run_dirs(profile: str) -> list[str]:
    """The run bundles a build profile carries, read when called so tests can patch RUN_DIRS."""
    if profile not in PROFILES:
        raise SystemExit(f"--profile must be one of {PROFILES}, got {profile!r}")
    return list(REVIEW_RUN_DIRS if profile == "review" else RUN_DIRS)

# Left out of the archive even though they sit under a packaged root, as (pattern, reason).
# Reproduction needs none of them. Patterns are fnmatch patterns on the repository path.
SUPPLEMENT_EXCLUDE = [
    # Outreach and film-production specs: audience, voice and music briefs for the public films.
    # The second pattern catches the film VO and music spec without naming the film: this file
    # ships scrubbed, and the leak scan reads the film titles. Inside an archive
    # (README_SUPPLEMENT.md present) the scalar audit skips its one check on that spec.
    ("docs/specs/*outreach-video*", "outreach and film-production specs"),
    ("docs/specs/*vo-music*", "outreach and film-production specs"),
    # Figures a run script drew inside a run bundle. Text drawn into an image is pixels that no
    # scrub can see, and no one has viewed these for names (REVIEWED_IMAGES); the numbers they
    # plot are in the JSON files beside them.
    ("artifacts/goal_stakes/*", "run-script figures not viewed for names"),
    ("fullruns/*", "run-script figures not viewed for names"),
    # The GitHub social-preview card and its source page print the repository address.
    ("docs/social-card/*", "the repository's social-preview card, which names the repository"),
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


def _sources(runs_root: str | None = None, dropped: list | None = None,
             run_dirs: list[str] | None = None):
    """(archive path, source path) for every file the archive carries, sorted, and the status
    of each run bundle (`run_dirs`, default RUN_DIRS) as (entry, file count), the count None
    when the entry is absent.
    The repository part follows git, not the filesystem, so gitignored scratch output under
    the included directories (artifacts/clip_audit/, about 400 MB) never ships. A file that
    SUPPLEMENT_EXCLUDE names is left out and, when `dropped` is a list, recorded there as
    (path, reason)."""
    runs_root = runs_root or os.path.join(ROOT, "fullruns")
    roots = INCLUDE_DIRS + DOC_DIRS
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
    for r in RUN_DIRS if run_dirs is None else run_dirs:
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
PREREG_HISTORY_DIR = "PREREG_HISTORY"


def _readme(status, anon_name: str | None, *, dropped=(), films: int = 0,
            provenance: str | None = None, profile: str = "full", left_out=(),
            commits_hidden: int = 0, digests_hidden: int = 0, branches_hidden: int = 0,
            title_checked: bool = True, histories: int = 0) -> str:
    """README_SUPPLEMENT.md: what the archive holds and leaves out, how to install and run it,
    and the state of every run bundle. `provenance` is None when PROVENANCE.md was written,
    else the reason it was not; `histories` counts the PREREG_HISTORY patch files written.
    `left_out` lists the full-build bundles a review build does not carry; `commits_hidden`, `digests_hidden` and `branches_hidden` count the commit IDs,
    SHA-256 digests and branch names replaced by pseudonyms; `title_checked` is False when the manuscript's title could not be
    read, so the identity scan did not look for it."""
    L = ["# Anonymized supplement", "",
         "This archive holds the code, the tests, every committed artifact and the method and "
         "correction documents of the project, anonymized for review. The project's public "
         "site, its films and illustrated walkthroughs, the Colab links and the git history "
         "are not part of it, so links to them in the documents read as plain text. Start with "
         "section 1 of docs/REPRODUCE.md and skip its clone step: the folder this file sits in "
         "is the repository root.", "",
         "In the Markdown documents, a link to the repository or to a file the archive does not "
         "carry becomes plain text. index.html, index.template.html and viz/player/index.html "
         "ship only as inputs to the scalar audit and the site tests. They do not render here, "
         "and their links to the repository and to files outside the archive do not resolve.",
         "",
         "## Install and run", "",
         "- Python 3.10 to 3.13. Create a virtual environment first (`python3 -m venv .venv && "
         ". .venv/bin/activate`; on Windows `py -m venv .venv`, then `.venv\\Scripts\\activate`), "
         "since a current Debian or Ubuntu system Python refuses a bare `pip install`, then "
         "install the dependencies with `pip install -r requirements-dev.txt`. Every `python` "
         "below is that environment's interpreter.",
         "- `python -m pytest -q` runs the test suite, about 20 minutes on a laptop CPU.",
         "- `python scripts/audit_stats_recheck.py` runs the scalar audit, a few seconds.",
         "- `python scripts/reproduce.py tables --check` compares the shipped tables and gates "
         "with a rebuild from the committed per-seed artifacts, without training, and runs the "
         "audit; it exits 3 when a shipped page is stale. Without `--check` it rewrites a stale "
         "page and still exits 3.",
         "- `python scripts/reproduce.py retrain C1` prints the commands that retrain a run "
         "(hours of CPU; costs in docs/REPRODUCE.md section 2). Every output goes under the "
         "retrain folder, dist/retrain_<date> by default (`--dest`), the follow-up readouts "
         "under its readouts/ folder; a recipe that runs a bash script needs Git Bash on "
         "Windows.",
         "- Bit-exact equality holds only on the stack and machine that wrote a result. To replay a "
         "saved-agent readout on another machine, use the pinned CPU stack "
         "(constraints-cpu-corrected.txt, Python 3.11) as in docs/REPRODUCE.md, \"Supported-stack "
         "replay route\", which lists the measured differences (a few thousandths of an AUROC).", "",
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
    if commits_hidden:
        L.append(f"- Commit IDs ({commits_hidden} distinct commits) are replaced by keyed "
                 "pseudonyms of the same length, in the documents, the artifacts, "
                 f"`{PROVENANCE_NAME}` and the file names. One commit reads the same everywhere "
                 "in the archive, and a short hash is the prefix of its full form, so references "
                 "between files still match. They resolve in no public history.")
    if digests_hidden:
        L.append(f"- SHA-256 digests ({digests_hidden} distinct) in the documents and artifacts "
                 "are replaced by keyed pseudonyms of the same length. `SHA256SUMS` lists the "
                 "real digest of every file. A check that hashes a shipped file and compares the "
                 "result with a recorded digest looks the result up in "
                 f"`{DIGEST_TABLE_NAME}`, which is keyed by the SHA-256 of each real digest and "
                 "so spells none of them.")
    if branches_hidden:
        L.append(f"- Branch names ({branches_hidden} distinct) read `anon-branch-1`, "
                 "`anon-branch-2` and so on, in the documents, the artifacts and the file names. "
                 "A pull request keeps the name of the branch it came from, so a branch name is "
                 "a search key.")
    if not title_checked:
        L.append("- The manuscript's title could not be read where this archive was built, so "
                 "the identity scan did not look for it.")
    L.append("- Figures that run scripts draw inside run bundles (under artifacts/goal_stakes/ "
             "and fullruns/) are left out whenever present: no one has viewed them for names, "
             "and the numbers they plot are in the JSON files beside them.")
    L += ["", "## Run bundles", "",
          "Run bundles are the outputs of training runs (cells, saved agents, state dumps). They "
          "are not kept in version control, so the archive carries those present where it was "
          "built.", ""]
    if profile == "review":
        L += ["This is the review build, sized for a 100 MB limit on supplementary material. "
              "It carries only the saved agents of the historical hidden-8 held-out run, from "
              "which the direction-of-rearing and world-sample readouts replay, those of the "
              "hidden-8 seed-2 run, which back no committed readout, and the saved agents and "
              "cells of the C1 and C2 reruns, from which their state dumps rebuild. It carries "
              "no state dumps.", ""]
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
    shipped = [RUN_INCLUDED_NOTES[r] for r, n in status if n is not None and r in RUN_INCLUDED_NOTES]
    if shipped:
        L.append("")
        L += [f"- {note}" for note in shipped]
    L.append("")
    if left_out:
        L += ["The review build leaves out these run bundles, or the parts of them named, which "
              "the full build carries:", ""]
        L += [f"- `{r}`: {REVIEW_LEFT_OUT[r]}" if r in REVIEW_LEFT_OUT else f"- `{r}`"
              for r in left_out]
        L += ["", "Their per-seed readouts are committed under artifacts/ (the cells of C1, C2 "
              "and L1-CORRECTED under artifacts/corrected_runs/, the B1 and B2 readouts under "
              "artifacts/balanced_confirmation/), so every table and gate rebuilds from this "
              "archive without them. Replaying a readout from the saved agents or state dumps "
              "of B1, B2 and L1-CORRECTED, including the matched-gate replay of B1 and B2 and "
              "the stream and surprise readouts of L1-CORRECTED, needs a retrain from the "
              "recorded recipe (docs/REPRODUCE.md section 2).", ""]
    if gone:
        both = " and ".join(gone)
        L += [f"The saved agents and state dumps of the corrected run{'s' if len(gone) > 1 else ''} "
              f"{both} are not in this archive. {both} {'were' if len(gone) > 1 else 'was'} "
              "trained in a 4-vCPU cloud container, and the bundles were not available where this "
              "archive was built. Their per-seed cell records are under "
              "artifacts/corrected_runs/, so every corrected table rebuilds without training. "
              "`python scripts/reproduce.py retrain C1 --execute` retrains C1 and `retrain C2 "
              "--execute` retrains C2 (about 3 h 20 min and 4 h 25 min on 4 CPU workers, "
              "docs/REPRODUCE.md section 2), under the retrain folder dist/retrain_<date> "
              "(`--dest`); each prints the readout-only follow-ups that need the saved agents, "
              "which write under dist/retrain_<date>/readouts/, and a compare step that sets the "
              "retrained cells beside the committed ones.", ""]
    L.append("A run whose bundle is not listed as included has no saved agents or state dumps "
             "here; its committed per-seed records are under artifacts/.")
    L += ["", "## Files", ""]
    if provenance is None:
        L.append(f"- `{PROVENANCE_NAME}` lists every commit to each preregistration and spec "
                 "in the archive, with its UTC date and time and "
                 + ("pseudonymized " if commits_hidden else "")
                 + "short hash and a count per file, read from git when the archive was built, "
                 "so the freeze timing of each and the date of every later edit can be checked.")
    else:
        L.append(f"- There is no `{PROVENANCE_NAME}`: {provenance}.")
    if histories:
        L.append(f"- `{PREREG_HISTORY_DIR}/` holds one patch file per preregistration and spec "
                 f"({histories} files): every committed change to it, oldest first, as `git log "
                 "-p --follow` printed it when the archive was built. Each commit's header gives "
                 f"its {'pseudonymized ' if commits_hidden else ''}short hash, as in "
                 f"`{PROVENANCE_NAME}`, and its committer time in UTC. Blob IDs, path prefixes, "
                 "names, e-mail addresses and commit messages are left out, and the text is "
                 "anonymized like every other file, so the registered text at each commit can "
                 "be read.")
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


# Commit IDs and SHA-256 digests. The archive carries no git history, but a commit ID is a
# search key: on 2026-10-16 one full hash in the review archive found the public repository in
# a single code search, and short hashes narrow to it in two or three. A 64-character SHA-256
# digest is a better one: the agent, fold and surrogate digests the artifacts record are on the
# public repository verbatim (second review, 2026-10-16).
#
# Every hex token of 7 to 40 characters, in either case, that is a prefix of a commit in this
# repository's object store is replaced by a keyed pseudonym of the same length and case, in
# text and in file names. The pseudonym belongs to the full commit ID, so a short hash stays
# the prefix of its full form and every mention of one commit reads the same; comparisons
# between shipped files (an artifact's commit against the audit's copy, a startswith on a
# short hash, a sorted list) still hold. A token in a commit field (a JSON key naming a commit
# or a head, or the word "commit" before it) is replaced even when it resolves to no local
# commit, and a history-checked build refuses to start when one does not (history_gap).
#
# Every 64-character hex token in a shipped text file is replaced the same way, by a keyed
# pseudonym of the same length; a token of 12 or more characters that is a prefix of one takes
# the prefix of its pseudonym. SHA256SUMS keeps the real digest of every member, so the
# archive's bytes can still be verified. A check that compares a digest it computes from a
# shipped file with the one an artifact records goes through archive_digest(), which inside an
# extracted archive looks the computed digest up in DIGEST_PSEUDONYMS.json. That table is keyed
# by the SHA-256 of each real digest, so it gives back no real digest, and no shipped text holds
# one. The real digests of shipped binary members (the saved agents, the surrogate exports) are
# still in SHA256SUMS: anyone who hashes a shipped file can search for the result.
#
# Digit runs inside numbers are left alone: a token of digits alone, or of digits and one
# exponent "e", right after "digit." (the decimals of -9.765625e-05), right before ".digit", or
# ending in an exponent "e" before its sign and digits is no ID. A token with any other hex
# letter is an ID candidate in every context, and the leak scan (commit_scan) uses the same
# pattern, so it sees exactly what the replacement sees. A build fails (number_guard) when the
# pseudonyms change a decimal number, leave a JSON member that no longer parses or holds a
# different number, or rewrite an all-digit token of 7 or more digits that nothing marks as a
# commit: a commit field or the word "commit" in some shipped file, the word "commit" (also as
# the end of a word, as in git_commit) right before it, or a Markdown code span around it.
#
# The key stays local, under dist/ (gitignored). The key alone does not give back a build's
# commit pseudonyms: they are handed out in the sorted order of every commit in the object
# store, so a later commit, or a gc that drops an unreachable one, shifts many of them. Each
# build therefore writes its own map beside the archive (commit_map_path), holding the commit
# list it drew from and the pseudonym of every spelling it replaced; keep it with the archive
# that was submitted.
#
# The scan only knows the commits in the local object store. Outside a checkout, in a shallow,
# partial or single-branch clone, or when the remote holds a branch tip the store lacks, commit
# IDs the archive carries would pass it unseen, so an anonymized build that needs the scan
# refuses to start there (history_gap). It refuses too when it cannot list a remote's branches
# (offline, or a remote that no longer exists): the build needs the remote, or the remote
# removed.
_HEX_TOKEN = re.compile(r"(?<![0-9A-Za-z])"
                        r"(?!(?<=\d\.)\d+(?:[eE]\d*)?(?![0-9A-Za-z]))"  # decimals after "digit."
                        r"(?!\d+\.\d)"                                  # digits before ".digit"
                        r"(?!\d+[eE][+-]\d)"                            # mantissa before "e-05"
                        r"[0-9A-Fa-f]{7,64}(?![0-9A-Za-z])")
_DIGEST_TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{64}(?![0-9A-Za-z])")
DIGEST_MIN_PREFIX = 12
COMMIT_KEY = os.path.join(ROOT, "dist", "supplement-commit-key.txt")
COMMIT_MAP_SUFFIX = ".commit-map.json"
DIGEST_TABLE_NAME = "DIGEST_PSEUDONYMS.json"
_DIGEST_LOOKUP_TAG = "digest-pseudonym:"
# Where a hex token names a commit: a JSON key with "commit" or "head" in it, or ending in
# "_fix" (the results manifest's padding_fix), holding a string or a list of strings; or the
# word "commit" followed by the token. A key that names a fingerprint, hash, sha256 or digest
# (fingerprint_committed) holds no commit.
_NOT_COMMIT_KEY = r'(?![A-Za-z_]*(?:fingerprint|hash|sha\d|digest))'
_COMMIT_FIELD = re.compile(r'(?i)"' + _NOT_COMMIT_KEY + r'[A-Za-z_]*(?:commit|head)[A-Za-z_]*"'
                           r'\s*:\s*(\[[^\]]*\]|"[^"]*")'
                           r'|"' + _NOT_COMMIT_KEY + r'[A-Za-z_]*_fix"\s*:\s*(\[[^\]]*\]|"[^"]*")')
# "paper commit X" names a commit of the manuscript's own repository, which this object store
# does not hold, so it is not read as one of this repository's commits; paper_commit_tokens
# collects those IDs so they are still given pseudonyms.
_COMMIT_PROSE = re.compile(r"(?i)(?<!paper )\bcommits?\b[\s`(:]*([0-9a-f]{7,40})(?![0-9A-Za-z])")
_PAPER_COMMIT_PROSE = re.compile(r"(?i)\bpaper commits?\b[\s`(:]*([0-9a-f]{7,40})(?![0-9A-Za-z])")


def commit_map_path(archive: str) -> str:
    """The pseudonym map of the archive at `archive`: dist/supplement-review.zip ->
    dist/supplement-review.commit-map.json, so no other build or test overwrites it."""
    return os.path.splitext(archive)[0] + COMMIT_MAP_SUFFIX


def commit_ids() -> list[str]:
    """Every commit object in this repository's object store, sorted; [] outside a checkout
    (an extracted archive, or a folder inside some other repository)."""
    top = (_git(["rev-parse", "--show-toplevel"]) or "").strip()
    if not top or os.path.normcase(os.path.realpath(top)) != os.path.normcase(os.path.realpath(ROOT)):
        return []
    out = _git(["cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)"])
    return sorted(ln.split()[0] for ln in (out or "").splitlines() if ln.endswith(" commit"))


def commit_field_tokens(text: str) -> set[str]:
    """The hex tokens (lowercase) that `text` names as commits (see _COMMIT_FIELD)."""
    out = set()
    for m in _COMMIT_FIELD.finditer(text):
        out |= {t.lower() for t in re.findall(r'"([0-9A-Fa-f]{7,40})"', m.group(1) or m.group(2))}
    out |= {m.group(1).lower() for m in _COMMIT_PROSE.finditer(text)}
    return out


def paper_commit_tokens(text: str) -> set[str]:
    """The hex tokens (lowercase) that `text` names as commits of the manuscript's repository."""
    return {m.group(1).lower() for m in _PAPER_COMMIT_PROSE.finditer(text)}


def _ls_remote(remote: str) -> str | None:
    """`git ls-remote --heads <remote>`, or None when the remote cannot be reached in 60 s."""
    try:
        res = subprocess.run(["git", "ls-remote", "--heads", remote], cwd=ROOT,
                             capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return res.stdout


def history_gap(commits: list[str], named=()) -> str | None:
    """Why the commit scan cannot see every commit ID the archive may carry, or None. `named`
    holds the tokens shipped files name as commits (commit_field_tokens); one that resolves to
    no local commit shows the object store is incomplete. A remote whose branches cannot be
    listed counts as a gap, since a branch tip the store lacks would then go unseen. An
    extracted archive (README_SUPPLEMENT.md at the root) has no history and no real commit IDs
    left, so rebuilding from one is allowed."""
    if os.path.exists(os.path.join(ROOT, README_NAME)):
        return None
    if not commits:
        return "this folder is not the top of a git checkout, so no commit ID is known"
    if (_git(["rev-parse", "--is-shallow-repository"]) or "").strip() == "true":
        return ("this is a shallow clone, which holds only the commits it fetched "
                "(run `git fetch --unshallow`)")
    if (_git(["config", "--get", "extensions.partialClone"]) or "").strip():
        return "this is a partial clone, which fetches objects only when they are read"
    have = set(commits)
    for remote in (_git(["remote"]) or "").split():
        specs = (_git(["config", "--get-all", f"remote.{remote}.fetch"]) or "").split()
        if not any(s.lstrip("+").split(":")[0] in ("refs/heads/*", "refs/*") for s in specs):
            return (f"remote {remote} fetches only some of its branches (a single-branch clone; "
                    f"set remote.{remote}.fetch to +refs/heads/*:refs/remotes/{remote}/* and "
                    "fetch)")
        listing = _ls_remote(remote)
        if listing is None:
            return (f"the branches of remote {remote} could not be listed (`git ls-remote` failed "
                    "or timed out), so nothing shows the object store holds every branch tip; "
                    f"build online, or remove the remote with `git remote remove {remote}` if it "
                    "is gone")
        missing = [ln.split()[1] for ln in listing.splitlines()
                   if len(ln.split()) == 2 and ln.split()[0] not in have]
        if missing:
            return (f"{len(missing)} branch tip(s) of remote {remote} are not in the object store "
                    f"(run `git fetch {remote}`), for example {missing[0]}")
    absent = sorted(t for t in named if not _resolves(commits, t))
    if absent:
        return (f"{len(absent)} commit ID(s) that shipped files name in commit fields resolve to "
                f"no local commit, for example {absent[0]}, so the object store lacks commits "
                "the archive may also name elsewhere")
    return None


def commit_key(path: str | None = None) -> bytes:
    """The pseudonym key, made on first use and kept so later builds agree."""
    path = path or COMMIT_KEY
    try:
        with open(path, encoding="ascii") as fh:
            return bytes.fromhex(fh.read().strip())
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        key = secrets.token_hex(32)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(key + "\n")
        return bytes.fromhex(key)


def _resolves(sorted_ids: list[str], tok: str) -> list[str]:
    """The entries of `sorted_ids` that `tok` (any case) is a prefix of."""
    tok = tok.lower()
    i = bisect.bisect_left(sorted_ids, tok)
    out = []
    while i < len(sorted_ids) and sorted_ids[i].startswith(tok):
        out.append(sorted_ids[i])
        i += 1
    return out


def _same_case(pseudonym: str, tok: str) -> str:
    return pseudonym.upper() if tok.isupper() else pseudonym


def digest_lookup_key(real: str) -> str:
    """The DIGEST_PSEUDONYMS.json key of a real digest: the SHA-256 of the tagged digest, which
    names it without spelling it."""
    return hashlib.sha256((_DIGEST_LOOKUP_TAG + real.lower()).encode("ascii")).hexdigest()


def digest_tokens(text: str) -> set[str]:
    """The 64-character hex tokens of `text` (lowercase) that look like digests: a run of one or
    two repeated characters is padding, not a search key."""
    return {m.group(0).lower() for m in _DIGEST_TOKEN.finditer(text) if len(set(m.group(0))) > 2}


class CommitPseudonyms:
    """Same-length keyed pseudonyms for the commit IDs and SHA-256 digests in a text (see
    _HEX_TOKEN). `sub` replaces them; `search` finds a commit ID that still resolves, with the
    interface of a compiled pattern, so the leak scan can use it. `mapping` records pseudonym ->
    commit ID as written, `digest_mapping` and `forced_mapping` the same for digests and for
    commit-field tokens that resolve to no local commit (`forced`)."""

    def __init__(self, commits: list[str], key: bytes, digests=(), forced=()):
        self.commits = sorted(commits)
        self.mapping: dict[str, str] = {}
        self.digest_mapping: dict[str, str] = {}
        self.forced_mapping: dict[str, str] = {}
        # One keyed 40-character draw per commit, each distinct at 7 characters and none the
        # prefix of a real commit, then handed out in sorted order. The pseudonyms therefore
        # sort as the commits do: the results manifest lists a run's commits sorted, and an
        # order that changed in the archive would make its own --check report it stale.
        draws, seen = [], set()
        for c in self.commits:
            n = 0
            while True:
                p = hmac.new(key, f"{c}:{n}".encode(), hashlib.sha256).hexdigest()[:40]
                if p[:7] not in seen and not self.resolves(p[:7]):
                    break
                n += 1
            seen.add(p[:7])
            draws.append(p)
        self.full = dict(zip(self.commits, sorted(draws)))     # commit ID -> pseudonym
        # Forced IDs get the same two properties as resolvable ones. A shorter spelling of an ID
        # (a 7-character commit_at_run beside a 40-character git_commit_at_run) takes the longer
        # spelling's pseudonym prefix, and the pseudonyms sort as the IDs do, because the results
        # manifest lists a run's commits sorted and its --check inside the archive would otherwise
        # report it stale (a shallow clone forces every ID, CI 2026-10-16).
        toks = sorted({t.lower() for t in forced if not self.resolves(t)})
        maximal = [t for t in toks if not any(o != t and o.startswith(t) for o in toks)]
        fdraws = sorted(self._draw(key, "commit", t, 40, seen) for t in maximal)
        ffull = dict(zip(maximal, fdraws))
        self.forced = {t: ffull[next(L for L in maximal if L.startswith(t))][:len(t)] for t in toks}
        # A forced ID is matched by prefix, as a resolvable one is: a field may spell the full ID
        # while code or prose quotes its first seven characters, and both must get one pseudonym
        # (a shallow clone resolves neither, 2026-10-16).
        self.forced_ids = sorted(self.forced)
        self.digests = sorted({d.lower() for d in digests})
        self.dfull = {d: self._draw(key, "sha256", d, 64, seen) for d in self.digests}

    def _draw(self, key: bytes, kind: str, tok: str, n: int, seen: set) -> str:
        """A keyed pseudonym of length `n` for `tok`, distinct at 7 characters from every other
        draw and from every real commit, and no real digest."""
        i = 0
        while True:
            p = hmac.new(key, f"{kind}:{tok}:{i}".encode(), hashlib.sha256).hexdigest()[:n]
            if p[:7] not in seen and not self.resolves(p[:7]) and p != tok:
                seen.add(p[:7])
                return p
            i += 1

    def resolves(self, tok: str) -> list[str]:
        """The commits `tok` is a prefix of. An ambiguous prefix counts: it still narrows a search."""
        return _resolves(self.commits, tok) if len(tok) <= 40 else []

    def sub(self, text: str) -> str:
        if not (self.commits or self.forced or self.digests):
            return text

        def rep(m):
            tok = m.group(0)
            low = tok.lower()
            hits = self.resolves(low)
            if hits:
                # An ambiguous prefix takes its first commit's pseudonym, which keeps the order.
                p = _same_case(self.full[hits[0]][:len(tok)], tok)
                self.mapping[p] = tok
            elif self._forced_hit(low):
                p = _same_case(self.forced[self._forced_hit(low)][:len(tok)], tok)
                self.forced_mapping[p] = tok
            elif len(tok) >= DIGEST_MIN_PREFIX and _resolves(self.digests, low):
                p = _same_case(self.dfull[_resolves(self.digests, low)[0]][:len(tok)], tok)
                self.digest_mapping[p] = tok
            else:
                return tok
            return p

        return _HEX_TOKEN.sub(rep, text)

    def _forced_hit(self, low: str) -> str | None:
        """The forced ID that `low` spells, or abbreviates in 7 to 12 characters as git does.
        A longer prefix is no abbreviation: a 16-character fingerprint that happens to start
        like a forced ID is left alone, as fingerprints are everywhere else."""
        if low in self.forced:
            return low
        hits = _resolves(self.forced_ids, low) if 7 <= len(low) <= 12 else []
        return hits[0] if hits else None

    def search(self, text: str):
        for m in _HEX_TOKEN.finditer(text):
            if self.resolves(m.group(0)) or self._forced_hit(m.group(0).lower()):
                return m
        return None

    def count(self) -> int:
        """Distinct commits replaced (an ambiguous prefix counts as its first commit)."""
        return len({self.resolves(t)[0] for t in self.mapping.values()})

    def digest_count(self) -> int:
        """Distinct digests replaced."""
        return len({_resolves(self.digests, t)[0] for t in self.digest_mapping.values()})

    def digest_table(self) -> dict:
        """DIGEST_PSEUDONYMS.json: digest_lookup_key(real) -> pseudonym, for every digest
        replaced."""
        used = {_resolves(self.digests, t)[0] for t in self.digest_mapping.values()}
        return {digest_lookup_key(d): self.dfull[d] for d in sorted(used)}


class DigestSet:
    """search() finds a 64-character hex token of `tokens`, in either case, with the interface
    of a compiled pattern, so the byte scan (_binary_hits) can use it."""

    def __init__(self, tokens):
        self.tokens = {t.lower() for t in tokens}

    def search(self, text: str):
        for m in _DIGEST_TOKEN.finditer(text):
            if m.group(0).lower() in self.tokens:
                return m
        return None


def tracked_digests() -> set[str]:
    """Every 64-character hex token in the text files git tracks here, the copies a code search
    of the public repository can find; empty outside a checkout."""
    out = _git(["grep", "-h", "-o", "-I", "-E", "[0-9a-fA-F]{64}"]) or ""
    return {t.lower() for t in out.split() if len(t) == 64}


def commit_scan(path: str, hider: CommitPseudonyms) -> list[str]:
    """Members of the archive at `path` whose name, text or bytes (zip containers opened) still
    hold a commit ID of this repository, in either case."""
    if not (hider.commits or hider.forced):
        return []
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if hider.search(n) or _binary_hits(z.read(n), [hider]):
                out.append(n)
    return out


def digest_scan(path: str, digests) -> list[str]:
    """Members of the archive at `path`, SHA256SUMS aside, whose name, text or bytes (zip
    containers opened) hold one of `digests`, in either case."""
    found = DigestSet(digests)
    if not found.tokens:
        return []
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n != "SHA256SUMS" and (found.search(n) or _binary_hits(z.read(n), [found])):
                out.append(n)
    return out


_DECIMAL = re.compile(r"\d+\.\d+(?:[eE][-+]?\d+)?")


def _json_docs(name: str, text: str) -> list[str]:
    if name.endswith((".json", ".ipynb")):
        return [text]
    if name.endswith(".jsonl"):
        return [ln for ln in text.splitlines() if ln.strip()]
    return []


def _leaves(obj):
    """The non-string leaves of a parsed JSON value, in document order."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    elif not isinstance(obj, str):
        yield repr(obj)


_INTEGER = re.compile(r"(?<![0-9A-Za-z])\d{7,}(?![0-9A-Za-z])")
# The word for a commit before the token, or a branch head written after the branch name in a
# code span (the 2026-10-20 spec names ITASORL's `main` head that way, an all-digit commit ID).
_COMMIT_BEFORE = re.compile(r"(?i)(?:commits?[\s`(:]*|`main`\s+)$")


def _marked_commit(text: str, start: int, end: int, known) -> bool:
    """Whether the token text[start:end] reads as a commit ID: one that `known` holds (the
    tokens some shipped file names in a commit field), one right after the word "commit" or
    after the branch name `main`, or one alone in a Markdown code span."""
    tok = text[start:end]
    if tok.lower() in known or _COMMIT_BEFORE.search(text[max(0, start - 16):start]):
        return True
    return text[start - 1:start] == "`" and text[end:end + 1] == "`"


def number_guard(name: str, before: str, after: str, known=frozenset()) -> str | None:
    """What the pseudonyms broke in the text member `name`, or None: a decimal number that
    changed, a JSON document that no longer parses, a JSON number, boolean or null that
    differs, or an all-digit token of 7 or more digits that changed although nothing marks it
    as a commit (_marked_commit; `known` adds the commit-field tokens of the other shipped
    files to this member's own). A document that did not parse before is not checked."""
    if before == after:
        return None
    if _DECIMAL.findall(before) != _DECIMAL.findall(after):
        return "a decimal number changed"
    for b, a in zip(_json_docs(name, before), _json_docs(name, after)):
        try:
            ob = json.loads(b)
        except ValueError:
            continue
        try:
            oa = json.loads(a)
        except ValueError as e:
            return f"no longer parses as JSON ({e})"
        if list(_leaves(ob)) != list(_leaves(oa)):
            return "a JSON number, boolean or null changed"
    if len(before) != len(after):
        return "the text changed length, so a same-length pseudonym cannot account for it"
    known = set(known) | commit_field_tokens(before)
    for m in _INTEGER.finditer(before):
        if after[m.start():m.end()] != m.group(0) and not _marked_commit(before, m.start(),
                                                                          m.end(), known):
            return (f"an integer of {len(m.group(0))} digits at character {m.start()} changed, "
                    "and nothing marks it as a commit")
    return None


def _digest_table(root: str) -> dict:
    try:
        with open(os.path.join(root, DIGEST_TABLE_NAME), encoding="utf-8") as fh:
            return json.load(fh)["pseudonyms"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def archive_digest(real: str) -> str:
    """The digest a shipped artifact records for the real digest `real`: inside an extracted
    anonymized archive (README_SUPPLEMENT.md at the root), its pseudonym from
    DIGEST_PSEUDONYMS.json; anywhere else, and for a digest the table does not hold, `real`.
    A check that compares a digest it computes from a shipped file with a recorded one compares
    archive_digest(computed) instead, and passes in both places."""
    if not os.path.exists(os.path.join(ROOT, README_NAME)):
        return real
    p = _digest_table(ROOT).get(digest_lookup_key(real))
    return _same_case(p, real) if p else real


def archive_digest_text(text: str) -> str:
    """`text` with every 64-character hex token passed through archive_digest()."""
    return _DIGEST_TOKEN.sub(lambda m: archive_digest(m.group(0)), text)


def _log_records(args: list[str]) -> list[tuple[str, str, list[str]]]:
    """[(short hash, committer time, [paths])] from a `git log --name-only` whose format line
    is `%x00%h %ct`, newest first; [] when git gives nothing."""
    out = _git(["-c", "core.quotepath=off", "log", "--name-only", "--format=%x00%h %ct", *args])
    recs = []
    for chunk in (out or "").split("\0")[1:]:
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if lines:
            h, ts = lines[0].split()
            recs.append((h, ts, lines[1:]))
    return recs


def provenance_rows(rels: list[str]):
    """([(path, [(UTC time, short hash, note), ...] oldest first, dirty)], None), or
    (None, reason) when the history is not there to read: outside a git checkout, or in a
    shallow clone, whose oldest commit is the clone boundary and not when a file entered the
    history.

    Every commit that changed the file is listed. The non-merge commits come from
    `git log --follow`, which walks every merged branch (an edit made on a side branch keeps its
    own hash even when the main line later carries the same text) and follows renames. A merge
    commit is listed when its version of the file differs from every parent, i.e. when it joins
    edits from two branches or resolves a conflict; a merge that takes one side unchanged added
    nothing. `note` is "merge", or the file's path when the commit made it under an earlier
    one. `dirty` is True when the file in the working tree differs from the last commit.
    Committer dates in UTC, no names, e-mail addresses or commit messages."""
    top = (_git(["rev-parse", "--show-toplevel"]) or "").strip()
    if not top or os.path.normcase(os.path.realpath(top)) != os.path.normcase(os.path.realpath(ROOT)):
        return None, "the archive was not built from a git checkout"
    if (_git(["rev-parse", "--is-shallow-repository"]) or "").strip() == "true":
        return None, "the archive was built from a shallow clone, whose history is incomplete"

    def utc(ts: str) -> str:
        return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M")

    status = _git(["-c", "core.quotepath=off", "status", "--porcelain", "--", *rels]) or ""
    changed = {ln[3:].split(" -> ")[-1].strip() for ln in status.splitlines() if len(ln) > 3}
    rows = []
    for rel in rels:
        edits = _log_records(["--follow", "--", rel])
        paths = sorted({p for _, _, ps in edits for p in ps} | {rel})
        merges = [(h, ts, ["merge"]) for h, ts, ps in
                  _log_records(["--full-history", "--merges", "-c", "--", *paths])
                  if set(ps) & set(paths)]
        seen, commits = set(), []
        for h, ts, ps in sorted(edits + merges, key=lambda r: int(r[1])):
            if h in seen:
                continue
            seen.add(h)
            note = "merge" if ps == ["merge"] else next((p for p in ps if p != rel), "")
            commits.append((utc(ts), h, note))
        rows.append((rel, commits, rel in changed))
    return rows, None


def provenance_md(rows, rename, pseudonyms: bool = False) -> str:
    """PROVENANCE.md: a summary line per file, then every commit to it. `pseudonyms` says the
    hashes will be replaced by commit pseudonyms (CommitPseudonyms) before the file ships."""
    hashes = ("short commit hashes, replaced by keyed pseudonyms of the same length (see "
              f"{README_NAME})" if pseudonyms else "short commit hashes")
    out = ["# Provenance of the preregistrations and specs", "",
           "Read from the git history when this archive was built: every commit that changed "
           f"each file, oldest first, with committer dates and times in UTC and {hashes}. "
           "Names, e-mail addresses and commit messages are not included. The first "
           "commit is when the file entered the history, and each later one is an edit; the "
           "file's own dated entries say what changed. Commits on every branch that was merged "
           "are counted, so an edit made on a side branch is listed under its own hash. A merge "
           "commit is listed when its version of the file differs from every parent, which "
           "happens when it joins edits from two branches or resolves a conflict. A commit made "
           "while the file had an earlier path names that path.", "",
           "## Summary", "",
           "| File | Commits | First commit (UTC) | Last commit (UTC) |",
           "|---|---|---|---|"]
    for rel, commits, _ in rows:
        if commits:
            (f, fh, _), (la, lh, _) = commits[0], commits[-1]
            out.append(f"| `{rename(rel)}` | {len(commits)} | {f} ({fh}) | {la} ({lh}) |")
        else:
            out.append(f"| `{rename(rel)}` | 0 | not committed | not committed |")
    out += ["", "## Every commit"]
    for rel, commits, dirty in rows:
        out += ["", f"### `{rename(rel)}`", ""]
        if not commits:
            out.append("Not committed when the archive was built.")
            continue
        n = len(commits)
        line = (f"{n} commit{'s' if n != 1 else ''}, first {commits[0][0]} ({commits[0][1]}), "
                f"last {commits[-1][0]} ({commits[-1][1]}).")
        if dirty:
            line += (" The copy in this archive has changes that were not committed when it was "
                     "built.")
        out += [line, "", "| # | Commit (UTC) | Hash | Note |", "|---|---|---|---|"]
        for i, (when, h, note) in enumerate(commits, 1):
            note = "merge" if note == "merge" else (f"path `{rename(note)}`" if note else "")
            out.append(f"| {i} | {when} | {h} | {note} |")
    return "\n".join(out) + "\n"


def _git_bytes_text(args: list[str]) -> str | None:
    """`git <args>` decoded as UTF-8 whatever the locale (the documents are UTF-8, and a Windows
    code page cannot decode every byte), or None when git fails."""
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.decode("utf-8", errors="replace")


def prereg_history(rel: str) -> str | None:
    """The full text history of the frozen document `rel` as one patch file, oldest commit
    first, or None outside a git checkout or in a shallow clone (as provenance_rows). From
    `git log -p --follow --cc --format=%x00%h %ct`: each commit's header becomes
    "commit <short hash> <YYYY-MM-DD HH:MM> UTC", the committer time in UTC, so no time-zone
    offset ships; the diffs have no a/ and b/ path prefixes, and the `index` lines, which name
    blob IDs, are dropped. Names, e-mail addresses and commit messages are not read. The caller
    scrubs the text like any other shipped file before it ships."""
    top = (_git(["rev-parse", "--show-toplevel"]) or "").strip()
    if not top or os.path.normcase(os.path.realpath(top)) != os.path.normcase(os.path.realpath(ROOT)):
        return None
    if (_git(["rev-parse", "--is-shallow-repository"]) or "").strip() == "true":
        return None
    out = _git_bytes_text(["-c", "core.quotepath=off", "log", "-p", "--follow", "--cc",
                           "--no-color", "--no-ext-diff", "--no-textconv", "--no-prefix",
                           "--format=%x00%h %ct", "--", rel])
    if out is None:
        return None
    chunks = []
    for chunk in out.split("\0")[1:]:
        lines = chunk.split("\n")
        h, ts = lines[0].split()
        when = datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M")
        body = [ln for ln in lines[1:] if not ln.startswith("index ")]
        chunks.append("\n".join([f"commit {h} {when} UTC", *body]).rstrip("\n") + "\n")
    if not chunks:
        return None
    head = (f"# Every committed change to {rel}, oldest first (git log -p --follow; "
            "committer times in UTC).\n\n")
    return head + "\n".join(reversed(chunks))


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


def paper_source(path: str | None = None) -> str | None:
    """The manuscript's LaTeX source, untracked and so never in the archive: `path` (main.tex
    or the folder that holds it) when given, else docs/paper/main.tex in this checkout, else
    the same file in the main checkout of a git worktree (a worktree has no docs/paper of its
    own). None when there is none."""
    def main_tex(p: str) -> str:
        return os.path.join(p, "main.tex") if os.path.isdir(p) else p
    if path:
        cands = [main_tex(path)]
    else:
        cands = [os.path.join(ROOT, "docs", "paper", "main.tex")]
        common = (_git(["rev-parse", "--git-common-dir"]) or "").strip()
        if common:
            common = os.path.normpath(os.path.join(ROOT, common))
            if os.path.basename(common) == ".git":
                cands.append(os.path.join(os.path.dirname(common), "docs", "paper", "main.tex"))
    return next((c for c in cands if os.path.isfile(c)), None)


_HEX_WORD = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{7,40}(?![0-9A-Za-z])")


def manuscript_commit_ids(path: str | None = None) -> list[str]:
    """Every commit ID of the manuscript's own git repository (the folder of
    paper_source(path)), sorted; [] when there is none. The archive may name these IDs in any
    form ("paper commit X", "paper repo at X"), and they are given pseudonyms like this
    repository's own."""
    src = paper_source(path)
    if not src:
        return []
    folder = os.path.dirname(os.path.abspath(src))
    try:
        top = subprocess.run(["git", "-C", folder, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True, timeout=60).stdout.strip()
        if os.path.normcase(os.path.realpath(top)) == os.path.normcase(os.path.realpath(ROOT)):
            return []        # no repository of its own: git walked up into this checkout
        res = subprocess.run(["git", "-C", folder, "rev-list", "--all"],
                             capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return sorted(ln.strip().lower() for ln in res.stdout.splitlines() if ln.strip())


def manuscript_tokens(text: str, manuscript: list[str], commits: list[str]) -> set[str]:
    """The hex words of `text` (lowercase) that abbreviate a manuscript commit and no commit of
    this repository."""
    return {t for t in (m.group(0).lower() for m in _HEX_WORD.finditer(text))
            if _resolves(manuscript, t) and not _resolves(commits, t)}


def paper_titles(path: str | None = None) -> list[str]:
    """The manuscript's main title and subtitle (the parts before and after the first colon,
    each of 8 or more characters), read at build time from paper_source(path), so no tracked
    file spells them; [] when there is no source or no title in it."""
    src = paper_source(path)
    if not src:
        return []
    try:
        with open(src, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    text = re.sub(r"(?<!\\)%.*", "", text)                 # LaTeX comments
    m = re.search(r"pdftitle=\{", text) or re.search(r"\\title\{", text)
    title = _braced(text, m.end()) if m else None
    if not title:
        return []
    while (t := title.find("\\thanks{")) >= 0:
        inner = _braced(title, t + len("\\thanks{"))
        title = title[:t] + " " + (title[t + len("\\thanks{") + len(inner) + 1:]
                                   if inner is not None else "")
    title = re.sub(r"\\\\|\\[A-Za-z]+\*?|[{}~]", " ", title)
    parts = [" ".join(p.split()) for p in title.split(":", 1)]
    return [p for p in parts if len(p) >= 8]


def _braced(text: str, i: int) -> str | None:
    """The text from `i`, just after an opening brace, to its matching closing brace."""
    depth, j = 1, i
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
        j += 1
    return None


def paper_title(path: str | None = None) -> str | None:
    """The manuscript's main title (before any colon), or None (paper_titles)."""
    return next(iter(paper_titles(path)), None)


def _leak_patterns(old_name: str | None, titles=None, branches=None) -> list:
    """Leak-only checks beyond the scrub rules: any GitHub Pages host or hosted-image URL that
    survived, and, when the package was renamed, any spelling of the original name; the build
    machine's account name; the manuscript's title and subtitle (`titles`, default
    paper_titles()) when its local source was found; and every branch name (`branches`, default
    branch_names()) and agent-session branch name in any case."""
    pats = [re.compile(r"[\w-]+\." + _GH + r"\.io\b", re.I), re.compile(_HOSTED, re.I),
            _branch_pattern(branch_names() if branches is None else branches, re.I)]
    if old_name:
        pats.append(re.compile(re.escape(old_name), re.I))
    acct = local_account()
    if acct:
        pats.append(re.compile(r"(?<![A-Za-z0-9])" + re.escape(acct) + r"(?![A-Za-z0-9])", re.I))
    for title in paper_titles() if titles is None else titles:
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


def leak_scan(path: str, rules, old_name: str | None, reviewed=None, titles=None,
              branches=None) -> list[str]:
    """Members of the archive at `path` that still carry an identity string, or may. Text
    members are scanned with every scrub rule; binary members (agent bundles, arrays, images)
    with every rule except the e-mail shape, which arbitrary bytes match by chance about once
    per few tens of MB. Member names are scanned too. An image member is a leak unless its
    bytes are on the reviewed list (REVIEWED_IMAGES, or `reviewed`), because text drawn into
    an image is invisible to every pattern. `titles` and `branches` as in _leak_patterns."""
    leak = _leak_patterns(old_name, titles, branches)
    text_pats = [p for p, _ in rules] + leak
    bin_pats = [p for p, r in rules if (p, r) != _EMAIL_RULE] + leak
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
               anon_name: str | None = DEFAULT_ANON_NAME, *, require_runs: bool = False,
               profile: str = "full", key_path: str | None = None,
               require_history: bool | None = None, paper: str | None = None) -> int:
    """Build the archive at `out`. `require_history` (default: on for the review profile; the
    command line always turns it on) refuses an anonymized build, writing nothing, when
    history_gap() finds the commit scan blind. `paper` is the manuscript source the title check
    reads (paper_source); an anonymized review build refuses to start when no title can be
    read, except in an extracted archive, which no longer holds anything to check it against.
    The build fails and removes the archive, writing no map, when any scan finds a leak, a
    commit ID that still resolves, a real digest, or a number the pseudonyms changed. A build
    that gets as far as writing the archive first removes the map of any earlier archive at
    `out`, so a failed rebuild leaves neither; a build that refuses to start leaves an earlier
    archive and its map as they were."""
    run_dirs = profile_run_dirs(profile)
    old_name = project_name() if anon_name else None
    if anon_name:
        check_anon_name(anon_name, old_name)
    if old_name and old_name.lower() == anon_name.lower():
        old_name = None          # already neutral (a rebuild from an extracted archive)
    rename_pkg = renamer(old_name, anon_name)
    films = film_rules()
    in_archive = os.path.exists(os.path.join(ROOT, README_NAME))
    # Every branch the remotes list too, not only those fetched here (an extracted archive has
    # no git and needs none: its names are pseudonyms already).
    branches = branch_names(remote=not in_archive)
    rules = scrub_rules(branches)
    branch_rule = rules[-1:]            # scrub_rules puts the branch rule last
    titles = paper_titles(paper)
    if anon_name and not titles:
        where = paper or "docs/paper/main.tex here or in the main checkout"
        if profile == "review" and not in_archive:
            print(f"no archive written: the manuscript title could not be read from {where}, so "
                  "the identity scan could not look for it. Pass --paper PATH.")
            return 1
        print(f"  WARNING: the manuscript title could not be read from {where}; the identity "
              "scan will not look for it", flush=True)
    commits = commit_ids()
    if require_history is None:
        require_history = profile == "review"

    dropped: list = []
    pairs, status = _sources(runs_root, dropped, run_dirs)
    # The commit IDs that commit fields name, and the digests to replace, from every text member.
    named: set = set()
    paper_ids: set = set()
    manuscript: list = []
    digests: set = set()
    if anon_name and not in_archive:
        manuscript = manuscript_commit_ids(paper)
        if not manuscript:
            print("  WARNING: the manuscript repository's commit IDs could not be listed; only "
                  "'paper commit X' spellings of them will get pseudonyms", flush=True)
        for rel, src in pairs:
            if is_text(rel):
                with open(src, "rb") as fh:
                    text = fh.read().decode("utf-8", errors="replace")
                named |= commit_field_tokens(text)
                paper_ids |= paper_commit_tokens(text) | manuscript_tokens(text, manuscript, commits)
                digests |= digest_tokens(text)
    gap = history_gap(commits, named) if anon_name and require_history else None
    if gap:
        print(f"no archive written: the commit scan cannot see every commit ID, because {gap}. "
              "Build from a full checkout.")
        return 1
    # The provenance table and the full text history of every frozen document, read before the
    # pseudonyms are drawn, so a digest that only an earlier version names is replaced too.
    # Neither is read in a shallow clone or outside a checkout (provenance_rows says why).
    rows, no_prov = provenance_rows([rel for rel, _ in pairs if _is_frozen_doc(rel)])
    histories = {}
    for rel, _, _ in rows or ():
        patch = prereg_history(rel)
        if patch:
            histories[rel] = patch
            if anon_name and not in_archive:
                digests |= digest_tokens(patch)
                paper_ids |= paper_commit_tokens(patch) | manuscript_tokens(patch, manuscript, commits)
    key_path = key_path or COMMIT_KEY
    # Manuscript-repository IDs get pseudonyms too; the history check above does not read them.
    forced = sorted(t for t in named | paper_ids if not _resolves(commits, t))
    hide = bool(commits or forced or digests)
    hider = CommitPseudonyms(commits, commit_key(key_path) if hide else b"", digests=digests,
                             forced=forced)

    def rename(path_or_text: str) -> str:
        """Package name, film titles, branch names and commit IDs, as in text, so a path a
        document quotes still resolves in the archive."""
        return hider.sub(rename_pkg(scrub(path_or_text, films + branch_rule)))

    left_out = [r for r in RUN_DIRS if r not in run_dirs] if profile == "review" else []
    print(f"profile {profile}; run bundles under {runs_root or os.path.join(ROOT, 'fullruns')}:")
    for r, n in status:
        if n is None:
            print(f"  WARNING: {r} is absent; the archive will NOT carry it", flush=True)
        else:
            print(f"  included {r} ({n} files)")
    for r in left_out:
        print(f"  not in the review profile: {r}")
    missing = [r for r, n in status if n is None]
    if require_runs and missing:
        print(f"--require-runs: {len(missing)} run bundle(s) absent; no archive written")
        return 1
    for rel, why in dropped:
        print(f"  left out {rel} ({why})")
    path = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shipped = {rename(rel) for rel, _ in pairs}
    sums = []
    broken = []
    # The map of an earlier build at this path names an archive this build overwrites, so it
    # goes first: a build that fails below then leaves neither archive nor map.
    if os.path.exists(commit_map_path(path)):
        os.remove(commit_map_path(path))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        def put(arc: str, data: bytes) -> None:
            z.writestr(arc, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")

        for rel, src in pairs:
            with open(src, "rb") as fh:
                data = fh.read()
            arc = rename(rel)
            if is_text(rel):
                before = rename_pkg(scrub(data.decode("utf-8", errors="replace"), rules))
                text = hider.sub(before)
                why = number_guard(arc, before, text, known=named)
                if why:
                    broken.append(f"{arc}: {why}")
                if arc.endswith(".md"):
                    text = delink(text, arc, shipped)
                data = text.encode("utf-8")
            put(arc, data)
        if rows is not None:
            put(PROVENANCE_NAME, hider.sub(provenance_md(rows, rename, bool(hider.commits)))
                .encode("utf-8"))
        # Each history goes through the same chain as a shipped text file: scrub and package
        # rename, then the pseudonyms, then the number check.
        for rel, patch in histories.items():
            arc = f"{PREREG_HISTORY_DIR}/{rename(rel)}.patch"
            before = rename_pkg(scrub(patch, rules))
            text = hider.sub(before)
            why = number_guard(arc, before, text, known=named)
            if why:
                broken.append(f"{arc}: {why}")
            put(arc, text.encode("utf-8"))
        table = hider.digest_table()
        if table:
            put(DIGEST_TABLE_NAME, (json.dumps(
                {"about": "SHA-256 digest pseudonyms. Each key is the SHA-256 of "
                          f"'{_DIGEST_LOOKUP_TAG}' followed by a real digest in lowercase; each "
                          "value is the pseudonym the shipped files record for it. See "
                          "archive_digest() in scripts/reproduce.py.",
                 "pseudonyms": table}, indent=1, sort_keys=True) + "\n").encode("utf-8"))
        elif in_archive and os.path.isfile(os.path.join(ROOT, DIGEST_TABLE_NAME)):
            with open(os.path.join(ROOT, DIGEST_TABLE_NAME), "rb") as fh:
                put(DIGEST_TABLE_NAME, fh.read())       # an extracted archive rebuilding itself
        # The README goes through the package rename like every shipped text: its run notes
        # name the fold switch (ITASORL_FOLDS), which the renamed code reads under the new name.
        put(README_NAME, hider.sub(rename_pkg(_readme(status, anon_name, dropped=dropped,
                                           films=len(film_slugs()), provenance=no_prov,
                                           profile=profile, left_out=left_out,
                                           commits_hidden=hider.count(),
                                           digests_hidden=hider.digest_count(),
                                           branches_hidden=len(set(branch_rule[0][1].used.values())),
                                           title_checked=bool(titles) or not anon_name,
                                           histories=len(histories))))
            .encode("utf-8"))
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    leaks = leak_scan(path, rules, old_name, titles=titles, branches=branches)
    unviewed = [n for n in leaks if is_image(n)]
    if unviewed:
        print(f"  images not on the reviewed list (view each, then add its SHA-256 to "
              f"REVIEWED_IMAGES in scripts/reproduce.py): {unviewed}")
    resolvable = commit_scan(path, hider)
    if resolvable:
        print(f"  COMMIT IDS THAT STILL RESOLVE in {resolvable}")
    real = (digest_scan(path, tracked_digests() | set(hider.digests))
            if anon_name and not in_archive else [])
    if real:
        print(f"  REAL SHA-256 DIGESTS in {real}")
    for b in broken:
        print(f"  NUMBER CHANGED BY A PSEUDONYM: {b}")
    print(f"wrote {out}: {len(sums) + 1} files ({len(sums)} listed in SHA256SUMS)"
          + (f"; POSSIBLE IDENTITY LEAKS in {leaks}" if leaks
             else "; no identity strings found in any file, binaries included, and every image "
                  "is on the reviewed list"))
    size = os.path.getsize(path)
    print(f"  {size:,} bytes")
    if leaks or resolvable or real or broken:
        os.remove(path)
        print("archive removed; no map written")
        return 1
    if profile == "review" and size > REVIEW_LIMIT:
        os.remove(path)
        print(f"--profile review: {size:,} bytes exceeds the limit of {REVIEW_LIMIT:,} bytes "
              "(100 MB); archive removed")
        return 1
    if hider.mapping or hider.digest_mapping or hider.forced_mapping:
        # Outside the archive, beside it, for the camera-ready version. With the key, the
        # commit list gives back every commit pseudonym of this build
        # (CommitPseudonyms(commits, key)); the digest and unresolved-commit pseudonyms are
        # listed in full.
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        with open(commit_map_path(path), "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"archive": os.path.basename(path), "archive_sha256": digest,
                       "commits": hider.commits,
                       "pseudonyms": dict(sorted(hider.mapping.items())),
                       "unresolved_commit_fields": dict(sorted(hider.forced_mapping.items())),
                       "digests": dict(sorted(hider.digest_mapping.items()))}, fh, indent=1)
            fh.write("\n")
        print(f"  {hider.count()} commit IDs replaced by keyed pseudonyms "
              f"({len(hider.mapping)} distinct spellings), {len(hider.forced_mapping)} unresolved "
              f"commit-field spellings and {hider.digest_count()} SHA-256 digests "
              f"({len(hider.digest_mapping)} spellings); map {commit_map_path(out)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--dumps", nargs="*", default=[])
    t.add_argument("--check", action="store_true",
                   help="compare the committed tables and pages with the builders' output and run "
                        "the audit, rewriting nothing; exit 3 when a page is stale")
    r = sub.add_parser("retrain")
    r.add_argument("run")
    r.add_argument("--execute", action="store_true")
    r.add_argument("--dest", default=None,
                   help="folder that receives the retrained run and every follow-up output "
                        "(default dist/retrain_<UTC date>); never the repository root, "
                        "artifacts/ or docs/")
    s = sub.add_parser("supplement")
    s.add_argument("--out", default=None,
                   help="archive path (default dist/supplement.zip, or dist/supplement-review.zip "
                        "for --profile review)")
    s.add_argument("--profile", choices=PROFILES, default="full",
                   help="full: every bundle in RUN_DIRS; review: only the held-out and seed-2 "
                        "saved agents and the C1 and C2 rerun agents and cells, failing above "
                        "100 MB, 100,000,000 bytes (default %(default)s)")
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
                   help="fail, writing nothing, when a run bundle of the profile is absent "
                        "(for the submission build)")
    s.add_argument("--paper", default=None,
                   help="the manuscript's main.tex, or its folder, for the title check (default: "
                        "docs/paper/ here, else in the main checkout of a worktree); a review "
                        "build refuses to start when no title can be read")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        return tables(a.dumps, check_only=a.check)
    if a.cmd == "retrain":
        return retrain(a.run, a.execute, a.dest)
    out = a.out or ("dist/supplement-review.zip" if a.profile == "review" else "dist/supplement.zip")
    return supplement(out, runs_root=a.runs_root, anon_name=None if a.keep_name else a.anon_name,
                      require_runs=a.require_runs, profile=a.profile, require_history=True,
                      paper=a.paper)


if __name__ == "__main__":
    raise SystemExit(main())
