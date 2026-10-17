"""Revision step 15: frozen surrogates, the reproduce entry point, and supplement scrubbing."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from itasorl.surrogate_l3 import GMotion, train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_gmotion_roundtrips_through_npz(tmp_path):
    g = train_g_motion(hidden=4, n_eps=10, steps=8, epochs=20, params=P, ray_steps=3)
    p = tmp_path / "g.npz"
    g.to_npz(str(p))
    h = GMotion.from_npz(str(p))
    for v, a in ((np.array([0.1, -0.2]), np.array([0.3, 0.0])), (np.zeros(2), np.ones(2))):
        assert np.array_equal(g(v, a), h(v, a))


def test_exported_surrogates_match_their_index_and_a_fresh_retrain():
    d = os.path.join(ROOT, "artifacts", "surrogates")
    with open(os.path.join(d, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    for name, meta in index["files"].items():
        with open(os.path.join(d, name), "rb") as fh:
            import reproduce
            assert reproduce.archive_digest(hashlib.sha256(fh.read()).hexdigest()) == meta["sha256"], name
    g = GMotion.from_npz(os.path.join(d, "gmotion_h8_s0.npz"))
    # The export trains single-threaded (as the run workers do); a multi-threaded CPU fit
    # moves the last bits (about 1e-5), so pin one thread for the comparison.
    import torch
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fresh = train_g_motion(hidden=8, seed=0, params=P)      # the recipe, on CPU
    finally:
        torch.set_num_threads(n_threads)
    rng = np.random.default_rng(0)
    for _ in range(5):
        v, a = rng.normal(size=2), rng.normal(size=2)
        assert np.allclose(g(v, a), fresh(v, a), atol=1e-6)


def test_scrub_removes_identity_words_without_touching_ordinary_words():
    import reproduce
    rules = [(p, r) for p, r in reproduce.GENERIC_SCRUB]
    import re
    # The fixture names are assembled at runtime so the supplement's own scrubber cannot
    # rewrite this test when it is packaged (the archive must pass its own test suite).
    family, given = "T" + "ate", "L" + "evy"
    handle = "i" + given + family
    for t in (family, given, handle):
        rules.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z0-9])"), "<author>"))
    # Likewise the path and e-mail fixtures: written literally they would be scrubbed too.
    win, email, home = "C:/" + "Users/someone/x", "a@" + "b.co", "/ho" + "me/user/x"
    text = f"state estimate {family} {given} {handle} {win} {email} {home}"
    out = reproduce.scrub(text, rules)
    assert out.startswith("state estimate <author> <author> <author>")
    assert "someone" not in out and email not in out and home not in out


def test_json_escaped_windows_paths_are_scrubbed():
    import reproduce
    # json.dumps doubles every backslash, which is how a Windows path sits in a JSON artifact.
    # Assembled at runtime so the scrubber does not rewrite this fixture inside the archive.
    win = "C:" + "\\Users\\someone\\repo\\x.json"
    text = json.dumps({"path": win, "p2": "C:" + "/Use" + "rs/other/y"})
    assert "\\\\Users" in text
    out = reproduce.scrub(text, reproduce.GENERIC_SCRUB)
    assert "someone" not in out and "other" not in out


def _in_git_checkout() -> bool:
    """True when ROOT is the top of a git checkout (an extracted archive placed inside some
    other repository is still an extracted archive)."""
    import subprocess
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and os.path.normcase(os.path.realpath(top)) == \
        os.path.normcase(os.path.realpath(ROOT))


def test_package_file_list_follows_git_not_the_filesystem():
    import reproduce
    if not _in_git_checkout():
        pytest.skip("git-following file list needs a checkout (an extracted archive walks the tree)")
    ignored = os.path.join(ROOT, "artifacts", "clip_audit", "_probe_ignored.json")
    os.makedirs(os.path.dirname(ignored), exist_ok=True)
    try:
        with open(ignored, "w", encoding="utf-8") as fh:
            fh.write("{}\n")
        files = reproduce._files()
        assert "artifacts/clip_audit/_probe_ignored.json" not in files
    finally:
        os.remove(ignored)
        try:
            os.rmdir(os.path.dirname(ignored))
        except OSError:
            pass
    assert "viz/collect.py" in files and "notebooks/colab_gpu.ipynb" in files
    assert "itasorl/experiment_b2.py" in files and "artifacts/gate_table.json" in files


@pytest.fixture(scope="module")
def built_supplement(tmp_path_factory):
    """One archive built from this checkout with no run bundles, shared by the tests that read
    a whole archive. Skipped in an extracted archive, which has no identity terms to check
    and is not where the archive is built."""
    import reproduce
    if not reproduce.identity_terms():
        pytest.skip("no identity terms here (an anonymized copy rebuilding itself)")
    d = tmp_path_factory.mktemp("supplement")
    out = d / "supp.zip"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(reproduce, "RUN_DIRS", [])
        assert reproduce.supplement(str(out), runs_root=str(d / "no-runs")) == 0
    return out


def test_supplement_scrubs_every_text_member_including_cff_html_js_and_license(built_supplement):
    import re
    import zipfile
    import reproduce
    terms = reproduce.identity_terms()
    pats = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z0-9])") for t in terms]
    pats += [p for p, _ in reproduce.film_rules()]
    with zipfile.ZipFile(built_supplement) as z:
        names = set(z.namelist())
        for must in ("CITATION.cff", "LICENSE", "index.html", "viz/player/brain/brain.js",
                     "viz/player/index.html", "viz/collect.py", "notebooks/colab_gpu.ipynb",
                     "SHA256SUMS", "README_SUPPLEMENT.md"):
            assert must in names, must
        assert not any(n.startswith("artifacts/clip_audit/") for n in names)
        for n in names:
            if reproduce.is_text(n):
                t = z.read(n).decode("utf-8", errors="replace")
                for pat in pats:
                    assert not pat.search(t), (n, pat.pattern)
        # SHA256SUMS lists every member but itself, the generated files included.
        listed = {ln.split("  ", 1)[1] for ln in z.read("SHA256SUMS").decode("utf-8").splitlines()}
        assert listed == names - {"SHA256SUMS"}
        readme = z.read("README_SUPPLEMENT.md").decode("utf-8")
        assert "pip install -r requirements-dev.txt" in readme and "python -m pytest -q" in readme
        assert "gitignored" not in readme


def test_the_extracted_supplement_passes_its_audit_and_site_tests(built_supplement, tmp_path):
    """The archive must pass its own checks (docs/REPRODUCE.md says so). Twice it did not: a
    file the scalar audit or the site tests read was not packaged, and the audit stopped with
    FileNotFoundError. Extract the archive and run the audit and the site tests inside it."""
    import subprocess
    import zipfile
    x = tmp_path / "x"
    with zipfile.ZipFile(built_supplement) as z:
        z.extractall(x)
    audit = subprocess.run([sys.executable, os.path.join("scripts", "audit_stats_recheck.py")],
                           cwd=x, capture_output=True, text=True, timeout=600)
    assert audit.returncode == 0, (audit.stdout[-4000:], audit.stderr[-4000:])
    assert "RESULT: all" in audit.stdout
    site = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           os.path.join("tests", "test_site_logic.py")],
                          cwd=x, capture_output=True, text=True, timeout=600)
    assert site.returncode == 0, site.stdout[-4000:]
    assert " failed" not in site.stdout


def test_the_scrubber_does_not_rewrite_its_own_source():
    """scripts/reproduce.py is packaged too. If its own rule patterns matched its rules, the
    archived copy would ship with the path and e-mail rules replaced by placeholders."""
    import reproduce
    src = open(os.path.join(ROOT, "scripts", "reproduce.py"), encoding="utf-8").read()
    assert reproduce.scrub(src, reproduce.scrub_rules()) == src


def test_identity_terms_ignore_the_scrubber_placeholders(tmp_path, monkeypatch):
    """An extracted archive rebuilding itself reads a scrubbed CITATION.cff; '<author>' must not
    become an identity term, or every scrubbed file reports as a leak."""
    import reproduce
    (tmp_path / "CITATION.cff").write_text(
        'authors:\n  - family-names: "<author>"\n    given-names: "<author>"\n', encoding="utf-8")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    assert reproduce.identity_terms() == []


def test_reproduce_tables_checks_the_committed_pages_before_rewriting(monkeypatch):
    import reproduce
    calls = []
    monkeypatch.setattr(reproduce, "_run", lambda cmd: calls.append(cmd) or 0)
    assert reproduce.tables([]) == 0
    builders = [c for c in calls if "build_" in c[1]]
    checks = [c for c in builders if "--check" in c]
    writes = [c for c in builders if "--check" not in c]
    assert checks and writes
    assert calls.index(checks[-1]) < calls.index(writes[0])



def test_the_email_rule_leaves_the_project_s_own_condition_labels_alone():
    """The record, the append-only deviation log and a frozen decision rule all write
    conditions as "train@0.45" and "eval@0.45". An earlier rule treated those as addresses and
    shipped the anonymized archive with the numbers stripped of the conditions they belong to.
    Fixtures are assembled at runtime so this file's own text is not scrubbed when packaged."""
    import reproduce
    rules = [(p, r) for p, r in reproduce.GENERIC_SCRUB]
    at = "@"
    keep = ["train" + at + "0.45", "eval" + at + "0.45", "react" + at + "18.3.1",
            "react-dom" + at + "18.3.1", "numpy" + at + "1.26"]
    for s in keep:
        assert reproduce.scrub(s, rules) == s, f"{s} must survive the scrub"

    strip = ["a" + at + "b.co", "first.last+tag" + at + "sub.example.com",
             "chain" + at + "itasorl.local", "me" + at + "1password.com",
             "12345" + at + "gmail.com"]
    for s in strip:
        assert reproduce.scrub(s, rules) == "<email>", f"{s} must be scrubbed"


def test_the_email_rule_still_scrubs_an_address_inside_a_sentence():
    import reproduce
    rules = [(p, r) for p, r in reproduce.GENERIC_SCRUB]
    at = "@"
    text = "write to someone" + at + "example.org about train" + at + "0.45"
    assert reproduce.scrub(text, rules) == "write to <email> about train" + at + "0.45"


def test_tables_dumps_passes_the_output_path_rescore_fold_split_requires(monkeypatch, tmp_path):
    """scripts/rescore_fold_split.py declares --json required=True, so the one documented
    reanalysis path exited 2 without it and had never run."""
    import reproduce

    calls = []
    monkeypatch.setattr(reproduce, "_run", lambda cmd, **kw: calls.append(cmd) or 0)
    monkeypatch.setattr(reproduce, "BUILDERS", [])
    d = tmp_path / "run" / "artifacts" / "states"
    d.mkdir(parents=True)
    reproduce.tables([str(d)])

    rescore = [c for c in calls if any("rescore_fold_split" in str(x) for x in c)]
    assert len(rescore) == 1, calls
    assert "--json" in rescore[0], rescore[0]
    out = rescore[0][rescore[0].index("--json") + 1]
    assert out.endswith(".json") and "dist" in out
    # A trailing separator must not collapse every run's output onto one name.
    calls.clear()
    reproduce.tables([str(d) + os.sep])
    other = [c for c in calls if any("rescore_fold_split" in str(x) for x in c)][0]
    assert other[other.index("--json") + 1] == out


def test_tables_names_each_run_s_outputs_after_the_run(monkeypatch, tmp_path):
    """The documented dump layout is fullruns/<run>/states, so the leaf is always "states".
    Two runs must not write the same rescore file, and the behavior control must be saved
    rather than printed."""
    import reproduce

    calls = []
    monkeypatch.setattr(reproduce, "_run", lambda cmd, **kw: calls.append(cmd) or 0)
    monkeypatch.setattr(reproduce, "BUILDERS", [])
    dirs = []
    for run in ("corrected_l3_h8_wm", "corrected_l3_h8_nowm"):
        d = tmp_path / "fullruns" / run / "states"
        d.mkdir(parents=True)
        dirs.append(str(d))
    reproduce.tables(dirs)

    def outputs(script):
        rows = [c for c in calls if any(script in str(x) for x in c)]
        assert len(rows) == 2, calls
        assert all("--json" in c for c in rows), rows
        return [os.path.basename(c[c.index("--json") + 1]) for c in rows]

    assert outputs("rescore_fold_split") == ["fold_rescore_corrected_l3_h8_wm.json",
                                             "fold_rescore_corrected_l3_h8_nowm.json"]
    assert outputs("audit_behavior_mediation") == ["behavior_audit_corrected_l3_h8_wm.json",
                                                   "behavior_audit_corrected_l3_h8_nowm.json"]
    assert reproduce.dump_tag("x/run_a/states/") == reproduce.dump_tag("x/run_a/states") == "run_a"
    assert reproduce.dump_tag("x/run_b/custom_dumps") == "custom_dumps"


# Fixture hosts and URLs are assembled at runtime: written literally they would be matched by
# the archive's own scrub and leak checks when this file is packaged.
_GH = "git" + "hub"
_ATTACH = "https://" + _GH + ".com/user-" + "attachments/assets/0000-aaaa"
_PAGES_IMG = "https://someone." + _GH + ".io/site/card.png"
_RAW_IMG = "https://raw." + _GH + "usercontent.com/someone/site/main/card.png"


def _hosted_image_re():
    import re
    host = (r"https?://(?:" + _GH + r"\.com/user-" + r"attachments/|[\w.-]*" + _GH
            + r"usercontent\.com/|[\w-]+\." + _GH + r"\.io/)")
    return re.compile(r"(<img\b[^>]*\bsrc\s*=\s*[\"']?|!\[[^\]]*\]\(\s*<?)" + host, re.I)


def test_externally_hosted_images_are_removed_and_local_images_kept():
    import reproduce
    text = "\n".join([
        f'<img width="2560" height="1280" alt="social preview" src="{_ATTACH}" />',
        f"![card]({_PAGES_IMG})",
        f"<img src='{_RAW_IMG}'>",
        '<img src="docs/figures/budget_curve.png" alt="local">',
        "![local figure](docs/figures/l3_summary.png)",
    ])
    out = reproduce.scrub(text, reproduce.IMAGE_SCRUB)
    assert out.count(reproduce.IMAGE_PLACEHOLDER) == 3
    assert not _hosted_image_re().search(out)
    assert 'src="docs/figures/budget_curve.png"' in out
    assert "![local figure](docs/figures/l3_summary.png)" in out


def test_no_packaged_page_keeps_an_externally_hosted_image():
    """README.md carried the social-preview image, which renders the owner's site address.
    Every packaged .md and .html file, once scrubbed, must be free of such images; images in
    general are fine (FINDINGS and the site embed local figures)."""
    import reproduce
    rules = reproduce.scrub_rules()
    pat = _hosted_image_re()
    bad = []
    for rel in reproduce._files():
        if rel.endswith((".md", ".html")):
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
                if pat.search(reproduce.scrub(fh.read(), rules)):
                    bad.append(rel)
    assert bad == []


def test_repository_urls_are_replaced_before_the_owner_rule(monkeypatch, tmp_path):
    """Applied after the owner rule, the URL patterns no longer match, and the repository name
    survives next to <author>. The URL rules run first and take the whole URL."""
    import reproduce
    owner, repo = "Some" + "Owner", "Some" + "Repo"
    monkeypatch.setattr(reproduce, "_remote", lambda: (owner, repo))
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))     # no CITATION.cff: owner only
    assert reproduce.identity_terms() == [owner]
    text = " | ".join([
        "https://" + owner.lower() + "." + _GH + ".io/" + repo + "/",
        "https://" + _GH + ".com/" + owner + "/" + repo + ".git",
        "git@" + _GH + ".com:" + owner + "/" + repo + ".git",
        "https://colab.research.google.com/" + _GH + "/" + owner + "/" + repo + "/blob/main/nb.ipynb",
        "https://" + _GH + ".com/" + owner + "/" + repo.upper() + "/issues",
        "written by " + owner,
    ])
    out = reproduce.scrub(text, reproduce.scrub_rules())
    ph = reproduce.REPO_PLACEHOLDER
    assert out == " | ".join([ph + "/", ph, ph, ph + "/blob/main/nb.ipynb", ph + "/issues",
                              "written by <author>"]), out
    assert repo.lower() not in out.lower()


def test_the_package_rename_follows_case_and_reaches_file_names():
    import reproduce
    old = reproduce.project_name()
    assert old, "pyproject.toml must name the package"
    new = "blind" + "pkg"
    ren = reproduce.renamer(old, new)
    mixed = "".join(c.upper() if i % 2 else c for i, c in enumerate(old))
    text = (f"import {old}.stats\n{old.upper()}_FOLDS=legacy\n{old.capitalize()}Overview\n"
            f"{mixed} site\nsee docs/{old.upper()}_world_spec.md")
    out = ren(text)
    assert out == (f"import {new}.stats\n{new.upper()}_FOLDS=legacy\n{new.capitalize()}Overview\n"
                   f"{new.capitalize()} site\nsee docs/{new.upper()}_world_spec.md")
    assert ren(f"{old}/world.py") == f"{new}/world.py"
    assert reproduce.renamer(old, None)(text) == text
    with pytest.raises(SystemExit):
        reproduce.check_anon_name("Not-An-Identifier", old)
    with pytest.raises(SystemExit):
        reproduce.check_anon_name(old + "2", old)
    reproduce.check_anon_name(old, old)        # an archive rebuilding itself: no-op, allowed


def test_supplement_renames_the_package_and_reports_each_run_bundle(tmp_path, monkeypatch):
    """Run bundles are read from --runs-root; a present one ships under fullruns/, an absent
    one is named in README_SUPPLEMENT.md. The package directory and every spelling of its
    name are replaced in member names and text."""
    import zipfile
    import reproduce
    old = reproduce.project_name()
    new = "blind" + "pkg"
    runs = tmp_path / "runs"
    (runs / "present" / "agents").mkdir(parents=True)
    (runs / "present" / "agents" / "agent_s0.pt").write_bytes(b"PK-free weights \x00\x01")
    monkeypatch.setattr(reproduce, "RUN_DIRS", ["fullruns/present/agents", "fullruns/absent"])
    monkeypatch.setattr(reproduce, "INCLUDE_DIRS", [old])
    monkeypatch.setattr(reproduce, "INCLUDE_FILES", ["pyproject.toml", "pytest.ini"])
    out = tmp_path / "supp.zip"
    assert reproduce.supplement(str(out), runs_root=str(runs), anon_name=new) == 0
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "fullruns/present/agents/agent_s0.pt" in names
        assert f"{new}/stats.py" in names and f"{new}/__init__.py" in names
        assert not any(old in n.lower() for n in names), names
        readme = z.read("README_SUPPLEMENT.md").decode("utf-8")
        # one Markdown bullet per bundle, so the list renders as a list
        assert "\n- included: `fullruns/present/agents` (1 files)\n" in readme
        assert "\n- not included: `fullruns/absent`\n" in readme
        for n in names:
            if reproduce.is_text(n):
                assert old not in z.read(n).decode("utf-8", errors="replace").lower(), n
        assert f'name = "{new}"' in z.read("pyproject.toml").decode("utf-8")
    # The submission build can insist on every bundle: it then writes nothing.
    strict = tmp_path / "strict.zip"
    assert reproduce.supplement(str(strict), runs_root=str(runs), anon_name=new,
                                require_runs=True) == 1
    assert not strict.exists()


def test_the_leak_scan_reads_binary_members_and_zip_containers(tmp_path):
    """Agent bundles (.pt) and arrays (.npz) are zip containers; an .npz member is deflated, so
    a raw byte scan alone would miss a path or name inside it. The e-mail shape is left out of
    the binary scan because arbitrary bytes match it by chance."""
    import io
    import zipfile
    import reproduce
    term = "Some" + "Owner"
    rules = reproduce.term_rules([term]) + list(reproduce.GENERIC_SCRUB)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zi:
        zi.writestr("arr/data.pkl", b"\x80\x02" + term.encode() + b"\x00" * 64)
    clean = io.BytesIO()
    with zipfile.ZipFile(clean, "w", zipfile.ZIP_DEFLATED) as zc:
        zc.writestr("arr/data.pkl", b"\x80\x02 weights only " + b"x" + b"@" + b"ab.cd")
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fullruns/r/agents/bad.npz", inner.getvalue())
        z.writestr("fullruns/r/agents/good.pt", clean.getvalue())
        z.writestr("notes.md", "nothing to see")
    assert term.encode() not in inner.getvalue(), "the fixture must hide the term in deflate"
    assert reproduce.leak_scan(str(path), rules, None) == ["fullruns/r/agents/bad.npz"]


def test_the_leak_scan_fails_on_any_image_not_on_the_reviewed_list(tmp_path):
    """A figure title is pixels, so no pattern can find a name drawn into it. An image ships
    only when its exact bytes were viewed and listed; a redrawn or new image fails."""
    import zipfile
    import reproduce
    ok = b"\x89PNG reviewed bytes"
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("docs/figures/ok.png", ok)
        z.writestr("docs/figures/new.png", b"\x89PNG never viewed")
        z.writestr("fullruns/r/expB2_survival.png", b"\x89PNG a run script figure")
        z.writestr("docs/figures/plot.svg", "<svg><text>chart</text></svg>")
    reviewed = {"docs/figures/ok.png": hashlib.sha256(ok).hexdigest()}
    assert reproduce.leak_scan(str(path), [], None, reviewed=reviewed) == [
        "docs/figures/new.png", "fullruns/r/expB2_survival.png", "docs/figures/plot.svg"]


def test_every_image_the_supplement_packages_is_on_the_reviewed_list():
    import reproduce
    pairs, _ = reproduce._sources(runs_root=os.path.join(ROOT, "no-such-runs-root"))
    images = [(arc, src) for arc, src in pairs if reproduce.is_image(arc)]
    assert images, "the supplement ships figures; the check must see them"
    for arc, src in images:
        with open(src, "rb") as fh:
            data = fh.read()
        assert reproduce.REVIEWED_IMAGES.get(arc) == reproduce.archive_digest(hashlib.sha256(data).hexdigest()), \
            f"{arc} is not on the reviewed list with these bytes; view it, then list its hash"


# ---- final review, 2026-10-10: remote parsing, path forms, films, links, provenance ----
# Every fixture below that looks like a path, an iD or a title is assembled at runtime, so the
# archive's own scrub leaves this file as written when it is packaged.
_PROF = "Use" + "rs"


def test_only_a_github_remote_names_the_owner(tmp_path, monkeypatch):
    """A clone of a local checkout has a filesystem path as its origin; parsing it took a path
    segment as the owner and rewrote ordinary words. Only a GitHub URL counts, and CITATION.cff
    is the fallback."""
    import reproduce
    owner, repo = "Some" + "Owner", "Some" + "Repo"
    host = _GH + ".com"
    for url in ("https://" + host + "/" + owner + "/" + repo + ".git",
                "https://" + host + "/" + owner + "/" + repo,
                "git@" + host + ":" + owner + "/" + repo + ".git",
                "ssh://git@" + host + "/" + owner + "/" + repo + ".git"):
        assert reproduce.parse_github(url) == (owner, repo), url
    for url in ("C:/" + _PROF + "/someone/Documents/" + _GH + "/" + repo,
                "/srv/mirrors/" + owner + "/" + repo + ".git",
                "https://gitlab.example.org/" + owner + "/" + repo, ""):
        assert reproduce.parse_github(url) is None, url
    # outside a checkout (no origin) the owner comes from CITATION.cff's repository-code
    (tmp_path / "CITATION.cff").write_text(
        'repository-code: "https://' + host + "/" + owner + "/" + repo + '"\n', encoding="utf-8")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    assert reproduce._remote() == (owner, repo)


def test_every_local_profile_form_is_scrubbed():
    import reproduce
    forms = ["/c/" + _PROF + "/someone/repo/x.json",          # Git Bash
             "/mnt/c/" + _PROF + "/someone/repo",             # WSL
             "/" + _PROF + "/someone/repo",                   # macOS
             "C:/" + _PROF + "/someone/repo",                 # Windows
             "c:\\" + _PROF.lower() + "\\someone\\repo",      # normcased Windows
             "C--" + _PROF + "-someone-Documents-repo"]       # a path flattened into a name
    for f in forms:
        out = reproduce.scrub("at " + f + " end", reproduce.GENERIC_SCRUB)
        assert "someone" not in out and "<local-path>" in out, (f, out)
    # a web URL with a lower-case users segment is not a local path
    url = "https://api.example.org/" + _PROF.lower() + "/someone"
    assert reproduce.scrub(url, reproduce.GENERIC_SCRUB) == url


def test_a_bare_orcid_id_is_an_identity_term(tmp_path, monkeypatch):
    import reproduce
    orcid = "0000-0002-" + "1825-0097"
    (tmp_path / "CITATION.cff").write_text(
        'authors:\n  - family-names: "Doe"\n    orcid: "https://orc' + 'id.org/' + orcid + '"\n',
        encoding="utf-8")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    monkeypatch.setattr(reproduce, "_remote", lambda: None)
    terms = reproduce.identity_terms()
    assert orcid in terms
    out = reproduce.scrub("iD " + orcid + ".", reproduce.term_rules(terms))
    assert orcid not in out and "<author>" in out


def test_the_build_machine_account_and_the_local_title_are_leak_checks(tmp_path, monkeypatch):
    import reproduce
    acct = "some" + "acct"
    monkeypatch.setenv("HOME", str(tmp_path / acct))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / acct))
    assert reproduce.local_account() == acct
    monkeypatch.setenv("HOME", str(tmp_path / "runner"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "runner"))
    assert reproduce.local_account() is None                  # a CI default is not a check
    monkeypatch.setattr(reproduce, "local_account", lambda: acct)
    paper = tmp_path / "docs" / "paper"
    paper.mkdir(parents=True)
    (paper / "main.tex").write_text("\\title{Quiet Words In A Row: A Subtitle}\n", encoding="utf-8")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    assert reproduce.paper_title() == "Quiet Words In A Row"
    pats = reproduce._leak_patterns(None)
    assert any(p.search("dir of " + acct.upper() + " here") for p in pats)
    assert any(p.search("quiet words\nin a row") for p in pats)
    assert not any(p.search(acct + "s and quiet words") for p in pats)


def test_film_titles_become_neutral_names_in_text_and_paths(tmp_path, monkeypatch):
    import reproduce
    film = tmp_path / "assets" / "film"
    film.mkdir(parents=True)
    for n in ("bright-north-star-web.mp4", "bright-north-star-poster.jpg", "quiet-hands-web.mp4",
              "README.md"):
        (film / n).write_bytes(b"x")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    assert reproduce.film_slugs() == ["bright-north-star", "quiet-hands"]
    rules = reproduce.film_rules()
    text = ("*Bright North Star* (released 24 March\n2025), then QUIET HANDS, quiet_hands.py, "
            "build-quiet-hands.ps1 and a spec released on 2025-" + "04-01; brightness stays, "
            "and with quiet hands in plain prose")
    out = reproduce.scrub(text, rules)
    assert out == ("*Film One* (released <date>), then FILM TWO, film_two.py, "
                   "build-film-two.ps1 and a spec released on <date>; brightness stays, "
                   "and with quiet hands in plain prose")
    # the leak check reads the same patterns, so ordinary prose is not a leak either
    assert not any(p.search("with quiet hands in plain prose") for p, _ in rules)
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path / "nowhere"))
    assert reproduce.film_slugs() == [] and reproduce.film_rules() == []


def test_outreach_specs_and_run_figures_are_left_out():
    import reproduce
    assert reproduce.excluded("docs/specs/2026-07-14-outreach-video-design.md")
    assert reproduce.excluded("artifacts/goal_stakes/T-touch/expB2_survival.png")
    assert reproduce.excluded("artifacts/goal_stakes/quarantine/x_2026-10-10/expB2_survival.png")
    assert reproduce.excluded("fullruns/corrected_l3_h8_wm/expB2_survival.png")
    for ships in ("artifacts/goal_stakes/T-touch/expB2_results.json",
                  "fullruns/corrected_l3_h8_wm/agents/agent_d0.45_s0_survival.pt",
                  "docs/specs/2026-10-06-primary-analysis-and-l0.md",
                  "artifacts/expB2/expB2_survival.png"):
        assert reproduce.excluded(ships) is None, ships


def test_links_the_archive_cannot_resolve_become_plain_text():
    import reproduce
    ph = reproduce.REPO_PLACEHOLDER
    shipped = {"docs/FINDINGS.md", "docs/specs/a.md", "artifacts/expB2/x.json", "README.md"}
    text = "\n".join([
        "[site](" + ph + "/)",
        "[![Badge](https://img.example.org/b.svg)](" + ph + "/blob/main/nb.ipynb)",
        "[film notes](assets/film/README.md)",
        "[findings](docs/FINDINGS.md#176-who)",
        "[folder](artifacts/expB2/)",
        "[web](https://example.org/x)",
        "[anchor](#top)",
        "![figure](docs/figures/f.png)",
    ])
    out = reproduce.delink(text, "README.md", shipped)
    assert out.split("\n") == ["site", "", "film notes", "[findings](docs/FINDINGS.md#176-who)",
                               "[folder](artifacts/expB2/)", "[web](https://example.org/x)",
                               "[anchor](#top)", "![figure](docs/figures/f.png)"]
    # relative to the file that holds the link
    assert reproduce.delink("[a](a.md) [up](../FINDINGS.md) [out](../../x.md)",
                            "docs/specs/b.md", shipped) == "[a](a.md) [up](../FINDINGS.md) out"


def _git_in(d, *args, when=None):
    import subprocess
    env = dict(os.environ)
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    res = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org",
                          "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                         cwd=d, env=env, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def test_provenance_lists_every_commit_in_a_built_history(tmp_path, monkeypatch):
    """Every commit to a file, oldest first: the first commit under an earlier path, side-branch
    edits (one of them later duplicated on the main line, so plain `git log` hides it), and the
    merge that joins edits from two branches; not the merges that add nothing to the file."""
    import shutil

    import reproduce
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    d = str(tmp_path)
    doc = "docs/PREREGISTRATION_X.md"
    n = [0]

    def commit(msg, files=None):
        n[0] += 1
        when = f"2026-01-01T00:{n[0]:02d}:00+00:00"
        for path, text in (files or {}).items():
            os.makedirs(os.path.dirname(os.path.join(d, path)) or d, exist_ok=True)
            with open(os.path.join(d, path), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            _git_in(d, "add", path)
        _git_in(d, "commit", "-q", "-m", msg, when=when)
        return _git_in(d, "rev-parse", "--short", "HEAD")

    def merge(branch):
        n[0] += 1
        when = f"2026-01-01T00:{n[0]:02d}:00+00:00"
        _git_in(d, "merge", "-q", "--no-ff", "--no-edit", branch, when=when)
        return _git_in(d, "rev-parse", "--short", "HEAD")

    def text(a="a", b="b", c="c", e="d"):
        return f"{a}\n{b}\n{c}\n{e}\n"

    _git_in(d, "init", "-q")
    _git_in(d, "symbolic-ref", "HEAD", "refs/heads/main")
    c1 = commit("add", {"PREREGISTRATION_X.md": text()})
    os.makedirs(os.path.join(d, "docs"))
    _git_in(d, "mv", "PREREGISTRATION_X.md", doc)
    c2 = commit("move")
    _git_in(d, "branch", "side")
    _git_in(d, "checkout", "-q", "side")
    c3 = commit("side a", {doc: text(a="a2")})
    _git_in(d, "checkout", "-q", "main")
    c4 = commit("main d", {doc: text(e="d2")})
    m1 = merge("side")                                    # joins a2 and d2: listed
    _git_in(d, "branch", "side2")
    _git_in(d, "checkout", "-q", "side2")
    c5 = commit("side2 c", {doc: text(a="a2", c="c2", e="d2")})
    _git_in(d, "checkout", "-q", "main")
    c6 = commit("main c, same text", {doc: text(a="a2", c="c2", e="d2")})
    merge("side2")                                        # same as both parents: not listed
    _git_in(d, "branch", "side3")
    _git_in(d, "checkout", "-q", "side3")
    commit("other file", {"other.txt": "x\n"})
    _git_in(d, "checkout", "-q", "main")
    c7 = commit("main b", {doc: text(a="a2", b="b2", c="c2", e="d2")})
    merge("side3")                                        # file as on main: not listed
    assert c5 not in _git_in(d, "log", "--format=%h", "--", doc).split()   # plain log hides it

    monkeypatch.setattr(reproduce, "ROOT", d)
    rows, why = reproduce.provenance_rows([doc])
    assert why is None
    (rel, commits, dirty), = rows
    assert rel == doc and not dirty
    assert [h for _, h, _ in commits] == [c1, c2, c3, c4, m1, c5, c6, c7]
    assert [note for *_, note in commits] == ["PREREGISTRATION_X.md", "", "", "", "merge",
                                               "", "", ""]
    assert [w for w, _, _ in commits] == [f"2026-01-01 00:{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 10)]
    md = reproduce.provenance_md(rows, lambda s: s)
    assert f"| `{doc}` | 8 | 2026-01-01 00:01 ({c1}) | 2026-01-01 00:10 ({c7}) |" in md
    assert f"8 commits, first 2026-01-01 00:01 ({c1}), last 2026-01-01 00:10 ({c7})." in md
    assert f"| 1 | 2026-01-01 00:01 | {c1} | path `PREREGISTRATION_X.md` |" in md
    assert f"| 5 | 2026-01-01 00:05 | {m1} | merge |" in md
    assert "@" not in md and "t@example" not in md
    with open(os.path.join(d, doc), "a", encoding="utf-8") as fh:
        fh.write("local edit\n")
    (_, _, dirty), = reproduce.provenance_rows([doc])[0]
    assert dirty
    assert "changes that were not committed" in reproduce.provenance_md(
        reproduce.provenance_rows([doc])[0], lambda s: s)


def test_provenance_lists_every_commit_to_a_registration_without_names(tmp_path, monkeypatch):
    import reproduce
    if not _in_git_checkout():
        pytest.skip("provenance is read from git; an extracted archive has no history")
    rel = "docs/PREREGISTRATION_L3.md"
    rows, why = reproduce.provenance_rows([rel])
    if rows is None:
        assert "shallow" in why
        if os.environ.get("CI") == "true":
            # ci.yml checks out the full history (fetch-depth 0), so a shallow clone in CI is
            # a broken checkout, not a reason to skip.
            pytest.fail(f"CI checkout is a shallow clone: {why}")
        pytest.skip(why)
    (got, commits, _), = rows
    assert got == rel and len(commits) >= 2
    listed = [h for _, h, _ in commits]
    assert len(set(listed)) == len(listed)
    assert [w for w, _, _ in commits] == sorted(w for w, _, _ in commits)
    # every edit on any merged branch, and the merges plain `git log` shows for the file
    every = _git_in(ROOT, "log", "--full-history", "--no-merges", "--format=%h", "--", rel)
    plain = _git_in(ROOT, "log", "--format=%h", "--", rel)
    assert set(every.split()) | set(plain.split()) <= set(listed)
    md = reproduce.provenance_md(rows, lambda s: s)
    n = len(commits)
    assert (f"{n} commits, first {commits[0][0]} ({commits[0][1]}), "
            f"last {commits[-1][0]} ({commits[-1][1]}).") in md
    for i, (when, h, _) in enumerate(commits, 1):
        assert f"| {i} | {when} | {h} |" in md
    assert "@" not in md
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    rows, why = reproduce.provenance_rows([rel])
    assert rows is None and "git checkout" in why


def test_supplement_carries_the_corrected_l1_bundle_and_names_its_absence():
    """The stream and surprise readouts replay from the corrected L1 bundle's agents and state
    dumps (review N09); the supplement ships it when present and says what its absence costs."""
    import reproduce
    key = "fullruns/corrected_l1_heldout"
    assert key in reproduce.RUN_DIRS
    absent = reproduce.RUN_NOTES[key]
    assert "L1-CORRECTED is absent" in absent and "stream" in absent and "surprise" in absent
    assert "docs/specs/2026-10-15-corrected-l1-organism-rerun-design.md" in absent
    present = reproduce.RUN_INCLUDED_NOTES[key]
    assert "60" in present and "state dumps" in present
    assert "stream" in present and "surprise" in present
    assert "artifacts/corrected_runs/l1-corrected" in present
