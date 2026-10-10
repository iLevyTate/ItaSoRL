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
            assert hashlib.sha256(fh.read()).hexdigest() == meta["sha256"], name
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
    text = json.dumps({"path": win, "p2": "C:" + "/Users/other/y"})
    assert "\\\\Users" in text
    out = reproduce.scrub(text, reproduce.GENERIC_SCRUB)
    assert "someone" not in out and "other" not in out


def _in_git_checkout() -> bool:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() == "true"
    except OSError:
        return False


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


def test_supplement_scrubs_every_text_member_including_cff_html_js_and_license(tmp_path, monkeypatch):
    import re
    import zipfile
    import reproduce
    monkeypatch.setattr(reproduce, "RUN_DIRS", [])
    out = tmp_path / "supp.zip"
    terms = reproduce.identity_terms()
    if not terms:
        pytest.skip("no identity terms here (an anonymized copy rebuilding itself)")
    assert reproduce.supplement(str(out)) == 0
    pats = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z0-9])") for t in terms]
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        for must in ("CITATION.cff", "LICENSE", "index.html", "viz/player/brain/brain.js",
                     "viz/collect.py", "notebooks/colab_gpu.ipynb", "SHA256SUMS"):
            assert must in names, must
        assert not any(n.startswith("artifacts/clip_audit/") for n in names)
        for n in names:
            if reproduce.is_text(n):
                t = z.read(n).decode("utf-8", errors="replace")
                for pat in pats:
                    assert not pat.search(t), (n, pat.pattern)


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
        assert "included      fullruns/present/agents (1 files)" in readme
        assert "not included  fullruns/absent" in readme
        for n in names:
            if reproduce.is_text(n):
                assert old not in z.read(n).decode("utf-8", errors="replace").lower(), n
        assert f'name = "{new}"' in z.read("pyproject.toml").decode("utf-8")


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
        assert reproduce.REVIEWED_IMAGES.get(arc) == hashlib.sha256(data).hexdigest(), \
            f"{arc} is not on the reviewed list with these bytes; view it, then list its hash"
