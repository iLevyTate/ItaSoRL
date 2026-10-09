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
