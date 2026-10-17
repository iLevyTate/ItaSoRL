"""What README_SUPPLEMENT.md says about links and about the C1 and C2 bundles of the review build.

Links: Markdown links to the repository or to files the archive does not carry become plain
text, and the site pages ship only as audit inputs that do not render (2026-10-17). Bundles: the
review build carries the saved agents and cells of the C1 and C2 reruns, and the README names
each of the four folders, says where it came from and how its state dumps rebuild. The README
spells no SHA-256 digest; those are in SHA256SUMS.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import reproduce  # noqa: E402

C1C2 = ["fullruns/corrected_l3_h8_wm/agents", "fullruns/corrected_l3_h8_nowm/agents",
        "fullruns/corrected_l3_h8_wm/cells", "fullruns/corrected_l3_h8_nowm/cells"]


def _review_readme(**kw):
    status = [(r, 3) for r in reproduce.profile_run_dirs("review")]
    left_out = [r for r in reproduce.RUN_DIRS if r not in reproduce.REVIEW_RUN_DIRS]
    return reproduce._readme(status, "blind" + "pkg", profile="review", left_out=left_out, **kw)


def test_the_readme_says_dead_links_read_as_plain_text_and_the_site_pages_do_not_render():
    for text in (reproduce._readme([], "blind" + "pkg"), _review_readme()):
        flat = " ".join(text.split())
        assert ("a link to the repository or to a file the archive does not carry becomes "
                "plain text") in flat
        for page in ("index.html", "index.template.html", "viz/player/index.html"):
            assert page in flat
        assert "ship only as inputs to the scalar audit and the site tests" in flat
        assert "They do not render here" in flat


def test_the_review_build_carries_the_c1_and_c2_agents_and_cells():
    review = reproduce.profile_run_dirs("review")
    assert all(r in review for r in C1C2)
    assert review[:2] == ["fullruns/l3_h8_heldout/agents", "fullruns/l3_h8_gseed2_gpu/agents"]
    # every full-build bundle the review build does not carry whole has a left-out line
    assert set(reproduce.RUN_DIRS) - set(review) <= set(reproduce.REVIEW_LEFT_OUT)


def test_the_review_readme_names_each_c1_c2_folder_with_its_note():
    text = _review_readme()
    flat = " ".join(text.split())
    for r in C1C2:
        assert f"- included: `{r}`" in text, r
        note = reproduce.RUN_INCLUDED_NOTES[r]
        assert " ".join(note.split()) in flat, r
        assert note.startswith(f"`{r}`")
        assert "rerun of C1" in note or "C1 rerun" in note or "rerun of C2" in note \
            or "C2 rerun" in note, r
        assert "regenerate_state_dumps.py" in note, r
    assert "saved agents and cells of the C1 and C2 reruns" in flat
    # the C1 and C2 bundles are not wholly left out: the left-out lines say what stays out
    for r in ("fullruns/corrected_l3_h8_wm", "fullruns/corrected_l3_h8_nowm"):
        assert f"- `{r}`: " in text and "agents/ and cells/" in reproduce.REVIEW_LEFT_OUT[r]


def test_the_readme_carries_no_digest_and_points_to_sha256sums():
    for text in (_review_readme(), reproduce._readme([(r, 3) for r in reproduce.RUN_DIRS],
                                                     "blind" + "pkg")):
        assert not re.search(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])", text)
        assert "`SHA256SUMS` lists the SHA-256 of every file" in text
    for note in reproduce.RUN_INCLUDED_NOTES.values():
        assert not re.search(r"[0-9A-Fa-f]{64}", note)
