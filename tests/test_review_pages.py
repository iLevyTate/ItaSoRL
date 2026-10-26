"""The anonymized build's rewrite of the public pages it carries (reproduce.review_page).

README.md, index.html and index.template.html ship as inputs to the scalar audit and the site
tests, and were written for the public repository. The build gives the README's opening in
the first person plural, replaces its film paragraphs with one line, and drops the site's
tagline from the page titles. The fixtures below are made up, since this file ships too and
must not quote the public pages. Each rewrite must be idempotent, so the tests that read the
real pages pass both in a checkout and in an extracted archive, where the pages are already
rewritten."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_index  # noqa: E402
import reproduce  # noqa: E402


def _intro_paragraphs(text: str) -> list[str]:
    head = re.search(r"^## ", text, re.M)
    return re.split(r"\n[ \t]*\n", text[:head.start() if head else len(text)])


# A check of its own, wider than the rewrite: any first person singular left in prose, contracted
# or before punctuation included, once code and numerals (Type I, Phase I) are set aside.
CODE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[^\n]*$|`[^`\n]+`", re.M | re.S)
NUMERAL = re.compile(r"\b(?:Type|Phase|Part|Experiment|Table|Figure|Stage|Level|Class|Section|"
                     r"Chapter|Appendix|Study|Round|Act|War) I\b")
SINGULAR = re.compile(r"(?<![\w`/.\-])I(?=['’](?:m|ve|d|ll)\b|\s+[a-z]|[,.;:)*])"
                      r"|\b[Mm](?:y|e|yself)\b(?!-|\.\w)")


def singular_left(text: str) -> list[str]:
    prose = NUMERAL.sub("", CODE.sub("", text))
    return [prose[max(0, m.start() - 30):m.end() + 30] for m in SINGULAR.finditer(prose)]


def test_the_first_person_singular_becomes_plural():
    text = ("# Title\n\nI built a thing and asked whether it worked. The bar, which I set\n"
            "earlier, held. Then I checked. I was sure, and I am sure.\n\n"
            "- I kept my notes, and they let me see it myself.\n\n"
            "My plan held.\n")
    assert reproduce.review_page("README.md", text) == (
        "# Title\n\nWe built a thing and asked whether it worked. The bar, which we set\n"
        "earlier, held. Then we checked. We were sure, and we are sure.\n\n"
        "- We kept our notes, and they let us see it ourselves.\n\n"
        "Our plan held.\n")


def test_contractions_and_a_heading_line():
    text = "# Title\nI'm here, I've been, I'd go and I'll see.\n"
    assert reproduce.review_page("README.md", text) == (
        "# Title\nWe're here, we've been, we'd go and we'll see.\n")


def test_code_numerals_and_other_words_are_left_alone():
    text = ("Run it.\n\n```\nI x = my_dir / me\n```\n\n~~~\nI ran\n~~~\n\n- item\n\n"
            "  ```\n  I ran\n  ```\n\nSee `I` and `cd my dir`, II, I/O, I-beam, README, mynah, "
            "meme, Phase IV, my.yaml. A Type I error, Phase I was short, Experiment I ran.\n")
    assert reproduce.review_page("README.md", text) == text
    assert singular_left(text) == []


def test_crlf_line_ends_are_kept():
    text = ("# Title\n\nI set it.\n\nThe film plays.\n\n## Next\n\nI kept it.\n")
    want = reproduce.review_page("README.md", text)
    out = reproduce.review_page("README.md", text.replace("\n", "\r\n"))
    assert out == want.replace("\n", "\r\n")
    assert reproduce.review_page("README.md", out) == out


def test_film_paragraphs_give_way_to_one_line():
    text = ("# Title\n\nOpening.\n\nThe film plays here.\n\nTwo Films are linked.\n\n"
            "Closing.\n\n## Next\n\nA film is named below the heading.\n")
    out = reproduce.review_page("README.md", text)
    assert out == ("# Title\n\nOpening.\n\n" + reproduce.FILM_NOTE + "\n\nClosing.\n\n"
                   "## Next\n\nA film is named below the heading.\n")
    assert reproduce.review_page("README.md", out) == out


def test_a_film_paragraph_right_above_the_heading_keeps_the_heading():
    text = "# Title\n\nOpening.\n\nThe film plays.\n## Next\n"
    out = reproduce.review_page("README.md", text)
    assert out == "# Title\n\nOpening.\n\n" + reproduce.FILM_NOTE + "\n## Next\n"
    assert reproduce.review_page("README.md", out) == out


def test_the_title_tagline_goes_and_nothing_else_moves():
    page = ('<meta property="og:title" content="Proj · A Tag, Line">\n'
            '<meta name="twitter:title" content="Proj · A Tag, Line">\n'
            '<meta property="og:description" content="Proj · kept">\n'
            "<title>Proj · A \"Tag\", Line</title>\n<p>Proj · kept</p>\n")
    want = ('<meta property="og:title" content="Proj">\n'
            '<meta name="twitter:title" content="Proj">\n'
            '<meta property="og:description" content="Proj · kept">\n'
            "<title>Proj</title>\n<p>Proj · kept</p>\n")
    for rel in ("index.html", "index.template.html"):
        assert reproduce.review_page(rel, page) == want
        assert reproduce.review_page(rel, want) == want
    assert reproduce.review_page("viz/player/index.html", page) == page


def test_other_files_come_back_unchanged():
    text = "I set it. The film plays.\n<title>Proj · Tag</title>\n"
    for rel in ("docs/FINDINGS.md", "docs/README.md", "CITATION.cff", "scripts/x.py"):
        assert reproduce.review_page(rel, text) == text


def test_the_real_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    out = reproduce.review_page("README.md", text)
    assert reproduce.review_page("README.md", out) == out
    assert singular_left(out) == []
    intro = _intro_paragraphs(out)
    assert [p for p in intro if re.search(r"film", p, re.I)] == [reproduce.FILM_NOTE]
    # Everything from the first second-level heading on is the same text, voice aside.
    rest = out[re.search(r"^## ", out, re.M).start():]
    assert rest == reproduce._plural_voice(text[re.search(r"^## ", text, re.M).start():])
    # The audit's README pins sit outside what the rewrite removes.
    for needle in ("docs/CORRECTIONS.md", "10.6.1", "VariBAD", "§10.8", "§10.9", "§15"):
        assert needle in out, needle


def test_the_real_site_pages_keep_rendering_alike():
    """tests/test_build_index.py requires index.html to equal the render of its template, in
    the archive too, so both pages must lose the same title text."""
    template = reproduce.review_page("index.template.html",
                                     (ROOT / "index.template.html").read_text(encoding="utf-8"))
    page = reproduce.review_page("index.html", (ROOT / "index.html").read_text(encoding="utf-8"))
    assert build_index.render(template, build_index.all_metrics(ROOT)) == page
    titles = re.findall(r"<title>[^<]*</title>|<meta\s+(?:property|name)=\"(?:og|twitter):title\""
                        r"\s+content=\"[^\"]*\"", page)
    assert len(titles) == 3 and not any("·" in t for t in titles), titles
