"""The social-preview card (docs/social-card) says only what the README's opening says.

The card is pixels once rendered, so the scalar audit cannot read it. Its source page can be
read: every percent drawn on it must appear in the README's opening paragraphs, the README
must show the committed card, and the anonymized supplement must leave it out, since it
prints the repository address.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs" / "social-card" / "index.html"
README = ROOT / "README.md"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _visible_text(html: str) -> str:
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    html = re.sub(r"<(style|head)\b.*?</\1>", " ", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _readme_opening() -> str:
    text = README.read_text(encoding="utf-8")
    return text[: text.index("docs/CORRECTIONS.md")]


def test_every_percent_on_the_card_is_in_the_readme_opening():
    shown = set(re.findall(r"\d+%", _visible_text(CARD.read_text(encoding="utf-8"))))
    assert shown, "the card shows no numbers"
    opening = _readme_opening().replace("**", "")
    missing = {p for p in shown if p not in opening}
    assert not missing, missing


def test_the_card_does_not_credit_survival_alone():
    text = _visible_text(CARD.read_text(encoding="utf-8")).lower()
    for banned in ("survival made it matter", "costs it dinner", "life depended",
                   "survival-specific", "because it had to survive"):
        assert banned not in text, banned
    assert "does not meet the registered rule" in text


def test_readme_shows_the_committed_card():
    head = README.read_text(encoding="utf-8")[:600]
    assert "docs/social-card/social-preview.png" in head
    assert (ROOT / "docs" / "social-card" / "social-preview.png").exists()


def test_the_supplement_leaves_the_card_out():
    import reproduce
    assert reproduce.excluded("docs/social-card/social-preview.png")
    assert reproduce.excluded("docs/social-card/index.html")
