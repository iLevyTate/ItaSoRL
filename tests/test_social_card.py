"""The social-preview card (docs/social-card) says only what the README's opening says.

The card is pixels once rendered, so the scalar audit cannot read it. Its source page can be
read: every percent drawn on it must appear in the README's opening paragraphs, the README
must show the committed card, and the anonymized supplement must leave it out, since it
prints the repository address. Outside a git checkout (the extracted supplement, which leaves
the card out) the tests that read the card skip; in a checkout a missing card still fails.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs" / "social-card" / "index.html"
README = ROOT / "README.md"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _in_git_checkout() -> bool:
    """True when ROOT is the top of a git checkout (an extracted archive placed inside some
    other repository is still an extracted archive)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and Path(top).resolve() == ROOT


def _card_file(path: Path) -> Path:
    if not path.exists() and not _in_git_checkout():
        pytest.skip(f"{path.relative_to(ROOT).as_posix()} is not shipped in the anonymized "
                    "supplement")
    return path


def _visible_text(html: str) -> str:
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    html = re.sub(r"<(style|head)\b.*?</\1>", " ", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _readme_opening() -> str:
    text = README.read_text(encoding="utf-8")
    return text[: text.index("docs/CORRECTIONS.md")]


def test_every_percent_on_the_card_is_in_the_readme_opening():
    shown = set(re.findall(r"\d+%", _visible_text(_card_file(CARD).read_text(encoding="utf-8"))))
    assert shown, "the card shows no numbers"
    opening = _readme_opening().replace("**", "")
    missing = {p for p in shown if p not in opening}
    assert not missing, missing


def test_the_own_state_tile_matches_the_readme_lead():
    """The every-percent test above lets a stale tile through whenever the README still
    quotes the old number somewhere in its opening (it quotes 53% as the first scoring), so
    the tile labeled "own state" is pinned to the number the README leads with (H82)."""
    html = _card_file(CARD).read_text(encoding="utf-8")
    tiles = re.findall(r'<div class="num">(\d+%)</div><div class="lab">(.*?)</div>', html, flags=re.S)
    own = [num for num, lab in tiles if "own state" in lab]
    assert len(own) == 1, tiles
    lead = re.search(r"own state reads\s*\*\*(\d+%)\*\*", _readme_opening())
    assert lead, "the README opening no longer leads with the own-state number"
    assert own[0] == lead.group(1)


def test_the_card_does_not_credit_survival_alone():
    text = _visible_text(_card_file(CARD).read_text(encoding="utf-8")).lower()
    for banned in ("survival made it matter", "costs it dinner", "life depended",
                   "survival-specific", "because it had to survive"):
        assert banned not in text, banned
    assert "does not meet the registered rule" in text


def test_readme_shows_the_committed_card():
    png = _card_file(ROOT / "docs" / "social-card" / "social-preview.png")
    head = README.read_text(encoding="utf-8")[:600]
    assert "docs/social-card/social-preview.png" in head
    assert png.exists()


def test_the_supplement_leaves_the_card_out():
    import reproduce
    assert reproduce.excluded("docs/social-card/social-preview.png")
    assert reproduce.excluded("docs/social-card/index.html")
