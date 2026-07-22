"""Bundle the outreach-video player into ONE self-contained HTML file.

Inlines style.css, beats.json, the recorded scene, the creature atlas, and the
sprite sheet (as a data URI) into a single page that plays with no network calls,
so it can be previewed in any browser (or published as an artifact). The four
runtime loaders in player.js are redirected to the inlined data; autoplay is
forced on. Google Fonts are dropped (system-font fallback) since a sandboxed
preview cannot reach external hosts; the final MP4 render still uses index.html
with the real fonts.

Usage:
    python viz/player/bundle_preview.py --out <path.html>
"""

from __future__ import annotations

import argparse
import base64
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# The #stage DOM from index.html (kept in sync by hand; it rarely changes).
STAGE = """
<div id="stage">
  <div id="world-card" class="layer">
    <canvas id="world" width="960" height="960"></canvas>
    <div class="pool-chip" id="chip-a">REAL WORLD<span class="chip-sub">true physics</span></div>
    <div class="pool-chip" id="chip-b">FLAWED COPY<span class="chip-sub">imitation physics</span></div>
  </div>
  <div id="gauge" class="layer">
    <div id="gauge-label"></div>
    <div id="gauge-row">
      <div id="gauge-track"><div id="gauge-fill"></div><div id="gauge-tick"></div></div>
      <div id="gauge-value">0.50</div>
    </div>
    <div id="gauge-scale"><span>0.50 = coin flip</span><span>1.00 = always right</span></div>
  </div>
  <div id="caption" class="layer">
    <div id="kicker"></div><div id="headline"></div><div id="subline"></div>
  </div>
  <div id="endcard" class="layer">
    <div class="glyph blob-glyph"></div>
    <div id="end-headline"></div><div id="end-url"></div><div id="end-foot"></div>
  </div>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    css = (HERE / "style.css").read_text(encoding="utf-8")
    js = (HERE / "player.js").read_text(encoding="utf-8")
    beats = (HERE / "beats.json").read_text(encoding="utf-8")
    frames = (HERE / "blob-frames.json").read_text(encoding="utf-8")
    scene = (DATA / "scene.json").read_text(encoding="utf-8")
    sprite = "data:image/png;base64," + base64.b64encode(
        (HERE / "blob-spritesheet.png").read_bytes()).decode()

    # Redirect the four runtime loaders to the inlined globals; force autoplay.
    repl = [
        ('const beats = await loadJSON("beats.json");', "const beats = window.__BEATS;"),
        ('scene = await loadJSON("../data/scene.json");', "scene = window.__SCENE;"),
        ('const meta = await loadJSON("blob-frames.json");', "const meta = window.__CREATURE;"),
        ("img.src = meta.image;", "img.src = window.__SPRITE;"),
        ('if (params.get("play") === "1") {', "if (true) {"),
    ]
    for a, b in repl:
        if a not in js:
            raise SystemExit(f"bundler: expected loader string not found: {a!r}")
        js = js.replace(a, b)

    # Guard against a stray </script> inside inlined JSON breaking the tag.
    for name, blob in (("scene", scene), ("beats", beats), ("frames", frames)):
        if "</script" in blob.lower():
            raise SystemExit(f"bundler: {name} contains a literal </script>; escape needed")

    html = (
        "<title>ItaSoRL - detectable is not learned (preview)</title>\n"
        "<style>\n" + css + "\n"
        "html,body{margin:0;background:#e9e6f2;}\n"
        "</style>\n"
        + STAGE +
        "<script>\n"
        f"window.__BEATS={beats};\n"
        f"window.__CREATURE={frames};\n"
        f"window.__SPRITE={sprite!r};\n"
        f"window.__SCENE={scene};\n"
        "</script>\n"
        "<script>\n" + js + "\n</script>\n"
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"wrote {out}  ({kb:.0f} KB, self-contained, autoplay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
