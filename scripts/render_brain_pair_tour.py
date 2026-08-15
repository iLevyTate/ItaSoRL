"""
Cinematic brain-pair tour: show the graphs in action with film pacing.

Loads training-step frames from viz/out/mind_brain_pair.gif, then builds a
short film with:
  - cold-open title
  - establishing full-brain playthrough
  - chapter cards for each network area
  - animated play-throughs while zoomed (with slow push-in)
  - crossfades, vignette, letterbox
  - MEMORY as the emotional peak (slower, teal-ring emphasis)
  - closing card that matches the open for a clean loop

Usage (from repo root):
  python scripts/render_brain_pair_tour.py
  python scripts/render_brain_pair_tour.py --gif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG = (10, 9, 16)
INK = (236, 233, 245)
MUTED = (120, 114, 145)
TEAL = (53, 181, 162)
SOFT = (42, 37, 71)


@dataclass(frozen=True)
class Beat:
    name: str
    box: tuple[float, float, float, float]
    chapter: str
    line: str
    plays: int = 1
    end_hold_s: float = 1.0
    # Extra in-betweens while playing this beat (slower = more cinematic).
    steps_between: int | None = None
    push: float = 0.04  # slow push-in amount over the beat
    # Static beats hold a single settled frame (no snapshot morph, no push-in) -
    # used for MEMORY, whose rings jitter across snapshots. end_hold_s is the
    # total on-screen time; a gentle brightness breath keeps it alive.
    static: bool = False
    # Dual close-up boxes: the SAME stage cropped from the real brain (box_l) and
    # the fake brain (box_r) within ONE source row, shown side by side (the MEMORY
    # hero treatment generalized). When set, `box` is ignored. Both must stay
    # inside a single stacked row so the crop never catches the divider seam.
    box_l: tuple[float, float, float, float] | None = None
    box_r: tuple[float, float, float, float] | None = None
    labels: tuple[str, str] = ("REAL", "FAKE")


# The source mind_brain_pair.gif is a 1600x1164 STACKED two-row image:
#   top row  (y~0.00-0.49): "NO. It cannot tell"  (real | fake, no teal rings)
#   divider  (y~0.49)
#   bottom row (y~0.51-1.0): "YES. It knows this world is fake" (real | fake + rings)
# Every chapter zooms into the BOTTOM ("can tell") row so the crop never straddles
# the divider seam. YROW is the brain-panel band inside that row; the L/R x-bands
# below isolate one functional stage in the real brain and the fake brain so they
# can sit side by side (same stage, real vs fake - the film's whole thesis).
YROW = (0.605, 0.955)


def _row_box(x0: float, x1: float) -> tuple[float, float, float, float]:
    return (x0, YROW[0], x1, YROW[1])


BEATS: list[Beat] = [
    Beat(
        "overview",
        (0.02, 0.02, 0.98, 0.98),
        "THE WHOLE BRAIN",
        "One mind. Two worlds. Watch it learn.",
        plays=1,
        end_hold_s=1.1,
        push=0.02,
    ),
    Beat(
        "senses",
        (0.0, 0.0, 0.0, 0.0),  # unused: dual close-up below
        "SENSES",
        "What it sees and feels - inputs waking up.",
        plays=1,
        end_hold_s=0.9,
        push=0.05,
        box_l=_row_box(0.015, 0.235),
        box_r=_row_box(0.505, 0.725),
    ),
    Beat(
        "process",
        (0.0, 0.0, 0.0, 0.0),
        "PROCESS",
        "Signals mix. Patterns start to form.",
        plays=1,
        end_hold_s=0.9,
        push=0.05,
        box_l=_row_box(0.190, 0.365),
        box_r=_row_box(0.665, 0.840),
    ),
    # MEMORY: the emotional peak. A calm, steady hold on the settled "can tell"
    # memory columns of both brains, teal rings as the hero - now a TIGHT dual
    # close-up so the rings read as large as possible. Static (settled frame only)
    # so the rings never jitter across snapshots.
    Beat(
        "memory",
        (0.0, 0.0, 0.0, 0.0),
        "MEMORY",
        "Teal rings = the cells that hold the clue.",
        end_hold_s=6.5,
        push=0.0,
        static=True,
        box_l=_row_box(0.255, 0.375),
        box_r=_row_box(0.735, 0.855),
    ),
    Beat(
        "actions",
        (0.0, 0.0, 0.0, 0.0),
        "ACTIONS + GUESSES",
        "What it does next - and what it expects.",
        plays=1,
        end_hold_s=1.0,
        push=0.05,
        box_l=_row_box(0.330, 0.475),
        box_r=_row_box(0.815, 0.960),
    ),
]

# Source single-row brain GIFs for the teal-ring reveal (real | fake side by side).
CAN_TELL_GIF = ROOT / "viz/out/mind_brain_can_tell.gif"
CANT_TELL_GIF = ROOT / "viz/out/mind_brain_cant_tell.gif"
# Tight MEMORY+ACTIONS crops on each half of the 1600x580 can/cant frames.
# (Wide whole-brain crops bury the teal rings; dual close-ups make them readable.)
MEM_L = (0.275, 0.18, 0.485, 0.95)
MEM_R = (0.745, 0.18, 0.945, 0.95)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf")
        if bold
        else ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf")
    )
    for name in candidates + ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - ((-2 * t + 2) ** 3) / 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    u = _ease_in_out_cubic(t)
    return tuple(_lerp(a[i], b[i], u) for i in range(4))  # type: ignore[return-value]


def _shrink_box(
    box: tuple[float, float, float, float], amount: float
) -> tuple[float, float, float, float]:
    """Push-in: shrink toward center by `amount` (0..0.2)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    s = 1.0 - amount
    return (cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s)


def _load_sequence(path: Path) -> list[Image.Image]:
    im = Image.open(path)
    frames: list[Image.Image] = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        frames.append(im.convert("RGB"))
    if len(frames) <= 1:
        return frames
    means = [float(np.asarray(fr.resize((32, 32))).mean()) for fr in frames]
    content = [fr for fr, m in zip(frames, means) if m > 18.0]
    return content if len(content) >= 2 else frames


def _crop_box(
    size: tuple[int, int], box: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = box
    return (
        max(0, int(round(x0 * w))),
        max(0, int(round(y0 * h))),
        min(w, int(round(x1 * w))),
        min(h, int(round(y1 * h))),
    )


def _blend(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, max(0.0, min(1.0, t)))


def _interp_sequence(seq: list[Image.Image], steps_between: int) -> list[Image.Image]:
    if steps_between <= 0 or len(seq) < 2:
        return list(seq)
    out: list[Image.Image] = []
    for i in range(len(seq) - 1):
        out.append(seq[i])
        for k in range(1, steps_between + 1):
            out.append(_blend(seq[i], seq[i + 1], _ease(k / (steps_between + 1))))
    out.append(seq[-1])
    return out


def _vignette(size: tuple[int, int], strength: float = 0.55) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rx, ry = w * 0.62, h * 0.62
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    alpha = np.clip((d - 0.35) / 1.1, 0, 1) ** 1.35
    alpha = (alpha * strength * 255).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


def _grade(im: Image.Image) -> Image.Image:
    """Slight cinematic grade: crush blacks, lift teal, soft contrast."""
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Color(im).enhance(1.12)
    im = ImageEnhance.Brightness(im).enhance(0.96)
    return im


def _letterbox(canvas: Image.Image, bar: int) -> None:
    if bar <= 0:
        return
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.rectangle([0, 0, w, bar], fill=(0, 0, 0))
    draw.rectangle([0, h - bar, w, h], fill=(0, 0, 0))


def _title_card(
    out_size: tuple[int, int],
    title: str,
    sub: str,
    *,
    hold_s: float,
    fps: int,
) -> list[Image.Image]:
    ow, oh = out_size
    font_t = _font(48 if ow >= 1280 else 34, bold=True)
    font_s = _font(22 if ow >= 1280 else 16)
    font_f = _font(16)
    base = Image.new("RGB", out_size, BG)
    draw = ImageDraw.Draw(base)
    draw.text((ow // 2, oh * 0.42), title, fill=INK, font=font_t, anchor="mm")
    draw.rectangle(
        [ow // 2 - 40, oh * 0.48, ow // 2 + 40, oh * 0.48 + 3], fill=TEAL
    )
    draw.text((ow // 2, oh * 0.54), sub, fill=MUTED, font=font_s, anchor="mm")
    draw.text(
        (ow // 2, oh * 0.88),
        "ItaSoRL",
        fill=SOFT,
        font=font_f,
        anchor="mm",
    )
    vig = _vignette(out_size, 0.7)
    black = Image.new("RGB", out_size, (0, 0, 0))
    base = Image.composite(black, base, vig)
    _letterbox(base, 52 if oh > 900 else 36)

    n = max(1, int(round(hold_s * fps)))
    # Fade in / out
    frames: list[Image.Image] = []
    fade = max(1, int(0.35 * fps))
    for i in range(n):
        fr = base.copy()
        if i < fade:
            fr = _blend(Image.new("RGB", out_size, BG), fr, (i + 1) / fade)
        elif i >= n - fade:
            fr = _blend(fr, Image.new("RGB", out_size, BG), (i - (n - fade) + 1) / fade)
        frames.append(fr)
    return frames


def _chapter_card(
    out_size: tuple[int, int],
    chapter: str,
    line: str,
    *,
    hold_s: float,
    fps: int,
) -> list[Image.Image]:
    ow, oh = out_size
    font_c = _font(18, bold=True)
    font_t = _font(40 if ow >= 1280 else 28, bold=True)
    font_s = _font(20 if ow >= 1280 else 15)
    base = Image.new("RGB", out_size, BG)
    draw = ImageDraw.Draw(base)
    draw.text((ow // 2, oh * 0.38), chapter, fill=TEAL, font=font_c, anchor="mm")
    draw.text((ow // 2, oh * 0.48), line, fill=INK, font=font_t, anchor="mm")
    draw.text(
        (ow // 2, oh * 0.58),
        "in action",
        fill=MUTED,
        font=font_s,
        anchor="mm",
    )
    vig = _vignette(out_size, 0.65)
    base = Image.composite(Image.new("RGB", out_size, (0, 0, 0)), base, vig)
    _letterbox(base, 52 if oh > 900 else 36)
    n = max(1, int(round(hold_s * fps)))
    fade = max(1, int(0.25 * fps))
    frames: list[Image.Image] = []
    for i in range(n):
        fr = base.copy()
        if i < fade:
            fr = _blend(Image.new("RGB", out_size, BG), fr, (i + 1) / fade)
        elif i >= n - fade:
            fr = _blend(fr, Image.new("RGB", out_size, BG), (i - (n - fade) + 1) / fade)
        frames.append(fr)
    return frames


def _compose_frame(
    src: Image.Image,
    box: tuple[float, float, float, float],
    out_size: tuple[int, int],
    chapter: str,
    line: str,
    *,
    progress: float,
    font_chapter: ImageFont.ImageFont,
    font_line: ImageFont.ImageFont,
    font_foot: ImageFont.ImageFont,
    vig: Image.Image,
) -> Image.Image:
    ow, oh = out_size
    canvas = Image.new("RGB", out_size, BG)
    crop = src.crop(_crop_box(src.size, box))
    # Soften slightly for film feel
    crop = crop.filter(ImageFilter.SMOOTH_MORE)

    bar = 56 if oh > 900 else 40
    caption_h = 130
    margin = 48
    avail_w = ow - 2 * margin
    avail_h = oh - caption_h - 2 * bar - 20
    scale = min(avail_w / max(1, crop.width), avail_h / max(1, crop.height))
    nw = max(1, int(round(crop.width * scale)))
    nh = max(1, int(round(crop.height * scale)))
    # Keep even dims
    nw -= nw % 2
    nh -= nh % 2
    fitted = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    fitted = _grade(fitted)
    x = (ow - nw) // 2
    y = bar + 10 + (avail_h - nh) // 2
    canvas.paste(fitted, (x, y))

    draw = ImageDraw.Draw(canvas)
    # Progress hairline under the plate
    px0, px1 = x, x + nw
    py = y + nh + 10
    draw.line([(px0, py), (px1, py)], fill=(60, 55, 80), width=2)
    draw.line(
        [(px0, py), (px0 + int((px1 - px0) * progress), py)],
        fill=TEAL,
        width=3,
    )

    # Caption band
    band_top = oh - caption_h - bar + 8
    for i in range(caption_h):
        a = i / max(1, caption_h - 1)
        shade = tuple(int(BG[c] * (0.4 + 0.6 * a)) for c in range(3))
        draw.line([(0, band_top + i), (ow, band_top + i)], fill=shade)

    draw.text((ow // 2, band_top + 28), chapter, fill=TEAL, font=font_chapter, anchor="mm")
    draw.text((ow // 2, band_top + 64), line, fill=INK, font=font_line, anchor="mm")
    draw.text(
        (ow // 2, band_top + 98),
        "ItaSoRL  ·  cinematic brain pair",
        fill=MUTED,
        font=font_foot,
        anchor="mm",
    )

    # Vignette + letterbox
    canvas = Image.composite(Image.new("RGB", out_size, (0, 0, 0)), canvas, vig)
    _letterbox(canvas, bar)
    return canvas


def _crossfade(
    a: list[Image.Image], b: list[Image.Image], n: int
) -> list[Image.Image]:
    if not a:
        return list(b)
    if not b or n <= 0:
        return a + b
    out = list(a)
    left, right = a[-1], b[0]
    for i in range(1, n + 1):
        out.append(_blend(left, right, _ease(i / (n + 1))))
    out.extend(b[1:])
    return out


def _find_teal_centers(im: Image.Image, max_centers: int = 10) -> list[tuple[float, float]]:
    """Locate teal ring centers on MEMORY columns (image pixel space)."""
    arr = np.asarray(im).astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Bright teal ring ink (exclude lavender nodes / yellow guesses).
    mask = (g > 155) & (b > 130) & (r < 110) & (g > r + 45) & (np.abs(g - b) < 45)
    if not mask.any():
        return []
    h, w = mask.shape
    band = np.zeros_like(mask)
    # Narrow MEMORY node columns only (skip ACTIONS "eat"/"move" halos).
    band[:, int(0.290 * w) : int(0.345 * w)] = True
    band[:, int(0.760 * w) : int(0.815 * w)] = True
    band[: int(0.22 * h), :] = False
    band[int(0.88 * h) :, :] = False
    mask = mask & band
    if not mask.any():
        return []

    cell = 18
    centers: list[tuple[float, float, int]] = []
    for y0 in range(0, h, cell):
        for x0 in range(0, w, cell):
            patch = mask[y0 : y0 + cell, x0 : x0 + cell]
            n = int(patch.sum())
            if n < 10:
                continue
            yy, xx = np.where(patch)
            centers.append((x0 + float(xx.mean()), y0 + float(yy.mean()), n))
    centers.sort(key=lambda t: -t[2])

    picked: list[tuple[float, float]] = []
    for x, y, _ in centers:
        # Rings are ~30px apart vertically; keep one center per ring.
        if all((x - px) ** 2 + (y - py) ** 2 > 30 ** 2 for px, py in picked):
            picked.append((x, y))
        if len(picked) >= max_centers:
            break
    return picked


def _memory_dual_plate(
    src: Image.Image,
    *,
    zoom: float = 0.0,
    labels: tuple[str, str] = ("REAL", "FAKE"),
) -> Image.Image:
    """Side-by-side MEMORY close-ups so teal rings read at film scale."""
    w, h = src.size
    z = max(0.0, min(0.12, zoom))

    def _zoomed(box: tuple[float, float, float, float]) -> Image.Image:
        x0, y0, x1, y1 = _shrink_box(box, z)
        return src.crop(_crop_box((w, h), (x0, y0, x1, y1)))

    left = _zoomed(MEM_L)
    right = _zoomed(MEM_R)
    # Match heights; keep width proportional.
    th = max(left.height, right.height)
    def _fit_h(im: Image.Image) -> Image.Image:
        scale = th / max(1, im.height)
        nw = max(1, int(round(im.width * scale)))
        return im.resize((nw, th), Image.Resampling.LANCZOS)

    left, right = _fit_h(left), _fit_h(right)
    gap = 28
    pad_top = 36
    pad_side = 18
    plate_w = left.width + right.width + gap + 2 * pad_side
    plate_h = th + pad_top + 12
    plate = Image.new("RGB", (plate_w, plate_h), BG)
    plate.paste(left, (pad_side, pad_top))
    plate.paste(right, (pad_side + left.width + gap, pad_top))

    draw = ImageDraw.Draw(plate)
    font = _font(15, bold=True)
    draw.text(
        (pad_side + left.width // 2, 16),
        labels[0],
        fill=TEAL,
        font=font,
        anchor="mm",
    )
    draw.text(
        (pad_side + left.width + gap + right.width // 2, 16),
        labels[1],
        fill=TEAL,
        font=font,
        anchor="mm",
    )
    # Thin divider
    mx = pad_side + left.width + gap // 2
    draw.line([(mx, pad_top + 8), (mx, pad_top + th - 8)], fill=SOFT, width=2)
    return plate


def _dual_plate(
    src: Image.Image,
    box_l: tuple[float, float, float, float],
    box_r: tuple[float, float, float, float],
    *,
    labels: tuple[str, str] = ("REAL", "FAKE"),
) -> Image.Image:
    """Side-by-side close-ups of the SAME stage in the real vs fake brain.

    Generalizes _memory_dual_plate to arbitrary crop boxes so every chapter can
    use the readable dual-close-up look, not just MEMORY. Push-in is applied
    afterward via _zoom_plate_smooth (constant output size = judder-free glide).
    """
    w, h = src.size
    left = src.crop(_crop_box((w, h), box_l))
    right = src.crop(_crop_box((w, h), box_r))
    th = max(left.height, right.height)

    def _fit_h(im: Image.Image) -> Image.Image:
        scale = th / max(1, im.height)
        nw = max(1, int(round(im.width * scale)))
        return im.resize((nw, th), Image.Resampling.LANCZOS)

    left, right = _fit_h(left), _fit_h(right)
    gap = 28
    pad_top = 36
    pad_side = 18
    plate_w = left.width + right.width + gap + 2 * pad_side
    plate_h = th + pad_top + 12
    plate = Image.new("RGB", (plate_w, plate_h), BG)
    plate.paste(left, (pad_side, pad_top))
    plate.paste(right, (pad_side + left.width + gap, pad_top))

    draw = ImageDraw.Draw(plate)
    font = _font(15, bold=True)
    draw.text(
        (pad_side + left.width // 2, 16), labels[0], fill=TEAL, font=font, anchor="mm"
    )
    draw.text(
        (pad_side + left.width + gap + right.width // 2, 16),
        labels[1],
        fill=TEAL,
        font=font,
        anchor="mm",
    )
    mx = pad_side + left.width + gap // 2
    draw.line([(mx, pad_top + 8), (mx, pad_top + th - 8)], fill=SOFT, width=2)
    return plate


def _zoom_plate_smooth(base: Image.Image, z: float) -> Image.Image:
    """Continuous center push-in on a fixed-size plate.

    Rebuilding the plate at a new crop each frame quantized its size/position
    (integer crop + even-snap rounding), so over the tiny zoom range the camera
    juddered and even stepped backward. Instead we build the plate ONCE at
    zoom=0 (constant dimensions) and zoom via a sub-pixel EXTENT transform that
    keeps the output size constant - so the downstream fit never moves and the
    push-in is perfectly smooth.
    """
    z = max(0.0, min(0.12, z))
    if z <= 1e-6:
        return base
    w, h = base.size
    cw, ch = w * (1.0 - z), h * (1.0 - z)
    x0, y0 = (w - cw) / 2.0, (h - ch) / 2.0
    return base.transform(
        (w, h),
        Image.Transform.EXTENT,
        (x0, y0, x0 + cw, y0 + ch),
        resample=Image.Resampling.BICUBIC,
    )


def _fade_in(frames: list[Image.Image], n: int) -> list[Image.Image]:
    """Ease the first n frames up from the background so a chapter card that
    faded to black flows into the plate instead of hard-cutting (less 'jumpy')."""
    for i in range(min(n, len(frames))):
        bg = Image.new("RGB", frames[i].size, BG)
        frames[i] = _blend(bg, frames[i], _ease((i + 1) / (n + 1)))
    return frames


def _map_centers_to_dual(
    centers: list[tuple[float, float]],
    src_size: tuple[int, int],
    plate: Image.Image,
    zoom: float = 0.0,
) -> list[tuple[float, float]]:
    """Map full-frame teal centers into dual-plate pixel coords."""
    sw, sh = src_size
    z = max(0.0, min(0.12, zoom))
    # Must match _memory_dual_plate layout.
    left_box = _shrink_box(MEM_L, z)
    right_box = _shrink_box(MEM_R, z)
    lx0, ly0, lx1, ly1 = _crop_box((sw, sh), left_box)
    rx0, ry0, rx1, ry1 = _crop_box((sw, sh), right_box)
    lw, lh = max(1, lx1 - lx0), max(1, ly1 - ly0)
    rw, rh = max(1, rx1 - rx0), max(1, ry1 - ry0)
    th = max(lh, rh)
    left_disp_w = max(1, int(round(lw * (th / lh))))
    right_disp_w = max(1, int(round(rw * (th / rh))))
    gap, pad_top, pad_side = 28, 36, 18

    mapped: list[tuple[float, float]] = []
    for x, y in centers:
        if lx0 <= x < lx1 and ly0 <= y < ly1:
            sx = (x - lx0) / lw
            sy = (y - ly0) / lh
            mapped.append((pad_side + sx * left_disp_w, pad_top + sy * th))
        elif rx0 <= x < rx1 and ry0 <= y < ry1:
            sx = (x - rx0) / rw
            sy = (y - ry0) / rh
            mapped.append(
                (
                    pad_side + left_disp_w + gap + sx * right_disp_w,
                    pad_top + sy * th,
                )
            )
    return mapped


def _spotlight_rings(
    plate: Image.Image,
    centers: list[tuple[float, float]],
    *,
    active_upto: int,
    phase: float,
    dim: float = 0.42,
) -> Image.Image:
    """Dim the plate, then pulse a soft halo on memory cells 0..active_upto-1."""
    if not centers:
        return plate
    base = plate.convert("RGBA")
    veil = Image.new("RGBA", base.size, (0, 0, 0, int(200 * dim)))
    dimmed = Image.alpha_composite(base, veil)

    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    left = sorted([p for p in centers if p[0] < plate.width * 0.5], key=lambda p: p[1])
    right = sorted([p for p in centers if p[0] >= plate.width * 0.5], key=lambda p: p[1])
    ordered = left + right

    n_active = max(0, min(len(ordered), active_upto))
    for i, (x, y) in enumerate(ordered[:n_active]):
        # Newest cell pops; earlier ones keep a quiet settled halo.
        is_new = i == n_active - 1
        birth = (0.55 + 0.45 * abs(np.sin(phase * np.pi))) if is_new else 0.75
        pulse = 0.6 + 0.4 * np.sin(phase * 2 * np.pi + i * 0.4)
        radius = 12 + (8 * birth if is_new else 4) + 2 * pulse
        draw.ellipse(
            [x - radius * 1.6, y - radius * 1.6, x + radius * 1.6, y + radius * 1.6],
            fill=(*TEAL, int(28 * birth)),
        )
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(*TEAL, int(170 * birth)),
            width=2,
        )

    cut = Image.new("L", base.size, 0)
    cut_d = ImageDraw.Draw(cut)
    for x, y in ordered[:n_active]:
        cut_d.ellipse([x - 26, y - 26, x + 26, y + 26], fill=255)
    cut = cut.filter(ImageFilter.GaussianBlur(5))
    revealed = Image.composite(base, dimmed, cut)
    out = Image.alpha_composite(revealed, glow)
    return out.convert("RGB")


def _compose_plate(
    plate: Image.Image,
    out_size: tuple[int, int],
    chapter: str,
    line: str,
    *,
    progress: float,
    font_chapter: ImageFont.ImageFont,
    font_line: ImageFont.ImageFont,
    font_foot: ImageFont.ImageFont,
    vig: Image.Image,
) -> Image.Image:
    """Fit a pre-built plate into the cinematic caption frame."""
    ow, oh = out_size
    canvas = Image.new("RGB", out_size, BG)
    crop = plate.filter(ImageFilter.SMOOTH_MORE)

    bar = 56 if oh > 900 else 40
    caption_h = 130
    margin = 40
    avail_w = ow - 2 * margin
    avail_h = oh - caption_h - 2 * bar - 20
    scale = min(avail_w / max(1, crop.width), avail_h / max(1, crop.height))
    nw = max(1, int(round(crop.width * scale)))
    nh = max(1, int(round(crop.height * scale)))
    nw -= nw % 2
    nh -= nh % 2
    fitted = _grade(crop.resize((nw, nh), Image.Resampling.LANCZOS))
    x = (ow - nw) // 2
    y = bar + 10 + (avail_h - nh) // 2
    canvas.paste(fitted, (x, y))

    draw = ImageDraw.Draw(canvas)
    px0, px1 = x, x + nw
    py = y + nh + 10
    draw.line([(px0, py), (px1, py)], fill=(60, 55, 80), width=2)
    draw.line(
        [(px0, py), (px0 + int((px1 - px0) * progress), py)],
        fill=TEAL,
        width=3,
    )

    band_top = oh - caption_h - bar + 8
    for i in range(caption_h):
        a = i / max(1, caption_h - 1)
        shade = tuple(int(BG[c] * (0.4 + 0.6 * a)) for c in range(3))
        draw.line([(0, band_top + i), (ow, band_top + i)], fill=shade)

    draw.text((ow // 2, band_top + 28), chapter, fill=TEAL, font=font_chapter, anchor="mm")
    draw.text((ow // 2, band_top + 64), line, fill=INK, font=font_line, anchor="mm")
    draw.text(
        (ow // 2, band_top + 98),
        "ItaSoRL  ·  cinematic brain pair",
        fill=MUTED,
        font=font_foot,
        anchor="mm",
    )
    canvas = Image.composite(Image.new("RGB", out_size, (0, 0, 0)), canvas, vig)
    _letterbox(canvas, bar)
    return canvas


def _teal_ring_reveal(
    out_size: tuple[int, int],
    fps: int,
    vig: Image.Image,
    fonts: tuple,
) -> list[Image.Image]:
    """
    MEMORY peak beat, readable at film scale:
      1) dual MEMORY close-up with no rings (cannot tell)
      2) survival turn card
      3) rings appear from real training frames
      4) spotlight cascade cell-by-cell, then breathe
    """
    font_chapter, font_line, font_foot = fonts
    if not CAN_TELL_GIF.exists() or not CANT_TELL_GIF.exists():
        print("WARN: can/cant tell GIFs missing; skipping teal reveal", flush=True)
        return

    cant = _load_sequence(CANT_TELL_GIF)
    can = _load_sequence(CAN_TELL_GIF)
    if len(cant) < 2 or len(can) < 2:
        return

    # Frame counts preserve the original beat timing. We deliberately no longer
    # morph *through* the raw training snapshots: each snapshot rings/edges
    # different memory cells, so blending them made the rings slide up and down
    # (the "forward and backward" jitter). Instead we push in on, and dissolve
    # between, the two settled frames only.
    n1 = len(cant) + (len(cant) - 1) * 4
    n2 = len(can) + (len(can) - 1) * 7
    centers = _find_teal_centers(can[-1])
    print(f"Teal-ring reveal: {len(centers)} memory cells tracked", flush=True)

    def plate_frame(
        src: Image.Image,
        chapter: str,
        line: str,
        *,
        progress: float,
        zoom: float = 0.0,
        centers_overlay: list[tuple[float, float]] | None = None,
        active_upto: int = 0,
        phase: float = 0.0,
        dim: float = 0.0,
    ) -> Image.Image:
        if centers_overlay is None:
            # Acts 2-3 (moving zoom, no ring overlay): build the plate once at
            # zoom=0 and push in with a sub-pixel transform, so the camera glides
            # instead of stair-stepping. Spotlight frames keep the crop-based
            # zoom below because their zoom is constant (no stepping) and the
            # ring centers must map into that same crop geometry.
            base = _memory_dual_plate(src, zoom=0.0)
            plate = _zoom_plate_smooth(base, zoom)
        else:
            plate = _memory_dual_plate(src, zoom=zoom)
            if active_upto > 0:
                mapped = _map_centers_to_dual(
                    centers_overlay, src.size, plate, zoom=zoom
                )
                plate = _spotlight_rings(
                    plate, mapped, active_upto=active_upto, phase=phase, dim=dim,
                )
        return _compose_plate(
            plate, out_size, chapter, line,
            progress=progress,
            font_chapter=font_chapter,
            font_line=font_line,
            font_foot=font_foot,
            vig=vig,
        )

    # --- Act 1: empty memory ---
    yield from _chapter_card(
        out_size,
        "MEMORY",
        "Zoom in. Look for teal rings.",
        hold_s=1.5,
        fps=fps,
    )
    fade_n = max(2, int(0.3 * fps))
    act1 = [
        plate_frame(
            cant[-1],
            "MEMORY",
            "No teal rings yet - memory has no world clue.",
            progress=0.08 + 0.22 * (i / max(1, n1 - 1)),
            zoom=0.02 * _ease(i / max(1, n1 - 1)),
        )
        for i in range(n1)
    ]
    act1 += [
        plate_frame(
            cant[-1],
            "MEMORY",
            "No teal rings yet - memory has no world clue.",
            progress=0.30,
            zoom=0.02,
        )
        for _ in range(int(1.0 * fps))
    ]
    yield from _fade_in(act1, fade_n)

    # --- Act 2: the turn ---
    yield from _chapter_card(
        out_size,
        "THEN SURVIVAL",
        "Make the fake cost dinner. Memory starts to care.",
        hold_s=1.7,
        fps=fps,
    )

    # --- Act 3: rings arrive. Show the SETTLED "can tell" graph only, faded up
    # from the card's black-out, then push in. We deliberately do NOT dissolve
    # cant[-1] -> can[-1]: their edges differ, so a cross-dissolve double-exposed
    # both edge sets (the "ghosting"). One fixed graph = no ghost, no jitter; the
    # rings "form" via the fade-in out of the survival card.
    act3 = [
        plate_frame(
            can[-1],
            "MEMORY",
            "Teal rings forming - these cells hold the clue.",
            progress=0.30 + 0.40 * (i / max(1, n2 - 1)),
            zoom=0.02 + 0.04 * _ease(i / max(1, n2 - 1)),
        )
        for i in range(n2)
    ]
    yield from _fade_in(act3, fade_n)

    # --- Act 4: spotlight each ringed cell, then breathe ---
    n_cells = max(1, len(centers))
    # ~0.28s per cell, then a full-group breath hold.
    per_cell = max(4, int(0.28 * fps))
    for i in range(n_cells * per_cell):
        t = i / max(1, n_cells * per_cell - 1)
        active = 1 + i // per_cell
        phase = (i % per_cell) / per_cell
        yield plate_frame(
            can[-1],
            "MEMORY",
            "Teal rings = memory cells that hold the clue.",
            progress=0.70 + 0.20 * t,
            zoom=0.06,
            centers_overlay=centers,
            active_upto=active,
            phase=phase,
            dim=0.38,
        )

    n_hold = int(2.4 * fps)
    for i in range(n_hold):
        t = i / max(1, n_hold - 1)
        yield plate_frame(
            can[-1],
            "MEMORY",
            "Teal rings = memory cells that hold the clue.",
            progress=1.0,
            # Constant zoom (the breath is carried by the dim pulse below); a
            # varying crop-based zoom here would stair-step.
            zoom=0.06,
            centers_overlay=centers,
            active_upto=n_cells,
            phase=t,
            dim=0.22 + 0.10 * (0.5 + 0.5 * np.sin(np.pi * t)),
        )


def _play_beat(
    seq: list[Image.Image],
    beat: Beat,
    out_size: tuple[int, int],
    fps: int,
    default_steps: int,
    vig: Image.Image,
    fonts: tuple,
):
    """Yield a beat's frames lazily so the film never holds them all at once."""
    font_chapter, font_line, font_foot = fonts
    dual = beat.box_l is not None and beat.box_r is not None

    def _frame(src: Image.Image, z: float, progress: float) -> Image.Image:
        """One composed cinematic frame for this beat, dual close-up or single."""
        if dual:
            # Build the dual plate at zoom 0, then push in with a sub-pixel
            # transform (constant output size) so the camera glides, matching the
            # MEMORY hero. box_l/box_r stay inside one row -> no seam artifacts.
            plate = _zoom_plate_smooth(
                _dual_plate(src, beat.box_l, beat.box_r, labels=beat.labels), z
            )
            return _compose_plate(
                plate, out_size, beat.chapter, beat.line,
                progress=progress,
                font_chapter=font_chapter,
                font_line=font_line,
                font_foot=font_foot,
                vig=vig,
            )
        box = _shrink_box(beat.box, z)
        return _compose_frame(
            src, box, out_size, beat.chapter, beat.line,
            progress=progress,
            font_chapter=font_chapter,
            font_line=font_line,
            font_foot=font_foot,
            vig=vig,
        )

    if beat.static:
        # Compose the settled frame ONCE (no per-frame recrop -> no jitter/step),
        # fade it up from the chapter card, hold steady with a faint brightness
        # breath so it is calm but not dead.
        base = _frame(seq[-1], beat.push, progress=1.0)
        bg = Image.new("RGB", out_size, BG)
        n = max(1, int(round(beat.end_hold_s * fps)))
        fade = max(2, int(0.4 * fps))
        for i in range(n):
            if i < fade:
                yield _blend(bg, base, _ease((i + 1) / fade))
            else:
                breath = 0.028 * np.sin(2 * np.pi * (i - fade) / max(1, n - fade))
                yield ImageEnhance.Brightness(base).enhance(1.0 + float(breath))
        return

    steps = beat.steps_between if beat.steps_between is not None else default_steps
    smooth = _interp_sequence(seq, steps_between=steps)
    total = max(1, len(smooth) * beat.plays)

    idx = 0
    for _ in range(beat.plays):
        for src in smooth:
            t = idx / max(1, total - 1)
            yield _frame(src, beat.push * _ease(t), progress=t)
            idx += 1

    # Settled hold with gentle breathing pulse. Dual plates hold at a fixed zoom
    # (the breath is carried by brightness) so the rings never stair-step; single
    # crops keep the original recrop-based breath.
    n_hold = max(1, int(round(beat.end_hold_s * fps)))
    last_src = seq[-1]
    held = _frame(last_src, beat.push, progress=1.0) if dual else None
    for i in range(n_hold):
        t = i / max(1, n_hold - 1)
        pulse = 0.04 * np.sin(np.pi * t)
        if dual:
            fr = held
        else:
            fr = _frame(
                last_src, beat.push * (1.0 + 0.15 * np.sin(np.pi * t)), progress=1.0
            )
        if abs(pulse) > 1e-3:
            fr = ImageEnhance.Brightness(fr).enhance(1.0 + float(pulse))
        yield fr


def _stream_crossfade(segments, n: int):
    """Concatenate frame-iterables, blending each boundary A[-1] -> B[0].

    Streams one frame at a time so the whole film is never resident in memory.
    Mirrors _crossfade: emits all of A, then n eased blends A[-1]->B[0], then
    B minus its first frame. Empty segments (e.g. a skipped teal reveal) are
    passed over without adding a bridge.
    """
    sentinel = object()
    prev_last = None
    for seg in segments:
        it = iter(seg)
        first = next(it, sentinel)
        if first is sentinel:
            continue
        if prev_last is None:
            yield first
        else:
            for i in range(1, n + 1):
                yield _blend(prev_last, first, _ease(i / (n + 1)))
            # 'first' is B[0]; it is replaced by the blend ramp, not emitted.
        last = first
        for fr in it:
            yield fr
            last = fr
        prev_last = last


def _iter_film(
    seq: list[Image.Image],
    out_size: tuple[int, int],
    fps: int,
    steps_between: int,
):
    """Yield the whole tour frame-by-frame (bounded memory, for streaming)."""
    vig = _vignette(out_size, 0.52)
    fonts = (
        _font(16 if out_size[0] < 1280 else 18, bold=True),
        _font(26 if out_size[0] < 1280 else 32, bold=True),
        _font(15),
    )
    xfade = max(1, int(0.4 * fps))

    def segments():
        yield _title_card(
            out_size,
            "Can a creature know\nit lives in a fake world?",
            "Watch the same brain in real vs fake - in action",
            hold_s=2.4,
            fps=fps,
        )
        for beat in BEATS:
            yield _chapter_card(
                out_size, beat.chapter, beat.line, hold_s=1.35, fps=fps,
            )
            yield _play_beat(
                seq, beat, out_size, fps, steps_between, vig, fonts,
            )
        yield _title_card(
            out_size,
            "Detectable is not noticed\nuntil survival makes it matter",
            "ItaSoRL  ·  ilevytate.github.io/ItaSoRL",
            hold_s=2.6,
            fps=fps,
        )

    first: Image.Image | None = None
    last: Image.Image | None = None
    for fr in _stream_crossfade(segments(), xfade):
        if first is None:
            first = fr
        last = fr
        yield fr

    if first is None:
        return
    # Loop seam: progressively fade the close back toward the open title,
    # ending exactly on frame 0 for a clean loop.
    seam_n = max(1, int(0.5 * fps))
    running = last
    for i in range(1, seam_n):
        running = _blend(running, first, _ease(i / (seam_n + 1)))
        yield running
    yield first.copy()


def _open_encoder(out: Path, fps: int) -> subprocess.Popen:
    """Start an ffmpeg process that reads PNG frames from stdin (near-lossless)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "14",
        "-preset", "slow",
        "-movflags", "+faststart",
        str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _feed(proc: subprocess.Popen, fr: Image.Image) -> None:
    w, h = fr.size
    if w % 2 or h % 2:  # yuv420p needs even dims
        fr = fr.crop((0, 0, w - w % 2, h - h % 2))
    fr.save(proc.stdin, format="PNG")


def _finish(proc: subprocess.Popen, out: Path) -> None:
    proc.stdin.close()
    _, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write(err.decode("utf-8", errors="replace"))
        raise RuntimeError(f"ffmpeg failed for {out}")


def _write_gif(frames: list[Image.Image], out: Path, fps: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = 640 / frames[0].width
    small = [
        fr.resize(
            (
                max(2, int(fr.width * scale) // 2 * 2),
                max(2, int(fr.height * scale) // 2 * 2),
            ),
            Image.Resampling.LANCZOS,
        )
        for fr in frames
    ]
    step = 3 if len(small) > 240 else 2
    sampled = small[::step]
    sampled[-1] = sampled[0].copy()
    duration = int(round(1000 * step / fps))
    sampled[0].save(
        out,
        save_all=True,
        append_images=sampled[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out}  ({len(sampled)} frames)", flush=True)


def _pad_one(
    fr: Image.Image,
    font: ImageFont.ImageFont,
    size: tuple[int, int] = (1080, 1920),
) -> Image.Image:
    """Reframe one wide frame to 9:16 with a URL footer (stream-friendly).

    The wide stream's final frame equals its first frame, so padding each frame
    independently keeps the vertical loop seamless without a post-pass fixup.
    """
    vw, vh = size
    canvas = Image.new("RGB", size, BG)
    url_y = vh - 88
    avail_w = vw - 48
    avail_h = vh - 220  # leave headroom + footer room
    scale = min(avail_w / fr.width, avail_h / fr.height)
    nw = max(2, int(round(fr.width * scale)) // 2 * 2)
    nh = max(2, int(round(fr.height * scale)) // 2 * 2)
    fitted = fr.resize((nw, nh), Image.Resampling.LANCZOS)
    # Center the plate in the band above the URL footer (slight upward bias) so
    # the 9:16 frame reads balanced instead of top-heavy with a dead void below.
    top = max(60, (url_y - nh) // 2 - 20)
    canvas.paste(fitted, ((vw - nw) // 2, top))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (vw // 2, url_y),
        "ilevytate.github.io/ItaSoRL",
        fill=MUTED,
        font=font,
        anchor="mm",
    )
    return canvas


def _section_frames(
    seq: list[Image.Image],
    beat: Beat,
    out_size: tuple[int, int],
    fps: int,
    steps_between: int,
    target_s: float,
):
    """Yield a standalone, seamlessly-looping clip for ONE section.

    No chapter card: the persistent caption band already names the section, and a
    card would flash on every GIF loop. The training progression plays once (the
    "watch it learn" beat), then the settled state holds with a gentle brightness
    breath and a slow continuous push-in so it never goes dead; a final crossfade
    returns to frame 0 so both the mp4 and the derived GIF loop without a seam.
    Streams frame-by-frame (only the opening frame is kept, for the seam).
    """
    vig = _vignette(out_size, 0.52)
    font_chapter = _font(16 if out_size[0] < 1280 else 18, bold=True)
    font_line = _font(26 if out_size[0] < 1280 else 32, bold=True)
    font_foot = _font(15)
    dual = beat.box_l is not None and beat.box_r is not None

    def _frame(src: Image.Image, z: float, progress: float) -> Image.Image:
        if dual:
            plate = _zoom_plate_smooth(
                _dual_plate(src, beat.box_l, beat.box_r, labels=beat.labels), z
            )
            return _compose_plate(
                plate, out_size, beat.chapter, beat.line,
                progress=progress,
                font_chapter=font_chapter,
                font_line=font_line,
                font_foot=font_foot,
                vig=vig,
            )
        box = _shrink_box(beat.box, z)
        return _compose_frame(
            src, box, out_size, beat.chapter, beat.line,
            progress=progress,
            font_chapter=font_chapter,
            font_line=font_line,
            font_foot=font_foot,
            vig=vig,
        )

    total = max(2, int(round(target_s * fps)))
    seam_n = max(2, int(round(0.6 * fps)))
    body_n = max(1, total - seam_n)
    zmax = 0.07 if dual else 0.045

    if beat.static:
        smooth: list[Image.Image] = [seq[-1]]
        anim_n = 0
    else:
        smooth = _interp_sequence(seq, steps_between)
        anim_n = min(len(smooth), body_n)

    first: Image.Image | None = None
    prev: Image.Image | None = None
    for i in range(body_n):
        t = i / max(1, body_n - 1)
        z = zmax * _ease(t)
        src = smooth[i] if i < anim_n else smooth[-1]
        fr = _frame(src, z, progress=t)
        if i >= anim_n:
            # Settled portion: one full breath that returns to 0 at the loop point.
            bt = (i - anim_n) / max(1, body_n - anim_n)
            breath = 0.03 * np.sin(2 * np.pi * bt)
            if abs(breath) > 1e-3:
                fr = ImageEnhance.Brightness(fr).enhance(1.0 + float(breath))
        if first is None:
            first = fr
        prev = fr
        yield fr

    # Seamless loop seam: crossfade the settled close back to the opening frame,
    # ending exactly on frame 0 (mirrors the full tour's seam).
    for k in range(1, seam_n):
        yield _blend(prev, first, _ease(k / (seam_n + 1)))
    yield first.copy()


def _render_section(
    seq: list[Image.Image],
    beat: Beat,
    out_size: tuple[int, int],
    fps: int,
    steps_between: int,
    target_s: float,
    out_dir: Path,
    want_gif: bool,
    vertical: bool,
) -> None:
    """Stream one section clip to wide + vertical mp4 (+ optional loop GIF).

    Memory-safe: only downscaled, pre-sampled frames are retained for the GIF, so
    a 12s 1080p section never holds the full frame list resident.
    """
    name = beat.name
    wide_out = out_dir / f"loop-brain-pair-{name}.mp4"
    vert_out = out_dir / f"loop-brain-pair-{name}-vertical.mp4"
    wide = _open_encoder(wide_out, fps)
    vert = _open_encoder(vert_out, fps) if vertical else None
    vfont = _font(20)

    gif_small: list[Image.Image] | None = [] if want_gif else None
    gif_step = 3
    gif_scale: float | None = None
    n = 0
    for fr in _section_frames(seq, beat, out_size, fps, steps_between, target_s):
        _feed(wide, fr)
        if vert is not None:
            _feed(vert, _pad_one(fr, vfont))
        if gif_small is not None and n % gif_step == 0:
            if gif_scale is None:
                gif_scale = 640 / fr.width
            gif_small.append(
                fr.resize(
                    (
                        max(2, int(fr.width * gif_scale) // 2 * 2),
                        max(2, int(fr.height * gif_scale) // 2 * 2),
                    ),
                    Image.Resampling.LANCZOS,
                )
            )
        n += 1

    _finish(wide, wide_out)
    print(f"Wrote {wide_out}  ({n} frames, {n / fps:.1f}s @ {fps}fps)", flush=True)
    if vert is not None:
        _finish(vert, vert_out)
        print(f"Wrote {vert_out}  ({n} frames)", flush=True)
    if gif_small:
        gif_small[-1] = gif_small[0].copy()  # guarantee a clean loop point
        gif_out = out_dir / f"loop-brain-pair-{name}-loop.gif"
        duration = int(round(1000 * gif_step / fps))
        gif_small[0].save(
            gif_out,
            save_all=True,
            append_images=gif_small[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"Wrote {gif_out}  ({len(gif_small)} frames)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "viz/out/mind_brain_pair.gif",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "assets/film/clips")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--steps-between", type=int, default=4)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--vertical", action="store_true", default=True)
    p.add_argument("--no-vertical", action="store_false", dest="vertical")
    p.add_argument(
        "--sections",
        action="store_true",
        help="render standalone, seamlessly-looping per-section clips "
        "(loop-brain-pair-<section>.mp4 / -vertical.mp4 / -loop.gif) "
        "instead of the single stitched tour",
    )
    p.add_argument(
        "--section-seconds",
        type=float,
        default=12.0,
        help="target length of each standalone section clip (10-15 works well)",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not a.source.exists():
        raise SystemExit(f"Missing source: {a.source}")
    seq = _load_sequence(a.source)
    if len(seq) < 2:
        raise SystemExit(f"Need multi-frame source; got {len(seq)} from {a.source}")
    print(f"Loaded {len(seq)} training frames from {a.source}", flush=True)

    w = a.width - a.width % 2
    h = a.height - a.height % 2
    out_dir = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir

    if a.sections:
        # Standalone looping clips: play the "learn" progression more gradually so
        # each section's forming animation reads on its own (min 12 in-betweens),
        # then dwell on the settled result. GIFs are written when --gif is passed.
        sec_steps = max(a.steps_between, 12)
        print(
            f"Rendering {len(BEATS)} standalone section loops "
            f"(~{a.section_seconds:.0f}s each @ {a.fps}fps"
            f"{', +gif' if a.gif else ''})",
            flush=True,
        )
        for beat in BEATS:
            _render_section(
                seq, beat, (w, h), a.fps, sec_steps, a.section_seconds,
                out_dir, a.gif, a.vertical,
            )
        return

    wide_out = out_dir / "loop-brain-pair-tour.mp4"
    vert_out = out_dir / "loop-brain-pair-tour-vertical.mp4"
    wide = _open_encoder(wide_out, a.fps)
    vert = _open_encoder(vert_out, a.fps) if a.vertical else None
    vfont = _font(20)
    gif_frames: list[Image.Image] | None = [] if a.gif else None

    # Single streaming pass: build each frame once, fan it out to both encoders.
    # Peak memory stays tiny (a few frames), so 2K/high-fps renders are safe.
    n = 0
    for fr in _iter_film(seq, (w, h), a.fps, a.steps_between):
        _feed(wide, fr)
        if vert is not None:
            _feed(vert, _pad_one(fr, vfont))
        if gif_frames is not None:
            gif_frames.append(fr)
        n += 1
        if n % 60 == 0:
            print(f"  ...{n} frames", flush=True)

    _finish(wide, wide_out)
    print(f"Wrote {wide_out}  ({n} frames, {n / a.fps:.1f}s @ {a.fps}fps)", flush=True)
    if vert is not None:
        _finish(vert, vert_out)
        print(f"Wrote {vert_out}  ({n} frames)", flush=True)
    if gif_frames:
        _write_gif(gif_frames, out_dir / "loop-brain-pair-tour-loop.gif", a.fps)


if __name__ == "__main__":
    main()
