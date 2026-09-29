# Film assets

What the site serves, where each file came from, and how to rebuild it. Two
films carry the project now. The first is the 90-second story with voice and
music. The second is Two Minds, the labeled brain film.

## Film one: Detectable All Along (90 s, voice and music)

| File | What it is | Source |
|------|------------|--------|
| `detectable-all-along-web.mp4` | 1280x720, 24 fps, H.264 CRF 23, AAC 128 kb/s stereo, 89.8 s, 12.5 MB. The cut the site plays. | Web encode of the released master (below). |
| `detectable-all-along-poster.jpg` | Poster frame at 1.5 s. | Same master. |

The master is `ItasorlOverview.mov`: 1920x1080, 24 fps, H.264 at 21 Mb/s with
a 320 kb/s AAC track, 239.8 MB, exported from DaVinci Resolve and released on
24 August 2026. It is too large for the repo and lives in the owner's Google
Drive. The picture is the `viz/player` capture (beats in `viz/player/beats.json`,
scene in `viz/data/scene.json`); the voice follows the script in
`docs/specs/2026-07-22-outreach-video-vo-music.md`. The three numbers on screen
(99%, 50%, 73%) are the published displays, sourced in `beats.json`.

Rebuild the web encode from the master with ffmpeg:

```bash
ffmpeg -i ItasorlOverview.mov -map 0:v:0 -map 0:a:0 \
  -vf "scale=1280:720:flags=lanczos" -r 24 \
  -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -profile:v high \
  -c:a aac -b:a 128k -ac 2 -movflags +faststart \
  assets/film/detectable-all-along-web.mp4
ffmpeg -ss 1.5 -i ItasorlOverview.mov -frames:v 1 -vf scale=1280:720 -q:v 4 \
  assets/film/detectable-all-along-poster.jpg
```

The captions-only 4:5 cut is no longer shipped as a file. It plays live in the
browser at `viz/player/` (same beats, same scene, transport bar included), and
`viz/player/capture/` re-renders it to MP4 when a silent social cut is needed.

## Film two: Two Minds (66 s, seven chapters)

Only `two-minds-web.mp4` and `two-minds-poster.jpg` are tracked. Everything
under `clips/` is a local build output, gitignored, rebuilt on demand with
`viz/player/capture/build-two-minds.ps1`; the site links the web encode.

| File | What it is | Source |
|------|------------|--------|
| `two-minds-web.mp4` | 1280x720, 30 fps, H.264, no audio track, 65.9 s (1977 frames). The cut the site plays. | Web encode of `clips/two-minds-tour.mp4`, end card corrected 2026-09-29 (below). |
| `two-minds-poster.jpg` | Poster frame at 2 s (the title card). | Same. |
| `clips/two-minds-tour.mp4` (local only) | 1920x1080 tour: title card (9.5 s), the five sections (overview 12, senses 9, process 9, memory 12, actions 12), end card (6 s), joined with 0.6 s fade-to-black. | `viz/player/capture/build-two-minds.ps1` |
| `clips/two-minds-tour-vertical.mp4` (local only) | 1080x1920 tour. | Same. |
| `clips/two-minds-tour-loop.gif` (local only) | 420 px wide, 7 fps loop GIF of the tour. | Same. |
| `clips/two-minds-<section>.mp4` (local only) | 12 s seamless loop per section: overview, senses, process, memory, actions. `title` and `end` are captured the same way but only feed the tour. | Same. |
| `clips/two-minds-<section>-vertical.mp4` (local only) | 9:16 of each section. | Same. |
| `clips/two-minds-<section>-loop.gif` (local only) | 560 px wide, 10 fps loop GIF of each section. | Same. |

The renderer is `viz/player/brain/` (open `index.html?sec=overview&layout=wide`
in a browser for a live loop). Every dot on screen is named after a real
sensor group, encoder row, recurrent unit, motor head, or readout head. The
twelve memory cells and the teal-ringed clue cells come from
`viz/player/brain/brain-data.js`, which `scripts/dump_brain_film_data.py`
computes from the saved held-out state pools: the representative brain is the
seed closest to the 0.752 pooled survival mean, and ring strength is that
unit's measured rank AUROC between worlds. Numbers on screen are canonical
FINDINGS values only, and one chapter carries the live meter: 0.488 floor
(49%), 0.65 bar, 0.726 survival (73%, the behaviour-independent component of
section 10.4, 90% CI 0.685 to 0.765), 0.928 watcher gate (93%). The pooled
0.752 and the masked-sense figures (0.686, 0.500) belong to the uncontrolled
family and are not shown as numbers; the senses chapter states that direction
in words.

**End card correction (2026-09-29).** The v3 end card said "A brain trained only
to survive kept a trace of which world it was in." FINDINGS 10.8 contradicts
that: without its next-observation predictor the survival brain reads 0.601,
under the bar. `brain.js` now says "A brain trained to survive and predict kept a
trace of which world it was in." The web encode was patched without the local
tour master: frames 0 to 1800 are the v3 web encode unchanged in content (PSNR
49 dB after one re-encode), and from frame 1801, where the fade to black bottoms
out, the corrected `?sec=end&layout=wide` capture fades up on the original's luma
ramp; re-encoded at CRF 22. That capture was rendered on Linux with Selawik,
Microsoft's metric-compatible open substitute, standing in for Segoe UI. The
local 1080p and 9:16 clips still carry the old card until the next
`build-two-minds.ps1` run, which renders the corrected text in Segoe UI and
supersedes this patch.

A voiced cut of Two Minds has not been committed. If one is produced, encode it
the same way as film one and replace `two-minds-web.mp4` in place so the site
and the player pick it up without a markup change.

Web encode:

```bash
ffmpeg -i assets/film/clips/two-minds-tour.mp4 -an \
  -vf "scale=1280:720:flags=lanczos" \
  -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -profile:v high \
  -movflags +faststart assets/film/two-minds-web.mp4
ffmpeg -ss 2 -i assets/film/clips/two-minds-tour.mp4 -frames:v 1 \
  -vf scale=1280:720 -q:v 4 assets/film/two-minds-poster.jpg
```

## Companion loops (the "More ways to see it" row)

Four 1280x720, 30 fps, H.264 CRF 23 loops with no audio. Each is a page of
`viz/player/loops/` (open `index.html?loop=<name>` for the live version) that reads
its numbers at load time from committed files and refuses to draw if a derived
mean differs from its FINDINGS value. Dots move between measured states as a
visual tween; a number is only on screen while the picture sits on a measured
state. Frame 0 equals the frame at the loop length, so every clip loops cleanly.

| File | `?loop=` | What it shows | Data |
|------|----------|---------------|------|
| `loop-two-worlds.mp4` | `worlds` | The recorded seed-0 survival brain in the real and the fake world from the same start, steps 0 to 140, with the measured distance between the two creatures. The paths part at step 27 (more than 5% of the world apart). 12 s. | `viz/data/scene.json` (`viz/collect.py`) |
| `loop-four-brains.mp4` | `brains` | Per-seed probe AUROC for untrained (0.488), prediction-only (0.573), survival without its predictor (0.601, CPU; 0.730 with it on the same CPU), and survival and prediction (0.752), against the coin flip and the 0.65 bar. 11 s. | `artifacts/expB2/heldout_l3_h8_summary.json`, `arch_baseline_l3_h8_nowm.json`, `device_control_l3_h8_wm_cpu.json`; FINDINGS 10.2, 10.8 |
| `loop-echoes.mp4` | `echoes` | The same ten survival seeds as behavior (0.726), senses (0.731), and both with a nonlinear fit (0.654) are subtracted. 12 s. | `artifacts/expB2/behavior_audit_l3_h8_heldout.json`, `sensory_echo_l3_h8.json`, `sensory_echo_l3_h8_mlp.json`; FINDINGS 10.4, 10.4.2 |
| `loop-seam-dial.mp4` | `dial` | The H2 graded seam: flaw left at 100, 75, 50, 25, 10, 0%; survival 0.752 falls to 0.506 while the untrained floor stays near chance. 11 s. | `artifacts/expH2/summary.json`; FINDINGS 14 |
| `clips/loop-brain-pair-*` (local only) | | The earlier brain-pair tour and its per-section loops (wide, vertical, GIF). Superseded by Two Minds for outreach. | `scripts/render_brain_pair_tour.py` |

The four loops that played here until 2026-09-29 (`loop-idle-clouds`,
`loop-survival-clouds`, `loop-race`, `loop-brain-pair`) were retired. They were
July demo renders of an early short run (training steps 0 to 24), not the n = 10
results, and their headlines ("It knows this world is fake", "separate islands")
claimed more than the overlapping clouds they drew. They remain in git history.

Rebuild (serve the repo root, so the page can reach `artifacts/` and
`viz/data/`; `ffmpeg` with libx264 on PATH; Playwright Chromium):

```bash
python -m http.server 8931 --bind 127.0.0.1 &   # from the repo root
cd viz/player/capture
for pair in worlds:two-worlds brains:four-brains echoes:echoes dial:seam-dial; do
  CAP_SOURCE=artifacts CAP_CRF=23 CAP_FPS=30 \
  CAP_URL="http://127.0.0.1:8931/viz/player/loops/index.html?loop=${pair%%:*}" \
  CAP_OUT="../../../assets/film/loop-${pair##*:}.mp4" \
  CHROME_EXE=/path/to/chromium node capture-brain.js
done
for f in ../../../assets/film/loop-*.mp4; do ffmpeg -nostdin -v error -i "$f" -f null -; done
```
