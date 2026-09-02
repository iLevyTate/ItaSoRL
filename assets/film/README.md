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

## Film two: Two Minds (58 s, five chapters)

| File | What it is | Source |
|------|------------|--------|
| `two-minds-web.mp4` | 1280x720, 30 fps, H.264 CRF 25, no audio track, 57.6 s. The cut the site plays. | Web encode of `clips/two-minds-tour.mp4`. |
| `two-minds-poster.jpg` | Poster frame at 2 s. | Same. |
| `clips/two-minds-tour.mp4` | 1920x1080 tour: the five sections joined with 0.6 s fade-to-black. | `viz/player/capture/build-two-minds.ps1` |
| `clips/two-minds-tour-vertical.mp4` | 1080x1920 tour. | Same. |
| `clips/two-minds-tour-loop.gif` | 420 px wide, 7 fps loop GIF of the tour. | Same. |
| `clips/two-minds-<section>.mp4` | 12 s seamless loop per section: overview, senses, process, memory, actions. | Same. |
| `clips/two-minds-<section>-vertical.mp4` | 9:16 of each section. | Same. |
| `clips/two-minds-<section>-loop.gif` | 560 px wide, 10 fps loop GIF of each section. | Same. |

The renderer is `viz/player/brain/` (open `index.html?sec=overview&layout=wide`
in a browser for a live loop). Every dot on screen is named after a real
sensor group, encoder row, recurrent unit, motor head, or readout head. The
twelve memory cells and the teal-ringed clue cells come from
`viz/player/brain/brain-data.js`, which `scripts/dump_brain_film_data.py`
computes from the saved held-out state pools: the representative brain is the
seed closest to the published 0.752 mean, and ring strength is that unit's
measured rank AUROC between worlds. Numbers on screen are canonical FINDINGS
values only (0.488 floor, 0.65 bar, 0.752 survival, 0.928 watcher gate,
vision masked 0.686, every sense masked 0.500).

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

| File | What it shows | Source |
|------|---------------|--------|
| `loop-idle-clouds.mp4` | Idle mind: memories from both worlds sit in one mixed blob. | `viz/player` capture |
| `loop-survival-clouds.mp4` | Surviving mind: the two worlds' memories drift into separate islands. | `viz/player` capture |
| `loop-race.mp4` | Two brains train side by side; the detection score only climbs when the fake matters. | `viz/player` capture |
| `loop-brain-pair.mp4` | Teal rings on the memory cells that hold the clue, present only under survival. | `scripts/render_brain_pair_tour.py` |
| `clips/loop-brain-pair-*` | The earlier brain-pair tour and its per-section loops (wide, vertical, GIF). Superseded by Two Minds for outreach; kept because posts already link them. | `scripts/render_brain_pair_tour.py` |

All loops start and end on the same frame.
