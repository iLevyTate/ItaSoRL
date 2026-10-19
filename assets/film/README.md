# Film assets

What the site serves, where each file came from, and how to rebuild it. Two
films carry the project now. The first is the 90-second story with voice and
music. The second is Two Minds, the labeled brain film.

## What is in git and what is not

This directory is gitignored by default. Only the six files the site actually
serves are allowlisted by name in `.gitignore`: the two 720p web encodes, their
two poster frames, the captions of the voiced film, and this README. The four
companion loop mp4s that used to be allowlisted were retired on 2026-10-10 (last
section). Everything
else here is a local build output and stays on local disk, including all of
`clips/`, every 1080p and 9:16 master, the textless passes, the loop gifs, and
the voiced `.mov` masters. That is about 930 MB kept out of the repo across
`clips/`, `viz/out/`, `viz/artifacts/`, `artifacts/clip_audit/`, and `promo/`.

The picture is byte-reproducible from `viz/` code plus `viz/data/scene.json`, so
re-render rather than commit. If you ever do need to publish a new file from
here, add an explicit `!/assets/film/<name>` line; without one it is ignored.

## Film one: Detectable All Along (90 s, voice and music)

| File | What it is | Source |
|------|------------|--------|
| `detectable-all-along-web.mp4` | 1280x720, 24 fps, H.264 CRF 23, AAC 128 kb/s stereo, 89.8 s, 12.5 MB. The cut the site plays. | Web encode of the released master (below). |
| `detectable-all-along-poster.jpg` | Poster frame at 1.5 s. | Same master. |
| `detectable-all-along-web.vtt` | WebVTT captions of the voiceover, 27 cues timed to the speech in the web encode (the last ends at 1:20.8). Added 2026-10-19. | The recorded script in `docs/specs/2026-07-22-outreach-video-vo-music.md`, checked against the encode's audio. |

The master is `ItasorlOverview.mov`: 1920x1080, 24 fps, H.264 at 21 Mb/s with
a 320 kb/s AAC track, 239.8 MB, exported from DaVinci Resolve and released on
24 August 2026. It is too large for the repo and lives in the owner's Google
Drive. The picture is the `viz/player` capture (beats in `viz/player/beats.json`,
scene in `viz/data/scene.json`); the voice follows the script in
`docs/specs/2026-07-22-outreach-video-vo-music.md`. The three numbers on screen
(99%, 50%, 73%) are the August displays. `beats.json` has since been rewritten on
the corrected record (below), so the August beats live in its git history.

The web encode is a historical cut, and the site labels it that way. Its
numbers come from the pre-correction trainer. The 50% is the untrained
network's floor (0.488) and the historical L0 check (0.517), not a trained
creature with nothing at stake; a creature trained only to predict reads 0.589
on the corrected trainer. The 73% is the 0.726 residual after a behavior
control; on the corrected agents the best-fitting control leaves 0.620, under
the bar (FINDINGS 17.8). From 45 s to the end, picture and voice credit
survival for the creature noticing ("It learned because its life depended on
it"; the end card's "The creature only noticed when survival made it matter").
That reading is withdrawn (FINDINGS 17.6): fed the survival agent's own
senses, a brain trained only to predict reads the world within about 0.01 of the
survival brain (0.721 against 0.733). The caveat
sits beside the player and the download link on the site. A downloaded copy
carries none of it, so a clean cut needs a new edit of the master (two VO lines
and the end card).

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
browser at `viz/player/` (transport bar included), and `viz/player/capture/`
re-renders it to MP4 when a silent social cut is needed.

On 2026-10-11 the live cut was rewritten on the corrected record: `beats.json`
carries 99% (the outside watcher, 0.991), 52% (an untrained mind, 0.523), 73% (the
forager's mind, 0.733), 72% (a mind trained only to predict, fed the same senses,
0.721) and the open gate (two different samples of real worlds read 56%, 0.559). The living
beat shows the recorded creature of the C1 rerun's seed 0 (`viz/data/scene.json`).

On 2026-10-16 the captions lost their awareness verbs ("can't tell", "can tell", "IT DOESN'T
REACT", "NOW IT NOTICES"). Removing the decoded direction while the creature acts changes its
behavior no more than a sham removal (FINDINGS 23.3), so the cut now says the fake shows in
its memory and that the creature never acts on it. The 52% beat used to walk the trained
creature; it now walks the untrained agent the 52% was read from, on the same seed
(`trajs.untr`, added by `viz/collect.py --untrained-only`). That agent starves at step 47, so
the beat plays its world clock at 0.7x (`"clock"` in `beats.json`) rather than show a death.
`viz/collect.py` checks every number on screen against the committed artifacts,
and `tests/test_viz_collect.py` runs that check in CI. The silent 4:5 and 9:16
masters of this cut are local build outputs. A voiced corrected cut needs a new
voice track over the new picture; until one exists, the web encode above stays the
historical cut.

## Film two: Two Minds (66 s, seven chapters)

Only `two-minds-web.mp4` and `two-minds-poster.jpg` are tracked. Everything
else, `clips/` included, is a local build output, gitignored, rebuilt on demand
with `viz/player/capture/build-two-minds.ps1`; the site links the web encode.

| File | What it is | Source |
|------|------------|--------|
| `two-minds-web.mp4` | 1280x720, 30 fps, H.264 CRF 25, no audio track, 65.9 s. The cut the site plays, intro and outro included. | Web encode of `clips/two-minds-tour.mp4`. |
| `two-minds-poster.jpg` | Poster frame at 2 s (the title card). | Same. |
| `clips/two-minds-tour.mp4` (local only) | 1920x1080 tour: intro title card (9.5 s), the five sections (overview 12, senses 9, process 9, memory 12, actions 12), outro end card (6 s), joined with 0.6 s fade-to-black. Trims sum to 69.5 s; the six 0.6 s crossfades overlap away 3.6 s, so the film runs 65.9 s. | `viz/player/capture/build-two-minds.ps1` |
| `clips/two-minds-tour-vertical.mp4` (local only) | 1080x1920 tour. | Same. |
| `clips/two-minds-tour-loop.gif` (local only) | 420 px wide, 7 fps loop GIF of the tour. | Same. |
| `clips/two-minds-<section>.mp4` (local only) | 12 s seamless loop per section: overview, senses, process, memory, actions. The intro and outro are captured the same way as full 12 s chapters, `clips/two-minds-title.mp4` and `clips/two-minds-end.mp4`, but only feed the tour, where they are trimmed to 9.5 s and 6 s. | Same. |
| `clips/two-minds-<section>-vertical.mp4` (local only) | 9:16 of each section. | Same. |
| `clips/two-minds-<section>-loop.gif` (local only) | 560 px wide, 10 fps loop GIF of each section. | Same. |

The tracked web encode is the October 2026 cut, rebuilt on 2026-10-11 from the
corrected renderer and the corrected C1 brain (seed 9) by the rebuild at the end
of this section. It replaced the September cut, whose numbers and brain data were
pre-correction, in place. All 22 build outputs were decode-checked: 14 section
clips at 12.0 s, two tours at 65.9 s (1977 frames), six loop GIFs.

### Labels on the published cuts

The live player of film one (`viz/player/`) had its end card reworded on
2026-10-08 to drop the reading that credited survival on its own (withdrawn in
FINDINGS 17.6), and its footer names the pre-correction trainer
(`viz/player/beats.json`). On 2026-10-10 its end-card tiles were relabeled
"untrained mind" and "trained mind" (they read "its mind, idle" and "its mind,
surviving"). On 2026-10-11 the 45 to 80 s beats were rewritten on the corrected
record (film one, above), so the two live cuts no longer tell the survival
story.

On 2026-10-09 the caveats were extended on every surface that publishes the films
(the site film block, the live player's end-card footer, and this file) with the corrected
record: the survival state reads 0.733 under the corrected trainer with the registered rule
as a whole not met, because the authentic-versus-authentic gate is open on the registered
pair of evaluation worlds (0.559, the largest of nine draws on record), and the level is not
a product of that draw: on the agents of the historical hidden-8 run, every one of eight
independent draws at drift 0.45 clears the bar (0.728 to 0.860, SECURE under a rule frozen
before the run; FINDINGS 17.5.1). The corrected agents are saved on the origin branch
`run/C1-rerun`, and on 2026-10-18 the same readout scored the draws on them: the eight
independent draws read 0.685 to 0.829, 8 of 8 at or above 0.65, and the registered draw, 0.733,
ranks 6 of 9 (R1 SECURE, post hoc; the 2026-10-19 note in FINDINGS 17.5.1). Propagated with
C1's own between-draw SD of 0.052, the corrected primary reads 0.733 [0.618, 0.848], a lower
bound under the 0.65 bar; the record's propagation with the historical 0.042 stays. The live
player's end-card footer gives the C1 range since 2026-10-19. The film-one mp4 is unchanged,
and its caveats sit beside it. The Two Minds mp4 was re-rendered on the corrected record on
2026-10-11 and carries its caveats on its end card.

A voiced cut of Two Minds has not been committed. If one is produced, encode it
the same way as film one and replace `two-minds-web.mp4` in place so the site
and the player pick it up without a markup change.

### What the renderer shows (corrected record, October 2026)

The renderer is `viz/player/brain/` (open `index.html?sec=overview&layout=wide`
in a browser for a live loop). One trained brain, frozen, runs in the real world
and in the fake copy. Every dot on screen is named after a real sensor group,
encoder row, recurrent unit, motor head, or readout head, and teal rings mark the
memory cells whose readings differ most between the two worlds.

Every number on screen comes from the corrected trainer (`docs/FINDINGS.md`
section 17) and sits in the `FILM` block of `brain.js`, one line each. Each is a
mean over ten brains of one kind, which is why the labels are plural; the brain
drawn on screen is one of the ten trained ones:

| On screen | Value | Source |
|---|---|---|
| untrained brains, 52% | 0.523, interval [0.491, 0.554], so chance | `artifacts/corrected_verdicts.json` `runs.C1.primary.untrained` |
| brains like this one, 73% | 0.733 [0.669, 0.797] | `runs.C1.primary.survival` |
| the bar, 65% | 0.65, registered before the run | `rules.bar` |
| predict-only brains fed the same senses, 72% | 0.721 | `artifacts/policy_controls/corrected_l3_h8_wm.json` `aggregate["predictor replay"]` |
| without the prediction job at 300 updates, 61% | 0.613, under the bar | `runs.C2.primary.survival` |
| without it at 450 updates, 69% | 0.685 | `artifacts/budget_curve.json` `series["decoder off"]` |
| two different samples of real worlds, 56% | 0.559, so the registered rule is not met | `runs.C1.gates.l0` |

`python scripts/dump_brain_film_data.py --check-only` compares each display with
those artifacts and stops on any mismatch. It also checks the sentences built on
them: the untrained interval includes 0.5, the trained arm clears the bar, the arm
without the prediction job is under it at 300 updates, and the L0 gate did not
pass. `tests/test_brain_film.py` runs the same check in CI, along with checks that
no withdrawn wording, historical figure or stack name renders.

What changed from the September cut, and why:

- The memory chapter's one meter climbed from 49% to 73% inside one brain. Those
  were two different agents, an untrained one (0.488, the pre-correction floor) and
  a trained one (0.726, the pre-correction section 10.4 residual after a behavior
  control), each read once after training. The chapter now has two labeled rows,
  "untrained brains" at 52% and "brains like this one" at 73%, against the 50%
  and 65% ticks. Each bar fills once and holds.
- The end card said "this brain kept a trace of which world it was in" over a
  single 73%. Under identical input the difference fades within a few steps, so
  the frozen verdict on storage is "retention not shown" (FINDINGS 17.7). The card
  now says the brain carries a readable sign of which world it is in and shows
  three tiles: 52%, 73%, and 72% for brains trained only to predict and fed the
  trained brains' own senses. That last reading sits within about 0.01 of 73%, so the
  reading that credited the survival objective is withdrawn (17.6).
- The card's second-job line now carries its budget: under the bar at 300 updates
  (61%), 69% at 450 (17.4, 17.10). A new line states the open gate: two identical
  real worlds read 56%, so the preregistered test is not passed (17.2).
- Labeled builds carry no on-screen narration. Narration renders only with
  `caption=1`, and the voice is recorded over the labeled cut. The labeled panels
  took the freed band: wide panels grew from 624 to 700 px tall, 9:16 panels from
  580 to 660.
- The narration that renders with `caption=1` lost the lines the record does not
  support: the masking line (its run is pre-correction and has no corrected
  counterpart), "the miss leaves a mark in memory", "that is what memory picks
  up", and the stack name in the caption subline.

Chapter lengths are unchanged (title 9.5 s, overview 12, senses 9, process 9,
memory 12, actions 12, end 6), so a music bed cut to the 65.9 s September picture
still fits.

### Brain data

The twelve memory cells and the ringed cells come from
`viz/player/brain/brain-data.js`, which `scripts/dump_brain_film_data.py` computes
from drift-0.45 state pools rebuilt out of the corrected C1 agents. The
representative brain is the seed whose committed survival target sits closest to
the C1 mean: seed 9, at 0.738 against 0.733. Ring strength is each unit's measured
rank AUROC between worlds. The script writes nothing unless that seed's rebuilt
survival and untrained pools reproduce the committed C1 cells' pooled targets
within `--atol` (default 1e-9, which only a rebuild on the stack that ran C1 can
meet).

The committed `brain-data.js` was rebuilt on the owner's Windows stack, not the
cloud stack that ran C1, so its pools agree with the committed targets to within
0.002 rather than to the bit: survival 0.7372 against 0.7384, untrained 0.5760
against 0.5756. The data records both readings and the tolerance used
(`rebuild_atol` 0.005), and `tests/test_brain_film.py` refuses a tolerance above
0.005.

### Rebuild

The C1 agents were not kept by the original run. A cloud rerun of C1 saved them
(`C1-RERUN` in docs/RESULTS_MANIFEST.md). Its pool targets equal the committed C1 cells to
the bit in 60 of 60 arm-cells, and every numeric field of its 20 cell files equals the
committed file. The files themselves are not byte-identical: each rerun cell records its own
commit (fb3a393, 5580816 or 2fce6a5 rather than f676b95) and carries extra knob and objective
keys that the committed cells lack. Only the seed-9 drift-0.45 survival and untrained agents
are needed. From the repo root, in PowerShell, one step at a time:

```powershell
$run = '<C1 rerun dir, holding agents\ and cells\>'
$sub = "$env:TEMP\two-minds-c1-s9"
New-Item -ItemType Directory -Force "$sub\agents", "$sub\cells", "$sub\states" | Out-Null
Copy-Item "$run\agents\agent_d0.45_s9_survival.pt", "$run\agents\agent_d0.45_s9_untrained.pt" "$sub\agents"
Copy-Item "$run\cells\cell_d0.45_s9.json" "$sub\cells"
$env:ITASORL_FOLDS = 'explicit'
python scripts/regenerate_state_dumps.py --run-dir $sub --out-dir "$sub\states" --report "$sub\states\report.json"
python scripts/dump_brain_film_data.py --states "$sub\states" --report "$sub\states\report.json"
```

Then serve the renderer on port 8877 from `viz/player/brain/` and run
`viz/player/capture/build-two-minds.ps1`. It decode-checks every clip and the
tour stitch; check the tours with `ffprobe` (65.9 s) before encoding the web cut
below.

Web encode:

```bash
ffmpeg -i assets/film/clips/two-minds-tour.mp4 -an \
  -vf "scale=1280:720:flags=lanczos" \
  -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -profile:v high \
  -movflags +faststart assets/film/two-minds-web.mp4
ffmpeg -ss 2 -i assets/film/clips/two-minds-tour.mp4 -frames:v 1 \
  -vf scale=1280:720 -q:v 4 assets/film/two-minds-poster.jpg
```

## Retired companion loops (2026-10-10)

Until 2026-10-10 the site's "More ways to see it" row autoplayed four short
loops: `loop-idle-clouds.mp4`, `loop-survival-clouds.mp4`, `loop-race.mp4` and
`loop-brain-pair.mp4`. They were July 2026 demo renders of an early short run,
eight frames each at under one frame a second, and each opened on a title card. Their
burned-in text stated readings the record no longer supports: "YES. It knows
this world is fake." over two clouds that mostly overlap, a race whose perfect
copy "never learns" and whose brain "is taught", and a detection score climbing
over training steps 0 to 24, a time course that was never measured. The
generator of the cloud and race loops is not on `main`, and the loops did not
start and end on the same frame. The row, the player gallery's copies, and the
four `.gitignore` allowlist lines are gone; the files stay in git history.

A data-drawn replacement set (`loop-two-worlds`, `loop-four-brains`,
`loop-echoes`, `loop-seam-dial`, drawn by `viz/player/loops/` from committed
artifacts) was written on a side branch (commit `47a6423`) and never merged. It
was not carried over here: its pictures carry pre-correction numbers with no
trainer label (0.752 as "the published brain"; 0.654 "at the bar" where the
corrected agents leave 0.620, under it), and re-rendering it on the corrected
record is a film build, left to the owner. The section-03 readout chart on the
site now draws the corrected per-seed readouts directly.
