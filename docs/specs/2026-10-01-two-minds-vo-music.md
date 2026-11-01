# Two Minds: voiceover script and music, for ElevenLabs and Suno

Status (2026-11-01): rewritten for the corrected 68.9 s picture, not recorded. *Two Minds*
ships silent (`assets/film/two-minds-web.mp4`, one video stream, no audio). This spec is
what a voiced cut would say and sound like. Companion to
`docs/specs/2026-07-22-outreach-video-vo-music.md`, which covers the other film,
*Detectable All Along*. The 2026-10-01 version of this spec, written for Hume and the
65.9 s September picture, is in git history. Its script no longer matched the record,
and the section after the script says what changed.

Same house rule as film one's spec: **the voice never reads on-screen text verbatim.**
The labeled cut carries labels and readouts, and the voice carries the story. Numbers
spoken must match the displays exactly. The voice speaks only the memory chapter's
three, 52%, 73% and the 65% bar, and leaves the end card's 72%, 61%, 69%, 56% and 75% to
the card, which holds 9 s so they can be read.

## The picture you are scoring

`clips/two-minds-tour.mp4` (local) or the web encode, 68.9 s. Seven chapters joined by
0.6 s fades to black. The times below are positions in the finished cut, read from the
renderer (`viz/player/brain/brain.js`: 12 s loops, the missed grab at u 0.32, the rings
from u 0.40, the untrained row at u 0.08 to 0.20, the trained row at u 0.40 to 0.54,
three grabs from u 0.26 in the actions chapter, 0.19 apart).

| Chapter | In | Out | On screen | Timed events |
|---|---|---|---|---|
| title | 0.0 | 8.9 | "Can a brain tell a real world from a fake one?" | none |
| overview | 8.9 | 20.3 | One brain in the real world and in the copy | missed reach 12.7, rings from 13.7 |
| senses | 20.3 | 28.7 | Ten labeled sensor groups | none |
| process | 28.7 | 37.1 | Encoder rows, no recurrence | none |
| memory | 37.1 | 48.5 | Twelve memory cells by unit number, two readout rows | untrained row fills 38.1 to 39.5 (52%), trained row fills 41.9 to 43.6 (73%) |
| actions | 48.5 | 59.9 | Five motor heads, three prediction heads | grabs at 51.6, 53.9 and 56.2 |
| end | 59.9 | 68.9 | End card: 52%, 73% and 72% tiles, four notes | none |

## Voiceover script

One take per chapter, each placed at the time shown, as `AGENTS.md` asks (a film's VO
split into takes at beat boundaries). 119 words at the default speed of 0.9. The pause
tags are for Eleven v3 and v4. For Multilingual v2 or Flash, write `[pause]` as
`<break time="0.7s" />`.

| Place at | Chapter | Take |
|---|---|---|
| 0:02 | title | "One brain. Two worlds. Only one of them is real." |
| 0:11 | overview | "Watch its reach. [pause] In the copy, it falls short. One rule of motion is slightly off there." |
| 0:20.6 | senses | "Everything comes in through ten senses. [pause] The flaw arrives as speed, a little wrong every step." |
| 0:29.2 | process | "Here the senses blend into patterns. [pause] Nothing here remembers. It passes straight through." |
| 0:37.5 | memory | "Untrained brains read fifty-two percent, a coin flip. [pause] Brains trained like this one read seventy-three. We fixed the bar at sixty-five beforehand." |
| 0:48.8 | actions | "Its two jobs: stay alive, and predict what comes next. In the copy, the same move keeps missing, [pause] so what it senses changes too." |
| 1:00.5 | end | "No reward ever pointed at the fake. [pause] And no test shows it acting on what we read." |

Landing points:

- "In the copy, it falls short" just after the missed reach at 0:12.7.
- "fifty-two" as the untrained row fills, 0:38.1 to 0:39.5.
- "seventy-three" as the trained row fills, 0:41.9 to 0:43.6.
- "keeps missing" across the three grabs at 0:51.6, 0:53.9 and 0:56.2.
- The end take finishes by 1:08.5.

## ElevenLabs

- Voice: the owner's ElevenLabs voice for the series, the same one as film one, so the
  two films sound like one project. If a voice has to be designed, describe it as a warm,
  unhurried narrator, mid-range and close-mic'd, with gentle curiosity and no broadcast
  polish, comfortable with silence.
- Model and pauses: Eleven v3 and v4 take the bracketed tags above and ignore SSML
  breaks. Multilingual v2 and Flash take `<break time="0.7s" />`, at most 3 s per break.
- Speed 0.9. Numbers are written as words, so every take reads them the same way.
- Generate two or three variants of the memory take and the end take and pick in
  context, since those two carry the film.
- Deliver one AAC stereo track matched to the 68.9 s picture, separate from the music.

## What the script leaves out, and why

The 2026-10-01 script was written for the September picture and the record of that
week. Three of its claims are now withdrawn or unsupported, and none is in the script
above.

- "Before the miss: forty-nine percent. A coin flip. After: seventy-three." The 49% and
  the 73% were two different agents, an untrained one and a trained one, each read once
  after training. The corrected picture shows two rows, 52% and 73%, and the script reads
  them as two kinds of brain.
- "Survival alone is not enough. Expecting alone is not enough. It takes both." Without
  the prediction job the reading is under the bar at 300 updates (0.613, FINDINGS 17.4)
  and 0.685 at 450 (17.10), and a brain trained only to predict, fed the trained brain's
  own senses, reads 0.721 (17.6). The script names the two jobs and leaves these numbers
  to the end card.
- "Nobody asked it to notice. It noticed anyway." The film makes no awareness claim.
  Removing the decoded direction while the brain acts changes its behavior no more than
  a sham removal (23.3). The script ends on what is shown: no reward pointed at the fake,
  and no test shows the brain acting on what is read.

### The one line with a shelf life

"And no test shows it acting on what we read." rests on FINDINGS 23.3. The nudge and the
erasure found no behavioral effect above a sham, and both are limited tests, since the
erasure is partial. If a later use test shows an effect above a sham, rewrite this line
and the end card's "Not shown" note before re-recording anything else.

The September script's line with a shelf life was "Survival alone is not enough", which
rested on the 0.601 adjudicated in 10.8. The corrected record retired it: the arm without
the prediction job reaches 0.685 at 450 updates, and the predict-only brain reads 0.721 on
the same senses.

## Suno: music

Palette continuity with *Detectable All Along* (felt piano, warm analog pad, soft tape
hiss, 80 BPM), but this film is a tour of a small mind rather than a reveal: less arc,
more patience, one warming as the trained row fills.

Settings, as in `AGENTS.md`: Suno v5.5 Advanced, Instrumental on, Weirdness 50, Style
Influence Strong, no Style wand, no Voices. Watch the labeled cut once before
generating. Name the style "Two Minds 80" so later tries keep the same clocks.

**Style:**

```
Intimate instrumental documentary underscore, 80 BPM, 4/4, about 69 seconds, a patient tour of a small mind. Felt piano, warm analog pad, soft tape hiss, a slow piano arpeggio, a low drone. It opens on one sustained pad and a single unresolved felt piano note. A curious three-note motif enters, and a low drone slides underneath on one beat as something goes slightly wrong. A slow, even arpeggio follows, flat and unhurried, one note at a time, never editorializing. Then the only build: the pad swells and the motif returns in a fuller voicing, staying cool at first and warming once, quietly pleased, not triumphant. After it, a steady patient pulse with three soft low piano accents about two seconds apart. The ending strips back to the opening motif, resolved on the tonic, and decays to room tone. Interior, minimal, warm, never epic.
```

**Exclude styles:**

```
vocals, choir, humming, trailer hits, braams, epic orchestral, cinematic risers, drum kit, kick drum, snare, EDM drop, distorted bass, sidechain pumping
```

**Lyrics** (Instrumental on, every line bracketed, cue lines carry musical direction and
BPM only):

```
[Intro]
[0:00]
[One sustained warm pad, a single unresolved felt piano note, tape hiss, no percussion, 80 BPM]

[Verse]
[0:09]
[Curious three-note felt piano motif enters over the pad]
[0:13]
[Low drone slides in underneath and stays]

[Verse]
[0:20]
[Slow even piano arpeggio, one note at a time, flat and patient]
[0:29]
[Arpeggio continues unchanged, pad thins slightly]

[Chorus]
[0:37]
[Pad swells, motif returns in a fuller voicing, still cool]
[0:42]
[Warmth arrives once on a sustained major chord, quietly, then holds]

[Verse]
[0:48]
[Steady patient pulse, three soft low piano accents about two seconds apart from 0:51]

[Outro]
[1:00]
[Strip back to the opening motif, resolved on the tonic, slow decay, room tone by 1:08]
[End]
```

**Tries**, in the order `AGENTS.md` sets:

1. Create: two takes.
2. Create again with one Style tweak: "a slow celesta arpeggio" in place of "a slow piano
   arpeggio".
3. One slider try on the best so far: Weirdness 40.
4. Remaster the keeper: Subtle first, Normal if Subtle is too flat.

**Length.** Suno returns generations longer than 68.9 s. Trim to the picture and let the
take's own decay cover the end card's last seconds. If no take decays cleanly, build the
ending from the calmest take in the editor.

**Mix notes.** Duck the music about 6 dB under the voice. Let it back up in the voice's
gaps: 0:06 to 0:11 (the title into the first chapter), around 0:41 (between the two
readings) and 0:59.9 to 1:00.5 (into the end card). The film has no sound design of its
own. If anything is added, a single soft click on the missed reach at 0:12.7 is enough.

## If this gets made

`assets/film/README.md` already says how to land it: encode the voiced cut the same
way as film one and replace `two-minds-web.mp4` in place, so the site and the player
pick it up without a markup change. Update that README's row for the file, which
currently records "no audio track", and the provenance line that says a voiced cut has
not been committed.
