# Two Minds: voiceover script and music, for Hume and Suno

Status (2026-10-01): proposed, not recorded. *Two Minds* ships silent today
(`assets/film/two-minds-web.mp4`, one video stream, no audio). This spec is what a
voiced cut would say and sound like. Companion to the existing
`docs/specs/2026-07-22-outreach-video-vo-music.md`, which covers the other film,
*Detectable All Along*, and was recorded and released on 2026-08-24.

Same house rule as that spec: **the voice never reads the on-screen captions
verbatim.** Captions carry the claims, the voice carries the story. Numbers spoken
must match the displays exactly, which here means 49%, 65%, and 73% only. The 93%
watcher gate is on screen in one chapter but the voice does not touch it, because
explaining a privileged oracle costs more seconds than this film has.

## The picture you are scoring

`clips/two-minds-tour.mp4`, 65.9 s. Seven chapters joined by 0.6 s fade-to-black.
Trims sum to 69.5 s and the six crossfades overlap away 3.6 s. The windows below are
positions in the finished 65.9 s cut, which is what the VO and music are cut against.

| Chapter | In | Out | Length | On screen |
|---|---|---|---|---|
| title | 0.0 | 8.9 | 8.9 | "Can a brain tell a real world from a fake one?" |
| overview | 8.9 | 20.3 | 11.4 | Two worlds side by side, the missed reach |
| senses | 20.3 | 28.7 | 8.4 | Ten labeled sensor groups |
| process | 28.7 | 37.1 | 8.4 | Encoder rows, no recurrence |
| memory | 37.1 | 48.5 | 11.4 | The live meter, 49% climbing to 73% |
| actions | 48.5 | 59.9 | 11.4 | Five motor heads, three prediction heads |
| end | 59.9 | 65.9 | 6.0 | End card, 73% |

## Voiceover script

About 53 s of speech inside 65.9 s. Roughly 150 words per minute. The silences are
load-bearing, especially the one before the end card.

| Time | Chapter | Line |
|---|---|---|
| 0:02-0:07 | title | "One brain. Two worlds. Only one of them is real." |
| 0:09-0:20 | overview | "We copied its world and broke one rule: how things move. Watch the reach. In the fake world it falls a hair short. (beat) Nobody told it a fake existed." |
| 0:21-0:28 | senses | "Everything it knows arrives through ten senses. The flaw comes in as speed, a little wrong at every step." |
| 0:30-0:36 | process | "Those signals mix into patterns and pass straight through. Nothing here remembers anything." |
| 0:38-0:48 | memory | "Memory is where it lands. Before the miss: forty-nine percent. A coin flip. (beat) After: seventy-three. We set the bar at sixty-five before we looked." |
| 0:49-0:59 | actions | "This brain has two jobs. Stay alive, and expect what it will feel next. (beat) Survival alone is not enough. Expecting alone is not enough. It takes both." |
| 1:02-1:05 | end | "(pause 2s) Nobody asked it to notice. It noticed anyway." |

### Why the actions chapter carries the weight

The line "it takes both" is the reason this spec exists. The end card used to read "a
brain trained only to survive", which FINDINGS 10.8 contradicts: strip the
next-observation decoder and the same protocol reads 0.601 on the same machine,
under the 0.65 bar, against 0.730 with it. Prediction alone reads 0.573. The card has
been rewritten and the voice should land the same point a beat earlier, on the
chapter that already shows the three prediction heads.

Do not soften "not enough" into "not the whole story". The claim is a threshold claim
and the film shows the threshold.

### The one line with a shelf life

"Survival alone is not enough" rests on the adjudicated 0.601. On 2026-09-30 a
model-free arm trained to a longer budget read 0.717, above the bar, but its match
overshot the frozen window so it carries no mediation verdict (FINDINGS 10.8.1). If
the 300-to-450 bisection is ever run and the match lands, revisit this line before
re-recording anything else.

## Hume: voice design and delivery

Octave takes a natural-language voice description plus per-utterance acting
instructions. Paste the description once, then feed each line with its own
instruction rather than recording the script as one block, because the beats matter
more than the continuity.

**Voice description (paste as the voice prompt):**

```
A warm, unhurried narrator, mid-range, close-mic'd and intimate, as if leaning in to
show you something strange they found rather than presenting it. Gentle curiosity,
never authority. Slight natural breath and a touch of rasp on the low notes. No
broadcast polish, no trailer gravitas, no upward news-reader inflection. Comfortable
with silence; lets pauses sit without rushing to fill them. Around 150 words per
minute.
```

**Per-line acting instructions:**

| Line | Acting instruction |
|---|---|
| "One brain. Two worlds..." | Quiet and confiding, like the first sentence of a story told late at night. Full stop between each fragment, no list cadence. |
| "We copied its world..." | Matter of fact through "how things move", then lift slightly on "Watch the reach" as an invitation. Drop to almost a whisper on "a hair short". |
| "Nobody told it a fake existed." | Flat, no emphasis anywhere. The flatness is the point; let it sound like an aside. |
| "Everything it knows arrives..." | Patient and explanatory, a half step warmer. Slight stress on "ten". |
| "The flaw comes in as speed..." | Slow the last four words. "every step" lands like a drip. |
| "Those signals mix into patterns..." | Brisk, almost throwaway. This chapter is a corridor, not a room. |
| "Nothing here remembers anything." | Small pointed pause before "Nothing". |
| "Memory is where it lands." | Settle. This is the arrival; drop pitch and slow. |
| "forty-nine percent. A coin flip." | Deadpan. No disappointment, no drama. |
| "After: seventy-three." | The only genuine lift in the film. Warm, quietly pleased, not triumphant. |
| "We set the bar at sixty-five before we looked." | Dry, almost legalistic. This is the honesty line and it should sound like one. |
| "This brain has two jobs." | Fresh and clear, resetting attention. |
| "Stay alive, and expect what it will feel next." | Two distinct beats, equal weight. Do not throw away the second. |
| "Survival alone is not enough. Expecting alone is not enough." | Parallel and even, same melody twice. Resist building. |
| "It takes both." | Land it. Short, certain, then stop. |
| "Nobody asked it to notice. It noticed anyway." | Nearly to yourself. Slower than everything before it. Let the last three words decay into silence. |

**Mechanical notes.** Render each line as its own take so the gaps are yours to place
in the edit rather than Hume's to guess. Ask for two or three variants of the
"seventy-three" line and the closing line and pick in context, since those two carry
the film. Keep the written numerals as words in the input ("forty-nine", not "49"),
which is what the other film's recorded script did. Target the same deliverable shape
as film one: AAC stereo, matched to a 65.9 s picture.

## Suno: music

Palette continuity with *Detectable All Along* (felt piano, warm analog pad, soft tape
hiss, roughly 80 BPM) so the two films sound like one project, but this one is more
interior. Film one is a reveal; this one is a tour of a small mind. Less arc, more
patience, one genuine warming around the meter climb.

**Style prompt:**

```
intimate ambient documentary underscore, felt piano, warm analog pad, soft tape hiss,
slow pulsing arpeggio, 80 BPM, curious and patient, minimal and unhurried, lots of
space, instrumental
```

**Exclude styles:**

```
drums, percussion, trailer braams, epic orchestral, cinematic swell, vocals, choir,
EDM, drop, distorted bass, sidechain pumping
```

**Shape to aim for, against the picture:**

| Window | Chapter | Music |
|---|---|---|
| 0.0-8.9 | title | Single sustained pad. One felt-piano note, unresolved. Nearly nothing. |
| 8.9-20.3 | overview | A three-note motif enters, curious rather than tense. A low drone slides underneath at the missed reach. |
| 20.3-37.1 | senses, process | Slow arpeggio begins, one note per sensor group feel. Keep it flat and even; these are the explanatory chapters and the music should not editorialise. |
| 37.1-48.5 | memory | The only build. Pad swells and the piano motif returns in a fuller voicing as the meter climbs, warmth peaking as 73% locks. Warm, not epic, and no percussion arrives. |
| 48.5-59.9 | actions | Hold the warmth, thin the texture. The motif plays once more, simpler, while the voice does the work. |
| 59.9-65.9 | end | Strip to the opening motif, now resolved on the tonic it avoided at the start. Decay to room tone by 65.0 so the last second is silent. |

**Practical note on length.** Suno returns fixed-length generations rather than a 65.9 s
cut, so generate two or three takes of the palette, then build the timeline in the
editor: a bed from the calmest take, the swell from the warmest, and a hand-placed
decay at the end. Ask for an instrumental generation; there are no lyrics in this
film and a stray vocal line will fight the VO.

**Mix notes.** Duck the music about 6 dB under the voice. Let it back up in the three
marked beats (0:19, 0:47, 1:00) so the swell lands on the silences rather than under
the words. The film has no sound design of its own; if anything is added, a single
soft click on the missed reach at roughly 0:14 is enough, and nothing else.

## If this gets made

`assets/film/README.md` already says how to land it: encode the voiced cut the same
way as film one and replace `two-minds-web.mp4` in place, so the site and the player
pick it up without a markup change. Update that README's row for the file, which
currently records "no audio track", and the provenance line that says a voiced cut has
not been committed.
