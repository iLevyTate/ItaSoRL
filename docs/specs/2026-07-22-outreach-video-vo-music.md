# Outreach video: voiceover script + music pacing

Status (2026-09-02): delivered. The voiced master (`ItasorlOverview.mov`,
DaVinci Resolve, 1920x1080, 24 fps, AAC stereo, 89.8 s) was released on
2026-08-24 under the public title *Detectable All Along*. The site serves a
720p web encode, `assets/film/detectable-all-along-web.mp4`; provenance and the
ffmpeg lines are in `assets/film/README.md`. The script and pacing below are the
ones that were recorded. (Note 2026-10-19: the recorded audio departs from the
script's wording in three places, "born knowing absolutely nothing", "one tiny
rule" and no "that violet ring"; the captions in
`assets/film/detectable-all-along-web.vtt` follow the audio.)

Companion to `viz/out/detectable-not-learned.mp4` (90s, beats from
`viz/player/beats.json`). The VO never reads on-screen text verbatim - captions
carry the claims, the voice carries the story. Numbers spoken must match the
published displays exactly: 99%, 50%, 73%.

Target delivery: warm, curious, unhurried - a friend showing you something
strange, not a trailer announcer. ~150 wpm. Total spoken time ~80s inside 90s;
the marked pauses are load-bearing, not optional.

## Voiceover script

| Time | Beat | Line |
|---|---|---|
| 0:00-0:10 | creature | "This little creature was born knowing nothing. No rules, no lessons. It just lives here, feeling its way around with those little rays." |
| 0:10-0:19 | trick | "So we ran an experiment. We copied its whole world and changed exactly one rule - how well its feet grip the ground. (beat) Can you tell which one is the fake?" |
| 0:19-0:25 | reveal | "(pause 1s) It's the one on the right. It slides a hair too far - every single step." |
| 0:25-0:35 | physics | "That's the whole difference. The real world follows rules. The copy just guesses - close... but wrong, every step." |
| 0:35-0:45 | observer | "So, can anyone tell? A watcher on the outside, who knows the real rules - (let gauge sweep) - catches the fake ninety-nine percent of the time." |
| 0:45-1:00 | nocare | "But here's the strange part. Read the creature's own mind - that violet ring - and ask it: is your world fake? (pause 1.5s while needle wobbles) Fifty percent. A coin flip. It has no idea. It never had a reason to care." |
| 1:00-1:20 | survival | "So we gave it one. In the fake world, the broken physics makes food hard to catch. Now, telling real from fake is the difference between eating and starving. (let gauge climb) Seventy-three percent. Nobody taught it. Nobody rewarded it. It learned because its life depended on it." |
| 1:20-1:30 | close | "(pause 1s) The fake was detectable all along. The creature only noticed... when survival made it matter. (silence to end)" |

### Corrected lines for a re-voice (2026-10-16)

The table above is what the released master says, and four of its lines no
longer match the record. "It has no idea" and "the creature only noticed" claim
awareness and use; removing the decoded direction while the creature acts
changes its behavior no more than a sham removal (FINDINGS 23.3). "It learned
because its life depended on it" is withdrawn: at matched input a mind trained
only to predict reads 72% against the forager's 73% (FINDINGS 17.6). "Fifty
percent" is now 52% on the corrected trainer. The live captions-only cut carries
the corrected wording; a re-voiced master would use these lines over that picture. (Note
2026-10-27: the close line and the end card read "only after a life there" until today. Brains
raised only in the real world, then shown the fake, read 75% too (FINDINGS 17.5.2), so the fake
does not show only after a life in the copy; both now read "only in a mind that had lived".)

| Time | Beat | Line |
|---|---|---|
| 0:45-1:00 | nocare | "But here's the strange part. Read the creature's own memory, that violet ring, and ask which world it is in. (pause 1.5s while needle wobbles) Fifty-two percent. A coin flip. Nothing there to read." |
| 1:00-1:20 | survival | "Now let it grow up in the copy. It forages there day after day, and nobody rewards it for the fake. (let gauge climb) Seventy-three percent. The fake shows up in its memory anyway. A mind trained only to predict what it senses, fed the same senses, reads seventy-two." |
| 1:20-1:30 | close | "(pause 1s) The fake was detectable all along. It showed only in a mind that had lived... and it never changed what the creature did. (silence to end)" |

Delivery notes:
- "ninety-nine percent" lands AS the gauge hits 99% (~0:40), not before.
- "Fifty percent" lands after the wobble settles (~0:52). The 1.5s pause while
  the needle flickers is the emotional center of the film - do not fill it.
- "Seventy-three percent" lands as the sweep locks (~1:14). The sweep runs
  1:04-1:14, so the two sentences before it should finish by ~1:12.
- Close line is slow, almost spoken to yourself. Last 2s are silent over the
  URL card.

## Music pacing

Palette: felt piano + warm analog pad + soft tape hiss. Minimal, honest,
slightly toy-like - matches the Soft Substrate look. ~80 BPM. No trailer
braams, no epic drums; the biggest dynamic moment is a HEARTBEAT, not a drop.

| Time | Beat | Music |
|---|---|---|
| 0:00 | creature | Sparse felt-piano motif (3-4 notes, majorish, curious) over a warm pad. No percussion. Room to breathe. |
| 0:10 | trick | Motif turns into a question: same notes, unresolved ending. A low drone fades in underneath - something is off. Soft ticking enters (a clock, not a beat). |
| 0:19 | reveal | One accented low piano note on "the right", then everything holds - pad only, ticking continues. |
| 0:25 | physics | Ticking becomes a steady metronome (= the rules). A faintly detuned echo of each tick trails it (= the copy's guess). Subtle, almost subliminal. |
| 0:35 | observer | Rising arpeggio synced to the gauge sweep (0:36-0:40), resolving on a bright, confident chord exactly at 99%. First moment of fullness. |
| 0:45 | nocare | Bottom drops out. Kill the arpeggio and metronome; leave only the drone and a thin, wavering high tone that mirrors the needle wobble. Near-silence under "a coin flip." Coldest point of the piece. |
| 1:00 | survival | A soft kick enters at heartbeat tempo (~60 BPM feel against the 80 BPM grid) - life is now at stake. Layers rebuild one at a time across the gauge climb (1:04-1:14): pad swell, piano motif returns in a fuller voicing, warmth peaks exactly as 73% locks at 1:14. This is the emotional high - warm, not epic. |
| 1:20 | close | Everything strips back to the opening piano motif, now resolved (ends on the tonic it avoided at 0:10). Decays naturally; silence by 1:28. End on room tone. |

Mix notes:
- Duck music ~6 dB under VO; let it back up in the marked pauses (0:20, 0:52,
  1:12-1:14) so the numbers and reveals get the swell, not the words.
- The three gauge moments (99 / 50 / 73) are the only sync-critical hits:
  bright chord at 0:40, near-silence at ~0:52, full warmth at 1:14.
- Master for phone speakers: keep the drone's energy above ~120 Hz or it
  disappears on mobile - and this film ships to social.
