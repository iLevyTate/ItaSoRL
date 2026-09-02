## Learned User Preferences

- Avoid en dashes and em dashes in docs, Colab wording, and generated copy; use plain punctuation instead.
- All markdown a person reads (posts, README, site/Colab copy, outreach, FINDINGS, specs, preregistration prose) follows Strip the AI Tells. Keep numbers, tables, and `[0:00]` cue lines. No em/en dashes, no "not just X it's Y," no participial tails, no Moreover/Furthermore/Additionally, no rule-of-three padding, no summary ending. Banned filler words stay banned. Do not swap in a synonym. Scientific "robust" (a result that holds under a named check) stays; marketing "robust" does not.
- Keep outreach and article drafts unpublished unless explicitly asked; PRs should not publish post content.
- Visualizations and film clips should be understandable to a general audience, not only researchers.
- For wide and vertical clips, distribute on-screen information to fill empty space in a design-focused way rather than leaving sparse unused regions.
- Prefer uploading media (videos, gifs) without requiring the accompanying post copy to go live.
- Prefer LinkedIn and YouTube Shorts for distribution; keep X/Twitter presence low-interaction when used.
- For each film clip, draft channel-specific posts (LinkedIn, X, YouTube Shorts, and relevant Reddit communities) in the user's first-person voice using writing-social-voice plus Strip the AI Tells; open with the claim, relate the clip to ML/AI, and do not use identical cross-posts. Treat clips as a short series posted over time, and include a brief refresher of earlier clips. The 90s full film is the watch-this-one post (voice and music in). LinkedIn uses uneven paragraphs, not one idea per line.
- In Colab setup, let the user choose hardware; do not default to T4.
- Prefer plain-English explanations and branding language suitable for public posts and designer handoff.
- For loop gifs and looping film clips, animations should start and end on the same frame so they loop cleanly.
- Prefer cinematic clip animation that shows the agent or creature in action, not only abstract graph motion; for story-arc deliverables, provide a continuous VO script and second-by-second event timeline so VO and music can be recorded in one take.
- Deliver VO and music as separate tracks. For Suno, start with `[80 BPM]` (or the user-set BPM), then `[0:00] [Role: what the music does, how it sounds, what is on screen]` lines; default style is felt piano / warm pad / tape hiss, but a named STYLE should be honored so other sounds can be explored on the same clocks; keep each music prompt under 1000 characters. For Hume VO, output one pasteable text block with `[pause]` / `[long pause]` whose spoken length matches the video duration (VO may exceed 1000 characters).

## Learned Workspace Facts

- ITASORL studies when RL agents internally represent environment distinctions they were not trained to care about.
- Primary end-to-end run output belongs under `fullruns/` (see `itasorl/results_io.py`).
- Colab GPU workflow lives in `notebooks/colab_gpu.ipynb`; runs should persist results to Google Drive and locally.
- Public site is GitHub Pages at `https://ilevytate.github.io/ItaSoRL/`; CI mirrors `main` onto `gh-pages`.
- Film clip deliverables live under `assets/film/clips/` (wide, vertical, and loop gif variants).
- Interactive player and clip capture tooling live under `viz/player/`.
- Reusable agent prompt for clip music lives at `artifacts/clip_audit/post-materials/suno-music-agent-prompt.md`.
- The site's `index.html` is generated: edit `index.template.html`, then run `python scripts/build_index.py`. CI (`scripts/audit_stats_recheck.py`) fails if the rendered page is stale.
- Film one is *Detectable All Along* (the 90 s story). Its voiced master, `ItasorlOverview.mov` (DaVinci Resolve export, 1920x1080, AAC), was released 2026-08-24 and lives on the owner's Google Drive, not in git. The site serves the 720p web encode `assets/film/detectable-all-along-web.mp4`; the captions-only cut plays live at `viz/player/`.
- Film two is *Two Minds* (the labeled brain film). Renderer: `viz/player/brain/`; rebuild: `viz/player/capture/build-two-minds.ps1`; site encode: `assets/film/two-minds-web.mp4`. No voiced cut of Two Minds is committed; if one is produced, replace the web encode in place. Provenance for every film file: `assets/film/README.md`.
