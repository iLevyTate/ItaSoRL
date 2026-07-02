# Ralph - Senior Developer + Research Loop

A [Ralph Wiggum loop](https://awesomeclaude.ai/ralph-wiggum): feed Claude Code
the same prompt over and over in a **fresh context each time** and let it
iterate toward a goal. Progress accumulates in **files and git history**, not in
the model's context window.

Each iteration the agent **re-reads current experiment results**, then either fixes
one bug or advances one **ready** research/next-step item (tests, tooling, docs).
Long GPU sweeps require human approval.

## What's here

| File | Purpose |
|------|---------|
| `PROMPT.md` | Persona, priorities, per-run procedure, and stop signal. The brain. |
| `ralph.sh` | The loop. Calls Claude with `PROMPT.md`, stops on completion / stall / max-iterations. |
| `EXPERIMENT_STATUS.md` | Living snapshot of latest `fullruns/`, canonical artifacts, open questions. |
| `NEXT_STEPS.md` | Prioritized research queue (`[ready]` / `[blocked]` / `[done]`). |
| `COLAB.md` | Colab playbook: RUN_PROFILE presets, resume, post-run compare. |
| `BACKLOG.md` | Bugs and gaps. Ralph reads this for P0-P3 work. |
| `JOURNAL.md` | Append-only record of what each run did. |
| `logs/` | Per-iteration stdout, timestamped. |

## The command to run locally

From the repo root, on a feature branch (the script refuses to run on
`main`/`master`):

```bash
chmod +x ralph/ralph.sh          # first time only
MAX_ITERATIONS=30 ./ralph/ralph.sh
```

That's it. Watch it work; logs stream to your terminal and to `ralph/logs/`.

### Tuning knobs (all optional env vars)

```bash
MAX_ITERATIONS=30 \   # hard cap; the loop can never run forever (default 20)
STALL_LIMIT=2 \       # stop after N runs with no new commit (default 2)
SLEEP_BETWEEN=3 \     # seconds between iterations (default 3)
./ralph/ralph.sh

DRY_RUN=1 ./ralph/ralph.sh    # print the command without calling Claude
```

### How it stops

The loop exits on the first of:
1. **Completion** - the agent prints `RALPH_COMPLETE` (no actionable bugs or `[ready]` research items).
2. **Stall** - `STALL_LIMIT` iterations in a row produce no new commit.
3. **Cap** - `MAX_ITERATIONS` reached.

Stop it yourself any time with **Ctrl-C**; committed work is already safe in git.

## Raw one-liner (no script)

If you want the bare Ralph in your shell:

```bash
for i in $(seq 1 20); do
  claude -p "$(cat ralph/PROMPT.md)" --dangerously-skip-permissions --permission-mode acceptEdits
done
```

The wrapper script is preferred - it adds the completion sentinel, stall
detection, the `main`-branch guard, and per-iteration logs.

## Safety notes

- Always run on a **feature branch**, never `main`. The script enforces this.
- `--dangerously-skip-permissions` lets Claude run unattended. It only ever
  executes the reviewed `PROMPT.md`, but treat the branch as untrusted until you
  review the diff: `git log --oneline` then read each commit.
- Each fix is its own commit, so anything bad is easy to revert.
