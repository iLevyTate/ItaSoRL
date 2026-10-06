# Reproducing the results

Two levels, by cost. Everything below runs from the repository root.

## 1. Inspect the central readouts without training (minutes)

```bash
pip install -r requirements-dev.txt
python scripts/reproduce.py tables
```

This rebuilds, from the committed per-seed artifacts under `artifacts/`:

- `docs/RESULTS_MANIFEST.md`: which commit, configuration, seeds, surrogate, budget, fold
  partition, device, and trainer produced every committed result, and whether it is
  historical or corrected;
- `docs/GATE_TABLE.md`: every registered gate per run and fold partition, with L0 recomputed;
- `docs/CONTRAST_INTERVALS.md`: the registered margins as seed-paired intervals;
- `artifacts/corrected_verdicts.json`: the frozen decision rules applied to the corrected
  runs `C1` and `C2`, with integrity checks and the correction effect;
- `docs/paper_tables/*.tex`: the same tables for the manuscript (`scripts/build_paper_tables.py`).
  They stay local like the manuscript: `docs/paper_tables/` and `docs/paper/` are both
  gitignored, nothing in them is committed, and the audit renders the tables in memory.

It then runs `scripts/audit_stats_recheck.py`. **What the audit checks:** that the numbers
quoted in `README.md`, `docs/FINDINGS.md`, `docs/PAPER_OUTLINE.md`, `CITATION.cff`, the site,
and the B-v3 preregistration's gate entry equal values recomputed from the committed per-seed
artifacts, that the generated pages above are current, and that retired wording has not
returned to the pages it watches. **What it does not check:** it reruns no experiment, it does
not read the local LaTeX manuscript (use `python scripts/build_paper_tables.py --manuscript docs/paper`),
and it can only be as good as the per-seed values committed.

With the raw state dumps of a run (in the supplement, or `fullruns/<run>/states` after a run),
the per-seed readouts themselves are recomputed from the recurrent states, under both fold
partitions and with the behavior control:

```bash
python scripts/reproduce.py tables --dumps fullruns/corrected_l3_h8_wm/states
```

## 2. Retrain (hours of CPU per run)

```bash
python scripts/reproduce.py retrain C1            # print the commands
python scripts/reproduce.py retrain C1 --execute  # run them
```

Recipes are recorded for the corrected runs `C1` and `C2`, for the historical CPU runs (with
`--gae-bootstrap pre_transition`, the historical trainer, kept for exact reproduction), and
for the published GPU headline. Every other run's configuration is in
`docs/RESULTS_MANIFEST.md`.

Measured cost on a 4-vCPU cloud container (torch 2.14.1+cpu), one cell (three arms at one
drift and seed, 300 survival updates): about 22 CPU-minutes uncontended, of which survival
training is about 9 and predictor training about 7. A 20-cell run on 4 workers takes about
two to three hours; the corrected runs, which extend the survival arm to 450 updates for the
budget curve, take longer. On this container the original code at `4b6e1f3` reproduced a
committed historical CPU cell bit for bit (156 of 156 values), so CPU reruns can be compared
with the historical CPU runs exactly.

The readout-only follow-ups on a run's saved agents (L0 world-sample audit, policy-controlled
readouts, controlled persistence, control diagnostics, both texture comparators, population
readout validation) are printed by `retrain C1` after the training commands, followed by
the steps that need both runs (verdicts, budget curve, and the exploratory cross-run
replay). `scripts/revision/run_corrected_readouts.sh` runs the per-run chain unattended. On
the revision container the C1 chain took about 2 h 50 min at reduced priority alongside
the C2 training run; C1 and C2 themselves took about 3 h 20 min and 4 h 25 min on 4 workers
(C2 shared the CPU with the readouts).

## Frozen surrogates

`artifacts/surrogates/` holds every learned fingerprint the runs used, serialized with
`GMotion.to_npz` (load with `GMotion.from_npz`), with SHA-256 sums in `index.json`. They were
trained on CPU and match the CPU runs; the GPU-published runs trained G on CUDA, whose last
bits can differ, so their integrity gates regenerate G on CUDA.

## Anonymized supplement

```bash
python scripts/reproduce.py supplement --out dist/itasorl_supplement.zip
```

The archive holds the code, the tests, every committed artifact, the method and correction
documents, and, when present, the corrected runs' cells, saved agents, and state dumps, with a
`SHA256SUMS` file. Author names, handles, e-mail addresses, ORCID identifiers, and local paths
are scrubbed from text files; the identity terms are read at build time from `CITATION.cff` and
the git remote, so the builder names no one. The build fails if any identity term survives.
The project and package name (`itasorl`) is not anonymized; rename it if a venue requires that.
Inside the extracted archive the test suite and the full scalar audit pass.

## Corrections and provenance

`docs/CORRECTIONS.md` is the dated record of what was wrong, what changed, and what each
change does not establish. `docs/REVISION_2026-10.md` holds the claim decisions and status.
