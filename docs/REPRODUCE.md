# Reproducing the results

Two levels, by cost. Everything below runs from the repository root.

A shallow clone of the repository is enough for both levels:

```bash
git clone --depth 1 <repository URL>
```

The history carries video, gif and PDF renders that the current tree no longer holds, so a
full clone is several times the size of the tree. Checking the commit hashes that
`docs/RESULTS_MANIFEST.md` records needs the full history (`git fetch --unshallow`), and
some of those commits are not ancestors of `main`; the manifest says where each of them
lives.

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

It then runs `scripts/audit_stats_recheck.py`. The script prints the count (2933 on
2026-10-13, after the balanced sensory-echo rows). **What the audit
checks:** it recomputes each summary statistic it guards from the committed per-seed
artifacts and compares it with a copy of the quoted value kept as a constant in the script.
Those constants mirror numbers quoted in `README.md`, `docs/FINDINGS.md`,
`docs/PAPER_OUTLINE.md`, `CITATION.cff`, and the B-v3 preregistration's gate entry. The
script reads the quoted numbers themselves only in a few passages of `docs/FINDINGS.md` (section 17, section
10.1.1, and selected passages of sections 13 and 14), where it requires them to equal values
rendered from the artifacts. It also checks that the generated pages above and the site's
`index.html` are current, and that retired wording has not returned to the pages it watches.
**What it does not check:** a number edited by hand in `README.md`, `docs/PAPER_OUTLINE.md`,
`CITATION.cff`, or a FINDINGS passage the script does not read passes the audit; it reruns no
experiment; it does not read the local LaTeX manuscript (use
`python scripts/build_paper_tables.py --manuscript docs/paper`); and it can only be as good as
the per-seed values committed.

The raw state dumps are not in git or in the supplement; they exist only after a run.
Retrain with `reproduce.py retrain C1` (section 2), and the dumps land in
`fullruns/<run>/states`. With them, the per-seed readouts themselves are recomputed from the
recurrent states, under both fold partitions and with the behavior control:

```bash
python scripts/reproduce.py tables --dumps fullruns/corrected_l3_h8_wm/states
```

## 2. Retrain (hours of CPU per run)

The two stacks a bit-for-bit claim depends on are pinned in
`constraints-cpu-corrected.txt` (Python 3.11.17, torch 2.14.1+cpu, numpy 2.4.6,
scikit-learn 1.9.1, scipy 1.17.1) and `constraints-workstation.txt` (Python 3.13.2,
torch 2.7.0+cu126, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.15.3). A cell written
from 2026-10-12 on records the stack it actually ran on.

Erratum, 2026-10-14. NumPy 1.26.4 supports Python 3.9 through 3.12. The PyPI
release has wheels for cp39 through cp312 and a source tarball, and no cp313
wheel, so `pip install numpy==1.26.4 --only-binary=:all:` on Python 3.13 has
nothing to fetch. The workstation copy is a local meson build, wheel tag
`cp313-cp313-win_amd64`, compiled with MSVC 19.29.30159 and Cython 3.0.12,
installed by pip for `C:\Python313\python.exe` into the user site-packages.
The installed tree (1209 files) hashes to
`675bd7f1f96e6fe6fba53bf7db13ddd50aba480cd34d44dce2ada19af9472c2a`.
The GPU numbers stay tied to that build. Swapping in a newer NumPy and
keeping a bit-exact claim against the recorded cells is not supported.
The original stack lines in the run artifacts are left as written.

Checked on 2026-10-14 from clean environments. A fresh Python 3.11 virtual environment
resolves `requirements.txt` under `constraints-cpu-corrected.txt` to the pinned torch
2.14.1+cpu, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1, and `scripts/audit_stats_recheck.py`
passes from it. On a fresh Python 3.13 environment `pip install numpy==1.26.4 --only-binary=:all:`
finds no wheel; on Python 3.12 the same command installs the official wheel.

```bash
python scripts/reproduce.py retrain C1            # print the commands
python scripts/reproduce.py retrain C1 --execute  # run them
```

The matched-world confirmations are readout replays on saved agents, not retrains:

```bash
python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_wm --objective survival --mortal on
python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_touch --objective touch --mortal off
```

If either bundle is missing from the supplement, README_SUPPLEMENT.md says so. Absence is not inclusion.

Recipes are recorded for the corrected runs `C1` and `C2`, for the historical CPU runs (with
`--gae-bootstrap pre_transition`, the historical trainer, kept for exact reproduction), and
for the published GPU headline. Every other run's configuration is in
`docs/RESULTS_MANIFEST.md`.

Measured cost on a 4-vCPU cloud container (torch 2.14.1+cpu), one cell (three arms at one
drift and seed, 300 survival updates): about 22 CPU-minutes uncontended, of which survival
training is about 9 and predictor training about 7. A 20-cell run on 4 workers takes about
two to three hours; the corrected runs, which extend the survival arm to 450 updates for the
budget curve, take longer. On this container the original code at `4b6e1f3` reran a committed
historical CPU cell and reproduced its recorded values, including survival target 0.66570 and
engagement return -0.65194, so CPU reruns can be compared with the historical CPU runs. That
rerun cell was not saved. A second rerun on 2026-10-11, saved this time, reproduces all 157
recorded values of both seed-0 cells of that run exactly
(`artifacts/repro_check/per_value_4b6e1f3_l3_h8_wm_cpu_s0.json`; `docs/CORRECTIONS.md`,
2026-10-10 and 2026-10-11).

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
trained single-threaded on CPU with the recipe the CPU runs used.
`tests/test_reproducibility_package.py` checks each file against its SHA-256 sum, and checks
that a fresh single-threaded CPU retrain of the headline instance (hidden 8, G seed 0) gives
the same outputs as the file within numpy's `allclose` tolerance (absolute 1e-6, relative
1e-5) on five random inputs. It does not check that the retrain is bit-identical, and on
another CPU stack or thread count the last bits can differ. The GPU-published runs trained G
on CUDA, whose last bits can also differ, so their integrity gates regenerate G on CUDA.

## Anonymized supplement

```bash
python scripts/reproduce.py supplement --out dist/supplement.zip
```

The archive holds the code, the tests, every committed artifact (the per-seed results, and
the corrected runs' cell records under `artifacts/corrected_runs/`), and the method and
correction documents, with a `SHA256SUMS` file. It also holds the run folders listed in
`RUN_DIRS` in `scripts/reproduce.py` that are present under the runs root (`--runs-root`,
default `fullruns/`):

- `fullruns/corrected_l3_h8_wm` and `fullruns/corrected_l3_h8_nowm`, the whole run folders of
  the corrected runs `C1` and `C2` (cells, saved agents and state dumps). They exist only on the
  machine where those runs were trained; built anywhere else, the archive carries no saved
  agents or state dumps of `C1` and `C2`, which then exist only after a retrain (section 2).
- `fullruns/l3_h8_heldout/agents`, the saved agents of the historical hidden-8 held-out run.
- `fullruns/l3_h8_gseed2_gpu/agents`, the saved agents of the hidden-8 seed-2 run.

The console and `README_SUPPLEMENT.md` name every folder that was absent. The saved agents
are not committed to git; for review they ship only in this supplement, and a DOI archive
record for them comes only after the review decision.

Author names, handles, e-mail addresses, ORCID identifiers, and local paths (Windows, Git Bash,
WSL, macOS and Linux forms) are scrubbed from text files; the identity terms are read at build
time from `CITATION.cff` and the git remote, so the builder names no one. The build fails if
any identity term survives in a text file. The check also scans every binary member, the
decompressed members of the agent and array files included, and fails on any image whose exact
bytes are not on the reviewed list in `scripts/reproduce.py` (`REVIEWED_IMAGES`), because a
name drawn into a figure is pixels that no pattern can find. A new or redrawn figure ships only
after someone views it and adds its SHA-256. The project and package name is replaced by a
neutral one in text and in file names (`anonpkg` by default; `--anon-name` picks another, and
`--keep-name` ships the original name, so that archive is not anonymous).

The titles of the project's public films become Film One and Film Two in text and in file
names, and a release date written after "released" reads `<date>`. The builder reads the titles
at build time from the names of the served film files, so it spells none of them. The outreach
and film-production specs are left out, and so is any figure a run script drew inside a run
bundle (under `artifacts/goal_stakes/` or `fullruns/`), since no one has viewed those for
names. A Markdown link to the repository placeholder or to a file the archive does not carry
becomes plain text. `PROVENANCE.md` lists every commit to each preregistration and spec in
the archive, oldest first, with UTC committer dates and times, short hashes and a count per
file, read from git at build time. Edits made on side branches are listed under their own
hashes, and a merge commit is listed when its version of the file differs from every parent.
No names, e-mail addresses or commit messages are included. It is left out, with the reason
in `README_SUPPLEMENT.md`, when the build runs outside a git checkout or from a shallow clone.
`--require-runs` stops the build with nothing written when a run bundle in `RUN_DIRS` is
absent.

Inside the extracted archive the test suite and the full scalar audit pass. The audit skips its
one check on the film VO and music spec, which the archive leaves out.

## Corrections and provenance

`docs/CORRECTIONS.md` is the dated record of what was wrong, what changed, and what each
change does not establish. `docs/REVISION_2026-10.md` holds the claim decisions and status.
