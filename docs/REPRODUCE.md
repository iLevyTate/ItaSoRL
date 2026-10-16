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
lives. Inside the anonymized supplement those hashes are pseudonyms (see "Anonymized
supplement" below), and the archive carries no history to check them against.

## 1. Inspect the central readouts without training (minutes)

```bash
pip install -r requirements-dev.txt
python scripts/reproduce.py tables --check
python scripts/reproduce.py tables
```

The first command compares the committed pages below with what the builders produce from the
committed per-seed artifacts, rewrites nothing, and runs the audit on the pages as shipped. It
prints `committed tables and pages: current` or `STALE`, and exits 3 when a page is stale. The
audit runs every builder's `--check` itself, so it fails on a stale page as well; the exit code
is still 3, and the audit's output names any other check that failed. The
second command rebuilds the pages, then runs the audit. It also exits 3 when a page was stale
before the rewrite, even though the audit then passes on the rewritten copy, so exit 0 from
either command means the shipped pages were already current (changed 2026-10-16; before that
the rebuild exited 0 after restoring a stale page). Any other non-zero exit is a failing
builder or audit check.

The rebuild writes, from the committed per-seed artifacts under `artifacts/`:

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

It then runs `scripts/audit_stats_recheck.py`. The script prints the count (3137 on
2026-10-16, after the critical-review checks). **What the audit
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
experiment; it does not read the local LaTeX manuscript; and it can only be as good as the
per-seed values committed.

**The manuscript check.** `python scripts/build_paper_tables.py --manuscript docs/paper` reads
the local LaTeX source and reports items to review. It compares these numbers with the
committed record: the C1 and C2 headline means and intervals of
`artifacts/corrected_verdicts.json` (each required interval quoted right after its own mean,
and no headline interval after a different number), the C1 and C2 per-seed lists, every
three-decimal number in the paragraph headed "The corrected primary run", the rule-ledger tally
in any sentence that counts rule-and-run pairs (against `docs/rule_ledger.json`), and the row
and set counts in any sentence that counts the gate table's distinct sets of per-seed values
(against `artifacts/gate_table.json`). It also checks that the generated tables are `\input`
and current, and flags retired wording and an unlabeled pre-correction headline. No other
number in the manuscript is compared with an artifact, the hand-typed appendix tables
included; those are traced by hand through the claims appendix, which names the artifact
behind every number that enters a decision, a gate or a headline comparison. "0 items to
review" means only that the checks above found nothing.

The raw state dumps of `C1` and `C2` are not in git or in the supplement. Where the saved agents
are at hand (a local run folder, or the full build made on the machine that trained the runs),
the dumps rebuild from them without retraining, and every pool is checked against its recorded
target:

```bash
python scripts/regenerate_state_dumps.py --run-dir fullruns/corrected_l3_h8_wm --out-dir fullruns/corrected_l3_h8_wm/states
```

The review build carries no `C1` or `C2` agents, so from it the dumps need a retrain:
`reproduce.py retrain C1 --execute` (section 2), after which they land in
`fullruns/<run>/states`. With the dumps, the per-seed readouts themselves are recomputed from
the recurrent states, under both fold partitions and with the behavior control:

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

### Supported-stack replay route

Measured on 2026-10-16. Bit-exact reproduction holds only on the stack and machine that wrote a
result. To replay a saved-agent readout elsewhere, use the pinned CPU stack and expect the
differences below, not equality.

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311 torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv311 -r requirements.txt -c constraints-cpu-corrected.txt
.venv311/Scripts/python scripts/run_l0_audit.py --agents-dir fullruns/corrected_l3_h8_wm/agents --out <dir>/l0.json
.venv311/Scripts/python scripts/run_control_diagnostics.py --run-dir fullruns/corrected_l3_h8_wm --out <dir>/cd.json --workers 2
.venv311/Scripts/python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_wm/agents --cells-dir fullruns/balanced_confirmation_wm/cells --out-dir <dir>/echo_b1 --balanced-seed-base 910000 --objective survival --mortal on --drifts 0.45 --device cpu --no-abort
```

Stack used: Python 3.11.16, torch 2.14.1+cpu, NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1, on
the owner's Windows workstation CPU, code at `8664583`. The committed C1 readouts were written on
the same pins in a Linux CPU container (Python 3.11.17); B1 was written on the workstation GPU
stack (Python 3.13.2, torch 2.7.0+cu126, the local NumPy 1.26.4 build).

| readout | committed artifact | equal values | largest per-seed difference | largest difference in a reported mean |
|---|---|---|---|---|
| L0 audit, C1 | `artifacts/l0_audit/corrected_l3_h8_wm.json` | 1136 of 1376 | 0.0025 (AUROC) | 0.0004 (pair mean); TOST p within 0.0055 |
| control diagnostics, C1 | `artifacts/control_diagnostics/corrected_l3_h8_wm.json` | 3 of 36 aggregate means | 0.028 (MLP control), 0.017 (sequence GRU) | 0.0041 (sequence GRU); 0.0007 for every linear and ridge readout; target 0.7326 against 0.7330 |
| sensory echo, B1, GPU to CPU | `artifacts/balanced_confirmation/sensory_echo_b1.json` | 22 of 30 cells pass the 0.0005 integrity gate | 0.0058 | 0.0005; the deciding survival `resid_obs` mean is 0.7608 on both |

On B1 eight cells miss the integrity gate by 0.0005 to 0.0012, so the tool withholds its decision,
as it must for a stored run read on another device: no reading is re-adjudicated on a replay. No
replayed value crosses a threshold that a committed reading relies on. The C1 state dumps were not
on this machine, so the control-diagnostics dump comparison did not run.

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
python scripts/reproduce.py supplement --profile review --require-runs   # dist/supplement-review.zip
python scripts/reproduce.py supplement --out dist/supplement.zip          # the full build
```

Both builds hold the code, the tests, every committed artifact (the per-seed results, and the
corrected runs' cell records under `artifacts/corrected_runs/`), and the method and correction
documents, with a `SHA256SUMS` file. They differ in the run bundles.

The review build (`--profile review`) is the one to submit. TMLR accepts at most 100 MB of
supplementary material, and the full build was 501 MB on 2026-10-16. The review build carries
only `fullruns/l3_h8_heldout/agents` and `fullruns/l3_h8_gseed2_gpu/agents`, the saved agents
from which the direction-of-rearing and world-sample readouts replay, and no state dumps. The
build fails, and removes the archive, when it exceeds 100 MB (100,000,000 bytes); a test
build on 2026-10-16 came to about 40 MB. Its `README_SUPPLEMENT.md` lists the bundles it leaves
out and says that replaying their readouts needs a retrain.

The full build (the default profile) also holds the run folders listed in `RUN_DIRS` in
`scripts/reproduce.py` that are present under the runs root (`--runs-root`, default
`fullruns/`):

- `fullruns/corrected_l3_h8_wm` and `fullruns/corrected_l3_h8_nowm`, the whole run folders of
  the corrected runs `C1` and `C2` (cells and saved agents; state dumps rebuild from the
  agents). They exist only on the
  machine where those runs were trained; built anywhere else, the archive carries no saved
  agents or state dumps of `C1` and `C2`, which then exist only after a retrain (section 2).
- `fullruns/l3_h8_heldout/agents`, the saved agents of the historical hidden-8 held-out run.
- `fullruns/l3_h8_gseed2_gpu/agents`, the saved agents of the hidden-8 seed-2 run.
- `fullruns/balanced_confirmation_wm` and `fullruns/balanced_confirmation_touch`, the run folders
  of the matched-world confirmations `B1` and `B2` (cells, saved agents, frozen normalizers and
  the matched-world sensory-control output), from which the matched gates replay (section 2).
- `fullruns/corrected_l1_heldout`, the run folder of `L1-CORRECTED` (20 cells, 60 saved agents
  and 60 state dumps), from which the stream and surprise readouts replay.

The console and `README_SUPPLEMENT.md` name every folder that was absent. The saved agents
are not committed to git. The review build carries only the held-out and seed-2 agents; the
others ship only in the full build and are otherwise kept by the authors. A DOI archive record
for them comes only after the review decision.

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
`--require-runs` stops the build with nothing written when a run bundle of the chosen profile
is absent.

Commit IDs and recorded digests are replaced too. A commit hash is a search key: on 2026-10-16
one full hash in the built archive found the public repository in a single code search. A
64-character SHA-256 digest recorded in an artifact (an agent, surrogate or fold hash) is an
even more specific one, since the same string sits in the public repository. Every hex token of
7 to 40 characters, in upper or lower case, that is the prefix of a commit in the repository's
object store becomes a keyed pseudonym of the same length and case, in text, in `PROVENANCE.md`
and in file names. So does a token in a commit field (a JSON key naming a commit, a head or a
fix, or the word "commit" before it), even one that resolves to no local commit. One commit
reads the same everywhere in the archive, a short hash stays the prefix of its full form, and
the pseudonyms sort as the commits do, so the results manifest's own check and the audit pass
inside the archive as they do in the repository.

Every 64-character hex digest in a text member becomes a keyed pseudonym too, the same one
wherever that digest appears, and a hex token of 12 or more characters that is the prefix of
such a digest becomes the same prefix of its pseudonym. A run of one or two repeated characters
is padding and stays. The rule covers digests of things the archive does not carry: the
workstation NumPy tree hash in section 2 and in `constraints-workstation.txt` reads as a
pseudonym inside the archive, and its real value is in the build's map. `SHA256SUMS` is the
exception: it keeps the real digest of every shipped file, so it still verifies the bytes, and
anyone who hashes a shipped binary such as a saved agent can still search for the result. A
check that compares a digest it computes from a shipped file with the one an artifact records
looks the computed digest up in `DIGEST_PSEUDONYMS.json`, which the archive carries
(`archive_digest()` in `scripts/reproduce.py`). That table is keyed by the SHA-256 of each real
digest, so it gives back none of them. Hex runs inside a number, such as the decimals of
`-9.765625e-05`, are not IDs and stay. Any other hex string stays too, such as a 16-character
configuration fingerprint, unless it is a commit prefix, a commit-field token or the prefix of
a replaced digest.

The key is written to `dist/supplement-commit-key.txt` and stays local. The key alone does not
give back a build's commit pseudonyms, because they are handed out in the sorted order of every
commit in the object store, and a later commit shifts many of them. Each build therefore writes
a map beside its archive (`dist/supplement-review.zip` gets
`dist/supplement-review.commit-map.json`) with the archive's SHA-256, the commit list it drew
from, and the pseudonym of every commit spelling, commit-field token and digest it replaced.
Keep that map with the submitted archive for the camera-ready version.

The build fails closed. It deletes the archive it wrote and exits non-zero when any shipped
name, text or binary member still holds an identity term or an unreviewed image, a token that
resolves to a commit, or a real digest that should have been replaced, and when a pseudonym
changes a decimal number or leaves a JSON member that no longer parses or holds a different
number. The commit scan only knows the commits in the local object store, so the command writes
nothing and refuses to build an anonymized archive outside a git checkout, from a shallow,
partial or single-branch clone, when a branch tip on the remote is missing from the object
store, or when a commit field in a shipped file names a commit the store lacks. Build from a
full checkout with every branch fetched (`git fetch --unshallow`, then `git fetch`). When the
remote cannot be reached, the build prints that the branch-tip check did not run and goes on.

Inside an extraction of the 2026-10-16 review archive, built before digests were replaced, the
full scalar audit passes, with its one check on the film VO and music spec skipped and counted,
and `python scripts/reproduce.py tables` passes (`docs/SUBMISSION_CHECK_2026-10-16.md`, section
2a). The full test suite has not yet passed there. The tests of the social-preview card and of
the web player's beat and brain data skip inside an archive, because it leaves those files out,
and still fail in a git checkout when a file is missing. One test does not skip yet:
`tests/test_optional_readouts.py` reads the optional-readout files under `docs/optional/`,
which the archive does not carry, and fails inside it.

## Corrections and provenance

`docs/CORRECTIONS.md` is the dated record of what was wrong, what changed, and what each
change does not establish. `docs/REVISION_2026-10.md` holds the claim decisions and status.
