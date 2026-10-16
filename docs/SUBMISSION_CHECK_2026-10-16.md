# Submission check, 2026-10-16

The submission copies were built and checked after PR #153 merged (`afe6ee5`, tag `v1.4.1`).
Experiment repo at `afe6ee5`; paper repo at `7fa50fa`. Machine: the owner's Windows 11
workstation. Python 3.13.2 for the builders; uv 0.12.7; MiKTeX 25.12 (pdfTeX 4.23), latexmk 4.88.

## 1. Repository checks at `afe6ee5`

| command | result |
|---|---|
| `python scripts/reproduce.py tables` | pass, all 2933 audit checks; see the note under the table |
| `python scripts/build_index.py` | `index.html` unchanged apart from line endings |
| `python scripts/build_paper_tables.py --manuscript docs/paper` | 0 items to review; see the note under the table |
| `python scripts/audit_stats_recheck.py` | all 2933 checks passed |

Correction, 2026-10-16, after the critical review. At `afe6ee5` the `tables` command rebuilt
any stale page before the audit and exited 0 either way, so its pass did not show that the
shipped pages were current. Rechecked the same day on a `git archive` export of `afe6ee5`: each
of the five builders' `--check` passes, so the shipped pages were current, and the audit passes
all 2933 checks. The command now exits 3 when a page was stale, and `tables --check` rewrites
nothing (`docs/REPRODUCE.md`, section 1).

The manuscript check compares only some numbers with the record: the C1 and C2 headline means,
intervals and per-seed lists of `artifacts/corrected_verdicts.json`, the corrected primary-run
paragraph, and, since the review, the rule-ledger tally and the gate table's row and set counts
(`docs/REPRODUCE.md`, section 1). Its "0 items" covered nothing else. With the two count checks
added, the same command on the paper at `7fa50fa` reports 4 items: `main.tex:551` gives 48
rule-and-run pairs and 20 fails against the ledger's 49 and 21, and `main.tex:725` gives
twenty-six rows and ten sets against the gate table's 27 and 11.

## 2. Supplement archive

`python scripts/reproduce.py supplement --out dist/supplement-review.zip --require-runs`

- 1699 files, 1698 listed in `SHA256SUMS`; 501,139,861 bytes.
- SHA-256 `117f70b4eb11c44d4bf2a1e88b40a02b7d653b68c9eea3e8ff4004dfb9bc1151`.
- Every `RUN_DIRS` bundle included: `corrected_l3_h8_wm` (111 files), `corrected_l3_h8_nowm`
  (111), `l3_h8_heldout/agents` (60), `l3_h8_gseed2_gpu/agents` (60),
  `balanced_confirmation_wm` (146), `balanced_confirmation_touch` (146),
  `corrected_l1_heldout` (179).
- No identity strings found in any file; every image on the reviewed list. Left out: four
  film-production specs and three run-script figures inside run bundles.
- Correction, 2026-10-16: the identity check did not look for commit IDs, and this archive
  carries them. Its text members hold 23 distinct full 40-character hashes, every one a commit
  object in this repository, and short hashes in the documents and in `PROVENANCE.md`. One full
  hash found the public repository in a single code search. The archive is not anonymous and
  must not be submitted. The builder now replaces every commit ID with a keyed pseudonym of the
  same length and fails if one survives (`docs/REPRODUCE.md`, "Anonymized supplement").
- The archive is also five times the 100 MB that TMLR accepts for supplementary material.
- The archive is 138 MB larger than the 2026-10-15 build, about the size of the corrected L1
  bundle it now carries.

## 2a. Review build, 2026-10-16 (a test build, not the submission archive)

`python scripts/reproduce.py supplement --profile review --require-runs`, built from the
working tree of the review fixes (uncommitted, while the erasure readout was being rerun) to a
directory outside the repository. 1011 files, 1010 listed in `SHA256SUMS`; 39,698,202 bytes,
under the 100 MB limit. The build enforced 100 MiB (104,857,600 bytes) when this archive was
made and now enforces 100,000,000 bytes, the reading the paper states; this archive is under
both. Run bundles:
`l3_h8_heldout/agents` (60 files) and `l3_h8_gseed2_gpu/agents` (60). 207 commits were replaced
by pseudonyms; the identity scan and the commit scan found nothing. Extracted into an empty
directory: every checksum verifies, and both `python scripts/reproduce.py tables --check` and
`python scripts/audit_stats_recheck.py` fail 1 of 3085 checks, the results-manifest check, on
the erasure artifact under rerun. The working tree fails the same check for the same reason.
The submission archive is still to be built and hashed after the rerun lands.

## 3. Clean extraction

Extracted into an empty directory outside the repository. Every checksum in `SHA256SUMS`
verifies. A new Python 3.11.16 environment from the archive's own files:

```bash
uv venv --python 3.11 venv
uv pip install --python venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python venv -r requirements.txt -c constraints-cpu-corrected.txt
```

resolves torch 2.14.1+cpu and NumPy 2.4.6.

| command, run inside the extraction | result |
|---|---|
| `python scripts/reproduce.py tables` | pass, all 2932 audit checks (inside an archive the one check on a left-out film spec is skipped, by design) |
| `python scripts/run_l0_audit.py --agents-dir fullruns/corrected_l3_h8_wm/agents --out <dir>/l0.json` | not completed: stopped after 3 of 10 seeds when the session reclaimed memory; the 27 L0 values it printed equal, to the last printed digit, a replay of the same command from the repository on the same stack that morning |

The `tables` pass in this table rebuilt any stale page before its audit, as in section 1, so it
does not show by itself that the shipped pages were current.

`run_l0_audit.py` and the other entry points import `scripts/_bootstrap.py` from their own
directory, so they must not be run with `python -I`; `reproduce.py` does not need it.

## 4. Source package

`python docs/paper/make_source_package.py` writes `docs/paper/dist/source.zip` (20 files),
SHA-256 `2e2999e8764533f9a9e056d81dfbd4cab9609077c13ccd7baea3f5bbb27952da`. Extracted into an
empty directory and compiled there with latexmk: 73 pages, no LaTeX errors, no undefined
references, an empty PDF Author field, and no author name in the text (the one "Benjamin" is a
cited author). The edits of the deep revision review read as made: "recorded as met", "above-sham
behavioral effect", "did not demonstrate a reliable improvement", the controlled-decodability
qualifier, the separate B1, B2 and L1-CORRECTED stacks, "470 of 640", "matched 24-step deviation"
and "not to the bit".

This check found one signed zero left in the contrast-interval table, a mean of -0.000 on the
hidden-7 rerun row. The table builder now prints it as +0.000 (`d84e80d`, this PR); the paper
was rebuilt (`7fa50fa`) and the source package above is the rebuilt one. The supplement was
built at `v1.4.1`, before that fix; the fix changes no artifact and no number.

## Remaining limitations

- The saved-agent replay from the clean extraction ran 3 of 10 seeds before it was stopped for
  memory; a full run from the extraction is still owed. The same command from the repository on
  the same stack completed and is compared with the committed artifact in `docs/REPRODUCE.md`.
- The full supplement is 501 MB, and TMLR accepts at most 100 MB. The review profile
  (`--profile review`, about 40 MB) carries only the held-out and seed-2 agents; the saved
  agents and state dumps of C1, C2, B1, B2 and L1-CORRECTED would need separate anonymous
  hosting if reviewers are to replay those readouts.
- Bit-exact equality holds only on the stack and machine that wrote a result; off that machine
  the replay route of `docs/REPRODUCE.md` gives the measured tolerances.
- The GPU results depend on the workstation's local NumPy 1.26.4 build, which no one else can
  install from PyPI (`docs/CORRECTIONS.md`, 2026-10-14).
