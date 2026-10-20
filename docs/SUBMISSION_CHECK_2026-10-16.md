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

## 2a. Review build, 2026-10-16 (after the critical review)

`python scripts/reproduce.py supplement --profile review --require-runs`, built from commit
`fdc3581` of `fix/critical-review-2026-10-16`, after the erasure rerun landed.

- 1014 files, 1013 listed in `SHA256SUMS`; 39,717,557 bytes, under TMLR's 100 MB limit (the
  build enforces 100,000,000 bytes).
- SHA-256 `96d1324fb8e31cc10ceca0a7a835b0ecbb2697c7bba8255a717fddc8ec50d6c7`.
- Run bundles: `l3_h8_heldout/agents` and `l3_h8_gseed2_gpu/agents` only. The corrected-run
  agents and state dumps are not in it.
- 208 commit IDs (224 spellings) replaced by keyed pseudonyms; the map stays local in
  `dist/supplement-review.commit-map.json` for the camera-ready. The identity scan found nothing.
- Correction, 2026-10-16, after the second review pass: that scan covered names, local paths,
  images and commit IDs, not recorded digests. Outside `SHA256SUMS` the archive's text members
  hold 83 distinct 64-character SHA-256 digests, most of them agent hashes recorded in the
  L0-audit and persistence artifacts. The same strings sit in the public repository, so each is
  a search key in the way a commit hash is. The builder now replaces them with keyed pseudonyms
  as well and keeps the real byte digests in `SHA256SUMS` (`docs/REPRODUCE.md`, "Anonymized
  supplement"). This archive predates that change, and a rebuild replaces it.
- Correction, 2026-10-16, after the third review pass: "the identity scan found nothing" is not
  a finding that the archive is anonymous. The scan looked for names, local paths, images and
  hex tokens. It did not look for branch names, and it cannot see that the code and documents
  match the public repository, which no token replacement changes. See 2b.

Extracted into an empty directory outside the repository: every checksum verifies,
`python scripts/audit_stats_recheck.py` passes all 3126 checks (the one check on the left-out
film spec is skipped and counted, by design), `python scripts/reproduce.py tables` passes, and
`tests/test_audit_selftest.py` passes (18 passed, 1 skipped: the checkout-only floor case).
A grep for the owner's name, handle and ORCID finds nothing.

Note, 2026-10-16, second review pass: the full test suite does not yet pass inside this archive.
In a fresh extraction of it, with the three test files patched after the build
(`tests/test_social_card.py`, `tests/test_viz_collect.py`, `tests/test_brain_film.py`) copied
in under the archive's package name, `python -E -m pytest -q` gave 1190 passed, 24 skipped and
1 failed in 16.6 minutes on Python 3.13.2. The failure is
`tests/test_optional_readouts.py::test_every_optional_readout_file_carries_its_provenance`: it
reads `docs/optional/*.json`, which `INCLUDE_FILES` does not list, so the archive carries none
and the test's `assert paths` fails. A rebuilt archive needs that test guarded or those files
shipped, and a full test run in a fresh extraction, before the suite can be said to pass there.
Decided after the third review pass: the archive ships `docs/optional/` (its README and the
readout files), the test reads them there as it does in a git checkout, and the audit
recomputes them. A full test run in a fresh extraction of a build with that change was made on
2026-10-17 (UTC) on the critical-review build `57c17a71...3822`: every checksum verified, 1214
passed, 28 skipped, 0 failed. The v1.5.0 build's run is in section 2c.

This file is not in the archive: `INCLUDE_FILES` in `scripts/reproduce.py` does not list it, so
no copy of it, current or earlier, ships to reviewers.

## 2b. Rebuilt review archive and what anonymity it has, 2026-10-16 (third review pass)

`dist/supplement-review.zip`, rebuilt with the digest replacement at `2fc8b8e`: 1019 files, 1018
listed in `SHA256SUMS`; 39,763,469 bytes; SHA-256
`e93024099b8bf60dd7d53ef7d5bade5abaa7e51cca80bec768156f3077f454e9`. Its commit map records 227
commit pseudonyms and 84 replaced digests. The third review pass found three ways back to the
public repository in it.

- Branch names. Its text members carry four public branch names with random suffixes, 16 times
  in all (the results manifest and its builder and test, FINDINGS, the Experiment C
  preregistration, the revision plan, one spec). A pull-request search on each name, with no
  repository named, returned only the public repository. The builder rule for commit IDs and
  digests has no branch-name case, so the leak scan passed. Fix: the builder replaces every
  branch name with a neutral label, and the build fails if any listed branch or pull-request head name survives (`docs/REPRODUCE.md`).
  This archive must not be submitted; a rebuild replaces it.
- `SHA256SUMS`. It keeps the real digest of every shipped file so that it verifies the bytes.
  83 of its lines (53 distinct digests: the shipped agents of both bundles, five surrogate
  exports and eight figures) hold a value that a tracked file on the public `main` also holds,
  so reading the file gives a search key without hashing anything. This is kept, so that the
  checksums still verify, and disclosed (`docs/REPRODUCE.md`, "Anonymized supplement").
- The code itself. Apart from the package name, class and function names and docstrings match
  the public repository verbatim, and so do the 16-character configuration fingerprints. The
  documents match too, apart from the replaced identifiers (commit IDs, recorded digests, branch
  names, identity terms and film titles). One code search on the world's class name returned 37 hits, every one in the
  repository. No token replacement removes this. It is disclosed, not fixed.

The archive is therefore not anonymous to a search while the repository and the site are
public. Pseudonymizing commit IDs, digests and branch names removes those search keys; it does
not hide that the archive matches a public repository. The owner decides, before the upload,
whether the repository and the site stay public during review, or whether the submission relies
on the venue's policy on prior public posting. Nothing in this record makes that decision.

Decided 2026-10-17 (D1, option c, owner): the repository and the site stay public during review,
and saved agent weights stay on the nine run and upload branches of the public remote that
already carry them (`AGENTS.md`). The owner confirms this in writing before the upload.

Superseded 2026-10-19: this archive (`e9302409...`) and the critical-review build
`dist/critical-review-2026-10-16/supplement-review.zip` (39,787,045 bytes, SHA-256
`57c17a7179b04fd57845dfe02d8f29679be685d1a12852b0f9d6ba435f2e3822`) are replaced by the v1.5.0
build in section 2c. Neither is to be uploaded.

Owner decisions of the 2026-10-17 closing pass, as chosen: runs launched on an explicit yes and
later narrowed in writing (four extension readouts not run: X02, X04, X05, X06); the corrected C1
and C2 agents and rerun cells ship in the review archive; the manuscript was retitled by the
owner's written override of the keep-title default, and the repository keeps its name; version
1.5.0 with the release date set to the tag date; no GitHub Release object; no training proposals
run; each push made with the owner's approval.

## 2c. Review archive for v1.5.0, 2026-10-19

*Superseded 2026-10-20 by the v1.5.1 archive (section 2d), which carries the records revised for
the owner's two blind reviews. The v1.5.0 tag and this archive stay as built; neither archive
below 2d is to be uploaded.*

`python scripts/reproduce.py supplement --profile review --require-runs --out dist/review-1.5.0/supplement-review.zip`,
built from the release commit `fc363fb` (tag `v1.5.0`) with the key in `dist/supplement-commit-key.txt`.

- `dist/review-1.5.0/supplement-review.zip`: 1406 files, 1405 listed in `SHA256SUMS`;
  94,236,313 bytes, under the 100,000,000 bytes the build enforces.
- SHA-256 `cfb851a47fe1662eb6dd3f6df702b38aedaba2d104eb4b4a0d8030ac784aa2f2`, equal to the
  `archive_sha256` of its commit map.
- Run bundles: `l3_h8_heldout/agents` (60 files), `l3_h8_gseed2_gpu/agents` (60), and the saved
  agents and rerun cells of the corrected C1 and C2 runs: `corrected_l3_h8_wm/agents` (90),
  `corrected_l3_h8_nowm/agents` (90), `corrected_l3_h8_wm/cells` (20) and
  `corrected_l3_h8_nowm/cells` (20). `README_SUPPLEMENT.md` names each folder and points to
  `SHA256SUMS` for the per-file digests.
- 373 commit IDs (394 spellings) and 163 recorded digests (164 spellings) replaced by keyed
  pseudonyms; 0 unresolved commit-field spellings. The map stays local in
  `dist/review-1.5.0/supplement-review.commit-map.json`. The identity scan found no identity
  string, and every image is on the reviewed list.
- In a fresh extraction: every line of `SHA256SUMS` verifies (1405 of 1405);
  `python -E scripts/reproduce.py tables --check` and `python -E scripts/audit_stats_recheck.py`
  pass 3873 checks with 1 skipped, against 3874 in a git checkout (one check reads a file the
  archive leaves out); `python -E -m pytest -q -rs` gives 1537 passed, 41 skipped, 0 failed in
  13:53 on Python 3.13.2 with NumPy 1.26.4. Against the critical-review reference run (28 skips
  on 22 lines) the skips differ only by additions: tests that need git history or a git
  checkout, a film-data test whose files the archive leaves out, and one more social-card file
  not shipped. No skip reason disappeared or fell in count.
- A first fresh-extraction run on a trial build from `ab68d05` failed 14 tests. Each asserted
  that a readout's provenance names a commit, which is empty by design where there is no git;
  the tests now check the field's name everywhere and its value only in a git checkout
  (PR #165). The trial build was deleted and never uploaded.
- The non-anonymous archives moved out of `dist/` before this build
  (`supplement-review.zip`, SHA-256 `117f70b4...1151`, 501 MB;
  `supplement-review-2026-10-15-pre-t21.zip`; `supplement-v1.3.0.zip`) carry real commit IDs
  and must never be uploaded.

## 2d. Review archive for v1.5.1, 2026-10-20

*Superseded 2026-10-20 by the v1.5.2 archive (section 2e), which carries the records revised for
the owner's third blind review. The v1.5.1 tag and this archive stay as built; neither archive
below 2e is to be uploaded.*

`python scripts/reproduce.py supplement --profile review --require-runs --out dist/review-1.5.1/supplement-review.zip`,
built from the release commit `196afed` (tag `v1.5.1`) with the key in `dist/supplement-commit-key.txt`.
Between v1.5.0 and this build, the records were brought in line with the manuscript after two
owner-run blind reviews (PRs #168 and #169: dated notes, one CORRECTIONS entry per review, no
recorded number, label or verdict changed), and the paper-table generator prints a lower bound
that would round onto the bar with the digit that shows its side (hidden 7 reads 0.6496).

- `dist/review-1.5.1/supplement-review.zip`: 1406 files, 1405 listed in `SHA256SUMS`;
  94,247,959 bytes, under the 100,000,000 bytes the build enforces.
- SHA-256 `2e6bf9ffff8390a8e1494f81c988e1b602ac7d873dbd095ab3ca098af750bea2`, equal to the
  `archive_sha256` of its commit map.
- Run bundles as in 2c: `l3_h8_heldout/agents` (60 files), `l3_h8_gseed2_gpu/agents` (60),
  `corrected_l3_h8_wm/agents` (90), `corrected_l3_h8_nowm/agents` (90),
  `corrected_l3_h8_wm/cells` (20) and `corrected_l3_h8_nowm/cells` (20), each named in
  `README_SUPPLEMENT.md` with digests in `SHA256SUMS`.
- 378 commit IDs (399 spellings) and 163 recorded digests (164 spellings) replaced by keyed
  pseudonyms; 0 unresolved commit-field spellings; map local in
  `dist/review-1.5.1/supplement-review.commit-map.json`. The identity scan found no identity
  string, and every image is on the reviewed list.
- In a fresh extraction: 1405 of 1405 `SHA256SUMS` lines verify;
  `python -E scripts/reproduce.py tables --check` and `python -E scripts/audit_stats_recheck.py`
  pass 3873 checks with 1 skipped, against 3874 in a git checkout; `python -E -m pytest -q -rs`
  gives 1538 passed, 41 skipped, 0 failed in 15:51 on Python 3.13.2 with NumPy 1.26.4 (one more
  pass than 2c: the new table-generator test). The skips match the v1.5.0 run line for line.
- Tag `v1.5.1` pushed 2026-10-20 01:29 UTC on `196afed`; its release run passed (the tag names
  the version, the package built, `dist` uploaded). It warns that `date-released` (2026-10-20, the
  tag date in UTC) differs from the release commit's own date, written in Eastern time (2026-10-19).

## 2e. Review archive for v1.5.2, 2026-10-20 (the upload candidate)

`python scripts/reproduce.py supplement --profile review --require-runs --out dist/review-1.5.2/supplement-review.zip`,
built from the release commit `0fd8b79` (tag `v1.5.2`) with the key in `dist/supplement-commit-key.txt`.
Between v1.5.1 and this build, the records were brought in line with the manuscript after the
owner's third blind review (accept, 3/5) and its code-check follow-up (PR #171: dated notes, one
CORRECTIONS entry, no recorded number, label or verdict changed), and the rule ledger gained a
timing column (41 prospective, 6 simultaneous, 2 retrospective) that the manuscript's Table 10
prints.

- `dist/review-1.5.2/supplement-review.zip`: 1406 files, 1405 listed in `SHA256SUMS`;
  94,256,134 bytes, under the 95,000,000-byte working cap and the 100,000,000 bytes the build
  enforces.
- SHA-256 `7864e5c7adfefa23883245881d6cbb333d7addafeff3c981037f86f892075049`, equal to the
  `archive_sha256` of its commit map.
- Run bundles as in 2c and 2d: `l3_h8_heldout/agents` (60 files), `l3_h8_gseed2_gpu/agents` (60),
  `corrected_l3_h8_wm/agents` (90), `corrected_l3_h8_nowm/agents` (90),
  `corrected_l3_h8_wm/cells` (20) and `corrected_l3_h8_nowm/cells` (20), each named in
  `README_SUPPLEMENT.md` with digests in `SHA256SUMS`. No B1 or B2 agents ship.
- 382 commit IDs (403 spellings) and 163 recorded digests (164 spellings) replaced by keyed
  pseudonyms; 0 unresolved commit-field spellings; map local in
  `dist/review-1.5.2/supplement-review.commit-map.json`. The identity scan found no identity
  string, and every image is on the reviewed list.
- In a fresh extraction: 1405 of 1405 `SHA256SUMS` lines verify;
  `python -E scripts/reproduce.py tables --check` and `python -E scripts/audit_stats_recheck.py`
  pass 3873 checks, against 3874 in a git checkout; `python -E -m pytest -q -rs` gives 1542
  passed, 41 skipped, 0 failed on Python 3.13.2 with NumPy 1.26.4 (four more passes than 2d: the
  new rule-ledger tests). A single background run was stopped by a low-memory reap about halfway,
  with no failure, so the suite ran in three foreground parts over the same 154 test files
  (4:39, 6:59 and 2:32). The skips match the v1.5.1 run line for line.
- Tag `v1.5.2` pushed 2026-10-20 on `0fd8b79`; its release run passed (the tag names the version,
  the package built, `dist` uploaded), with the same `date-released` warning as v1.5.1 (2026-10-20
  in UTC against the release commit's Eastern-time date, 2026-10-19).

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
| `python scripts/run_l0_audit.py --agents-dir fullruns/corrected_l3_h8_wm/agents --out <dir>/l0.json` | completed 2026-10-16 on the second attempt (the first stopped after 3 of 10 seeds when the session reclaimed memory): integrity gate PASS, worst deviation 0.0008 against 0.01; all 1488 numeric fields of the output equal the repository replay of the same command on the same stack, wall time aside; the extraction's `l0.json` was not kept |

The `tables` pass in this table rebuilt any stale page before its audit, as in section 1, so it
does not show by itself that the shipped pages were current.

`run_l0_audit.py` and the other entry points import `scripts/_bootstrap.py` from their own
directory, so they must not be run with `python -I`; `reproduce.py` does not need it.

## 4. Source package

*Final, 2026-10-20 (after review 3):* paper commit `2e9ee73`, 92 pages, `docs/paper/dist/source.zip`
(21 files, 301,490 bytes), SHA-256 `6c0c6017d9dd0d17f8539add93bb99a4b736d6e7a8db79905d54c3745d46376f`,
the same bytes with and without the standalone compile, which gave 92 pages with no undefined
reference or overfull box and page-by-page text identical to `docs/paper/main.pdf`. It answers the
owner's three blind reviews (the third: accept, 3/5); the abstract now fits on one page and
Table 10 prints the ledger's timing column. The packaged `main.tex` carries "Anonymous authors".

*Superseded, 2026-10-20:* paper commit `4667176`, 92 pages, `docs/paper/dist/source.zip` (21 files),
SHA-256 `78ae2255cfdbff09022cd5135bbfdb135c443cf87027314dc36bf11961940189`, rebuilt from the
committed FINAL with the same bytes as the pre-push builds. It answers the owner's two blind
reviews (the second: leaning accept). The packaged `main.tex` carries "Anonymous authors".

*Superseded, 2026-10-19:* paper commit `55e741b`, 91 pages, `docs/paper/dist/source.zip` (21 files),
SHA-256 `b900ad35f51da66111e50b58697514b60e509ba3c6707eb8bd37ccf85a16b982`, the same bytes on two
builds. The builder compiled it standalone and passed; the packaged `main.tex` carries
"Anonymous authors" and no author name or e-mail. The entries below record earlier packages.

*Rebuilt 2026-10-16 after the critical review:* paper commit `eb79ca2`, 74 pages,
`docs/paper/dist/source.zip` SHA-256
`26a834918832cb6907a347807713af44bb9a70ece9c05e42a7245707fabb56b6`. The builder compiles it
standalone and refuses on any overfull box; it passed. The checks below were made on the earlier
package.

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

- The review archive is not anonymous to a search while the repository and the site are public
  (section 2b). The owner decided on 2026-10-17 that they stay public during review (2b).
- The full supplement is 501 MB, and TMLR accepts at most 100 MB. The review profile
  (`--profile review`, about 40 MB) carries only the held-out and seed-2 agents; the saved
  agents and state dumps of C1, C2, B1, B2 and L1-CORRECTED would need separate anonymous
  hosting if reviewers are to replay those readouts.
  [Correction, 2026-10-19: since 2026-10-17 the review profile also carries the saved agents and
  rerun cells of the corrected C1 and C2 runs (`fullruns/corrected_l3_h8_wm/` and `_nowm/`,
  per-file digests in `SHA256SUMS`), so the C1 and C2 state dumps rebuild from the archive; the
  about 40 MB above predates them, and the size check before upload decides whether both ship.
  The B1, B2 and L1-CORRECTED bundles are still left out.]
- Bit-exact equality holds only on the stack and machine that wrote a result; off that machine
  the replay route of `docs/REPRODUCE.md` gives the measured tolerances.
- The GPU results depend on the workstation's local NumPy 1.26.4 build, which no one else can
  install from PyPI (`docs/CORRECTIONS.md`, 2026-10-14).
