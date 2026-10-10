# Figure provenance

Every committed figure in this directory is drawn by a deterministic script from committed
artifacts (run from the repo root). Figures are committed only deliberately, together with
the numbers they support, and a provenance row here is updated in the same commit. No
figure carries a title; the caption or the text around a figure says what it shows.

| Figure | Generating script | Reads | What it shows | Backing doc section |
|--------|-------------------|-------|---------------|---------------------|
| `expA_ceiling.png` | `scripts/plot_record_figures.py` | `artifacts/expA/summary.json`, `expA_l1.calibration` | L1 oracle AUROC against grid spacing, in units of the sensor noise (σ = 0.01) | FINDINGS.md sec. 2.1 |
| `expA_L2_ceiling.png` | `scripts/plot_record_figures.py` | `artifacts/expA/summary.json`, `expA_l2.calibration` | L2 oracle AUROC against drift strength | FINDINGS.md sec. 2.2 |
| `expB_incidental.png` | `scripts/plot_record_figures.py` | `artifacts/expB/summary.json`, `expB_full.drift_sweep` | Recurrent-state probe on the prediction-only agent across the drift sweep: the world-identity target (mean ± std over 3 seeds), the speed positive control and the shuffled-label negative control. The artifact records no across-seed std for the two controls, so they are drawn as means. | FINDINGS.md sec. 3.1 |
| `expB_channels.png` | `scripts/plot_record_figures.py` | `artifacts/expB/summary.json`, `expB_surprise.drift_sweep_with_std` and `expB_full.drift_sweep` | Recurrent-state and prediction-error channels, mean ± std over 3 seeds | FINDINGS.md sec. 3.2 |
| `expB_kstep.png` | `scripts/plot_record_figures.py` | `artifacts/expB/summary.json`, `expB_kstep_rerun_20260713.horizons` | Target AUROC across open-loop horizons at drift 0.45 and at the drift-zero control, mean ± std over 3 seeds, from the recorded 2026-07-13 rerun that the sec. 3.3 table quotes | FINDINGS.md sec. 3.3 |
| `budget_curve.png` | `scripts/build_budget_curve.py` | `artifacts/budget_curve.json` | The corrected arms' survival return and world-identity target against the training budget | FINDINGS.md sec. 17.10 |

`artifacts/expA/summary.json` and `artifacts/expB/summary.json` were promoted by
`scripts/promote_ab_summaries.py` from the full run `fullruns/06302026` (commit `4c16be6`,
Tesla T4) and the k-step rerun log `fullruns/kstep_rerun_20260713.log`. They are the same
artifacts the FINDINGS tables quote.

`scripts/plot_record_figures.py` also redraws the two B-v2 figures in `artifacts/expB2/`:
`expB2_survival.png` from `expB2_results.json` and `expB2_survival_confirmatory_n3.png` from
`expB2_results_confirmatory_n3.json`. Each shows the pooled world-identity AUROC per arm at
drift 0 and 0.45 (mean ± std over 3 seeds) against chance and the registered bar, and each
is labeled as a historical run under the pre-correction trainer.

Redraw every figure the script owns:

```bash
python scripts/plot_record_figures.py
```

For print, `python scripts/plot_record_figures.py --size print --format pdf --out-dir <dir>`
draws each figure at the width it is printed at, with 7.5 pt type. Every `plot_*` function
in the script takes the output path, the width and height in inches and the font size, so a
print build can import it and choose its own sizes.

## The run scripts' figures are superseded

`scripts/run_expA.py`, `run_expA_l2.py`, `run_expB_full.py`, `run_expB_surprise.py` and
`run_expB_kstep.py` (and `scripts/run_e2e.py`, which calls them) still save a titled figure
of whatever run they just made over the file of the same name in this directory. Those
images are run outputs and do not replace the committed figures. After such a run, restore
the committed copies with `git checkout -- docs/figures` or rerun
`scripts/plot_record_figures.py`. `tests/test_plot_record_figures.py` fails when a committed
copy does not have this script's pixel size. `scripts/run_expB2.py` likewise writes
`expB2_survival.png` into its own `--out-dir`; the committed copy in `artifacts/expB2/` is
redrawn by `scripts/plot_record_figures.py`.

## Correction (2026-10-10)

Until this date the committed `expA_ceiling.png`, `expA_L2_ceiling.png`,
`expB_incidental.png` and `expB_channels.png` did not show the numbers the tables quote.
They were last redrawn on 2026-06-28 on the local stack, and the local quick-mode run
`fullruns/07042026` (commit `888eb3e`, RTX 4050 laptop GPU) reproduces them byte for byte.
The tables quote the promoted full run `fullruns/06302026` (commit `4c16be6`, Tesla T4).
The plotted values differed from the quoted ones by up to 0.055 (L1 at Δ = 0.015: 0.457
plotted, 0.402 quoted; L2 at drift 0.01: 0.681 plotted, 0.706 quoted). Quick mode scales
only B-v2, so the gap comes from the different run and stack. The images also carried the
project name in a raster title, and `expB_incidental.png` left out the speed positive
control. `expB_kstep.png` had been corrected on 2026-07-13 (FINDINGS.md sec. 3.3) and kept
its source. All five are now drawn from the committed artifacts by
`scripts/plot_record_figures.py`, and the qualitative reading of each is the same as before.

The regeneration contract: if a figure changes when redrawn, either an artifact or the
plotting code changed (find the commit and record it here) or the environment differs
(report the matplotlib version). A silently changed figure is a bug in process, not a
result.
