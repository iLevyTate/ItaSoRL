"""Draw the Experiment A, B and B-v2 record figures from committed artifacts.

Nothing here trains, rolls out or evaluates an agent. Every plotted number is read from a
committed JSON under artifacts/, and nothing is typed in by hand. No figure carries a
title: the caption or the surrounding text says what a figure shows.

    figure                              reads
    expA_ceiling                        artifacts/expA/summary.json, expA_l1.calibration
    expA_L2_ceiling                     artifacts/expA/summary.json, expA_l2.calibration
    expB_incidental                     artifacts/expB/summary.json, expB_full.drift_sweep
    expB_channels                       artifacts/expB/summary.json,
                                        expB_surprise.drift_sweep_with_std and
                                        expB_full.drift_sweep
    expB_kstep                          artifacts/expB/summary.json,
                                        expB_kstep_rerun_20260713.horizons
    expB2_survival                      artifacts/expB2/expB2_results.json
    expB2_survival_confirmatory_n3      artifacts/expB2/expB2_results_confirmatory_n3.json

Run from the repository root:

    python scripts/plot_record_figures.py
        rewrites the committed copies in docs/figures/ and artifacts/expB2/
    python scripts/plot_record_figures.py --size print --format pdf --out-dir <dir>
        draws every figure at the width it is printed at, with 7.5 pt type

Each plot_* function takes the output path, the figure width and height in inches and the
base font size in points, so a print build can draw a figure at its printed width instead
of shrinking a larger drawing (which shrinks its type with it). Every figure_* function
returns the matplotlib Figure without saving it, which is what the tests inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

# Sizes for the committed copies, which are read on screen at roughly 900 px wide.
WEB_SIZE = (5.2, 3.6)
WEB_FONT = 9.5
WEB_DPI = 200

# Printed sizes in inches for a 6.5 in text width: a side-by-side pair at 0.48 of it and the
# k-step figure at 0.6 of it. Drawn at this size, PRINT_FONT is the printed type size.
PRINT_SIZES = {
    "expA_ceiling": (3.12, 2.35),
    "expA_L2_ceiling": (3.12, 2.35),
    "expB_incidental": (3.12, 2.35),
    "expB_channels": (3.12, 2.35),
    "expB_kstep": (3.9, 2.5),
    "expB2_survival": (3.12, 2.45),
    "expB2_survival_confirmatory_n3": (3.12, 2.45),
}
PRINT_FONT = 7.5
PRINT_DPI = 300

# Experiment A's L1 sensor-noise floor, a configuration value (SIGMA in scripts/run_expA.py)
# that the summary artifact does not repeat. The x axis of expA_ceiling is spacing over it.
L1_SENSOR_SIGMA = 0.01
REGISTERED_BAR = 0.65

# Validated categorical slots on a light surface (the same slots as budget_curve.png).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, CONTROL = "#2a78d6", "#eb6834", "#1baf7a", "#8f8d87"
ARM_COLOR = {"survival": BLUE, "predictor": ORANGE, "untrained": AQUA}
ARM_MARKER = {"survival": "o", "predictor": "s", "untrained": "^"}
PM = "±"
# Behind every printed value, so a reference line never runs through its digits.
LABEL_BOX = {"boxstyle": "square,pad=0.08", "facecolor": SURFACE, "edgecolor": "none"}


def load_artifact(rel: str) -> dict:
    with open(ART / rel, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- series (pure, tested)

def expA_l1_series() -> tuple[list[float], list[float]]:
    """Grid spacing in units of the sensor noise, and the oracle AUROC, fine to coarse."""
    rows = sorted(load_artifact("expA/summary.json")["expA_l1"]["calibration"],
                  key=lambda r: r["delta"])
    return [r["delta"] / L1_SENSOR_SIGMA for r in rows], [r["oracle_auroc"] for r in rows]


def expA_l2_series() -> tuple[list[float], list[float]]:
    rows = sorted(load_artifact("expA/summary.json")["expA_l2"]["calibration"],
                  key=lambda r: r["drift_sigma"])
    return [r["drift_sigma"] for r in rows], [r["oracle_auroc"] for r in rows]


def expB_full_series() -> dict[str, list[float]]:
    rows = sorted(load_artifact("expB/summary.json")["expB_full"]["drift_sweep"],
                  key=lambda r: r["drift"])
    keys = ("drift", "target_mean", "target_std", "shuffled_mean", "speed_mean")
    return {k: [r[k] for r in rows] for k in keys}


def expB_surprise_series() -> dict[str, list[float]]:
    rows = sorted(load_artifact("expB/summary.json")["expB_surprise"]["drift_sweep_with_std"],
                  key=lambda r: r["drift"])
    return {k: [r[k] for r in rows] for k in ("drift", "mean", "std")}


def expB_kstep_series() -> dict[str, list[float]]:
    """The recorded k-step rerun that FINDINGS 3.3 and its table cite (2026-07-13)."""
    rows = sorted(load_artifact("expB/summary.json")["expB_kstep_rerun_20260713"]["horizons"],
                  key=lambda r: r["open_horizon"])
    keys = ("open_horizon", "drift_045_target_mean", "drift_045_target_std",
            "control_target_mean", "control_target_std")
    return {k: [r[k] for r in rows] for k in keys}


def expB2_series(results: str = "expB2/expB2_results.json") -> dict:
    """Per arm, the seed mean and population std (ddof 0, as the run reported it) of the
    pooled world-identity target at each drift."""
    doc = load_artifact(results)
    drifts = sorted(doc, key=float)
    out: dict = {"drift": [float(d) for d in drifts]}
    for arm in ARM_COLOR:
        means, stds = [], []
        for d in drifts:
            vals = [float(v) for v in doc[d][arm]["pool_target"]]
            m = sum(vals) / len(vals)
            means.append(m)
            stds.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
        out[arm] = {"mean": means, "std": stds}
    return out


# ---------------------------------------------------------------- drawing helpers

class _Style:
    """Type sizes from the base font size, and mark weights scaled with it so a figure drawn
    small for print does not get heavy lines."""

    def __init__(self, fontsize: float):
        self.font = fontsize
        self.small = fontsize - 0.5
        k = min(1.0, max(0.7, fontsize / WEB_FONT))
        self.lw, self.ms, self.mew, self.elw, self.cap = 1.8 * k, 6.0 * k, 1.4 * k, 1.2 * k, 3 * k


def _new(width_in: float, height_in: float, fontsize: float):
    st = _Style(fontsize)
    fig = Figure(figsize=(width_in, height_in), facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=st.small, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax, st


def _labels(ax, st: _Style, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=st.font, color=INK)
    ax.set_ylabel(ylabel, fontsize=st.font, color=INK)


def _ticks(ax, x, pad: float) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlim(min(x) - pad, max(x) + pad)


def _chance(ax, st: _Style) -> None:
    """The chance line, named in the right margin where no data point can sit under it."""
    ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate("chance", xy=(1.0, 0.5), xycoords=("axes fraction", "data"), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center", fontsize=st.small,
                color=INK2, annotation_clip=False)


def _bar(ax, st: _Style) -> None:
    ax.axhline(REGISTERED_BAR, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"registered bar {REGISTERED_BAR:g}", xy=(1.0, REGISTERED_BAR),
                xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=st.small, color=INK2)


def _series(ax, st: _Style, x, y, *, color, marker, gid, label, yerr=None, dashed=False,
            hollow=False, line=True):
    """One data series. The line carries a gid so a test can read back what was plotted."""
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=st.elw,
                    capsize=st.cap, capthick=st.elw, zorder=2)
    art, = ax.plot(x, y, color=color, linewidth=st.lw,
                   linestyle=("--" if dashed else "-") if line else "none",
                   marker=marker, markersize=st.ms * (1.25 if marker == "^" else 1.0),
                   markeredgewidth=st.mew,
                   markerfacecolor=SURFACE if hollow else color,
                   markeredgecolor=color if hollow else SURFACE, label=label, zorder=3)
    art.set_gid(gid)
    return art


def _value(ax, st: _Style, x: float, y: float, whisker: float, above: bool) -> None:
    """Print a value past the end of its whisker, above or below the point."""
    ax.annotate(f"{y:.3f}", xy=(x, y + whisker if above else y - whisker),
                xytext=(0, 4 if above else -4), textcoords="offset points", ha="center",
                va="bottom" if above else "top", fontsize=st.small, color=INK, bbox=LABEL_BOX)


def _legend(ax, st: _Style, **kw) -> None:
    ax.legend(frameon=False, fontsize=st.small, labelcolor=INK2, handlelength=2.2,
              borderaxespad=0.2, **kw)


def _note(fig, st: _Style, text: str, width_in: float) -> None:
    """A one- or two-line note under the plot, wrapped to the figure width."""
    per_line = max(20, int(width_in * 72 / (st.small * 0.5)))
    fig.supxlabel(textwrap.fill(text, per_line, break_on_hyphens=False), fontsize=st.small, color=INK2, x=0.02,
                  ha="left", multialignment="left")


# ---------------------------------------------------------------- figures

def figure_expA_ceiling(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                        fontsize: float = WEB_FONT) -> Figure:
    """L1 oracle AUROC against grid spacing, in units of the sensor noise."""
    x, y = expA_l1_series()
    fig, ax, st = _new(width_in, height_in, fontsize)
    _series(ax, st, x, y, color=BLUE, marker="o", gid="expA_l1.oracle_auroc", label="oracle")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.set_ylim(0.3, 1.06)
    _chance(ax, st)
    for xi, yi in zip(x, y):
        if yi < 0.99:                   # label every point below saturation
            _value(ax, st, xi, yi, 0.0, above=yi >= 0.45)
    _labels(ax, st, f"grid spacing Δ in units of sensor noise σ = {L1_SENSOR_SIGMA:g}",
            "oracle AUROC")
    return fig


def figure_expA_l2_ceiling(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                           fontsize: float = WEB_FONT) -> Figure:
    """L2 oracle AUROC against drift strength."""
    x, y = expA_l2_series()
    fig, ax, st = _new(width_in, height_in, fontsize)
    _series(ax, st, x, y, color=BLUE, marker="o", gid="expA_l2.oracle_auroc", label="oracle")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.set_xlim(-0.002, max(x) * 1.25)
    ax.set_ylim(0.3, 1.06)
    _chance(ax, st)
    for xi, yi in zip(x, y):
        if yi < 0.992:                  # label the rising part; the plateau reads 0.992-0.993
            ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(5, -4), textcoords="offset points",
                        ha="left", va="top", fontsize=st.small, color=INK, bbox=LABEL_BOX)
    _labels(ax, st, "drift strength (log scale above 0.01)", "oracle AUROC")
    return fig


def figure_expB_incidental(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                           fontsize: float = WEB_FONT) -> Figure:
    """Recurrent-state probe on the prediction-only agent across the drift sweep, with the
    speed positive control and the shuffled-label negative control. The artifact records an
    across-seed std for the target only, so the two controls are plotted as means."""
    s = expB_full_series()
    x = s["drift"]
    fig, ax, st = _new(width_in, height_in, fontsize)
    _series(ax, st, x, s["speed_mean"], color=ORANGE, marker="^", gid="expB_full.speed_mean",
            label="speed (positive control)")
    _series(ax, st, x, s["shuffled_mean"], color=CONTROL, marker="s", dashed=True, hollow=True,
            gid="expB_full.shuffled_mean", label="shuffled labels (negative control)")
    _series(ax, st, x, s["target_mean"], yerr=s["target_std"], color=BLUE, marker="o",
            gid="expB_full.target_mean", label=f"world identity, {PM} std")
    for xi, m, sd in zip(x, s["target_mean"], s["target_std"]):
        _value(ax, st, xi, m, sd, above=False)
    _ticks(ax, x, 0.04)
    ax.set_ylim(0.3, 1.0)
    _chance(ax, st)
    _legend(ax, st, loc="center left", bbox_to_anchor=(0.0, 0.62))
    _labels(ax, st, "drift strength (0 = two authentic worlds)", "pooled AUROC, mean of 3 seeds")
    return fig


def figure_expB_channels(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                         fontsize: float = WEB_FONT) -> Figure:
    """The recurrent-state and prediction-error channels side by side."""
    b = expB_full_series()
    p = expB_surprise_series()
    fig, ax, st = _new(width_in, height_in, fontsize)
    _series(ax, st, b["drift"], b["target_mean"], yerr=b["target_std"], color=BLUE, marker="o",
            gid="expB_full.target_mean", label="recurrent-state probe")
    _series(ax, st, p["drift"], p["mean"], yerr=p["std"], color=AQUA, marker="D",
            gid="expB_surprise.mean", label="prediction-error probe")
    for xi, m, sd, other in zip(p["drift"], p["mean"], p["std"], b["target_mean"]):
        _value(ax, st, xi, m, sd, above=m >= other)     # on the side away from the other series
    _ticks(ax, b["drift"], 0.04)
    ax.set_ylim(0.3, 1.0)
    _chance(ax, st)
    _legend(ax, st, loc="upper left")
    _labels(ax, st, "drift strength (0 = two authentic worlds)",
            f"pooled AUROC, mean {PM} std of 3 seeds")
    return fig


def figure_expB_kstep(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                      fontsize: float = WEB_FONT) -> Figure:
    """Target AUROC across open-loop horizons at drift 0.45 and at the drift-zero control,
    from the recorded 2026-07-13 rerun."""
    s = expB_kstep_series()
    x = s["open_horizon"]
    fig, ax, st = _new(width_in, height_in, fontsize)
    _series(ax, st, x, s["drift_045_target_mean"], yerr=s["drift_045_target_std"], color=BLUE,
            marker="o", gid="expB_kstep.drift_045_target_mean", label="drift 0.45 (surrogate)")
    _series(ax, st, x, s["control_target_mean"], yerr=s["control_target_std"], color=CONTROL,
            marker="s", dashed=True, hollow=True, gid="expB_kstep.control_target_mean",
            label="drift 0 (control)")
    for xi, m, sd in zip(x, s["drift_045_target_mean"], s["drift_045_target_std"]):
        _value(ax, st, xi, m, sd, above=True)
    _ticks(ax, x, 1.5)
    ax.set_ylim(0.3, 1.0)
    _chance(ax, st)
    _legend(ax, st, loc="upper left")
    _labels(ax, st, "open-loop prediction horizon (steps)",
            f"pooled AUROC, mean {PM} std of 3 seeds")
    return fig


def figure_expB2_survival(width_in: float = WEB_SIZE[0], height_in: float = WEB_SIZE[1],
                          fontsize: float = WEB_FONT,
                          results: str = "expB2/expB2_results.json",
                          note: str = "Historical run (pre-correction trainer).") -> Figure:
    """B-v2 pooled world-identity target per arm at drift 0 and 0.45, mean and std over the
    three seeds. Plotted as AUROC against chance and the registered bar, so no whisker can
    cross a floor the quantity cannot reach."""
    s = expB2_series(results)
    drifts = s["drift"]
    fig, ax, st = _new(width_in, height_in, fontsize)
    offsets = {"survival": -0.16, "predictor": 0.0, "untrained": 0.16}
    for arm in ARM_COLOR:
        xs = [i + offsets[arm] for i in range(len(drifts))]
        _series(ax, st, xs, s[arm]["mean"], yerr=s[arm]["std"], color=ARM_COLOR[arm],
                marker=ARM_MARKER[arm], gid=f"expB2.{arm}.mean", label=arm, line=False)
    for i, m in enumerate(s["survival"]["mean"]):      # survival is the leftmost arm
        ax.annotate(f"{m:.3f}", xy=(i + offsets["survival"], m), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=st.small,
                    color=INK, bbox=LABEL_BOX)
    ax.set_xticks(range(len(drifts)))
    ax.set_xticklabels([f"{d:g}" for d in drifts])
    ax.set_xlim(-0.5, len(drifts) - 0.5)
    ax.set_ylim(0.3, 1.0)
    _chance(ax, st)
    _bar(ax, st)
    _legend(ax, st, loc="upper left", ncols=3, columnspacing=1.0, handletextpad=0.3)
    _labels(ax, st, "drift strength (0 = two authentic worlds)", "pooled AUROC")
    _note(fig, st, f"World-identity target, mean {PM} std over 3 seeds. {note}", width_in)
    return fig


def figure_expB2_survival_confirmatory_n3(width_in: float = WEB_SIZE[0],
                                          height_in: float = WEB_SIZE[1],
                                          fontsize: float = WEB_FONT) -> Figure:
    return figure_expB2_survival(
        width_in, height_in, fontsize, results="expB2/expB2_results_confirmatory_n3.json",
        note="Archived initial run, before the rigor hardening; historical "
             "(pre-correction trainer).")


# ---------------------------------------------------------------- saving

def _save(fig: Figure, out_path, dpi: int) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A PDF would otherwise carry a creation date that changes on every rebuild.
    meta = {"CreationDate": None} if out.suffix.lower() == ".pdf" else {}
    fig.savefig(out, dpi=dpi, facecolor=SURFACE, metadata=meta)
    return out


def plot_expA_ceiling(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1], fontsize=WEB_FONT,
                      dpi=WEB_DPI) -> Path:
    return _save(figure_expA_ceiling(width_in, height_in, fontsize), out_path, dpi)


def plot_expA_l2_ceiling(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1],
                         fontsize=WEB_FONT, dpi=WEB_DPI) -> Path:
    return _save(figure_expA_l2_ceiling(width_in, height_in, fontsize), out_path, dpi)


def plot_expB_incidental(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1],
                         fontsize=WEB_FONT, dpi=WEB_DPI) -> Path:
    return _save(figure_expB_incidental(width_in, height_in, fontsize), out_path, dpi)


def plot_expB_channels(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1],
                       fontsize=WEB_FONT, dpi=WEB_DPI) -> Path:
    return _save(figure_expB_channels(width_in, height_in, fontsize), out_path, dpi)


def plot_expB_kstep(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1], fontsize=WEB_FONT,
                    dpi=WEB_DPI) -> Path:
    return _save(figure_expB_kstep(width_in, height_in, fontsize), out_path, dpi)


def plot_expB2_survival(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1],
                        fontsize=WEB_FONT, dpi=WEB_DPI) -> Path:
    return _save(figure_expB2_survival(width_in, height_in, fontsize), out_path, dpi)


def plot_expB2_survival_confirmatory_n3(out_path, width_in=WEB_SIZE[0], height_in=WEB_SIZE[1],
                                        fontsize=WEB_FONT, dpi=WEB_DPI) -> Path:
    return _save(figure_expB2_survival_confirmatory_n3(width_in, height_in, fontsize),
                 out_path, dpi)


# name -> (plot function, figure builder, committed copy relative to the repository root)
FIGURES = {
    "expA_ceiling": (plot_expA_ceiling, figure_expA_ceiling, "docs/figures/expA_ceiling.png"),
    "expA_L2_ceiling": (plot_expA_l2_ceiling, figure_expA_l2_ceiling,
                        "docs/figures/expA_L2_ceiling.png"),
    "expB_incidental": (plot_expB_incidental, figure_expB_incidental,
                        "docs/figures/expB_incidental.png"),
    "expB_channels": (plot_expB_channels, figure_expB_channels,
                      "docs/figures/expB_channels.png"),
    "expB_kstep": (plot_expB_kstep, figure_expB_kstep, "docs/figures/expB_kstep.png"),
    "expB2_survival": (plot_expB2_survival, figure_expB2_survival,
                       "artifacts/expB2/expB2_survival.png"),
    "expB2_survival_confirmatory_n3": (plot_expB2_survival_confirmatory_n3,
                                       figure_expB2_survival_confirmatory_n3,
                                       "artifacts/expB2/expB2_survival_confirmatory_n3.png"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="draw only these figures")
    ap.add_argument("--size", choices=("web", "print"), default="web",
                    help="web: the committed copies; print: each figure at its printed width")
    ap.add_argument("--format", choices=("png", "pdf"), default="png")
    ap.add_argument("--out-dir", help="write <name>.<format> here instead of over the "
                    "committed copies (required with --size print or --format pdf)")
    a = ap.parse_args(argv)
    if a.out_dir is None and (a.size != "web" or a.format != "png"):
        ap.error("--size print and --format pdf need --out-dir; the committed copies are "
                 "web-sized PNGs")
    for name in a.only or FIGURES:
        plot, _, committed = FIGURES[name]
        out = ROOT / committed if a.out_dir is None else Path(a.out_dir) / f"{name}.{a.format}"
        if a.size == "print":
            w, h = PRINT_SIZES[name]
            path = plot(out, w, h, PRINT_FONT, dpi=PRINT_DPI)
        else:
            path = plot(out)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
