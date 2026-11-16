"""Extension of the draw-crossed readout (frozen 2026-11-16): the C1 arms on their own walks
over further draws, and each arm's reading net of its own control.

docs/specs/2026-11-16-draw-crossed-extension-design.md. The crossed interval, the level, band
and margin labels and the integrity gate are those of itasorl.draw_crossed, unchanged. This
module adds one contrast and its label: an arm's reading at drift 0.45 minus the same arm's
reading at drift zero, paired by agent seed index and by draw.
"""

from __future__ import annotations

import numpy as np

from itasorl import draw_crossed as dc

RUN = "C1"
ARMS = ("survival", "predictor", "untrained")
FEATURE_SETS = dc.OWN                      # each arm's own walk; no replayed stream
PROBES = dc.PROBES                         # the penalty chosen in fold (primary), and C = 1
DRIFT_ZERO, DRIFT_SURROGATE = 0.0, 0.45
NET_FLOOR = round(dc.BAR - 0.5, 10)        # the bar's distance from chance, 0.15

# Seed-and-draw-paired leads over the baseline arms at the surrogate drift, as in the first
# readout: (name, minuend, subtrahend).
LEADS = tuple(c for c in dc.CONTRASTS if c[2] in dc.OWN)


def label_net(ci: dict, floor: float = NET_FLOOR) -> str:
    """Where the crossed interval of a reading net of its control lies, tested in this order:
    at or above the bar's distance from chance, above zero, or neither."""
    if not ci.get("computable"):
        return dc.NOT_COMPUTABLE
    if ci["lo"] >= floor:
        return "at least the bar's distance from chance"
    if ci["lo"] > 0.0:
        return "above its control"
    return "not shown above its control"


def net_name(arm: str) -> str:
    return f"{arm} arm net of its control, own walk"


def _m(cells: list, drift: float, feature_set: str, probe: str, bases: list) -> np.ndarray:
    return np.asarray(dc.matrix(cells, RUN, drift, feature_set, probe, bases), float)


def summarize(cells: list, bases: list, registered_bases, ok: bool, labels: bool = True) -> dict:
    """Every summary of one set of draws, recomputable from `cells` alone.

    Per probe: the crossed interval of each arm's own-walk reading at each drift, the two
    leads over the baseline arms at the surrogate drift, and each arm's reading net of its
    control. With `labels` the pre-stated labels are attached (VOID when the integrity gate
    failed); without it every label is None, which is how a set of draws that had been read
    before the design was frozen is reported.
    """
    def lab(value):
        if not labels:
            return None
        return value if ok else dc.VOID

    out = {}
    for probe in PROBES:
        block = {"readings": {}, "leads": {}, "net_of_control": {}, "registered_draw": {}}
        mats = {}
        for drift in (DRIFT_ZERO, DRIFT_SURROGATE):
            dkey = f"{drift:.2f}"
            block["readings"][dkey] = {}
            block["registered_draw"][dkey] = {}
            for fs in FEATURE_SETS:
                M = _m(cells, drift, fs, probe, bases)
                mats[(dkey, fs)] = M
                ci = dc.crossed_interval(M)
                if drift == DRIFT_ZERO:
                    label = lab(dc.label_band(ci))
                elif fs == "own/survival":
                    label = lab(dc.label_level(ci))
                else:
                    label = None            # baselines at the surrogate drift carry no label
                block["readings"][dkey][fs] = {**ci, "label": label}
                if registered_bases is not None:
                    col = [r[0] for r in dc.matrix(cells, RUN, drift, fs, probe, [registered_bases])]
                    block["registered_draw"][dkey][fs] = {
                        "per_seed": col,
                        "mean": (float(np.mean(col)) if col and np.all(np.isfinite(col))
                                 else float("nan"))}
        skey = f"{DRIFT_SURROGATE:.2f}"
        for name, plus, minus in LEADS:
            ci = dc.crossed_interval(mats[(skey, plus)] - mats[(skey, minus)])
            block["leads"][name] = {**ci, "label": lab(dc.label_margin(ci))}
        for arm in ARMS:
            fs = f"own/{arm}"
            ci = dc.crossed_interval(mats[(skey, fs)] - mats[(f"{DRIFT_ZERO:.2f}", fs)])
            block["net_of_control"][net_name(arm)] = {
                **ci, "label": (lab(label_net(ci)) if arm == "survival" else None)}
        out[probe] = block
    return out


def first_cells(first_artifact: dict) -> list:
    """The C1 cells of the first readout's artifact, reduced to the own-walk readings, so that
    its eight draws can stand beside the further ones. Nothing is rescored."""
    out = []
    for c in first_artifact["cells"]:
        if c["run"] != RUN:
            continue
        out.append({"run": RUN, "drift": c["drift"], "seed": c["seed"],
                    "draws": [{"bases": d["bases"],
                               "readings": {fs: d["readings"][fs] for fs in FEATURE_SETS}}
                              for d in c["draws"]]})
    return out


def merged_cells(cells: list, first: list) -> list:
    """One cell per (drift, seed) holding the draws of both readouts, for the pooled summary."""
    by_key = {(float(c["drift"]), c["seed"]): c for c in first}
    out = []
    for c in cells:
        other = by_key.get((float(c["drift"]), c["seed"]))
        seen = {tuple(d["bases"]) for d in c["draws"]}
        extra = [d for d in (other["draws"] if other else []) if tuple(d["bases"]) not in seen]
        out.append({"run": RUN, "drift": c["drift"], "seed": c["seed"],
                    "draws": list(c["draws"]) + extra})
    return out


def summarize_all(cells: list, first: list, further_bases: list, first_bases: list,
                  registered_bases, ok: bool) -> dict:
    """The three sets of draws of the frozen design.

    `further`: the further draws alone, prospective, with every label.
    `first_eight`: the reading net of its control on the eight draws of the first readout,
    post hoc there, without labels; the other summaries of those draws are FINDINGS 25's.
    `all_fresh`: both sets together, descriptive, without labels.
    """
    further = summarize(cells, further_bases, registered_bases, ok, labels=True)
    eight = summarize(first, first_bases, None, ok, labels=False)
    pooled = summarize(merged_cells(cells, first), list(first_bases) + list(further_bases),
                       None, ok, labels=False)
    return {"further": further,
            "first_eight": {p: {"net_of_control": eight[p]["net_of_control"]} for p in PROBES},
            "all_fresh": {p: {k: v for k, v in pooled[p].items() if k != "registered_draw"}
                          for p in PROBES}}
