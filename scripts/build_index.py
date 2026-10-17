"""Regenerate index.html from index.template.html + the single-source metrics.

The template is the hand-maintained page with every result number replaced by a `{{key}}`
placeholder. Two sources fill them: `site_metrics.derive_metrics()` supplies the L3
headline set from artifacts/expB2/, and `derive_record_metrics()` below supplies the
numbers the page quotes from the wider committed record (the policy controls, persistence
and control-diagnostics readouts of the corrected run, the drag comparator, the historical
behavior audit, the graded seam, the texture knockout and the L1 organism runs).
Placeholders are plain text substitution, so they are valid in every context the numbers
live in (HTML text, JS string literals, JS numeric literals, and `data-target="..."`
attributes) - which an HTML-comment anchor could not be.

Usage:
  python scripts/build_index.py           # write index.html from the template
  python scripts/build_index.py --check    # exit 1 if index.html is stale (CI/gate use)

The forward cross-recipe number (transfer_rff_target ~0.684) is intentionally a template
literal, not a placeholder: its run lives in an untracked fullruns/ dir, so it has no
committed artifact to regenerate from. It collides with the (owned) reverse-cg 0.684, so
the string stays covered; promoting the cross-recipe run into artifacts/expB2/ would let a
future revision own it too.

Three other kinds of number stay template literals on purpose: the L2 oracle readings
("~0.99", "0.62 to 0.65", "0.60 and 0.50"), which are loose rounds or ranges over noise
draws with no single artifact field; the pre-correction numbers the audit's phrase list
already watches on index.html (0.730, 0.601, 0.639, 0.731, 0.654, 0.509); and the
percentages beside the Two Minds film, which describe the numbers burned into its frames.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from itasorl.stats import paired_contrast  # noqa: E402
from site_metrics import derive_metrics  # noqa: E402

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, metrics: dict[str, str]) -> str:
    """Fill `{{key}}` placeholders from `metrics`; raise KeyError on any unknown key."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in metrics:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no metric")
        return metrics[key]
    return _PLACEHOLDER.sub(_sub, template)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _trace_mean(doc: dict, drift: str, agent: str, metric: str) -> float:
    """Mean over the per-seed cells of a behavior-audit trace file (one cell per seed)."""
    return fmean(float(c[metric]) for c in doc["cells"] if c["drift"] == drift and c["agent"] == agent)


def derive_record_metrics(root: str | Path = ROOT) -> dict[str, str]:
    """Display strings for the result numbers the page quotes from committed artifacts
    outside the L3 headline set of `site_metrics`. Every key is formatted exactly as the
    page prints it, so a re-run that moves a number moves the page in the same step, and
    `--check` fails until the page is rebuilt. Keys are grouped by the artifact they read."""
    art = Path(root) / "artifacts"

    # FINDINGS 17.6: who drives the evaluation episodes (corrected run, drift 0.45).
    pc = _load_json(art / "policy_controls" / "corrected_l3_h8_wm.json")["aggregate"]
    scripted = [pc[f"{arm} scripted"]["mean"]
                for arm in ("untrained", "predictor", "predictor_logged", "survival")]
    # FINDINGS 17.7: the controlled persistence test (window means over the tail).
    ps = _load_json(art / "persistence" / "corrected_l3_h8_wm.json")["aggregate"]
    # FINDINGS 17.8: the best-fitting control and the stream decoder that never sees the brain.
    cd = _load_json(art / "control_diagnostics" / "corrected_l3_h8_wm.json")["aggregate"]
    # FINDINGS 17.9: the hand-written quadratic drag, read through the L3 direction and by
    # a fresh probe, with the seed-paired margins over the untrained network.
    qd = _load_json(art / "texture" / "corrected_l3_h8_wm_qd.json")["aggregate"]
    qd_transfer = paired_contrast(qd["survival transfer"]["per_seed"], qd["untrained transfer"]["per_seed"])
    qd_fresh = paired_contrast(qd["survival fresh"]["per_seed"], qd["untrained fresh"]["per_seed"])
    # FINDINGS 10.2 to 10.4: the historical hidden-8 behavior audit (pre-correction trainer).
    h8 = _load_json(art / "expB2" / "behavior_audit_l3_h8_traces.json")
    # FINDINGS 14: the graded-seam sweep's alpha = 0 anchor.
    h2 = _load_json(art / "expH2" / "summary.json")["cells"]["survival"]
    # FINDINGS 14.5: the matched-detectability jitter comparator at hidden 8.
    tk = _load_json(art / "expH2" / "texture_knockout_h8.json")
    # FINDINGS 14.7 and 22.1: the L1 organism run as first scored, its corrected-trainer
    # rerun, and the historical run re-scored on the partition used now.
    org = _load_json(art / "expL1" / "organism_summary.json")
    l1_first = fmean(float(v) for v in org["arms"][org["headline_drift"]]["survival"]["pool_target"])
    l1_oracle = float(org["gate0"]["delta"]["chosen_auroc"])
    l1c = _load_json(art / "expB2" / "corrected_L1-corrected.json")
    l1_corrected = float(l1c["arms"][l1c["dmax"]]["survival"]["pool_target"]["mean"])
    l1r = _load_json(art / "fold_rescore" / "l1_heldout.json")
    l1_rescored = fmean(float(c["explicit"]["target"]) for c in l1r["cells"]
                        if c["agent"] == "survival" and abs(float(c["drift"]) - 0.023) < 1e-9)

    return {
        "pc_predictor_replay": f"{pc['predictor replay']['mean']:.3f}",
        "pc_predictor_logged_replay": f"{pc['predictor_logged replay']['mean']:.3f}",
        "pc_scripted_lo": f"{min(scripted):.2f}",
        "pc_scripted_hi": f"{max(scripted):.2f}",
        "persist_replay_prefix": f"{ps['d=0.45 survival replay:prefix']['window_mean']:.3f}",
        "persist_reset_hidden": f"{ps['d=0.45 survival reset_hidden:prefix']['window_mean']:.3f}",
        "cd_best_control": f"{cd['d=0.45 survival resid_joint_mlp']['mean']:.3f}",
        "cd_stream_decoder": f"{cd['d=0.45 survival seq_gru']['mean']:.3f}",
        "qd_transfer": f"{qd['survival transfer']['mean']:.3f}",
        "qd_fresh": f"{qd['survival fresh']['mean']:.3f}",
        "qd_untrained_fresh": f"{qd['untrained fresh']['mean']:.3f}",
        "qd_margin_transfer": f"{qd_transfer['mean']:+.3f}",
        "qd_margin_fresh": f"{qd_fresh['mean']:+.3f}",
        "hist_untrained": f"{_trace_mean(h8, '0.45', 'untrained', 'target'):.3f}",
        "hist_resid_trace": f"{_trace_mean(h8, '0.45', 'survival', 'resid_trace'):.3f}",
        "hist_l0": f"{_trace_mean(h8, '0.00', 'survival', 'target'):.3f}",
        "hist_behavior_only_approx": f"~{_trace_mean(h8, '0.45', 'survival', 'behavior_only'):.2f}",
        "hist_resid_trace_approx": f"~{_trace_mean(h8, '0.45', 'survival', 'resid_trace'):.2f}",
        "hist_predictor_approx": f"~{_trace_mean(h8, '0.45', 'predictor', 'target'):.2f}",
        "hist_untrained_approx": f"~{_trace_mean(h8, '0.45', 'untrained', 'target'):.2f}",
        "h2_seam_floor": f"{fmean(h2['a0.00']['per_seed']):.3f}",
        "tk_gn_oracle": f"{tk['gate0']['gn']['selected_row']['oracle_auroc']:.3f}",
        "tk_gn_survival": f"{tk['channels']['gn']['survival']['mean']:.3f}",
        "l1_oracle": f"{l1_oracle:.3f}",
        "l1_oracle_hero": f"{l1_oracle:.2f}",
        "l1_survival": f"{l1_corrected:.3f}",
        "l1_survival_hero": f"{l1_corrected:.2f}",
        "l1_survival_rescored": f"{l1_rescored:.3f}",
        "l1_survival_first": f"{l1_first:.3f}",
    }


def all_metrics(root: str | Path = ROOT) -> dict[str, str]:
    """The full placeholder set: the expB2 headline metrics plus the record metrics. A key
    both sources define would let one silently shadow the other, so it raises."""
    root = Path(root)
    site = derive_metrics(root / "artifacts" / "expB2")
    record = derive_record_metrics(root)
    shared = sorted(set(site) & set(record))
    if shared:
        raise KeyError(f"metric keys defined by both site_metrics and build_index: {shared}")
    return {**site, **record}


def build(root: str | Path = ROOT, check: bool = False) -> int:
    root = Path(root)
    template = (root / "index.template.html").read_text(encoding="utf-8")
    rendered = render(template, all_metrics(root))
    target = root / "index.html"
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print("index.html is STALE - run: python scripts/build_index.py")
            return 1
        print("index.html is up to date with artifacts + template.")
        return 0
    target.write_text(rendered, encoding="utf-8", newline="")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify index.html is current; do not write")
    args = ap.parse_args()
    raise SystemExit(build(check=args.check))
