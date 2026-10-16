"""Two Minds: the renderer's on-screen numbers and wording, and the brain-data dump checks.

Runs in CI with no agents, no dumps and no torch. The number check reads brain.js and the
committed corrected-trainer artifacts; the dump check runs on a small synthetic pool with
an explicit feature function (itasorl.experiment_b.episode_features imports torch).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dump_brain_film_data as dbf  # noqa: E402

BRAIN_JS = ROOT / "viz" / "player" / "brain" / "brain.js"
BRAIN_DATA = ROOT / "viz" / "player" / "brain" / "brain-data.js"


def _features(H):
    """itasorl.experiment_b.episode_features (mean and final state), without torch."""
    return np.concatenate([H.mean(1), H[:, -1]], axis=-1)


def _film(**over):
    film = {"untrained": (0.523, "52%"), "trained": (0.733, "73%"), "replay": (0.721, "72%"),
            "noAux300": (0.613, "61%"), "noAux450": (0.685, "69%"), "l0": (0.559, "56%"),
            "bar": (0.65, "65%")}
    film.update(over)
    return film


def _record(**over):
    rec = {"untrained": 0.522603, "untrained_t90": [0.4914, 0.553806], "trained": 0.733017,
           "replay": 0.720992, "noAux300": 0.613471, "noAux450": 0.685372, "l0": 0.559421,
           "l0_pass": False, "c1_met": False, "bar": 0.65}
    rec.update(over)
    return rec


# ---------------------------------------------------------------- the numbers

def test_committed_renderer_numbers_match_the_corrected_record():
    """What brain.js puts on screen agrees with the committed corrected-trainer artifacts."""
    dbf.verify_film_numbers(dbf.renderer_numbers(BRAIN_JS), dbf.load_record(ROOT))


def test_verify_passes_on_published_values():
    got = dbf.verify_film_numbers(_film(), _record())
    assert got["trained"] == 0.733 and got["untrained"] == 0.5226


@pytest.mark.parametrize("key,bad", [
    ("trained", (0.726, "73%")),     # the pre-correction residual behind the same whole percent
    ("untrained", (0.488, "49%")),   # the pre-correction floor
    ("replay", (0.721, "73%")),
    ("noAux450", (0.717, "72%")),    # the pre-correction 450-update reading
])
def test_verify_fails_on_a_drifted_number(key, bad):
    with pytest.raises(SystemExit):
        dbf.verify_film_numbers(_film(**{key: bad}), _record())


def test_verify_fails_when_a_number_is_missing():
    film = _film()
    del film["l0"]
    with pytest.raises(SystemExit):
        dbf.verify_film_numbers(film, _record())


def test_verify_fails_when_untrained_leaves_chance():
    with pytest.raises(SystemExit):
        dbf.verify_film_numbers(_film(untrained=(0.6, "60%")),
                                _record(untrained=0.6, untrained_t90=[0.57, 0.63]))


def test_verify_fails_when_the_preregistered_test_passes():
    with pytest.raises(SystemExit):
        dbf.verify_film_numbers(_film(), _record(l0_pass=True))


def test_verify_fails_when_the_arm_without_prediction_clears_the_bar_at_300():
    with pytest.raises(SystemExit):
        dbf.verify_film_numbers(_film(noAux300=(0.66, "66%")), _record(noAux300=0.66))


def test_renderer_numbers_refuse_a_reshaped_line(tmp_path):
    src = BRAIN_JS.read_text(encoding="utf-8").replace(
        'replay: { value: 0.721, display: "72%" },', 'replay: { display: "72%", value: 0.721 },')
    p = tmp_path / "brain.js"
    p.write_text(src, encoding="utf-8")
    with pytest.raises(SystemExit):
        dbf.renderer_numbers(p)


# ------------------------------------------------------------------ the brain

def test_representative_seed_is_the_committed_seed_closest_to_the_mean():
    rec = dbf.load_record(ROOT)
    assert dbf.representative_seed(rec["cells"], rec["trained"]) == 9
    assert all(c["gae_bootstrap"] == "successor" for c in rec["cells"].values())
    assert dbf.representative_seed({3: {"survival": 0.5}, 1: {"survival": 0.75}}, 0.625) == 1  # tie


def test_unit_auroc_orientation_and_ties():
    a, s = np.zeros((4, 3)), np.zeros((4, 3))
    s[:, 0] = 1.0   # unit 0 runs hotter in the fake world
    a[:, 1] = 1.0   # unit 1 runs hotter in the real world
    assert dbf.unit_auroc(a, s).tolist() == [1.0, 0.0, 0.5]


def _save_pool(path, shift=1.0, drift=0.45, seed=0):
    rng = np.random.default_rng(seed)
    ha = rng.normal(size=(24, 4, 6))
    hs = rng.normal(size=(24, 4, 6))
    hs[..., 0] += shift
    np.savez_compressed(path, Ha=ha, Hs=hs, drift_sigma=np.float64(drift), steps=np.int64(4))
    return ha, hs


def test_dump_check_accepts_a_pool_that_reproduces_and_refuses_one_that_does_not(tmp_path):
    ha, hs = _save_pool(tmp_path / "states_d0.45_s9_survival.npz")
    target = dbf.pool_target(ha, hs, _features)
    assert 0.5 < target <= 1.0
    _, _, got = dbf.load_checked_pool(tmp_path, 9, "survival", target, 1e-9, _features)
    assert got == target
    with pytest.raises(SystemExit):
        dbf.load_checked_pool(tmp_path, 9, "survival", target + 0.01, 1e-9, _features)


def test_dump_check_refuses_a_pool_from_the_wrong_drift(tmp_path):
    ha, hs = _save_pool(tmp_path / "states_d0.45_s9_survival.npz", drift=0.0)
    with pytest.raises(SystemExit):
        dbf.load_checked_pool(tmp_path, 9, "survival", dbf.pool_target(ha, hs, _features), 1e-9, _features)


def test_dump_check_refuses_a_missing_pool(tmp_path):
    with pytest.raises(SystemExit):
        dbf.load_checked_pool(tmp_path, 9, "untrained", 0.5, 1e-9, _features)


def test_check_report_needs_every_cell_reproduced(tmp_path):
    cells = [{"drift": 0.45, "seed": 9, "arm": a, "reproduced": True} for a in ("survival", "untrained")]
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"all_reproduced": True, "cells": cells}), encoding="utf-8")
    dbf.check_report(good, 9)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"all_reproduced": False, "cells": cells}), encoding="utf-8")
    with pytest.raises(SystemExit):
        dbf.check_report(bad, 9)
    with pytest.raises(SystemExit):
        dbf.check_report(good, 3)


def test_brain_data_keeps_the_ring_cells_on_screen():
    rng = np.random.default_rng(1)
    fa, fs = rng.normal(size=(200, 96)), rng.normal(size=(200, 96))
    fs[:, 5] += 2.0   # one unit that clearly separates the worlds
    ua, us = rng.normal(size=(200, 96)), rng.normal(size=(200, 96))
    out = dbf.brain_data(fa, fs, ua, us)
    assert len(out["display_units"]) == 12 and len(out["ring_units"]) == 6
    assert set(out["ring_units"]) <= set(out["display_units"])
    assert out["ring_units"][0] == 5
    assert max(out["unit_signal"]) == pytest.approx(1.0, abs=1e-3)


def _in_git_checkout() -> bool:
    """True when ROOT is the top of a git checkout (an extracted archive placed inside some
    other repository is still an extracted archive)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and Path(top).resolve() == ROOT


def _brain_data_js():
    # The anonymized supplement ships brain.js but not brain-data.js, so outside a git
    # checkout this check skips; in a checkout a missing file still fails.
    if not BRAIN_DATA.exists() and not _in_git_checkout():
        pytest.skip("viz/player/brain/brain-data.js is not shipped in the anonymized supplement")
    txt = BRAIN_DATA.read_text(encoding="utf-8")
    return json.loads(txt.split("window.BRAIN_DATA = ", 1)[1].rsplit(";", 1)[0])


def test_committed_brain_data_comes_from_the_corrected_c1_brain():
    data = _brain_data_js()
    if data.get("trainer") != "corrected":
        pytest.skip("brain-data.js still holds the pre-correction pools; rebuild it "
                    "(assets/film/README.md, Two Minds)")
    rec = dbf.load_record(ROOT)
    seed = dbf.representative_seed(rec["cells"], rec["trained"])
    assert data["seed"] == seed
    cell = rec["cells"][seed]
    # The committed readings are carried exactly; the rebuilt pools agree with them within the
    # recorded tolerance (a rebuild on another software stack is close, not bit-identical),
    # and that tolerance is never loose enough to hide a different brain.
    assert data["committed_seed_pooled_target"] == round(cell["survival"], 4)
    assert data["committed_untrained_pooled_target"] == round(cell["untrained"], 4)
    atol = float(data["rebuild_atol"])
    assert atol <= 0.005
    assert abs(data["seed_pooled_target"] - cell["survival"]) <= atol + 5e-5
    assert abs(data["untrained_pooled_target"] - cell["untrained"]) <= atol + 5e-5
    assert data["record_mean"] == round(rec["trained"], 4)
    assert data["film_numbers"] == dbf.verify_film_numbers(dbf.renderer_numbers(BRAIN_JS), rec)
    assert len(data["display_units"]) == 12 and set(data["ring_units"]) <= set(data["display_units"])


# ------------------------------------------------------------ what it says

def _rendered(src: str) -> str:
    """brain.js without its comments: the code that can put words on screen."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    keep = [re.sub(r"\s//\s.*$", "", ln) for ln in src.splitlines() if not ln.lstrip().startswith("//")]
    return "\n".join(keep)


def test_the_film_renders_no_withdrawn_claim_or_stack_name():
    code = _rendered(BRAIN_JS.read_text(encoding="utf-8"))
    for banned in ("AUROC", "A2C", "0.726", "0.752", "0.488", "49%", "75%",
                   "hold the clue", "holds the clue", "kept a trace", "leaves a mark",
                   "memory picks up", "survival pressure", "survival-specific",
                   "behavior-independent", "behaviour-independent", "Cover the eyes"):
        assert banned not in code, banned
    assert not re.search(r"\bGRU\b", code)


def test_every_percent_on_screen_is_a_checked_number():
    code = _rendered(BRAIN_JS.read_text(encoding="utf-8"))
    checked = {display for _, display in dbf.renderer_numbers(BRAIN_JS).values()}
    shown = set(re.findall(r"\d+%", code))
    assert shown <= checked | {"50%"}, shown - checked


def test_labeled_builds_ship_no_narration():
    """House rule: narration renders only with caption=1; the voice is recorded over the
    labeled build."""
    src = BRAIN_JS.read_text(encoding="utf-8")
    render = src[src.index("function render(t)"):]
    assert render.count("drawNarration(") == 1 and render.count("drawCaptionBand(") == 1
    assert re.search(r"if \(SHOW_CAPTION && SHOW_LABELS && !CARD\) \{\s*drawCaptionBand\(\);"
                     r"\s*drawNarration\(t\);\s*\}", render)


def test_the_memory_readout_shows_two_brains_not_one_climbing_meter():
    src = BRAIN_JS.read_text(encoding="utf-8")
    assert 'label: "untrained brains"' in src and 'label: "brains like this one"' in src
    assert not re.search(r"lerp\([^)]*untrained[^)]*trained", src)
