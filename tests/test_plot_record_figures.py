"""The record figures (scripts/plot_record_figures.py) are drawn from committed artifacts only.

Review findings R141 to R144: the Experiment A and B figures had been drawn by an earlier run
than the one the tables quote, carried the project name in a raster title, left out the speed
control their caption describes, and printed their type at 4 pt. These tests pin each of
those properties: no title and no project name in any figure, the plotted values equal the
committed artifacts and the record's tables, every text artist at the size asked for, and the
committed copies drawn by this script rather than by a run script.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_record_figures as prf  # noqa: E402

NAMES = sorted(prf.FIGURES)


def _build(name, fontsize=prf.PRINT_FONT):
    w, h = prf.PRINT_SIZES[name]
    fig = prf.FIGURES[name][1](w, h, fontsize)
    fig.draw_without_rendering()        # creates the tick labels and runs the layout
    return fig


def _texts(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _lines(fig):
    return {ln.get_gid(): list(ln.get_ydata()) for ax in fig.axes for ln in ax.get_lines()
            if ln.get_gid()}


def _png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", head[16:24])


@pytest.mark.parametrize("name", NAMES)
def test_no_title_and_no_project_name(name):
    fig = _build(name)
    assert getattr(fig, "_suptitle", None) is None
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            assert ax.get_title(loc=loc) == "", f"{name} has a {loc} title"
    for t in _texts(fig):
        assert "itasorl" not in t.get_text().lower(), f"{name} prints the project name"


@pytest.mark.parametrize("name", NAMES)
def test_type_is_drawn_at_the_size_asked_for(name):
    fig = _build(name, fontsize=7.5)
    sizes = {round(t.get_fontsize(), 2) for t in _texts(fig)}
    assert sizes and min(sizes) >= 7.0 and max(sizes) <= 7.5, (name, sorted(sizes))


def test_plotted_values_are_the_committed_artifacts():
    a = prf.load_artifact("expA/summary.json")
    b = prf.load_artifact("expB/summary.json")
    l1 = sorted(a["expA_l1"]["calibration"], key=lambda r: r["delta"])
    l2 = sorted(a["expA_l2"]["calibration"], key=lambda r: r["drift_sigma"])
    assert _lines(_build("expA_ceiling"))["expA_l1.oracle_auroc"] == [r["oracle_auroc"] for r in l1]
    assert _lines(_build("expA_L2_ceiling"))["expA_l2.oracle_auroc"] == [
        r["oracle_auroc"] for r in l2]
    sweep = b["expB_full"]["drift_sweep"]
    got = _lines(_build("expB_incidental"))
    for key in ("target_mean", "shuffled_mean", "speed_mean"):
        assert got[f"expB_full.{key}"] == [r[key] for r in sweep], key
    got = _lines(_build("expB_channels"))
    assert got["expB_surprise.mean"] == [r["mean"] for r in b["expB_surprise"]["drift_sweep_with_std"]]
    assert got["expB_full.target_mean"] == [r["target_mean"] for r in sweep]
    rerun = b["expB_kstep_rerun_20260713"]["horizons"]
    got = _lines(_build("expB_kstep"))
    assert got["expB_kstep.drift_045_target_mean"] == [r["drift_045_target_mean"] for r in rerun]
    assert got["expB_kstep.control_target_mean"] == [r["control_target_mean"] for r in rerun]


def test_plotted_values_match_the_record_tables():
    """The numbers FINDINGS 2 and 3 and the artifacts README quote, read back off the figures."""
    r3 = lambda xs: [round(v, 3) for v in xs]  # noqa: E731
    assert r3(_lines(_build("expA_L2_ceiling"))["expA_l2.oracle_auroc"])[:4] == [
        0.44, 0.706, 0.881, 0.991]
    assert r3(_lines(_build("expA_ceiling"))["expA_l1.oracle_auroc"])[:3] == [0.508, 0.402, 0.576]
    inc = _lines(_build("expB_incidental"))
    assert r3(inc["expB_full.speed_mean"]) == [0.913, 0.898, 0.902]
    assert r3(inc["expB_full.target_mean"])[1:] == [0.509, 0.510]
    assert r3(_lines(_build("expB_channels"))["expB_surprise.mean"]) == [0.511, 0.444, 0.596]
    ks = _lines(_build("expB_kstep"))
    assert r3(ks["expB_kstep.drift_045_target_mean"]) == [0.506, 0.480, 0.484]
    assert r3(ks["expB_kstep.control_target_mean"]) == [0.453, 0.448, 0.410]
    s = prf.expB2_series()
    assert round(s["survival"]["mean"][-1], 3) == 0.523 and round(s["survival"]["std"][-1], 3) == 0.045


def test_the_speed_control_is_in_the_incidental_figure():
    labels = [t.get_text() for t in _texts(_build("expB_incidental"))]
    assert any("speed" in s for s in labels)
    assert any("shuffled" in s for s in labels)


def test_smoke_every_plot_function_writes_a_file(tmp_path):
    assert prf.main(["--out-dir", str(tmp_path / "print"), "--size", "print"]) == 0
    for name in NAMES:
        out = tmp_path / "print" / f"{name}.png"
        w, h = prf.PRINT_SIZES[name]
        assert _png_size(out) == (round(w * prf.PRINT_DPI), round(h * prf.PRINT_DPI)), name
        assert b"itasorl" not in out.read_bytes().lower(), name
    for name, (plot, _, _) in prf.FIGURES.items():
        out = plot(tmp_path / "web" / f"{name}.png")
        assert _png_size(out) == (round(prf.WEB_SIZE[0] * prf.WEB_DPI),
                                  round(prf.WEB_SIZE[1] * prf.WEB_DPI)), name
    pdf = prf.plot_expB_kstep(tmp_path / "kstep.pdf", 3.9, 2.5, 7.5)
    assert pdf.read_bytes()[:5] == b"%PDF-"


def test_print_size_never_overwrites_the_committed_copies():
    with pytest.raises(SystemExit):
        prf.main(["--size", "print"])
    with pytest.raises(SystemExit):
        prf.main(["--format", "pdf"])


def test_committed_copies_are_drawn_by_this_script():
    """A run script (run_expA.py, run_expB_*.py, run_e2e.py) writes 910 x 546 px titled images
    over docs/figures; the committed copies must be this script's web size."""
    want = (round(prf.WEB_SIZE[0] * prf.WEB_DPI), round(prf.WEB_SIZE[1] * prf.WEB_DPI))
    for name, (_, _, committed) in prf.FIGURES.items():
        assert _png_size(ROOT / committed) == want, f"{committed}: rerun scripts/plot_record_figures.py"


def test_script_reads_nothing_outside_artifacts():
    src = (ROOT / "scripts" / "plot_record_figures.py").read_text(encoding="utf-8")
    assert "docs/paper" not in src and "docs.paper" not in src
    assert "fullruns" not in src
