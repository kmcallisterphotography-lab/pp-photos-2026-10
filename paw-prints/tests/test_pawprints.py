"""End-to-end and unit tests. Run from paw-prints/:  .venv/bin/python -m pytest tests"""

from __future__ import annotations

import csv
import io
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pawprints as pp  # noqa: E402
from make_samples import make_samples  # noqa: E402

resvg_py = pytest.importorskip("resvg_py")


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    scans = tmp_path_factory.mktemp("batch") / "scans"
    expected = make_samples(scans)
    code = pp.main([str(scans), "--workers", "2"])
    out = scans.with_name("scans-svg")
    with open(out / "report.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return dict(scans=scans, out=out, expected=expected, rows=rows, code=code)


def render(svg_path: Path, width: int = 400) -> np.ndarray:
    png = resvg_py.svg_to_bytes(svg_string=svg_path.read_text(encoding="utf-8"), dpi=96, width=width)
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGBA"))


def test_every_pdf_becomes_svgs_named_after_it(batch):
    made = {str(p.relative_to(batch["out"])) for p in batch["out"].rglob("*.svg")}
    wanted = {o for e in batch["expected"].values() for o in e["outputs"]}
    assert made == wanted  # nothing landed in NEEDS_REVIEW either
    assert batch["code"] == 2  # broken.pdf is reported as an error, the rest still convert


def test_report_lists_every_page(batch):
    rows = {(r["source_pdf"], r["page"]): r for r in batch["rows"]}
    assert rows[("Buddy Smith.pdf", "1")]["output_svg"] == "Buddy Smith.svg"
    assert rows[("Daisy O'Brien.pdf", "2")]["output_svg"] == "Daisy O'Brien - 2.svg"
    assert rows[("broken.pdf", "1")]["status"] == "error"


def test_svgs_are_grayscale_paws_on_transparent_background(batch):
    expected = {o: ink for e in batch["expected"].values() for o, ink in zip(e["outputs"], e["inks"])}
    for name, ink in expected.items():
        svg = batch["out"] / name
        text = svg.read_text(encoding="utf-8")
        fills = re.findall(r'fill="#([0-9A-Fa-f]{6})"', text)
        assert fills and all(c[0:2] == c[2:4] == c[4:6] for c in fills), f"{name}: not grayscale"
        px = render(svg)
        alpha = px[..., 3]
        border = np.concatenate([alpha[:3].ravel(), alpha[-3:].ravel(), alpha[:, :3].ravel(), alpha[:, -3:].ravel()])
        assert border.max() == 0, f"{name}: background is not transparent at the edges"
        opaque = px[alpha == 255][:, :3].astype(int)
        assert 0.15 < len(opaque) / alpha.size < 0.7, f"{name}: paw coverage looks wrong"
        gray = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
        if name != "Brownie Cole.svg":  # the faint print is lighter than its ink
            assert abs(np.median(opaque) - gray) < 30, f"{name}: ink shade changed"
        assert (opaque.min(axis=1) > 240).mean() < 0.01, f"{name}: paper-white fill left in the SVG"
        # The paw is 76 mm across (82 mm with margins). A label, heart, line or card left in the
        # SVG would make it bigger.
        root = pp.ET.fromstring(text)
        if "- 2" not in name:  # page 2 of Daisy O'Brien has a smaller paw
            assert 80 < float(root.get("width").removesuffix("mm")) < 85, f"{name}: more than the paw"
            assert 76 < float(root.get("height").removesuffix("mm")) < 81, f"{name}: more than the paw"


def test_second_page_keeps_its_own_paw(batch):
    root = pp.ET.fromstring((batch["out"] / "Daisy O'Brien - 2.svg").read_text(encoding="utf-8"))
    assert 55 < float(root.get("width").removesuffix("mm")) < 75  # the smaller 59 mm paw on page 2


def test_rerun_never_overwrites(batch, tmp_path):
    scans = tmp_path / "scans"
    shutil.copytree(batch["scans"], scans)
    first = scans.with_name("scans-svg")
    shutil.copytree(batch["out"], first)
    before = {p: p.read_bytes() for p in first.rglob("*") if p.is_file()}
    pp.main([str(scans), "--workers", "2"])
    assert {p: p.read_bytes() for p in first.rglob("*") if p.is_file()} == before
    assert (tmp_path / "scans-svg (2)" / "report.csv").exists()


def test_same_name_in_two_subfolders_gets_numbered(batch, tmp_path):
    scans = tmp_path / "scans"
    for sub in ("monday", "tuesday"):
        (scans / sub).mkdir(parents=True)
        shutil.copy(batch["scans"] / "Coco.pdf", scans / sub / "Coco.pdf")
    pp.main([str(scans), "--workers", "2"])
    assert sorted(p.name for p in (tmp_path / "scans-svg").glob("*.svg")) == ["Coco (2).svg", "Coco.svg"]


def test_colored_marks_dropped_only_when_paw_ink_is_gray():
    rgb = np.full((40, 40, 3), 240, np.uint8)
    rgb[:30, :30] = (30, 30, 32)  # black paw ink
    rgb[32:, 32:] = (200, 30, 45)  # red heart
    ink = rgb.min(axis=2) < 200
    kept = pp.drop_colored_marks(ink, rgb)
    assert kept[:30, :30].all() and not kept[32:, 32:].any()
    rgb[:30, :30] = (92, 48, 30)  # paw printed in brown ink: color can't separate it, keep all
    assert pp.drop_colored_marks(ink, rgb).sum() == ink.sum()


def test_lines_alone_are_not_a_paw():
    ink = np.zeros((1000, 1000), bool)
    ink[500:506, 100:900] = True  # a 0.5 mm pen line
    ink[100:900, 200:206] = True
    assert pp.paw_region(ink, 300, 1.0, 0.35) is None
    assert pp.paw_region(ink, 300, 2.5, 0.2) is None


def test_label_text_block_is_not_a_paw():
    # Three lines of bold label text, like a pharmacy sticker: they fill in to a solid block, but
    # unlike a pad there is no solid ink in it.
    ink = np.zeros((600, 900), bool)
    for top in (100, 160, 220):
        for left in range(100, 700, 22):
            ink[top:top + 36, left:left + 6] = True  # letter stems, ~0.5 mm strokes
            ink[top:top + 6, left:left + 16] = True
            ink[top + 30:top + 36, left:left + 16] = True
    assert pp.paw_region(ink, 300, 1.0, 0.35) is None
    assert pp.paw_region(ink, 300, 2.5, 0.2) is None


def test_background_check_catches_full_canvas_shape():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="10mm" viewBox="0 0 100 100">'
           '<path d="M0 0 L100 0 L100 100 L0 100 Z" fill="#FFFFFF"/></svg>')
    assert "covers the whole image" in pp.background_problem(svg)


def test_background_check_catches_pale_card_but_not_a_dark_paw():
    head = '<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="10mm" viewBox="0 0 1261 1865">'
    card = '<path d="M0 0 L1189 0 L1189 1793 L0 1793 Z" fill="#FEFEFE" transform="translate(36,36)"/>'
    paw = '<path d="M0 0 L1100 0 L1100 1700 L0 1700 Z" fill="#1C1C1C" transform="translate(80,80)"/>'
    assert "pale shape" in pp.background_problem(head + card + "</svg>")  # 94% wide: the Brown - 2 case
    assert pp.background_problem(head + paw + "</svg>") == ""
