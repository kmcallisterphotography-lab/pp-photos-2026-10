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
needs_ocr = pytest.mark.skipif(
    pp.pick_ocr_backend("auto") == "none", reason="needs Apple Vision (macOS) or Tesseract to read image-only scans"
)


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


@needs_ocr
def test_every_scan_gets_the_right_file(batch):
    made = {str(p.relative_to(batch["out"])) for p in batch["out"].rglob("*.svg")}
    wanted = {o for e in batch["expected"].values() for o in e["outputs"]}
    assert made == wanted
    assert (batch["out"] / "NEEDS_REVIEW" / "scan_004-label.png").exists()
    assert batch["code"] == 2  # broken.pdf is reported as an error, the rest still convert


@needs_ocr
def test_report_lists_every_page(batch):
    rows = {(r["source_pdf"], r["page"]): r for r in batch["rows"]}
    assert rows[("scan_001.pdf", "1")]["dog_name"] == "Buddy Smith"
    assert rows[("scan_002.pdf", "1")]["text_from"] == "pdf text"
    assert rows[("scan_004.pdf", "1")]["status"] == "needs review"
    assert "numbered" in rows[("scan_005.pdf", "1")]["note"]
    assert rows[("scan_006.pdf", "2")]["dog_name"] == "Daisy O'Brien"
    assert rows[("broken.pdf", "1")]["status"] == "error"


def test_svgs_are_grayscale_paws_on_transparent_background(batch):
    expected = {o: ink for e in batch["expected"].values() for o, ink in zip(e["outputs"], e["inks"])}
    for name, ink in expected.items():
        svg = batch["out"] / name
        if not svg.exists():
            continue  # naming depends on OCR; covered by test_every_scan_gets_the_right_file
        text = svg.read_text(encoding="utf-8")
        fills = re.findall(r'fill="#([0-9A-Fa-f]{6})"', text)
        assert fills and all(c[0:2] == c[2:4] == c[4:6] for c in fills), f"{name}: not grayscale"
        px = render(svg)
        alpha = px[..., 3]
        border = np.concatenate([alpha[:3].ravel(), alpha[-3:].ravel(), alpha[:, :3].ravel(), alpha[:, -3:].ravel()])
        assert border.max() == 0, f"{name}: background is not transparent at the edges"
        opaque = px[alpha == 255][:, :3].astype(int)
        assert 0.2 < len(opaque) / alpha.size < 0.7, f"{name}: paw coverage looks wrong"
        gray = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
        assert abs(np.median(opaque) - gray) < 30, f"{name}: ink shade changed"
        assert (opaque.min(axis=1) > 200).mean() < 0.01, f"{name}: white or pale fill left in the SVG"
        # The paw is 76 mm across (82 mm with margins). A label, heart or line left in the SVG
        # would make it bigger.
        root = pp.ET.fromstring(text)
        if "- 2" not in name:  # page 2 of scan_006 has a smaller paw
            assert 80 < float(root.get("width").removesuffix("mm")) < 85, f"{name}: more than the paw"
            assert 76 < float(root.get("height").removesuffix("mm")) < 81, f"{name}: more than the paw"


def test_second_page_keeps_its_own_paw(batch):
    second = batch["out"] / "Daisy O'Brien - 2.svg"
    if not second.exists():
        pytest.skip("needs OCR naming")
    root = pp.ET.fromstring(second.read_text(encoding="utf-8"))
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


def test_dry_run_writes_nothing(batch, tmp_path):
    scans = tmp_path / "scans"
    shutil.copytree(batch["scans"], scans)
    assert pp.main([str(scans), "--dry-run", "--workers", "2"]) == 0
    assert not scans.with_name("scans-svg").exists()


def lines(*specs):
    """TextLine list from (text, height) pairs stacked down the page."""
    out, y = [], 100
    for text, height in specs:
        out.append(pp.TextLine(text, 0.95, (100, y, 100 + 20 * len(text), y + height)))
        y += height * 2
    return out


@pytest.mark.parametrize("specs, name", [
    ([("BUDDY SMITH", 80)], "Buddy Smith"),
    ([("Name: luna garcia", 60), ("Left front 10/27/2026", 30)], "Luna Garcia"),
    ([("Dog's Name -", 50), ("Rex", 50)], "Rex"),
    ([("Name: Buddy Smith 10/27/2026", 50)], "Buddy Smith"),
    ([("Name:", 50)], None),
    ([("First name: Bo", 50), ("Last name: Diddley", 50)], "Bo Diddley"),
    ([("In Loving Memory of", 40), ("Biscuit", 90)], "Biscuit"),
    ([("Owner: Jane Doe", 90), ("Pepper", 60)], "Pepper"),
    ([("Mary-Kate McDonald", 70), ("2026", 40)], "Mary-Kate McDonald"),
    ([("Bella", 70), ("Coco", 70)], None),  # two names the same size: flag, don't guess
    ([("10/27/2026", 70)], None),
    ([], None),
])
def test_choose_name(specs, name):
    assert pp.choose_name(lines(*specs), pp.Options())[0] == name


def test_low_confidence_is_flagged():
    line = pp.TextLine("Buddy", 0.3, (0, 0, 100, 50))
    name, conf, reason = pp.choose_name([line], pp.Options())
    assert name is None and "confidence" in reason


def test_name_pattern_option():
    found = lines(("Clinic: Happy Paws", 80), ("Pet: Ziggy Stardust", 40))
    assert pp.choose_name(found, pp.Options(name_pattern=r"Pet: (?P<name>.+)"))[0] == "Ziggy Stardust"


@pytest.mark.parametrize("raw, clean", [
    ("o'brien", "O'Brien"),
    ("Max/Power", "Max Power"),
    ("  Daisy   Mae. ", "Daisy Mae"),
    ("McDONALD", "McDONALD"),  # mixed case is kept as printed
    ("Chloé", "Chloé"),
])
def test_clean_name(raw, clean):
    assert pp.clean_name(raw) == clean


def test_row_fragments_merge():
    parts = [pp.TextLine("Smith", 0.9, (300, 100, 420, 150)), pp.TextLine("Buddy", 0.9, (150, 102, 270, 150))]
    assert [l.text for l in pp.merge_row_fragments(parts)] == ["Buddy Smith"]


def test_background_check_catches_full_canvas_shape():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="10mm" viewBox="0 0 100 100">'
           '<path d="M0 0 L100 0 L100 100 L0 100 Z" fill="#FFFFFF"/></svg>')
    assert "covers the whole image" in pp.background_problem(svg)


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
    assert pp.paw_region(ink, 300) is None
