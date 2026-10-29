"""Make fake paw-print scans for testing: textured ink paws and printed labels on off-white paper.

    python tests/make_samples.py OUT_DIR
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter

DPI = 300
PAGE_W, PAGE_H = 612, 792  # US Letter, points
PAPER = (246, 243, 236)
NAVY = (38, 34, 70)
BROWN = (92, 48, 30)
BLACK = (28, 28, 30)
RED = (200, 30, 45)


def paw_image(ink: tuple[int, int, int], seed: int, size_px: int = 900, faint: bool = False) -> Image.Image:
    """RGBA paw: a main pad and four toes with mottled ink and paper-colored skin gaps."""
    rng = np.random.default_rng(seed)
    s = size_px
    shape = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(shape)
    d.ellipse((0.22 * s, 0.45 * s, 0.78 * s, 0.95 * s), fill=255)
    d.polygon([(0.30 * s, 0.62 * s), (0.70 * s, 0.62 * s), (0.82 * s, 0.90 * s), (0.18 * s, 0.90 * s)], fill=255)
    for cx, cy in ((0.13, 0.42), (0.36, 0.18), (0.64, 0.18), (0.87, 0.42)):
        d.ellipse(((cx - 0.11) * s, (cy - 0.14) * s, (cx + 0.11) * s, (cy + 0.14) * s), fill=255)
    shape = shape.filter(ImageFilter.GaussianBlur(3))
    # Skin texture: little paper-colored gaps where the ink did not take.
    gaps = Image.new("L", (s, s), 0)
    g = ImageDraw.Draw(gaps)
    for _ in range(900 if faint else 260):
        x, y, r = rng.uniform(0, s), rng.uniform(0, s), rng.uniform(2, 7)
        g.ellipse((x - r, y - r, x + r, y + r), fill=255)
    alpha = np.asarray(shape, np.float32) / 255 * (1 - np.asarray(gaps, np.float32) / 255 * 0.85)
    if faint:  # light pressure: patchy, half-strength ink
        alpha *= 0.55 + 0.45 * (rng.random((s, s)) > 0.5)
    shade = 1 + rng.normal(0, 0.12, (s // 30 + 1, s // 30 + 1))
    shade = np.asarray(Image.fromarray(shade.astype(np.float32)).resize((s, s), Image.BILINEAR))
    color = np.clip(np.array(ink, np.float32)[None, None, :] * shade[..., None], 0, 255)
    rgba = np.dstack([color, alpha * 255]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def scan_page(paws: list[tuple[Image.Image, tuple[int, int]]], frame: tuple | None = None) -> Image.Image:
    page = Image.new("RGB", (PAGE_W * DPI // 72, PAGE_H * DPI // 72), PAPER)
    for paw, xy in paws:
        page.paste(paw, xy, paw)
    if frame:
        ImageDraw.Draw(page).rectangle(frame, outline=(120, 120, 120), width=4)
    return page


def card_page(lid: tuple, card: tuple, paw: Image.Image) -> Image.Image:
    """A 4x6" card on the scanner glass: lid all around, shadow lines on the card's top and left edges."""
    page = Image.new("RGB", (PAGE_W * DPI // 72, PAGE_H * DPI // 72), lid)
    d = ImageDraw.Draw(page)
    x0, y0, x1, y1 = 150, 150, 150 + 4 * DPI, 150 + 6 * DPI
    d.rectangle((x0, y0, x1, y1), fill=card)
    d.line([(x0, y1), (x0, y0), (x1, y0)], fill=(70, 70, 70), width=5)
    page.paste(paw, (x0 + 150, y0 + 250), paw)
    return page


def heart(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color, width: int | None = None) -> None:
    """A drawn heart, filled, or as an outline when width is given."""
    r = size // 4
    if width:
        draw.arc((x, y, x + 2 * r, y + 2 * r), 135, 360, fill=color, width=width)
        draw.arc((x + 2 * r, y, x + 4 * r, y + 2 * r), 180, 45, fill=color, width=width)
        draw.line([(x + 0.3 * r, y + 1.7 * r), (x + 2 * r, y + size), (x + 3.7 * r, y + 1.7 * r)], fill=color, width=width)
    else:
        draw.ellipse((x, y, x + 2 * r, y + 2 * r), fill=color)
        draw.ellipse((x + 2 * r, y, x + 4 * r, y + 2 * r), fill=color)
        draw.polygon([(x + 0.1 * r, y + 1.4 * r), (x + 2 * r, y + size), (x + 3.9 * r, y + 1.4 * r)], fill=color)


def decorate(page: Image.Image) -> Image.Image:
    """What staff add to a sheet: hearts, a pen line, a marker stroke."""
    d = ImageDraw.Draw(page)
    heart(d, 1750, 1950, 260, RED)  # red heart near the paw
    heart(d, 250, 250, 300, BLACK, width=7)  # outlined heart
    heart(d, 2050, 250, 220, BLACK)  # solid black heart, away from the paw
    d.line([(150, 2300), (2400, 2300)], fill=BLACK, width=5)  # pen line
    d.line([(250, 1700), (650, 2050)], fill=BLACK, width=18)  # 1.5 mm marker stroke
    return page


def add_scan(doc: pymupdf.Document, page_img: Image.Image, labels: list[tuple[str, float, float, float]],
             searchable: bool, seed: int) -> None:
    """Put the 'scanned' page in the PDF. labels: (text, x_pt, y_pt, fontsize)."""
    # Print the label onto the paper, then "scan" it: noise + JPEG like a real scanner.
    tmp = pymupdf.open()
    p = tmp.new_page(width=PAGE_W, height=PAGE_H)
    buf = io.BytesIO()
    page_img.save(buf, "PNG")
    p.insert_image(p.rect, stream=buf.getvalue())
    for text, x, y, size in labels:
        p.insert_text((x, y), text, fontsize=size, fontname="helv", color=(0.1, 0.1, 0.1))
    pix = p.get_pixmap(dpi=DPI)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(np.int16)
    rgb = np.clip(rgb + np.random.default_rng(seed).normal(0, 2.5, rgb.shape), 0, 255).astype(np.uint8)
    jpg = io.BytesIO()
    Image.fromarray(rgb).save(jpg, "JPEG", quality=85)

    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=jpg.getvalue())
    if searchable:  # invisible OCR text layer, the way scanners make "searchable PDFs"
        for text, x, y, size in labels:
            page.insert_text((x, y), text, fontsize=size, fontname="helv", render_mode=3)


def make_samples(out: Path) -> dict[str, dict]:
    """Write the test PDFs into out/ and return what each one should produce."""
    out.mkdir(parents=True, exist_ok=True)
    expected = {}

    def save(name: str, doc: pymupdf.Document, **expect) -> None:
        doc.save(out / name)
        expected[name] = expect

    doc = pymupdf.open()
    add_scan(doc, scan_page([(paw_image(NAVY, 1), (820, 600))]), [("Buddy Smith", 200, 600, 30)], False, 1)
    save("scan_001.pdf", doc, outputs=["Buddy Smith.svg"], inks=[NAVY])

    doc = pymupdf.open()
    add_scan(doc, scan_page([(paw_image(BROWN, 2), (820, 500))]),
             [("Name: Luna Garcia", 180, 560, 26), ("Left front   10/27/2026", 180, 600, 12)], True, 2)
    save("scan_002.pdf", doc, outputs=["Luna Garcia.svg"], inks=[BROWN])

    doc = pymupdf.open()
    add_scan(doc, scan_page([(paw_image(NAVY, 3), (820, 900))]), [("MAX", 260, 160, 40)], False, 3)
    save("scan_003.pdf", doc, outputs=["Max.svg"], inks=[NAVY])

    doc = pymupdf.open()
    add_scan(doc, scan_page([(paw_image(BROWN, 4), (820, 900))]), [], False, 4)
    save("scan_004.pdf", doc, outputs=["NEEDS_REVIEW/scan_004.svg"], inks=[BROWN])

    doc = pymupdf.open()  # second dog named Max; label in a sticker frame
    add_scan(doc, scan_page([(paw_image(NAVY, 5), (820, 500))], frame=(700, 2380, 1850, 2700)),
             [("Max", 230, 620, 34)], False, 5)
    save("scan_005.pdf", doc, outputs=["Max (2).svg"], inks=[NAVY])

    doc = pymupdf.open()  # front and back paw on two pages, label only on page 1
    add_scan(doc, scan_page([(paw_image(BROWN, 6), (820, 500))]), [("Daisy O'Brien", 200, 600, 30)], True, 6)
    add_scan(doc, scan_page([(paw_image(NAVY, 7, 700), (900, 900))]), [], False, 7)
    save("scan_006.pdf", doc, outputs=["Daisy O'Brien - 1.svg", "Daisy O'Brien - 2.svg"], inks=[BROWN, NAVY])

    doc = pymupdf.open()  # black ink plus the hearts and lines staff draw on the sheet
    add_scan(doc, decorate(scan_page([(paw_image(BLACK, 8), (820, 600))])), [("Rosie Parker", 200, 640, 30)], False, 8)
    save("scan_007.pdf", doc, outputs=["Rosie Parker.svg"], inks=[BLACK])

    doc = pymupdf.open()  # white card on a grayer scanner lid, with a faint print
    add_scan(doc, card_page((226, 226, 224), (254, 254, 253), paw_image(BLACK, 9, faint=True)),
             [("Brownie Cole", 80, 400, 26)], False, 9)
    save("scan_008.pdf", doc, outputs=["Brownie Cole.svg"], inks=[BLACK])

    doc = pymupdf.open()  # cream card on a white lid
    add_scan(doc, card_page((250, 250, 250), (240, 232, 214), paw_image(BLACK, 10)),
             [("Pepper Lane", 80, 400, 26)], False, 10)
    save("scan_009.pdf", doc, outputs=["Pepper Lane.svg"], inks=[BLACK])

    (out / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    expected["broken.pdf"] = dict(outputs=[], inks=[])
    return expected


if __name__ == "__main__":
    for name, expect in make_samples(Path(sys.argv[1])).items():
        print(name, "->", expect["outputs"])
