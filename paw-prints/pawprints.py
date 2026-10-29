#!/usr/bin/env python3
"""Bulk-convert paw-print PDF scans into grayscale SVGs with a transparent background.

Each PDF is already named after the patient (e.g. "Brown.pdf"). For every page the script:

  1. renders the page and finds the ink: anything darker than the paper around it,
  2. keeps only the paw: drops colored marks, thin lines, writing, and anything outside
     the outline of the main pad and its toe pads (pages with no paw are skipped),
  3. removes the paper so only the paw is left, in grayscale,
  4. traces it with vtracer and saves "<PDF name>.svg" ("<PDF name> - 2.svg" for a second page).

Usage:
    ./pawprints.py ~/Desktop/paw-scans    # writes ~/Desktop/paw-scans-svg/

Run ./pawprints.py --help for all options.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
import vtracer
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

VERSION = "2026-10-29b"
REVIEW_DIR = "NEEDS_REVIEW"
REPORT_NAME = "report.csv"

# Tracing settings. Tracing at 2x the scan resolution keeps fine skin texture and fur strokes
# crisp; higher detail also keeps more shades of gray and makes bigger files.
DETAIL_PRESETS = {
    "low": dict(scale=1, filter_speckle=2, color_precision=7, layer_difference=8),
    "medium": dict(scale=2, filter_speckle=4, color_precision=7, layer_difference=10),
    "high": dict(scale=2, filter_speckle=2, color_precision=8, layer_difference=4),
}


@dataclass
class Options:
    dpi: int = 300
    detail: str = "medium"
    threshold: float | None = None


@dataclass
class PageResult:
    source: str
    page: int
    page_count: int
    status: str  # "ok", "review", "skipped" or "error"
    reason: str = ""
    svg: str | None = None
    preview_png: bytes | None = None


# ---------------------------------------------------------------- finding the ink


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def paper_brightness(lum: np.ndarray, dpi: int) -> np.ndarray:
    """How bright the paper is around each pixel: the bright end of a ~8 cm neighborhood.

    Measured locally, not from the page edge, because a card or sheet often sits on a scanner
    lid of a slightly different white. Comparing against the wrong white turned whole cards into "ink".
    """
    h, w = lum.shape
    f = max(1, round(dpi / 10))  # 10 DPI is plenty for something that changes this slowly
    hh, ww = math.ceil(h / f), math.ceil(w / f)
    padded = np.pad(lum, ((0, hh * f - h), (0, ww * f - w)), mode="edge").astype(np.float32)
    small = padded.reshape(hh, f, ww, f).mean(axis=(1, 3))
    # 90th percentile, so a big dark pad still has enough paper in its window to measure.
    small = ndimage.percentile_filter(small, 90, size=max(3, round(80 / 25.4 * dpi / f)))
    return np.asarray(Image.fromarray(small, "F").resize((w, h), Image.BILINEAR))


def ink_mask(rgb: np.ndarray, threshold: float | None, dpi: int) -> np.ndarray:
    """True wherever the page is noticeably darker than the paper around it."""
    h, w, _ = rgb.shape
    lum = luminance(rgb)
    darker = paper_brightness(lum, dpi) - lum
    ink = ndimage.binary_opening(darker > (threshold or 25), structure=np.ones((2, 2), bool))

    # Scanner edge shadows hug the page border; nothing on the outer 1% is paw.
    edge = max(2, int(min(h, w) * 0.01))
    ink[:edge] = ink[-edge:] = False
    ink[:, :edge] = ink[:, -edge:] = False
    return ink


# ---------------------------------------------------------------- keeping only the paw


def grow(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilate by a disk (via a distance transform, much faster than a big structuring element)."""
    if not mask.any():
        return mask.copy()  # the transform needs at least one set pixel to measure from
    return ndimage.distance_transform_edt(~mask) <= radius


def shrink(mask: np.ndarray, radius: float) -> np.ndarray:
    """Erode by a disk."""
    return ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] > radius


def drop_colored_marks(ink: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Remove red or other colored marks (hearts, stickers) when the paw is printed in black/gray ink."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    if not ink.any() or np.median(chroma[ink]) > 50:
        return ink  # the paw itself is in colored ink, so color can't tell it apart
    return ink & (chroma <= 50)


def paw_region(ink: np.ndarray, dpi: int, close_mm: float, density: float) -> np.ndarray | None:
    """Region covering each main pad and the toe pads near it, or None if nothing looks like a paw."""
    h, w = ink.shape
    f = max(1, round(dpi / 100))  # work at ~100 DPI; plenty for finding shapes
    hh, ww = math.ceil(h / f), math.ceil(w / f)
    padded = np.zeros((hh * f, ww * f), bool)
    padded[:h, :w] = ink
    fraction = padded.reshape(hh, f, ww, f).mean(axis=(1, 3))
    coarse = fraction >= density
    mm = dpi / f / 25.4  # coarse pixels per mm

    # Fill the skin texture so each pad is one solid shape, then drop everything thinner than
    # 2.5 mm: pen lines, drawn heart outlines, card edges.
    solid = shrink(grow(coarse, close_mm * mm), close_mm * mm)
    solid = grow(shrink(solid, 1.25 * mm), 1.25 * mm)
    cores, count = ndimage.label(solid)
    areas = np.bincount(cores.ravel(), minlength=count + 1)
    # Pads are partly truly solid ink (spots 2 mm across): 13-90% of a pad on real scans. Printed
    # letters are thin strokes, so a block of label text that the filling-in turned into a shape
    # is ~0% solid.
    solid_ink = shrink(fraction >= 0.5, 1.0 * mm)
    solid_share = ndimage.sum(solid_ink, cores, index=np.arange(count + 1)) / np.maximum(areas, 1)
    has_solid_ink = solid_share >= 0.05
    blobs = []  # roundish solid shapes, like pads: not bars, frames, cards, stickers or words
    for idx, (ys, xs) in enumerate(ndimage.find_objects(cores), start=1):
        bh, bw = ys.stop - ys.start, xs.stop - xs.start
        fill = areas[idx] / (bh * bw)
        if (0.3 <= fill <= 0.9 and max(bh, bw) <= 3 * min(bh, bw) and max(bh, bw) <= 160 * mm
                and has_solid_ink[idx]):
            blobs.append(idx)
    if not blobs:
        return None
    largest = max(areas[i] for i in blobs)
    region = Image.new("1", (ww, hh), 0)
    for pad in [i for i in blobs if areas[i] >= 0.35 * largest]:  # main pads (two if two paws)
        # Toe pads sit within about 3/4 of a pad-width of the main pad; hearts and marks further out go.
        reach = 0.75 * 2 * math.sqrt(areas[pad] / math.pi)
        dist = ndimage.distance_transform_edt(cores != pad)
        paw = [i for i, nearest in zip(blobs, ndimage.minimum(dist, labels=cores, index=blobs)) if nearest <= reach]
        # Keep everything inside the outline wrapped around the pad and its toes, so fur streaks
        # and texture between the pads stay.
        ys, xs = np.nonzero(np.isin(cores, paw))
        points = np.column_stack([xs, ys])
        if len(points) >= 3:
            hull = points[ConvexHull(points).vertices]
            ImageDraw.Draw(region).polygon([tuple(p) for p in hull.tolist()], fill=1)
    region = grow(np.asarray(region, bool), 1.5 * mm)
    return np.repeat(np.repeat(region, f, axis=0), f, axis=1)[:h, :w]


def isolate_paw(ink: np.ndarray, rgb: np.ndarray, dpi: int) -> tuple[np.ndarray, str]:
    """(mask of just the paw, reason there is no paw)."""
    if not ink.any():
        return ink, "no paw found (page is blank)"
    ink = drop_colored_marks(ink, rgb)
    # A faint, patchy print needs more filling-in before its pads look solid.
    for close_mm, density in ((1.0, 0.35), (2.5, 0.2)):
        region = paw_region(ink, dpi, close_mm, density)
        if region is not None:
            break
    else:
        return ink, "no paw found (nothing paw-shaped on the page)"
    ink = ink & region
    if ink.sum() < 50 * (dpi / 25.4) ** 2:  # under ~50 mm² of ink
        return ink, "no paw found (only small marks on the page)"
    return ink, ""


def paw_rgba(rgb: np.ndarray, mask: np.ndarray, dpi: int) -> tuple[np.ndarray, np.ndarray]:
    """(grayscale crop around the paw, which of its pixels are paw), with a transparent margin."""
    ys, xs = np.nonzero(mask)
    margin = round(2 / 25.4 * dpi)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    gray = luminance(rgb[y0:y1, x0:x1])
    # A margin of nothing around the paw makes vtracer key out the background (it samples the edges).
    pad = ((margin, margin), (margin, margin))
    return np.pad(gray, pad, constant_values=255), np.pad(mask[y0:y1, x0:x1], pad)


# ---------------------------------------------------------------- tracing


def trace_svg(gray: np.ndarray, keep: np.ndarray, dpi: int, detail: str) -> str:
    preset = DETAIL_PRESETS[detail]
    scale = preset["scale"]
    if scale > 1:  # smooth enlargement, so curves follow the texture instead of the pixel grid
        size = (gray.shape[1] * scale, gray.shape[0] * scale)
        gray = np.asarray(Image.fromarray(gray).resize(size, Image.BICUBIC))
        keep = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) >= 128
    h, w = gray.shape
    rgba = np.dstack([gray, gray, gray, np.where(keep, 255, 0).astype(np.uint8)])
    rgba[~keep, :3] = 255
    png = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(png, "PNG", compress_level=1)
    svg = vtracer.convert_raw_image_to_svg(
        # Positional on purpose: vtracer 0.6.15 segfaults on Python 3.14 when given keyword arguments.
        png.getvalue(),
        "png",  # img_format
        "color",  # colormode
        "stacked",  # hierarchical
        "spline",  # mode
        preset["filter_speckle"],
        preset["color_precision"],
        preset["layer_difference"],
        60,  # corner_threshold
        4.0,  # length_threshold
        10,  # max_iterations
        45,  # splice_threshold
        2,  # path_precision
    )
    # Real-world size, so the paw imports into Illustrator/Cricut at its true scale.
    mm_w, mm_h = w / (dpi * scale) * 25.4, h / (dpi * scale) * 25.4
    return re.sub(
        r'<svg ([^>]*?)width="\d+" height="\d+"',
        rf'<svg \1width="{mm_w:.2f}mm" height="{mm_h:.2f}mm" viewBox="0 0 {w} {h}"',
        svg,
        count=1,
    )


def background_problem(svg: str) -> str:
    """Describe any shape that looks like a leftover background, or return ''."""
    root = ET.fromstring(svg)
    if root.get("viewBox"):
        _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
    else:
        vw, vh = float(root.get("width")), float(root.get("height"))
    for path in root.iter("{http://www.w3.org/2000/svg}path"):
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", path.get("d", ""))]
        if len(nums) < 4:
            continue
        tx = ty = 0.0
        m = re.search(r"translate\(([-\d.e]+)[ ,]+([-\d.e]+)\)", path.get("transform", ""))
        if m:
            tx, ty = float(m.group(1)), float(m.group(2))
        xs, ys = [n + tx for n in nums[0::2]], [n + ty for n in nums[1::2]]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bw >= 0.95 * vw and bh >= 0.95 * vh:
            return f"a shape filled {path.get('fill')} covers the whole image"
        fill = path.get("fill", "")
        pale = re.fullmatch(r"#[0-9A-Fa-f]{6}", fill) and min(int(fill[i:i + 2], 16) for i in (1, 3, 5)) >= 200
        if pale and bw * bh >= 0.5 * vw * vh:
            return f"a pale shape filled {fill} covers most of the image"
    return ""


# ---------------------------------------------------------------- one PDF


def process_pdf(pdf_path: str, source: str, opts: Options) -> list[PageResult]:
    """Worker: convert every page of one PDF. Never raises; errors become results."""
    results = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:  # noqa: BLE001 - report unreadable files and keep going
        return [PageResult(source, 1, 1, "error", reason=f"could not open PDF: {e}")]
    with doc:
        if doc.page_count == 0:
            return [PageResult(source, 1, 0, "error", reason="PDF has no pages")]
        for index, page in enumerate(doc):
            result = PageResult(source, index + 1, doc.page_count, "error")
            try:
                convert_page(page, opts, result)
            except Exception as e:  # noqa: BLE001
                result.status, result.reason = "error", f"{type(e).__name__}: {e}"
            results.append(result)
    return results


def convert_page(page: pymupdf.Page, opts: Options, result: PageResult) -> None:
    pix = page.get_pixmap(dpi=opts.dpi, colorspace=pymupdf.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    ink = ink_mask(rgb, opts.threshold, opts.dpi)
    mask, no_paw = isolate_paw(ink, rgb, opts.dpi)
    if no_paw:
        result.status, result.reason = "skipped", no_paw
        return
    svg = trace_svg(*paw_rgba(rgb, mask, opts.dpi), opts.dpi, opts.detail)
    problem = background_problem(svg)
    result.status, result.svg = ("review" if problem else "ok"), svg
    if problem:
        result.reason = f"possible background left in SVG: {problem}"
        result.preview_png = page_preview(rgb)


def page_preview(rgb: np.ndarray) -> bytes:
    """Small PNG of the original page, to compare against an SVG that needs checking."""
    img = Image.fromarray(rgb)
    img.thumbnail((1200, 1200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- the batch


def unique_path(folder: Path, stem: str, suffix: str, taken: set[str]) -> Path:
    candidate, n = stem, 1
    while candidate.lower() in taken or (folder / f"{candidate}{suffix}").exists():
        n += 1
        candidate = f"{stem} ({n})"
    taken.add(candidate.lower())
    return folder / f"{candidate}{suffix}"


def default_out_dir(in_dir: Path) -> Path:
    out, n = in_dir.with_name(f"{in_dir.name}-svg"), 1
    while out.exists() and any(out.iterdir()):
        n += 1
        out = in_dir.with_name(f"{in_dir.name}-svg ({n})")
    return out


def find_pdfs(in_dir: Path, out_dir: Path) -> list[Path]:
    pdfs = []
    for path in sorted(in_dir.rglob("*")):
        if path.suffix.lower() == ".pdf" and path.is_file() and not path.name.startswith("._"):
            if out_dir not in path.parents:
                pdfs.append(path)
    return pdfs


def write_outputs(results: list[PageResult], out_dir: Path) -> list[dict]:
    """Save every SVG, named after its PDF, in input order; return the report rows."""
    taken: set[str] = set()
    review_taken: set[str] = set()
    rows = []
    by_pdf: dict[str, list[PageResult]] = {}
    for r in results:
        by_pdf.setdefault(r.source, []).append(r)
    for source, pages in by_pdf.items():
        traced = [p for p in pages if p.svg is not None]
        for n, page in enumerate(traced, start=1):
            stem = Path(source).stem + (f" - {n}" if len(traced) > 1 else "")
            folder, used = (out_dir, taken) if page.status == "ok" else (out_dir / REVIEW_DIR, review_taken)
            folder.mkdir(parents=True, exist_ok=True)
            path = unique_path(folder, stem, ".svg", used)
            path.write_text(page.svg, encoding="utf-8")
            page.svg = str(path.relative_to(out_dir))
            if page.preview_png:
                (folder / f"{path.stem}-page.png").write_bytes(page.preview_png)
        for page in pages:
            rows.append({
                "source_pdf": page.source,
                "page": page.page,
                "status": {"ok": "converted", "review": "needs review"}.get(page.status, page.status),
                "output_svg": page.svg or "",
                "note": page.reason,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a folder of paw-print PDF scans into grayscale SVGs with transparent backgrounds.",
    )
    parser.add_argument("folder", type=Path, help="folder of PDF scans (subfolders are included)")
    parser.add_argument("--out", type=Path, help="output folder (default: '<folder>-svg' next to the input)")
    parser.add_argument("--detail", choices=DETAIL_PRESETS, default="medium", help="ink texture detail (default: medium)")
    parser.add_argument("--dpi", type=int, default=300, help="scan resolution to work at (default: 300)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 2, 8),
                        help="PDFs to process at once (default: one per CPU core, up to 8)")
    parser.add_argument("--threshold", type=float,
                        help="how much darker than the paper a pixel must be to count as ink (0-255, default: 25)")
    parser.add_argument("--open", action="store_true", help="open the output folder in Finder when done (macOS)")
    args = parser.parse_args(argv)

    in_dir = args.folder.expanduser().resolve()
    if not in_dir.is_dir():
        parser.error(f"{in_dir} is not a folder")
    out_dir = (args.out.expanduser().resolve() if args.out else default_out_dir(in_dir))
    opts = Options(args.dpi, args.detail, args.threshold)
    pdfs = find_pdfs(in_dir, out_dir)
    print(f"Paw Prints converter, version {VERSION}")
    if not pdfs:
        print(f"No PDF files found in {in_dir}")
        return 1
    print(f"Converting {len(pdfs)} PDF(s) from {in_dir} · detail: {opts.detail} · workers: {args.workers}")

    start = time.monotonic()
    results: dict[str, list[PageResult]] = {}
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(process_pdf, str(p), str(p.relative_to(in_dir)), opts): p for p in pdfs}
            for done, future in enumerate(as_completed(futures), start=1):
                pages = future.result()
                results[pages[0].source] = pages
                icons = {"ok": "✓", "review": "⚑", "skipped": "·", "error": "✗"}
                summary = ", ".join(f"{icons[p.status]} {p.reason or 'done'}" for p in pages)
                print(f"[{done}/{len(pdfs)}] {pages[0].source}: {summary}", flush=True)
    except BrokenProcessPool:
        # A worker died outright (a crash inside a library). Keep everything already finished.
        print("\nThe converter crashed partway through. Finished files are saved below; "
              "the rest are listed as errors in report.csv.")
    sources = [str(p.relative_to(in_dir)) for p in pdfs]
    for source in sources:
        results.setdefault(source, [PageResult(source, 1, 1, "error", reason="converter crashed; not converted")])
    ordered = [page for source in sources for page in results[source]]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = write_outputs(ordered, out_dir)
    with open(out_dir / REPORT_NAME, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count = {s: sum(r["status"] == s for r in rows) for s in ("converted", "needs review", "skipped", "error")}
    print(f"\n{count['converted']} converted · {count['needs review']} need review · "
          f"{count['skipped']} skipped · {count['error']} errors · {time.monotonic() - start:.0f}s")
    print(f"Output: {out_dir}")
    if count["needs review"]:
        print(f"Check {out_dir / REVIEW_DIR}: compare each SVG with its -page.png and erase anything that isn't paw.")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out_dir)], check=False)
    return 0 if not count["error"] else 2


if __name__ == "__main__":
    sys.exit(main())
