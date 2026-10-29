#!/usr/bin/env python3
"""Bulk-convert paw-print PDF scans into transparent-background SVGs.

Each PDF holds a scanned paw print on white paper plus a printed label with the
dog's name. For every page the script:

  1. renders the page and reads the label (PDF text layer, else OCR),
  2. picks the dog's name and cleans it into a file name,
  3. removes the label and the paper so only the inked paw is left,
  4. traces the paw in color with vtracer into an SVG with no background,
  5. saves "<Dog Name>.svg", or puts it in NEEDS_REVIEW/ when no name is found.

Usage:
    ./pawprints.py ~/Desktop/paw-scans            # writes ~/Desktop/paw-scans-svg/
    ./pawprints.py ~/Desktop/paw-scans --dry-run  # only show the names it would use

Run ./pawprints.py --help for all options.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
import vtracer
from PIL import Image, ImageFilter
from scipy import ndimage

REVIEW_DIR = "NEEDS_REVIEW"
REPORT_NAME = "report.csv"

# vtracer settings. Higher detail keeps more of the ink texture and makes bigger files.
DETAIL_PRESETS = {
    "low": dict(filter_speckle=8, color_precision=5, layer_difference=32),
    "medium": dict(filter_speckle=4, color_precision=6, layer_difference=16),
    "high": dict(filter_speckle=2, color_precision=7, layer_difference=8),
}

# Label fields that hold the dog's name ("Name: Buddy Smith", "Pet's name - Buddy").
NAME_FIELD_RE = re.compile(
    r"^\s*(?:(?:dog|pet|puppy|patient)(?:'?s)?\s+)?name\b\s*[:\-–—=]?\s*(?P<name>.*)$",
    re.IGNORECASE,
)
FIRST_FIELD_RE = re.compile(r"^\s*first(?:\s+name)?\s*[:\-–—=]\s*(?P<name>.*)$", re.IGNORECASE)
LAST_FIELD_RE = re.compile(r"^\s*(?:last|family|sur)\s*(?:name)?\s*[:\-–—=]\s*(?P<name>.*)$", re.IGNORECASE)
# Any other "Key: value" line (Owner:, Date:, Breed:) is never the dog's name.
OTHER_FIELD_RE = re.compile(r"^\s*[^\W\d_][\w' ]{0,24}\s*[:=]")
MEMORIAL_PREFIX_RE = re.compile(r"^\s*(?:in\s+(?:loving\s+)?memory\s+of|r\.?i\.?p\.?)\s+", re.IGNORECASE)
NAME_WORD_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|['.\-])*$")
# Words that show up on labels but are never part of a dog's name.
STOPWORDS = {
    "paw", "paws", "print", "prints", "pawprint", "pawprints", "left", "right", "front",
    "back", "rear", "hind", "nose", "date", "dob", "breed", "owner", "clinic", "hospital",
    "veterinary", "vet", "scan", "page", "name", "memory", "loving", "forever", "rip",
}
FILENAME_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
NO_NAME = "no name found on label"


@dataclass
class Options:
    dpi: int = 300
    detail: str = "medium"
    ocr: str = "auto"
    threshold: float | None = None
    min_confidence: float = 0.5
    name_pattern: str | None = None
    dry_run: bool = False


@dataclass
class TextLine:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class PageResult:
    source: str
    page: int
    page_count: int
    status: str  # "ok", "review", "skipped" or "error"
    name: str | None = None
    confidence: float | None = None
    text_source: str = ""
    reason: str = ""
    svg: str | None = None
    preview_png: bytes | None = None
    label_missing: bool = False  # no name text at all, as opposed to an unclear one
    stem: str = ""


# ---------------------------------------------------------------- reading text


def text_layer_lines(page: pymupdf.Page, dpi: int) -> list[TextLine]:
    """Lines from the PDF's own text layer (searchable scans), in page pixels."""
    scale = dpi / 72
    lines = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = " ".join(s["text"].strip() for s in spans)
            x0 = min(s["bbox"][0] for s in spans) * scale
            y0 = min(s["bbox"][1] for s in spans) * scale
            x1 = max(s["bbox"][2] for s in spans) * scale
            y1 = max(s["bbox"][3] for s in spans) * scale
            lines.append(TextLine(text, 1.0, (x0, y0, x1, y1)))
    return lines


def pick_ocr_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if sys.platform == "darwin":
        try:
            import ocrmac  # noqa: F401

            return "vision"
        except ImportError:
            pass
    if shutil.which("tesseract"):
        return "tesseract"
    return "none"


def ocr_vision(img: Image.Image) -> list[TextLine]:
    from ocrmac import ocrmac

    results = ocrmac.OCR(img, recognition_level="accurate", language_preference=["en-US"]).recognize(px=True)
    return [TextLine(text, float(conf), tuple(box)) for text, conf, box in results]


def ocr_tesseract(img: Image.Image) -> list[TextLine]:
    # Tesseract is faster and finds sparse label text more reliably at ~150 DPI than at 300.
    scale = 0.5 if min(img.size) > 2000 else 1.0
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS) if scale < 1 else img
    buf = io.BytesIO()
    small.save(buf, "PNG")
    out = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
        input=buf.getvalue(),
        capture_output=True,
        check=True,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},  # the batch already runs one worker per core
    ).stdout.decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE))
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("level") == "5" and (row.get("text") or "").strip():
            key = (row["block_num"], row["par_num"], row["line_num"])
            grouped.setdefault(key, []).append(row)
    lines = []
    for words in grouped.values():
        x0 = min(int(w["left"]) for w in words) / scale
        y0 = min(int(w["top"]) for w in words) / scale
        x1 = max(int(w["left"]) + int(w["width"]) for w in words) / scale
        y1 = max(int(w["top"]) + int(w["height"]) for w in words) / scale
        conf = min(float(w["conf"]) for w in words) / 100
        lines.append(TextLine(" ".join(w["text"].strip() for w in words), conf, (x0, y0, x1, y1)))
    return lines


def run_ocr(img: Image.Image, backend: str) -> list[TextLine]:
    if backend == "vision":
        return ocr_vision(img)
    if backend == "tesseract":
        return ocr_tesseract(img)
    return []


def merge_row_fragments(lines: list[TextLine]) -> list[TextLine]:
    """Join pieces of one printed line that OCR returned separately ("Buddy" + "Smith")."""
    merged: list[TextLine] = []
    for line in sorted(lines, key=lambda l: l.box[0]):
        for i, prev in enumerate(merged):
            h = max(prev.height, line.height, 1)
            same_row = abs((prev.box[1] + prev.box[3]) / 2 - (line.box[1] + line.box[3]) / 2) < 0.4 * h
            similar = 0.75 <= line.height / max(prev.height, 1) <= 1.33
            gap = line.box[0] - prev.box[2]
            if same_row and similar and -0.2 * h <= gap < 1.5 * h:
                merged[i] = TextLine(
                    f"{prev.text} {line.text}",
                    min(prev.conf, line.conf),
                    (prev.box[0], min(prev.box[1], line.box[1]), line.box[2], max(prev.box[3], line.box[3])),
                )
                break
        else:
            merged.append(line)
    return merged


# ---------------------------------------------------------------- picking the name


def clean_name(raw: str) -> str:
    name = unicodedata.normalize("NFC", raw).replace("’", "'").replace("‘", "'")
    name = MEMORIAL_PREFIX_RE.sub("", name)
    name = FILENAME_BAD_CHARS.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip(" .-_,;")
    letters = [c for c in name if c.isalpha()]
    if letters and (all(c.isupper() for c in letters) or all(c.islower() for c in letters)):
        name = re.sub(r"[^\W\d_]+", lambda m: m.group(0).capitalize(), name)
    return name[:80].strip()


def looks_like_name(text: str) -> bool:
    text = MEMORIAL_PREFIX_RE.sub("", text).strip(" .,;")
    words = text.replace("’", "'").split()
    if not 1 <= len(words) <= 4:
        return False
    if any(not NAME_WORD_RE.match(w) or w.lower().strip(".") in STOPWORDS for w in words):
        return False
    return sum(c.isalpha() for c in text) >= 2


def name_words(value: str) -> str:
    """The name at the start of a field value: "Buddy Smith 10/27/26" -> "Buddy Smith"."""
    words = []
    for word in value.replace("’", "'").split():
        if not NAME_WORD_RE.match(word) or word.lower().strip(".") in STOPWORDS or len(words) == 4:
            break
        words.append(word)
    return " ".join(words)


def field_value(lines: list[TextLine], idx: int, regex: re.Pattern) -> tuple[str, float] | None:
    """Value of a "Key: value" field, or of the line just below a bare "Key:" line."""
    line = lines[idx]
    value = regex.match(line.text).group("name").strip()
    if value:
        return name_words(value), line.conf
    below = [
        l for l in lines
        if l is not line
        and 0 <= l.box[1] - line.box[3] < 2 * line.height
        and l.box[0] < line.box[2] + line.height and l.box[2] > line.box[0]
    ]
    if below:
        nearest = min(below, key=lambda l: l.box[1])
        return name_words(nearest.text), nearest.conf
    return None


def choose_name(lines: list[TextLine], opts: Options) -> tuple[str | None, float | None, str]:
    """Return (name, confidence, reason). A None name means the page needs review."""
    if opts.name_pattern:
        pattern = re.compile(opts.name_pattern, re.IGNORECASE)
        for line in lines:
            m = pattern.search(line.text)
            if m:
                raw = m.groupdict().get("name") or (m.group(1) if m.groups() else m.group(0))
                return finish_name(raw, line.conf, opts, "matched --name-pattern")
        return None, None, NO_NAME + " (nothing matched --name-pattern)"

    full = first = last = None
    for i, line in enumerate(lines):
        if NAME_FIELD_RE.match(line.text):
            full = full or field_value(lines, i, NAME_FIELD_RE)
        elif FIRST_FIELD_RE.match(line.text):
            first = first or field_value(lines, i, FIRST_FIELD_RE)
        elif LAST_FIELD_RE.match(line.text):
            last = last or field_value(lines, i, LAST_FIELD_RE)
    if first:
        raw = first[0] + (" " + last[0] if last else "")
        return finish_name(raw, min(first[1], last[1]) if last else first[1], opts, "first/last name fields")
    if full:
        return finish_name(full[0], full[1], opts, "name field")

    candidates = [l for l in lines if not OTHER_FIELD_RE.match(l.text) and looks_like_name(l.text)]
    if not candidates:
        return None, None, NO_NAME if lines else NO_NAME + " (no text on page)"
    candidates.sort(key=lambda l: l.height, reverse=True)
    best = candidates[0]
    rivals = [
        c for c in candidates[1:]
        if c.height >= 0.9 * best.height and clean_name(c.text).lower() != clean_name(best.text).lower()
    ]
    if rivals:
        options = " / ".join(f"'{clean_name(c.text)}'" for c in [best, *rivals[:2]])
        return None, best.conf, f"more than one possible name: {options}"
    return finish_name(best.text, best.conf, opts, "largest name-like text")


def finish_name(raw: str, conf: float, opts: Options, how: str) -> tuple[str | None, float, str]:
    name = clean_name(raw)
    if not name or not any(c.isalpha() for c in name):
        return None, conf, f"{how}, but it was empty"
    if conf < opts.min_confidence:
        return None, conf, f"low OCR confidence ({conf:.2f}) reading '{name}'"
    return name, conf, how


# ---------------------------------------------------------------- finding the paw


def color_distance2(rgb: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Squared color distance of every pixel from the paper (lookup tables keep big scans fast)."""
    lut = ((np.arange(256)[None, :] - paper[:, None]) ** 2).astype(np.int32)
    return lut[0][rgb[..., 0]] + lut[1][rgb[..., 1]] + lut[2][rgb[..., 2]]


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


def is_printed_text(line: TextLine, ink: np.ndarray) -> bool:
    """False for "text" OCR imagined on the paw: real letters are thin strokes, toes are solid blobs."""
    h, w = ink.shape
    x0, y0 = max(0, int(line.box[0])), max(0, int(line.box[1]))
    x1, y1 = min(w, int(math.ceil(line.box[2]))), min(h, int(math.ceil(line.box[3])))
    box = ink[y0:y1, x0:x1]
    if box.size == 0 or not box.any():
        return box.size > 0  # invisible text-layer text with no ink under it is still text
    if box.mean() > 0.45:
        return False
    k = max(1, round(0.04 * line.height))  # close ink texture so a mottled toe counts as solid
    closed = ndimage.binary_closing(np.pad(box, k + 1), iterations=k)
    return ndimage.distance_transform_edt(closed).max() <= 0.2 * line.height


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


def paw_region(ink: np.ndarray, dpi: int) -> np.ndarray | None:
    """Where the paw is: each main pad plus the toe pads around it. None if no paw-like shape."""
    h, w = ink.shape
    f = max(1, round(dpi / 100))  # work at ~100 DPI; plenty for finding shapes
    hh, ww = math.ceil(h / f), math.ceil(w / f)
    padded = np.zeros((hh * f, ww * f), bool)
    padded[:h, :w] = ink
    coarse = padded.reshape(hh, f, ww, f).mean(axis=(1, 3)) >= 0.35
    mm = dpi / f / 25.4  # coarse pixels per mm
    # Fill the skin texture so each pad is one solid shape, then drop everything thinner than
    # 2.5 mm: pen lines, drawn heart outlines, handwriting.
    solid = shrink(grow(coarse, 1.0 * mm), 1.0 * mm)
    solid = grow(shrink(solid, 1.25 * mm), 1.25 * mm)
    cores, count = ndimage.label(solid)
    if count == 0:
        return None
    areas = np.bincount(cores.ravel(), minlength=count + 1)
    blobs = []  # roundish solid shapes, like pads: not bars, frames, cards or stickers
    for idx, (ys, xs) in enumerate(ndimage.find_objects(cores), start=1):
        bh, bw = ys.stop - ys.start, xs.stop - xs.start
        fill = areas[idx] / (bh * bw)
        if 0.3 <= fill <= 0.9 and max(bh, bw) <= 3 * min(bh, bw) and max(bh, bw) <= 160 * mm:
            blobs.append(idx)
    if not blobs:
        return None
    largest = max(areas[i] for i in blobs)
    keep = np.zeros(count + 1, bool)
    for pad in [i for i in blobs if areas[i] >= 0.35 * largest]:  # main pads (two if two paws)
        keep[pad] = True
        # Toe pads sit within about 3/4 of a pad-width of the main pad; hearts and marks further out go.
        reach = 0.75 * 2 * math.sqrt(areas[pad] / math.pi)
        dist = ndimage.distance_transform_edt(cores != pad)
        for idx, nearest in zip(blobs, ndimage.minimum(dist, labels=cores, index=blobs)):
            keep[idx] |= nearest <= reach
    region = grow(keep[cores], 1.5 * mm)
    return np.repeat(np.repeat(region, f, axis=0), f, axis=1)[:h, :w]


def isolate_paw(ink: np.ndarray, rgb: np.ndarray, lines: list[TextLine], dpi: int) -> tuple[np.ndarray, str, str]:
    """(mask of just the paw, reason there is no paw, warning that needs a person to check)."""
    ink = ink.copy()
    h, w = ink.shape
    labels, count = ndimage.label(ink, structure=np.ones((3, 3), bool))
    if count == 0:
        return ink, "no paw found (page is blank)", ""
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    largest = areas.max()

    # Remove the label: letters, underlines and frames that sit on OCR text boxes.
    text_boxes = []
    for line in lines:
        if sum(c.isalnum() for c in line.text) >= 2:
            pad = 0.25 * line.height
            text_boxes.append((max(0, int(line.box[0] - pad)), max(0, int(line.box[1] - pad)),
                               min(w, int(line.box[2] + pad) + 1), min(h, int(line.box[3] + pad) + 1)))
    if text_boxes:
        on_text = np.zeros_like(ink)
        for x0, y0, x1, y1 in text_boxes:
            on_text[y0:y1, x0:x1] = True
        overlap = np.bincount(labels[on_text].ravel(), minlength=count + 1)
        slices = ndimage.find_objects(labels)
        remove = np.zeros(count + 1, bool)
        for idx in np.nonzero(overlap[1:])[0] + 1:
            ys, xs = slices[idx - 1]
            inside_box = any(x0 <= xs.start and xs.stop <= x1 and y0 <= ys.start and ys.stop <= y1
                             for x0, y0, x1, y1 in text_boxes)
            if inside_box or (overlap[idx] / areas[idx] >= 0.5 and areas[idx] < 0.1 * largest):
                remove[idx] = True  # a letter, or a mark sitting on the label
                continue
            encloses_text = any(xs.start <= (x0 + x1) / 2 < xs.stop and ys.start <= (y0 + y1) / 2 < ys.stop
                                for x0, y0, x1, y1 in text_boxes)
            fill = areas[idx] / ((ys.stop - ys.start) * (xs.stop - xs.start))
            if encloses_text and fill < 0.15:
                remove[idx] = True  # a thin frame or sticker edge around the label
        ink[remove[labels]] = False

    ink = drop_colored_marks(ink, rgb)
    region = paw_region(ink, dpi)
    warning = ""
    if region is None:
        # Very faint or broken-up print: keep all the ink and let a person check it.
        warning = "couldn't find a clear paw shape; check that only the paw was kept"
    else:
        ink &= region

    if ink.sum() < 50 * (dpi / 25.4) ** 2:  # under ~50 mm² of ink
        return ink, "no paw found (only small marks on the page)", ""
    return ink, "", warning


def paw_rgba(rgb: np.ndarray, mask: np.ndarray, ink: np.ndarray, dpi: int) -> np.ndarray:
    """Crop to the paw and make everything that is not ink fully transparent."""
    ys, xs = np.nonzero(mask)
    margin = round(2 / 25.4 * dpi)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    crop = np.array(Image.fromarray(rgb[y0:y1, x0:x1]).filter(ImageFilter.MedianFilter(3)))
    bare = ~ink[y0:y1, x0:x1]
    paper = np.median(crop[bare], axis=0) if bare.any() else np.full(3, 255.0)  # paper right here
    dist2 = color_distance2(crop, paper)
    keep = mask[y0:y1, x0:x1]
    # Pixels that are mostly paper (under 30% of the ink's strength) become transparent too,
    # so the paw has no pale fringe and its skin-texture gaps show through.
    strong = np.percentile(dist2[keep], 90)
    keep &= dist2 >= 0.09 * strong
    # Edge pixels are part paper, so they scan paler than the ink. Give them the color of the
    # nearest solid ink so no light outline shows when the SVG sits on a dark background.
    solid = keep & (dist2 >= 0.8 * strong)
    if solid.any():
        edge = keep & ~ndimage.binary_erosion(keep, iterations=max(1, round(0.6 / 25.4 * dpi)))
        iy, ix = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        paler = edge & (dist2 < dist2[iy, ix])
        crop[paler] = crop[iy[paler], ix[paler]]
    gray = (crop.astype(np.uint32) @ np.array([299, 587, 114], np.uint32) // 1000).astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, np.where(keep, 255, 0).astype(np.uint8)])
    rgba[~keep, :3] = 255
    # A fully transparent frame makes vtracer key out the background (it samples the edges).
    return np.pad(rgba, ((margin, margin), (margin, margin), (0, 0)), constant_values=0)


# ---------------------------------------------------------------- tracing


def trace_svg(rgba: np.ndarray, dpi: int, detail: str) -> str:
    h, w, _ = rgba.shape
    preset = DETAIL_PRESETS[detail]
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
    mm_w, mm_h = w / dpi * 25.4, h / dpi * 25.4
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
    backend = pick_ocr_backend(opts.ocr)
    with doc:
        if doc.page_count == 0:
            return [PageResult(source, 1, 0, "error", reason="PDF has no pages")]
        for index, page in enumerate(doc):
            result = PageResult(source, index + 1, doc.page_count, "error")
            try:
                convert_page(page, opts, backend, result)
            except Exception as e:  # noqa: BLE001
                result.status, result.reason = "error", f"{type(e).__name__}: {e}"
            results.append(result)
    # A page with no label of its own (e.g. the second paw) takes the name from the labeled page.
    doc_name = next((r.name for r in results if r.name), None)
    for r in results:
        if doc_name and r.name is None and r.label_missing and r.status in ("ok", "review"):
            r.name = doc_name
            if r.reason.startswith(NO_NAME):  # flagged only because this page had no label
                r.status, r.reason = "ok", "name from another page of this PDF"
    return results


def convert_page(page: pymupdf.Page, opts: Options, backend: str, result: PageResult) -> None:
    pix = page.get_pixmap(dpi=opts.dpi, colorspace=pymupdf.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    img = Image.fromarray(rgb)

    ink = ink_mask(rgb, opts.threshold, opts.dpi)

    def real_text(found: list[TextLine]) -> list[TextLine]:
        return merge_row_fragments([l for l in found if l.conf >= 0.2 and is_printed_text(l, ink)])

    lines, result.text_source = real_text(text_layer_lines(page, opts.dpi)), "pdf text"
    name, conf, reason = choose_name(lines, opts)
    if name is None and backend != "none":
        ocr_lines = real_text(run_ocr(img, backend))
        if ocr_lines:
            lines, result.text_source = ocr_lines, backend
            name, conf, reason = choose_name(lines, opts)
    if not lines:
        result.text_source = backend if backend != "none" else "no OCR available"
    result.name, result.confidence = name, conf
    result.label_missing = name is None and reason.startswith(NO_NAME)

    mask, no_paw, warning = isolate_paw(ink, rgb, lines, opts.dpi)
    if no_paw:
        result.status, result.reason = "skipped", no_paw
        return
    result.status = "ok" if name and not warning else "review"
    result.reason = warning or reason
    if result.status == "review":
        result.preview_png = label_preview(img, [] if warning else lines)
    if opts.dry_run:
        return
    svg = trace_svg(paw_rgba(rgb, mask, ink, opts.dpi), opts.dpi, opts.detail)
    problem = background_problem(svg)
    if problem:
        result.status, result.reason = "review", f"possible background left in SVG: {problem}"
        result.preview_png = result.preview_png or label_preview(img, lines)
    result.svg = svg


def label_preview(img: Image.Image, lines: list[TextLine]) -> bytes:
    """PNG of the label area (or the whole page) so a person can read the name fast."""
    boxes = [l.box for l in lines if sum(c.isalnum() for c in l.text) >= 2]
    if boxes:
        pad = 0.05 * min(img.size)
        crop = img.crop((
            max(0, min(b[0] for b in boxes) - pad), max(0, min(b[1] for b in boxes) - pad),
            min(img.width, max(b[2] for b in boxes) + pad), min(img.height, max(b[3] for b in boxes) + pad),
        ))
    else:
        crop = img.copy()
    crop.thumbnail((1600, 1600))
    buf = io.BytesIO()
    crop.save(buf, "PNG")
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
    """Name and save every SVG in input order; return the report rows."""
    review_dir = out_dir / REVIEW_DIR
    taken: set[str] = set()
    review_taken: set[str] = set()
    rows = []
    by_pdf: dict[str, list[PageResult]] = {}
    for r in results:
        by_pdf.setdefault(r.source, []).append(r)
    for source, pages in by_pdf.items():
        traced = [p for p in pages if p.svg is not None]
        for page in traced:
            page.stem = page.name or Path(source).stem
        for page in traced:
            same = [p for p in traced if p.stem == page.stem]
            stem = f"{page.stem} - {same.index(page) + 1}" if len(same) > 1 else page.stem
            folder, used = (out_dir, taken) if page.status == "ok" else (review_dir, review_taken)
            folder.mkdir(parents=True, exist_ok=True)
            path = unique_path(folder, stem, ".svg", used)
            if path.stem != stem and folder == out_dir:
                page.reason += "; another dog has this name, so it was numbered"
            path.write_text(page.svg, encoding="utf-8")
            page.svg = str(path.relative_to(out_dir))
            if page.status == "review" and page.preview_png:
                (folder / f"{path.stem}-label.png").write_bytes(page.preview_png)
        for page in pages:
            rows.append({
                "source_pdf": page.source,
                "page": page.page,
                "status": {"ok": "converted", "review": "needs review"}.get(page.status, page.status),
                "output_svg": page.svg or "",
                "dog_name": page.name or "",
                "confidence": "" if page.confidence is None else f"{page.confidence:.2f}",
                "text_from": page.text_source,
                "note": page.reason,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a folder of paw-print PDF scans into transparent SVGs named after each dog.",
    )
    parser.add_argument("folder", type=Path, help="folder of PDF scans (subfolders are included)")
    parser.add_argument("--out", type=Path, help="output folder (default: '<folder>-svg' next to the input)")
    parser.add_argument("--detail", choices=DETAIL_PRESETS, default="medium", help="ink texture detail (default: medium)")
    parser.add_argument("--dpi", type=int, default=300, help="scan resolution to work at (default: 300)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 2, 8),
                        help="PDFs to process at once (default: one per CPU core, up to 8)")
    parser.add_argument("--ocr", choices=["auto", "vision", "tesseract", "none"], default="auto",
                        help="text reader for labels (default: Apple Vision on a Mac)")
    parser.add_argument("--threshold", type=float,
                        help="how much darker than the paper a pixel must be to count as ink (0-255, default: 25)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="flag names read with lower OCR confidence (0-1, default: 0.5)")
    parser.add_argument("--name-pattern",
                        help="regex that finds the name on each label, e.g. 'Pet: (?P<name>.+)'")
    parser.add_argument("--dry-run", action="store_true", help="only show the names it would use; write nothing")
    parser.add_argument("--open", action="store_true", help="open the output folder in Finder when done (macOS)")
    args = parser.parse_args(argv)

    in_dir = args.folder.expanduser().resolve()
    if not in_dir.is_dir():
        parser.error(f"{in_dir} is not a folder")
    out_dir = (args.out.expanduser().resolve() if args.out else default_out_dir(in_dir))
    if args.name_pattern:
        re.compile(args.name_pattern)
    opts = Options(args.dpi, args.detail, args.ocr, args.threshold, args.min_confidence, args.name_pattern, args.dry_run)
    pdfs = find_pdfs(in_dir, out_dir)
    if not pdfs:
        print(f"No PDF files found in {in_dir}")
        return 1
    backend = pick_ocr_backend(opts.ocr)
    if backend == "none":
        print("Warning: no OCR available, so only searchable PDFs will get names. "
              "On a Mac run setup.sh; elsewhere install Tesseract.")
    print(f"Converting {len(pdfs)} PDF(s) from {in_dir}")
    print(f"Label reader: {backend} · detail: {opts.detail} · workers: {args.workers}")

    start = time.monotonic()
    results: dict[str, list[PageResult]] = {}
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(process_pdf, str(p), str(p.relative_to(in_dir)), opts): p for p in pdfs}
            for done, future in enumerate(as_completed(futures), start=1):
                pages = future.result()
                results[pages[0].source] = pages
                icons = {"ok": "✓", "review": "⚑", "skipped": "·", "error": "✗"}
                summary = ", ".join(f"{icons[p.status]} {p.name or p.reason}" for p in pages)
                print(f"[{done}/{len(pdfs)}] {pages[0].source}: {summary}", flush=True)
    except BrokenProcessPool:
        # A worker died outright (a crash inside a library). Keep everything already finished.
        print("\nThe converter crashed partway through. Finished files are saved below; "
              "the rest are listed as errors in report.csv.")
    sources = [str(p.relative_to(in_dir)) for p in pdfs]
    for source in sources:
        results.setdefault(source, [PageResult(source, 1, 1, "error", reason="converter crashed; not converted")])
    ordered = [page for source in sources for page in results[source]]

    if opts.dry_run:
        print(f"\nDry run finished in {time.monotonic() - start:.0f}s; nothing was written.")
        return 0
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
        print(f"Check {out_dir / REVIEW_DIR}: each SVG is ready, it just needs the dog's name as its file name.")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out_dir)], check=False)
    return 0 if not count["error"] else 2


if __name__ == "__main__":
    sys.exit(main())
