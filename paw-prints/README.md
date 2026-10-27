# Paw Prints → SVG

Turns a folder of scanned paw-print PDFs into SVGs with a **transparent background**,
each named after the dog on its printed label (`Buddy Smith.svg`, or `Buddy.svg` when the
label only has a first name). Anything it can't name confidently goes to a
`NEEDS_REVIEW` folder for you to name by hand. The whole folder is processed in one go,
several PDFs at a time.

## One-time setup (Mac)

1. Put this `paw-prints` folder somewhere handy, e.g. your Documents folder.
2. Double-click **Convert Paw Prints.command**. The first time, it sets itself up (about a
   minute).
   - If macOS says Apple "could not verify" it is free of malware, see
     [the macOS warning](#the-macos-malware-warning) below. You only need to do this once.
   - If it says Python 3.10 or newer is needed, install Python from
     <https://www.python.org/downloads/macos/>, then double-click the file again.

The first run sets everything up for you. Running `./setup.sh` in Terminal does the same
thing by hand.

## The macOS malware warning

macOS shows this warning for any script downloaded from the internet that isn't signed with
a paid Apple developer certificate. It doesn't mean anything harmful was found. Every file
here is plain text, so you can open them in TextEdit and read them first.

Clear the warning once for the whole folder, either way:

- **Terminal (covers every file at once):** open Terminal, type
  `xattr -dr com.apple.quarantine ` (with a space at the end), drag the `paw-prints`
  folder onto the Terminal window, and press Return. This removes the "downloaded from the
  internet" tag from the folder and everything in it.
- **System Settings:** after the warning appears, click **Done**. Go to **System Settings →
  Privacy & Security**, scroll down to the message about "Convert Paw Prints.command", click
  **Open Anyway**, and confirm with your password.

Getting the folder with `git clone` instead of a zip download avoids the warning
altogether, because git doesn't add the tag.

## Converting a batch

1. Double-click **Convert Paw Prints.command**.
2. Choose the folder of PDF scans. Subfolders are included.
3. Watch the progress. When it finishes, the results folder opens in Finder.

Results go in a new folder next to your scans. For example, `paw-scans` gives you
`paw-scans-svg`. If you run the same folder again you get `paw-scans-svg (2)`, so nothing
is ever overwritten. Your PDFs are never changed.

```
paw-scans-svg/
  Buddy Smith.svg
  Max.svg
  Max (2).svg              ← a second dog named Max
  Daisy O'Brien - 1.svg    ← a PDF with two paws (two pages)
  Daisy O'Brien - 2.svg
  NEEDS_REVIEW/
    scan_0042.svg          ← finished and transparent; it just needs the dog's name
    scan_0042-label.png    ← picture of the label (or the page) so you can read the name
  report.csv               ← one line per page: name read, confidence, and any problem
```

**What to do with NEEDS_REVIEW:** open the `-label.png`, then rename the matching `.svg` to
the dog's name. The SVG is already done. A file lands here when:

- there's no label, or the label has no readable name;
- two different lines look equally like the name (for example "Bella" and "Coco" in the same size);
- OCR wasn't confident about what it read.

## What it does to each scan

1. Reads the label from the PDF's own text if the scanner made it searchable. Otherwise it
   uses Apple's built-in text recognition, which is free, runs on your Mac, and needs no
   account.
2. Picks the name. A `Name: ...` field wins. Otherwise it takes the largest line that looks
   like a name. Lines like `Owner: ...`, dates, and "left front" are ignored. ALL-CAPS names
   are changed to normal capitalization (`BUDDY SMITH` → `Buddy Smith`).
3. Removes the label and any dust specks, so only the paw is left.
4. Removes the paper. Anything close to the paper color becomes transparent, including
   the little gaps in the paw's skin texture. Edges are cleaned up so there's no white
   halo when the paw sits on a dark background.
5. Traces the paw in its real ink colors and saves the SVG at its **true physical size**
   (in mm), so it imports into Illustrator, Cricut Design Space, and similar tools at the
   right scale.

## Options

To use options, run the script in Terminal from this folder:

```
.venv/bin/python pawprints.py ~/Desktop/paw-scans [options]
```

| Option | What it does |
|---|---|
| `--dry-run` | List the name it would use for each PDF without writing anything. Good for a quick check. |
| `--detail low/medium/high` | How much ink texture to keep. `high` keeps the most texture and makes bigger files. Default is `medium`. |
| `--out FOLDER` | Put the results somewhere else. |
| `--name-pattern REGEX` | Tell it exactly where the name is, if your labels have a fixed layout. Example: `--name-pattern 'Pet: (?P<name>.+)'` |
| `--min-confidence 0.5` | Flag names read with OCR confidence below this (0–1). |
| `--threshold 40` | How different from the paper a pixel must be to count as ink. Lower it if faint ink is lost; raise it if paper smudges show up. It's automatic by default. |
| `--workers N` | How many PDFs to process at once. Defaults to one per CPU core, up to 8. |
| `--open` | Open the results folder in Finder when done. The launcher uses this. |

## Good to know

- **Speed:** each PDF takes 1–2 seconds of work, and several run at once (one per CPU
  core), so around 100 scans should take under a minute on a recent Mac.
- **Scanning tips:** 300 DPI color scans on plain white paper work best. Keep the label a
  little away from the paw, about half an inch.
- **Not on a Mac?** The script also runs on Windows or Linux with
  [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) installed for
  reading labels. Labels on a big colored sticker aren't removed automatically yet.

## Tests

```
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
```

The tests generate fake scans (`tests/make_samples.py`) and check the following:

- file names and which files get flagged
- `report.csv` contents
- each SVG has a transparent background and keeps the ink color
- the label is removed
- no pale fringe is left around the paw
- re-running never overwrites earlier results
