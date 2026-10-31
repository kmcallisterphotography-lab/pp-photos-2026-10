# Paw Prints → SVG

Turns a folder of scanned paw-print PDFs into grayscale SVGs of just the paw, on a
**transparent background**. Each SVG gets the same name as its PDF (`Brown.pdf` →
`Brown.svg`). The whole folder is processed in one go, several PDFs at a time.

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
  Bandit Oswalt K9.svg     ← the label side of the card has no paw, so it's skipped
  Brown.svg
  Daisy O'Brien - 1.svg    ← a PDF with a paw on each of two pages gives one SVG per paw
  Daisy O'Brien - 2.svg
  NEEDS_REVIEW/
    Coco.svg               ← finished, but worth a look
    Coco-page.png          ← the original page, to compare against
  report.csv               ← one line per page, including skipped ones and why
```

The terminal also shows the converter's version when it starts. If a fix doesn't seem to
take, check that line.

**What to do with NEEDS_REVIEW:** compare the SVG with its `-page.png` and fix anything
that's off. A file lands here when the SVG seems to contain a leftover background.

**Pages with no paw** are skipped, such as the label side of a card or a blank page.
`report.csv` lists them. If a real paw page is ever skipped, its print was too faint to
find. Try `--threshold 15`.

## What it does to each scan

1. Finds the ink: anything noticeably darker than the paper around it. Paper is judged
   locally, so a card that's a different white from the scanner lid doesn't count as ink.
2. Keeps only the paw. It removes:
   - red or other colored marks, such as hearts, when the paw is in black ink;
   - anything thinner than about 2.5 mm, such as pen lines, card edges, heart outlines and
     writing;
   - anything outside the outline around the main pad and its toe pads, such as the label
     or stamped hearts elsewhere on the sheet. Inside that outline everything stays, such as
     the fur streaks between the pads.
3. Removes the paper. Paper and the little gaps in the paw's skin texture become
   transparent. All the ink keeps its shade of gray, light skin texture included.
4. Traces the paw in **grayscale** at twice the scan's resolution, so fine texture and fur
   strokes stay crisp. The SVG is saved at its **true physical size** (in mm), so it imports
   into Illustrator, Cricut Design Space, and similar tools at the right scale.

## Options

To use options, run the script in Terminal from this folder:

```
.venv/bin/python pawprints.py ~/Desktop/paw-scans [options]
```

| Option | What it does |
|---|---|
| `--detail low/medium/high` | How much ink texture and shading to keep. `medium` (default) traces at double resolution, about 5–7 MB per paw. `high` keeps even more shades of gray, about 8 MB. `low` makes smaller files, about 2 MB, but the fine texture gets softer. |
| `--out FOLDER` | Put the results somewhere else. |
| `--threshold 25` | How much darker than the paper around it a spot must be to count as ink (0–255). Lower it if faint ink is lost; raise it if paper smudges or card edges show up. Default is 25. |
| `--workers N` | How many PDFs to process at once. Defaults to one per CPU core, up to 8. |
| `--open` | Open the results folder in Finder when done. The launcher uses this. |

## Good to know

- **Speed:** each paw takes about 5 seconds of work, and several run at once (one per CPU
  core), so around 100 scans take a minute or two on a recent Mac.
- **Light prints on dark products:** the SVG keeps the print's real shades. On a dark
  background, the light-gray parts of a light print will show as light gray.
- **Scanning tips:** 300 DPI color scans on plain white paper work best. Keep the label,
  and any hearts or drawings, a little away from the paw, about an inch. A solid black mark
  touching the paw or sitting right between the toes can't be told apart from the paw, so it
  stays in.
- **Not on a Mac?** The script also runs on Windows or Linux; only the double-click launcher
  is Mac-specific.

## Tests

```
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
```

The tests generate fake scans (`tests/make_samples.py`) and check the following:

- each SVG is named after its PDF, with no overwrites and same-named PDFs numbered
- `report.csv` contents
- each SVG is grayscale on a transparent background
- the label, hearts, pen lines and marker strokes are removed, leaving only the paw
- no pale fringe is left around the paw
- re-running never overwrites earlier results
