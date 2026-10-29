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
  Brown.svg
  Buddy Smith.svg
  Daisy O'Brien - 1.svg    ← a PDF with two pages gives one SVG per page
  Daisy O'Brien - 2.svg
  NEEDS_REVIEW/
    Coco.svg               ← finished, but worth a look
    Coco-page.png          ← the original page, to compare against
  report.csv               ← one line per page, with any problem noted
```

The terminal also shows the converter's version when it starts. If a fix doesn't seem to
take, check that line.

**What to do with NEEDS_REVIEW:** compare the SVG with its `-page.png` and erase anything
that isn't paw. A file lands here when:

- the print is too faint or broken up to pick out the paw's shape, so every solid mark on
  the sheet was kept;
- the SVG seems to contain a leftover background.

## What it does to each scan

1. Finds the ink: anything noticeably darker than the paper around it. Paper is judged
   locally, so a card that's a different white from the scanner lid doesn't count as ink.
2. Keeps only the paw. It removes:
   - red or other colored marks, such as hearts, when the paw is in black ink;
   - anything thinner than about 2.5 mm, such as pen lines, card edges, heart outlines and
     writing;
   - anything that isn't the main pad or one of the toe pads close to it, such as the label
     or stamped hearts elsewhere on the sheet.
3. Removes the paper. Anything close to the paper color becomes transparent, including
   the little gaps in the paw's skin texture. Edges are cleaned up so there's no white
   halo when the paw sits on a dark background.
4. Traces the paw in **grayscale** and saves the SVG at its **true physical size** (in mm),
   so it imports into Illustrator, Cricut Design Space, and similar tools at the right
   scale.

## Options

To use options, run the script in Terminal from this folder:

```
.venv/bin/python pawprints.py ~/Desktop/paw-scans [options]
```

| Option | What it does |
|---|---|
| `--detail low/medium/high` | How much ink texture and shading to keep. `high` keeps the most texture and makes bigger files. Default is `medium`. |
| `--out FOLDER` | Put the results somewhere else. |
| `--threshold 25` | How much darker than the paper around it a spot must be to count as ink (0–255). Lower it if faint ink is lost; raise it if paper smudges or card edges show up. Default is 25. |
| `--workers N` | How many PDFs to process at once. Defaults to one per CPU core, up to 8. |
| `--open` | Open the results folder in Finder when done. The launcher uses this. |

## Good to know

- **Speed:** each PDF takes 1–2 seconds of work, and several run at once (one per CPU
  core), so around 100 scans should take under a minute on a recent Mac.
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
