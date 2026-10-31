#!/bin/bash
# Double-click in Finder: choose a folder of paw-print PDFs, convert them all, then open the results.
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting things up (takes a minute)..."
  if ! ./setup.sh; then
    read -r -p "Setup failed. Press Return to close this window."
    exit 1
  fi
fi

FOLDER=$(osascript -e 'POSIX path of (choose folder with prompt "Choose the folder of paw-print PDF scans:")' 2>/dev/null) || exit 0

.venv/bin/python pawprints.py "$FOLDER" --open
echo
read -r -p "Done. Press Return to close this window."
