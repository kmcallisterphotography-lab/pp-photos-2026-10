#!/bin/bash
# One-time setup: builds a private Python environment (.venv) with everything the converter needs.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done

if [ -n "$PY" ]; then
  echo "Using $("$PY" --version)"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
elif command -v uv >/dev/null 2>&1; then
  echo "Using uv to fetch Python 3.12"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo "This needs Python 3.10 or newer, and none was found."
  echo "Install it from https://www.python.org/downloads/macos/ (or: brew install python),"
  echo "then run this setup again."
  exit 1
fi

echo
echo "Setup complete. Double-click 'Convert Paw Prints.command' to convert a folder of scans."
