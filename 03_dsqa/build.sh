#!/bin/bash
# Builds 03_dsqa_submission.zip at the top of the repository, ready to upload
# to Codabench, and checks it.
#
# The zip is written by tools/build_entry.py with Python's own zipfile, so no
# `zip` command is needed. Run it with the virtualenv's interpreter:
#
#     PYTHON=../.venv/bin/python ./build.sh
#     PYTHON=../.venv/Scripts/python.exe ./build.sh --fetch-embedder
#
# What is passed here is passed on: --fetch-embedder downloads the embedding
# model first, --allow-dirty builds from a tree with uncommitted changes.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
python="${PYTHON:-python}"
case "$python" in
    /* | [A-Za-z]:*) ;;
    */*) python="$here/$python" ;;
esac
cd "$here/.."
"$python" -m tools.build_entry "$@"
