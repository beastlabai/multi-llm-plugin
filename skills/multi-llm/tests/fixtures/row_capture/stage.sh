#!/bin/bash
# Stages the capture kit's shared files into the output dir mounted at
# /work/out: the in-container driver, its servers, the rendered argv and the
# prompt. Run once before any capture of a session (run_row.sh, batch.sh), never
# while a capture reads them.
#
#   stage.sh <out dir>
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: stage.sh <out dir>" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
mkdir -p "$OUT/rows" "$OUT/scenarios"
cp "$HERE/incontainer.py" "$HERE/scriptserver.py" "$HERE/scripts.json" "$HERE/argv.json" "$HERE/prompt.txt" "$OUT/"
cp -r "$HERE/scenarios/." "$OUT/scenarios/"
chmod -R a+rwX "$OUT"
