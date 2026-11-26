#!/bin/sh
# capture.sh <row> <stub|fail|script> <mode> [capture_probe.py options]
# One container per row from the probe image (README.md), network none,
# uid 10001. Writes $CAPTURE_OUT/<row>.json (default
# ~/.cache/multi-llm-capture/hermes); Docker Desktop cannot mount /tmp.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
skill=$(cd "$here/../../.." && pwd)
out=${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/hermes}
image=${HERMES_PROBE_IMAGE:-mllm-hermes-probe:0.23.4}
mkdir -p "$out"
row=$1; shift
docker run --rm --network none --read-only --tmpfs /tmp:exec,mode=1777 \
  -v "$skill:/skill:ro" -v "$skill/tests/container:/rig:ro" \
  -v "$skill/tests/fixtures/row_capture:/kit:ro" \
  --entrypoint python3 "$image" /skill/tests/fixtures/hermes/capture_probe.py "$row" "$@" \
  > "$out/$row.json"
