#!/bin/bash
# Host-side driver for one dsh fixture row (README.md here). One container per
# row from this directory's throwaway image, --network none, uid 10001, a
# read-only root, no credential, the server running inside the container.
#
#   run_row.sh <row> <stub|fail|script> [<failserver mode>]
#
# Env: DSH_CAPTURE_IMAGE (default mllm-dsh-capture:0.2.2-rc.5); CAPTURE_OUT
# (default ~/.cache/multi-llm-capture/dsh; Docker Desktop refuses /tmp mounts);
# SCENARIO, TOOL, TOOL_ARGS, CAPTURE_DEADLINE_S, MODEL_OVERRIDE and
# REASONING_OVERRIDE pass through to incontainer.py.
set -euo pipefail
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: run_row.sh <row> <stub|fail|script> [<failserver mode>]" >&2
  exit 2
fi
ROW=$1; SERVER=$2; MODE=${3:-}
if [[ ! $ROW =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "run_row.sh: '$ROW' is not a row name" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=$(cd "$HERE/../../row_capture" && pwd)
RIG=$(cd "$HERE/../../../container" && pwd)
SKILL=$(cd "$HERE/../../../.." && pwd)
OUT=${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/dsh}
IMAGE=${DSH_CAPTURE_IMAGE:-mllm-dsh-capture:0.2.2-rc.5}
if [ -z "${DSH_CAPTURE_STAGED:-}" ]; then
  mkdir -p "$OUT/rows/dsh" "$OUT/scenarios"
  cp "$HERE/incontainer.py" "$KIT/scriptserver.py" "$KIT/prompt.txt" "$OUT/"
  cp -r "$KIT/scenarios/." "$OUT/scenarios/"
  (cd "$SKILL" && uv run --project . -- python tests/fixtures/dsh/capture/render.py "$OUT" >/dev/null)
  chmod -R a+rwX "$OUT"
fi
for suffix in stdout.txt stderr.txt meta.json server.jsonl server.log; do
  rm -f "$OUT/rows/dsh/$ROW.$suffix"
done
timeout --signal=KILL 900 docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --pids-limit 512 --memory 4g --cpus 2.0 \
  --tmpfs /work/homes:rw,exec,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,exec,mode=0755,uid=10001,gid=10001 \
  --tmpfs /work/outside:rw,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,exec,mode=1777 \
  -v "$RIG:/rig:ro" -v "$SKILL:/opt/mllm/skill:ro" -v "$OUT:/work/out:rw" \
  -e ROW="$ROW" -e SERVER="$SERVER" -e MODE="$MODE" \
  -e SCENARIO="${SCENARIO:-}" -e TOOL="${TOOL:-}" -e TOOL_ARGS="${TOOL_ARGS:-"{}"}" \
  -e CAPTURE_DEADLINE_S="${CAPTURE_DEADLINE_S:-240}" \
  ${MODEL_OVERRIDE:+-e MODEL_OVERRIDE="$MODEL_OVERRIDE"} \
  ${REASONING_OVERRIDE+-e REASONING_OVERRIDE="$REASONING_OVERRIDE"} \
  "$IMAGE" python3 /work/out/incontainer.py
